"""The trading cycle, on a venue that is a recording double.

The loop is the file that decides how much money moves, so it is tested the way the
provider is tested: by watching exactly what it does and proving that the rest did not
happen.

* **the caps.** ``MAX_TRADE_SIZE_USD=50`` against a wallet of $68 must produce $13.60, not
  $50 — and no amount of hunter confidence may change that, because the fraction is the
  answer to "what is the worst one mistake can do".
* **the dry run.** ``DRY_RUN`` unset means dry, and a dry cycle calls ``execute_stake``
  **not at all**. The test asserts on an empty ledger, not on a flag inside the venue.
* **the refusals.** A pool the book does not name, a verdict that convicts, a wallet that
  cannot be read, a feed that died: each is a row with a reason, and none of them is a
  transaction.
* **the memory.** A position is written to the state file and read back, and an
  unreadable state file stops the loop instead of pretending it holds nothing — that is
  the difference between "new stake" and "the same stake twice".
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from agent.physics.portfolio import Protocol
from agent.physics.real_testnet_provider import Balance
from agent.physics.volatility_hunter import Dislocation, PanicReport, PanicSignal
from agent.trade import capital as C
from agent.trade import loop as L

VAULT = "0x" + "11" * 20
OTHER = "0x" + "44" * 20
SIGNER = "0x" + "33" * 20
TOKEN = "0x" + "22" * 20


def vault(name: str = "Aave V3 USDC", address: str = VAULT) -> Protocol:
    """A published vault: sixty days at 0.01 % a day, which compounds to ~3.7 %."""
    return Protocol(
        name=name,
        bucket="safe",
        claim_apy=0.05,
        returns=np.full(60, 0.0001),
        address=address,
    )


def signal(
    verdict: str = "PANIC",
    name: str = "Aave V3 USDC",
    reason: str = "a measured dislocation",
) -> PanicSignal:
    """A hunter verdict with a controlled confidence.

    ``confidence`` is a property computed from the dislocation, so it cannot be passed:
    a PANIC with ``z`` past its saturation, a gap past its saturation and a halved depth
    is a confidence of exactly 1.0, and every other verdict is 0.0 by construction.
    """
    return PanicSignal(
        name=name,
        address=VAULT,
        verdict=verdict,
        reason=reason,
        dislocation=Dislocation(
            resting=4.0,
            recent=12.0,
            sigma=1.0,
            gap_pp=8.0,
            gap_ratio=3.0,
            z=10.0,
            exodus=-0.99,
            points=60,
        ),
        resting_verdict="HONEST",
    )


def report(*signals: PanicSignal, error: str = "") -> PanicReport:
    return PanicReport(
        venue="defillama", scanned=len(signals), signals=tuple(signals), error=error
    )


# ── the venue, as a recording double ────────────────────────────────────────


@dataclass
class FakeVenue:
    """What a cycle asks of a venue, answered from a dict and written down.

    Four calls and two facts — the same surface ``RealTestnetProvider`` answers. Anything
    the loop does *not* call stays empty, which is how "the dry run signed nothing" is
    checked without trusting a boolean inside the loop.
    """

    name: str = "fake-chain"
    chain_id: int = 11155111
    block_number: int = 101
    native_eth: float = 0.008
    token_balance: float = 50.0
    balance_error: str = ""
    raise_on_balance: Optional[BaseException] = None
    refuse: bool = False
    published: List[Protocol] = field(default_factory=lambda: [vault()])
    stakes: List[Tuple[str, float]] = field(default_factory=list)
    withdrawals: List[Tuple[str, float]] = field(default_factory=list)
    refusals: List[Dict[str, Any]] = field(default_factory=list)

    def get_protocol_list(self) -> List[Protocol]:
        return list(self.published)

    def balance(self, address: Optional[str] = None) -> Balance:
        if self.raise_on_balance is not None:
            raise self.raise_on_balance
        if self.balance_error:
            return Balance(
                address=SIGNER, native_eth=self.native_eth, error=self.balance_error
            )
        return Balance(
            address=SIGNER,
            native_eth=self.native_eth,
            native_units=int(self.native_eth * 10 ** 18),
            token=TOKEN,
            token_balance=self.token_balance,
            token_units=int(self.token_balance * 10 ** 6),
            decimals=6,
            chain_id=self.chain_id,
        )

    def execute_stake(self, address: str, amount: float) -> bool:
        if self.refuse:
            self.refusals.append({"action": "stake", "reason": "not in the address book"})
            return False
        self.stakes.append((str(address), float(amount)))
        return True

    def withdraw(self, address: str, amount: float) -> bool:
        if self.refuse:
            self.refusals.append({"action": "withdraw", "reason": "dead protocol"})
            return False
        self.withdrawals.append((str(address), float(amount)))
        return True


def sizing(dry_run: bool = True, **overrides: Any) -> C.Sizing:
    fields: Dict[str, Any] = {
        "max_trade_usd": 50.0,
        "max_fraction": 0.20,
        "eth_price_usd": 2250.0,
        "gas_reserve_eth": 0.0005,
        "dry_run": dry_run,
    }
    fields.update(overrides)
    return C.Sizing(**fields)


def trader(
    venue: FakeVenue,
    *signals: PanicSignal,
    dry_run: bool = True,
    state: Optional[Path] = None,
    error: str = "",
) -> L.Trader:
    return L.Trader(
        venue=venue,
        sizing=sizing(dry_run),
        state_path=state,
        market=lambda: report(*signals, error=error),
        now=lambda: 1_700_000_000.0,
    )

# ── the caps, without a chain in sight ──────────────────────────────────────


def test_a_fifth_of_the_wallet_beats_a_fifty_dollar_ceiling():
    """$50 and $68: the smaller ceiling is the fifth, so the trade is $13.60."""
    size = C.size_trade(
        token_balance=50.0, native_eth=0.008, sizing=sizing()
    )

    assert size.ok
    assert size.amount == pytest.approx(13.6)
    assert size.bound == C.ENV_MAX_FRACTION
    assert size.equity_usd == pytest.approx(68.0)
    assert "20.0% of the wallet" in size.reason


def test_the_operator_ceiling_binds_when_the_wallet_is_large():
    size = C.size_trade(
        token_balance=1_000.0, native_eth=1.0, sizing=sizing()
    )
    assert size.amount == pytest.approx(50.0)
    assert size.bound == C.ENV_MAX_TRADE_USD


def test_ether_without_credits_is_not_a_balance():
    """Gas money cannot be staked: the wallet has $18 and nothing the vault can pull."""
    size = C.size_trade(token_balance=0.0, native_eth=0.008, sizing=sizing())

    assert size.ok is False
    assert size.bound == "token_balance"
    assert "no credits" in size.refusal


def test_a_wallet_below_the_gas_reserve_trades_nothing():
    size = C.size_trade(token_balance=50.0, native_eth=0.0001, sizing=sizing())

    assert size.ok is False
    assert size.bound == "gas_reserve"
    assert "gas reserve" in size.refusal


def test_confidence_cannot_lift_the_fraction():
    """A panic may raise the operator's ceiling; it may not raise the fifth."""
    calm = C.size_trade(
        token_balance=50.0, native_eth=0.008, sizing=sizing(), confidence=0.0
    )
    panic = C.size_trade(
        token_balance=50.0, native_eth=0.008, sizing=sizing(), confidence=1.0
    )

    assert panic.amount == pytest.approx(calm.amount) == pytest.approx(13.6)
    assert panic.bound == C.ENV_MAX_FRACTION


def test_a_panic_lifts_a_ceiling_that_would_otherwise_stop_it():
    """$2 per trade is the operator's number; half again is the most a panic earns."""
    sizing_tight = sizing(max_trade_usd=2.0, max_fraction=1.0)

    calm = C.size_trade(
        token_balance=100.0, native_eth=0.01, sizing=sizing_tight, confidence=0.0
    )
    panic = C.size_trade(
        token_balance=100.0, native_eth=0.01, sizing=sizing_tight, confidence=1.0
    )

    assert calm.amount == pytest.approx(2.0)
    assert panic.amount == pytest.approx(3.0)
    assert panic.bound == C.ENV_MAX_TRADE_USD + "+alpha"


def test_dry_run_is_what_an_unset_variable_means():
    """An empty environment is not an instruction to spend money."""
    assert C.Sizing.from_env({}).dry_run is True
    assert C.Sizing.from_env({C.ENV_DRY_RUN: "0"}).dry_run is False
    assert C.Sizing.from_env({C.ENV_DRY_RUN: "ture"}).dry_run is True
    assert C.Sizing.from_env({C.ENV_DRY_RUN: "false"}).dry_run is False


def test_a_misspelt_number_is_an_error_not_a_zero():
    with pytest.raises(C.SizingError):
        C.Sizing.from_env({C.ENV_MAX_TRADE_USD: "fifty"})
    with pytest.raises(C.SizingError):
        C.Sizing(max_fraction=0.0)

# ── the cycle: what it signs, and what it refuses ───────────────────────────


def test_a_dry_cycle_decides_and_signs_nothing():
    venue = FakeVenue()
    cycle = trader(venue, signal()).run_once()

    assert cycle.dry_run is True
    assert [action.kind for action in cycle.actions] == ["stake"]
    action = cycle.actions[0]
    assert action.sent is False
    assert action.amount == pytest.approx(13.6)
    assert action.verdict == "PANIC"
    assert action.confidence == pytest.approx(1.0)
    assert venue.stakes == []           # the venue was never asked
    assert cycle.sent == 0
    assert "would stake" in cycle.summary_text()


def test_a_live_cycle_sends_the_capped_stake_and_remembers_it(tmp_path):
    state = tmp_path / "trader_state.json"
    venue = FakeVenue()
    cycle = trader(venue, signal(), dry_run=False, state=state).run_once()

    assert venue.stakes == [(VAULT, pytest.approx(13.6))]
    assert cycle.sent == 1
    assert cycle.actions[0].sent is True
    stored = json.loads(state.read_text())
    assert stored["positions"][VAULT]["amount"] == pytest.approx(13.6)
    assert stored["positions"][VAULT]["verdict"] == "PANIC"
    assert stored["cycles"] == 1


def test_a_vault_the_market_does_not_name_is_not_traded():
    """A signal about a pool is not a signal about an address; the loop does not guess."""
    cycle = trader(FakeVenue(), signal(name="not-this-vault")).run_once()

    assert cycle.actions[0].kind == "hold"
    assert "no pool named like this vault" in cycle.actions[0].reason


def test_a_convicted_pool_is_not_bought():
    for verdict in ("FORMULA", "DISHONEST", "THIN", "UNMEASURED"):
        venue = FakeVenue()
        cycle = trader(venue, signal(verdict=verdict, reason=f"{verdict} reason")).run_once()

        assert cycle.actions[0].kind == "hold", verdict
        assert cycle.actions[0].verdict == verdict
        assert cycle.actions[0].reason == f"{verdict} reason"
        assert venue.stakes == []


def test_a_calm_pool_is_bought_for_carry():
    """``CALM`` is an ordinary day in an approved pool: a fifth of the wallet, no panic."""
    venue = FakeVenue()
    cycle = trader(venue, signal(verdict="CALM", reason="an ordinary day")).run_once()

    assert cycle.actions[0].kind == "stake"
    assert cycle.actions[0].amount == pytest.approx(13.6)
    assert cycle.actions[0].confidence == 0.0


def test_a_wallet_that_cannot_be_read_refuses_the_trade():
    venue = FakeVenue(balance_error="token balance: the venue refused (nope)")
    cycle = trader(venue, signal()).run_once()

    assert cycle.actions[0].kind == "hold"
    assert "could not be read" in cycle.actions[0].reason
    assert venue.stakes == []


def test_a_balance_that_raises_is_a_row_and_not_a_crash():
    venue = FakeVenue(raise_on_balance=TimeoutError("the node went quiet"))
    cycle = trader(venue, signal()).run_once()

    assert cycle.actions[0].kind == "hold"
    assert any("could not be read" in warning for warning in cycle.warnings)
    assert venue.stakes == []


def test_a_dead_feed_is_a_warning_and_nothing_is_bought():
    venue = FakeVenue()
    cycle = trader(venue, error="the feed published nothing").run_once()

    assert cycle.market["error"] == "the feed published nothing"
    assert cycle.actions[0].kind == "hold"
    assert any("no pool named like this vault" in a.reason for a in cycle.actions)
    assert venue.stakes == []


def test_the_market_arm_that_raises_does_not_stop_the_cycle():
    def boom():
        raise ConnectionError("no route to host")

    made = L.Trader(venue=FakeVenue(), sizing=sizing(), market=boom)
    cycle = made.run_once()

    assert any("market arm failed" in warning for warning in cycle.warnings)
    assert cycle.actions[0].kind == "hold"


def test_an_empty_book_says_so_in_one_line():
    venue = FakeVenue(published=[])
    cycle = trader(venue, signal()).run_once()

    # nothing to price, so nothing to buy — and the panic is still reported as a signal
    assert [action.kind for action in cycle.actions] == ["signal"]
    assert any("nothing to price" in warning for warning in cycle.warnings)

# ── the memory: positions, exits, and a file that must not lie ──────────────


def seed(path: Path, amount: float = 13.6, name: str = "Aave V3 USDC") -> Path:
    """A state file that says this loop already holds one position."""
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cycles": 3,
                "positions": {
                    VAULT: {
                        "address": VAULT,
                        "name": name,
                        "amount": amount,
                        "at": 1_699_000_000.0,
                        "verdict": "PANIC",
                        "confidence": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_convicted_position_is_closed(tmp_path):
    """The pool is no longer what it was bought as: the position leaves, once."""
    state = seed(tmp_path / "state.json")
    venue = FakeVenue()
    made = trader(venue, signal(verdict="DISHONEST", reason="stops paying"), dry_run=False, state=state)
    cycle = made.run_once()

    kinds = [action.kind for action in cycle.actions]
    assert kinds == ["hold", "withdraw"]
    withdrawal = cycle.actions[-1]
    assert withdrawal.amount == pytest.approx(13.6)
    assert withdrawal.verdict == "DISHONEST"
    assert venue.withdrawals == [(VAULT, pytest.approx(13.6))]
    assert made.positions == {}
    assert json.loads(state.read_text())["positions"] == {}


def test_a_dry_exit_also_signs_nothing(tmp_path):
    state = seed(tmp_path / "state.json")
    venue = FakeVenue()
    cycle = trader(venue, signal(verdict="FORMULA"), dry_run=True, state=state).run_once()

    assert [action.kind for action in cycle.actions] == ["hold", "withdraw"]
    assert venue.withdrawals == []
    assert "would withdraw" in cycle.summary_text()


def test_an_unmeasured_pool_keeps_the_position(tmp_path):
    """A feed that stopped publishing is not evidence against a record."""
    state = seed(tmp_path / "state.json")
    made = trader(FakeVenue(), signal(verdict="UNMEASURED"), dry_run=False, state=state)
    cycle = made.run_once()

    assert [action.kind for action in cycle.actions] == ["hold"]
    assert list(made.positions) == [VAULT]


def test_a_position_whose_vault_stopped_publishing_is_kept_and_flagged(tmp_path):
    state = seed(tmp_path / "state.json")
    made = trader(FakeVenue(published=[]), signal(), dry_run=False, state=state)
    cycle = made.run_once()

    assert any("published no record" in warning for warning in cycle.warnings)
    assert list(made.positions) == [VAULT]


def test_a_second_cycle_does_not_buy_the_same_vault_twice():
    venue = FakeVenue()
    made = trader(venue, signal(), dry_run=False)
    made.printer = lambda text: None
    made.run(cycles=2, interval=0.0)

    assert venue.stakes == [(VAULT, pytest.approx(13.6))]
    assert made.cycles == 2
    assert list(made.positions) == [VAULT]


def test_the_cycle_counter_survives_a_restart(tmp_path):
    state = seed(tmp_path / "state.json")
    first = trader(FakeVenue(), signal(), state=state).run_once()
    assert first.number == 4          # the seeded file had lived three cycles

    again = trader(FakeVenue(), signal(), state=state).run_once()
    assert again.number == 5
    assert json.loads(state.read_text())["cycles"] == 5


def test_an_unreadable_state_file_stops_the_loop(tmp_path):
    """An empty portfolio means \"a new stake\"; a corrupt file must not mean that."""
    broken = tmp_path / "state.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(L.TradeError) as raised:
        trader(FakeVenue(), signal(), state=broken)
    assert str(broken) in str(raised.value)


def test_a_refused_send_quotes_the_venue(tmp_path):
    venue = FakeVenue(refuse=True)
    cycle = trader(venue, signal(), dry_run=False, state=tmp_path / "s.json").run_once()

    assert cycle.actions[0].kind == "stake"
    assert cycle.actions[0].sent is False
    assert "address book" in cycle.actions[0].reason
    assert cycle.sent == 0


# ── the entry point the container runs ──────────────────────────────────────


def wire(monkeypatch, venue: FakeVenue, *signals: PanicSignal) -> None:
    """Replace the two collaborators ``main`` builds from the environment."""
    monkeypatch.setattr(L, "_venue_from_env", lambda: venue)
    monkeypatch.setattr(
        L, "_market_arm", lambda mode, limit, hunt: (lambda: report(*signals))
    )
    for name in (
        "DRY_RUN",
        "MAX_TRADE_SIZE_USD",
        "MAX_TRADE_FRACTION",
        "ETH_USD_PRICE",
        L.ENV_SCAN,
        L.ENV_MEASURE,
    ):
        monkeypatch.delenv(name, raising=False)


def test_the_cli_will_not_run_without_a_node(monkeypatch, capsys):
    def no_node():
        raise RuntimeError("no node: set TESTNET_RPC_URL")

    monkeypatch.setattr(L, "_venue_from_env", no_node)
    assert L.main(["--once"]) == 2
    assert "no node" in capsys.readouterr().err


def test_the_cli_runs_one_dry_cycle_by_default(monkeypatch, tmp_path, capsys):
    venue = FakeVenue()
    wire(monkeypatch, venue, signal())
    state = tmp_path / "state.json"

    assert L.main(["--once", "--state", str(state)]) == 0
    printed = capsys.readouterr().out

    assert "DRY RUN" in printed
    assert "would stake" in printed
    assert venue.stakes == []                    # the whole point of the default
    assert json.loads(state.read_text())["dry_run"] is True


def test_the_cli_signs_only_when_told_twice(monkeypatch, tmp_path, capsys):
    venue = FakeVenue()
    wire(monkeypatch, venue, signal())
    monkeypatch.setenv("DRY_RUN", "0")

    assert L.main(["--once", "--state", str(tmp_path / "s.json")]) == 0
    # $10: the fifth is taken of the stable balance, because no ETH price was given and
    # an unpriced ether balance is not a dollar amount.
    assert venue.stakes == [(VAULT, pytest.approx(10.0))]
    assert "LIVE" in capsys.readouterr().out


def test_the_cli_live_flag_overrides_the_environment(monkeypatch, tmp_path):
    venue = FakeVenue()
    wire(monkeypatch, venue, signal())
    monkeypatch.setenv("DRY_RUN", "1")

    assert L.main(["--once", "--live", "--state", str(tmp_path / "s.json")]) == 0
    assert venue.stakes == [(VAULT, pytest.approx(10.0))]


def test_the_cli_writes_the_cycle_as_json(monkeypatch, tmp_path):
    wire(monkeypatch, FakeVenue(), signal())
    report_path = tmp_path / "cycle.json"

    assert (
        L.main(
            [
                "--once",
                "--state",
                str(tmp_path / "s.json"),
                "--json",
                str(report_path),
            ]
        )
        == 0
    )
    payload = json.loads(report_path.read_text())
    assert payload["cycle"] == 1
    assert payload["actions"][0]["kind"] == "stake"
    assert payload["dry_run"] is True


def test_the_cli_number_flags_reach_the_policy(monkeypatch, tmp_path):
    venue = FakeVenue()
    wire(monkeypatch, venue, signal(verdict="CALM", reason="an ordinary day"))

    L.main(
        [
            "--once",
            "--live",
            "--state",
            str(tmp_path / "s.json"),
            "--max-usd",
            "2",
            "--max-fraction",
            "1",
            "--eth-price",
            "2250",
        ]
    )
    # a CALM pool earns no alpha lift, so the operator's $2 is the whole ceiling
    assert venue.stakes == [(VAULT, pytest.approx(2.0))]


def test_a_panic_with_no_vault_is_a_signal_and_not_a_trade():
    """The most interesting row of a cycle is the one it must not act on."""
    venue = FakeVenue()
    cycle = trader(venue, signal(name="some-other-pool")).run_once()

    assert [action.kind for action in cycle.actions] == ["hold", "signal"]
    row = cycle.actions[-1]
    assert row.verdict == "PANIC"
    assert row.amount == 0.0          # no vault means no token, so no amount
    assert row.sent is False
    assert "no vault in the operator's book" in row.reason
    assert venue.stakes == []
    assert "signal" in cycle.summary_text()


def blocked_state(tmp_path: Path) -> Path:
    """A state path whose directory cannot be written: a read-only parent."""
    parent = tmp_path / "readonly"
    parent.mkdir()
    parent.chmod(0o500)
    return parent / "state.json"


def test_a_live_loop_that_cannot_write_its_state_stops(tmp_path):
    """Without the file, a live loop would stake the same money every cycle."""
    if os.getuid() == 0:  # pragma: no cover - root ignores the mode bits
        pytest.skip("root can write into a read-only directory")
    made = trader(FakeVenue(), signal(), dry_run=False, state=blocked_state(tmp_path))

    with pytest.raises(L.TradeError) as raised:
        made.run_once()
    assert "could not be written" in str(raised.value)


def test_a_dry_run_warns_about_an_unwritable_state_file_instead(tmp_path):
    if os.getuid() == 0:  # pragma: no cover - root ignores the mode bits
        pytest.skip("root can write into a read-only directory")
    cycle = trader(
        FakeVenue(), signal(), dry_run=True, state=blocked_state(tmp_path)
    ).run_once()

    assert any("could not be written" in warning for warning in cycle.warnings)


# ── how wide the hunt is: SCAN_LIMIT and MEASURE_LIMIT ──────────────────────


def test_the_scan_width_comes_from_the_environment():
    """``SCAN_LIMIT=350`` is a wider hunt than the loop's own 200, not a new rule."""
    assert L._limit(L.ENV_SCAN, L.DEFAULT_SCAN_LIMIT, {}) == L.DEFAULT_SCAN_LIMIT
    assert L._limit(L.ENV_SCAN, L.DEFAULT_SCAN_LIMIT, {"SCAN_LIMIT": "350"}) == 350
    assert L._limit(L.ENV_MEASURE, L.DEFAULT_MEASURE_LIMIT, {"MEASURE_LIMIT": "8"}) == 8
    # a `.env` is written by hand and a hand writes "200.0"; an empty value is a default
    assert L._limit(L.ENV_SCAN, 200, {"SCAN_LIMIT": "200.0"}) == 200
    assert L._limit(L.ENV_SCAN, 200, {"SCAN_LIMIT": "  "}) == 200


def test_a_width_of_nothing_is_a_refusal_with_a_name():
    """``0`` is not "unlimited": a loop that read it that way would hammer the feed."""
    with pytest.raises(L.TradeError) as raised:
        L._count("0", "SCAN_LIMIT")
    assert "SCAN_LIMIT=0" in str(raised.value)
    with pytest.raises(L.TradeError):
        L._count("-5", "--hunt")
    with pytest.raises(L.TradeError) as raised:
        L._count("many", "--limit")
    assert "--limit='many'" in str(raised.value)


def test_the_default_scan_width_is_the_funnels_own():
    """The loop does not carry a second opinion about how big the market is."""
    from agent.physics.defillama_adapter import POOLS_LIMIT, WATCH_LIMIT

    assert L.DEFAULT_SCAN_LIMIT == POOLS_LIMIT
    assert L.DEFAULT_MEASURE_LIMIT <= WATCH_LIMIT


def arm_recorder(
    monkeypatch, seen: List[Tuple[str, int, int]], *signals: PanicSignal
) -> None:
    """Replace the market arm with one that writes down how wide it was asked to read."""

    def arm(mode: str, limit: int, hunt: int):
        seen.append((str(mode), int(limit), int(hunt)))
        return lambda: report(*signals)

    monkeypatch.setattr(L, "_market_arm", arm)


def test_the_cli_reads_the_width_of_the_hunt_from_the_environment(
    monkeypatch, tmp_path, capsys
):
    seen: List[Tuple[str, int, int]] = []
    wire(monkeypatch, FakeVenue(), signal())
    arm_recorder(monkeypatch, seen, signal())
    monkeypatch.setenv(L.ENV_SCAN, "300")
    monkeypatch.setenv(L.ENV_MEASURE, "25")

    assert L.main(["--once", "--state", str(tmp_path / "s.json")]) == 0

    assert seen == [("real", 300, 25)]
    assert "scan 300, measure 25" in capsys.readouterr().out


def test_the_flag_beats_the_environment_and_the_environment_beats_the_default(
    monkeypatch, tmp_path
):
    seen: List[Tuple[str, int, int]] = []
    wire(monkeypatch, FakeVenue(), signal())
    arm_recorder(monkeypatch, seen, signal())

    monkeypatch.delenv(L.ENV_SCAN, raising=False)
    monkeypatch.delenv(L.ENV_MEASURE, raising=False)
    assert L.main(["--once", "--state", str(tmp_path / "a.json")]) == 0
    assert seen[-1] == ("real", L.DEFAULT_SCAN_LIMIT, L.DEFAULT_MEASURE_LIMIT)

    monkeypatch.setenv(L.ENV_SCAN, "300")
    monkeypatch.setenv(L.ENV_MEASURE, "25")
    flags = ["--once", "--limit", "9", "--hunt", "3", "--state", str(tmp_path / "b.json")]
    assert L.main(flags) == 0
    assert seen[-1] == ("real", 9, 3)


def test_a_bad_width_stops_the_cli_before_a_cycle(monkeypatch, tmp_path, capsys):
    seen: List[Tuple[str, int, int]] = []
    wire(monkeypatch, FakeVenue(), signal())
    arm_recorder(monkeypatch, seen, signal())
    monkeypatch.setenv(L.ENV_SCAN, "lots")

    assert L.main(["--once", "--state", str(tmp_path / "s.json")]) == 2
    assert "SCAN_LIMIT='lots' is not a count" in capsys.readouterr().err
    assert seen == []

    monkeypatch.delenv(L.ENV_SCAN)
    assert L.main(["--once", "--limit", "0", "--state", str(tmp_path / "s.json")]) == 2
    assert "--limit=0 is not a count" in capsys.readouterr().err
    assert seen == []


# ── the address book: one example the loader accepts, and why it is empty ────


def example_book_path() -> Path:
    """``book.example.json`` in the repo root — the file the operator is told to copy."""
    return Path(L.__file__).resolve().parents[2] / "book.example.json"


def test_the_example_book_is_a_book_the_venue_accepts():
    """The example is loaded by the real loader, so it cannot document a shape it fails."""
    from agent.physics.real_testnet_provider import ProtocolBook

    book = ProtocolBook.load(str(example_book_path()))

    assert len(book) == 2
    for row in book:
        assert row.name and row.address.startswith("0x") and len(row.address) == 42
        assert row.decimals == 6
        assert row.token.startswith("0x")
    # the reader's notes are ignored by the loader, so the file can explain itself
    payload = json.loads(example_book_path().read_text(encoding="utf-8"))
    assert "_readme" in payload and "protocols" in payload


def book_of(*rows: Dict[str, Any]) -> Any:
    from agent.physics.real_testnet_provider import ProtocolBook

    return ProtocolBook.from_payload({"protocols": list(rows)})


def book_entry(address: str = VAULT, name: str = "Aave V3 USDC") -> Dict[str, Any]:
    return {"address": address, "name": name, "bucket": "active", "decimals": 6}


def with_book(
    venue: FakeVenue,
    book: Any,
    lookback: int = 5_000,
    history: str = "chain",
) -> FakeVenue:
    """Give a stub the one attribute the diagnostics need: the book it was built from."""
    venue.config = SimpleNamespace(  # type: ignore[attr-defined]
        book=book, lookback=lookback, history=history
    )
    return venue


def test_an_empty_book_names_the_configuration_and_not_the_network():
    venue = with_book(FakeVenue(published=[]), book_of())
    cycle = trader(venue).run_once()

    assert any("the operator's book is empty" in warning for warning in cycle.warnings)
    assert any("book.example.json" in warning for warning in cycle.warnings)


def test_a_book_whose_vaults_published_nothing_names_each_one():
    """The line that turns "book empty" into "this address is not a vault"."""
    venue = with_book(FakeVenue(published=[]), book_of(book_entry()))
    cycle = trader(venue).run_once()

    warnings = [warning for warning in cycle.warnings if VAULT in warning]
    assert warnings, cycle.warnings
    assert "Aave V3 USDC" in warnings[0]
    assert L.YIELD_EVENT_SIGNATURE in warnings[0]
    assert "5000 blocks" in warnings[0]


def test_a_venue_without_a_book_still_says_nothing_to_price():
    """A stub that answers no ``config`` keeps the old one-line behaviour."""
    cycle = trader(FakeVenue(published=[]), signal()).run_once()

    assert any("nothing to price" in warning for warning in cycle.warnings)


# ── the book report: the diagnostic that answers "why is it empty?" ─────────


@dataclass
class BookVenue(FakeVenue):
    """A venue that also answers for its records, address by address."""

    records: Dict[str, List[float]] = field(default_factory=dict)
    config: Any = None
    #: Where each address's record came from, the way ``RealTestnetProvider`` answers it:
    #: ``chain`` for the address's own log, the feed's label otherwise.
    source: Dict[str, str] = field(default_factory=dict)

    def fetch_protocol_history(
        self, address: str, steps: Optional[int] = None
    ) -> List[float]:
        return list(self.records.get(str(address).lower(), ()))

    def history_source(self, address: str) -> str:
        return self.source.get(str(address).lower(), "")


def test_the_book_report_says_which_entry_has_no_record():
    venue = BookVenue(published=[], records={})
    with_book(venue, book_of(book_entry(), book_entry(OTHER, "UNI V3 USDC-ETH")))
    printed: List[str] = []

    assert L._book_report(venue, None, printed.append) == 4

    text = "\n".join(printed)
    assert "2 entry/entries" in text
    assert "5000 blocks" in text
    for address in (VAULT, OTHER):
        assert address in text
    assert "no record" in text
    assert "nothing to price" in text
    assert L.YIELD_EVENT_SIGNATURE in text


def test_the_book_report_says_which_entry_published_and_what_the_market_thinks():
    venue = BookVenue(published=[], records={VAULT: [0.0001] * 60})
    with_book(venue, book_of(book_entry()))
    printed: List[str] = []

    assert L._book_report(venue, lambda: report(signal()), printed.append) == 0

    text = "\n".join(printed)
    assert "60 record(s)" in text
    assert "PANIC" in text
    assert "nothing to price" not in text


def test_the_book_report_survives_a_market_that_cannot_be_read():
    def boom():
        raise ConnectionError("no route to host")

    venue = BookVenue(published=[], records={VAULT: [0.0001]})
    with_book(venue, book_of(book_entry()))
    printed: List[str] = []

    assert L._book_report(venue, boom, printed.append) == 0
    assert any("market unreadable" in line for line in printed)


def test_the_book_report_says_empty_and_nothing_else():
    printed: List[str] = []

    assert L._book_report(FakeVenue(), None, printed.append) == 4
    assert "empty" in "\n".join(printed)


def test_the_cli_reports_the_book_and_runs_no_cycle(monkeypatch, capsys):
    wire(monkeypatch, FakeVenue(), signal())

    assert L.main(["--book-report"]) == 4
    printed = capsys.readouterr().out
    assert "book report" in printed
    assert "empty" in printed
    assert "cycle 1" not in printed   # no cycle was run, and no state file was written


# ── a record the feed vouched for: measured, printed, never signed ──────────
#
# A book of real dApp addresses publishes no ``YieldDistributed``, so the venue may price an
# entry from DeFiLlama's history of its *name* (``TESTNET_HISTORY=auto``). That makes the
# book non-empty and the hunter's verdict real — and it is not evidence that the address
# implements ``stake(uint256)``, which is the one thing a signature would need.

FEED_LABEL = "the feed aave-v3 USDC (Arbitrum) [d9fa8e]"


def test_a_feed_priced_vault_is_measured_and_not_staked():
    """A PANIC on a name-matched pool is a verdict, and not permission to sign."""
    venue = BookVenue(published=[vault()], source={VAULT: FEED_LABEL})
    cycle = trader(venue, signal(), dry_run=False).run_once()

    assert cycle.book[0]["source"] == FEED_LABEL
    assert cycle.book[0]["implied_apy"] is not None     # it *is* priced
    action = cycle.actions[0]
    assert action.kind == "hold"
    assert action.verdict == "PANIC"                     # the verdict is still printed
    assert FEED_LABEL in action.reason
    assert venue.stakes == []                            # and nothing was signed


def test_a_record_read_from_the_address_itself_is_still_staked():
    """The guard is narrow on purpose: the vault's own log is what a stake may be based on."""
    venue = BookVenue(
        published=[vault()], source={VAULT: L.RECORD_FROM_CHAIN}
    )
    cycle = trader(venue, signal(), dry_run=False).run_once()

    assert cycle.actions[0].kind == "stake"
    assert venue.stakes and venue.stakes[0][0] == VAULT


def test_the_cycle_prints_where_a_record_came_from():
    venue = BookVenue(published=[vault()], source={VAULT: FEED_LABEL})
    text = trader(venue, signal()).run_once().summary_text()

    assert "book  Aave V3 USDC" in text
    assert FEED_LABEL in text
    assert "signs only into a vault whose record it has read itself" in text


def test_a_quiet_book_names_the_switch_that_would_price_it():
    """"No log" is a diagnosis; the warning also says which variable changes it."""
    venue = with_book(FakeVenue(published=[]), book_of(book_entry()), history="chain")
    cycle = trader(venue).run_once()

    assert any("TESTNET_HISTORY=auto" in warning for warning in cycle.warnings), (
        cycle.warnings
    )


def test_a_feed_that_named_nothing_is_reported_as_such():
    venue = with_book(FakeVenue(published=[]), book_of(book_entry()), history="auto")
    cycle = trader(venue).run_once()

    assert any(
        "TESTNET_HISTORY=auto was asked and named none of these entries" in warning
        for warning in cycle.warnings
    ), cycle.warnings


def test_the_book_report_shows_a_feed_record_and_says_it_is_not_stakable():
    venue = BookVenue(
        published=[], records={VAULT: [0.0001] * 60}, source={VAULT: FEED_LABEL}
    )
    with_book(venue, book_of(book_entry()), history="auto")
    printed: List[str] = []

    assert L._book_report(venue, None, printed.append) == 0

    text = "\n".join(printed)
    assert "TESTNET_HISTORY=auto" in text
    assert f"60 record(s) · {FEED_LABEL}" in text
    assert "not stakable" in text
    assert "1 of 1 record(s) came from the feed" in text


def test_the_book_report_points_at_the_switch_when_the_feed_was_not_asked():
    venue = BookVenue(published=[], records={})
    with_book(venue, book_of(book_entry()), history="chain")
    printed: List[str] = []

    assert L._book_report(venue, None, printed.append) == 4
    text = "\n".join(printed)
    assert "the feed was not asked" in text
    assert "TESTNET_HISTORY=auto" in text
