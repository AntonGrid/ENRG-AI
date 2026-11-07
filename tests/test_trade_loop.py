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
from dataclasses import dataclass, field, replace
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
    *,
    project: str = "",
    chain: str = "",
    uuid: str = "",
) -> PanicSignal:
    """A hunter verdict with a controlled confidence.

    ``confidence`` is a property computed from the dislocation, so it cannot be passed:
    a PANIC with ``z`` past its saturation, a gap past its saturation and a halved depth
    is a confidence of exactly 1.0, and every other verdict is 0.0 by construction.

    ``project``/``chain``/``uuid`` are the feed's own spelling of the row, which is what an
    ``aave-v3`` book entry may be matched on when a label and a pool name disagree.
    """
    return PanicSignal(
        name=name,
        address=VAULT,
        verdict=verdict,
        reason=reason,
        uuid=uuid,
        project=project,
        chain=chain,
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


# ── the micro cap, and a position that can be topped up ─────────────────────
#
# Two ceilings the operator asked for by name. ``TEST_STAKE_AMOUNT_USD`` is the one a
# first, unproven loop cannot argue with: it lowers every transaction, is never lifted by
# a panic, and is off unless it is set. ``held_usd`` is the same number in the other
# direction: a pool whose ``supply`` grows a position is sized against a *target*, and the
# transaction carries the difference — with a floor below which a fee is not worth paying.


def test_the_micro_cap_is_a_ceiling_like_the_others():
    size = C.size_trade(
        token_balance=50.0, native_eth=0.008, sizing=sizing(micro_cap_usd=1.0)
    )

    assert size.ok
    assert size.amount == pytest.approx(1.0)
    assert size.bound == C.ENV_MICRO_CAP
    assert size.bound in size.reason


def test_the_micro_cap_is_never_lifted_by_a_panic():
    calm = C.size_trade(
        token_balance=50.0,
        native_eth=0.008,
        sizing=sizing(micro_cap_usd=1.0),
        confidence=0.0,
    )
    panic = C.size_trade(
        token_balance=50.0,
        native_eth=0.008,
        sizing=sizing(micro_cap_usd=1.0),
        confidence=1.0,
    )

    assert panic.amount == pytest.approx(calm.amount) == pytest.approx(1.0)


def test_an_unset_micro_cap_is_not_a_cap_of_zero():
    """The ordinary case: no micro cap configured, and the fraction decides as before."""
    assert C.Sizing.from_env({}).micro_cap_usd == 0.0
    assert C.Sizing.from_env({C.ENV_MICRO_CAP: "1.5"}).micro_cap_usd == pytest.approx(1.5)
    with pytest.raises(C.SizingError):
        C.Sizing.from_env({C.ENV_MICRO_CAP: "one dollar"})
    with pytest.raises(C.SizingError):
        C.Sizing(micro_cap_usd=-1.0)


def test_a_held_position_is_sized_as_a_target_and_moved_as_a_difference():
    """$1 already supplied, a $13.82 wallet: the transaction carries what is missing."""
    size = C.size_trade(
        token_balance=12.82,
        native_eth=0.00195,
        sizing=sizing(eth_price_usd=0.0),
        held_usd=1.0,
    )

    assert size.ok
    assert size.usd == pytest.approx(size.amount) == pytest.approx(1.764)
    assert "top_up" in size.bound
    assert "already supplied" in size.reason


def test_a_position_that_is_already_at_the_ceiling_is_nothing_to_do():
    """The fixed point of "a fifth of the wallet": 17 held, and a fifth of 85 is 17."""
    size = C.size_trade(
        token_balance=50.0, native_eth=0.008, sizing=sizing(), held_usd=17.0
    )

    assert size.ok is False
    assert size.bound == "top_up"
    assert "already at the ceiling" in size.refusal


def test_a_top_up_below_the_minimum_is_a_refusal_not_a_transaction():
    size = C.size_trade(
        token_balance=50.0, native_eth=0.008, sizing=sizing(), held_usd=16.7
    )

    assert size.ok is False
    assert "minimum top-up" in size.refusal
    assert C.MIN_TOP_UP_USD == pytest.approx(0.5)

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
        C.ENV_MICRO_CAP,
        L.ENV_FORCE,
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
    #: How money reaches each address, the way the venue answers ``protocol_kind``: a
    #: ``vault`` of this interface, or an ``aave-v3`` pool the deployment table names.
    #: An address that is absent answers nothing, and the loop then takes the strict path.
    kinds: Dict[str, str] = field(default_factory=dict)
    #: How the feed spells each entry, the way the venue answers ``protocol_identity``: the
    #: project, symbol, chain and pinned uuid the loop may match a verdict on. An address
    #: that is absent answers ``None``, which is the strict path — a name and nothing else.
    identities: Dict[str, Any] = field(default_factory=dict)
    #: What the venue already holds of the entry — for Aave, the aToken balance, which is
    #: the supplied amount plus interest, and which the chain reports rather than the loop.
    position: float = 0.0

    def fetch_protocol_history(
        self, address: str, steps: Optional[int] = None
    ) -> List[float]:
        return list(self.records.get(str(address).lower(), ()))

    def history_source(self, address: str) -> str:
        return self.source.get(str(address).lower(), "")

    def protocol_kind(self, address: str) -> str:
        return self.kinds.get(str(address).lower(), "")

    def protocol_identity(self, address: str) -> Any:
        return self.identities.get(str(address).lower())

    def balance(self, address: Optional[str] = None) -> Balance:
        read = super().balance(address)
        if self.protocol_kind(str(address or "")) != "aave-v3":
            return read
        units = int(round(float(self.position) * 10 ** 6))
        return replace(
            read,
            position_token=TOKEN,
            position_units=units,
            position_balance=units / 10 ** 6,
        )


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


# ── an Aave position: the record is the feed's, the signature is the pool's ──
#
# The one exemption to "a feed record is never signed", and the reason it is safe is a
# property of the *address* rather than of the record: an ``aave-v3`` entry names a pool
# the venue verified against its deployment table on the chain id the node reported. So
# the question "is this address a stranger whose interface I cannot see" has an answer,
# and it is not the operator's optimism. What follows tests the three consequences: the
# supply is signed, the position is topped up to the ceiling rather than re-staked, and
# the amount held is read from the chain rather than remembered.

AAVE = "0x" + "55" * 20
AAVE_NAME = "Aave V3 USDC (Arbitrum)"


def aave_pool() -> Protocol:
    """The pool as the venue publishes it: a name, a record, and the pool's address."""
    return vault(name=AAVE_NAME, address=AAVE)


def identity(**overrides: Any) -> Any:
    """The ``ProtocolIdentity`` a real venue answers: what the feed calls this entry.

    ``aave-v3``/``USDC``/``Arbitrum`` is DeFiLlama's own spelling of the Aave row, and the
    uuid is the one ``book.example.json`` pins, so a test says "the row the feed publishes
    for this pool" rather than repeating a label that happens to normalise the same way.
    """
    fields: Dict[str, Any] = {
        "kind": "aave-v3",
        "project": "aave-v3",
        "symbol": "USDC",
        "chain": "Arbitrum",
        "pool": "d9fa8e14-0447-4207-9ae8-7810199dfa1f",
        "protocol": "Aave",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def pool_venue(**overrides: Any) -> BookVenue:
    fields: Dict[str, Any] = {
        "published": [aave_pool()],
        "source": {AAVE: FEED_LABEL},
        "kinds": {AAVE: "aave-v3"},
        "identities": {AAVE: identity()},
    }
    fields.update(overrides)
    return BookVenue(**fields)


def test_a_feed_priced_aave_pool_is_staked():
    """The verdict is about the name, and the address is a pool this file can name."""
    venue = pool_venue()
    cycle = trader(venue, signal(name=AAVE_NAME), dry_run=False).run_once()

    action = cycle.actions[0]
    assert action.kind == "stake"
    assert action.sent is True
    assert venue.stakes == [(AAVE, pytest.approx(13.6))]
    assert "Aave v3 Pool" in action.reason
    assert "aToken" in action.reason
    assert cycle.book[0]["kind"] == "aave-v3"


def test_the_same_record_on_a_vault_is_still_not_signed():
    """The exemption is the kind, not the record: a vault of unknown interface is held."""
    venue = pool_venue(kinds={AAVE: "vault"})
    cycle = trader(venue, signal(name=AAVE_NAME), dry_run=False).run_once()

    assert cycle.actions[0].kind == "hold"
    assert FEED_LABEL in cycle.actions[0].reason
    assert venue.stakes == []


def test_a_position_is_topped_up_not_staked_twice(tmp_path):
    """$1 already in the pool: the cycle moves the difference to the ceiling, and no more."""
    state = tmp_path / "trader_state.json"
    venue = pool_venue(position=1.0)
    cycle = trader(venue, signal(name=AAVE_NAME), dry_run=False, state=state).run_once()

    action = cycle.actions[0]
    assert action.kind == "stake"
    assert action.amount == pytest.approx(12.8)  # 20% of (50 free + 1 held + 18 gas), minus 1
    assert "top_up" in action.bound
    stored = json.loads(state.read_text())
    assert stored["positions"][AAVE]["amount"] == pytest.approx(13.8)


def test_a_position_at_the_ceiling_is_a_hold():
    """The chain says 17 supplied and a fifth of 85 is 17: there is nothing to add."""
    venue = pool_venue(position=17.0)
    cycle = trader(venue, signal(name=AAVE_NAME), dry_run=False).run_once()

    action = cycle.actions[0]
    assert action.kind == "hold"
    assert "already at the ceiling" in action.reason
    assert venue.stakes == []


def test_the_position_is_read_from_the_chain_and_not_from_the_state_file():
    """A lost state file is not a reason to supply the same dollars again."""
    venue = pool_venue(position=17.0)
    cycle = trader(venue, signal(name=AAVE_NAME), dry_run=False).run_once()

    assert cycle.actions[0].kind == "hold"
    assert venue.stakes == []


def test_the_micro_cap_sizes_the_first_supply():
    venue = pool_venue()
    cycle = L.Trader(
        venue=venue,
        sizing=sizing(dry_run=False, micro_cap_usd=1.0),
        market=lambda: report(signal(name=AAVE_NAME)),
        now=lambda: 1_700_000_000.0,
    ).run_once()

    action = cycle.actions[0]
    assert action.kind == "stake"
    assert action.amount == pytest.approx(1.0)
    assert action.bound == C.ENV_MICRO_CAP
    assert venue.stakes == [(AAVE, pytest.approx(1.0))]


def test_the_book_report_calls_a_feed_priced_pool_stakable():
    venue = pool_venue(records={AAVE: [0.0001] * 60})
    with_book(venue, book_of(book_entry(AAVE, AAVE_NAME)), history="auto")
    printed: List[str] = []

    assert L._book_report(venue, None, printed.append) == 0

    text = "\n".join(printed)
    assert "aave-v3" in text
    assert "stakable: yes" in text
    assert "supply() moves the money" in text
    assert "not stakable" not in text
    assert "each one a pool this file knows by address, so they are staked" in text


# ── a label is not a key: the declaration the book may carry ────────────────
#
# The book spells its entries for a human (``aave-v3 USDC (Arbitrum)``) and DeFiLlama spells
# its rows for a database (``USDC``, project ``aave-v3``, chain ``Arbitrum``). Matching the
# two by spelling turned a verified pool into "the market has no pool named like this vault",
# so an ``aave-v3`` entry may carry the feed's own words and be recognised by what it *is*.
# The name is still tried first; a vault still has nothing but its name, and that is the
# asymmetry these tests hold in place.


def test_a_pool_whose_row_the_feed_spells_differently_is_still_supplied():
    """The row is ``USDC`` under project ``aave-v3``: same pool, different spelling."""
    venue = pool_venue()
    cycle = trader(
        venue,
        signal(name="USDC", project="aave-v3", chain="Arbitrum"),
        dry_run=False,
    ).run_once()

    action = cycle.actions[0]
    assert action.kind == "stake"
    assert action.verdict == "PANIC"                  # the row's verdict, not a guess
    assert action.sent is True
    assert venue.stakes == [(AAVE, pytest.approx(13.6))]


def test_a_pinned_uuid_finds_the_row_whatever_the_feed_calls_it():
    """The strongest key in the book: an identity, not a similarity.

    The uuid is pinned in ``book.example.json``, so a row whose label shares no words with
    the entry is still this pool's row — and the comparison ignores case, because a uuid
    pasted out of a web page is not always lower case.
    """
    venue = pool_venue()
    cycle = trader(
        venue,
        signal(name="Pool #42 (whatever)", uuid="D9FA8E14-0447-4207-9AE8-7810199DFA1F"),
        dry_run=False,
    ).run_once()

    assert cycle.actions[0].kind == "stake"
    assert cycle.actions[0].verdict == "PANIC"


def test_a_row_on_another_chain_is_not_this_pools_verdict():
    """The chain is a filter: the same project and symbol elsewhere is another pool."""
    venue = pool_venue()
    cycle = trader(
        venue,
        signal(name="USDC", project="aave-v3", chain="Ethereum"),
        dry_run=False,
    ).run_once()

    action = cycle.actions[0]
    assert action.verdict == L.UNMATCHED     # the row was about another chain's pool
    assert action.kind == "stake"            # and the entry does not need it to be real


def test_an_aave_pool_the_report_never_reached_is_still_supplied():
    """No signal at all: the deployment table, not a label, is what makes it stakeable."""
    venue = pool_venue()
    cycle = trader(venue, dry_run=False).run_once()      # an empty report

    action = cycle.actions[0]
    assert action.kind == "stake"
    assert action.verdict == L.UNMATCHED
    assert "deployment table" in action.reason
    assert "not a refusal" in action.reason
    assert venue.stakes == [(AAVE, pytest.approx(13.6))]
    assert not [row for row in cycle.actions if row.kind == "signal"]


def test_a_vault_the_report_never_reached_is_still_a_hold():
    """The exemption is the kind: an unnamed vault stays a stranger, whatever it paid."""
    venue = BookVenue(published=[vault()], source={VAULT: L.RECORD_FROM_CHAIN})
    cycle = trader(venue, dry_run=False).run_once()

    action = cycle.actions[0]
    assert action.kind == "hold"
    assert "no pool named like this vault" in action.reason
    assert venue.stakes == []


def test_the_book_report_says_when_a_verdict_came_from_the_declaration():
    """A diagnostic that disagreed with the decision would be worse than none."""
    venue = pool_venue(records={AAVE: [0.0001] * 60})
    with_book(venue, book_of(book_entry(AAVE, AAVE_NAME)), history="auto")
    printed: List[str] = []

    L._book_report(
        venue,
        lambda: report(signal(name="USDC", project="aave-v3", chain="Arbitrum")),
        printed.append,
    )

    text = "\n".join(printed)
    assert "PANIC" in text
    assert "matched by declaration" in text
    assert "no pool in the report is named like this entry" not in text


# ── FORCE_TRADE: prove the pipe, not the thesis ─────────────────────────────
#
# A first real run has one job — approve, supply, a receipt — and the market's silence (or a
# calm day) is no reason to postpone it. The flag therefore lifts exactly one veto, on
# exactly one kind of entry, and prints the sentence an operator will look for in the log.


def forced(venue: BookVenue, *signals: PanicSignal, dry_run: bool = True, micro: float = 1.0):
    """A trader with ``FORCE_TRADE`` on, recording every line it prints."""
    printed: List[str] = []
    cycle = L.Trader(
        venue=venue,
        sizing=sizing(dry_run=dry_run, micro_cap_usd=micro),
        market=lambda: report(*signals),
        now=lambda: 1_700_000_000.0,
        force=True,
        printer=printed.append,
    ).run_once()
    return cycle, printed


def test_force_trade_is_off_unless_the_operator_asks():
    """The default is the one that cannot spend: an unset flag is not an instruction."""
    venue = pool_venue()
    cycle = trader(
        venue, signal(verdict="UNMEASURED", name=AAVE_NAME), dry_run=False
    ).run_once()

    action = cycle.actions[0]
    assert action.kind == "hold"
    assert action.verdict == "UNMEASURED"
    assert venue.stakes == []


def test_force_trade_supplies_through_a_verdict_that_is_not_an_entry():
    """``UNMEASURED`` is not a conviction, so the flag may carry a pool it cannot measure."""
    venue = pool_venue()
    cycle, printed = forced(
        venue, signal(verdict="UNMEASURED", name=AAVE_NAME), dry_run=False
    )

    action = cycle.actions[0]
    assert action.kind == "stake"
    assert action.sent is True
    assert action.amount == pytest.approx(1.0)             # the micro cap still decides
    assert action.bound == C.ENV_MICRO_CAP
    assert venue.stakes == [(AAVE, pytest.approx(1.0))]
    assert "FORCE_TRADE" in action.reason
    assert printed == ["Force trade: I supplied $1.00 USDC to Aave."]


def test_force_trade_in_a_dry_run_says_what_it_would_supply():
    """The rehearsal line, word for word — and the venue is never called."""
    venue = pool_venue()
    cycle, printed = forced(venue, signal(verdict="UNMEASURED", name=AAVE_NAME))

    assert printed == ["Force trade: I would supply $1.00 USDC to Aave."]
    assert venue.stakes == []
    assert cycle.actions[0].sent is False
    assert cycle.dry_run is True


def test_force_trade_still_respects_a_tighter_ceiling():
    """It lifts a verdict, never an amount: a $0.50 micro cap sends $0.50."""
    venue = pool_venue()
    cycle, printed = forced(
        venue, signal(verdict="UNMEASURED", name=AAVE_NAME), dry_run=False, micro=0.5
    )

    assert cycle.actions[0].amount == pytest.approx(0.5)
    assert venue.stakes == [(AAVE, pytest.approx(0.5))]
    assert printed == ["Force trade: I supplied $0.50 USDC to Aave."]


def test_force_trade_does_not_outrank_a_conviction():
    """``DISHONEST`` is evidence about a pool; a flag on a command line is not."""
    printed: List[str] = []
    venue = pool_venue()
    cycle = L.Trader(
        venue=venue,
        sizing=sizing(dry_run=False, micro_cap_usd=1.0),
        market=lambda: report(signal(verdict="DISHONEST", name=AAVE_NAME)),
        now=lambda: 1_700_000_000.0,
        force=True,
        printer=printed.append,
    ).run_once()

    action = cycle.actions[0]
    assert action.kind == "hold"
    assert action.verdict == "DISHONEST"
    assert venue.stakes == []
    assert printed == []


def test_force_trade_does_not_reach_a_vault():
    """A vault staked twice stakes the same money twice; the flag is Aave's alone."""
    venue = BookVenue(published=[vault()], source={VAULT: FEED_LABEL})
    cycle, printed = forced(venue, signal(verdict="UNMEASURED"), dry_run=False)

    assert cycle.actions[0].kind == "hold"
    assert venue.stakes == []
    assert printed == []


def test_force_trade_reports_a_refused_supply_honestly():
    """A refused send is not a supply: the line quotes the venue instead of claiming one."""
    venue = pool_venue(refuse=True)
    cycle, printed = forced(
        venue, signal(verdict="UNMEASURED", name=AAVE_NAME), dry_run=False
    )

    action = cycle.actions[0]
    assert action.sent is False
    assert "not in the address book" in action.reason
    assert printed == [
        "Force trade: I tried to supply $1.00 USDC to Aave and the venue refused — "
        "not in the address book."
    ]


def test_the_cli_force_flag_dry_runs_the_supply(monkeypatch, tmp_path, capsys):
    """``--force-trade --once`` is the rehearsal an operator runs before the first dollar."""
    venue = pool_venue()
    wire(monkeypatch, venue, signal(verdict="UNMEASURED", name=AAVE_NAME))
    monkeypatch.setenv(C.ENV_MICRO_CAP, "1.0")

    assert L.main(["--once", "--force-trade", "--state", str(tmp_path / "s.json")]) == 0

    printed = capsys.readouterr().out
    assert "Force trade: I would supply $1.00 USDC to Aave." in printed
    assert "FORCE_TRADE is on" in printed
    assert venue.stakes == []                          # DRY_RUN is still the default


def test_force_trade_comes_from_the_environment_and_the_flag_wins(
    monkeypatch, tmp_path, capsys
):
    """``FORCE_TRADE=1`` in ``.env``, and ``--force-trade`` on the run that needs it."""
    venue = pool_venue()
    wire(monkeypatch, venue, signal(verdict="UNMEASURED", name=AAVE_NAME))
    monkeypatch.setenv(C.ENV_MICRO_CAP, "1.0")
    monkeypatch.setenv(L.ENV_FORCE, "1")

    assert L.main(["--once", "--state", str(tmp_path / "a.json")]) == 0
    assert "Force trade: I would supply $1.00 USDC to Aave." in capsys.readouterr().out

    # the environment says no, the flag says yes, and the flag is the one that was typed
    monkeypatch.setenv(L.ENV_FORCE, "0")
    assert L.main(["--once", "--force-trade", "--state", str(tmp_path / "b.json")]) == 0
    assert "Force trade: I would supply $1.00 USDC to Aave." in capsys.readouterr().out

    # and with neither, nothing is forced — the quiet default is the safe one
    monkeypatch.delenv(L.ENV_FORCE)
    assert L.main(["--once", "--state", str(tmp_path / "c.json")]) == 0
    assert "Force trade" not in capsys.readouterr().out

