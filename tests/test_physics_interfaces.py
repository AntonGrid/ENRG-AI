"""The adapter: the same decisions, and a venue that can be replaced in one line.

This is the test of a *seam*, and a seam is only worth writing if both of its sides can
be held to something. So it holds three:

* **the contract.** Both providers answer :class:`IWeb3Provider`, and so does the
  one-line subclass a third party would write: the layer above asks for five operations
  and three facts, not for a particular class;
* **the same numbers.** :class:`MockWeb3Provider` publishes exactly the declared
  market — the seven protocols, the same seeds, the same two traps at the same rug
  steps — and an empire launched through the adapter ends on the same credit as one
  launched through the market. Backwards compatibility is asserted, not argued;
* **nothing that can move real money.** The adapter's imports are parsed out of the file
  and checked: ``typing`` and (lazily) ``portfolio``, and no ``web3``, no ``requests``,
  no endpoint, no key. A provider that *could* sign would be a provider that could lose
  money, and this file is the proof that the one shipped cannot.

The negative cases matter as much as the positive ones: a venue that refuses a stake is
an :class:`~agent.physics.portfolio.StakeRefused` and not a position, a refusal is never
recorded as a success, and a proof for an address the venue does not publish is
``False`` rather than a claim to have transmitted something.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from agent.physics import portfolio as P
from agent.physics import risk_assessment as R
from agent.physics.interfaces import IWeb3Provider, MockWeb3Provider, NoVenueProvider

INTERFACES = Path(__file__).resolve().parents[1] / "agent" / "physics" / "interfaces.py"

#: Everything a provider must answer, spelled once so a rename cannot pass silently.
CONTRACT = (
    "get_protocol_list",
    "fetch_protocol_history",
    "submit_validation_proof",
    "execute_stake",
    "withdraw",
    "advance",
    "opportunities",
    "alive",
    "protocol",
    "records",
    "describe",
)


@pytest.fixture()
def provider():
    return MockWeb3Provider(steps=360, seed=7)


@pytest.fixture()
def market(provider):
    return provider.market


class RefusingVenue(MockWeb3Provider):
    """A venue that says no to capital — the one thing the mock never does itself."""

    def execute_stake(self, protocol_address: str, amount: float) -> bool:
        return False


class OneWayVenue(MockWeb3Provider):
    """A venue that takes capital in and will not let it out: a paused contract."""

    def withdraw(self, protocol_address: str, amount: float) -> bool:
        return False


class SafeOnlyVenue(MockWeb3Provider):
    """A venue that publishes one bucket; for the agent, the rest does not exist."""

    def get_protocol_list(self):
        return [
            protocol
            for protocol in super().get_protocol_list()
            if protocol.bucket == P.BUCKET_SAFE
        ]


class ShortMemoryVenue(MockWeb3Provider):
    """A venue that keeps only the last few points of every record.

    Nodes prune. An adapter over a real chain can be asked for a history nobody stored
    any more, and what comes back is a stub — which is precisely the situation
    ``risk_assessment`` refuses to have an opinion about. Building it here is how the
    refusal is measured instead of assumed.
    """

    RETENTION = 32

    def get_protocol_list(self):
        published = []
        for protocol in super().get_protocol_list():
            record = protocol.record(self.step)[-self.RETENTION :]
            if record.size == 0:
                # A node with nothing stored about an address does not list it; the
                # protocol type refuses an empty record, and so does this adapter.
                continue
            published.append(
                P.Protocol(
                    name=protocol.name,
                    bucket=protocol.bucket,
                    claim_apy=protocol.claim_apy,
                    returns=record,
                    address=protocol.address,
                    alive=protocol.alive,
                    rug_step=protocol.rug_step,
                )
            )
        return published


# ── the contract ────────────────────────────────────────────────────────────


def test_both_providers_answer_the_interface():
    for venue in (MockWeb3Provider(), NoVenueProvider()):
        assert isinstance(venue, IWeb3Provider)
        for name in CONTRACT:
            assert callable(getattr(venue, name)), name
        assert isinstance(venue.step, int)
        assert isinstance(venue.whitelist, P.WhiteList)
        assert venue.name in ("mock", "none")


def test_an_adapter_subclasses_the_interface_for_the_read_helpers(market):
    """Five operations, three facts, and the read helpers come for free.

    An object wrapping ``web3.py`` has no reason to know this package exists, and it
    does not have to: subclassing the interface costs one line and supplies
    ``opportunities``/``alive``/``protocol``/``records``/``describe`` from the list it
    publishes, so the layer above can never meet a provider that reads a record
    differently from another one.
    """

    class Wrapped(IWeb3Provider):
        name = "wrapped"
        step = 0

        def __init__(self, whitelist):
            self._whitelist = whitelist

        @property
        def whitelist(self):
            return self._whitelist

        def get_protocol_list(self):
            return []

        def fetch_protocol_history(self, address, steps=None):
            return []

        def submit_validation_proof(self, protocol_address, proof_data):
            return False

        def execute_stake(self, protocol_address, amount):
            return True

        def withdraw(self, protocol_address, amount):
            return True

        def advance(self, steps=1):
            return 0

    venue = Wrapped(market.whitelist)
    assert isinstance(venue, IWeb3Provider)
    assert venue.opportunities() == []
    assert venue.describe() == {"name": "wrapped", "step": 0, "protocols": 0}
    book = P.Portfolio(provider=venue, whitelist=market.whitelist, cash=50.0)
    assert book.deposit(market.protocol("StakingVault"), 20.0) == pytest.approx(20.0)
    assert book.cash == pytest.approx(30.0)
    assert R.review(venue) == {}  # a venue publishing nothing is a refusal, not a crash


# ── the same market, behind the seam ────────────────────────────────────────


def test_the_mock_publishes_exactly_the_declared_market(provider):
    """The adapter returns the same data — names, addresses, claims and records."""
    declared = P.ProtocolMarket.mock(steps=360, seed=7)
    published = provider.get_protocol_list()
    assert [item.name for item in published] == [
        item.name for item in declared.protocols
    ]
    for mine, theirs in zip(published, declared.protocols):
        assert mine.address == theirs.address
        assert mine.claim_apy == pytest.approx(theirs.claim_apy)
        assert np.allclose(mine.returns, theirs.returns)
    assert {"YieldMax", "AurumDAO"} <= {item.name for item in published}


def test_the_two_traps_keep_their_dates_through_the_adapter():
    venue = MockWeb3Provider(steps=720, seed=7)
    for name, rug in P.TRAP_RUGS.items():
        assert venue.protocol(name).rug_step == rug
        assert venue.protocol(name).returns[int(rug)] == -1.0


def test_a_record_never_runs_past_the_venues_clock(provider):
    """The clock is the frontier: a step that has not come cannot be published."""
    trap = provider.protocol("YieldMax")
    assert provider.fetch_protocol_history(trap.address) == []
    provider.advance(120)
    published = provider.fetch_protocol_history(trap.address)
    assert len(published) == 120
    assert provider.fetch_protocol_history(trap.address, 10) == pytest.approx(
        [float(value) for value in trap.returns[110:120]]
    )
    # the rug is in the array and outside the published window: the future is behind the
    # clock, exactly as it is behind the next block on a chain
    assert trap.returns[300] == -1.0
    assert len(published) < 300


def test_an_unpublished_address_is_an_empty_answer_and_not_an_exception(provider):
    stranger = "0x" + "0" * 40
    assert provider.fetch_protocol_history(stranger) == []
    assert provider.fetch_protocol_history(stranger, 5) == []
    assert provider.submit_validation_proof(stranger, {"verdict": R.SCAM_TRAP}) is False
    assert provider.execute_stake(stranger, 10.0) is False
    assert provider.withdraw(stranger, 10.0) is False
    # three refusals and not four: reading an unpublished address is an empty answer,
    # and the two operations that would move value are the ones that say no
    assert provider.refusals == 3


# ── capital, moved only through the interface ───────────────────────────────


def test_the_book_moves_capital_only_through_the_provider(provider, market):
    book = P.Portfolio(provider=provider, whitelist=market.whitelist, cash=100.0)
    vault = market.protocol("StakingVault")
    assert book.provider is provider
    assert book.deposit(vault, 60.0, step=0) == pytest.approx(60.0)
    assert provider.stakes == [{"step": 0, "address": vault.address, "amount": 60.0}]
    assert book.cash == pytest.approx(40.0)
    assert book.withdraw(vault.address, 15.0) == pytest.approx(15.0)
    assert provider.withdrawals[-1]["amount"] == pytest.approx(15.0)
    assert book.cash == pytest.approx(55.0)
    # a short balance clamps, and the venue sees exactly what the book placed
    assert book.deposit(vault, 1000.0) == pytest.approx(55.0)
    assert provider.stakes[-1]["amount"] == pytest.approx(55.0)


def test_a_venue_that_refuses_a_stake_leaves_no_position(market):
    venue = RefusingVenue(market)
    book = P.Portfolio(provider=venue, whitelist=market.whitelist, cash=100.0)
    with pytest.raises(P.StakeRefused):
        book.deposit(market.protocol("StakingVault"), 10.0)
    assert book.cash == 100.0
    assert not book.holdings
    assert not [row for row in book.events if row["action"] == "deposit"]


def test_a_venue_that_refuses_a_sale_cannot_be_talked_into_one(market):
    """The kill switch's exit is the same call, so it fails the same way: loudly."""
    venue = OneWayVenue(market)
    book = P.Portfolio(provider=venue, whitelist=market.whitelist, cash=100.0)
    vault = market.protocol("StakingVault")
    book.deposit(vault, 30.0)
    with pytest.raises(P.StakeRefused):
        book.withdraw(vault.address, 10.0)
    with pytest.raises(P.StakeRefused):
        book.liquidate([P.BUCKET_SAFE])
    assert book.value(vault.address) == pytest.approx(30.0)


def test_a_bare_book_moves_capital_through_the_no_venue_provider(market):
    """No venue is an implementation of the interface, not a branch that skips it."""
    book = P.Portfolio(whitelist=market.whitelist, cash=100.0)
    venue = book.provider
    assert isinstance(venue, NoVenueProvider) and venue.name == "none"
    assert venue.get_protocol_list() == [] and venue.opportunities() == []
    assert venue.advance(10) == 0 and venue.step == 0
    vault = market.protocol("StakingVault")
    assert book.deposit(vault, 25.0) == pytest.approx(25.0)
    assert venue.stakes[-1]["amount"] == pytest.approx(25.0)
    # a proof with nowhere to go is refused, never counted as sent
    assert venue.submit_validation_proof(
        vault.address, {"verdict": R.SCAM_TRAP}
    ) is False


# ── the empire: one venue, handed to the book ────────────────────────────────


def test_the_empire_hands_its_venue_to_the_book(market):
    book = P.Portfolio(whitelist=market.whitelist, cash=100.0)
    assert isinstance(book.provider, NoVenueProvider)  # nothing attached yet
    empire = P.Empire(market=market, portfolio=book, kill_switch=R.KillSwitch())
    assert empire.portfolio.provider is empire.provider
    assert isinstance(empire.provider, MockWeb3Provider)


def test_two_venues_in_one_run_are_refused(market):
    """Assessing one market and trading another would make every number meaningless."""
    other = MockWeb3Provider(P.ProtocolMarket.mock(steps=360, seed=7))
    with pytest.raises(P.PortfolioError):
        P.Empire.launch(market=market, provider=other)
    book = P.Portfolio(provider=MockWeb3Provider(market), whitelist=market.whitelist)
    with pytest.raises(P.PortfolioError):
        P.Empire(  # a book that already trades somewhere else
            market=market,
            portfolio=book,
            kill_switch=R.KillSwitch(),
            provider=MockWeb3Provider(P.ProtocolMarket.mock(steps=360, seed=7)),
        )


def test_the_empire_files_its_convictions_with_the_venue():
    """The fourth operation is used: a conviction is published, and counted once."""
    venue = MockWeb3Provider(steps=400, seed=7)
    empire = P.Empire.launch(provider=venue)
    empire.run(400)
    assert empire.provider is venue and empire.portfolio.provider is venue
    filed = {row["protocol"]: row for row in empire.proofs}
    # Without a body the structural arm is blind, so the conviction on the record comes
    # from the sterility arm alone — the notebook's AurumDAO verdict needs the oracle.
    # What is asserted here is the plumbing, not the detection: the conviction travels.
    assert "YieldMax" in filed
    for row in empire.proofs:
        assert row["verdict"] in R.CONVICTIONS
        assert row["reason"]
        assert row["points"] >= P.MIN_RECORD
        assert venue.whitelist.allows(row["address"])
    assert len(empire.proofs) == len(venue.proofs)  # nothing counted that never left
    assert all(decision["proofs"] >= 0 for decision in empire.decisions)
    filed_by_the_empire = sum(decision["proofs"] for decision in empire.decisions)
    assert filed_by_the_empire == len(empire.proofs)


def test_every_position_the_empire_holds_was_a_stake_the_venue_saw():
    venue = MockWeb3Provider(steps=240, seed=7)
    empire = P.Empire.launch(provider=venue)
    empire.run(240)
    deposits = [row for row in empire.portfolio.events if row["action"] == "deposit"]
    assert deposits and len(venue.stakes) == len(deposits)
    assert venue.refusals == 0  # the book never asked for something the venue forbids
    assert empire.report()["provider"]["stakes"] == len(venue.stakes)


def test_the_adapter_does_not_move_a_single_number():
    """The same life, with and without the injection, ends on the same credit."""
    through_market = P.Empire.launch(market=P.ProtocolMarket.mock(steps=240, seed=7))
    through_market.run(240)
    through_venue = P.Empire.launch(provider=MockWeb3Provider(steps=240, seed=7))
    through_venue.run(240)
    assert through_venue.portfolio.equity == pytest.approx(
        through_market.portfolio.equity, rel=1e-12
    )
    assert [row["equity"] for row in through_venue.trace] == pytest.approx(
        [row["equity"] for row in through_market.trace]
    )


def test_a_narrower_venue_narrows_the_empire_and_nothing_else():
    """Swap the provider, and the agent's world changes — not a line of the rules."""
    empire = P.Empire.launch(provider=SafeOnlyVenue(steps=240, seed=7))
    empire.run(240)
    funded = {
        name
        for decision in empire.decisions
        for name in decision["plan"].values()
        if name
    }
    assert funded <= {"StakingVault", "FixedBond"}
    assert all(row["active"] == 0.0 for row in empire.trace)
    assert all(row["experimental"] == 0.0 for row in empire.trace)
    assert empire.portfolio.equity > 0.0


def test_a_venue_with_pruned_records_makes_the_agent_hold_cash():
    """Short history is not an invitation to guess: every bucket stays in cash."""
    empire = P.Empire.launch(provider=ShortMemoryVenue(steps=240, seed=7))
    empire.run(240)
    assert empire.decisions
    assert all(
        name is None
        for decision in empire.decisions
        for name in decision["plan"].values()
    )
    assert not empire.provider.stakes
    assert empire.portfolio.equity == pytest.approx(P.START_CAPITAL)
    judged = next(iter(empire.assessments.values()))
    assert {row["verdict"] for row in judged.values()} == {R.UNMEASURED}


# ── what the file is not allowed to contain ─────────────────────────────────


def test_the_adapter_imports_nothing_that_can_speak_to_a_network():
    """Parsed, not promised: ``typing``, and ``portfolio`` only where it is needed."""
    imports = set()
    for node in ast.walk(ast.parse(INTERFACES.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert imports <= {"__future__", "typing", "agent.physics.portfolio"}, imports


def test_there_is_no_endpoint_and_no_key_in_the_adapter():
    source = INTERFACES.read_text(encoding="utf-8").lower()
    for token in ("http://", "https://", "private_key", "privatekey", "secret_key"):
        assert token not in source, token
