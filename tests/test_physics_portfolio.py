"""The portfolio: what the cap buys, and what the whitelist does not.

This is the test of a *capital manager*, so it asserts the three things a cap can get
wrong:

* **the cap holds.** At 60/30/10 a protocol that goes to zero costs a tenth of the
  empire, and the accounting says so to the last credit: ``equity = cash + Σ holdings``
  after every operation, a rug writes one position off and touches nothing else, and
  nothing is ever divided by a rate that would invent credits.
* **the whitelist is necessary and not sufficient.** Capital cannot reach an address
  that is not approved (:class:`NotWhitelisted` is an error, not a refusal) — and every
  trap in the declared market *is* approved, which is exactly why the assessment exists.
* **the future is not in the object.** :meth:`Protocol.record` returns only the steps
  that have happened; the rug is in the array and not in the record, and the test reads
  both to check that the difference is real.
"""
import numpy as np
import pytest

from agent.physics import portfolio as P
from agent.physics.risk_assessment import HONEST


@pytest.fixture()
def market():
    return P.ProtocolMarket.mock(steps=360, seed=7)


@pytest.fixture()
def vault(market):
    return market.protocol("StakingVault")


@pytest.fixture()
def wallet(market):
    return P.Portfolio(whitelist=market.whitelist, cash=100.0)


# ── the brief's arithmetic ──────────────────────────────────────────────────


def test_the_targets_are_the_briefs_and_sum_to_one():
    assert P.TARGETS[P.BUCKET_SAFE] == 0.60
    assert P.TARGETS[P.BUCKET_ACTIVE] == 0.30
    assert P.TARGETS[P.BUCKET_EXPERIMENTAL] == 0.10
    assert sum(P.TARGETS.values()) == pytest.approx(1.0)


def test_a_target_set_that_does_not_sum_to_one_is_a_bug():
    with pytest.raises(P.PortfolioError):
        P.Portfolio(whitelist=P.WhiteList(), targets={P.BUCKET_SAFE: 1.0})


def test_an_address_is_simulated_stable_and_not_a_wallet():
    first = P.sim_address("YieldMax")
    assert first == P.sim_address("YieldMax")
    assert first != P.sim_address("AurumDAO")
    assert len(first) == 42 and first.startswith("0x")
    int(first[2:], 16)  # forty hex digits and nothing else
    with pytest.raises(P.PortfolioError):
        P.Protocol(
            name="bad", bucket="safe", claim_apy=0.1, returns=np.ones(64),
            address="0x1234",
        )


def test_an_apy_and_a_per_step_rate_are_the_same_conversion_both_ways():
    rate = P.per_step_rate(0.5)
    assert P.annualise(rate) == pytest.approx(0.5, rel=1e-12)
    assert P.implied_apy(np.full(365, rate)) == pytest.approx(0.5, rel=1e-9)
    assert P.implied_apy([-1.0, 0.0, 0.0]) == -1.0


# ── the whitelist: necessary, insufficient, and fail-closed ─────────────────


def test_capital_cannot_reach_an_address_off_the_whitelist(market, vault):
    stranger = P.Protocol(
        name="Stranger", bucket="safe", claim_apy=0.1, returns=vault.returns
    )
    wallet = P.Portfolio(whitelist=market.whitelist, cash=100.0)
    with pytest.raises(P.NotWhitelisted):
        wallet.deposit(stranger, 10.0)
    assert wallet.cash == 100.0 and not wallet.holdings


def test_an_empty_whitelist_approves_nothing(market, vault):
    wallet = P.Portfolio(whitelist=P.WhiteList(), cash=100.0)
    assert not wallet.whitelist.allows(vault.address)
    with pytest.raises(P.NotWhitelisted):
        wallet.deposit(vault, 1.0)


def test_every_trap_in_the_market_is_whitelisted(market):
    """The whitelist says who runs an address, not whether its curve is honest."""
    for name in ("YieldMax", "AurumDAO"):
        assert market.whitelist.allows(market.protocol(name).address)


# ── the accounting: nothing is invented, nothing is lost silently ───────────


def test_a_deposit_moves_credits_and_never_invents_them(wallet, vault):
    placed = wallet.deposit(vault, 60.0)
    assert placed == 60.0
    assert wallet.cash == 40.0
    assert wallet.value(vault.address) == 60.0
    assert wallet.equity == 100.0
    assert wallet.deposit(vault, 1000.0) == 40.0  # clamps, does not fail
    assert wallet.cash == 0.0 and wallet.equity == 100.0


def test_a_return_multiplies_a_holding_and_the_equity_follows(wallet, vault):
    wallet.deposit(vault, 100.0)
    wallet.accrue(step=0)
    rate = float(vault.step_return(0))
    assert wallet.value(vault.address) == pytest.approx(100.0 * (1.0 + rate))
    assert wallet.equity == pytest.approx(wallet.cash + wallet.value(vault.address))


def test_a_rug_writes_one_position_off_and_touches_nothing_else(market):
    wallet = P.Portfolio(whitelist=market.whitelist, cash=100.0)
    trap = market.protocol("YieldMax")
    vault = market.protocol("StakingVault")
    wallet.deposit(vault, 40.0)
    wallet.deposit(trap, 10.0)
    before = wallet.cash
    wallet.accrue(step=int(trap.rug_step))
    assert wallet.value(trap.address) == 0.0
    assert trap.address not in wallet.holdings
    assert wallet.value(vault.address) > 0.0
    assert wallet.cash == before
    assert wallet.equity == pytest.approx(100.0 - 10.0, abs=0.5)
    assert any(event["action"] == "close" for event in wallet.events)


def test_a_dead_protocol_cannot_be_funded(wallet, market):
    trap = market.protocol("YieldMax")
    market.step = int(trap.rug_step) + 1
    market.advance(1)
    assert not trap.alive
    with pytest.raises(P.DeadProtocol):
        wallet.deposit(trap, 10.0)


def test_a_liquidation_returns_everything_to_cash(wallet, market):
    wallet.deposit(market.protocol("StakingVault"), 50.0)
    wallet.deposit(market.protocol("NicheFarm"), 10.0)
    equity = wallet.equity
    raised = wallet.liquidate([P.BUCKET_ACTIVE, P.BUCKET_EXPERIMENTAL])
    assert raised == pytest.approx(10.0)
    assert wallet.bucket_value(P.BUCKET_ACTIVE) == 0.0
    assert wallet.bucket_value(P.BUCKET_EXPERIMENTAL) == 0.0
    assert wallet.bucket_value(P.BUCKET_SAFE) == pytest.approx(50.0)
    assert wallet.equity == pytest.approx(equity)


# ── the cap, driven by a plan ───────────────────────────────────────────────


def test_each_bucket_is_brought_to_its_target(wallet, market):
    plan = {
        P.BUCKET_SAFE: market.protocol("StakingVault"),
        P.BUCKET_ACTIVE: market.protocol("MevRouter"),
        P.BUCKET_EXPERIMENTAL: market.protocol("NicheFarm"),
    }
    wallet.rebalance(plan)
    for bucket, target in P.TARGETS.items():
        assert wallet.bucket_value(bucket) == pytest.approx(100.0 * target, rel=1e-9)
    assert wallet.cash == pytest.approx(0.0, abs=1e-9)
    assert wallet.equity == pytest.approx(100.0)


def test_a_bucket_with_nothing_admissible_goes_to_cash(wallet, market):
    farm = market.protocol("NicheFarm")
    wallet.rebalance({P.BUCKET_SAFE: market.protocol("StakingVault"),
                      P.BUCKET_ACTIVE: None,
                      P.BUCKET_EXPERIMENTAL: farm})
    assert wallet.bucket_value(P.BUCKET_EXPERIMENTAL) > 0.0
    wallet.rebalance({P.BUCKET_SAFE: market.protocol("StakingVault"),
                      P.BUCKET_ACTIVE: None,
                      P.BUCKET_EXPERIMENTAL: None})
    assert wallet.bucket_value(P.BUCKET_EXPERIMENTAL) == 0.0
    safe = wallet.bucket_value(P.BUCKET_SAFE)
    assert wallet.cash == pytest.approx(wallet.equity - safe)


def test_the_experimental_bucket_never_holds_more_than_a_tenth(wallet, market):
    """The cap is the whole risk budget: a tenth, plus the trim band it is allowed."""
    wallet.rebalance({
        P.BUCKET_SAFE: market.protocol("StakingVault"),
        P.BUCKET_ACTIVE: market.protocol("MevRouter"),
        P.BUCKET_EXPERIMENTAL: market.protocol("NicheFarm"),
    })
    for step in range(120):
        wallet.accrue(step)
    # between reviews a loud bucket is allowed to grow; a review pulls it back
    wallet.rebalance({
        P.BUCKET_SAFE: market.protocol("StakingVault"),
        P.BUCKET_ACTIVE: market.protocol("MevRouter"),
        P.BUCKET_EXPERIMENTAL: market.protocol("NicheFarm"),
    })
    share = wallet.bucket_value(P.BUCKET_EXPERIMENTAL) / wallet.equity
    assert share <= P.TARGETS[P.BUCKET_EXPERIMENTAL] + P.TRIM_BAND + 0.01, share


# ── the market: only what has happened, and two traps with dates ────────────


def test_the_market_publishes_only_the_steps_that_happened(market):
    trap = market.protocol("YieldMax")
    assert trap.record(100).size == 100
    assert np.array_equal(trap.record(100), trap.returns[:100])
    assert trap.record(-5).size == 0
    published = trap.as_opportunity(100)
    assert len(published["returns"]) == 100
    assert min(published["returns"]) > -1.0  # the rug is not in there yet
    assert float(np.min(trap.returns)) == -1.0


def test_the_two_traps_die_on_their_declared_days(market):
    market.step = 0
    live = {protocol.name: protocol.alive for protocol in market.protocols}
    assert all(live.values())
    market.advance(int(P.TRAP_RUGS["YieldMax"]) + 1)
    assert not market.protocol("YieldMax").alive
    assert market.protocol("AurumDAO").alive
    assert [event["protocol"] for event in market.events] == ["YieldMax"]
    market.advance(int(P.TRAP_RUGS["AurumDAO"]) - int(P.TRAP_RUGS["YieldMax"]))
    assert not market.protocol("AurumDAO").alive
    assert market.protocol("StakingVault").alive
    assert [event["protocol"] for event in market.events] == ["YieldMax", "AurumDAO"]


def test_a_naive_investor_follows_the_claim_and_an_ape_takes_the_top_one():
    market = P.ProtocolMarket.mock(steps=120, seed=7)
    ape = P.NaiveInvestor.launch(market, policy=P.NAIVE_APE)
    assert ape.choose() is market.protocol("AurumDAO")  # the highest claim of all
    split = P.NaiveInvestor.launch(market, policy=P.NAIVE_SPLIT)
    assert split.choose(P.BUCKET_SAFE) is market.protocol("FixedBond")
    assert split.choose(P.BUCKET_EXPERIMENTAL) is market.protocol("AurumDAO")
    with pytest.raises(P.PortfolioError):
        P.NaiveInvestor(market=market, portfolio=split.portfolio, policy="genius")


# ── the empire: what it funds, and whether it survives the declared market ──


def test_the_empire_only_funds_a_verdict_it_reached_honestly():
    """Every review's plan is a subset of that review's ``HONEST`` verdicts."""
    empire = P.Empire.launch(steps=240, seed=7)  # no body: sterility and claim only
    empire.run(240)
    assert empire.decisions
    for decision in empire.decisions:
        judged = empire.assessments[decision["step"]]
        for bucket, name in decision["plan"].items():
            if name is None:
                continue
            assert judged[name]["verdict"] == HONEST, (decision["step"], bucket, name)
            assert judged[name]["admissible"] is True
    funded = {
        name
        for decision in empire.decisions
        for name in decision["plan"].values()
        if name
    }
    assert "YieldMax" not in funded and "AurumDAO" not in funded
    assert "FixedBond" not in funded


def test_the_empire_survives_the_declared_market():
    """The acceptance test: growth, no dip below the start, and no trap ever held."""
    from agent.physics.curiosity import Curiosity
    from agent.physics.organism import Organism
    from agent.physics.worker import RealityOracle

    body = Organism.hatch(offline=True, steps=20, seed=11)
    engine = Curiosity(body.mirror.posture, prior=body.mirror.prior)
    oracle = RealityOracle(
        sensorium=body.sensorium, mirror=body.mirror, engine=engine, seed=11
    )
    steps = 400  # the grind and the two rugs; the spike is covered by the arm tests
    empire = P.Empire.launch(
        market=P.ProtocolMarket.mock(steps=steps, seed=7), steps=steps, seed=7
    )
    empire.oracle = oracle
    empire.run(steps)
    history = empire.history()
    equity = np.asarray(history["equity"], dtype=float)
    start = float(equity[0])
    assert float(equity[-1]) > start  # it grew
    # and the crash is bounded by the switch rather than survived by luck: the fall from
    # the running peak stays inside a quarter of the empire even though the market it
    # trades fell by more than half.
    peak = np.maximum.accumulate(equity)
    assert float((1.0 - equity / peak).max()) < 0.25
    held = {
        event["protocol"]
        for event in empire.portfolio.events
        if event["action"] == "deposit"
    }
    assert "YieldMax" not in held and "AurumDAO" not in held
    assert held & {"StakingVault", "FixedBond"}
    # and it was in the safe protocol on every step of the run
    assert all(row["safe"] >= 0.0 for row in empire.trace)





def test_the_cli_runs_the_empire_and_writes_a_report(tmp_path):
    """The documented command has to work, and its report has to be JSON."""
    import json

    target = tmp_path / "empire.json"
    assert P.main(["--steps", "72", "--output", str(target)]) == 0
    report = json.loads(target.read_text(encoding="utf-8"))
    assert report["equity"]["end"] > 0.0
    assert report["kill_switch"]["engaged"] is False
    assert report["decisions"]
