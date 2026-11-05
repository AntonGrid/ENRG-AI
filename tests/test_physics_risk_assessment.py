"""The risk assessment: whether a yield curve is a record or a formula.

This is the test of a *verdict about somebody else's numbers*, so it asserts the four
things such a verdict can get wrong:

* **the sterility floor separates a record from a formula.** A contractual fixed rate
  and a Ponzi both have no residuals in them; a staking vault and a degen farm both do.
  If the statistic cannot make that cut, nothing downstream matters.
* **the two arms must agree before anything is called a scam.** Measured here: a bond is
  sterile and the assessment refuses it as ``SYNTHETIC_COLLUSION``, never as a trap —
  the difference between the two verdicts is the claim, because no statistic of a record
  can tell a promise that is kept from a promise that is not.
* **a claim the record does not support is a suspicion, not a conviction.** The bounty
  board pays 26 % over its first sixty-four steps while claiming 177 %; the honest
  answer is "wait", and the test pins that the answer is ``SUSPECT`` rather than
  ``SCAM_TRAP``.
* **the switch fires on each of its three arms, alone.** A drawdown with a calm market,
  a volatility regime with no loss yet, and a collapsed resonance with neither — three
  independent measurements, three independent reasons, and one action.
"""
import numpy as np
import pytest

from agent.physics import portfolio as P
from agent.physics import risk_assessment as R
from agent.physics.organism import Organism
from agent.physics.curiosity import Curiosity
from agent.physics.worker import RealityOracle


@pytest.fixture(scope="module")
def oracle():
    """One body, its mirror, and the oracle that prices both contracts and records."""
    body = Organism.hatch(offline=True, steps=20, seed=11)
    engine = Curiosity(body.mirror.posture, prior=body.mirror.prior)
    return RealityOracle(
        sensorium=body.sensorium, mirror=body.mirror, engine=engine, seed=11
    )


@pytest.fixture(scope="module")
def market():
    return P.ProtocolMarket.mock(steps=720, seed=7)


def opportunity(returns, *, claim, bucket="experimental", name="Test", alive=True):
    """A published opportunity, in the shape an adapter would hand over."""
    return {
        "name": name,
        "address": P.sim_address(name),
        "bucket": bucket,
        "claim_apy": float(claim),
        "returns": [float(value) for value in np.asarray(returns, dtype=float)],
        "alive": bool(alive),
    }


# ── the sterility floor: a record has residuals, a formula does not ──────────


def test_the_sterility_floor_separates_a_formula_from_a_record():
    rng = np.random.default_rng(3)
    lived = P.staking_returns(rng, 400)
    formula = P.formula_returns(rng, 400)
    a2_lived, cv_lived, points = R.sterility_of(lived)
    a2_formula, cv_formula, _ = R.sterility_of(formula)
    assert points == 400
    assert 1.0 < a2_lived < 10.0, a2_lived
    assert cv_lived > 10.0 * R.STERILE_CV, cv_lived
    assert cv_formula < R.STERILE_CV / 10.0, cv_formula
    assert a2_formula < 2.0


def test_a_fixed_rate_is_refused_without_being_called_a_scam(market):
    bond = market.protocol("FixedBond")
    verdict = R.evaluate_opportunity(bond.as_opportunity(200))
    assert verdict.verdict == R.SYNTHETIC_COLLUSION
    assert verdict.sterile
    assert not verdict.promised
    assert "CV" in verdict.reason and "sterile" in verdict.reason
    assert not verdict.admissible


def test_a_thousand_percent_with_no_noise_in_it_is_a_scam_trap(market):
    trap = market.protocol("YieldMax")
    verdict = R.evaluate_opportunity(trap.as_opportunity(200))
    assert verdict.verdict == R.SCAM_TRAP
    assert verdict.sterile and verdict.promised
    assert verdict.claim_apy > R.PLAUSIBLE_APY["experimental"]
    assert verdict.scam and not verdict.admissible


def test_the_reason_only_quotes_comparisons_that_are_true(market):
    """``sterile`` is an *or*: a reason that prints both arms can print a lie.

    ``YieldMax`` carries ``A²`` of order one and ``CV`` of ``1e-4`` — the spread arm is
    the one that fires, the residual arm is not, and the printed reason has to say so.
    """
    for name in ("FixedBond", "YieldMax", "NicheFarm"):
        verdict = R.evaluate_opportunity(market.protocol(name).as_opportunity(200))
        if "sterile (" not in verdict.reason:
            continue
        arms = verdict.reason.split("sterile (", 1)[1].split(")", 1)[0]
        if "A²" in arms:
            assert verdict.a2 < R.STERILE_A2, verdict.reason
        if "CV" in arms:
            assert verdict.cv < R.STERILE_CV, verdict.reason
    trap = R.evaluate_opportunity(market.protocol("YieldMax").as_opportunity(200))
    assert "CV" in trap.reason and "A²" not in trap.reason, trap.reason
    assert trap.a2 > R.STERILE_A2


def test_a_noisy_farm_claiming_the_same_kind_of_yield_is_honest(market):
    farm = market.protocol("NicheFarm")
    verdict = R.evaluate_opportunity(farm.as_opportunity(200))
    assert verdict.verdict == R.HONEST, verdict.reason
    assert not verdict.sterile and not verdict.promised
    assert verdict.admissible


def test_the_same_record_is_a_bond_or_a_ponzi_depending_only_on_the_claim():
    """The measured limit of Statement #87: sterility alone cannot convict."""
    rng = np.random.default_rng(5)
    curve = P.bond_returns(rng, 400, rate=5.0e-4)
    honest = R.evaluate_opportunity(
        opportunity(curve, claim=P.annualise(5.0e-4), bucket="safe")
    )
    absurd = R.evaluate_opportunity(
        opportunity(curve, claim=10.0, bucket="experimental")
    )
    assert honest.verdict == R.SYNTHETIC_COLLUSION
    assert absurd.verdict == R.SCAM_TRAP
    assert honest.a2 == pytest.approx(absurd.a2)
    assert honest.cv == pytest.approx(absurd.cv)


# ── the structural and mirror arms: through the body, on a real record ───────


def test_a_replayed_record_is_caught_by_the_structural_arm(oracle, market):
    """The same measurement that names a re-served contract names a re-served record."""
    trap = market.protocol("AurumDAO")
    verdict = R.evaluate_opportunity(trap.as_opportunity(340), oracle=oracle)
    assert verdict.verdict == R.SCAM_TRAP, verdict.reason
    assert verdict.repeat
    assert verdict.reducible >= R.REPEAT_R2, verdict.reducible
    assert verdict.structure > 5.0, verdict.structure
    assert verdict.reality < R.REALITY_FLOOR, verdict.reality
    assert verdict.resonance < R.REALITY_FLOOR, verdict.resonance
    assert "repeats" in verdict.reason


def test_an_honest_record_is_named_by_nothing(oracle, market):
    staking = market.protocol("StakingVault")
    verdict = R.evaluate_opportunity(staking.as_opportunity(400), oracle=oracle)
    assert verdict.verdict == R.HONEST, verdict.reason
    assert not verdict.repeat
    assert verdict.reducible < 0.5, verdict.reducible
    assert verdict.reality > R.REALITY_FLOOR, verdict.reality
    assert verdict.resonance > R.RESONANCE_FLOOR, verdict.resonance


def test_lookalike_replayed_numbers_are_not_caught_without_a_body(market):
    trap = market.protocol("AurumDAO")
    blind = R.evaluate_opportunity(trap.as_opportunity(340))
    seeing = R.evaluate_opportunity(trap.as_opportunity(340), oracle=_stub_oracle())
    # The claim alone can only accuse: the record is loud, and nothing without a body
    # can see that the loudness is a loop. The second arm is what convicts.
    assert blind.verdict == R.SUSPECT
    assert seeing.verdict == R.SCAM_TRAP
    assert seeing.repeat and not blind.repeat
    assert blind.details["measured"] is False
    assert seeing.details["measured"] is True


def _stub_oracle():
    """A body is expensive; one fixture-sized hatch is enough for a two-arm check."""
    body = Organism.hatch(offline=True, steps=4, seed=3)
    engine = Curiosity(body.mirror.posture, prior=body.mirror.prior)
    return RealityOracle(sensorium=body.sensorium, mirror=body.mirror, engine=engine)


# ── what the claim is allowed to say ────────────────────────────────────────


def test_a_claim_the_record_does_not_support_is_suspicion_not_a_conviction():
    rng = np.random.default_rng(9)
    modest = rng.normal(1.0e-3, 5.0e-3, size=400)
    verdict = R.evaluate_opportunity(opportunity(modest, claim=10.0, bucket="active"))
    assert verdict.inflated and verdict.promised
    assert verdict.claim_z > R.CLAIM_Z, verdict.claim_z
    assert not verdict.sterile and not verdict.repeat
    assert verdict.verdict == R.SUSPECT
    assert not verdict.scam


def test_a_claim_inside_three_standard_errors_is_not_inflated():
    rng = np.random.default_rng(10)
    noisy = rng.normal(2.0e-3, 1.0e-2, size=400)
    verdict = R.evaluate_opportunity(
        opportunity(noisy, claim=P.annualise(2.0e-3), bucket="active")
    )
    assert not verdict.inflated, verdict.claim_z
    assert verdict.verdict == R.HONEST, verdict.reason


def test_a_record_that_lost_money_cannot_be_sold_as_a_profit():
    rng = np.random.default_rng(11)
    losing = rng.normal(-4.0e-3, 6.0e-3, size=300)
    verdict = R.evaluate_opportunity(opportunity(losing, claim=0.9, bucket="active"))
    assert verdict.ghost
    assert verdict.verdict in (R.SUSPECT, R.SCAM_TRAP)
    assert not verdict.admissible


def test_a_record_too_short_to_judge_is_unmeasured(market):
    staking = market.protocol("StakingVault")
    verdict = R.evaluate_opportunity(staking.as_opportunity(20))
    assert verdict.verdict == R.UNMEASURED
    assert not verdict.measured
    assert not verdict.admissible
    assert verdict.points == 20


def test_a_dead_protocol_is_refused_for_being_dead(market):
    trapless = market.protocol("YieldMax")
    dead = dict(trapless.as_opportunity(400), alive=False)
    verdict = R.evaluate_opportunity(dead)
    assert verdict.verdict == R.SUSPECT
    assert "stopped paying" in verdict.reason


def test_the_assessment_cannot_see_a_step_that_has_not_happened(market):
    """The rug is in the array and not in the record: that is the whole property."""
    trap = market.protocol("YieldMax")
    published = trap.as_opportunity(200)
    assert len(published["returns"]) == 200
    assert min(published["returns"]) > -1.0
    assert float(np.min(trap.returns)) == -1.0
    verdict = R.evaluate_opportunity(published)
    assert verdict.points == 200
    assert verdict.details["alive"] is True


# ── the switch: three arms, one action ──────────────────────────────────────


def test_the_switch_fires_on_fifteen_percent_per_cycle():
    switch = R.KillSwitch()
    trigger = switch.check(step=10, equity=84.0, reference=100.0)
    assert trigger is not None
    assert trigger.arm == R.ARM_DRAWDOWN
    assert trigger.drawdown == pytest.approx(0.16)
    assert switch.engaged
    assert "15%" in trigger.reason


def test_a_thirteen_percent_fall_is_not_yet_a_fire():
    switch = R.KillSwitch()
    assert switch.check(step=10, equity=87.0, reference=100.0) is None
    assert not switch.engaged


def test_the_switch_fires_on_a_volatility_regime_change_without_a_loss():
    switch = R.KillSwitch()
    trigger = switch.check(step=10, equity=101.0, reference=100.0, volatility=3.4)
    assert trigger is not None
    assert trigger.arm == R.ARM_VOLATILITY
    assert trigger.drawdown == 0.0
    assert "3.4x" in trigger.reason


def test_the_switch_fires_when_the_resonance_collapses():
    switch = R.KillSwitch()
    trigger = switch.check(step=10, equity=101.0, reference=100.0, resonance=0.0)
    assert trigger is not None
    assert trigger.arm == R.ARM_RESONANCE
    assert "0.00" in trigger.reason


def test_an_unmeasured_arm_cannot_fire():
    """Not measuring a risk is not the same as measuring it as zero."""
    switch = R.KillSwitch()
    assert switch.check(step=10, equity=101.0, reference=100.0) is None
    assert not switch.engaged


def test_the_switch_stays_engaged_until_the_market_is_calm():
    switch = R.KillSwitch()
    assert switch.check(step=10, equity=84.0, reference=100.0) is not None
    assert switch.check(step=11, equity=83.0, reference=100.0) is None  # already out
    assert switch.release(
        step=12, equity=95.0, reference=100.0, volatility=4.0
    ) is None  # calm price, wild market
    released = switch.release(step=13, equity=100.0, reference=100.0, volatility=1.1)
    assert released is not None
    assert released.action == "reinvest"
    assert not switch.engaged
    assert [event.arm for event in switch.triggers] == ["drawdown", "release"]


def test_a_jackpot_is_not_a_regime_change():
    """One protocol's payout must not trip a market-wide switch."""
    market = P.ProtocolMarket.mock(steps=720, seed=7)
    empire = P.Empire.launch(market=market)
    # The clock belongs to the venue: the arm measures what the provider publishes at
    # *its* step, so asking about step 301 means letting 301 steps happen first. Before
    # the adapter this was ``empire.step = 301`` alone; the number the arm returns is the
    # same one, because in a real run the two clocks move together.
    empire.provider.advance(301)
    empire.step = 301
    before = empire.market_volatility()
    market.protocol("BountyBoard").returns[300] = 0.25
    after = empire.market_volatility()
    assert before is not None and after is not None
    assert after < R.KILL_VOLATILITY, after


def test_the_volatility_ratio_survives_its_own_baseline():
    """A baseline that includes the crash cannot see the crash."""
    rng = np.random.default_rng(13)
    calm = rng.normal(1.0e-3, 5.0e-4, size=300)
    crash = rng.normal(-7.5e-2, 3.0e-2, size=22)
    record = np.concatenate([calm, crash])
    assert R.volatility_ratio(record) > R.KILL_VOLATILITY
    assert R.volatility_ratio(calm) < R.KILL_VOLATILITY


