"""The panic detector, mocked: two series in, one verdict out, and no way to move a credit.

Everything here is invented, and invented *quietly*: the point of the file is that a verdict
is a function of two published series and of nothing else, so a test can draw the series and
know the answer in advance.

What is being pinned down, in five groups.

**The arithmetic has floors and they are the argument.** A tokenised Treasury bill whose rate
steps by seven basis points has a median absolute deviation of nearly nothing, so a raw z
ranks it as the most panicked asset in DeFi — measured, ``19.70 σ``. With the floors the same
series reads ``2.63 σ`` and ``0.66 pp``, which is a bill stepping with the Fed. The floor is
therefore tested directly: a step function is not a panic, at any z.

**The two arms cannot be sold short.** A dislocation without an organic record is
``FORMULA`` (a panic in a formula is the trap), and an organic record without a dislocation
is ``CALM``. Each arm alone is refused, which is the same rule the core keeps for its own
two arms.

**"Honest at rest" is what gets measured.** The claim a panicking pool is judged against is
its *resting* level, not the rate the panic advertises — otherwise the claim arm refuses
every panic by construction, and the module would be a filter that always says no.

**A replay cannot see the future.** :meth:`PanicDetector.walk` measures each day on that
day's prefix, and the tests assert it by comparing a walked signal to a fresh
:meth:`PanicDetector.measure` on the same prefix; :func:`forward_apy` starts strictly *after*
the decision, so the outcome column cannot leak into the verdict.

**Sizing is the book's, and it is capped.** A fired signal buys at most
:data:`agent.physics.portfolio.ALPHA_BOOST_CAP` of its own bucket, out of cash above
:data:`agent.physics.portfolio.ALPHA_CASH_FLOOR`, and nothing at all when the measurement did
not fire. And no surface in either file — hunter, detector, signal, report — has a method
that could sign, stake or transfer.
"""
import json
import math

import numpy as np
import pytest

from agent.physics import Portfolio, PortfolioError, Protocol, WhiteList
from agent.physics import portfolio as P
from agent.physics import volatility_hunter as V
from agent.physics.defillama_adapter import PoolFrame, PoolRow, sim_address


# ── the series a test draws ─────────────────────────────────────────────────


def bumpy(level: float, points: int, *, noise: float = 0.3, seed: int = 7):
    """A pool paying around ``level`` with the noise of a market on it."""
    rng = np.random.default_rng(seed)
    return (float(level) + rng.normal(0.0, float(noise) * float(level), int(points))).tolist()


def stepped(level: float, points: int, *, step: float, at: int):
    """A contract: one level, then one step, then the same level again."""
    return [float(level)] * int(at) + [float(level) + float(step)] * (int(points) - int(at))


def panic(
    level: float = 4.0,
    high: float = 12.0,
    *,
    quiet: int = 240,
    spike: int = 14,
    noise: float = 0.3,
    seed: int = 3,
):
    """``(apy, tvl)`` of a pool that pays ``level`` and then pays ``high`` on half the depth.

    The shape the whole file is about: liquidity leaves, the same yield is shared by fewer
    dollars, and the APY the venue publishes triples. The noise survives the jump — a panic
    is a *market* repricing, and a flat spike would be an announcement.
    """
    apy = bumpy(level, quiet, noise=noise, seed=seed) + bumpy(
        high, spike, noise=noise, seed=seed + 1
    )
    depth = [1.0e9] * int(quiet) + [4.0e8] * int(spike)
    return apy, depth


def organic_only(
    level: float = 4.0, points: int = 120, *, depth: float = 1.0e9, seed: int = 5
):
    """An ordinary organic pool: noisy, level, and no dislocation at all."""
    return bumpy(level, points, seed=seed), [float(depth)] * int(points)


def row(name: str = "pool", *, apy: float = 4.0, tvl: float = 1.0e9, uuid: str = "u-1"):
    return PoolRow(
        uuid=uuid,
        project="project",
        chain="Ethereum",
        symbol="SYM",
        tvl_usd=float(tvl),
        apy=float(apy),
    )


class Funnel:
    """The three things :meth:`VolatilityHunter.scan` asks a funnel, and nothing else."""

    def __init__(self, pools, *, approved=None, error: str = "") -> None:
        self.pools = dict(pools)
        self._approved = list(self.pools) if approved is None else list(approved)
        self.error = error
        self.venue = "defillama"

    def approved_names(self):
        return list(self._approved)


class Wire:
    """A scanner with no wire: frames come out of a dictionary and never off a socket."""

    def __init__(self, frames, *, fail=()) -> None:
        self.frames = dict(frames)
        self.fail = set(fail)
        self.asked = []

    def fetch_pool_frame(self, uuid, *, refresh: bool = False):
        self.asked.append(str(uuid))
        if str(uuid) in self.fail:
            raise RuntimeError("the wire is down")
        return self.frames.get(str(uuid), PoolFrame(uuid=str(uuid)))


def frame(apy, tvl, *, uuid: str = "u-1"):
    return PoolFrame(uuid=str(uuid), apy=tuple(apy), tvl=tuple(tvl))


def book(cash: float = 100.0, *protocols) -> Portfolio:
    """A book that may fund exactly these protocols — approval, not a proposal."""
    wallet = Portfolio(whitelist=WhiteList.of(*protocols), cash=float(cash))
    wallet.approve(list(protocols))
    return wallet


def protocol(
    name: str = "pool", *, apy: float = 0.04, returns=None, bucket: str = "safe"
) -> Protocol:
    record = (
        np.full(400, float(apy) / 365.0)
        if returns is None
        else np.asarray(returns, dtype=float)
    )
    return Protocol(
        name=name,
        bucket=bucket,
        claim_apy=float(apy),
        returns=record,
        address=sim_address(name),
    )


def signal(
    verdict: str = V.PANIC,
    *,
    confidence: float = 1.0,
    bucket: str = "safe",
    name: str = "pool",
):
    """A signal of the right *shape*, for the sizing tests that are not about measuring."""
    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "bucket": bucket,
        "name": name,
    }


# ── the arithmetic ──────────────────────────────────────────────────────────


def test_robust_scale_is_a_median_and_a_mad():
    centre, scale = V.robust_scale([1.0, 2.0, 3.0, 4.0, 5.0])
    assert centre == 3.0
    assert scale == pytest.approx(V.ROBUST_K * 1.0)


def test_robust_scale_ignores_a_single_spike():
    plain = V.robust_scale([5.0] * 50 + [5.1, 4.9])
    spiked = V.robust_scale([5.0] * 50 + [5.1, 4.9, 900.0])
    assert spiked[0] == plain[0]
    assert spiked[1] == plain[1]


def test_robust_scale_of_nothing_is_not_a_number():
    assert V.robust_scale([]) == (0.0, 0.0)
    assert V.robust_scale([float("nan"), float("inf")]) == (0.0, 0.0)


def test_resting_claim_ignores_the_last_window():
    apy = [2.0] * 100 + [40.0] * 14
    assert V.resting_claim(apy) == pytest.approx(2.0)


def test_resting_claim_of_a_short_series_uses_what_there_is():
    # Even two points get the rule: the last one is "recent", so the rest is a single point.
    assert V.resting_claim([3.0, 4.0]) == pytest.approx(3.0)
    assert V.resting_claim([]) == 0.0


def test_dislocation_is_the_gap_over_the_robust_scale():
    apy, _ = panic(4.0, 12.0, quiet=240, spike=14)
    price = V.measure_dislocation(apy, [1.0e9] * 254)
    assert price.resting == pytest.approx(4.0, abs=0.5)
    assert price.recent == pytest.approx(12.0, abs=1.5)
    assert price.gap_pp == pytest.approx(price.recent - price.resting)
    assert price.gap_ratio == pytest.approx(price.recent / price.resting)
    assert price.z > V.PANIC_Z


def test_a_treasury_bill_stepping_is_not_nineteen_sigmas():
    bill = stepped(3.22, 224, step=0.66, at=193)
    price = V.measure_dislocation(bill, [1.0e9] * 224)
    assert price.gap_pp == pytest.approx(0.66, abs=1e-9)
    assert price.z < 3.0, "the floor is what keeps a T-bill out of the panic list"
    assert price.z == pytest.approx(0.66 / V.PANIC_SIGMA_FLOOR, abs=1e-6)


def test_a_pool_paying_nothing_cannot_produce_an_astronomical_z():
    apy = [0.0] * 200 + [3.2] * 14
    price = V.measure_dislocation(apy, [1.0e9] * 214)
    assert math.isfinite(price.z)
    assert price.z < 100.0, "without an absolute floor this reads 2.9e8"


def test_exodus_is_the_relative_change_in_depth():
    apy, depth = panic(4.0, 12.0)
    price = V.measure_dislocation(apy, depth)
    assert price.exodus == pytest.approx(0.4 - 1.0, abs=1e-9)


def test_no_depth_is_not_depth_intact():
    apy, _ = organic_only()
    assert math.isnan(V.measure_dislocation(apy).exodus)


def test_a_constant_record_is_sterile():
    record = V.measure_organic([5.0] * 100)
    assert record.sterile is True
    assert record.organic is False


def test_a_noisy_record_is_organic():
    apy, _ = organic_only(4.0, 200)
    record = V.measure_organic(apy)
    assert record.sterile is False
    assert record.organic is True
    assert record.a2 > 0.1 and record.cv > 0.005


def test_a_flat_fortnight_after_a_noisy_year_is_not_organic():
    apy = bumpy(4.0, 200, seed=11) + [300.0] * 14
    record = V.measure_organic(apy)
    assert record.sterile is False, "the year is a market"
    assert record.flat is True, "the fortnight is an announcement"
    assert record.organic is False


def test_forward_apy_starts_after_the_decision():
    apy = [1.0] * 10 + [9.0] * 30
    assert V.forward_apy(apy, 9, 30) == pytest.approx(9.0)
    assert math.isnan(V.forward_apy(apy, 49, 30))


# ── the verdicts: each arm alone is refused ─────────────────────────────────


def test_a_short_record_is_unmeasured_not_refused():
    apy, depth = organic_only(4.0, 40)
    verdict = V.PanicDetector().measure(apy, depth, name="young")
    assert verdict.verdict == V.UNMEASURED
    assert "points of history" in verdict.reason
    assert verdict.resting_verdict == "", "no history, so no assessment at rest"


def test_a_thin_pool_has_no_exit_whatever_the_yield_says():
    apy, depth = panic(4.0, 12.0)
    thin = [value * 1.0e-6 for value in depth]
    verdict = V.PanicDetector().measure(apy, thin, name="small")
    assert verdict.verdict == V.THIN
    assert "no exit" in verdict.reason


def test_an_ordinary_pool_is_calm():
    apy, depth = organic_only(4.0, 200)
    verdict = V.PanicDetector().measure(apy, depth, name="ordinary")
    assert verdict.verdict == V.CALM
    assert not verdict.fired
    assert verdict.confidence == 0.0


def test_yield_up_with_depth_up_is_growth_not_panic():
    apy = bumpy(4.0, 240, seed=3) + bumpy(12.0, 14, seed=4)
    depth = [1.0e9] * 240 + [3.0e9] * 14
    verdict = V.PanicDetector().measure(apy, depth, name="crowded")
    assert verdict.verdict == V.GROWTH
    assert "growth" in verdict.reason


def test_a_dislocation_in_a_formula_is_the_trap():
    apy = stepped(3.22, 254, step=6.0, at=240)
    verdict = V.PanicDetector().measure(apy, [1.0e9] * 240 + [4.0e8] * 14, name="formula")
    assert verdict.verdict == V.FORMULA
    assert "flat" in verdict.reason, "a step is an announcement, not a repricing"
    assert not verdict.fired


def test_a_flat_spike_after_a_noisy_year_is_a_formula():
    apy = bumpy(4.0, 300, seed=11) + [300.0] * 14
    verdict = V.PanicDetector().measure(apy, [1.0e9] * 300 + [4.0e8] * 14, name="announced")
    assert verdict.verdict == V.FORMULA
    assert "flat" in verdict.reason


def test_an_honest_panic_fires_on_both_arms():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=14)
    verdict = V.PanicDetector().measure(apy, depth, name="honest panic")
    assert verdict.verdict == V.PANIC
    assert verdict.fired is True
    assert verdict.resting_verdict == "HONEST"
    assert 0.0 < verdict.confidence <= 1.0
    assert "depth fell" in verdict.reason and "organic" not in verdict.reason
    assert verdict.bucket == "safe", "the bucket is the one the resting level belongs to"


def test_the_claim_that_is_judged_is_the_resting_one():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=14)
    verdict = V.PanicDetector().measure(apy, depth, name="honest panic")
    assert verdict.resting_apy == pytest.approx(0.04, abs=0.01)
    assert verdict.claim_apy == pytest.approx(verdict.resting_apy), (
        "a panic may advertise anything; an operator who passes claim_apy overrides it"
    )
    judged = verdict.resting
    assert judged is not None and judged.claim_apy == pytest.approx(0.04, abs=0.01)


def test_an_overridden_claim_is_still_judged_at_rest():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=14)
    verdict = V.PanicDetector().measure(
        apy, depth, name="honest panic", claim_apy=0.11
    )
    assert verdict.claim_apy == pytest.approx(0.11)
    assert verdict.resting is not None
    assert verdict.resting.claim_apy == pytest.approx(0.04, abs=0.01)


def test_a_protocol_that_does_not_pay_its_promise_at_rest_is_refused():
    # A farm advertising 80 % in a bucket that may promise 35 %: at rest the assessment
    # says SUSPECT, and no panic makes an unpaid claim worth buying.
    apy = bumpy(80.0, 300, seed=13) + bumpy(200.0, 14, seed=14)
    depth = [1.0e9] * 300 + [4.0e8] * 14
    verdict = V.PanicDetector().measure(apy, depth, name="loud", bucket="safe")
    assert verdict.verdict == V.DISHONEST
    assert verdict.resting_verdict == "SUSPECT"
    assert not verdict.fired


def test_confidence_grows_with_the_dislocation():
    small = V.PanicDetector().measure(
        bumpy(4.0, 300, seed=21) + bumpy(8.0, 14, seed=22),
        [1.0e9] * 300 + [7.0e8] * 14,
        name="small",
    )
    big = V.PanicDetector().measure(
        bumpy(4.0, 300, seed=21) + bumpy(40.0, 14, seed=22),
        [1.0e9] * 300 + [1.0e8] * 14,
        name="big",
    )
    assert small.fired and big.fired
    assert big.confidence > small.confidence
    assert big.dislocation.z > small.dislocation.z
    assert big.confidence <= 1.0


def test_a_threshold_moved_moves_the_verdict():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=14)
    strict = V.PanicDetector(z_min=1000.0).measure(apy, depth, name="strict")
    assert strict.verdict == V.CALM, "a threshold nobody can clear is a refusal, not a no-op"


def test_the_signal_is_json_able():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=14)
    payload = V.PanicDetector().measure(apy, depth, name="honest panic").as_dict()
    assert json.loads(json.dumps(payload, allow_nan=False))["verdict"] == V.PANIC


# ── the replay: no look-ahead, one bet per episode, and the outcome attached after ──


def test_walk_measures_each_day_on_that_days_prefix():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=14)
    detector = V.PanicDetector()
    fired = detector.walk(apy, depth, name="honest panic")
    assert fired, "a fifteen-point jump on 60 % of the depth has to fire"
    day, walked = fired[0]
    fresh = detector.measure(apy[: day + 1], depth[: day + 1], name="honest panic")
    assert walked.verdict == fresh.verdict == V.PANIC
    assert walked.confidence == pytest.approx(fresh.confidence)
    assert walked.dislocation.recent == pytest.approx(fresh.dislocation.recent)


def test_walk_of_a_calm_pool_is_empty():
    apy, depth = organic_only(4.0, 200)
    assert V.PanicDetector().walk(apy, depth) == []


def test_one_episode_is_one_bet():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=20)
    events = V.PanicDetector().replay(apy, depth, name="episode", forward=10)
    assert len(events) == 1, "a fortnight of panic is not fourteen bets"


def test_replay_attaches_the_outcome_after_the_verdict():
    held, depth = panic(4.0, 12.0, quiet=300, spike=60)
    reverted = bumpy(4.0, 300, seed=3) + bumpy(12.0, 14, seed=4) + bumpy(4.0, 40, seed=5)
    depth_reverted = [1.0e9] * 300 + [4.0e8] * 14 + [4.0e8] * 40
    keeper = V.PanicDetector().replay(held, depth, forward=30)
    fader = V.PanicDetector().replay(reverted, depth_reverted, forward=30)
    assert keeper and fader
    assert keeper[0].paid_pp > 0.0 and keeper[0].held is True
    assert fader[0].paid_pp == pytest.approx(0.0, abs=1.0), "the spike went back to rest"


def test_an_event_is_json_able():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=30)
    payload = V.PanicDetector().replay(apy, depth, forward=10)[0].as_dict()
    assert json.loads(json.dumps(payload, allow_nan=False))["verdict"] == V.PANIC


# ── the scan: the funnel's approved pools and nothing else ──────────────────


def test_the_scan_reads_only_what_the_funnel_approved():
    panicking, deep = panic(4.0, 12.0, quiet=300, spike=14)
    calm, flat_depth = organic_only(4.0, 320)
    wire = Wire(
        {
            "u-panic": frame(panicking, deep, uuid="u-panic"),
            "u-calm": frame(calm, flat_depth, uuid="u-calm"),
        }
    )
    funnel = Funnel(
        {
            "panic pool": row("panic pool", uuid="u-panic"),
            "calm pool": row("calm pool", uuid="u-calm"),
            "refused pool": row("refused pool", uuid="u-never"),
        },
        approved=["panic pool", "calm pool"],
    )
    report = V.VolatilityHunter(scanner=wire, workers=1).scan(funnel)
    assert report.scanned == 2
    assert wire.asked == ["u-panic", "u-calm"], "an unapproved pool is never fetched"
    assert [signal.name for signal in report.panics()] == ["panic pool"]
    assert report.verdicts() == {V.CALM: 1, V.PANIC: 1}
    assert report.measured == 2


def test_the_scan_reports_a_dead_feed_instead_of_a_traceback():
    report = V.VolatilityHunter(scanner=Wire({}), workers=1).scan(
        Funnel({}, error="the feed answered HTTP 429")
    )
    assert report.error.startswith("the feed answered")
    assert report.signals == ()
    assert "no scan" in report.summary_text()


def test_a_pool_the_wire_fails_on_is_skipped_by_name():
    panicking, deep = panic(4.0, 12.0, quiet=300, spike=14)
    wire = Wire({"u-panic": frame(panicking, deep, uuid="u-panic")}, fail={"u-broken"})
    funnel = Funnel(
        {"panic pool": row("panic pool", uuid="u-panic"), "broken": row("broken", uuid="u-broken")}
    )
    report = V.VolatilityHunter(scanner=wire, workers=2).scan(funnel)
    assert report.fired == 1
    assert len(report.unmeasured()) == 1
    assert "the wire is down" in report.warnings[0]


def test_a_pool_with_no_aligned_history_is_unmeasured():
    wire = Wire({})
    funnel = Funnel({"empty": row("empty", uuid="u-empty")})
    report = V.VolatilityHunter(scanner=wire, workers=1).scan(funnel)
    assert report.signals[0].verdict == V.UNMEASURED
    assert "no day with both an APY and a depth" in report.signals[0].reason


def test_the_limit_takes_the_best_of_the_queue():
    panicking, deep = panic(4.0, 12.0, quiet=300, spike=14)
    wire = Wire(
        {
            "u-1": frame(panicking, deep, uuid="u-1"),
            "u-2": frame(panicking, deep, uuid="u-2"),
        }
    )
    funnel = Funnel({"first": row("first", uuid="u-1"), "second": row("second", uuid="u-2")})
    report = V.VolatilityHunter(scanner=wire, workers=1).scan(funnel, limit=1)
    assert report.scanned == 1
    assert wire.asked == ["u-1"]


def test_the_report_is_json_able_and_says_what_it_read():
    panicking, deep = panic(4.0, 12.0, quiet=300, spike=14)
    wire = Wire({"u-1": frame(panicking, deep, uuid="u-1")})
    report = V.VolatilityHunter(scanner=wire, workers=1).scan(
        Funnel({"pool": row("pool", apy=12.0, uuid="u-1")})
    )
    payload = json.loads(json.dumps(report.as_dict(), allow_nan=False))
    assert payload["venue"] == "defillama"
    assert payload["fired"] == 1
    assert payload["signals"][0]["verdict"] == V.PANIC
    assert "panics: 1" in report.summary_text()


# ── the sizing: the book's, and it is capped ────────────────────────────────


def test_the_weight_is_one_without_a_signal():
    wallet = book(100.0, protocol("a"))
    assert wallet.alpha_hunting_weight(None) == 1.0
    assert wallet.alpha_hunting_weight(signal(V.CALM, confidence=1.0)) == 1.0


def test_the_weight_is_capped_by_the_policy():
    wallet = book(100.0, protocol("a"))
    assert wallet.alpha_hunting_weight(signal(confidence=1.0)) == pytest.approx(
        1.0 + P.ALPHA_BOOST_CAP
    )
    assert wallet.alpha_hunting_weight(signal(confidence=99.0)) == pytest.approx(
        1.0 + P.ALPHA_BOOST_CAP
    ), "a confidence the signal invents cannot widen the cap"


def test_the_weight_is_the_book_reading_a_measured_signal():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=14)
    measured = V.PanicDetector().measure(apy, depth, name="honest panic")
    wallet = book(100.0, protocol("a"))
    assert measured.fired
    assert wallet.alpha_hunting_weight(measured) == pytest.approx(
        1.0 + P.ALPHA_BOOST_CAP * measured.confidence
    )


def test_the_book_and_the_hunter_agree_on_the_word_panic():
    apy, depth = panic(4.0, 12.0, quiet=300, spike=14)
    measured = V.PanicDetector().measure(apy, depth, name="honest panic")
    assert V.PANIC == "PANIC", "the literal both files share, held by this test"
    assert P._signal_fired(measured) is True
    assert P._signal_fired(signal(V.CALM)) is False
    assert P._signal_fired(None) is False


def test_no_signal_leaves_the_targets_alone():
    wallet = book(100.0, protocol("a"))
    assert wallet.boosted_targets([]) == dict(P.TARGETS)
    assert wallet.boosted_targets([signal(V.CALM, confidence=1.0)]) == dict(P.TARGETS)
    assert wallet.alpha_log == []


def test_the_boost_is_capped_at_half_the_bucket():
    wallet = book(100.0, protocol("a"))
    targets = wallet.boosted_targets([signal(confidence=1.0)])
    assert targets["safe"] == pytest.approx(0.60 + P.ALPHA_BOOST_CAP * 0.60)
    assert targets["active"] == pytest.approx(0.30), "a panic never widens another bucket"
    assert targets["experimental"] == pytest.approx(0.10)
    granted = sum(targets.values()) - sum(P.TARGETS.values())
    assert granted <= wallet.alpha_capacity() + 1e-12, "cash above the floor is all there is"


def test_the_boost_is_funded_out_of_cash_above_the_floor():
    wallet = book(100.0, protocol("a"))
    wallet.deposit(protocol("a"), 90.0)
    assert wallet.alpha_capacity() == pytest.approx(0.10 - P.ALPHA_CASH_FLOOR)
    targets = wallet.boosted_targets([signal(confidence=1.0)])
    assert targets["safe"] == pytest.approx(0.60 + wallet.alpha_capacity())


def test_a_book_with_no_spare_cash_buys_nothing():
    wallet = book(100.0, protocol("a"))
    wallet.deposit(protocol("a"), 100.0)
    assert wallet.alpha_capacity() == 0.0
    assert wallet.boosted_targets([signal(confidence=1.0)]) == dict(P.TARGETS)
    assert wallet.alpha_log == []


def test_the_loudest_signal_is_served_first():
    wallet = book(100.0, protocol("a"))
    wallet.deposit(protocol("a"), 90.0)
    targets = wallet.boosted_targets(
        [
            signal(confidence=0.2, bucket="active", name="quiet"),
            signal(confidence=1.0, bucket="safe", name="loud"),
        ]
    )
    assert targets["safe"] > 0.60, "the strongest measurement gets the room"
    assert targets["active"] == pytest.approx(0.30), "and the rest gets none of it"
    assert [row["protocol"] for row in wallet.alpha_log] == ["loud"]


def test_the_grant_is_logged_with_the_numbers_that_granted_it():
    wallet = book(100.0, protocol("a"))
    wallet.boosted_targets([signal(confidence=0.5, name="measured")], step=42)
    row = wallet.alpha_log[0]
    assert row["step"] == 42 and row["bucket"] == "safe" and row["protocol"] == "measured"
    assert row["weight"] == pytest.approx(1.0 + 0.5 * P.ALPHA_BOOST_CAP)
    assert row["granted"] <= row["wanted"] + 1e-12
    assert json.loads(json.dumps(row, allow_nan=False))["granted"] == row["granted"]


def test_a_boosted_target_deploys_more_credits():
    calm_book = book(100.0, protocol("a"))
    hunting_book = book(100.0, protocol("a"))
    plan = {"safe": protocol("a"), "active": None, "experimental": None}
    calm_book.rebalance(plan, step=0)
    boosted = hunting_book.boosted_targets([signal(confidence=1.0)])
    hunting_book.rebalance(plan, step=0, targets=boosted)
    placed = hunting_book.value(protocol("a").address)
    assert placed > calm_book.value(protocol("a").address)
    assert placed == pytest.approx(100.0 * boosted["safe"])
    assert hunting_book.cash >= P.ALPHA_CASH_FLOOR * hunting_book.equity


def test_rebalance_without_targets_is_the_book_it_always_was():
    plain = book(100.0, protocol("a"))
    explicit = book(100.0, protocol("a"))
    plan = {"safe": protocol("a"), "active": None, "experimental": None}
    plain.rebalance(plan, step=0)
    explicit.rebalance(plan, step=0, targets=dict(P.TARGETS))
    assert plain.value(protocol("a").address) == explicit.value(protocol("a").address)
    assert plain.alpha_log == []


def test_a_target_map_missing_a_bucket_is_an_error():
    wallet = book(100.0, protocol("a"))
    with pytest.raises(PortfolioError, match="missing"):
        wallet.rebalance({"safe": None}, step=0, targets={"safe": 0.6})


def test_a_signal_for_a_bucket_that_does_not_exist_is_ignored():
    wallet = book(100.0, protocol("a"))
    targets = wallet.boosted_targets([signal(confidence=1.0, bucket="moonshot")])
    assert targets == dict(P.TARGETS)
    assert wallet.alpha_log == []


def test_an_empire_with_no_signal_keeps_its_own_targets():
    empire = P.Empire.launch(steps=240, seed=7)
    assert empire._targets_with_alpha() is None, "no hunter attached, no new weight"
    empire.alpha = [signal(V.CALM, confidence=1.0)]
    assert empire._targets_with_alpha() == dict(P.TARGETS), "a calm market prices nothing in"
    assert empire.portfolio.alpha_log == []


def test_an_empire_prices_a_panic_into_the_bucket_it_measured():
    calm = P.Empire.launch(steps=480, seed=7)
    hunting = P.Empire.launch(steps=480, seed=7)
    for empire in (calm, hunting):
        empire.advance(96)  # the book finds where to stand and spends its cash
        empire.portfolio.cash += 40.0  # a top-up: the only capital that answers a surprise
    calm.review()
    hunting.alpha = [signal(confidence=1.0, bucket="safe", name="measured")]
    hunting.review()
    grant = hunting.portfolio.alpha_log[0]
    assert grant["protocol"] == "measured" and grant["granted"] > 0.0
    assert grant["granted"] <= grant["wanted"] + 1e-12
    assert grant["capacity_left"] >= 0.0, "the boost stops at the cash floor, never below it"
    assert hunting.portfolio.bucket_value("safe") > calm.portfolio.bucket_value("safe")


# ── the seam: no surface here can move a credit ─────────────────────────────


def test_nothing_in_the_hunter_can_sign_stake_or_transfer():
    dangerous = (
        "execute_stake",
        "withdraw",
        "sign",
        "send",
        "transfer",
        "approve",
        "whitelist",
        "execution_whitelist",
        "private_key",
        "account",
    )
    surface = (V.VolatilityHunter, V.PanicDetector, V.PanicSignal, V.PanicReport, V.PanicEvent)
    for item in surface:
        for name in dangerous:
            assert not hasattr(item, name), f"{item.__name__} must not have {name}"
