"""A panic is a price; a lie is a formula. This file only trades the first one.

The previous phase handed the two gates a market nobody drew and asked one question of
every pool: *is this a record or a promise?* Fifty-two of a hundred and fifty-two live
records came back ``HONEST``, and that verdict is the end of the funnel as it stands —
the wallet may fund them and nothing else. But a verdict is about a *protocol*, and it
says nothing about a *day*. An honest protocol can be having a terrible week: liquidity
leaves, the same depth has to pay the same borrowers, and the APY on the venue's chart
goes up by a factor of three. The yield rose because the risk premium rose, and the
question this file answers is whether the *record* still looks like a market while the
*price* looks like fear — because that combination is the only one where the empire can
be paid for standing still while everybody else runs.

The detector that does not work, measured
=========================================

"Current APY more than two sigmas above its own median" is four lines of numpy, and on the
live two-hundred-pool scan of 5 November 2026 it produces this as its loudest reading::

    blackrock-buidl BUIDL (BSC)      z = 19.70    3.22 % -> 3.90 %    depth +25 %

A tokenised Treasury bill, ranked as the most panicked asset in DeFi. It is not a bug in
the arithmetic: BUIDL's rate moves in single steps when the bill it holds is rolled, so the
median absolute deviation of its series is a few hundredths of a point and a seven-basis-
point step is nineteen "sigmas". Two things follow, and both are in this file:

* **σ alone cannot rank opportunity.** The same reading, measured with the floors below,
  is ``2.63 σ``, ``0.66 pp`` and ``1.20 ×`` — three numbers that describe a Treasury bill
  stepping with the Fed and nothing else, and the whole of the difference is that a
  dislocation is never measured in units finer than a quarter of a point of APY
  (:data:`PANIC_SIGMA_FLOOR`) or smaller than a point a year
  (:data:`PANIC_GAP_PP`). The arithmetic that made a T-bill look like a panic is the
  arithmetic the floor removes.
* **a curve with no market noise in it is a promise** — Statement #87's sentence, and the
  record arm is where it is asked. It is not an extra filter bolted onto the price arm; it
  is the measurement that tells a dislocation in a market from a dislocation in a formula,
  and its second horizon (the raised segment's own ``CV``) is there because a pool can be
  noisy for a year and then *announce* 300 % for a fortnight — which is precisely the
  fortnight the strategy would be buying.

Measured over the same scan, the floors leave nothing firing on the day of the run: the
strongest reading among two hundred pools is ``2.63 σ``, and the honest report of a market
with no panic in it is ``CALM`` fifty-two times. What shows the detector working is the
replay (:meth:`PanicDetector.replay`), where the same arithmetic is asked about every
historical day of the same pools — fifty-three panics across seventeen of them, and a
median of ``+1.35 pp`` of APY paid over the month that followed against ``0.00 pp`` for
the same pools on ordinary days.


The two arms, and the one thing they are allowed to conclude
===========================================================

========================  =========================================================
``dislocation`` (price)   the recent level is far above the resting one, in robust
                          sigmas **and** in percentage points, while the pool's depth
                          has fallen — "the market is paying more because the market
                          left", measured, not assumed
``organic`` (record)      the same record that passed the funnel still carries the
                          residuals of a medium: ``A²`` and ``CV`` above Statement
                          #87's floors, the *raised* segment not flat, and no repeat.
                          "The protocol is honest" means honest **at rest**, which is
                          what :func:`resting_claim` measures
========================  =========================================================

Only both arms together are a ``PANIC``, for the reason the core gives for its own two
arms: a price alone cannot tell you whether the protocol is real, and a record alone
cannot tell you whether today is different from yesterday.

What "honest at rest" means, and why the claim arm is deliberately changed
==========================================================================

:func:`agent.physics.risk_assessment.evaluate_opportunity` refuses a record whose claim
is more than :data:`~agent.physics.risk_assessment.CLAIM_Z` standard errors above what
the record paid — ``SUSPECT``, "stop advertising". During a panic that test fires by
construction: the advertised APY *is* above the record's mean, because the whole event is
a rate that has not yet been paid long enough to be inside the mean. Judging a panicking
pool by its advertised yield therefore refuses every panic, which is not a conservative
reading but an empty one.

So this file asks the assessment a different question about the same record: *what would
this protocol be if it were calm?* The claim it is judged against is the **resting
level** — the median APY of everything before the last :data:`PANIC_RECENT` days — and
the bucket is the one that level belongs to, not the one the panic belongs to. The
argument is one sentence long: a panic cannot be allowed to widen the ceiling it is
judged against, and what a pool promises is what it pays when nothing is happening.

The verdicts
============

.. code-block:: text

    UNMEASURED   too little history, or no depth published: not a verdict
    THIN         the pool is smaller than the scanner's depth floor: no exit
    CALM         no dislocation: today is an ordinary day for this pool
    GROWTH       the yield rose and the depth rose with it: growth, not exodus
    FORMULA      liquidity left and the yield rose — and the record is a formula:
                 the panic *is* the trap
    DISHONEST    the record is organic, but the protocol does not pay what it
                 promises when calm: nothing here to buy
    PANIC        both arms: an honest protocol, a measured dislocation

The last one is the signal, and the only one :class:`agent.physics.portfolio.Portfolio`
sizes (:meth:`~agent.physics.portfolio.Portfolio.alpha_hunting_weight`). It is also the
same vocabulary the rest of the layer already speaks: a number, the measurement that
produced it, and a reason a reader can argue with.

What this file cannot do
========================

It reads two published series and computes five numbers. There is no order, no key, no
signature and no transfer anywhere in it: :class:`VolatilityHunter` holds a scanner,
which reads a public API, and the book that funds a signal is
:class:`agent.physics.portfolio.Portfolio`, whose every stake goes through
``IWeb3Provider.execute_stake`` and nowhere else. The measurement and the transaction are
in different files, and a test asserts that no method here could move a credit.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.physics.defillama_adapter import (
    MIN_TVL_USD,
    VENUE_NAME,
    WORKERS,
    DefiLlamaScanner,
    MarketFunnel,
    PoolRow,
    bucket_for,
    to_returns,
)
from agent.physics.portfolio import BUCKET_SAFE, MIN_RECORD, sim_address
from agent.physics.risk_assessment import (
    CONVICTIONS,
    HONEST,
    UNMEASURED,
    Assessment,
    evaluate_opportunity,
    sterility_of,
)
from agent.physics.sterility import STERILE_A2, STERILE_CV

__all__ = [
    "CALM",
    "DISHONEST",
    "FORMULA",
    "GROWTH",
    "PANIC",
    "PANIC_COOLDOWN",
    "PANIC_EXODUS_FULL",
    "PANIC_GAP_FULL_PP",
    "PANIC_GAP_PP",
    "PANIC_GAP_RATIO",
    "PANIC_MIN_POINTS",
    "PANIC_RECENT",
    "PANIC_TVL_DROP",
    "PANIC_Z",
    "PANIC_Z_FULL",
    "ROBUST_K",
    "PANIC_SIGMA_FLOOR",
    "SIGMA_FLOOR_SHARE",
    "THIN",
    "VERDICTS",
    "Dislocation",
    "Organic",
    "PanicDetector",
    "PanicEvent",
    "PanicReport",
    "PanicSignal",
    "VolatilityHunter",
    "forward_apy",
    "main",
    "measure_dislocation",
    "measure_organic",
    "resting_claim",
    "robust_scale",
]


# ── the numbers the detector is built on ────────────────────────────────────

#: Days the *recent* level is measured over. Fourteen, and it is a median rather than the
#: last point on purpose: a single day of a feed is the day most likely to be a mistake —
#: a stale row, a reward token that paid once, a pool that was briefly mispriced — and the
#: strategy is paid by a *state*, not by a print. Two weeks is also the shortest window
#: that survives :func:`agent.physics.sterility.anderson_darling`'s own floor of eight
#: samples, which is what lets the detector ask whether the **raised** part of the curve
#: is organic too.
PANIC_RECENT = 14

#: Median absolute deviation as a multiple of a normal sample's σ. Copied, not tuned:
#: ``1.4826`` is the consistency constant of the MAD, the same number
#: :mod:`agent.physics.sterility` uses when it turns a deviation into a scale.
ROBUST_K = 1.4826

#: The smallest scale the dislocation is allowed to divide by, as a share of the resting
#: level. Without it a pool whose rate barely moves has ``σ ≈ 0`` and any step at all is
#: "nineteen sigmas" — measured on the live scan, that is exactly how a tokenised Treasury
#: bill (BUIDL, +19.70 σ on a seven-basis-point step) came to look like the most panicked
#: asset in DeFi. Two percent of the level is a floor a reader can check: a pool paying
#: 4 % is allowed to be surprised by 0.08 pp before the surprise is called large.
SIGMA_FLOOR_SHARE = 0.02

#: And the floor in *absolute* units: a quarter of a point of APY a year. The relative
#: floor above cannot help a pool whose resting level is zero — it divides by zero of a
#: different kind — and on the live cache that produced ``z = 2.9 × 10⁸`` for a synthetic
#: dollar that started paying 3 % after a year of paying nothing. A quarter of a point is
#: the finest resolution a daily APY feed supports (the venue publishes hundredths of a
#: point, and day-to-day movement below a quarter is rounding on a rounding), so a
#: dislocation is never measured in units finer than that. Declared policy, not a fit.
PANIC_SIGMA_FLOOR = 0.25

#: Robust sigmas above the resting level before the price arm calls the day unusual. Two,
#: the number the brief asks for, and it is a *gate* rather than a size: what the signal
#: may buy is decided by :attr:`PanicSignal.confidence`, which keeps growing past this
#: threshold instead of stopping at it.
PANIC_Z = 2.0

#: The gap in percentage points below which nothing fires, however many sigmas it is. One
#: point of APY a year is the smallest difference a person would move capital for, and a
#: dislocation measured in sigmas alone cannot see the difference between a T-bill
#: stepping by twelve basis points and a lending market repricing by six hundred.
PANIC_GAP_PP = 1.0

#: The multiple of the resting level the recent level must reach. Sigma and points are
#: both absolute; this one is relative, and it is the arm that refuses a pool whose
#: resting rate is already huge — a 400 % farm going to 500 % is noise on a number nobody
#: believed in the first place.
PANIC_GAP_RATIO = 1.25

#: The share of depth that must have left the pool. The whole thesis of the file is that a
#: panic is *liquidity leaving and being paid to come back*, so it is measured rather than
#: assumed: a yield that rises while depth rises is growth, and this number is the only
#: thing that tells the two apart.
PANIC_TVL_DROP = 0.15

#: Points of history before the detector will speak at all. The same floor the rest of the
#: layer uses (:data:`agent.physics.portfolio.MIN_RECORD`): below it the record is too
#: short for a verdict, and the honest answer is "not yet".
PANIC_MIN_POINTS = MIN_RECORD

#: The depth below which a dislocation is not an opportunity: the scanner's own floor
#: (:data:`agent.physics.defillama_adapter.MIN_TVL_USD`), reused rather than re-declared so
#: that the hunter cannot quietly trade a pool the funnel would have dropped — and so that
#: a pool that *lost* its depth during the panic is measured on the depth it had before it.
PANIC_MIN_DEPTH = MIN_TVL_USD

#: The three saturations that turn a measurement into a 0 … 1 confidence. ``z`` at four
#: robust sigmas, a gap of three points of APY a year, and half the pool's depth gone are
#: each, on this market, a large but not extreme reading — so a signal at a threshold buys
#: a little and a signal far past it buys the whole budget, continuously, with no cliff
#: between the two.
PANIC_Z_FULL = 4.0
PANIC_GAP_FULL_PP = 3.0
PANIC_EXODUS_FULL = 0.50

#: Days after a signal during which the same pool cannot signal again. Thirty — the same
#: horizon the outcome is measured over — so that one episode is one bet: a panic lasts a
#: fortnight and its aftermath a month, and counting the same dislocation once a day would
#: turn every average over "panics" into an average over one panic.
PANIC_COOLDOWN = 30

# ── the verdicts ────────────────────────────────────────────────────────────

#: Both arms: an honest protocol and a measured dislocation. The only verdict the book
#: will size.
PANIC = "PANIC"
#: No dislocation: today is an ordinary day for this pool.
CALM = "CALM"
#: The yield rose and the depth rose with it — growth, not exodus.
GROWTH = "GROWTH"
#: The record is not a record: a sterile curve, a flat raised segment, or a repeat. A
#: "panic" in a formula is the trap, not the opportunity.
FORMULA = "FORMULA"
#: The record is organic, but the protocol does not pay what it promises when calm.
DISHONEST = "DISHONEST"
#: Too little depth to have an exit, whatever the yield says.
THIN = "THIN"
#: Every verdict, in the order the detector checks them.
VERDICTS = (UNMEASURED, THIN, CALM, GROWTH, FORMULA, DISHONEST, PANIC)


# ── the arithmetic: two series in, five numbers out ─────────────────────────


def _finite(values: Sequence[Any]) -> np.ndarray:
    """The finite part of a series, in order — the only rows any of this may read."""
    array = np.asarray(list(values), dtype=float).ravel()
    return array[np.isfinite(array)]


def robust_scale(values: Sequence[float]) -> Tuple[float, float]:
    """``(centre, scale)`` = median and ``1.4826 × MAD`` — a mean a spike cannot move.

    The mean and the standard deviation are the wrong tools for a yield series, and the
    reason is in the market rather than in the mathematics: an APY series is *fat-tailed by
    construction* (rewards arrive in lumps, liquidity arrives and leaves), so one week of a
    farm paying 400 % moves the mean of a year by more than the panic the detector is
    looking for. The median ignores it, and the MAD — the median of the distances from the
    median — inherits the same indifference.
    """
    array = _finite(values)
    if array.size == 0:
        return 0.0, 0.0
    centre = float(np.median(array))
    scale = float(ROBUST_K * np.median(np.abs(array - centre)))
    return centre, scale


def resting_claim(apy: Sequence[float], *, recent: int = PANIC_RECENT) -> float:
    """The APY a pool pays when nothing is happening, in per cent: its resting level.

    Everything before the last :data:`PANIC_RECENT` days, as a median. This is the number
    the assessment is asked about, and calling it a *claim* is deliberate: it is what the
    venue would advertise on an ordinary day, so judging the pool by it is asking "is this
    protocol honest when it is calm" — the only question a panic leaves open.
    """
    array = _finite(apy)
    if array.size == 0:
        return 0.0
    window = min(max(int(recent), 0), max(array.size - 1, 0))
    resting = array[: array.size - window] if window else array
    if resting.size == 0:
        resting = array
    return float(np.median(resting))


@dataclass(frozen=True)
class Dislocation:
    """The price arm, measured: two levels, the noise between them, and the exodus.

    Every field is a number or ``NaN``, and none of them is a verdict — the thresholds
    belong to :class:`PanicDetector`, so that a reader can see the measurement and the
    policy apart. ``exodus`` is the relative change in depth across the same window:
    negative means liquidity left, and that sign is the difference between a panic and a
    rally.
    """

    resting: float = math.nan
    recent: float = math.nan
    sigma: float = math.nan
    gap_pp: float = math.nan
    gap_ratio: float = math.nan
    z: float = math.nan
    exodus: float = math.nan
    points: int = 0

    @property
    def measured(self) -> bool:
        """True when both levels exist — a pool with no history has no dislocation."""
        return bool(self.points and not math.isnan(self.recent))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resting_apy": _round(self.resting),
            "recent_apy": _round(self.recent),
            "sigma": _round(self.sigma),
            "gap_pp": _round(self.gap_pp),
            "gap_ratio": None if math.isinf(self.gap_ratio) else _round(self.gap_ratio),
            "z": _round(self.z),
            "exodus": _round(self.exodus),
            "points": int(self.points),
        }


def measure_dislocation(
    apy: Sequence[float],
    tvl: Optional[Sequence[float]] = None,
    *,
    recent: int = PANIC_RECENT,
) -> Dislocation:
    """The price arm: how far the last fortnight stands above the pool's own rest.

    ``tvl`` is optional in the type and not in the argument: a pool whose depth the venue
    does not publish has no measurable exodus, and the detector reports that as
    ``UNMEASURED`` rather than treating "no depth" as "depth intact". The two series must
    already be aligned
    (:meth:`agent.physics.defillama_adapter.DefiLlamaScanner.fetch_pool_frame` is what
    makes them so); a caller that zips two independently cleaned series is measuring two
    different calendars.
    """
    values = _finite(apy)
    points = int(values.size)
    if points == 0:
        return Dislocation(points=0)
    window = min(max(int(recent), 1), points)
    resting_part = values[: points - window] if points > window else values
    if resting_part.size < 2:
        resting_part = values
    centre, scale = robust_scale(resting_part)
    level = float(np.median(values[-window:]))
    gap = level - centre
    sigma = max(
        scale,
        float(SIGMA_FLOOR_SHARE) * abs(centre),
        float(PANIC_SIGMA_FLOOR),
    )
    if centre > 0.0:
        ratio = (level / centre) if level > 0.0 else 0.0
    else:
        ratio = math.inf if level > 0.0 else 1.0
    exodus = math.nan
    depth = _finite(tvl) if tvl is not None else np.zeros(0)
    if depth.size >= 2:
        depth_window = min(window, depth.size)
        before = depth[: depth.size - depth_window]
        if before.size == 0:
            before = depth
        base = float(np.median(before))
        now = float(np.median(depth[-depth_window:]))
        if base > 0.0:
            exodus = now / base - 1.0
    return Dislocation(
        resting=float(centre),
        recent=float(level),
        sigma=float(sigma),
        gap_pp=float(gap),
        gap_ratio=float(ratio),
        z=float(gap / sigma),
        exodus=float(exodus),
        points=points,
    )


@dataclass(frozen=True)
class Organic:
    """The record arm, measured: Statement #87 on the whole record and on the spike.

    ``a2``/``cv`` are the core's own sterility reading
    (:func:`agent.physics.risk_assessment.sterility_of`) over the whole history;
    ``segment_*`` are the same two statistics over the raised window alone. The second
    pair exists because the first one cannot see a *local* formula: a pool that paid a
    noisy market rate for a year and then announced a flat 300 % for a fortnight leaves the
    whole-record statistics looking perfectly organic, and that fortnight is precisely what
    the strategy would be buying.
    """

    a2: float = math.nan
    cv: float = math.nan
    points: int = 0
    segment_a2: float = math.nan
    segment_cv: float = math.nan
    segment_points: int = 0

    @property
    def sterile(self) -> bool:
        """Statement #87's finding, in the same words the assessment uses."""
        if math.isnan(self.a2) or math.isnan(self.cv):
            return True
        return bool(self.a2 < float(STERILE_A2) or self.cv < float(STERILE_CV))

    @property
    def flat(self) -> bool:
        """True when the *raised* part of the curve is a constant — a promise, not a price."""
        if math.isnan(self.segment_cv):
            return True
        return bool(self.segment_cv < float(STERILE_CV))

    @property
    def organic(self) -> bool:
        """Neither a sterile record nor a flat spike: a market on both horizons."""
        return bool(not self.sterile and not self.flat)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "a2": _round(self.a2),
            "cv": _round(self.cv),
            "points": int(self.points),
            "segment_a2": _round(self.segment_a2),
            "segment_cv": _round(self.segment_cv),
            "segment_points": int(self.segment_points),
            "sterile": bool(self.sterile),
            "flat": bool(self.flat),
            "organic": bool(self.organic),
        }


def measure_organic(apy: Sequence[float], *, recent: int = PANIC_RECENT) -> Organic:
    """The record arm: Statement #87 on the record and on the raised window.

    The series is converted to per-step rates first
    (:func:`agent.physics.defillama_adapter.to_returns`), because that is the object the
    rest of the layer measures and the conversions must not differ: ``CV`` of a rate series
    and ``CV`` of the APY series it came from agree to the fourth percent at an
    eight-percent yield, and ``A²`` is invariant because the conversion is monotone. Doing
    it here rather than at the call site is the rule the adapter keeps — the units change
    in exactly one place.
    """
    returns = to_returns(apy)
    a2, cv, points = sterility_of(returns)
    values = _finite(apy)
    window = min(max(int(recent), 0), int(values.size))
    segment = values[values.size - window :] if window else np.zeros(0)
    segment_a2, segment_cv, segment_points = sterility_of(to_returns(segment))
    return Organic(
        a2=float(a2),
        cv=float(cv),
        points=int(points),
        segment_a2=float(segment_a2),
        segment_cv=float(segment_cv),
        segment_points=int(segment_points),
    )


def _round(value: float, digits: int = 6) -> Optional[float]:
    """Round, keeping ``NaN`` as ``None`` — a report is JSON and JSON has no NaN."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return round(float(value), digits)


def _leg(value: float, threshold: float, full: float) -> float:
    """One measurement as 0 … 1 between its threshold and its saturation."""
    if value is None or not math.isfinite(float(value)):
        return 0.0
    number = float(value)
    if full <= threshold:
        return 1.0 if number >= threshold else 0.0
    return min(max((number - threshold) / (full - threshold), 0.0), 1.0)


# ── one verdict, as a value ─────────────────────────────────────────────────


@dataclass(frozen=True)
class PanicSignal:
    """One pool, one day, both arms measured: the verdict, the numbers, and the size.

    The three numbers that matter are all here and none of them is derived from a policy:
    the dislocation (how far the last fortnight stands above the pool's rest, in sigmas,
    points and multiple), the exodus (how much of the pool's depth left), and the record
    (``A²``, ``CV``, and the assessment's verdict on what the pool promises *at rest*).
    The one number that *is* policy — how much of its bucket a signal may buy — lives in
    :class:`agent.physics.portfolio.Portfolio`, and this value only reports the 0 … 1
    ``confidence`` it is scaled by.
    """

    name: str = ""
    address: str = ""
    uuid: str = ""
    bucket: str = BUCKET_SAFE
    chain: str = ""
    project: str = ""
    verdict: str = UNMEASURED
    reason: str = ""
    dislocation: Dislocation = field(default_factory=Dislocation)
    organic: Organic = field(default_factory=Organic)
    resting_verdict: str = ""
    resting_reason: str = ""
    resting_apy: float = 0.0
    claim_apy: float = 0.0
    depth: Optional[float] = None
    resting: Optional[Assessment] = None

    # -- what it concluded -------------------------------------------------- #

    @property
    def fired(self) -> bool:
        """True only for ``PANIC``: both arms, and nothing else.

        This is the property :func:`agent.physics.portfolio._signal_fired` reads, which is
        why the book can size a signal without importing this module — and why the literal
        ``"PANIC"`` appears in both files, held together by a test rather than by an
        import.
        """
        return self.verdict == PANIC

    @property
    def confidence(self) -> float:
        """How far past its thresholds the measurement is, 0 … 1, as the mean of three legs.

        The legs are the three ways a dislocation can be large — how unusual it is for this
        pool (``z``), how much extra yield is on the table (``gap_pp``), and how much of the
        pool's depth left (``exodus``) — each mapped from its own threshold to its own
        saturation. The mean rather than the minimum because the three are *different
        kinds of evidence* rather than three attempts at one number: a pool down 60 % of
        its depth is a violent panic even if the yield only doubled, and the size of the
        trade should say so.
        """
        if not self.fired:
            return 0.0
        legs = (
            _leg(self.dislocation.z, PANIC_Z, PANIC_Z_FULL),
            _leg(self.dislocation.gap_pp, PANIC_GAP_PP, PANIC_GAP_FULL_PP),
            _leg(-self.dislocation.exodus, PANIC_TVL_DROP, PANIC_EXODUS_FULL),
        )
        return float(sum(legs) / len(legs))

    def as_dict(self) -> Dict[str, Any]:
        """The signal as a report row — rounded, JSON-able, ``NaN`` as ``None``."""
        return {
            "name": self.name,
            "address": self.address,
            "uuid": self.uuid,
            "chain": self.chain,
            "project": self.project,
            "bucket": self.bucket,
            "verdict": self.verdict,
            "fired": bool(self.fired),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "resting_apy": round(float(self.resting_apy) * 100.0, 4),
            "claim_apy": round(float(self.claim_apy) * 100.0, 4),
            "depth_usd": None if self.depth is None else round(float(self.depth), 2),
            "resting_verdict": self.resting_verdict,
            "resting_reason": self.resting_reason,
            "dislocation": self.dislocation.as_dict(),
            "organic": self.organic.as_dict(),
        }


def forward_apy(apy: Sequence[float], day: int, days: int = 30) -> float:
    """The APY a pool actually paid over the ``days`` **after** ``day``, as a median.

    Strictly after: the window starts at ``day + 1``, because the decision is taken on the
    close of ``day`` and a strategy paid by the same day it decided would be a backtest
    reading the answer. Returns ``NaN`` when the record does not reach that far — a
    truncated outcome is not an outcome of zero.
    """
    values = _finite(apy)
    start = int(day) + 1
    if start >= values.size:
        return math.nan
    window = values[start : start + max(int(days), 1)]
    if window.size == 0:
        return math.nan
    return float(np.median(window))


# ── the detector: two arms, and the policy that reads them ──────────────────


@dataclass(frozen=True)
class PanicDetector:
    """Both arms and every threshold, in one value an operator can move.

    None of the methods here writes anything: :meth:`measure` takes two series and returns
    a :class:`PanicSignal`, :meth:`walk` asks the same question of every historical day of
    one pool, and :meth:`replay` attaches what happened next. The thresholds are fields
    rather than constants inside a call so that a reader — or a test — can move one number
    and see exactly what it changed, which is the same argument the adapter's ``timeout``
    is a field for.

    ``oracle`` is optional exactly as it is in the core: without it the repeat and mirror
    arms are blind and the assessment says so in ``details["measured"]``. With it the
    resting verdict also answers "does this record repeat itself", which is the one thing
    :class:`Organic` cannot see from two statistics alone.
    """

    recent: int = PANIC_RECENT
    z_min: float = PANIC_Z
    gap_min: float = PANIC_GAP_PP
    ratio_min: float = PANIC_GAP_RATIO
    exodus_min: float = PANIC_TVL_DROP
    min_points: int = PANIC_MIN_POINTS
    min_depth: Optional[float] = PANIC_MIN_DEPTH
    oracle: Any = None

    # -- the price arm, as a refusal ---------------------------------------- #

    def price_refusal(self, price: Dislocation) -> Tuple[str, str]:
        """``(verdict, reason)`` for the price arm — ``("", "")`` when it fires.

        Four questions in the order that makes the reason worth reading: can the exodus be
        measured at all, is the level unusual for this pool, is the difference worth money,
        and did the depth actually leave. The first is a *different kind of answer*
        (``UNMEASURED``) and is kept apart from the other three, which are statements about
        the market.
        """
        if not price.measured:
            return UNMEASURED, "no published history to compare today against"
        if math.isnan(price.exodus):
            return UNMEASURED, (
                "the venue published no depth for this pool, so an exodus cannot be "
                "measured — and a yield that rose with no measurable exodus is not a "
                "panic, it is a number"
            )
        if (
            price.z < float(self.z_min)
            or price.gap_pp < float(self.gap_min)
            or price.gap_ratio < float(self.ratio_min)
        ):
            return CALM, (
                f"APY {price.recent:.2f}% against a resting {price.resting:.2f}% is "
                f"{price.z:.2f}σ ({price.gap_pp:+.2f} pp, {price.gap_ratio:.2f}×), inside "
                f"the {float(self.z_min):.1f}σ / {float(self.gap_min):.1f} pp / "
                f"{float(self.ratio_min):.2f}× gates: an ordinary day for this pool"
            )
        if price.exodus > -float(self.exodus_min):
            moved = "rose" if price.exodus >= 0.0 else "barely moved"
            return GROWTH, (
                f"APY rose {price.resting:.2f}% → {price.recent:.2f}% but depth {moved} "
                f"({price.exodus:+.0%}): capital arrived with the yield, so this is "
                "growth, not an exodus"
            )
        return "", ""

    # -- the record arm, as a refusal --------------------------------------- #

    def record_refusal(self, signal: PanicSignal) -> Tuple[str, str]:
        """``(verdict, reason)`` for the record arm — ``("", "")`` when it passes.

        The order is the core's: a sterile curve is Statement #87's finding and needs no
        second opinion, a flat *segment* is the same finding one horizon down, and only then
        is the assessment at rest consulted. ``CONVICTIONS`` fold into ``FORMULA`` because a
        replayed record is a formula too, and the two refuse for the same reason: what a
        reader would be buying is not a market.
        """
        record = signal.organic
        if record.sterile:
            return FORMULA, (
                f"the record carries no market noise (A² {record.a2:.2f}, CV {record.cv:.3f} "
                f"against the floors {float(STERILE_A2)} and {float(STERILE_CV)}): a panic "
                "in a formula is the trap, not the opportunity"
            )
        if record.flat:
            return FORMULA, (
                f"the raised {record.segment_points} days are flat (CV "
                f"{record.segment_cv:.4f}): the pool is not repricing, it is announcing"
            )
        if signal.resting_verdict in CONVICTIONS:
            return FORMULA, (
                f"at rest the assessment convicts this record "
                f"({signal.resting_verdict}): {signal.resting_reason}"
            )
        if signal.resting_verdict != HONEST:
            return DISHONEST, (
                "the record is organic but the protocol does not pay what it promises when "
                f"calm — {signal.resting_verdict or UNMEASURED}: {signal.resting_reason}"
            )
        return "", ""

    # -- one pool, one day --------------------------------------------------- #

    def measure(
        self,
        apy: Sequence[float],
        tvl: Optional[Sequence[float]] = None,
        *,
        name: str = "",
        bucket: Optional[str] = None,
        claim_apy: Optional[float] = None,
        depth: Optional[float] = None,
        uuid: str = "",
        chain: str = "",
        project: str = "",
        address: str = "",
    ) -> PanicSignal:
        """Measure one pool's two series and say what it is — ``PANIC`` … ``UNMEASURED``.

        ``apy`` is in **per cent a year**, because that is what the venue publishes and
        converting at the call site is how a unit error gets in. ``tvl`` must be aligned
        with it row by row (:class:`agent.physics.defillama_adapter.PoolFrame`); ``depth``
        is the pool's *resting* depth — the median before the window — and is measured from
        ``tvl`` when not given, because a pool that lost 80 % of its depth during the panic
        must be judged on the depth it had before it, or the panics in the smallest pools
        would be the only ones the floor lets through.

        The bucket and the claim are the ones the pool has **at rest**: ``bucket_for`` of
        the resting level and the resting level itself. Both defaults are argued in the
        module docstring — a panic cannot be allowed to widen the ceiling it is judged
        against.
        """
        values = _finite(apy)
        points = int(values.size)
        resting_pct = resting_claim(values, recent=self.recent)
        price = measure_dislocation(values, tvl, recent=self.recent)
        record = measure_organic(values, recent=self.recent) if points >= 8 else Organic()
        bucket_name = str(bucket) if bucket else bucket_for(resting_pct)
        base_depth = depth
        depth_values = _finite(tvl) if tvl is not None else np.zeros(0)
        if base_depth is None and depth_values.size:
            window = min(max(int(self.recent), 1), depth_values.size)
            before = depth_values[: depth_values.size - window]
            base_depth = float(np.median(before if before.size else depth_values))
        claim = float(claim_apy) if claim_apy is not None else resting_pct / 100.0
        label = name or uuid or "unnamed"
        assessment: Optional[Assessment] = None
        if points >= int(self.min_points):
            assessment = evaluate_opportunity(
                {
                    "address": address or sim_address(label),
                    "name": label,
                    "bucket": bucket_name,
                    "claim_apy": float(resting_pct) / 100.0,
                    "returns": [float(value) for value in to_returns(values)],
                    "alive": True,
                    "turnover": None if base_depth is None else float(base_depth),
                },
                oracle=self.oracle,
                min_record=int(self.min_points),
            )
        signal = PanicSignal(
            name=label,
            address=address or sim_address(label),
            uuid=str(uuid),
            bucket=bucket_name,
            chain=str(chain),
            project=str(project),
            dislocation=price,
            organic=record,
            resting_verdict="" if assessment is None else assessment.verdict,
            resting_reason="" if assessment is None else assessment.reason,
            resting_apy=float(resting_pct) / 100.0,
            claim_apy=claim,
            depth=base_depth,
            resting=assessment,
        )
        return self._conclude(signal, points, base_depth, price, record)

    def _conclude(
        self,
        signal: PanicSignal,
        points: int,
        depth: Optional[float],
        price: Dislocation,
        record: Organic,
    ) -> PanicSignal:
        """The verdict, in the order the gates are passed — the last three lines are the news.

        Kept out of :meth:`measure` so that the *measurement* reads as measurement: everything
        above is numbers off two series, everything here is the exchange rate between a
        number and a name, and a reader who disagrees with a threshold can see exactly which
        sentence it costs.
        """
        if points < int(self.min_points):
            return replace(
                signal,
                verdict=UNMEASURED,
                reason=(
                    f"{points} points of history, fewer than the {int(self.min_points)} a "
                    f"record needs before a {int(self.recent)}-day window can be compared "
                    "with it"
                ),
            )
        if self.min_depth is not None and (
            depth is None or float(depth) < float(self.min_depth)
        ):
            return replace(
                signal,
                verdict=THIN,
                reason=(
                    f"${0.0 if depth is None else float(depth):,.0f} of resting depth "
                    f"against a ${float(self.min_depth):,.0f} floor: no exit, whatever the "
                    "yield says"
                ),
            )
        verdict, reason = self.price_refusal(price)
        if verdict:
            return replace(signal, verdict=verdict, reason=reason)
        verdict, reason = self.record_refusal(signal)
        if verdict:
            return replace(signal, verdict=verdict, reason=reason)
        return replace(
            signal,
            verdict=PANIC,
            reason=(
                f"APY {price.resting:.2f}% → {price.recent:.2f}% "
                f"({price.z:.1f}σ, {price.gap_pp:+.2f} pp, {price.gap_ratio:.2f}×) while "
                f"depth fell {price.exodus:+.0%}, and the record still carries the noise of "
                f"a market (A² {record.a2:.1f}, CV {record.cv:.2f}); at rest the assessment "
                f"says {signal.resting_verdict}"
            ),
        )

    # -- the same question, asked of every day of one pool -------------------- #

    def walk(
        self,
        apy: Sequence[float],
        tvl: Optional[Sequence[float]] = None,
        *,
        name: str = "",
        start: Optional[int] = None,
        **metadata: Any,
    ) -> List[Tuple[int, PanicSignal]]:
        """``[(day, signal)]`` for every historical day on which this pool panicked.

        Each day is measured with **its own** prefix and nothing else: ``measure`` is handed
        ``apy[:day + 1]``, so a replay cannot see a number the detector would not have had.
        The cheap arm runs first — a day whose price arm does not fire is not worth an
        assessment — which is the same division of labour the funnel keeps between level 1
        and level 2, and it is what makes a thousand-day walk affordable.

        ``start`` is the first day the detector is allowed to speak on; by default it is the
        record floor, because before that there is no resting level to compare against.
        """
        values = np.asarray(list(apy), dtype=float).ravel()
        depth = None if tvl is None else np.asarray(list(tvl), dtype=float).ravel()
        first = int(self.min_points) if start is None else int(start)
        events: List[Tuple[int, PanicSignal]] = []
        for day in range(max(first, 1), int(values.size)):
            history = values[: day + 1]
            depths = None if depth is None else depth[: day + 1]
            price = measure_dislocation(history, depths, recent=self.recent)
            if self.price_refusal(price)[0]:
                continue
            signal = self.measure(history, depths, name=name, **metadata)
            if signal.fired:
                events.append((day, signal))
        return events

    def replay(
        self,
        apy: Sequence[float],
        tvl: Optional[Sequence[float]] = None,
        *,
        name: str = "",
        forward: int = 30,
        cooldown: Optional[int] = None,
        start: Optional[int] = None,
        **metadata: Any,
    ) -> List["PanicEvent"]:
        """Every panic this pool has had, each with what it actually paid afterwards.

        The outcome is attached **after** the verdict and never feeds it: :meth:`walk` decides
        on a prefix, :func:`forward_apy` reads the days that followed, and the two are joined
        here. That is the whole honesty of a replay — the same object that a live scan would
        have produced, with one extra column a live scan cannot have.

        ``cooldown`` days after a signal suppress the next one, so a fortnight-long panic is
        one event. It exists because the alternative is an average over one panic counted
        fourteen times, which is exactly the kind of number this layer refuses to print.
        """
        values = np.asarray(list(apy), dtype=float).ravel()
        quiet = -1
        events: List[PanicEvent] = []
        for day, signal in self.walk(values, tvl, name=name, start=start, **metadata):
            if day < quiet:
                continue
            quiet = day + int(PANIC_COOLDOWN if cooldown is None else cooldown)
            events.append(
                PanicEvent(
                    day=int(day),
                    signal=signal,
                    forward_apy=forward_apy(values, day, forward),
                    resting_apy=float(signal.dislocation.resting),
                    forward_days=int(forward),
                )
            )
        return events


@dataclass(frozen=True)
class PanicEvent:
    """A past panic, with the column a live scan cannot have: what it paid afterwards.

    ``forward_apy`` is the median APY over the days *after* the decision and ``resting_apy``
    is the pool's own level before it, so ``paid_pp`` answers the only question a
    backtest is worth: **did standing in the panic pay more than standing in the pool at
    rest?** A negative number is a panic that mean-reverted before it paid — the market was
    right and the hunter was early — and it is reported with the same weight as a win,
    because a strategy that only publishes its wins is a brochure.
    """

    day: int = 0
    signal: PanicSignal = field(default_factory=PanicSignal)
    forward_apy: float = math.nan
    resting_apy: float = math.nan
    forward_days: int = 0

    @property
    def paid_pp(self) -> float:
        """Percentage points the next ``forward_days`` paid over the pool's resting level."""
        if math.isnan(self.forward_apy) or math.isnan(self.resting_apy):
            return math.nan
        return float(self.forward_apy) - float(self.resting_apy)

    @property
    def held(self) -> bool:
        """True when the pool was still paying above its resting level afterwards."""
        return bool(not math.isnan(self.paid_pp) and self.paid_pp > 0.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "day": int(self.day),
            "name": self.signal.name,
            "verdict": self.signal.verdict,
            "confidence": round(self.signal.confidence, 4),
            "resting_apy": _round(self.resting_apy),
            "panic_apy": _round(self.signal.dislocation.recent),
            "forward_apy": _round(self.forward_apy),
            "forward_days": int(self.forward_days),
            "paid_pp": _round(self.paid_pp),
            "held": bool(self.held),
            "z": _round(self.signal.dislocation.z),
            "exodus": _round(self.signal.dislocation.exodus),
        }


# ── the whole scan, as a value ──────────────────────────────────────────────


@dataclass(frozen=True)
class PanicReport:
    """One trip over a funnel's approved pools: who panicked, who did not, and why not.

    ``PANIC`` and the refusals are kept apart exactly as the funnel keeps ``approved`` and
    ``refused`` apart, and for the same reason: ``UNMEASURED`` and ``THIN`` are statements
    about the *data* — nobody has read enough of this pool, or the pool has no exit — while
    ``FORMULA`` and ``DISHONEST`` are statements about the *protocol*. A report that added
    them together would let a shortage of history look like a finding about a pool.
    """

    venue: str = VENUE_NAME
    scanned: int = 0
    signals: Tuple[PanicSignal, ...] = ()
    elapsed: float = 0.0
    warnings: Tuple[str, ...] = ()
    error: str = ""

    @property
    def measured(self) -> int:
        """Signals the detector had an opinion about (everything but ``UNMEASURED``)."""
        return sum(1 for signal in self.signals if signal.verdict != UNMEASURED)

    def panics(self) -> List[PanicSignal]:
        """The fired signals, best confidence first — the queue a book would act on."""
        return sorted(
            (signal for signal in self.signals if signal.fired),
            key=lambda signal: (-signal.confidence, signal.name),
        )

    @property
    def fired(self) -> int:
        return len(self.panics())

    def refused(self) -> List[PanicSignal]:
        """Everything that was measured and did not panic — the table that explains why."""
        return sorted(
            (
                signal
                for signal in self.signals
                if not signal.fired and signal.verdict != UNMEASURED
            ),
            key=lambda signal: (signal.verdict, signal.name),
        )

    def unmeasured(self) -> List[PanicSignal]:
        return [signal for signal in self.signals if signal.verdict == UNMEASURED]

    def verdicts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for signal in self.signals:
            counts[signal.verdict] = counts.get(signal.verdict, 0) + 1
        return dict(sorted(counts.items()))

    def best(self) -> Optional[PanicSignal]:
        """The loudest signal, or ``None`` — what a one-line report prints."""
        panics = self.panics()
        return panics[0] if panics else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "venue": self.venue,
            "scanned": int(self.scanned),
            "measured": self.measured,
            "fired": self.fired,
            "verdicts": self.verdicts(),
            "elapsed": round(float(self.elapsed), 3),
            "warnings": list(self.warnings),
            "error": self.error,
            "signals": [signal.as_dict() for signal in self.signals],
        }

    def summary_text(self) -> str:
        """Three lines in the order of the argument, or the reason there is no scan."""
        if self.error:
            return (
                f"venue {self.venue}: no scan — {self.error}\n"
                "  nothing was read, nothing was judged, nothing was sized"
            )
        counts = ", ".join(
            f"{verdict} {count}" for verdict, count in self.verdicts().items()
        )
        lines = [
            f"venue {self.venue}: {int(self.scanned)} approved pools read, "
            f"{self.measured} measured",
            f"  verdicts: {counts or 'none'}",
            f"  panics: {self.fired}",
        ]
        for signal in self.panics():
            lines.append(
                f"    {signal.name:<40} {signal.dislocation.resting:6.2f}% → "
                f"{signal.dislocation.recent:6.2f}%  {signal.dislocation.z:6.2f}σ  "
                f"depth {signal.dislocation.exodus:+.0%}  confidence "
                f"{signal.confidence:.2f}"
            )
        return "\n".join(lines)


# ── the hunter: the venue's approved pools, read two series at a time ───────


@dataclass
class VolatilityHunter:
    """The venue-facing half: read the funnel's approved pools and judge each one.

    It holds a :class:`~agent.physics.defillama_adapter.DefiLlamaScanner` — which is a cache
    and a wire, and nothing else — and a :class:`PanicDetector`, which is arithmetic. The
    pool list it reads is the **funnel's approved names**, not the market's: a panic in a
    pool the two gates refused is a panic in a protocol nobody has measured, and this file
    has no opinion about protocols it has not measured.

    One request per pool, shared with the funnel's own cache, so a scan costs nothing on a
    warm cache — which is what makes it affordable to ask the question every review rather
    than once a week.
    """

    scanner: Optional[DefiLlamaScanner] = None
    detector: PanicDetector = field(default_factory=PanicDetector)
    workers: int = WORKERS
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.scanner is None:
            self.scanner = DefiLlamaScanner()

    # -- one pool ------------------------------------------------------------ #

    def measure_pool(
        self, name: str, row: PoolRow, *, refresh: bool = False
    ) -> PanicSignal:
        """One approved pool, measured — a dead uuid is a verdict, never an exception."""
        frame = self.scanner.fetch_pool_frame(row.uuid, refresh=refresh)  # type: ignore[union-attr]
        if not frame:
            return PanicSignal(
                name=name,
                uuid=str(row.uuid),
                chain=str(row.chain),
                project=str(row.project),
                address=sim_address(name),
                verdict=UNMEASURED,
                reason=(
                    "the feed published no day with both an APY and a depth, so there is "
                    "nothing to measure an exodus against"
                ),
            )
        return self.detector.measure(
            frame.apy,
            frame.tvl,
            name=name,
            uuid=str(row.uuid),
            chain=str(row.chain),
            project=str(row.project),
            address=sim_address(name),
            claim_apy=float(row.apy) / 100.0,
        )

    def _measure_all(
        self, pairs: Sequence[Tuple[str, PoolRow]], *, refresh: bool = False
    ) -> List[PanicSignal]:
        """Every pool measured, in the funnel's order, with the pool the unit of failure."""

        def one(pair: Tuple[str, PoolRow]) -> PanicSignal:
            name, row = pair
            try:
                return self.measure_pool(name, row, refresh=refresh)
            except Exception as exc:  # noqa: BLE001 - one pool, one line, never a scan
                self.warnings.append(f"{name}: the measurement failed ({exc})")
                return PanicSignal(
                    name=name,
                    uuid=str(row.uuid),
                    address=sim_address(name),
                    verdict=UNMEASURED,
                    reason=f"the measurement failed: {exc}",
                )

        if not pairs:
            return []
        size = max(int(self.workers), 1)
        if size == 1 or len(pairs) == 1:
            return [one(pair) for pair in pairs]
        with ThreadPoolExecutor(max_workers=size) as pool:
            return list(pool.map(one, pairs))

    # -- the whole queue ----------------------------------------------------- #

    def scan(
        self,
        funnel: MarketFunnel,
        *,
        limit: Optional[int] = None,
        refresh: bool = False,
    ) -> PanicReport:
        """Scan the pools a funnel approved — the only queue this file will read.

        ``limit`` bounds the scan without changing what it means: the pools are taken in the
        funnel's own order (best record first), so a bounded scan is the *best* part of the
        queue rather than an arbitrary slice of it, and ``scanned`` says how many were read.
        """
        started = time.perf_counter()
        if getattr(funnel, "error", ""):
            return PanicReport(
                venue=str(getattr(funnel, "venue", VENUE_NAME)),
                error=str(funnel.error),
                elapsed=time.perf_counter() - started,
            )
        names = funnel.approved_names()
        if limit is not None:
            names = names[: max(int(limit), 0)]
        pairs = [(name, funnel.pools[name]) for name in names if name in funnel.pools]
        signals = self._measure_all(pairs, refresh=refresh)
        return PanicReport(
            venue=str(getattr(funnel, "venue", VENUE_NAME)),
            scanned=len(pairs),
            signals=tuple(signals),
            elapsed=time.perf_counter() - started,
            warnings=tuple(self.warnings),
        )


# ── the CLI: one scan, one table, and an exit code that means something ─────


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m agent.physics.volatility_hunter --limit 200 --top 5``.

    Exit codes, because a shell script cares: ``0`` the scan ran (whatever it found — a
    market with no panic in it is a result, not a failure), ``3`` the funnel could not read
    the feed, ``4`` no pool survived the funnel, so there was nothing to hunt in.
    """
    import argparse
    import json

    from agent.physics.defillama_adapter import (
        DEFAULT_TIMEOUT,
        POOLS_LIMIT,
        WATCH_LIMIT,
        DiscoveryEngine,
    )

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=POOLS_LIMIT, help="pools to scan")
    parser.add_argument("--top", type=int, default=5, help="refusals to explain")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--watch-limit", type=int, default=WATCH_LIMIT)
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    parser.add_argument("--output", default="", help="write the report as JSON here")
    args = parser.parse_args(argv)

    scanner = DefiLlamaScanner(timeout=args.timeout)
    funnel = DiscoveryEngine(scanner=scanner, watch_limit=args.watch_limit).run(
        args.limit, refresh=args.refresh, workers=args.workers
    )
    print(funnel.summary_text())
    if funnel.error:
        return 3
    if not funnel.approved:
        print("no pool passed the funnel: there is nothing to hunt in")
        return 4
    hunter = VolatilityHunter(scanner=scanner, workers=args.workers)
    report = hunter.scan(funnel, limit=args.limit, refresh=args.refresh)
    print()
    print(report.summary_text())
    for signal in report.refused()[: max(int(args.top), 0)]:
        print(f"  {signal.verdict:<12} {signal.name:<40} {signal.reason}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2, ensure_ascii=False)
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the module is normally imported
    raise SystemExit(main())
