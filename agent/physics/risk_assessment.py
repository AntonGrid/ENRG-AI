"""Statement #87 pointed at a yield curve: is this a record, or is it a formula?

The sterility index was written for a memory relief and it answers one question: *does
this trace carry the residuals a lived medium leaves behind?* A protocol's return record
is the same object seen in a different market — a trace, one number per step, published
by somebody with an interest in what it looks like — so the same statistic applies
without a single new theorem. What this module adds is the reading: **a curve with no
market noise in it is not a good yield, it is a promise.**

Three measurements, and what each one can see
=============================================

``A²`` and ``CV``  (:mod:`agent.physics.sterility`, Statement #87, thresholds
                   ``STERILE_A2 = 0.1`` / ``STERILE_CV = 0.005``, copied not tuned). The
                   residuals of the record against its own mean rate: a lived record
                   leaves non-normal residuals (``A²`` of order one) and a spread of
                   order the yield; a formula leaves none of either. Measured on this
                   market: staking ``A² = 2.48, CV = 3.02``; bounty
                   ``A² = 219.7, CV = 2.80``; the formula trap
                   ``A² = 0.26, CV = 0.0001``.
``R²``             (:func:`agent.physics.curiosity.reducibility`, through the body's
                   own posture). Does the record repeat itself? A trap that serves the
                   same eight numbers every lap comes back at ``R² = 1.00``, ``14.5``
                   nats of structure, at exactly the lag its loop has — the *same*
                   measurement that names a re-served contract in ``worker.py``. It
                   fires on nothing honest here: the honest records measure
                   ``R² = 0.00``.
``reality`` / ``internal_resonance``
                   the mirror (``self_model``) judging the same windows. The replayed
                   trap scores ``reality = 0.00`` and ``resonance = 0.0000`` on every
                   fixation: the body has heard a formula before and knows the
                   difference between a trace and a thought. The honest curves sit at
                   ``0.65 … 0.86``.

The exchange rate between a measurement and a verdict
====================================================

None of the three can convict on its own, and the reason is measured rather than argued:
a **bond is sterile too**. A contractual fixed rate has ``CV = 1e-4`` and no residuals,
which is exactly the signature of the formula trap, and no statistic of the record can
tell the two apart — one is a promise that is kept and the other a promise that is not,
and both are promises. That is why the verdict needs a second input, the *claim*, and
why the rules are two arms that must agree:

.. code-block:: text

    sterile | repeat | unreal   ->  the record is not a record
    promised | inflated         ->  the claim is not one a market could pay

    both                        ->  SCAM_TRAP            (refuse: the brief's trap) the
    first alone             ->  SYNTHETIC_COLLUSION (refuse; the notebook prices
                                                        the yield it forgoes)
    the second alone            ->  SUSPECT             (refuse: a tenth of the empire
                                                        is not worth an unexplained
                                                        claim)
    neither                     ->  HONEST              (admissible)

What a claim may legally be is *declared per bucket* (:data:`PLAUSIBLE_APY`): a thousand
percent a year is absurd in a staking vault and merely ambitious in a farm, because the
farm is where the empire has already decided how much it can lose. Those numbers are
policy — an operator sets them — and every verdict prints them, so a reader can disagree
with one number instead of the whole rule.

The kill switch
===============

Three arms, each with its own reason string, because a risk manager that can only say
"down" cannot say *why*:

``drawdown``    the equity fell :data:`KILL_DRAWDOWN` (15 %) over one cycle
                (:data:`agent.physics.portfolio.CYCLE` steps) — the brief's rule, and
                the arm that fires when a market grinds down instead of gapping.
``volatility``  the realised volatility of the held protocols is
                :data:`KILL_VOLATILITY` times its own baseline — the arm that fires
                when a crash arrives as a *change of regime* rather than as a loss.
``resonance``   the mean ``internal_resonance`` of the records being held fell below
                :data:`RESONANCE_FLOOR`. This one does not detect a crash — measured,
                the crash's records keep ``resonance ≈ 0.80``, because a crash is still
                a lived trace — it detects that what is being held stopped being a trace
                at all.

``drawdown`` is checked first because it is the one the brief names and the one that
cannot be argued with; every trigger records the numbers it fired on, so a false
positive is found by reading the log rather than by guessing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agent.physics.curiosity import CURIOUS_MIN_WINDOWS, SURVEY_FIXATIONS
from agent.physics.portfolio import (
    BUCKETS,
    MIN_RECORD,
    PortfolioError,
    annualise,
    per_step_rate,
)
from agent.physics.self_model import WORLD_JUDGEMENT
from agent.physics.sensorium import WORLD, Perception
from agent.physics.sterility import (
    STERILE_A2,
    STERILE_CV,
    anderson_darling,
    variation,
)

# ── verdicts: what an assessment can say, and what it costs to be wrong ──────

#: The record is a record and the claim is a claim: the only verdict that funds.
HONEST = "HONEST"
#: There is not enough record to judge. Neither a refusal nor an approval.
UNMEASURED = "UNMEASURED"
#: The claim is implausible and the record does not explain it.
SUSPECT = "SUSPECT"
#: The record carries no market noise, or repeats itself: Statement #87's finding.
SYNTHETIC_COLLUSION = "SYNTHETIC_COLLUSION"
#: Both at once — a sterile curve *and* a promise no market pays. The brief's trap.
SCAM_TRAP = "SCAM_TRAP"
VERDICTS = (HONEST, UNMEASURED, SUSPECT, SYNTHETIC_COLLUSION, SCAM_TRAP)

#: The highest APY whose *sterile* curve is still a plausible contract, per bucket.
#: Declared policy, not a measurement: a hundred percent a year is impossible in a
#: staking vault and unremarkable in a farm, and the bucket is where the empire has
#: already said how much it is willing to lose.
PLAUSIBLE_APY = {"safe": 0.35, "active": 3.0, "experimental": 4.0}

#: Steps of record a *claim* is checked against: the recent ones, because a claim is
#: about the present. The sterility and structural arms keep the whole history — a
#: formula does not stop being one — but a claim is measured against the record it can
#: be measured on, and after a crash the honest answer is "not this quarter".
RATIO_WINDOW = 90

#: Standard errors between a claim and the record's own mean rate before the claim is
#: called unsupported. Three, the z a physicist asks for, and a *suspicion* threshold
#: rather than a conviction: measured on this market, a bursty bounty record pays 26 %
#: over its first sixty-four steps while claiming 177 % — nine standard errors, so the
#: claim is not supported *yet*, and it becomes supported when the jackpots arrive.
#: ``SUSPECT`` says exactly that ("wait"); only the record arm may say "liar".
CLAIM_Z = 3.0

#: ``R²`` above which a record is called a repeat. The gate inside
#: :func:`~agent.physics.curiosity.reducibility` has already decided significance; this
#: is the size a repeat has to be to matter, and measured it is not close: honest
#: records score ``0.00``, a re-served block ``1.00``.
REPEAT_R2 = 0.9

#: The mirror's reality score below which the arm treats its answer as *not the world*.
#: It is not a new number: :data:`agent.physics.self_model.WORLD_JUDGEMENT` is the
#: boundary the mirror itself already judges a stream by, so this arm and the mirror
#: cannot disagree about what the mirror said. Measured on this market, honest records
#: score ``0.63 … 0.99`` over four to eight windows and the replayed formula ``0.20 …
#: 0.37`` — the cut has somewhere to go, which is the only thing a threshold needs.
REALITY_FLOOR = float(WORLD_JUDGEMENT)

# ── the kill switch's numbers ────────────────────────────────────────────────

#: The brief's fifteen percent, measured over one cycle.
KILL_DRAWDOWN = 0.15
#: Realised volatility, as a multiple of the protocol's own baseline, that counts as a
#: change of regime rather than a bad day.
KILL_VOLATILITY = 3.0
#: The mean ``internal_resonance`` below which the held record stopped being a trace.
RESONANCE_FLOOR = 0.25
#: The switch is released only when the cycle is this calm again and the market's
#: volatility is back inside this multiple of its baseline.
RESUME_DRAWDOWN = 0.05
RESUME_VOLATILITY = 1.5
#: Steps over which "recent" volatility is measured, and steps of history it is compared
#: against. The pair is what makes the ratio mean "now" against "normally".
VOL_WINDOW = 24
VOL_BASELINE = 90
#: The smallest volatility the ratio is allowed to divide by. Below it the record is not
#: quiet, it is motionless, and a ratio against zero says nothing about risk.
VOL_FLOOR = 1.0e-6

#: The arms of the switch, in the order they are checked.
ARM_DRAWDOWN = "drawdown"
ARM_VOLATILITY = "volatility"
ARM_RESONANCE = "resonance"
ARMS = (ARM_DRAWDOWN, ARM_VOLATILITY, ARM_RESONANCE)


# ── the verdict, as a value ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Assessment:
    """One opportunity, measured: the two arms, their numbers, and the verdict.

    Everything the decision used is here, including the numbers that did *not* fire, so
    a reader can recompute the verdict with a pencil: ``sterile``/``repeat``/``unreal``
    are properties of the fields above them, ``promised``/``inflated`` are properties of
    the claim against the bucket's declared ceiling, and the verdict is a table lookup
    of the two flags.
    """

    name: str
    address: str
    bucket: str
    verdict: str
    reason: str
    claim_apy: float
    implied_apy: float
    claim_z: float = math.nan
    a2: float = math.nan
    cv: float = math.nan
    reducible: float = math.nan
    lag: int = 0
    structure: float = 0.0
    reality: float = math.nan
    reality_spread: float = math.nan
    resonance: float = math.nan
    points: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    # -- what the record said --------------------------------------------- #

    @property
    def measured(self) -> bool:
        """True when there was a record long enough to have an opinion about."""
        return self.verdict != UNMEASURED

    @property
    def sterile(self) -> bool:
        """Statement #87: no residuals, or no spread — a formula, not a record."""
        if math.isnan(self.a2) or math.isnan(self.cv):
            return False
        return bool(self.a2 < STERILE_A2 or self.cv < STERILE_CV)

    @property
    def repeat(self) -> bool:
        """The structural arm: the record re-serves itself at a lag."""
        return bool(not math.isnan(self.reducible) and self.reducible >= REPEAT_R2)

    @property
    def unreal(self) -> bool:
        """The mirror's arm: the body says it has never heard a stream like this."""
        return bool(not math.isnan(self.reality) and self.reality < REALITY_FLOOR)

    # -- what the claim said ---------------------------------------------- #

    @property
    def ratio(self) -> float:
        """The claim divided by what the recent record implies; ``inf`` on a loss."""
        if self.implied_apy <= 0.0:
            return math.inf if self.claim_apy > 0.0 else 0.0
        return float(self.claim_apy / self.implied_apy)

    @property
    def promised(self) -> bool:
        """The claim is above what this bucket may legally promise."""
        ceiling = float(PLAUSIBLE_APY.get(self.bucket, PLAUSIBLE_APY["safe"]))
        return bool(self.claim_apy > ceiling)

    @property
    def inflated(self) -> bool:
        """The claim is more than :data:`CLAIM_Z` standard errors above the record."""
        if math.isnan(self.claim_z):
            return False
        return bool(self.claim_z > CLAIM_Z)

    @property
    def ghost(self) -> bool:
        """A promise of profit on a record that has lost money."""
        return bool(self.implied_apy <= 0.0 < self.claim_apy)

    # -- what may be done with it ----------------------------------------- #

    @property
    def admissible(self) -> bool:
        """Only ``HONEST`` funds. Everything else, including ``UNMEASURED``, waits."""
        return self.verdict == HONEST

    @property
    def scam(self) -> bool:
        """Statement #87's finding, with or without an impossible claim attached."""
        return self.verdict in (SYNTHETIC_COLLUSION, SCAM_TRAP)

    def as_dict(self) -> Dict[str, Any]:
        """Every number the verdict was computed from, JSON-able and rounded."""
        return {
            "name": self.name,
            "address": self.address,
            "bucket": self.bucket,
            "verdict": self.verdict,
            "admissible": bool(self.admissible),
            "reason": self.reason,
            "claim_apy": round(float(self.claim_apy), 6),
            "implied_apy": round(float(self.implied_apy), 6),
            "claim_z": _round(self.claim_z, 4),
            "ratio": None if math.isinf(self.ratio) else round(self.ratio, 4),
            "a2": _round(self.a2),
            "cv": _round(self.cv),
            "sterile": bool(self.sterile),
            "reducible": _round(self.reducible),
            "lag": int(self.lag),
            "structure": round(float(self.structure), 4),
            "repeat": bool(self.repeat),
            "reality": _round(self.reality),
            "reality_spread": _round(self.reality_spread),
            "resonance": _round(self.resonance),
            "unreal": bool(self.unreal),
            "promised": bool(self.promised),
            "inflated": bool(self.inflated),
            "ghost": bool(self.ghost),
            "points": int(self.points),
            "details": dict(self.details),
        }


def _round(value: float, digits: int = 6) -> Optional[float]:
    """Round, keeping ``NaN`` as ``None`` — JSON has no NaN and a report is JSON."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


# ── the two arms, measured ──────────────────────────────────────────────────


def claim_statistics(
    returns: Sequence[float], claim_apy: float, *, window: int = RATIO_WINDOW
) -> Tuple[float, float]:
    """``(implied_apy, z)`` of a claim against the recent record: does it pay it?

    The record's last :data:`RATIO_WINDOW` steps give a mean rate and a standard error;
    the claim gives the rate it *says* it pays. ``z`` is how many standard errors apart
    they are, which is the only honest way to compare a promise with a bursty record: a
    bounty board's payouts arrive in jackpots, so its mean over ninety steps wobbles by
    half, and a raw ratio would call that wobble a lie.

    The implied APY is the recent window's own mean, compounded — the same reading a
    person takes off a yield chart, and the number printed beside the claim.
    """
    values = np.asarray(returns, dtype=float).ravel()
    recent = values[-int(window) :] if window and window > 0 else values
    n = int(recent.size)
    if n < 8:
        return 0.0, math.nan
    mean = float(np.mean(recent))
    spread = float(np.std(recent, ddof=1))
    error = spread / math.sqrt(float(n))
    try:
        claimed = float(per_step_rate(claim_apy))
    except PortfolioError:
        return float(annualise(mean)), math.nan
    if error <= 0.0:
        return float(annualise(mean)), 0.0 if claimed <= mean else math.inf
    return float(annualise(mean)), float((claimed - mean) / error)


def sterility_of(returns: Sequence[float]) -> Tuple[float, float, int]:
    """``(A², CV, points)`` of a return record — Statement #87 on a yield curve.

    The model being tested is *"a constant rate plus the noise a market carries"*, so
    the residuals are the record minus its own mean. ``A²`` re-centres internally, which
    is why the mean is not passed in: the statistic is already invariant to the level of
    the yield, and passing a *claimed* rate instead would make any wrong claim look
    sterile.

    ``A² = 0`` means every step paid the same number — the failure mode, not a perfect
    score. ``CV = std/mean`` is the same statement in the units a person uses, and it is
    the one that catches a curve whose residuals are *nearly* equal: measured, the
    formula trap has ``A² = 0.26`` (close to the 0.1 floor) but ``CV = 0.0001``, four
    hundred times below its own floor.
    """
    values = np.asarray(returns, dtype=float).ravel()
    if values.size < 8:
        return math.nan, math.nan, int(values.size)
    return float(anderson_darling(values)), float(variation(values)), int(values.size)


def measure_structure(returns: Sequence[float], oracle: Any) -> Dict[str, float]:
    """Judge a record the way a contract is judged: through the body's own posture.

    ``oracle`` is a :class:`agent.physics.worker.RealityOracle` — the same object that
    prices a contract — so this arm is not a second opinion, it is the same opinion
    asked about a different stream. It is passed in rather than built here because
    building one means hatching an organism, which is a decision the caller should make
    explicitly: an assessment without an oracle is a weaker assessment and says so.
    """
    values = np.asarray(returns, dtype=float).ravel()
    perception = Perception(
        values=values, origin=WORLD, source="protocol:record", domain="finance"
    )
    episode = oracle.sensorium.fixate(perception, limit=int(SURVEY_FIXATIONS))
    moments = oracle.mirror.observe(episode, learn=False)
    windows = int(len(episode.readings))
    if windows < CURIOUS_MIN_WINDOWS:
        # Fewer windows than the survey's floor is not "no repeat", it is no
        # measurement: the arm reports nothing rather than a zero that would read as
        # a clean record.
        return {"windows": windows}
    episodes = [episode]
    oracle.engine.reset()
    traits = oracle.engine.survey(episodes, moments=[moments])
    if not traits:
        return {"windows": windows}
    trait = traits[0]
    resonance = np.array([float(moment.resonance) for moment in moments], dtype=float)
    reality = np.array([float(moment.reality_score) for moment in moments], dtype=float)
    return {
        "windows": windows,
        "reducible": float(trait.reducible),
        "lag": int(trait.lag),
        "structure": float(trait.structure),
        "reality": float(trait.reality),
        "reality_spread": float(reality.std()) if reality.size else math.nan,
        "resonance": float(resonance.mean()) if resonance.size else math.nan,
    }


# ── the function the brief asked for ────────────────────────────────────────


def _read(opportunity_data: Any) -> Dict[str, Any]:
    """Normalise an opportunity: a mapping (what an adapter publishes) or an object."""
    if isinstance(opportunity_data, Mapping):
        get = opportunity_data.get
    else:
        get = lambda key, default=None: getattr(opportunity_data, key, default)
    address = str(get("address", "") or "")
    name = str(get("name", "") or address or "unnamed")
    bucket = str(get("bucket", "") or "safe")
    returns = get("returns", None)
    if returns is None:
        returns = get("record", ())
    claim = get("claim_apy", None)
    if claim is None:
        claim = get("apy", 0.0)
    return {
        "name": name,
        "address": address,
        "bucket": bucket if bucket in BUCKETS else "safe",
        "claim_apy": float(claim or 0.0),
        "returns": np.asarray(list(returns), dtype=float).ravel(),
        "alive": bool(get("alive", True)),
    }


def _record_reason(assessment: "Assessment") -> List[str]:
    """What the record itself said, in the fewest words that are still checkable."""
    parts: List[str] = []
    if assessment.sterile:
        # The arm is an *or*, so the reason must name the one that actually fired —
        # printing both with a "<" between them would claim a comparison that is false.
        arms = []
        if assessment.a2 < STERILE_A2:
            arms.append(f"A² {assessment.a2:.2f} < {STERILE_A2}")
        if assessment.cv < STERILE_CV:
            arms.append(f"CV {assessment.cv:.4f} < {STERILE_CV}")
        parts.append(f"sterile ({', '.join(arms)})")
    if assessment.repeat:
        parts.append(
            f"repeats (R² {assessment.reducible:.2f} at lag {assessment.lag}, "
            f"{assessment.structure:.1f} nats)"
        )
    if assessment.unreal:
        parts.append(f"unreal (reality {assessment.reality:.2f} < {REALITY_FLOOR})")
    return parts


def _claim_reason(assessment: "Assessment") -> List[str]:
    """What the promise said, against the bucket's ceiling and its own record."""
    parts: List[str] = []
    ceiling = float(PLAUSIBLE_APY.get(assessment.bucket, PLAUSIBLE_APY["safe"]))
    if assessment.promised:
        parts.append(
            f"promises {assessment.claim_apy:.0%} against the {ceiling:.0%} allowed "
            f"in {assessment.bucket}"
        )
    if assessment.inflated:
        parts.append(
            f"claims {assessment.claim_apy:.0%} while its own record pays "
            f"{assessment.implied_apy:.0%}: {assessment.claim_z:.1f} standard errors "
            f"apart"
        )
    if assessment.ghost:
        parts.append(
            f"promises {assessment.claim_apy:.0%} on a record that lost money "
            f"({assessment.implied_apy:.0%})"
        )
    return parts


def _convict(assessment: "Assessment") -> Tuple[str, str]:
    """The table from the module docstring, in code: two arms, four outcomes."""
    record = _record_reason(assessment)
    claim = _claim_reason(assessment)
    if record and claim:
        return SCAM_TRAP, "; ".join(record + claim)
    if record:
        return SYNTHETIC_COLLUSION, "; ".join(record)
    if claim:
        return SUSPECT, "; ".join(claim)
    return HONEST, (
        f"the record carries noise (A² {assessment.a2:.2f}, CV {assessment.cv:.2f}) "
        f"and pays what it claims ({assessment.claim_apy:.0%})"
    )


def evaluate_opportunity(
    opportunity_data: Any,
    *,
    oracle: Any = None,
    min_record: int = MIN_RECORD,
) -> Assessment:
    """Measure one opportunity and say what it is: ``HONEST`` … ``SCAM_TRAP``.

    ``opportunity_data`` is what an adapter publishes — a mapping with ``address``,
    ``name``, ``bucket``, ``claim_apy``, ``returns`` and ``alive`` — or any object with
    those attributes (:class:`agent.physics.portfolio.Protocol` is one). Only the record
    up to now may be in it: this function has no way to ask for a step that has not
    happened, which is what makes the refusal in the log a decision and not a script.

    ``oracle`` is optional and its absence is not hidden: without it the structural and
    mirror arms report ``NaN`` and the verdict rests on Statement #87 and the claim.
    With it (a :class:`agent.physics.worker.RealityOracle`) the arms that catch a
    re-served record come alive, and ``details["measured"]`` says which assessment this
    was.

    Raises:
        Nothing. A malformed opportunity is not an exception — a protocol that publishes
        nothing is a protocol to refuse, and a rule that crashes on a bad feed cannot be
        run on a real one.
    """
    record = _read(opportunity_data)
    returns = record["returns"]
    points = int(returns.size)
    implied, claim_z = claim_statistics(returns, record["claim_apy"])
    base: Dict[str, Any] = {
        "name": record["name"],
        "address": record["address"],
        "bucket": record["bucket"],
        "claim_apy": float(record["claim_apy"]),
        "implied_apy": float(implied) if points else 0.0,
        "claim_z": claim_z,
        "points": points,
    }
    if points < int(min_record):
        return Assessment(
            verdict=UNMEASURED,
            reason=(
                f"{points} steps of record, fewer than the {int(min_record)} a record "
                "needs before anybody may have an opinion about it"
            ),
            details={"min_record": int(min_record), "measured": False},
            **base,
        )
    a2, cv, _ = sterility_of(returns)
    structure = measure_structure(returns, oracle) if oracle is not None else {}
    draft = Assessment(
        verdict=UNMEASURED,
        reason="",
        a2=a2,
        cv=cv,
        reducible=float(structure.get("reducible", math.nan)),
        lag=int(structure.get("lag", 0)),
        structure=float(structure.get("structure", 0.0)),
        reality=float(structure.get("reality", math.nan)),
        reality_spread=float(structure.get("reality_spread", math.nan)),
        resonance=float(structure.get("resonance", math.nan)),
        details={
            "min_record": int(min_record),
            "measured": bool(structure),
            "windows": int(structure.get("windows", 0)),
            "alive": bool(record["alive"]),
            "a2_threshold": float(STERILE_A2),
            "cv_threshold": float(STERILE_CV),
            "repeat_threshold": float(REPEAT_R2),
            "claim_ceiling": float(
                PLAUSIBLE_APY.get(record["bucket"], PLAUSIBLE_APY["safe"])
            ),
        },
        **base,
    )
    if not record["alive"]:
        return replace(
            draft,
            verdict=SUSPECT,
            reason="the protocol has stopped paying: its record ends in a total loss",
        )
    verdict, reason = _convict(draft)
    return replace(draft, verdict=verdict, reason=reason)


def review(
    market: Any, *, oracle: Any = None, min_record: int = MIN_RECORD
) -> Dict[str, Assessment]:
    """Assess every protocol in a market's *published* state, by name."""
    step = int(getattr(market, "step", 0))
    return {
        protocol.name: evaluate_opportunity(
            protocol.as_opportunity(step),
            oracle=oracle,
            min_record=min_record,
        )
        for protocol in market.protocols
    }


# ── the kill switch: three arms, one action ─────────────────────────────────


def volatility(returns: Sequence[float], window: int = VOL_WINDOW) -> float:
    """Standard deviation of the last ``window`` returns; ``0.0`` when too short.

    Twenty-four by default — one cycle (:data:`agent.physics.portfolio.CYCLE`), which is
    the shortest window a spread can be estimated on without the estimate itself
    becoming the noise. Measured on this market: over twelve steps, a *robust* spread
    (``1.4826 × MAD``) reads 0.5 at one step and 28.8 at the next, because a
    twelve-sample median is improvised; over twenty-four steps the same record reads
    between 0.8 and 2.4 in calm weather and above 10 in a crash.
    """
    values = np.asarray(returns, dtype=float).ravel()
    if values.size < 2:
        return 0.0
    tail = values[-int(max(window, 2)) :]
    if tail.size < 2:
        return 0.0
    return float(np.std(tail, ddof=1))


def volatility_ratio(returns: Sequence[float], *, window: int = VOL_WINDOW) -> float:
    """Recent spread over the record's own history: "now" against "before".

    The baseline is the *whole* published history before the stretch being judged, and
    both sides are the same estimator. Both choices were forced by measurement. A
    baseline of the last ninety steps catches up with a crash within six steps, and a
    ``−7.5 %``-a-step regime reads 1.6 against its own aftermath, a calm market by this
    rule — and a baseline that widens only enough to include the *previous* crash still
    reads 2.6 during the spike. And a robust estimator on a short window is improvised
    statistics: ``1.4826 × MAD`` over twelve points swings between 0.5 and 28.8 on a
    record that has not changed at all. So: plain standard deviations, one cycle on the
    recent side, the whole life on the other.

    The denominator is floored at :data:`VOL_FLOOR` so a record that has never moved
    does not divide by zero — a bond's volatility is zero, and ``inf`` would make a bond
    look like the most dangerous thing in the market.
    """
    values = np.asarray(returns, dtype=float).ravel()
    if values.size < 2 * int(window):
        return 1.0
    past = values[: -int(window)]
    now = volatility(values, window=int(window))
    return float(now / max(volatility(past, window=int(past.size)), VOL_FLOOR))


@dataclass(frozen=True)
class Trigger:
    """One firing of the switch: which arm, on what number, and what was done."""

    step: int
    arm: str
    reason: str
    drawdown: float
    volatility: Optional[float] = None
    resonance: Optional[float] = None
    action: str = "cash"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "arm": self.arm,
            "reason": self.reason,
            "drawdown": round(float(self.drawdown), 6),
            "volatility": (
                None if self.volatility is None else _round(self.volatility)
            ),
            "resonance": None if self.resonance is None else _round(self.resonance),
            "action": self.action,
        }


@dataclass
class KillSwitch:
    """Capital preservation, as three independent measurements and one action.

    The switch is *stateful* and that is the whole point: once it fires, the risky
    buckets stay in cash until the measurements say the market is calm again, and
    neither the portfolio nor the assessment can talk it out of that. Release needs
    every measuring arm quiet — the cycle drawdown back inside :data:`RESUME_DRAWDOWN`,
    the market's volatility back inside :data:`RESUME_VOLATILITY` of its baseline, and
    the resonance of what would be bought back above :data:`RESONANCE_FLOOR`.
    """

    drawdown_limit: float = KILL_DRAWDOWN
    volatility_multiple: float = KILL_VOLATILITY
    resonance_floor: float = RESONANCE_FLOOR
    resume_drawdown: float = RESUME_DRAWDOWN
    resume_volatility: float = RESUME_VOLATILITY
    engaged: bool = False
    triggers: List[Trigger] = field(default_factory=list)

    def drawdown(self, equity: float, reference: float) -> float:
        """How far equity has fallen from a reference level, as a positive fraction."""
        base = float(reference)
        if base <= 0.0:
            return 0.0
        return float(max(0.0, 1.0 - float(equity) / base))

    def check(
        self,
        *,
        step: int,
        equity: float,
        reference: float,
        volatility: Optional[float] = None,
        resonance: Optional[float] = None,
    ) -> Optional[Trigger]:
        """Fire, or say nothing. ``reference`` is the equity one cycle ago.

        The arms are checked in the order the brief lists them, and the first one over
        its limit names the trigger. An arm that was not measured (``None``) cannot
        fire: unmeasured risk is not a safe one, it is an unmeasured one, and the
        trigger's own fields say which numbers were live by leaving the others out.
        """
        if self.engaged:
            return None
        fall = self.drawdown(equity, reference)
        if fall >= float(self.drawdown_limit):
            return self._fire(
                step,
                ARM_DRAWDOWN,
                f"equity fell {fall:.1%} over one cycle "
                f"(limit {self.drawdown_limit:.0%})",
                fall,
                volatility,
                resonance,
            )
        if volatility is not None and float(volatility) >= float(
            self.volatility_multiple
        ):
            return self._fire(
                step,
                ARM_VOLATILITY,
                f"realised volatility {float(volatility):.1f}x its baseline "
                f"(limit {self.volatility_multiple:.1f}x)",
                fall,
                volatility,
                resonance,
            )
        if resonance is not None and float(resonance) < float(self.resonance_floor):
            return self._fire(
                step,
                ARM_RESONANCE,
                f"internal_resonance {float(resonance):.2f} < floor "
                f"{self.resonance_floor:.2f}",
                fall,
                volatility,
                resonance,
            )
        return None

    def release(
        self,
        *,
        step: int,
        equity: float,
        reference: float,
        volatility: Optional[float] = None,
        resonance: Optional[float] = None,
    ) -> Optional[Trigger]:
        """Disengage when every arm is quiet again; returns the release, or ``None``."""
        if not self.engaged:
            return None
        fall = self.drawdown(equity, reference)
        calm = fall <= float(self.resume_drawdown)
        if volatility is not None:
            calm = calm and float(volatility) <= float(self.resume_volatility)
        if resonance is not None:
            calm = calm and float(resonance) >= float(self.resonance_floor)
        if not calm:
            return None
        self.engaged = False
        event = Trigger(
            step=int(step),
            arm="release",
            reason=(
                f"calm again (drawdown {fall:.1%}, volatility {_show(volatility)}, "
                f"resonance {_show(resonance)}): risk may be taken again"
            ),
            drawdown=fall,
            volatility=volatility,
            resonance=resonance,
            action="reinvest",
        )
        self.triggers.append(event)
        return event

    def _fire(
        self,
        step: int,
        arm: str,
        reason: str,
        drawdown: float,
        volatility: Optional[float],
        resonance: Optional[float],
    ) -> Trigger:
        self.engaged = True
        event = Trigger(
            step=int(step),
            arm=arm,
            reason=reason,
            drawdown=float(drawdown),
            volatility=volatility,
            resonance=resonance,
        )
        self.triggers.append(event)
        return event

    def as_dict(self) -> Dict[str, Any]:
        return {
            "engaged": bool(self.engaged),
            "limits": {
                "drawdown": float(self.drawdown_limit),
                "volatility": float(self.volatility_multiple),
                "resonance": float(self.resonance_floor),
            },
            "resume": {
                "drawdown": float(self.resume_drawdown),
                "volatility": float(self.resume_volatility),
            },
            "triggers": [event.as_dict() for event in self.triggers],
        }


def _show(value: Optional[float]) -> str:
    """A number for a log line, or a word that admits it was not measured."""
    return "unmeasured" if value is None else f"{float(value):.2f}"


__all__ = [
    "ARM_DRAWDOWN",
    "ARM_RESONANCE",
    "ARM_VOLATILITY",
    "ARMS",
    "Assessment",
    "CLAIM_RATIO_MAX",
    "HONEST",
    "KILL_DRAWDOWN",
    "KILL_VOLATILITY",
    "KillSwitch",
    "PLAUSIBLE_APY",
    "REALITY_FLOOR",
    "REPEAT_R2",
    "RESONANCE_FLOOR",
    "RESUME_DRAWDOWN",
    "RESUME_VOLATILITY",
    "SCAM_TRAP",
    "SUSPECT",
    "SYNTHETIC_COLLUSION",
    "Trigger",
    "UNMEASURED",
    "VERDICTS",
    "VOL_BASELINE",
    "VOL_FLOOR",
    "VOL_WINDOW",
    "evaluate_opportunity",
    "measure_structure",
    "review",
    "sterility_of",
    "volatility",
    "volatility_ratio",
]








