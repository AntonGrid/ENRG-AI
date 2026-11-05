"""Curiosity: what one more look is worth, and the questions it pays for.

The mirror answered *"is this the world, or is this something I made?"*. This
module answers the next question a body has to answer if it is not to waste its
life: **"which of the streams in front of me is worth another look?"** A creature
that spends equal attention on everything learns everything badly; a creature
that is only surprised learns noise. The number that decides is an *expected
information gain*, and the whole of this file is one honest construction of it.

Where the number comes from
===========================

Active inference (Friston) calls the quantity *epistemic value*; statistics calls
it the Lindley gain, and the two agree. For a Gaussian belief with uncertainty
``τ²`` about a quantity buried in noise ``σ²``, one more observation buys

.. math:: \\Delta H = \\tfrac12 \\ln\\left(1 + \\frac{\\tau^2}{\\sigma^2}\\right)

nats, and :func:`lindley_gain` is that formula, kept in the file so the
demonstration can print it. It is honest and it is useless as a *choice* rule,
because it does not depend on the stream: for two streams the same body has
looked at the same number of times, it returns the same number. What differs
between streams is not how much a look buys but **how much of the stream there
is to buy**, and that is measured here in two parts.

``movement``  — how much this stream moves at all, relative to the streams this
                body actually has. A frozen feed and a feed of pure noise differ
                in nothing that a variance cannot see: one has none, the other
                has nothing else.

``structure`` — how much of that movement the stream explains *itself*: the
                best single-lag copy predictor, ``f(t) ≈ a + b·f(t−k)``, scored
                by its :math:`R^2` **against the null distribution of `R^2`**,
                so a repeat is only believed when a repeat is unlikely by
                chance. This is the "edge of chaos" of the whole exercise: a
                stream with no self-similarity (noise) gives ``R² = 1/n`` and
                nothing to learn, a stream that is *one* window long (frozen)
                has no movement to explain, and only between the two is there
                something a body can be taught.

The product of the two, gated by the mirror's own verdict and discounted by the
questions already asked, is :attr:`Trait.curiosity`:

.. math:: c = R \\cdot \\min\\left(1, \\frac{2V}{V + V_\\text{ref}}\\right) \\cdot
              \\left(1 - e^{-\\Lambda/0.25}\\right) \\cdot \\frac{1}{1 + 0.5\\,\\text{looks}}

``V_ref`` is not a constant — it is the **median** ``movement`` of the streams in
the same survey, i.e. "what this body is used to". Turning ``V`` into
``2V/(V+V_ref)`` and clamping at one is what makes movement a *gate* instead of a
fifth opinion: in this frame a noise stream and a structured stream are equally
loud, so ranking by variance would rank noise highest, and the only thing
movement is asked to decide is whether there is anything there at all. Ranking is
the structure's job.

The gate ``R`` is the mirror's mean reality score for the stream, which is why the
organism is never curious about its own noise, and why a *believable* forgery is
exactly the thing it has to be suspicious of.

The last factor is the only one *declared* rather than measured: each question
about a stream costs a fixed share of that stream's curiosity (:data:`LOOK_COST`),
so the second question is worth two-thirds of the first and the third one half.
What that buys is termination. Curiosity is a price of attention, and a body that
wants to look at everything would spend its life nowhere.

What this metric is not
=======================

* It is not surprise. Surprise is maximal on noise, and a body that chases
  surprise chases noise; the notebook measures the two rankings on the same
  survey and shows the difference.
* It is not the Lindley gain: that number is printed next to it, and it is
  constant across streams, which is precisely why the choice needs the two
  factors above.
* It is not a proof of anomaly. A stream that repeats itself gets attention
  whether it is a genuine cycle or a copied block, and the module says so —
  deciding *which* it is takes a probe, and the probe is what this module asks
  for.

What the repeat test can and cannot see
=======================================

The structure arm compares the feature vectors of whole windows, so what it catches
is a **payload re-served**, not a *periodicity of a summary*. Measured on the
offline sensorium at the survey's 24 windows of 64 samples:

* a counterfeit — a block of a real feed repeated — comes back at exactly its own
  period: blocks of 128, 192, 256 and 512 samples at lags 1, 3, 4 and 8 with
  ``R² = 1.00``, and with ``σ = 0.25`` of added noise still ``R² ≥ 0.98``. The
  128-sample block reports lag 1 rather than lag 2 because two alternating windows
  predict each other exactly through a slope of ``-1``, and the tie band reports
  the smallest lag that is nearly as good;
* a replay of an earlier stretch comes back at its own offset — 768 samples → lag
  12 — with ``R² = 0.83``, and what is missing from 1.00 is arithmetic rather than
  noise: the survey's window is 128 samples at a 64-sample hop, so the single
  window that straddles the point where the copy begins mixes both halves and
  cannot pair with its partner exactly. Eleven pairs of twelve are identical to the
  last bit, and the twelfth is the price of a 768-sample period measured in a
  128-sample frame;
* a *clean* rhythm is caught, and the answer it gets is the frame's own: a sine of
  period 512 samples returns ``R² = 1.00`` at lag 4, because the aggregates are
  invariant to a sign flip and so its half period is the fundamental period *in
  this frame*. Its reality is 0.00, because the self model has never heard a
  noiseless signal and says so — structure without reality is worth nothing;
* the same rhythm with ``σ = 0.3`` of noise is **not** called a repeat. That is a
  limitation, not a bug: a sine's window aggregates vary by less than the noise
  does, so once the phase is gone there is no self-similarity left to find. What
  the arm reports is "this is that same window again" — which is exactly the
  question a forgery has to answer, and not the question a melody asks.

On the same evidence thirty permutations of a real feed, a random walk, an AR(1)
process at the survey's resolution, a frozen feed and the honest feeds themselves
fire **none** of the lags: 34 streams with no repeat in them, 0 false positives.

Probes
======

When a stream's curiosity clears :data:`PROBE_THRESHOLD` the organism stops
waiting for data and *asks* for it: :class:`Probe` is the question, and the
sensorium's ``request_probe(stream, anchor=..., kind=...)`` is the answer — a
window of samples centred on the fixation that was least expected, or the next
package of the same stream. Three kinds of question are used, and they are the
three available to a body that can only look:

``context``  show me what happened around the anomaly (±``context`` samples).
``next``     show me the following package of the same stream.
``compare``  show me this stream beside the one I find least interesting.

How sure the survey is
======================

A ranking is not a belief, and a body that is paid for being right has to know the
difference. :meth:`Curiosity.certainty` answers *"how much of my answer is
arithmetic and how much is the draw"* by nudging the survey's own rows by
:data:`PERTURBATION_JITTER` of their standard deviation and re-running the same
test: a margin that survives five such nudges is a fact about the feed, and one
that changes its mind is a tie — reported as ``suspect = None`` (no answer) rather
than as the least bad answer. Two arms are carried: the structure margin with its
significance, and the spread of the mirror's ``internal_resonance`` over the
suspect — the second is not multiplied in, it is reported beside the first.

:class:`UCB` is the same appetite one level up: not *which stream* to look at, but
*which kind of question* to keep bidding for. Its score is the UCB1 rule with the
rewards coming in as credit rates, so the optimism about an untried kind and the
win probability of a finished one are on one scale and can be compared.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.physics.homeostasis import Posture
from agent.physics.self_model import WORLD_PRIOR, reality_score

#: Shortest stream worth having an opinion about, in survey windows. Below this
#: the numbers exist but their confidence intervals overlap everything, and the
#: engine says "blind" rather than inventing a ranking.
CURIOUS_MIN_WINDOWS = 4

#: The survey's geometry: the sensorium's own window and hop, capped at this many
#: fixations. The geometry is *not* a new one, deliberately — the posture was
#: standardised on the sensorium's windows, and a survey at another window length
#: would measure features in a frame it was never fitted to, which shows up as
#: every stream looking equally unreal. Capping, on the other hand, is a real
#: limitation: a stream longer than twenty-four fixations is judged on its first
#: twenty-four — twenty-three hops plus one window, 1600 samples at the sensorium's
#: default window of 128 and hop of 64.
#:
#: Consecutive, evenly spaced hops are the other requirement. A repeat is only
#: measurable as a repeat when the fixations tile the stream at one fixed hop, so
#: a forgery that re-serves a cached block of samples appears as *the same
#: fixation* a whole number of hops later — exactly, not approximately.
SURVEY_FIXATIONS = 24

#: Samples per fixation, used only when an episode does not report it. The
#: survey itself runs in the sensorium's geometry; this is the floor under a
#: caller that hands in a hand-built episode.
DEFAULT_WINDOW = 128

#: Features are clipped at this many standard deviations before any magnitude is
#: used. The posture's standardiser divides by a median absolute deviation, which
#: for a feature that is constant in most of the evidence (``stillness`` is 0 for
#: every continuous feed) is a division by almost zero; unclipped, one such
#: feature would be the only one the metric ever saw.
Z_CLIP = 6.0

#: Fraction of the best :math:`R^2` a shorter lag must reach to be reported
#: instead. A repeat with a period is also a repeat at every multiple of the
#: period, and the two :math:`R^2` then differ only by noise — without this band
#: the report says "every twenty fixations" about a stream that repeats every
#: two. The smallest lag that is nearly as good is the fundamental period, and it
#: is the one the probe's question is about.
LAG_TIE = 0.9

#: Nats of structure that saturate the score: ``1 - exp(-Λ/GAIN_SCALE)``.
GAIN_SCALE = 0.25

#: Family-wise error rate for "this stream repeats itself". Several lags are
#: tried, so this is the level of the whole family, not of one test.
PROBE_ALPHA = 1e-2

#: Curiosity at which the organism stops watching and asks.
PROBE_THRESHOLD = 0.35
#: Curiosity at which the stream is called an anomaly in the report.
ANOMALY_THRESHOLD = 0.60
#: Questions per stream. A body that can ask forever never learns anything else.
MAX_PROBES = 3
#: How much of a stream's curiosity one question is worth: the fade is
#: ``1/(1 + LOOK_COST * looks)``, so the second question is worth two-thirds of
#: the first and the third one half. With ``1/(1+looks)`` the third question is
#: unreachable arithmetic — it would need a starting curiosity above 1.05 out of
#: a score that cannot exceed 1 — and :data:`KINDS` would have a move in it that
#: no organism could ever play. That this number is *declared* rather than
#: measured is the honest weak point of the file, and it is the one field of the
#: four a user should set for themselves.
LOOK_COST = 0.5
#: Samples either side of the anomaly a probe asks for: "a hundred steps before
#: the jump and a hundred after", in one number.
PROBE_CONTEXT = 96

KIND_CONTEXT = "context"
KIND_NEXT = "next"
KIND_COMPARE = "compare"
KINDS = (KIND_CONTEXT, KIND_NEXT, KIND_COMPARE)

#: Perturbations of the survey the certainty of an answer is measured against.
#: Five, because the question is "does the same stream win when the evidence is
#: nudged", and five draws already resolve ``1/5 = 0.2`` of stability — finer
#: than the threshold anyone would act on.
PERTURBATION_DRAWS = 5
#: Size of one nudge, as a fraction of each feature's own standard deviation
#: inside the survey. Small on purpose: this measures whether the *answer* is
#: stable, not whether it survives a different world. Five percent moves the
#: structure of an honest stream by less than the noise in its own estimate, and
#: moves a coin-flip ranking by all of it.
PERTURBATION_JITTER = 0.05
#: UCB1 exploration constant. One, in the units the table is fed: rewards are
#: credit rates in ``[-1, 1]``, so a bonus of one is "what one completely unknown
#: job is worth" — enough to buy a lesson, not enough to buy a habit.
UCB_C = 1.0
#: Plays an arm that has never been played is credited with, for the bonus only.
#: Not infinity: a table that scores an unseen arm ``+∞`` cannot be argued with,
#: and this one has to be arguable — the worker refuses a lesson when it cannot
#: afford it, and the notebook prints the score it refused at.
UNSEEN_PLAYS = 0.5



def lindley_gain(learnable: float, noise: float, looks: float = 0.0) -> float:
    """``½ ln(1 + τ²/σ²)``: the textbook expected information gain, in nats.

    Kept as the reference the metric is *not*. With ``τ²`` the posterior
    uncertainty of a stream's mean after ``looks`` observations of noise ``σ²``
    (``τ² = learnable / (1 + looks)``), the number is the same for every stream a
    body has looked at the same number of times — which is exactly why
    :class:`Curiosity` measures the two factors above instead of using it
    directly. The notebook prints both, side by side, so the difference is
    visible and not just claimed.
    """
    tau = max(float(learnable), 0.0) / (1.0 + max(float(looks), 0.0))
    sigma = max(float(noise), 1e-12)
    return float(0.5 * math.log1p(tau / sigma))


def structure_of(reducible: float) -> float:
    """Nats of structure implied by an ``R²``: ``-ln(1-R²)``, floored at 0."""
    r = float(min(max(reducible, 0.0), 1.0 - 1e-9))
    return float(-math.log1p(-r))


def reducibility(
    rows: Sequence[Sequence[float]], alpha: float = PROBE_ALPHA
) -> Tuple[float, int, float]:
    """How much of a stream its own past explains, and at what lag.

    For every lag ``k`` in ``1 .. T-3`` the stream's windows are regressed on
    themselves ``k`` windows earlier — one slope and one intercept per feature,
    pooled into a single :math:`R^2` — and a lag is believed only if the result is
    unlikely under the null hypothesis of no self-similarity. For one regressor
    that null distribution is exact:

    .. math:: P(R^2 > r) = (1-r)^{(n-2)/2}

    so the gate is a computation and not a convention. Without it, the maximum
    over five lags of an honest :math:`1/n` looks like structure, and a body that
    believes it spends its life on noise — measured on the offline sensorium:
    white noise scored 0.23 to 0.41 before the gate and 0.00 after.

    "Pooled" is meant literally: the residual sum of squares is accumulated over
    all :data:`~agent.physics.sensorium.FEATURES` at once and divided by the
    accumulated total sum of squares. Pooling rather than averaging per-feature
    :math:`R^2` is what stops a feature that is constant in the evidence
    (``stillness`` on a continuous feed) from contributing a spurious 1.0.

    Before pooling, each feature is divided by its spread *across these windows*,
    and a feature with no spread at all is dropped. The posture already centres
    and scales, but its scale is a median absolute deviation over the whole
    infancy, and a feature that is almost always equal to its median gets a tiny
    divisor and then a handful of enormous z-scores — the pooled :math:`R^2` is
    then one feature's :math:`R^2` wearing a disguise. Dividing again by the
    survey's own spread is what makes the answer "does this stream repeat"
    instead of "did the wildest feature happen to repeat".

    Args:
        rows: standardised feature vectors, in the order they were heard.
        alpha: family-wise error rate over however many lags are tried. A *bound*,
            not an operating point: measured on two hundred pure-noise surveys the
            procedure fires zero times at every ``alpha`` from 0.5 down to 0.01,
            because with three pairs left at a short lag the tail is steep — firing
            at ``alpha = 0.5`` there would need :math:`R^2 \\ge 0.9994`. The gate
            errs only towards silence, and the price is sensitivity: a noisy rhythm
            is invisible to it, which the module docstring measures.

    Returns:
        ``(R², lag, tail)`` — the winning lag's pooled :math:`R²`, the lag itself
        (0 when nothing was significant) and the tail probability that beat
        ``alpha / lags``. ``(0.0, 0, 1.0)`` when there is little to test.

    The *smallest* lag comes back when several are equally predictive, which they
    are whenever the repeat has a period: a stream that repeats every four windows
    also repeats every eight, and "every four" is the answer to the question the
    probe is going to ask.
    """
    matrix = np.clip(np.asarray(rows, dtype=float), -Z_CLIP, Z_CLIP)
    if matrix.ndim != 2 or matrix.shape[0] < CURIOUS_MIN_WINDOWS:
        return 0.0, 0, 1.0
    spread = matrix.std(axis=0)
    live = spread > 1e-12
    if not bool(live.any()):
        return 0.0, 0, 1.0
    matrix = matrix[:, live] / spread[live]
    windows, dims = matrix.shape
    lags = list(range(1, windows - 2))
    if not lags:
        return 0.0, 0, 1.0
    level = float(alpha) / float(len(lags))
    hits: List[Tuple[float, int, float]] = []
    for lag in lags:
        left, right = matrix[:-lag], matrix[lag:]
        pairs = left.shape[0]
        if pairs < 3:
            continue
        sse = 0.0
        sst = 0.0
        for j in range(dims):
            x, y = left[:, j], right[:, j]
            variance = float(y.var())
            sst += variance
            if variance <= 1e-15 or float(x.var()) <= 1e-15:
                sse += variance  # an undefined slope predicts the mean, and loses
                continue
            slope = float(np.cov(x, y, ddof=0)[0, 1] / x.var())
            residual = y - (y.mean() + slope * (x - x.mean()))
            sse += float(np.mean(residual ** 2))
        if sst <= 1e-15:
            continue
        r2 = float(min(max(1.0 - sse / sst, 0.0), 1.0))
        # ``pairs``, not ``pairs * dims``: the features are correlated, so counting
        # each one as an independent observation would turn seven opinions into
        # seven hundred and make the gate believe anything.
        tail = float((1.0 - min(r2, 1.0 - 1e-12)) ** ((pairs - 2) / 2.0))
        if tail <= level:
            hits.append((r2, int(lag), tail))
    if not hits:
        return 0.0, 0, 1.0
    top = max(hit[0] for hit in hits)
    for hit in sorted(hits, key=lambda item: item[1]):
        if hit[0] >= LAG_TIE * top:
            return hit
    return max(hits, key=lambda item: item[0])


def score(
    movement: float,
    reference: float,
    structure: float,
    reality: float,
    looks: int = 0,
) -> float:
    """The product, in ``[0, 1]``: the one place the four factors meet.

    ``movement`` enters as ``2V/(V+V_ref)`` clamped at 1, because ``V_ref`` is the
    survey's median: a stream that moves at the body's habitual rate is then 1.0
    and the factor acts as a **gate** on "is there anything here to learn at all"
    rather than as a fifth opinion. That is deliberate. In this frame any two
    streams that move at all have almost the same variance — noise and structure
    are equally *loud*, they differ only in how much of the loudness is
    predictable — so a factor that tried to rank them by variance would be
    ranking noise highest. The ranking is the structure's job; the gate's job is
    to zero out the frozen feed, which is the one thing a variance knows and
    structure does not.

    Separated from :class:`Curiosity` so the notebook (and a reader) can sweep any
    one of the factors and watch the answer move, which is the only way to know
    that a score has the shape it claims.
    """
    spread = max(float(movement), 0.0)
    typical = max(float(reference), 1e-9)
    moving = min(1.0, 2.0 * spread / (spread + typical)) if spread > 0.0 else 0.0
    gain = 1.0 - math.exp(-max(float(structure), 0.0) / GAIN_SCALE)
    gate = min(max(float(reality), 0.0), 1.0)
    fade = 1.0 / (1.0 + LOOK_COST * max(int(looks), 0))
    return float(max(0.0, min(1.0, gate * moving * gain * fade)))


@dataclass(frozen=True)
class Trait:
    """One stream's standing, in every number the decision is made from.

    Nothing here is a summary of a summary: each field is one of the arms of the
    product in the module docstring, so a disagreement about the score can be
    traced to the factor that caused it instead of to "the model".
    """

    stream: str
    source: str
    domain: str = ""
    origin: str = ""
    fingerprint: str = ""
    window: int = DEFAULT_WINDOW
    stride: int = 0
    fixations: int = 0
    looks: int = 0
    movement: float = 0.0
    reference: float = 0.0
    reducible: float = 0.0
    lag: int = 0
    evidence: float = 1.0
    structure: float = 0.0
    reality: float = 0.0
    curiosity: float = 0.0
    anchor: int = 0
    note: str = ""

    @property
    def blind(self) -> bool:
        """True when the stream was too short for the survey to say anything."""
        return self.fixations < CURIOUS_MIN_WINDOWS

    @property
    def anomaly(self) -> bool:
        """True when the stream is worth announcing, not merely worth a look."""
        return self.curiosity >= ANOMALY_THRESHOLD

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stream": self.stream,
            "source": self.source,
            "domain": self.domain,
            "origin": self.origin,
            "fixations": int(self.fixations),
            "looks": int(self.looks),
            "window": int(self.window),
            "stride": int(self.stride),
            "movement": round(float(self.movement), 6),
            "reference": round(float(self.reference), 6),
            "reducible": round(float(self.reducible), 6),
            "lag": int(self.lag),
            "evidence": float(f"{self.evidence:.3e}"),
            "structure": round(float(self.structure), 6),
            "reality": round(float(self.reality), 6),
            "curiosity": round(float(self.curiosity), 6),
            "anchor": int(self.anchor),
            "note": self.note,
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.stream:<34} c={self.curiosity:.3f} R={self.reality:.3f} "
            f"move={self.movement:.2f} r2={self.reducible:.2f}@{self.lag} "
            f"lam={self.structure:.2f} looks={self.looks}"
            + (f"  [{self.note}]" if self.note else "")
        )


@dataclass(frozen=True)
class Probe:
    """A question asked of the sensorium, with the reason it was asked.

    A probe is an *action*, not an inference: the organism cannot make the feed
    say anything, it can only choose where to point and how much to take in. The
    three kinds in :data:`KINDS` are therefore the three moves available, and
    :attr:`kind` records which one this was so the journal can be read back.
    """

    stream: str
    kind: str
    curiosity: float
    anchor: int = 0
    context: int = PROBE_CONTEXT
    lag: int = 0
    against: str = ""
    step: int = 0
    origin: str = ""

    def question(self) -> str:
        """The question in words, for the journal and the notebook."""
        if self.kind == KIND_CONTEXT:
            return (
                f"{self.stream}: what happened {self.context} samples either side "
                f"of sample {self.anchor}?"
            )
        if self.kind == KIND_NEXT:
            return f"{self.stream}: what comes next, after sample {self.anchor}?"
        return (
            f"{self.stream}: is this {self.stream} at all, next to "
            f"{self.against or 'the least interesting stream'}?"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stream": self.stream,
            "kind": self.kind,
            "curiosity": round(float(self.curiosity), 6),
            "anchor": int(self.anchor),
            "context": int(self.context),
            "lag": int(self.lag),
            "against": self.against,
            "step": int(self.step),
            "question": self.question(),
        }


@dataclass(frozen=True)
class Certainty:
    """How far the survey trusts its own answer, and what would talk it out of it.

    A ranking is not a belief. ``Trait.curiosity`` says which stream is most worth
    a look; this says **how much of that is arithmetic and how much is the draw**
    — the same distinction that separates a detector from a decoration. Two
    questions are asked, and both are answered by measurement:

    ``margin`` / ``significance``
        How much structure the winner has that the runner-up does not (nats, from
        :attr:`Trait.structure`), and how unlikely that winner's ``R²`` is under
        the file's own null. A margin of twenty nats is not a preference, it is a
        fact about the feed; a margin of zero is the survey saying "nothing here
        repeats", which is a perfectly good answer and is stored as
        ``suspect = None`` — *no answer*, not *the least bad answer*.

    ``stability`` / ``uncertainty``
        The survey's rows are nudged by :data:`PERTURBATION_JITTER` of their own
        standard deviation, :data:`PERTURBATION_DRAWS` times, and the same
        arithmetic is re-run. ``stability`` is the share of those draws that name
        the same stream; ``uncertainty`` is its complement. This is epistemic
        uncertainty in the only form this file can honestly produce it: not a
        variance of the answer, but **the probability that the answer changes**.
        For a real repeat it is 0 (twenty nats do not move for five percent of the
        noise); for a survey of three honest feeds it is about 2/3, because the
        winner is then whichever stream happened to win the tie. ``spread`` is the
        same question asked in the currency of the answer rather than of the
        ranking: the standard deviation of the margin itself across those draws.

    ``resonance``
        The spread of the mirror's own ``internal_resonance`` over the suspect's
        fixations — the second arm the question asked for, and the reason it asked
        for the mirror rather than only for ``R²``. It is carried, not used to
        decide: two arms that measure the same thing should be reported side by
        side and not multiplied.

    ``probability``
        The one number a worker can spend: ``1 - e^(-margin/GAIN_SCALE)`` — the
        file's own saturation map, applied to the *gap* instead of to the level —
        floored at :attr:`chance` when the winner is not significant, and capped
        just below one. It is **declared**: nothing in the data says the map is
        right, and the notebook prints the calibration (what the agent said
        against what happened) so the declaration can be judged instead of
        believed.
    """

    suspect: Optional[str] = None
    margin: float = 0.0
    significance: float = 0.0
    stability: float = 0.0
    uncertainty: float = 1.0
    resonance: float = 0.0
    chance: float = 0.0
    draws: int = 0
    spread: float = 0.0

    @property
    def decisive(self) -> bool:
        """True when the survey names a stream at all (not *which one is sure*)."""
        return self.suspect is not None

    @property
    def probability(self) -> float:
        """``P(this stream is the odd one out)``, as far as structure can say."""
        if self.suspect is None:
            return float(self.chance)
        mapped = 1.0 - math.exp(-max(self.margin, 0.0) / GAIN_SCALE)
        return float(min(1.0 - 1e-9, max(self.chance, mapped)))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "suspect": self.suspect,
            "margin": round(float(self.margin), 6),
            "significance": float(f"{self.significance:.3e}"),
            "stability": round(float(self.stability), 4),
            "uncertainty": round(float(self.uncertainty), 4),
            "resonance": round(float(self.resonance), 6),
            "chance": round(float(self.chance), 6),
            "draws": int(self.draws),
            "probability": round(float(self.probability), 6),
        }


def perturb(
    rows: Sequence[np.ndarray],
    jitter: float = PERTURBATION_JITTER,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """The same windows, nudged: Gaussian noise at ``jitter`` of each feature's σ.

    Why a nudge and not a bootstrap: a resample changes *which windows exist*, so
    it measures sampling error — how surprised a different twenty-four windows
    would be. The question :meth:`Curiosity.certainty` asks is narrower and
    harder: given these very windows, would a five-percent change in their values
    change the answer? A detector that survives a re-draw but not a rounding is a
    detector that has memorised its sample.

    Returns one matrix, not a list of them: the arithmetic downstream
    (:func:`reducibility`) takes rows, and a wrapper list was measured to be a
    silent zero — ``np.asarray`` of a one-element list is three-dimensional, and a
    three-dimensional survey is refused as "little to test".
    """
    stream = rng if rng is not None else np.random.default_rng(0)
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return matrix.copy()
    # A feature that does not move in this survey is left exactly where it is.
    # Nudging it by a fraction of zero is zero, and nudging it by an absolute
    # amount would *invent* movement: after ``reducibility`` re-normalises by the
    # survey's own spread, that invented movement arrives with unit variance and
    # the pooled R² of a perfect repeat collapses to zero. Measured, on a cut of
    # the digits recording where ``stillness`` is constant: R² = 1.00 → 0.00.
    sigma = matrix.std(axis=0)
    sigma[sigma <= 0.0] = 0.0
    return matrix + stream.normal(0.0, float(jitter) * sigma, matrix.shape)


@dataclass
class Curiosity:
    """The organism's appetite: a survey of what it hears, and the questions it buys.

    One posture, one survey budget, one ranking. The engine is *rebuilt* from
    evidence rather than remembered from scores: :meth:`observe` stores the
    standardised feature rows and the reality of every stream it has heard, and
    every number in :class:`Trait` is recomputed from those rows on demand. A
    score that cannot be recomputed cannot be checked, and a decision made on an
    uncheckable number is not a decision.

    Args:
        posture: the mirror's posture — the frame the features are standardised
            in, and the model whose likelihoods become the reality gate.
        prior: ``P(world)`` used when the caller supplies no moments. The same
            constant the self model uses, so a stream is gated by the same
            question the mirror asks.
        threshold: curiosity at which :meth:`propose` returns a question.
        anomaly: curiosity at which :attr:`Trait.anomaly` is true.
        context: samples either side of an anchor a ``context`` probe asks for.
        max_probes: questions per stream, so curiosity cannot become obsession.
    """

    posture: Posture
    prior: float = WORLD_PRIOR
    threshold: float = PROBE_THRESHOLD
    anomaly: float = ANOMALY_THRESHOLD
    context: int = PROBE_CONTEXT
    max_probes: int = MAX_PROBES
    reference: float = 0.0
    step: int = 0
    _rows: Dict[str, List[np.ndarray]] = field(default_factory=dict, init=False)
    _reality: Dict[str, List[float]] = field(default_factory=dict, init=False)
    _errors: Dict[str, List[float]] = field(default_factory=dict, init=False)
    _info: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    _looks: Dict[str, int] = field(default_factory=dict, init=False)
    _resonance: Dict[str, List[float]] = field(default_factory=dict, init=False)
    _traits: Dict[str, Trait] = field(default_factory=dict, init=False)
    questions: List[Probe] = field(default_factory=list, init=False)

    # -- hearing ------------------------------------------------------------ #

    def observe(
        self,
        episode: Any,
        moments: Optional[Sequence[Any]] = None,
        key: Optional[str] = None,
        survey: bool = True,
    ) -> Trait:
        """Take in one episode, and answer what it did to that stream's standing.

        Args:
            episode: an :class:`~agent.physics.sensorium.Episode`.
            moments: the mirror's :class:`~agent.physics.self_model.Moment` per
                fixation, if the caller has them. Their ``reality_score`` carries
                the ledger — the organism's knowledge that it emitted these bytes
                itself — which no recomputation from the posture can recover.
                Without them reality is the likelihood arm alone, and the module
                says so in the trait's note.
            key: the stream's name; defaults to ``episode.source``.
            survey: ``False`` for a probe's episode. A probe is chosen *because*
                of an anomaly, so its windows are not independent evidence about
                the stream — counting them would double-count the very repeat
                that made the stream interesting. Probes therefore move
                ``reality`` and ``looks``, never the structure.
        """
        name = key or episode.source
        rows = [self.posture.features(reading) for reading in episode.readings]
        if not rows:
            return self._score(name)
        judged = (
            [float(moment.reality_score) for moment in moments]
            if moments is not None
            else [self._reality_of(row) for row in rows]
        )
        errors = [float(self.posture.world.distance(row)) for row in rows]
        starts = [int(getattr(reading, "start", 0) or 0) for reading in episode.readings]
        info = self._info.setdefault(
            name,
            {
                "source": episode.source,
                "domain": episode.domain,
                "origin": episode.origin,
                "fingerprint": episode.fingerprint,
                "window": self._window_of(episode),
                "starts": [],
            },
        )
        info.update(
            source=episode.source,
            domain=episode.domain,
            origin=episode.origin,
            fingerprint=episode.fingerprint,
        )
        self._looks.setdefault(name, 0)
        self._reality.setdefault(name, []).extend(judged)
        self._errors.setdefault(name, []).extend(errors)
        if moments is not None:
            self._resonance.setdefault(name, []).extend(
                float(moment.resonance) for moment in moments
            )
        if survey:
            self._rows.setdefault(name, []).extend(rows)
            info["starts"] = starts
            info["window"] = self._window_of(episode)
            info["ledger"] = moments is not None
        else:
            info["probed"] = int(info.get("probed", 0)) + len(rows)
        self.reference = self._reference_of()
        self._traits.clear()
        return self._score(name)

    # -- the numbers -------------------------------------------------------- #

    def _window_of(self, episode: Any) -> int:
        """Samples per fixation, as the episode reports it (``hours`` is the core's)."""
        readings = getattr(episode, "readings", ()) or ()
        if not readings:
            return DEFAULT_WINDOW
        hours = getattr(readings[0].observation, "hours", None)
        try:
            value = int(round(float(hours)))
        except (TypeError, ValueError):
            return DEFAULT_WINDOW
        return value if value > 0 else DEFAULT_WINDOW

    def _reality_of(self, row: np.ndarray) -> float:
        """Reality from the posture alone: the likelihood arm, no ledger."""
        return reality_score(self.posture.log_ratio(row), 0.0, self.prior)

    def movement_of(self, stream: str) -> float:
        """Mean per-feature variance of a stream's survey windows, in the frame.

        Unit-free by construction (the frame is the posture's standardiser) and
        therefore comparable between streams but not between organisms. The
        mean over features rather than the sum keeps it the size of "one
        feature's variance", which is what makes ``V_ref`` a meaningful scale.
        """
        rows = self._rows.get(stream, [])
        if not rows:
            return 0.0
        matrix = np.clip(np.asarray(rows, dtype=float), -Z_CLIP, Z_CLIP)
        return float(np.mean(matrix.var(axis=0)))

    def resonance_of(self, stream: str) -> float:
        """Spread of the mirror's ``internal_resonance`` over a stream's fixations.

        The second arm of the mandated uncertainty estimate, and the only one that
        asks the *mirror* rather than the arithmetic: a stream the body's own
        resonator cannot settle on is one its self model is unsure about, whatever
        its ``R²`` says. Stored from the moments :meth:`observe` was given, so it
        costs nothing to carry and cannot be recomputed from the posture alone —
        like the ledger, it is knowledge of a *heartbeat*, not of a stream.
        """
        values = self._resonance.get(stream, [])
        if len(values) < 2:
            return 0.0
        return float(np.std(np.asarray(values, dtype=float)))

    def certainty(
        self,
        *,
        draws: int = PERTURBATION_DRAWS,
        jitter: float = PERTURBATION_JITTER,
        seed: int = 0,
    ) -> Certainty:
        """How far this survey trusts its own ranking, measured against nudges.

        The honest version of "am I sure?" for a body whose model is a ranking: it
        is not asked how confident the *model* is, it is asked how much the answer
        moves when the evidence is perturbed by an amount smaller than the noise in
        the evidence itself. A detector that answers the same way five times out of
        five under :data:`PERTURBATION_JITTER` has a margin worth spending money
        on; one that changes its mind has a tie, and a tie must be reported as
        ``suspect = None`` rather than as the alphabetically first stream.

        ``seed`` makes the draw reproducible: two runs of the same survey must
        report the same confidence, or the notebook cannot be re-run and the
        refusal in the log cannot be audited.
        """
        traits = self.traits()
        if not traits:
            return Certainty(draws=0)
        names = [trait.stream for trait in traits]
        top = traits[0]
        second = traits[1] if len(traits) > 1 else None
        margin = float(top.structure) - (float(second.structure) if second else 0.0)
        # ``evidence`` is the p-tail of the winner's R², so significance is 1 - p.
        # Without it a margin over *nothing* would be an answer: the largest of
        # twenty-one honest draws is always the largest of twenty-one.
        significance = 1.0 - float(top.evidence)
        suspect = top.stream if (margin > 0.0 and top.evidence <= PROBE_ALPHA) else None
        chance = 1.0 / max(len(names), 1)
        usable = [name for name in names if len(self._rows.get(name, ())) >= 2]
        rng = np.random.default_rng(int(seed))
        hits = 0
        margins: List[float] = []
        for _ in range(max(int(draws), 0)):
            # ``perturb`` returns rows, not a list of row-sets: a one-element list
            # would arrive at ``reducibility`` as a three-dimensional array and be
            # refused, which is how a perfectly stable repeat once measured as
            # zero stability.
            levels = {
                name: structure_of(
                    reducibility(perturb(self._rows[name], jitter, rng))[0]
                )
                for name in usable
            }
            ranked = sorted(levels.values(), reverse=True)
            margins.append(float(ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)))
            if suspect is not None:
                best = max(levels, key=lambda name: (levels[name], name))
                hits += int(best == suspect)
        stability = (
            float(hits) / len(margins) if (suspect is not None and margins) else 0.0
        )
        spread = float(np.std(margins)) if len(margins) > 1 else 0.0
        if suspect is not None:
            resonance = self.resonance_of(suspect)
        else:
            spreads = [self.resonance_of(name) for name in names]
            resonance = float(np.mean(spreads)) if spreads else 0.0
        return Certainty(
            suspect=suspect,
            margin=margin,
            significance=significance,
            stability=stability,
            uncertainty=1.0 - stability,
            resonance=resonance,
            chance=chance,
            draws=len(margins),
            spread=spread,
        )

    def _reference_of(self) -> float:
        """``V_ref``: the median movement of the streams in this survey.

        Measured, not declared — "what this body is used to" is a property of the
        survey, and a constant here would quietly fix the scale of curiosity to
        one organism's sensorium. Falls back to 1.0 (one standardised feature's
        variance) only when nothing in the survey moves at all.
        """
        eligible = [
            self.movement_of(name)
            for name, rows in self._rows.items()
            if len(rows) >= CURIOUS_MIN_WINDOWS
        ] or [self.movement_of(name) for name in self._rows]
        moving = [value for value in eligible if value > 0.0]
        return float(np.median(moving)) if moving else 1.0

    def _score(self, stream: str) -> Trait:
        """Recompute one stream's trait from the stored rows. Pure; no side effects."""
        cached = self._traits.get(stream)
        if cached is not None:
            return cached
        info = self._info.get(stream, {})
        rows = self._rows.get(stream, [])
        reality = self._reality.get(stream, [])
        errors = self._errors.get(stream, [])
        reducible, lag, tail = reducibility(rows)
        nats = structure_of(reducible)
        gate = float(np.mean(reality)) if reality else 0.0
        movement = self.movement_of(stream)
        looks = int(self._looks.get(stream, 0))
        curiosity = score(movement, self.reference, nats, gate, looks)
        window = int(info.get("window") or DEFAULT_WINDOW)
        starts = list(info.get("starts") or [])
        stride = int(starts[1] - starts[0]) if len(starts) > 1 else window
        anchor = 0
        if starts:
            pool = np.asarray(errors[: len(starts)], dtype=float)
            anchor = int(starts[int(np.argmax(pool)) if pool.size else 0]) + window // 2
        note = ""
        if len(rows) < CURIOUS_MIN_WINDOWS:
            note = f"{len(rows)} fixations: too short to judge"
        elif lag and nats >= 1.0:
            note = f"repeats every {lag} fixations ({lag * stride} samples)"
        elif movement <= 0.0:
            note = "frozen: nothing moves"
        if info.get("ledger") is False:
            note = (note + "; " if note else "") + "reality without the ledger"
        trait = Trait(
            stream=stream,
            source=str(info.get("source") or stream),
            domain=str(info.get("domain") or ""),
            origin=str(info.get("origin") or ""),
            fingerprint=str(info.get("fingerprint") or ""),
            window=window,
            stride=stride,
            fixations=len(rows),
            looks=looks,
            movement=movement,
            reference=self.reference,
            reducible=reducible,
            lag=lag,
            evidence=tail,
            structure=nats,
            reality=gate,
            curiosity=curiosity,
            anchor=anchor,
            note=note,
        )
        self._traits[stream] = trait
        return trait

    # -- the ranking -------------------------------------------------------- #

    def traits(self) -> List[Trait]:
        """Every stream's trait, most interesting first."""
        found = [self._score(name) for name in self._rows]
        return sorted(found, key=lambda t: (-t.curiosity, t.stream))

    def ranking(self) -> List[Dict[str, Any]]:
        """The same list as plain rows: what a report wants, and what a table prints."""
        return [trait.as_dict() for trait in self.traits()]

    def trait(self, stream: str) -> Trait:
        return self._score(stream)

    def curious(self) -> List[Trait]:
        """Streams that clear :attr:`threshold` and have questions left in them."""
        return [
            trait
            for trait in self.traits()
            if trait.curiosity >= self.threshold and trait.looks < self.max_probes
        ]

    def survey(self, episodes: Sequence[Any], moments: Optional[Sequence[Any]] = None) -> List[Trait]:
        """Take in a whole round of hearing, then answer with the ranking.

        Args:
            episodes: one :class:`~agent.physics.sensorium.Episode` per stream.
            moments: either a list of per-fixation moments, parallel to
                ``episodes``, or ``None``. Passing them is better: only the
                ledger knows that the organism emitted some of these bytes
                itself, and a forger that re-serves the organism its own output
                is otherwise indistinguishable from the world.
        """
        for index, episode in enumerate(episodes):
            judged = None
            if moments is not None:
                judged = moments[index]
            self.observe(episode, moments=judged)
        return self.traits()

    # -- the questions ------------------------------------------------------ #

    def propose(self, step: Optional[int] = None) -> Optional[Probe]:
        """The next question worth asking, or ``None`` when nothing is worth asking.

        The kind is a function of how often this stream has already been asked, so
        the three moves in :data:`KINDS` are spent in the order that information
        is worth: *where* (context), then *what next* (next), then *what is it
        really* (compare). Returning ``None`` is the normal case and must stay
        cheap — a body that always has a question is not curious, it is
        compulsive.
        """
        pending = self.curious()
        if not pending:
            return None
        trait = pending[0]
        against = ""
        if trait.looks == 0:
            kind = KIND_CONTEXT
        elif trait.looks == 1:
            kind = KIND_NEXT
        else:
            others = [t for t in self.traits() if t.stream != trait.stream]
            if others:
                kind = KIND_COMPARE
                against = min(others, key=lambda t: (t.curiosity, t.stream)).stream
            else:
                kind = KIND_NEXT
        self.step = int(step) if step is not None else self.step + 1
        return Probe(
            stream=trait.stream,
            kind=kind,
            curiosity=trait.curiosity,
            anchor=trait.anchor,
            context=self.context,
            lag=trait.lag,
            against=against,
            step=self.step,
            origin=trait.origin,
        )

    def answered(
        self,
        probe: Probe,
        episode: Any = None,
        moments: Optional[Sequence[Any]] = None,
    ) -> Trait:
        """Record that a question was asked and, if it was, what came back.

        The answer is deliberately *not* added to the stream's survey: it was
        drawn where the anomaly was, so it is a biased sample of the stream, and
        the structure it would add is the structure whose significance is already
        in question. What it does add is reality — the mirror judges the answer
        like any other window — and one to ``looks``, which is what makes
        curiosity fade: ``1/(1 + LOOK_COST·looks)`` hands the second question
        two-thirds of the stream's worth and the third one half, so a body sticks
        with an anomaly for a while and then moves on. That decay is declared, not
        measured, and it is the one number in this file a user should set for
        themselves.
        """
        self._looks[probe.stream] = int(self._looks.get(probe.stream, 0)) + 1
        self.questions.append(probe)
        self._traits.clear()
        if episode is not None:
            self.observe(episode, moments=moments, key=probe.stream, survey=False)
        return self._score(probe.stream)

    # -- the report --------------------------------------------------------- #

    def summary(self) -> Dict[str, Any]:
        """Everything the report prints about curiosity, in one dictionary."""
        return {
            "streams": len(self._rows),
            "reference": round(float(self.reference), 6),
            "threshold": float(self.threshold),
            "anomaly": float(self.anomaly),
            "probes_asked": len(self.questions),
            "questions": [probe.as_dict() for probe in self.questions],
            "traits": self.ranking(),
        }

    def reset(self) -> None:
        """Forget every stream; keep the posture, the threshold and the budget."""
        self._rows.clear()
        self._reality.clear()
        self._errors.clear()
        self._info.clear()
        self._looks.clear()
        self._resonance.clear()
        self._traits.clear()
        self.questions.clear()
        self.reference = 0.0
        self.step = 0

    def __len__(self) -> int:
        return len(self._rows)


# --------------------------------------------------------------------------- #
# The attention table: which kind of job is still worth trying.
# --------------------------------------------------------------------------- #

@dataclass
class UCB:
    """A table of past rewards that stays optimistic about what it has not tried.

    The appetite above decides *which stream* to look at; this decides *which kind
    of question* to keep bidding for, and the two are not the same. A body that
    only does what it has already mastered stops growing; a body that only tries
    new things starves. The classical answer is UCB1 (Auer, Cesa-Bianchi, Fischer,
    2002)::

        score(a) = mean(a) + c · sqrt(ln N / n_a)

    ``mean`` is what the arm has paid so far — exploitation — and the root is how
    little is known about it — exploration. The bonus shrinks as ``1/sqrt(n)``, so
    a kind that keeps disappointing is eventually left alone *by arithmetic*, not
    by a course of action someone wrote down.

    Why here and not in the worker: this is the same appetite as
    :class:`Curiosity`, one level up. Curiosity asks "how much would one more look
    at this stream buy?"; the table asks "how much would one more job of this kind
    buy?" — and the second question is the first one with the credits attached, so
    it belongs in the file that already owns the notion of "worth".

    Rewards are fed as **credit rates in ``[-1, 1]``** (``credits / reward`` of the
    contract, clipped), so ``c`` is on the scale of "one job" and the score can be
    compared with a win probability without a conversion table. An arm that has
    never been played is credited with :data:`UNSEEN_PLAYS` at the running mean:
    optimistic, finite, and therefore arguable — a table that returns ``+∞`` for
    an unseen arm forces the exploration and cannot be refused on a bad day.
    """

    c: float = UCB_C
    plays: Dict[str, int] = field(default_factory=dict)
    totals: Dict[str, float] = field(default_factory=dict)

    def observe(self, arm: str, reward: float = 0.0) -> None:
        """Record one played job: what kind it was, and what it paid."""
        name = str(arm)
        self.plays[name] = int(self.plays.get(name, 0)) + 1
        self.totals[name] = float(self.totals.get(name, 0.0)) + float(reward)

    def count(self, arm: str) -> int:
        return int(self.plays.get(str(arm), 0))

    def mean(self, arm: str, default: float = 0.0) -> float:
        """What the arm has paid per job; ``default`` when never played."""
        plays = self.count(arm)
        if not plays:
            return float(default)
        return float(self.totals.get(str(arm), 0.0)) / plays

    def score(self, arm: str, *, default: float = 0.0) -> float:
        """``mean + c·sqrt(ln N / n)``: the optimism this kind is still owed."""
        plays = self.count(arm)
        total = max(sum(self.plays.values()), 1)
        prior = self.mean(arm, default)
        virtual = float(max(plays, UNSEEN_PLAYS))
        return float(prior + float(self.c) * math.sqrt(math.log(total + 1.0) / virtual))

    def best(self, arms: Sequence[str], *, default: float = 0.0) -> Optional[str]:
        """The arm with the highest score; ties broken by name for reproducibility."""
        names = [str(arm) for arm in arms]
        if not names:
            return None
        return max(names, key=lambda name: (self.score(name, default=default), name))

    def table(self, arms: Optional[Sequence[str]] = None, *, default: float = 0.0):
        """The table as rows, most optimistic first — what a notebook prints."""
        names = [str(arm) for arm in arms] if arms is not None else sorted(
            set(self.plays) | set(self.totals)
        )
        rows = [
            {
                "arm": name,
                "plays": self.count(name),
                "mean": round(self.mean(name, default), 6),
                "score": round(self.score(name, default=default), 6),
            }
            for name in names
        ]
        return sorted(rows, key=lambda row: (-row["score"], row["arm"]))

    def as_dict(self) -> Dict[str, Any]:
        return {"c": float(self.c), "arms": self.table()}


__all__ = [
    "ANOMALY_THRESHOLD",
    "CURIOUS_MIN_WINDOWS",
    "Certainty",
    "Curiosity",
    "DEFAULT_WINDOW",
    "GAIN_SCALE",
    "KINDS",
    "KIND_COMPARE",
    "KIND_CONTEXT",
    "KIND_NEXT",
    "LAG_TIE",
    "LOOK_COST",
    "MAX_PROBES",
    "PERTURBATION_DRAWS",
    "PERTURBATION_JITTER",
    "PROBE_ALPHA",
    "PROBE_CONTEXT",
    "PROBE_THRESHOLD",
    "Probe",
    "SURVEY_FIXATIONS",
    "Trait",
    "UCB",
    "UCB_C",
    "UNSEEN_PLAYS",
    "Z_CLIP",
    "lindley_gain",
    "perturb",
    "reducibility",
    "score",
    "structure_of",
]
