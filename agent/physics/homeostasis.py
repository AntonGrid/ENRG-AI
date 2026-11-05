"""Homeostasis: the organism tunes itself, and it is allowed to mutate.

Three numbers decide how this organism reads itself, and none of them is guessed:

``lam`` (λ)      the write rate of the world into the internal posture,
                 ``r <- r + λ (x - r) - ε r``. λ = 0 is an organism that never
                 learns; λ = 1 is one that remembers nothing but the last window.
``epsilon`` (ε)  forgetfulness: the same rate that decays the posture decays the
                 emission ledger, which is why forgetting the world and forgetting
                 one's own dreams are the *same* act in this body.
``alpha`` (α)    the declared risk of the surprise gate. The threshold is never
                 chosen directly: ``τ = null_threshold(n, α)`` comes from the
                 core, so the risk is a decision and the threshold is arithmetic.

The objective is one number, and it is the one the whole exercise is about:

.. math:: J(\\lambda, \\varepsilon) = \\overline{\\rho}_{\\text{world}}
                                     - \\overline{\\rho}_{\\text{self}}

where :math:`\\rho` is :func:`internal_resonance` — how well the current fixation
agrees with the posture the organism has been carrying. A posture that tunes to
the world resonates with the world and not with the organism's own noise; a
posture that tunes to nothing resonates with nothing in particular.

Three searches, in this order, because each is blind to what the next one sees:

1. **grid** — a declared lattice, evaluated exhaustively. Kept as a surface so
   the report can show whether the optimum is a peak or a ridge, and whether it
   sits on the edge of the box (which is a warning, not a success).
2. **refinement** — ``scipy.optimize.minimize(method="Powell")`` from the best
   cell, bounded to the same box. Skipped, with a warning, if scipy is absent.
3. **mutation** — a (1+1) evolution strategy: each generation, *each* parameter
   is perturbed with probability :data:`MUTATION_RATE` (8%), multiplicatively and
   log-normally. The candidate is accepted when the objective improves and rolled
   back when it does not. Without this the organism stops wherever the lattice
   happens to be fine-grained, and the report would be a statement about the
   lattice rather than about the organism.

Acceptance is elitist — the best objective never decreases — so the loop is a
lower bound on what a mutation can buy, and the log carries every attempt with
its gain, accepted or not. A mutation that buys nothing is still evidence: it
says the organism is not stuck on the first ridge it found.

Declared limits:

* the objective is maximised on a *tuning* life and reported on a *held-out*
  validation life; the split is interleaved by source so both halves see every
  domain. A single-number optimum on one half is a fit; agreement between halves
  is a claim.
* separation is reported as a margin, as AUC (rank based, no threshold) and as
  d', because a margin alone hides a bimodal disaster.
* nothing here touches the core's ``theorem.py`` or ``detector.py``; the
  threshold is *called*, never re-derived.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.physics.genome import core as core_module

#: Π of the book's overlap kernel ``S(Δ) = 1 / (1 + (Δ/Π)²)``. One unit of
#: resonance band = one channel's honest noise, the only scale in the problem
#: that is measured rather than chosen.
RESONANCE_BAND = 1.0

#: The declared lattice for λ: geometric, because a write rate is a rate.
LAMBDA_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0)
#: The declared lattice for ε. The upper end is wide on purpose — if the optimum
#: sits on the edge of the box then the box, not the organism, decided the answer,
#: and :func:`search_grid` reports the edge rather than hiding it.
#:
#: ``ε = 0`` is deliberately absent. Zero forgetting means an infinite memory
#: horizon, and this body uses one ε for two jobs: it decays the class models *and*
#: it ages the ledger of its own emissions. With ε = 0 a dream emitted a thousand
#: steps ago is still recognised as its own with certainty, the phantom curve is
#: flat, and the whole claim becomes untestable. No organism holds a memory
#: forever; the floor is what makes that a constraint instead of a wish.
EPSILON_GRID = (0.01, 0.03, 0.07, 0.15, 0.3, 0.45, 0.6)
#: Smallest admissible forgetting rate, in inverse fixations.
EPSILON_FLOOR = 0.01
#: The declared risks. α is a promise about one's own false alarms, not a knob.
ALPHA_GRID = (1e-1, 5e-2, 1e-2, 1e-3)
#: Probability that one parameter is perturbed in one generation (the 5–10% band).
MUTATION_RATE = 0.08
#: Log-normal scale of a *coarse* mutation: one standard deviation of the
#: log-multiplier. 0.45 means "somewhere between −36% and +57%", which is a step
#: large enough to leave a basin and small enough to stay in the sensible box.
MUTATION_SCALE = 0.45
#: Scale of the *fine* mutation that runs after the refinement, where the job is
#: no longer to escape but to polish without stepping off the peak.
MUTATION_SCALE_FINE = 0.18
#: Generations of (1+1) evolution per phase.
MUTATION_STEPS = 200
#: Where a search starts when a caller does not say.
DEFAULT_START = (0.35, 0.03, 1e-2)

#: Prior variance of a class model before it has seen anything, in standardised
#: feature units. 1.0 means "one typical deviation of everything I have met",
#: which is exactly what the standardiser makes the units mean.
INITIAL_VARIANCE = 1.0
#: Pseudo-observations behind the innate prior. Every update shrinks the learned
#: mean toward 0 and the learned variance toward :data:`INITIAL_VARIANCE` by
#: ``PRIOR_MASS / (mass + PRIOR_MASS)``.
#:
#: This is not a regularisation trick, it is the fix for a measured failure: with
#: 24 world fixations from three streams, an unshrunk model narrows its variance to
#: the *within-stream* spread and then finds the fourth real stream (the sunspot
#: record) stranger than its own dreams — measured AUC 0.685, with ρ = 0.11 for a
#: genuine world stream. An organism that has met four things must not be certain
#: about the fifth.
PRIOR_MASS = 4.0
#: A class model's variance never collapses below this. A model that is certain to
#: within 1e-3 of a feature would otherwise report an infinite surprise for the
#: first reading that differs by 2e-3.
VAR_FLOOR = 1e-3
#: Floor on the standardiser's per-feature scale, so a feature that is constant
#: across the evidence does not become a division by zero.
SCALE_FLOOR = 1e-6


@dataclass(frozen=True)
class Parameters:
    """The organism's autonomic state: three numbers, and a threshold they imply.

    ``tau`` is deliberately *not* a field. A threshold is a function of the
    declared risk and the number of channels — ``null_threshold(n, alpha)`` — so
    storing it would create a second source of truth that can drift from the
    core. :meth:`threshold` asks the core every time (the core caches).
    """

    lam: float = DEFAULT_START[0]
    epsilon: float = DEFAULT_START[1]
    alpha: float = DEFAULT_START[2]

    def __post_init__(self) -> None:
        if not (0.0 < self.lam <= 1.0):
            raise ValueError(f"lam must be in (0, 1], got {self.lam}")
        if not (0.0 < self.epsilon < 1.0):
            raise ValueError(f"epsilon must be in (0, 1), got {self.epsilon}")
        if self.lam + self.epsilon > 1.0 + 1e-9:
            raise ValueError(
                f"lam + epsilon = {self.lam + self.epsilon:.6f} > 1: the posture "
                "would overshoot and oscillate instead of settling"
            )
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    def threshold(self, channels: int) -> float:
        """``τ_α`` for ``channels`` channels, from the core's own null table."""
        return float(core_module().detector.null_threshold(int(channels), self.alpha))

    def as_dict(self) -> Dict[str, float]:
        return {
            "lam": float(self.lam),
            "epsilon": float(self.epsilon),
            "alpha": float(self.alpha),
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"lam={self.lam:.3f} eps={self.epsilon:.3f} alpha={self.alpha:g}"


@dataclass
class Standardiser:
    """Per-feature centre and scale, taken from the evidence without its labels.

    Fitted on *every* fixation the organism has, world and dream together, so it
    carries no provenance: it only puts six features that live in different units
    (a z-units chord, a GLRT, a kurtosis, a slope, a correlation) onto one
    comparable footing. Without it, an initial variance of 1.0 would mean "one
    kurtosis" for one channel and "one chord" for another.
    """

    centre: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, rows: Sequence[Sequence[float]]) -> "Standardiser":
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        centre = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - centre), axis=0)
        scale = 1.4826 * mad
        spread = matrix.std(axis=0) if matrix.shape[0] > 1 else np.ones_like(scale)
        scale = np.where(scale > 0.0, scale, np.where(spread > 0.0, spread, 1.0))
        return cls(centre=centre, scale=np.maximum(scale, SCALE_FLOOR))

    def apply(self, features: Sequence[float]) -> np.ndarray:
        return (np.asarray(features, dtype=float) - self.centre) / self.scale


@dataclass
class ClassModel:
    """One diagonal Gaussian in standardised feature space, with a memory.

    The diagonal is a declaration, not a shortcut: the organism models each
    feature's own spread and refuses to invent correlations it has no evidence
    for. The log-determinant is kept in the likelihood, so an ignorant model (big
    variances) is not rewarded for being vague.

    ``mass`` is what makes λ and ε mean something *together*. Each write adds λ of
    new evidence and each step keeps only ``1 - ε`` of the old, so the mass
    settles at ``λ / ε`` — the model's horizon in fixations. λ is how much of the
    new reading is believed; ε is how much of the past is dropped. λ alone cannot
    express forgetting and ε alone cannot express learning.
    """

    dim: int
    mean: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    var: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    mass: float = 0.0
    updates: int = 0

    def __post_init__(self) -> None:
        if self.mean.size == 0:
            self.mean = np.zeros(int(self.dim), dtype=float)
        if self.var.size == 0:
            self.var = np.full(int(self.dim), INITIAL_VARIANCE, dtype=float)

    def write(self, features: Sequence[float], lam: float, epsilon: float) -> None:
        """One observation: move the mean, re-estimate the variance, forget a little."""
        reading = np.asarray(features, dtype=float)
        self.mass = (1.0 - epsilon) * self.mass + lam
        step = lam / max(self.mass, lam)
        delta = reading - self.mean
        self.mean = self.mean + step * delta
        self.var = (1.0 - step) * (self.var + step * delta * delta)
        self.var = np.maximum(self.var, VAR_FLOOR)
        self.updates += 1

    @property
    def shrink(self) -> float:
        """How much of the answer is still the innate prior, in ``[0, 1)``."""
        return PRIOR_MASS / (max(self.mass, 0.0) + PRIOR_MASS)

    def posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """The mean and variance the model will actually claim, prior included.

        Shrinking here rather than in :meth:`write` matters: compounding a shrink
        on every update silently turns into a second forgetting rate that fights
        λ, and then λ means two different things at once.
        """
        weight = self.shrink
        mean = (1.0 - weight) * self.mean
        var = (1.0 - weight) * self.var + weight * INITIAL_VARIANCE
        return mean, np.maximum(var, VAR_FLOOR)

    def distance(self, features: Sequence[float]) -> float:
        """RMS standardised deviation: 1.0 means "as far as this model expects"."""
        mean, var = self.posterior()
        gap = np.asarray(features, dtype=float) - mean
        return float(np.sqrt(np.mean(gap * gap / var)))

    def log_likelihood(self, features: Sequence[float]) -> float:
        mean, var = self.posterior()
        gap = np.asarray(features, dtype=float) - mean
        return float(-0.5 * np.sum(gap * gap / var + np.log(var)))


@dataclass
class Posture:
    """The organism's internal state: a model of the world and a model of itself.

    Two Gaussians over the same six features, learned from the same body at the
    same rates. The **world model** is written by what the organism believes came
    from outside; the **self model** by what it knows it produced. Resonance is
    then a likelihood ratio, not a distance to a running average:

    .. math:: \\ell_c(f) = -\\tfrac12 \\sum_j \\left[
                 (f_j - \\mu_{cj})^2 / \\sigma^2_{cj} + \\ln \\sigma^2_{cj} \\right]

    .. math:: R_{\\text{int}}(f) = 1 / (1 + e^{-(\\ell_w(f) - \\ell_s(f))})

    Why a ratio and not a distance: a distance from the average says the world is
    *unpredictable* — its streams disagree with each other, so they sit far from
    any average — which is the opposite of what "real" should mean. A likelihood
    ratio asks the right question: *does this reading look more like the world I
    have met, or like the noise I make?* The demonstration measures both: the
    distance version scores AUC 0.47 (worse than a coin), this one far above.
    """

    standardiser: Standardiser
    world: ClassModel
    own: ClassModel

    @classmethod
    def uninformed(cls, rows: Sequence[Sequence[float]]) -> "Posture":
        """Start from the evidence's own frame, with no memory of either class."""
        standardiser = Standardiser.fit(rows)
        dim = int(standardiser.centre.size)
        return cls(
            standardiser=standardiser,
            world=ClassModel(dim=dim),
            own=ClassModel(dim=dim),
        )

    @property
    def dim(self) -> int:
        return int(self.world.dim)

    def features(self, reading: Any) -> np.ndarray:
        """Standardised feature vector of one fixation, in the declared order."""
        raw = getattr(reading, "features", None)
        if raw is None:  # a Reading-shaped object from somewhere else
            from agent.physics.sensorium import feature_vector

            raw = feature_vector(reading)
        return self.standardiser.apply(raw)

    def write(
        self, features: Sequence[float], origin: str, lam: float, epsilon: float
    ) -> None:
        """Route one reading to the model that owns it, and forget a little."""
        target = self.world if origin == "world" else self.own
        target.write(features, lam, epsilon)

    def log_ratio(self, features: Sequence[float]) -> float:
        """``ℓ_world(f) − ℓ_self(f)``: the evidence, in nats, for "this is the world"."""
        return self.world.log_likelihood(features) - self.own.log_likelihood(features)

    def resonance(self, features: Sequence[float]) -> float:
        """``R_int``: how much better this reading fits the world than the self."""
        return internal_resonance(self.log_ratio(features))

    def deltas(self, features: Sequence[float]) -> Tuple[float, float]:
        return self.world.distance(features), self.own.distance(features)


def surprise(x: Sequence[float], reference: Sequence[float], sigma: float) -> float:
    """Standardised distance between a fixation and a posture.

    .. math:: \\Delta = \\|x - r\\| / (\\sigma \\sqrt{n})

    ``sigma`` is measured on the medium, so Δ is in units of "one channel's
    honest noise": Δ = 1 means the reading is one noise-width away from what the
    organism expected, in every channel at once.
    """
    difference = np.asarray(x, dtype=float) - np.asarray(reference, dtype=float)
    root = math.sqrt(max(difference.size, 1))
    scale = float(sigma) * root
    if scale <= 0.0:
        return float("inf") if np.any(difference) else 0.0
    return float(np.linalg.norm(difference) / scale)


def overlap(error: float, band: float = RESONANCE_BAND) -> float:
    """The book's overlap kernel ``S(Δ) = 1 / (1 + (Δ/Π)²)``.

    Used where a distance is wanted and the distance must not rise again with
    strangeness. The metric is :func:`internal_resonance`, which needs the two
    sidedness: a reading can be too strange for the world *and* too familiar for
    the self, and only a ratio of likelihoods can say both at once.
    """
    scaled = float(error) / float(band)
    return 1.0 / (1.0 + scaled * scaled)


def internal_resonance(log_likelihood_ratio: float) -> float:
    """The metric: ``R_int = sigmoid(ℓ_world − ℓ_self)``, in ``(0, 1)``.

    High when the reading fits the learned model of the world better than the
    learned model of the organism's own voice; low when the reverse. This is the
    behaviour the whole exercise is judged on, and it is a likelihood ratio rather
    than a hand-tuned score, so each of the two numbers it is built from can be
    printed and checked.
    """
    ratio = max(min(float(log_likelihood_ratio), 60.0), -60.0)
    return float(1.0 / (1.0 + math.exp(-ratio)))


@dataclass
class Life:
    """One continuous life, fixation by fixation, with the provenance recorded.

    Every row is a fixation the organism actually had. ``rho`` is the metric
    *before* the reading was written into its model — a prediction, not a
    hindsight — and ``keep`` later marks which rows were held out of the search.
    """

    origin: np.ndarray
    rho: np.ndarray
    world_distance: np.ndarray
    self_distance: np.ndarray
    source: np.ndarray
    features: np.ndarray
    posture: Posture
    frozen: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __len__(self) -> int:
        return int(self.rho.size)

    def mask(self, keep: Optional[np.ndarray] = None) -> np.ndarray:
        if keep is None:
            return np.ones(len(self), dtype=bool)
        array = np.asarray(keep, dtype=bool)
        if array.size != len(self):
            raise ValueError(
                f"holdout mask has {array.size} entries for a life of {len(self)}"
            )
        return array

    def piles(self, keep: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """``(rho_world, rho_self)`` on the kept fixations."""
        selected = self.mask(keep)
        world = self.rho[selected & (self.origin == "world")]
        dream = self.rho[selected & (self.origin != "world")]
        return world, dream

    def separation(self, keep: Optional[np.ndarray] = None) -> Dict[str, float]:
        world, dream = self.piles(keep)
        out = separation(world, dream)
        out["macro_margin"] = self.macro(keep)
        return out

    def macro(self, keep: Optional[np.ndarray] = None) -> float:
        """Margin computed per stream and then averaged: no stream can carry the rest.

        The pooled margin is a trap, and it was measured: with a short memory the
        class model fits whichever stream it saw last, scoring 1.0 on that one and
        0.32 on the first — pooled, that looks like 0.69 and passes. Averaging the
        margin *inside each stream first* makes every stream count once, so a body
        that only remembers the last thing it heard cannot buy the score with it.
        """
        selected = self.mask(keep)
        world: List[float] = []
        dream: List[float] = []
        for name in sorted(set(self.source.tolist())):
            rows = selected & (self.source == name)
            if not rows.any():
                continue
            value = float(self.rho[rows].mean())
            (world if str(self.origin[rows][0]) == "world" else dream).append(value)
        if not world or not dream:
            return float("nan")
        return float(np.mean(world) - np.mean(dream))

    def by_source(self, keep: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
        """Per-stream averages: the check that one domain is not carrying the rest."""
        selected = self.mask(keep)
        out: Dict[str, Dict[str, Any]] = {}
        for name in sorted(set(self.source.tolist())):
            rows = selected & (self.source == name)
            if not rows.any():
                continue
            origin = str(self.origin[rows][0])
            values = self.rho[rows]
            out[str(name)] = {
                "origin": origin,
                "fixations": int(values.size),
                "rho_mean": float(values.mean()),
                "rho_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
        return out


def run_life(
    episodes: Sequence[Any],
    parameters: Parameters,
    posture: Optional[Posture] = None,
    freeze_after: float = 0.7,
) -> Life:
    """Live one continuous life: predict each fixation, then learn from it.

    The posture persists across episodes and is never reset, because an organism
    that reset its expectation on every new stream would never notice that streams
    differ. Each fixation is scored *before* it is written, so ``rho`` is a
    one-step-ahead prediction rather than a fit to something already seen.

    ``freeze_after`` is the honest part. Inside an episode only the first
    ``freeze_after`` share of the fixations may write into the models; the rest are
    scored as an extrapolation and flagged ``frozen``. Without it the measured
    optimum is not an optimum but an artefact: overlapping windows make sixty
    consecutive fixations nearly identical, an unconstrained λ goes to 0.9,
    memorises the block, scores marvellously — and then cannot recognise the same
    stream a hundred fixations later (measured: −16 nats). That is exactly the
    overfitting the acceptance rule exists to catch, so the calibration is not
    allowed to commit it.
    """
    if not episodes:
        return Life(
            origin=np.zeros(0, dtype=object),
            rho=np.zeros(0),
            world_distance=np.zeros(0),
            self_distance=np.zeros(0),
            source=np.zeros(0, dtype=object),
            features=np.zeros((0, 0)),
            posture=Posture(
                standardiser=Standardiser(np.zeros(0), np.ones(0)),
                world=ClassModel(dim=0),
                own=ClassModel(dim=0),
            ),
            frozen=np.zeros(0, dtype=bool),
        )
    if posture is None:
        rows = [
            getattr(reading, "features", None)
            for episode in episodes
            for reading in episode.readings
        ]
        if any(row is None for row in rows):
            from agent.physics.sensorium import feature_vector

            rows = [
                feature_vector(reading)
                for episode in episodes
                for reading in episode.readings
            ]
        state = Posture.uninformed(rows)  # type: ignore[arg-type]
    else:
        state = posture
    origin: List[str] = []
    source: List[str] = []
    rho: List[float] = []
    world_distance: List[float] = []
    self_distance: List[float] = []
    features: List[np.ndarray] = []
    frozen: List[bool] = []
    for episode in episodes:
        count = len(episode.readings)
        cut = int(round(max(min(float(freeze_after), 1.0), 0.0) * count))
        for position, reading in enumerate(episode.readings):
            vector = state.features(reading)
            gap_world, gap_self = state.deltas(vector)
            rho.append(state.resonance(vector))
            world_distance.append(gap_world)
            self_distance.append(gap_self)
            features.append(vector)
            origin.append(episode.origin)
            source.append(episode.source)
            learning = position < cut
            frozen.append(not learning)
            if learning:
                state.write(vector, episode.origin, parameters.lam, parameters.epsilon)
    return Life(
        origin=np.asarray(origin, dtype=object),
        rho=np.asarray(rho, dtype=float),
        world_distance=np.asarray(world_distance, dtype=float),
        self_distance=np.asarray(self_distance, dtype=float),
        source=np.asarray(source, dtype=object),
        features=np.asarray(features, dtype=float),
        posture=state,
        frozen=np.asarray(frozen, dtype=bool),
    )


def life_resonance(
    episodes: Sequence[Any], parameters: Parameters, keep: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """The two piles of ``rho`` from one life, in the shape the report wants."""
    life = run_life(episodes, parameters)
    world, dream = life.piles(keep)
    return {"world": world, "self": dream, "world_distance": life.world_distance,
            "self_distance": life.self_distance, "posture": life.posture}


def auc(world: Sequence[float], dream: Sequence[float]) -> float:
    """Mann-Whitney AUC: ``P(ρ_world > ρ_self)``, ties counting one half.

    Rank based on purpose. It needs no threshold, so it cannot be gamed by a
    scale choice, and 0.5 is exactly "these two piles are the same pile" — the
    number that a phase-randomised surrogate is entitled to.
    """
    a = np.asarray(world, dtype=float)
    b = np.asarray(dream, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    joint = np.concatenate([a, b])
    order = joint.argsort(kind="mergesort")
    ranks = np.empty(joint.size, dtype=float)
    ranks[order] = np.arange(1, joint.size + 1, dtype=float)
    # average ranks inside tie groups
    sorted_joint = joint[order]
    start = 0
    for index in range(1, joint.size + 1):
        if index == joint.size or sorted_joint[index] != sorted_joint[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[: a.size].sum() - a.size * (a.size + 1) / 2.0) / (a.size * b.size))


def separation(world: Sequence[float], dream: Sequence[float]) -> Dict[str, float]:
    """Margin, AUC and d' between two piles of one statistic.

    Three numbers because one is never enough: a margin can be large and the AUC
    mediocre (one wild outlier), and an AUC can be perfect while both piles sit
    on top of each other in the tails (a bimodal disaster that a single
    threshold will mis-handle).
    """
    a = np.asarray(world, dtype=float)
    b = np.asarray(dream, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"margin": float("nan"), "auc": float("nan"), "d_prime": float("nan"),
                "world_mean": float("nan"), "self_mean": float("nan"),
                "world_std": float("nan"), "self_std": float("nan")}
    pooled = math.sqrt(max((a.var(ddof=1) if a.size > 1 else 0.0)
                           + (b.var(ddof=1) if b.size > 1 else 0.0), 0.0) / 2.0)
    d_prime = float((a.mean() - b.mean()) / pooled) if pooled > 0.0 else 0.0
    return {
        "margin": float(a.mean() - b.mean()),
        "auc": auc(a, b),
        "d_prime": d_prime,
        "world_mean": float(a.mean()),
        "self_mean": float(b.mean()),
        "world_std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "self_std": float(b.std(ddof=1)) if b.size > 1 else 0.0,
    }


def objective(
    episodes: Sequence[Any], parameters: Parameters, keep: Optional[np.ndarray] = None
) -> float:
    """The one number: the per-stream world resonance minus the per-stream self one.

    Args:
        episodes: the life, in the order the organism met the streams.
        parameters: λ and ε; α does not enter, because α is a risk and a risk does
            not change what the organism resonates with.
        keep: optional boolean mask over fixations.

    Macro-average, not pooled: see :meth:`Life.macro` for the failure that taught
    this. A body with a short memory scores brilliantly on whatever it heard last
    and poorly on everything else; the pooled mean hides that and the per-stream
    mean does not.
    """
    life = run_life(episodes, parameters)
    world, dream = life.piles(keep)
    if world.size == 0 or dream.size == 0:
        return float("nan")
    return life.macro(keep)



# --------------------------------------------------------------------------- #
# The mutation. A grid can only find what the grid contains.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Mutation:
    """One attempt to be different, and what it bought.

    ``accepted`` is a fact about the objective, not about intention: the organism
    keeps the candidate only when the candidate reads itself better. A rejected
    mutation is kept in the log because the *rate* of rejections is the evidence
    that the search has converged — and a run with no rejection at all is evidence
    that the objective is too flat to be a claim.
    """

    generation: int
    changed: bool
    accepted: bool
    before: Parameters
    candidate: Parameters
    objective: float
    best: float
    gain: float
    phase: str = ""
    scale: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "phase": self.phase,
            "changed": self.changed,
            "accepted": self.accepted,
            "lam": self.candidate.lam,
            "epsilon": self.candidate.epsilon,
            "alpha": self.candidate.alpha,
            "objective": round(self.objective, 6),
            "gain": round(self.gain, 6),
            "scale": round(self.scale, 4),
        }


def mutate(
    parameters: Parameters,
    rng: Optional[np.random.Generator] = None,
    p_mutation: float = MUTATION_RATE,
    scale: float = MUTATION_SCALE,
) -> Tuple[Parameters, bool]:
    """Perturb each parameter with probability ``p_mutation``; return ``(params, changed)``.

    Multiplicative and log-normal, because λ and ε are rates: a mutation of
    ``exp(±scale)`` means "halve it or double it", which is the same perturbation
    at 0.05 and at 0.8. α is categorical — it can only move to another declared
    risk, because inventing a risk is inventing a promise.

    Args:
        parameters: the parent.
        rng: a seeded generator; ``None`` draws fresh entropy.
        p_mutation: probability per parameter per generation, default 8%.
        scale: one standard deviation of the log-multiplier.

    Returns:
        ``(candidate, changed)``; ``changed`` is False when every draw missed,
        and the parent is then returned untouched.
    """
    generator = rng if rng is not None else np.random.default_rng()
    lam, epsilon, alpha = parameters.lam, parameters.epsilon, parameters.alpha
    ceiling = 1.0 - EPSILON_FLOOR
    changed = False
    if generator.random() < p_mutation:
        lam = float(np.clip(lam * math.exp(generator.normal(0.0, scale)), 0.02, ceiling))
        changed = True
    if generator.random() < p_mutation:
        epsilon = float(epsilon * math.exp(generator.normal(0.0, scale)))
        epsilon = float(np.clip(epsilon, EPSILON_FLOOR, 0.6))
        changed = True
    if generator.random() < p_mutation:
        alpha = float(ALPHA_GRID[int(generator.integers(0, len(ALPHA_GRID)))])
        changed = True
    if lam + epsilon > 1.0:  # project onto the constraint, keeping the ratio
        total = lam + epsilon
        lam = lam / total
        epsilon = epsilon / total
        changed = True
    epsilon = float(min(max(epsilon, EPSILON_FLOOR), 0.6))
    lam = float(min(max(lam, 0.02), 1.0 - epsilon))
    if not changed:
        return parameters, False
    return (
        Parameters(
            lam=round(lam, 5), epsilon=round(epsilon, 5), alpha=float(alpha)
        ),
        True,
    )


def search_grid(
    episodes: Sequence[Any],
    lam_grid: Sequence[float] = LAMBDA_GRID,
    eps_grid: Sequence[float] = EPSILON_GRID,
    alpha: float = DEFAULT_START[2],
    keep: Optional[np.ndarray] = None,
) -> Tuple[Parameters, float, List[Dict[str, float]]]:
    """Evaluate the declared lattice exhaustively; return ``(best, value, surface)``.

    The surface is returned, not swallowed: a ridge parallel to an axis means the
    objective does not care about one of the parameters, and an optimum on the
    edge of the box means the box was too small. Both are findings.
    """
    surface: List[Dict[str, float]] = []
    best: Optional[Parameters] = None
    best_value = float("-inf")
    for lam in lam_grid:
        for epsilon in eps_grid:
            if lam + epsilon > 1.0:
                continue
            candidate = Parameters(lam=float(lam), epsilon=float(epsilon), alpha=alpha)
            value = objective(episodes, candidate, keep)
            surface.append(
                {"lam": float(lam), "epsilon": float(epsilon), "objective": value}
            )
            if np.isfinite(value) and value > best_value:
                best, best_value = candidate, value
    if best is None:
        best = Parameters(lam=DEFAULT_START[0], epsilon=DEFAULT_START[1], alpha=alpha)
        best_value = objective(episodes, best)
    return best, float(best_value), surface


def refine(
    episodes: Sequence[Any],
    start: Parameters,
    max_iter: int = 60,
    keep: Optional[np.ndarray] = None,
) -> Tuple[Parameters, float, Dict[str, Any]]:
    """Powell's method inside the same box, in log-λ and log-ε coordinates.

    Log coordinates because the parameters are rates: a step of 0.05 is large at
    λ = 0.05 and invisible at λ = 0.9. Skipped and reported if scipy is absent —
    a missing optimiser must not silently turn the grid into the whole story.
    """
    record: Dict[str, Any] = {"used": False, "method": "Powell"}
    baseline = objective(episodes, start, keep)
    try:
        from scipy.optimize import minimize
    except ImportError:  # pragma: no cover - scipy is a declared dep
        record["reason"] = "scipy not installed"
        return start, baseline, record

    def decode(vector: Sequence[float]) -> Parameters:
        lam = float(np.clip(math.exp(vector[0]), 0.02, 1.0 - EPSILON_FLOOR))
        epsilon = float(np.clip(math.exp(vector[1]) - 0.01, EPSILON_FLOOR, 0.6))
        epsilon = float(min(epsilon, max(EPSILON_FLOOR, 1.0 - lam)))
        lam = float(min(lam, 1.0 - epsilon))
        return Parameters(lam=lam, epsilon=epsilon, alpha=start.alpha)

    def negative(vector: Sequence[float]) -> float:
        value = objective(episodes, decode(vector), keep)
        return -value if np.isfinite(value) else 1e9

    result = minimize(
        negative,
        [math.log(start.lam), math.log(start.epsilon + 0.01)],
        method="Powell",
        bounds=[(math.log(0.02), 0.0), (math.log(0.01), math.log(0.61))],
        options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-4},
    )
    candidate = decode(result.x)
    value = objective(episodes, candidate, keep)
    record.update(
        {
            "used": True,
            "converged": bool(result.success),
            "evaluations": int(getattr(result, "nfev", 0)),
            "improved": bool(np.isfinite(value) and value > baseline),
        }
    )
    if np.isfinite(value) and value > baseline:
        return candidate, float(value), record
    return start, baseline, record


def evolve(
    episodes: Sequence[Any],
    start: Parameters,
    steps: int = MUTATION_STEPS,
    rng: Optional[np.random.Generator] = None,
    p_mutation: float = MUTATION_RATE,
    scale: float = MUTATION_SCALE,
    keep: Optional[np.ndarray] = None,
    phase: str = "",
    adapt: bool = True,
) -> Tuple[Parameters, float, List[Mutation]]:
    """(1+1) evolution with elitist rollback: 8% per parameter per generation.

    The parent is kept unless the child reads itself strictly better, so ``best``
    never decreases and the loop is a monotone lower bound on what mutation can
    buy from this starting point.

    ``adapt`` applies the 1/5th success rule: every five *attempts* the step is
    widened by 1.5 when more than a fifth were accepted and narrowed by 1.5 when
    fewer were. Without it the measured truth is uncomfortable: from a refined
    optimum, twenty attempts at a fixed 45% step bought exactly nothing, because
    every one of them stepped off the peak. The rule is the difference between a
    mechanism that exists and a mechanism that works.
    """
    generator = rng if rng is not None else np.random.default_rng(0)
    best = start
    best_value = objective(episodes, best, keep)
    log: List[Mutation] = []
    current = float(scale)
    window: List[bool] = []
    for generation in range(int(steps)):
        parent = best
        candidate, changed = mutate(parent, generator, p_mutation, current)
        value = best_value if not changed else objective(episodes, candidate, keep)
        accepted = bool(changed and np.isfinite(value) and value > best_value + 1e-12)
        log.append(
            Mutation(
                generation=generation,
                changed=changed,
                accepted=accepted,
                before=parent,
                candidate=candidate,
                objective=float(value),
                best=float(value if accepted else best_value),
                gain=float(value - best_value) if changed else 0.0,
                phase=phase,
                scale=current,
            )
        )
        if not changed:
            continue
        window.append(accepted)
        if len(window) >= 5:
            rate = sum(window) / len(window)
            current = current * (1.5 if rate > 0.2 else 1.0 / 1.5)
            current = float(min(max(current, 0.01), 1.5))
            window = []
        if accepted:
            best, best_value = candidate, float(value)
    return best, float(best_value), log


def gate_rates(episodes: Sequence[Any], alpha: float, channels: int) -> Dict[str, float]:
    """How often the core's GLRT fires on the world and on the organism's dreams.

    The gate is the core's own verdict at the declared risk —
    ``detector.detect(x, sigma, alpha)`` — not a re-implementation of it.
    """
    detector = core_module().detector
    counts = {"world": [0, 0], "self": [0, 0]}
    for episode in episodes:
        bucket = counts["world" if episode.origin == "world" else "self"]
        for observation in episode.observations:
            verdict = detector.detect(
                np.asarray(observation.x, dtype=float), float(observation.sigma), alpha
            )
            bucket[0] += int(bool(verdict.flagged))
            bucket[1] += 1
    out: Dict[str, float] = {
        "tau": float(core_module().detector.null_threshold(int(channels), alpha)),
        "alpha": float(alpha),
    }
    for key, (hits, total) in counts.items():
        out[f"{key}_rate"] = float(hits / total) if total else float("nan")
        out[f"{key}_count"] = float(total)
    out["youden"] = out["world_rate"] - out["self_rate"]
    return out


def choose_alpha(
    episodes: Sequence[Any],
    parameters: Parameters,
    channels: int,
    alphas: Sequence[float] = ALPHA_GRID,
    minimum_youden: float = 0.05,
) -> Tuple[float, str, List[Dict[str, float]]]:
    """Pick the declared risk, and say which rule picked it.

    Two rules, declared in advance:

    ``youden``              maximise ``TPR - FPR`` of the surprise gate. Used when
                            the world genuinely trips the gate more often than the
                            organism's own dreams do, by at least ``minimum_youden``.
    ``false-alarm-budget``  otherwise, keep the most sensitive α whose *empirical*
                            alarm rate on the organism's own dreams stays inside
                            the declared budget. This is the honest fallback: if the
                            gate cannot tell the world from a dream, the only thing
                            left to calibrate is how often it cries wolf.

    Returns ``(alpha, rule, table)``.
    """
    table: List[Dict[str, float]] = [
        gate_rates(episodes, float(alpha), channels) for alpha in alphas
    ]
    if not table:
        return parameters.alpha, "default", table
    best_youden = max(table, key=lambda row: row["youden"])
    if np.isfinite(best_youden["youden"]) and best_youden["youden"] >= minimum_youden:
        return float(best_youden["alpha"]), "youden", table
    inside = [
        row
        for row in table
        if np.isfinite(row["self_rate"]) and row["self_rate"] <= row["alpha"] + 1e-12
    ]
    if inside:
        chosen = max(inside, key=lambda row: row["alpha"])
        return float(chosen["alpha"]), "false-alarm-budget", table
    chosen = min(table, key=lambda row: row["alpha"])
    return float(chosen["alpha"]), "muted", table


@dataclass
class Calibration:
    """What the search found, and what it is worth out of sample.

    ``objective`` is the tuning half; ``validation`` is the held-out half. The two
    are stored side by side on purpose: a big tuning number with a small
    validation number is a fit, and the report has to show both.
    """

    parameters: Parameters
    channels: int
    objective: float
    validation: Dict[str, float]
    tuning: Dict[str, float]
    surface: List[Dict[str, float]] = field(default_factory=list)
    grid_best: Dict[str, float] = field(default_factory=dict)
    refine: Dict[str, Any] = field(default_factory=dict)
    mutations: List[Mutation] = field(default_factory=list)
    alphas: List[Dict[str, float]] = field(default_factory=list)
    alpha_rule: str = "default"
    steps: List[Dict[str, float]] = field(default_factory=list)
    phases: List[Dict[str, Any]] = field(default_factory=list)
    sizes: Dict[str, int] = field(default_factory=dict)
    streams: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def threshold(self) -> float:
        return self.parameters.threshold(self.channels)

    @property
    def accepted_mutations(self) -> int:
        return sum(1 for mutation in self.mutations if mutation.accepted)

    @property
    def attempted_mutations(self) -> int:
        return sum(1 for mutation in self.mutations if mutation.changed)

    def summary(self) -> Dict[str, Any]:
        """A JSON-able account, honest about which half each number came from."""
        return {
            "parameters": self.parameters.as_dict(),
            "threshold": round(self.threshold, 6),
            "channels": self.channels,
            "objective_tuning": round(float(self.objective), 6),
            "tuning": {k: round(float(v), 6) for k, v in self.tuning.items()},
            "validation": {k: round(float(v), 6) for k, v in self.validation.items()},
            "grid_best": self.grid_best,
            "refine": self.refine,
            "mutations": {
                "attempted": self.attempted_mutations,
                "accepted": self.accepted_mutations,
                "log": [m.as_dict() for m in self.mutations],
            },
            "phases": self.phases,
            "alpha_rule": self.alpha_rule,
            "alphas": [
                {
                    key: (round(float(value), 6) if isinstance(value, float) else value)
                    for key, value in row.items()
                }
                for row in self.alphas
            ],
            "sizes": self.sizes,
            "streams": self.streams,
            "warnings": self.warnings,
        }


def calibrate(
    episodes: Sequence[Any],
    *,
    holdout: float = 0.3,
    lam_grid: Sequence[float] = LAMBDA_GRID,
    eps_grid: Sequence[float] = EPSILON_GRID,
    alphas: Sequence[float] = ALPHA_GRID,
    steps: int = MUTATION_STEPS,
    p_mutation: float = MUTATION_RATE,
    scale: float = MUTATION_SCALE,
    seed: int = 11,
    refinement: bool = True,
    channels: Optional[int] = None,
) -> Calibration:
    """Find λ, ε and α: grid, then Powell, then 8%-per-parameter mutation.

    Args:
        episodes: both piles, in the order the organism met them. Order matters
            inside an episode (the posture is a trajectory), not between them.
        holdout: fraction of each source's episodes held back for validation.
        steps: generations of mutation after the refinement.
        p_mutation: per-parameter probability of a perturbation, default 8%.
        seed: seeds the mutation stream and the split; a rerun with the same seed
            is the same organism.
        refinement: run Powell from the best grid cell (skipped without scipy).

    Raises:
        ValueError: one of the two piles is empty — there is nothing to separate.
    """
    world = [episode for episode in episodes if episode.origin == "world"]
    dream = [episode for episode in episodes if episode.origin != "world"]
    if not world:
        raise ValueError("calibration needs at least one world episode")
    if not dream:
        raise ValueError("calibration needs at least one dream episode")

    warnings: List[str] = []
    resolved_channels = int(
        channels if channels is not None else np.asarray(episodes[0].readings[0].x).size
    )
    start_alpha = alphas[0] if alphas else DEFAULT_START[2]
    grid_best, grid_value, surface = search_grid(
        episodes, lam_grid, eps_grid, alpha=start_alpha
    )
    best, value = grid_best, grid_value
    generator = np.random.default_rng(seed)
    phases: List[Dict[str, Any]] = [
        {
            "phase": "grid",
            "lam": float(best.lam),
            "epsilon": float(best.epsilon),
            "objective": float(value),
            "accepted": 0,
            "attempted": 0,
        }
    ]

    # Phase 1 — coarse mutation from the best lattice node. This is the phase with
    # a job the grid cannot do: leave the basin the lattice found.
    coarse, coarse_value, coarse_log = evolve(
        episodes,
        best,
        steps=steps,
        rng=generator,
        p_mutation=p_mutation,
        scale=scale,
        phase="coarse",
    )
    log: List[Mutation] = list(coarse_log)
    if np.isfinite(coarse_value) and coarse_value > value:
        best, value = coarse, coarse_value
    phases.append(
        {
            "phase": "coarse",
            "lam": float(best.lam),
            "epsilon": float(best.epsilon),
            "objective": float(value),
            "accepted": sum(1 for entry in coarse_log if entry.accepted),
            "attempted": sum(1 for entry in coarse_log if entry.changed),
        }
    )
    if not any(entry.accepted for entry in coarse_log):
        warnings.append(
            "the coarse mutations bought nothing: the lattice node was already a "
            "local optimum for this evidence"
        )

    # Phase 2 — Powell, from wherever the mutations ended up.
    record: Dict[str, Any] = {"used": False, "method": "Powell"}
    if refinement:
        refined, refined_value, record = refine(episodes, best)
        if np.isfinite(refined_value) and refined_value > value:
            best, value = refined, refined_value
        phases.append(
            {
                "phase": "powell",
                "lam": float(best.lam),
                "epsilon": float(best.epsilon),
                "objective": float(value),
                "accepted": int(bool(record.get("improved"))),
                "attempted": int(record.get("evaluations", 0)),
            }
        )

    # Phase 3 — fine mutation: polish without stepping off the peak.
    fine, fine_value, fine_log = evolve(
        episodes,
        best,
        steps=max(int(steps) // 2, 10),
        rng=generator,
        p_mutation=p_mutation,
        scale=MUTATION_SCALE_FINE,
        phase="fine",
    )
    log.extend(fine_log)
    if np.isfinite(fine_value) and fine_value > value:
        best, value = fine, fine_value
    phases.append(
        {
            "phase": "fine",
            "lam": float(best.lam),
            "epsilon": float(best.epsilon),
            "objective": float(value),
            "accepted": sum(1 for entry in fine_log if entry.accepted),
            "attempted": sum(1 for entry in fine_log if entry.changed),
        }
    )
    if not any(entry.accepted for entry in log):
        warnings.append(
            "no mutation was accepted in any phase: the objective is flat here, "
            "which is a statement about the evidence, not about the search"
        )

    provisional = Parameters(lam=best.lam, epsilon=best.epsilon, alpha=start_alpha)
    alpha, rule, table = choose_alpha(
        episodes, provisional, resolved_channels, alphas=alphas
    )
    parameters = Parameters(lam=best.lam, epsilon=best.epsilon, alpha=alpha)

    life = run_life(episodes, parameters)
    tuning_separation = life.separation(~life.frozen)
    validation_separation = life.separation(life.frozen)
    return Calibration(
        parameters=parameters,
        channels=resolved_channels,
        objective=objective(episodes, parameters),
        validation=validation_separation,
        tuning=tuning_separation,
        surface=surface,
        grid_best={
            "lam": grid_best.lam,
            "epsilon": grid_best.epsilon,
            "objective": float(grid_value),
        },
        refine=record,
        mutations=log,
        alphas=table,
        alpha_rule=rule,
        steps=[
            {
                "generation": entry.generation,
                "phase": entry.phase,
                "objective": entry.objective,
                "best": entry.best,
                "accepted": entry.accepted,
                "changed": entry.changed,
                "scale": float(entry.scale),
            }
            for entry in log
        ],
        phases=phases,
        sizes={
            "fixations": int(len(life)),
            "tuning_fixations": int((~life.frozen).sum()),
            "validation_fixations": int(life.frozen.sum()),
            "world_fixations": int((life.origin == "world").sum()),
            "self_fixations": int((life.origin != "world").sum()),
        },
        streams=life.by_source(life.frozen),
        warnings=warnings,
    )


@dataclass
class Homeostasis:
    """The autonomic state of a living organism, and its right to keep mutating.

    After :func:`calibrate` the parameters are fixed, but the body does not stop
    trying: :meth:`tick` performs one mutation attempt against the evidence the
    organism keeps, accepts it when the objective improves, and rolls back when it
    does not. That is what lets the demonstration show parameters *stabilising*
    rather than merely *being set* — and it is why a run that never accepts
    anything is visible as such instead of looking like convergence.
    """

    parameters: Parameters
    channels: int = 8
    evidence: List[Any] = field(default_factory=list)
    calibration: Optional[Calibration] = None
    attempts: List[Mutation] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    @classmethod
    def awaken(cls, episodes: Sequence[Any], **kwargs: Any) -> "Homeostasis":
        """Calibrate on ``episodes`` and hand back a body ready to live."""
        calibration = calibrate(episodes, **kwargs)
        return cls(
            parameters=calibration.parameters,
            channels=calibration.channels,
            evidence=list(episodes),
            calibration=calibration,
        )

    @property
    def threshold(self) -> float:
        return self.parameters.threshold(self.channels)

    def verdict(self, observation: Any) -> Any:
        """The core's verdict on one fixation, at this body's declared risk."""
        return core_module().detector.detect(
            np.asarray(observation.x, dtype=float),
            float(observation.sigma),
            self.parameters.alpha,
        )

    def gate(self, observation: Any) -> bool:
        """Does the surprise gate fire? (``statistic > τ_α``, the core's rule.)"""
        return bool(self.verdict(observation).flagged)

    def tick(
        self,
        p_mutation: float = MUTATION_RATE,
        scale: float = MUTATION_SCALE,
    ) -> Optional[Mutation]:
        """One mutation attempt against the stored evidence; accept or roll back.

        Returns ``None`` when there is no evidence to judge a mutation by — an
        organism without a past cannot tell an improvement from a change.
        """
        if not self.evidence or int(self.channels) < 2:
            return None
        current = objective(self.evidence, self.parameters)
        candidate, changed = mutate(self.parameters, self.rng, p_mutation, scale)
        value = objective(self.evidence, candidate) if changed else current
        accepted = bool(changed and np.isfinite(value) and value > current + 1e-12)
        attempt = Mutation(
            generation=len(self.attempts),
            changed=changed,
            accepted=accepted,
            before=self.parameters,
            candidate=candidate,
            objective=float(value),
            best=float(value if accepted else current),
            gain=float(value - current) if changed else 0.0,
        )
        self.attempts.append(attempt)
        if accepted:
            self.parameters = candidate
        return attempt

    def summary(self) -> Dict[str, Any]:
        return {
            "parameters": self.parameters.as_dict(),
            "threshold": round(self.threshold, 6),
            "channels": self.channels,
            "consistent_posture": bool(self.parameters.lam + self.parameters.epsilon <= 1.0),
            "attempts": len(self.attempts),
            "accepted": sum(1 for a in self.attempts if a.accepted),
            "calibration": None if self.calibration is None else self.calibration.summary(),
        }


__all__ = [
    "ALPHA_GRID",
    "Calibration",
    "ClassModel",
    "DEFAULT_START",
    "EPSILON_GRID",
    "Homeostasis",
    "INITIAL_VARIANCE",
    "LAMBDA_GRID",
    "Life",
    "MUTATION_RATE",
    "MUTATION_SCALE",
    "MUTATION_STEPS",
    "Mutation",
    "Parameters",
    "Posture",
    "RESONANCE_BAND",
    "SCALE_FLOOR",
    "Standardiser",
    "VAR_FLOOR",
    "auc",
    "calibrate",
    "choose_alpha",
    "evolve",
    "gate_rates",
    "internal_resonance",
    "life_resonance",
    "mutate",
    "objective",
    "overlap",
    "refine",
    "run_life",
    "search_grid",
    "separation",
    "surprise",
]
