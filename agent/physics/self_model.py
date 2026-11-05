"""The mirror: the organism reading itself reading the world.

``internal_resonance`` — the metric, from :mod:`agent.physics.homeostasis` — is
already the right shape: high when a reading fits the learned model of the world
better than the learned model of the organism's own voice. This module turns it
into a *decision*, and the decision needs three arms, only one of which is a
statistic:

1. **the metric** — ``R_int = sigmoid(ℓ_world − ℓ_self)``, computed before the
   reading is learned from. It is a one-step-ahead prediction, not a fit.
2. **the prior** — ``P(world)``, declared, because an organism that expects the
   world more often than its own dreams should say so. The demonstration sweeps
   this number instead of hiding it.
3. **the ledger** — what the organism *did*. By the book's own statement the
   orientation of a split is not in the data, so no statistic can decide "I made
   this". Only memory can. Every emission is recorded with a content address, and
   the memory of having emitted it decays at ε — the same ε that forgets the world,
   because it is the same body.

The ledger is not a convenience. The demonstration measures the honest limit: a
phase-randomised surrogate of a real stream carries the world's spectrum and the
world's exact marginal distribution, and it scored ``R_int = 0.644`` against a
world mean of 0.72 — a coin with a slight tilt. Nothing local in the reading can
refuse it. What refuses it is remembering that one made it.

So:

.. math:: s(f) = \\bigl(1 - r(f)\\bigr) \\cdot \\sigma\\bigl(
              \\operatorname{logit} R_{\\text{int}}(f) + \\log \\tfrac{P}{1-P} \\bigr)

where ``r(f)`` is the decaying recall of having emitted exactly these bytes. For
the world ``r = 0`` and the score is the calibrated posterior; for a fresh dream
``r ≈ 1`` and the score collapses to zero. For an old dream, ``r = e^{-εΔt}``
decays and the dream is re-heard as the world — the *phantom*, which is the
prediction this whole construction has to make and which
:func:`SelfModel.phantom` measures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agent.physics import sensorium
from agent.physics.homeostasis import (
    Homeostasis,
    Parameters,
    Posture,
    internal_resonance,
    overlap,
    run_life,
    separation,
    surprise,
)

#: Declared prior odds that a stream the organism meets is the world. 2:1 is the
#: ratio a body can defend: it hears the world in more streams than it invents, but
#: not ten times more. :func:`prior_sweep` measures what the choice costs.
WORLD_PRIOR = 2.0 / 3.0

#: A reading whose score exceeds this is written into the world model; below it,
#: into the self model. 0.5 is the only threshold that is not a preference.
WORLD_JUDGEMENT = 0.5

#: How long the ledger keeps a content address, in fixations. Pruning is by ε, not
#: by this number; the number only bounds the bookkeeping of a long life.
LEDGER_LIMIT = 4096


def logit(probability: float) -> float:
    """``log(p / (1 - p))``, clipped away from the ends."""
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    """``1 / (1 + exp(-x))``, clipped so a wide likelihood ratio cannot overflow."""
    x = min(max(float(value), -60.0), 60.0)
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class Emission:
    """One thing the organism made, and when it made it."""

    fingerprint: str
    source: str
    label: str
    step: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint[:16],
            "source": self.source,
            "label": self.label,
            "step": self.step,
        }


@dataclass
class Ledger:
    """What the organism has emitted: the only certain knowledge it has.

    The *act* of emission is certain at the moment it happens — an organism knows
    what its own muscles did, and that certainty is not an inference. What decays
    is the memory of *having* done it, at rate ε:

    .. math:: r(f, t) = \\max_{e: e.fp = f} e^{-\\varepsilon (t - e.t)}

    A perfect match is required. A near match is not a recall, because the question
    is "did I make this?" and not "does this resemble what I made?" — the second
    question is the metric's job and it is much worse at it.
    """

    emissions: Dict[str, Emission] = field(default_factory=dict)
    limit: int = LEDGER_LIMIT

    def record(
        self, perception: Any, step: int, label: str = ""
    ) -> Emission:
        """Write down one emission, replacing any earlier copy of the same bytes."""
        entry = Emission(
            fingerprint=perception.fingerprint,
            source=perception.source,
            label=label or perception.label,
            step=int(step),
        )
        self.emissions[entry.fingerprint] = entry
        if len(self.emissions) > self.limit:
            oldest = sorted(self.emissions.values(), key=lambda item: item.step)
            for stale in oldest[: len(self.emissions) - self.limit]:
                self.emissions.pop(stale.fingerprint, None)
        return entry

    def recall(self, fingerprint: str, step: int, epsilon: float) -> float:
        """``e^{-ε Δt}`` if these exact bytes were emitted, else 0."""
        entry = self.emissions.get(fingerprint)
        if entry is None:
            return 0.0
        gap = max(int(step) - entry.step, 0)
        return float(math.exp(-max(float(epsilon), 0.0) * gap))

    def __len__(self) -> int:
        return len(self.emissions)


@dataclass(frozen=True)
class Moment:
    """One heartbeat of self-observation: what was heard, and what was concluded.

    ``origin`` is the *truth*, kept for the report and never used to decide.
    ``judged`` is what the organism concluded at the time; the difference between
    them is the only honest measure of whether it knows itself.
    """

    step: int
    source: str
    origin: str
    resonance: float
    reality_score: float
    recall: float
    ratio: float
    peak: float
    gate: bool
    judged_world: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "source": self.source,
            "origin": self.origin,
            "resonance": round(float(self.resonance), 6),
            "reality_score": round(float(self.reality_score), 6),
            "recall": round(float(self.recall), 6),
            "ratio": round(float(self.ratio), 4),
            "peak": round(float(self.peak), 4),
            "gate": bool(self.gate),
            "judged_world": bool(self.judged_world),
        }


def reality_score(
    ratio: float,
    recall: float = 0.0,
    prior: float = WORLD_PRIOR,
) -> float:
    """The headline number: a calibrated posterior, discounted by what was emitted.

    Args:
        ratio: ``ℓ_world − ℓ_self`` in nats.
        recall: ``r`` from the ledger, in ``[0, 1]``; 1 means "these exact bytes
            are mine, emitted just now".
        prior: ``P(world)``, declared. 0.5 gives the likelihood ratio unadorned.

    Returns:
        ``(1 - recall) · sigmoid(ratio + logit(prior))``, in ``[0, 1)``.

    The two factors are not interchangeable and should not be blended more subtly:
    the first is evidence, the second is *knowledge of one's own act*. A dream the
    organism has forgotten scores exactly like a stranger with the same features —
    which is what a phantom is.
    """
    posterior = sigmoid(float(ratio) + logit(prior))
    return float(max(0.0, min(1.0 - 1e-12, (1.0 - float(recall)) * posterior)))


def _auc(world: Sequence[float], dream: Sequence[float]) -> float:
    from agent.physics.homeostasis import auc

    return auc(world, dream)


@dataclass
class SelfModel:
    """The organism observing itself: the metric, the prior, the ledger.

    Built by :meth:`awaken` from a labelled infancy (:func:`run_life` with the true
    provenance), after which it lives without labels: every fixation is judged by
    its own score and the models learn from that judgement. A wrong judgement
    therefore poisons the model — which is measurable, and is why
    :meth:`summary` reports ``misjudged`` next to the means instead of in a
    footnote.
    """

    parameters: Parameters
    posture: Posture
    ledger: Ledger = field(default_factory=Ledger)
    prior: float = WORLD_PRIOR
    channels: int = 8
    step: int = 0
    trace: List[Moment] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @classmethod
    def awaken(
        cls,
        episodes: Sequence[Any],
        parameters: Parameters,
        prior: float = WORLD_PRIOR,
        channels: Optional[int] = None,
    ) -> "SelfModel":
        """Fit both models on a labelled infancy, then hand back a living mirror."""
        life = run_life(episodes, parameters)
        resolved = (
            int(channels)
            if channels is not None
            else int(np.asarray(episodes[0].readings[0].x).size)
        )
        return cls(
            parameters=parameters,
            posture=life.posture,
            prior=float(prior),
            channels=resolved,
        )

    # -- the two things a self-model does ---------------------------------- #

    def emit(self, perception: Any, label: str = "") -> Emission:
        """Record an emission *at the moment of making it*: certain, and ageing."""
        return self.ledger.record(perception, self.step, label)

    def observe(self, episode: Any, learn: bool = True) -> List[Moment]:
        """Live through one episode, judging each fixation before learning from it."""
        moments: List[Moment] = []
        threshold = self.parameters.threshold(self.channels)
        for reading in episode.readings:
            vector = self.posture.features(reading)
            ratio = self.posture.log_ratio(vector)
            resonance = internal_resonance(ratio)
            recall = self.ledger.recall(
                episode.fingerprint, self.step, self.parameters.epsilon
            )
            score = reality_score(ratio, recall, self.prior)
            _, peak = sensorium.core_shape(reading)
            judged_world = score >= WORLD_JUDGEMENT
            moments.append(
                Moment(
                    step=self.step,
                    source=episode.source,
                    origin=episode.origin,
                    resonance=float(resonance),
                    reality_score=float(score),
                    recall=float(recall),
                    ratio=float(ratio),
                    peak=float(peak),
                    gate=bool(peak > threshold),
                    judged_world=bool(judged_world),
                )
            )
            if learn:
                # Route by judgement, not by truth: this is the loop that can
                # poison itself, and the ledger is what keeps it recoverable.
                target = "world" if judged_world else "self"
                self.posture.write(
                    vector, target, self.parameters.lam, self.parameters.epsilon
                )
            self.step += 1
        self.trace.extend(moments)
        return moments

    # -- what it can say about itself ------------------------------------- #

    def history(self) -> Dict[str, np.ndarray]:
        """The trace as arrays, in step order, for plotting and for tests."""
        return {
            "step": np.array([m.step for m in self.trace], dtype=int),
            "resonance": np.array([m.resonance for m in self.trace], dtype=float),
            "reality_score": np.array(
                [m.reality_score for m in self.trace], dtype=float
            ),
            "recall": np.array([m.recall for m in self.trace], dtype=float),
            "ratio": np.array([m.ratio for m in self.trace], dtype=float),
            "origin": np.array([m.origin for m in self.trace], dtype=object),
            "source": np.array([m.source for m in self.trace], dtype=object),
            "judged_world": np.array([m.judged_world for m in self.trace], dtype=bool),
        }

    def piles(self, field_name: str = "reality_score") -> Tuple[np.ndarray, np.ndarray]:
        """``(world, self)`` values of one trace field."""
        world = np.array(
            [getattr(m, field_name) for m in self.trace if m.origin == "world"],
            dtype=float,
        )
        dream = np.array(
            [getattr(m, field_name) for m in self.trace if m.origin != "world"],
            dtype=float,
        )
        return world, dream

    def summary(self) -> Dict[str, Any]:
        """Everything the mirror can say, with the arms of the decision separated."""
        world, dream = self.piles("reality_score")
        world_rho, dream_rho = self.piles("resonance")
        blind_world = np.array(
            [
                sigmoid(m.ratio + logit(self.prior))
                for m in self.trace
                if m.origin == "world"
            ],
            dtype=float,
        )
        blind_dream = np.array(
            [
                sigmoid(m.ratio + logit(self.prior))
                for m in self.trace
                if m.origin != "world"
            ],
            dtype=float,
        )
        per_source: Dict[str, Dict[str, Any]] = {}
        for name in sorted({m.source for m in self.trace}):
            rows = [m for m in self.trace if m.source == name]
            scores = np.array([m.reality_score for m in rows], dtype=float)
            per_source[name] = {
                "origin": rows[0].origin,
                "fixations": len(rows),
                "reality_score": float(scores.mean()),
                "resonance": float(np.mean([m.resonance for m in rows])),
                "recall": float(np.mean([m.recall for m in rows])),
            }
        misjudged = sum(
            1
            for m in self.trace
            if (m.reality_score >= WORLD_JUDGEMENT) != (m.origin == "world")
        )
        return {
            "steps": len(self.trace),
            "parameters": self.parameters.as_dict(),
            "prior": float(self.prior),
            "world": {
                "fixations": int(world.size),
                "reality_score": float(world.mean()) if world.size else float("nan"),
                "median": float(np.median(world)) if world.size else float("nan"),
                "resonance": float(world_rho.mean()) if world_rho.size else float("nan"),
            },
            "self": {
                "fixations": int(dream.size),
                "reality_score": float(dream.mean()) if dream.size else float("nan"),
                "median": float(np.median(dream)) if dream.size else float("nan"),
                "resonance": float(dream_rho.mean()) if dream_rho.size else float("nan"),
            },
            "auc_reality_score": float(_auc(world, dream)),
            "auc_resonance": float(_auc(world_rho, dream_rho)),
            "comparator_only": {
                "world": float(blind_world.mean()) if blind_world.size else float("nan"),
                "self": float(blind_dream.mean()) if blind_dream.size else float("nan"),
                "auc": float(_auc(blind_world, blind_dream)),
            },
            "ledger": len(self.ledger),
            "misjudged": int(misjudged),
            "misjudged_rate": float(misjudged / max(len(self.trace), 1)),
            "per_source": per_source,
            "warnings": list(self.warnings),
        }

    def acceptance(
        self, world_floor: float = 0.7, self_ceiling: float = 0.3
    ) -> Dict[str, Any]:
        """The hard rule: after a life, is the world above the floor, the rest below?

        Declared before any run, in the words it is judged by: *if after the
        integration the mean score for real data is not above 0.7, and for the
        organism's own simulations not below 0.3, then it overfitted or stalled.*
        """
        world, dream = self.piles("reality_score")
        world_mean = float(world.mean()) if world.size else float("nan")
        self_mean = float(dream.mean()) if dream.size else float("nan")
        return {
            "steps": len(self.trace),
            "world_mean": world_mean,
            "self_mean": self_mean,
            "world_floor": float(world_floor),
            "self_ceiling": float(self_ceiling),
            "world_ok": bool(np.isfinite(world_mean) and world_mean > world_floor),
            "self_ok": bool(np.isfinite(self_mean) and self_mean < self_ceiling),
            "passed": bool(
                np.isfinite(world_mean)
                and np.isfinite(self_mean)
                and world_mean > world_floor
                and self_mean < self_ceiling
            ),
        }

    def phantom(
        self, waits: Sequence[int] = (0, 1, 2, 5, 10, 20, 40, 80)
    ) -> List[Dict[str, float]]:
        """What its own dreams score once it has forgotten making them.

        Recall decays as ``e^{-εΔt}``, so the score of a dream rises from zero
        toward the comparator's opinion as the wait grows. The prediction is sharp
        and falsifiable: at ``Δt ≈ 1/ε`` the mean score of the organism's own
        output should cross :data:`WORLD_JUDGEMENT`. A model that cannot produce
        this crossing has not implemented forgetting at all.
        """
        world_ratio = [m.ratio for m in self.trace if m.origin == "world"]
        dream_ratio = [m.ratio for m in self.trace if m.origin != "world"]
        rows: List[Dict[str, float]] = []
        for wait in waits:
            recall = math.exp(-max(self.parameters.epsilon, 0.0) * float(wait))
            dream_scores = [
                (1.0 - recall) * sigmoid(r + logit(self.prior)) for r in dream_ratio
            ]
            world_scores = [
                (1.0 - recall) * sigmoid(r + logit(self.prior)) for r in world_ratio
            ]
            rows.append(
                {
                    "wait": float(wait),
                    "recall": float(recall),
                    "self": float(np.mean(dream_scores)) if dream_scores else float("nan"),
                    "world": float(np.mean(world_scores)) if world_scores else float("nan"),
                }
            )
        return rows

    def blind(self) -> "SelfModel":
        """The same trace, judged without the ledger: the honest limit, in one call.

        Used by the demonstration to answer "how much of this is memory?" by
        rebuilding the scores with ``recall = 0`` and leaving everything else alone.
        """
        clone = SelfModel(
            parameters=self.parameters,
            posture=self.posture,
            ledger=Ledger(),
            prior=self.prior,
            channels=self.channels,
            step=self.step,
        )
        clone.trace = [
            Moment(
                step=m.step,
                source=m.source,
                origin=m.origin,
                resonance=m.resonance,
                reality_score=reality_score(m.ratio, 0.0, self.prior),
                recall=0.0,
                ratio=m.ratio,
                peak=m.peak,
                gate=m.gate,
                judged_world=reality_score(m.ratio, 0.0, self.prior) >= WORLD_JUDGEMENT,
            )
            for m in self.trace
        ]
        return clone


def prior_sweep(
    model: SelfModel, priors: Sequence[float] = (0.5, 0.6, 2.0 / 3.0, 0.75, 0.85)
) -> List[Dict[str, float]]:
    """What the declared prior costs: the summary at several ``P(world)``.

    The prior is the one number in the decision that is a *choice*. Printing the
    whole curve instead of the chosen point is the difference between a parameter
    and a fiddle.
    """
    rows: List[Dict[str, float]] = []
    for prior in priors:
        world = np.array(
            [reality_score(m.ratio, m.recall, prior) for m in model.trace
             if m.origin == "world"],
            dtype=float,
        )
        dream = np.array(
            [reality_score(m.ratio, m.recall, prior) for m in model.trace
             if m.origin != "world"],
            dtype=float,
        )
        rows.append(
            {
                "prior": float(prior),
                "world": float(world.mean()) if world.size else float("nan"),
                "self": float(dream.mean()) if dream.size else float("nan"),
                "auc": float(_auc(world, dream)),
            }
        )
    return rows


__all__ = [
    "Emission",
    "Ledger",
    "Moment",
    "SelfModel",
    "WORLD_JUDGEMENT",
    "WORLD_PRIOR",
    "logit",
    "prior_sweep",
    "reality_score",
    "sigmoid",
]
