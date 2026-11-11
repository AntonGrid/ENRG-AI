"""The learned policy: what the hunter's verdict is worth, in one probability.

The engine's rule is a sentence — *buy every panic above a confidence floor* — and the sweep
measured what that sentence is worth: 23.5 % of trades won, and the losses were not spread
evenly (risk-off was significant at t = −3.20, risk-on was not significant at all). A rule
with one threshold cannot say "this panic, not that one", so this file learns the missing
part: a calibrated probability that an entry pays more than the venue keeps.

Why logistic regression, and not something larger
=================================================

The dataset is about a thousand panics over five years and the label is noisy by nature. A
model with more capacity than that fits the noise and reports a triumph in-sample — the
first sweep was that mistake with fewer parameters. A linear model on fifteen standardised
features has fifteen numbers a reader can argue with, and :meth:`LogisticPolicy.contributions`
prints them per bar, so "why did the book buy this bar" has an answer that is not "the
network liked it".

The second reason is architectural: ``agent.physics.genome`` freezes the detector's own
mathematics. The policy is an *organ* — it may learn, be replaced, be switched off — while
the DNA stays provably untouched.

What "calibrated" means here
============================

``fit`` learns weights; ``calibrate`` fits two more numbers that turn the raw logit into a
frequency. That step matters because a panic that pays is the minority: a model fitted on
rare events is confident in the wrong places. Without calibration the policy's ``0.3`` means
nothing, and a threshold on nothing is a knob to overfit. With it, ``0.3`` says "three in
ten bars that looked like this paid", which is a sentence an operator can hold to.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agent.backtest.features import FEATURES, feature_vector

__all__ = [
    "DEFAULT_L2",
    "LogisticPolicy",
    "load_policy",
]

#: Regularisation strength. Large on purpose: fifteen correlated features on a thousand
#: noisy labels is a setting where the ridge term is doing the real work.
DEFAULT_L2 = 1.0


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """A numerically quiet logistic function — ``exp`` never sees a large positive number."""
    out = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    out[~positive] = exponent / (1.0 + exponent)
    return out


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


class LogisticPolicy:
    """A fitted model: what a bar looks like, and how often such a bar paid.

    The object is a value, serialisable and comparable: two policies with the same numbers
    answer the same on the same bar, and :meth:`to_dict` is the whole of it. Nothing here
    reaches for a file, a clock or a venue — the caller owns those, which is what makes a
    policy testable on a hand-built series and shippable as signed weights.
    """

    def __init__(
        self,
        features: Sequence[str] = FEATURES,
        *,
        mean: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        weights: Optional[Sequence[float]] = None,
        bias: float = 0.0,
        calibration: Tuple[float, float] = (1.0, 0.0),
        threshold: float = 0.5,
        horizon: int = 3,
        cost: float = 0.004,
        samples: int = 0,
        positives: int = 0,
    ) -> None:
        self.features: Tuple[str, ...] = tuple(str(name) for name in features)
        size = len(self.features)
        self.mean = np.asarray(mean if mean is not None else [0.0] * size, dtype=float)
        self.scale = np.asarray(scale if scale is not None else [1.0] * size, dtype=float)
        self.weights = np.asarray(
            weights if weights is not None else [0.0] * size, dtype=float
        )
        for name, value in (
            ("mean", self.mean),
            ("scale", self.scale),
            ("weights", self.weights),
        ):
            if value.shape != (size,):
                raise ValueError(
                    f"a policy of {size} features cannot hold {name} of shape {value.shape}"
                )
        self.bias = float(bias)
        self.calibration = (float(calibration[0]), float(calibration[1]))
        self.threshold = float(threshold)
        self.horizon = int(horizon)
        self.cost = float(cost)
        self.samples = int(samples)
        self.positives = int(positives)

    # -- reading ------------------------------------------------------------- #

    @property
    def base_rate(self) -> float:
        """How often an entry paid in the record this policy was fitted on."""
        return (float(self.positives) / float(self.samples)) if self.samples else 0.0

    @property
    def temperature(self) -> float:
        return float(self.calibration[0])

    def _standard(self, features: Mapping[str, float]) -> np.ndarray:
        vector = np.asarray(feature_vector(features, self.features), dtype=float)
        safe = np.where(np.abs(self.scale) > 1e-12, self.scale, 1.0)
        return (vector - self.mean) / safe

    def logit(self, features: Mapping[str, float]) -> float:
        """The raw score, before calibration — the number the weights actually produce."""
        return float(np.dot(self.weights, self._standard(features)) + self.bias)

    def score(self, features: Mapping[str, float]) -> float:
        """The calibrated probability that an entry here pays more than the venue keeps."""
        a, b = self.calibration
        return float(_sigmoid(np.asarray([a * self.logit(features) + b]))[0])

    def decision(self, features: Mapping[str, float]) -> bool:
        """Whether the probability clears this policy's own threshold."""
        return self.score(features) >= float(self.threshold)

    def contributions(self, features: Mapping[str, float]) -> Dict[str, float]:
        """Each feature's share of the score, in logits — the answer to "why this bar".

        A zero weight is a feature the fit found no use for, and it is printed as zero
        rather than hidden: an operator reading fifteen numbers learns which of them the
        record supports and which are decoration.
        """
        standard = self._standard(features)
        return {
            name: float(weight * value)
            for name, weight, value in zip(self.features, self.weights, standard)
        }

    def describe(self, features: Optional[Mapping[str, float]] = None, *, top: int = 5) -> str:
        """A one-line reading for a report: the score, then its loudest arguments."""
        if features is None:
            ranked = sorted(zip(self.features, self.weights), key=lambda kv: -abs(kv[1]))
            parts = [f"{name} {weight:+.3f}" for name, weight in ranked[:top]]
            return f"{len(self.features)} features, base rate {self.base_rate:.1%}: " + ", ".join(parts)
        shares = sorted(self.contributions(features).items(), key=lambda kv: -abs(kv[1]))
        parts = [f"{name} {share:+.3f}" for name, share in shares[:top]]
        return f"p={self.score(features):.3f} from " + ", ".join(parts)

    # -- writing ------------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "features": list(self.features),
            "mean": [round(float(value), 10) for value in self.mean],
            "scale": [round(float(value), 10) for value in self.scale],
            "weights": [round(float(value), 10) for value in self.weights],
            "bias": round(float(self.bias), 10),
            "calibration": [round(float(value), 10) for value in self.calibration],
            "threshold": round(float(self.threshold), 6),
            "horizon": int(self.horizon),
            "cost": round(float(self.cost), 8),
            "samples": int(self.samples),
            "positives": int(self.positives),
            "base_rate": round(float(self.base_rate), 6),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LogisticPolicy":
        calibration = data.get("calibration") or (1.0, 0.0)
        return cls(
            features=data.get("features") or FEATURES,
            mean=data.get("mean"),
            scale=data.get("scale"),
            weights=data.get("weights"),
            bias=float(data.get("bias", 0.0)),
            calibration=(float(calibration[0]), float(calibration[1])),
            threshold=float(data.get("threshold", 0.5)),
            horizon=int(data.get("horizon", 3)),
            cost=float(data.get("cost", 0.004)),
            samples=int(data.get("samples", 0)),
            positives=int(data.get("positives", 0)),
        )

    def save(self, path: Any) -> Path:
        """Write the weights as JSON — the same document ``agent.fed`` can sign."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: Any) -> "LogisticPolicy":
        """Read the weights back — a saved policy is a policy, not a description of one."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- fitting ------------------------------------------------------------- #

    @classmethod
    def fit(
        cls,
        samples: Iterable[Tuple[Mapping[str, float], Optional[float]]],
        *,
        features: Sequence[str] = FEATURES,
        l2: float = DEFAULT_L2,
        iterations: int = 60,
        tolerance: float = 1e-8,
        threshold: float = 0.5,
        horizon: int = 3,
        cost: float = 0.004,
    ) -> "LogisticPolicy":
        """Fit the standardiser and the weights by Newton steps (IRLS), then calibrate.

        The labels are unbalanced by construction — a panic that pays is the minority — so
        the fit balances the two classes while learning, and :meth:`_platt` undoes the
        distortion afterwards. The order matters: balancing first gives every positive the
        attention it deserves, calibration second gives back the honest frequency.
        """
        names = tuple(str(name) for name in features)
        rows: List[List[float]] = []
        labels: List[float] = []
        for case, label in samples:
            if label is None:
                continue
            rows.append(feature_vector(case, names))
            labels.append(float(label))
        n = len(labels)
        if n == 0:
            raise ValueError("a policy cannot be fitted on zero samples")
        X = np.asarray(rows, dtype=float)
        y = np.asarray(labels, dtype=float)
        positives = int(np.sum(y > 0.5))
        if positives == 0 or positives == n:
            raise ValueError(
                f"every one of {n} samples carries the same label ({positives} positive): "
                "there is nothing here to tell apart"
            )
        mean = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        Z = (X - mean) / scale

        # Balanced weights: the minority class carries the weight it lacks in count.
        positive_weight = 0.5 * n / positives
        negative_weight = 0.5 * n / (n - positives)
        sample_weight = np.where(y > 0.5, positive_weight, negative_weight)

        design = np.hstack([Z, np.ones((n, 1))])
        beta = np.zeros(design.shape[1])
        beta[-1] = _logit(float(np.average(y, weights=sample_weight)))
        ridge = np.eye(design.shape[1]) * float(l2)
        ridge[-1, -1] = 0.0  # the intercept is not shrunk
        for _ in range(int(iterations)):
            probability = _sigmoid(design @ beta)
            gradient = design.T @ (sample_weight * (probability - y)) / n + ridge @ beta
            curvature = sample_weight * probability * (1.0 - probability)
            hessian = (design * curvature[:, None]).T @ design / n + ridge
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:  # pragma: no cover - the ridge keeps it invertible
                break
            beta = beta - step
            if float(np.max(np.abs(step))) < float(tolerance):
                break

        policy = cls(
            features=names,
            mean=mean,
            scale=scale,
            weights=beta[:-1],
            bias=float(beta[-1]),
            calibration=(1.0, 0.0),
            threshold=threshold,
            horizon=horizon,
            cost=cost,
            samples=n,
            positives=positives,
        )
        policy.calibration = policy._platt(X, y)
        return policy

    def _platt(self, X: np.ndarray, y: np.ndarray, *, iterations: int = 40) -> Tuple[float, float]:
        """Two numbers that turn the raw score into a frequency.

        Fitted on the same rows the weights saw, which is the honest limit of this method
        and is stated rather than hidden: a policy adopted through
        :mod:`agent.backtest.walkforward` is calibrated on its training block and then
        *measured* on bars it has never seen, so an over-confident calibration shows up as a
        worse score, not as a better one.
        """
        raw = np.asarray(
            [self.logit(dict(zip(self.features, row))) for row in X], dtype=float
        )
        a, b = 1.0, 0.0
        for _ in range(int(iterations)):
            p = _sigmoid(a * raw + b)
            error = p - y
            curvature = p * (1.0 - p) + 1e-9
            g_a = float(np.sum(error * raw))
            g_b = float(np.sum(error))
            h_aa = float(np.sum(curvature * raw * raw)) + 1e-6
            h_ab = float(np.sum(curvature * raw))
            h_bb = float(np.sum(curvature)) + 1e-6
            determinant = h_aa * h_bb - h_ab * h_ab
            if abs(determinant) < 1e-12:
                break
            step_a = (h_bb * g_a - h_ab * g_b) / determinant
            step_b = (h_aa * g_b - h_ab * g_a) / determinant
            a -= step_a
            b -= step_b
            if max(abs(step_a), abs(step_b)) < 1e-9:
                break
        # A negative temperature would invert the ranking; the ridge of the fit makes it
        # unlikely, and this floor makes it impossible.
        return (max(float(a), 1e-3), float(b))


def load_policy(path: Any) -> LogisticPolicy:
    """Read a saved policy — a named function so a caller need not import the class."""
    return LogisticPolicy.load(path)



