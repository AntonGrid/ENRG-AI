"""Statement #87 — the sterility index: is this relief too perfect to be lived?

The book states a detector, not a test of the world. A memory relief that follows
the theoretical decay ``R₀·e^(−εt)`` *too* closely is a sign of synthetic
collusion, because a lived medium always carries residuals: what was written has
a 1 % miss in it, and the miss moves. Two readings decide:

* Anderson–Darling ``A²`` of the normalised residuals
  ``δ_t = relief(t) − R₀·e^(−εt)``. A lived relief leaves residuals that are
  not normal (A² of order one), a formula leaves none at all (A² = 0 — caught).
  A² = 0 is therefore a *failure* here, not a perfect score;
* ``CV(relief) = std/mean``. The written 1 % miss puts a floor under the spread,
  so a coefficient below 0.5 % is below the physical floor of this record.

Fail — ``A² < 0.1`` **or** ``CV < 0.005`` — and the statement is marked
``synthetic_collusion_detected``. Both thresholds are named before the run in
``entrain/julia/kuramoto/consistency.jl`` (``STERILE_A2`` / ``STERILE_CV``) and
are policy, not physics: they are carried through here, never tuned.

The module also ports Statement #88 — the LogSumExp smoothing of the ``λ*``
operator and Banach's contraction estimate — so the oracle reasons about the
same operator the book proves theorems about. The residual statistics are
compared with the Julia run in ``tests/test_physics_sterility.py``.

One deliberate choice: Julia's ``Base`` has no ``erf``, and the book uses the
Abramowitz–Stegun approximation 7.1.26 (error ≤ 1.5e-07). The port computes Φ
the same way on purpose — the point of a port is that both sides read the same
number off the same array, so ``A²`` agrees with Julia to ~1e-11 (the remainder
is the summation order: numpy sums pairwise, the book's loop sums in order).
Accuracy is tested separately against ``math.erf``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

#: Thresholds of Statement #87, copied from ``consistency.jl`` (named there
#: before the run). A relief below either one is too close to the formula.
STERILE_A2 = 0.1
STERILE_CV = 0.005

#: Residuals below this count are not a sample: the normal quantiles they are
#: compared against would be read off too few points.
STERILE_MIN = 16

#: Reason carried by a failed statement — the same string the Axis-core Policy
#: Engine reports (``synthetic_collusion_detected``).
FAILED = "synthetic_collusion_detected"


def normal_cdf(z: float) -> float:
    """Φ(z) — the standard normal distribution function.

    Abramowitz–Stegun 7.1.26, the same formula ``consistency.jl`` uses (Julia's
    ``Base`` has no ``erf``): error ≤ 1.5e-07, which is four orders of magnitude
    below the 0.1 threshold ``A²`` is compared with.
    """
    x = abs(float(z)) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    erf = 1.0 - poly * math.exp(-x * x)
    return 0.5 * (1.0 + (erf if z >= 0 else -erf))


def anderson_darling(values) -> float:
    """Anderson–Darling ``A²`` of a sample against normality.

    Mean and spread are estimated from the sample itself, hence the Stephens
    correction ``(1 + 0.75/n + 2.25/n²)``: without it the criterion is stricter
    than the one named in the book. ``0.0`` means the residuals are all equal —
    an exact match with the formula, which is the failure mode Statement #87
    looks for, not a perfect score.
    """
    x = np.asarray(values, dtype=float).ravel()
    n = int(x.size)
    if n < 8:
        return math.inf
    sigma = float(np.std(x, ddof=1)) if n > 1 else 0.0
    if sigma <= 0.0:
        return 0.0
    z = np.sort((x - float(np.mean(x))) / sigma)
    lower = np.clip([normal_cdf(v) for v in z], 1e-12, 1.0 - 1e-12)
    upper = np.clip([normal_cdf(v) for v in z[::-1]], 1e-12, 1.0 - 1e-12)
    weights = 2.0 * np.arange(1, n + 1) - 1.0
    total = float(np.sum(weights * (np.log(lower) + np.log1p(-upper))))
    a2 = -n - total / n
    return float(a2 * (1.0 + 0.75 / n + 2.25 / (n * n)))


def variation(values) -> float:
    """Coefficient of variation ``std/mean``; ``inf`` when the mean is zero."""
    x = np.asarray(values, dtype=float).ravel()
    mean = float(np.mean(x)) if x.size else 0.0
    if mean == 0.0:
        return math.inf
    return float(np.std(x, ddof=1) / abs(mean)) if x.size > 1 else math.inf


@dataclass
class SterilityVerdict:
    """What Statement #87 says about one relief trace.

    ``verdict`` is ``passed`` (a lived relief), ``failed`` (too close to the
    formula — ``synthetic_collusion_detected``) or ``short`` (fewer residuals
    than :data:`STERILE_MIN`, so there is nothing to decide on).
    """

    a2: float
    cv: float
    r0: float
    verdict: str
    reason: str = "ok"
    points: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def synthetic_collusion(self) -> bool:
        """Duck-typed flag the Policy Engine reads (``evaluate_sterility``)."""
        return self.verdict == "failed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "a2": self.a2,
            "cv": self.cv,
            "r0": self.r0,
            "verdict": self.verdict,
            "reason": self.reason,
            "points": self.points,
            **self.details,
        }


def sterility_index(
    timestamps,
    relief,
    memory: float,
    level: Optional[float] = None,
) -> SterilityVerdict:
    """Statement #87 for one trace: ``A²`` of the residuals and ``CV(relief)``.

    ``timestamps`` — seconds since the event stopped, ``relief`` — the relief
    read at those times (``relief_place(δ, place)[1]`` in the book's terms),
    ``memory`` — ε of the medium, ``level`` — R₀; when omitted it is the first
    reading, which is what "the relief at the moment the event stopped" means.

    The residuals are ``δ_t = relief(t) − R₀·e^(−εt)``. They are normalised
    inside :func:`anderson_darling` (mean and spread of the sample), which is
    the same estimate the Julia statement makes.
    """
    times = np.asarray(timestamps, dtype=float).ravel()
    trace = np.asarray(relief, dtype=float).ravel()
    if times.size != trace.size:
        raise ValueError(
            "timestamps and relief must have the same length, "
            f"got {times.size} and {trace.size}"
        )
    n = int(trace.size)
    if n < STERILE_MIN:
        return SterilityVerdict(
            a2=math.inf,
            cv=math.inf,
            r0=math.nan,
            verdict="short",
            reason="insufficient_residuals",
            points=n,
        )
    r0 = float(trace[0] if level is None else level)
    residuals = trace - r0 * np.exp(-float(memory) * times)
    a2 = anderson_darling(residuals)
    cv = variation(trace)
    failed = a2 < STERILE_A2 or cv < STERILE_CV
    return SterilityVerdict(
        a2=a2,
        cv=cv,
        r0=r0,
        verdict="failed" if failed else "passed",
        reason=FAILED if failed else "ok",
        points=n,
        details={
            "memory": float(memory),
            "a2_threshold": STERILE_A2,
            "cv_threshold": STERILE_CV,
        },
    )


# ── Statement #88: the smoothed operator and Banach's estimate ───────────────


def smooth_max(values: Iterable[float], beta: float) -> float:
    """LogSumExp ``(1/β)·ln Σ e^(β·h_k)`` — the smoothing of ``max``.

    ``β → ∞`` recovers the maximum; a finite ``β`` makes the operator smooth in
    λ while keeping the Lipschitz constant of ``max``: the shift by the largest
    value is what keeps ``exp`` from overflowing and does not change the number.
    """
    values = np.asarray(list(values), dtype=float).ravel()
    if beta <= 0.0:
        raise ValueError("beta must be positive: smoothing tends to max as beta → ∞")
    if values.size == 0:
        raise ValueError("nothing to smooth: empty set")
    top = float(np.max(values))
    if not math.isfinite(top):
        return top
    return top + float(np.log(np.sum(np.exp(beta * (values - top))))) / beta


@dataclass
class BanachResult:
    """The limit of an iteration and the path that led to it."""

    lam: float
    path: List[float]
    converged: bool

    @property
    def rate(self) -> float:
        """Geometric ratio of the last two steps (an estimate of q)."""
        if len(self.path) < 3:
            return math.nan
        last = abs(self.path[-1] - self.path[-2])
        prev = abs(self.path[-2] - self.path[-3])
        if prev == 0.0:
            return math.nan
        return last / prev


def banach_iteration(
    operator: Callable[[float], float],
    start: float,
    tol: float = 1e-12,
    steps: int = 200,
) -> BanachResult:
    """``λ_{n+1} = T(λ_n)`` until ``|Δλ| ≤ tol``.

    Convergence is checked, not assumed: if ``T`` is a contraction with a
    constant ``q < 1``, Banach's theorem gives existence, uniqueness and
    ``|λ_n − λ*| ≤ qⁿ·|λ₀ − λ*|``. The caller can read ``rate`` next to the
    limit — without it "it converged" is an observation, not a proof.
    """
    lam = float(start)
    path = [lam]
    for _ in range(int(steps)):
        nxt = float(operator(lam))
        path.append(nxt)
        if abs(nxt - lam) <= tol:
            return BanachResult(lam=nxt, path=path, converged=True)
        lam = nxt
    return BanachResult(lam=lam, path=path, converged=False)


# ── the statement in the oracle's language ───────────────────────────────────


@dataclass
class SterilitySignal:
    """Statement #87 shaped for ``PolicyEngine.evaluate_sterility``.

    The engine duck-types ``synthetic_collusion``, ``a2`` and ``cv`` — the same
    way ``evaluate_ai_anomaly`` reads ``is_anomaly`` and ``score`` — so this
    layer adds an observation and never a decision: the thresholds belong to
    the policy, and the policy is the operator's.
    """

    device_id: str
    a2: float
    cv: float
    synthetic_collusion: bool
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_verdict(cls, device_id: str, verdict: SterilityVerdict) -> "SterilitySignal":
        return cls(
            device_id=device_id,
            a2=verdict.a2,
            cv=verdict.cv,
            synthetic_collusion=verdict.synthetic_collusion,
            details={
                "verdict": verdict.verdict,
                "reason": verdict.reason,
                "r0": verdict.r0,
                "points": verdict.points,
                **verdict.details,
            },
        )


