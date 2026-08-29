"""Energy forecast model — lightweight state-space trend with intervals.

Fits Holt's linear trend (local level + trend) by one-step SSE grid search.
Numpy only, matching the ENRG-AI convention of dependency-light, serializable
models. Prediction intervals grow with ``sqrt(horizon)`` — the standard
approximation for the state-space ETS(A,A,N) forecast error variance — so
uncertainty widens sensibly with distance, and the lower bound is clipped at
zero (energy cannot be negative).

This is intentionally a solid baseline. For zero-shot deep forecasting
(TimesFM) or full time-series ML (aeon), see ``skills/`` and the docs of the
corresponding skills — the CLI writes the same CSV contract so the engine can
be swapped without changing callers.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from agent.forecast.energy import ProofSeries

#: Normal quantile for the 80% prediction interval (10% each tail).
_Z_90 = 1.2816
#: Normal quantile for the 95% interval (2.5% each tail).
_Z_95 = 1.96


class HoltTrend:
    """Holt's linear trend (additive): level + trend, two smoothing weights.

    ``alpha`` controls level smoothing, ``beta`` trend smoothing. Both are
    fitted by a small grid search that minimises one-step-ahead SSE.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.1) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.level = 0.0
        self.trend = 0.0
        self.residual_rmse: float = 0.0
        self._fitted = False

    def _sse(self, y: np.ndarray, alpha: float, beta: float) -> float:
        n = len(y)
        if n == 1:
            return 0.0
        level = float(y[0])
        trend = float(y[1]) - float(y[0]) if n > 1 else 0.0
        sse = 0.0
        for t in range(1, n):
            forecast = level + trend
            err = float(y[t]) - forecast
            sse += err * err
            prev_level = level
            level = alpha * float(y[t]) + (1.0 - alpha) * forecast
            trend = beta * (level - prev_level) + (1.0 - beta) * trend
        return sse

    def fit(self, y: np.ndarray) -> "HoltTrend":
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.size == 0:
            raise ValueError("cannot fit an empty series")
        if y.size == 1:
            self.level = float(y[0])
            self.trend = 0.0
            self.residual_rmse = 0.0
            self._fitted = True
            return self

        best = (self.alpha, self.beta, float("inf"))
        for alpha in np.linspace(0.05, 0.95, 10):
            for beta in np.linspace(0.01, 0.5, 6):
                sse = self._sse(y, float(alpha), float(beta))
                if sse < best[2]:
                    best = (float(alpha), float(beta), sse)
        self.alpha, self.beta, best_sse = best

        # Refit the chosen parameters to keep the final level/trend state.
        level = float(y[0])
        trend = float(y[1]) - float(y[0])
        for t in range(1, y.size):
            forecast = level + trend
            prev_level = level
            level = self.alpha * float(y[t]) + (1.0 - self.alpha) * forecast
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend
        self.level = level
        self.trend = trend
        self.residual_rmse = float(np.sqrt(best_sse / max(1, y.size - 1)))
        self._fitted = True
        return self

    def forecast(self, steps: int) -> np.ndarray:
        """Point forecast (mean path) for ``steps`` buckets ahead."""
        if not self._fitted:
            raise RuntimeError("fit() before forecast()")
        steps = max(int(steps), 0)
        if steps == 0:
            return np.array([], dtype=float)
        horizon = np.arange(1, steps + 1, dtype=float)
        return self.level + self.trend * horizon

    def in_sample_residuals(self, y: np.ndarray) -> np.ndarray:
        """One-step-ahead residuals over the fitted series (same recursion).

        Uses the *fitted* smoothing weights, so callers can build their own
        anomaly detector (MAD / thresholds) without re-fitting.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        n = len(y)
        if n < 2:
            return np.array([], dtype=float)
        level = float(y[0])
        trend = float(y[1]) - float(y[0])
        residuals: List[float] = []
        for t in range(1, n):
            forecast = level + trend
            residuals.append(float(y[t]) - forecast)
            prev_level = level
            level = self.alpha * float(y[t]) + (1.0 - self.alpha) * forecast
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend
        return np.array(residuals, dtype=float)

    def interval(self, steps: int, quantile: float = _Z_90) -> np.ndarray:
        """Half-width of the prediction interval at ``quantile`` z-score.

        Grows like ``sqrt(horizon)`` — the ETS(A,A,N) forecast error variance
        for a model with a trend component. Returns the half-width per step.
        """
        steps = max(int(steps), 0)
        if steps == 0:
            return np.array([], dtype=float)
        sigma = max(self.residual_rmse, 1e-9)
        horizon = np.arange(1, steps + 1, dtype=float)
        return quantile * sigma * np.sqrt(horizon)


@dataclass
class ForecastResult:
    """Forecast for a ProofSeries: point path plus interval per horizon."""

    bucket_minutes: int
    labels: List[str]  # ISO bucket starts, UTC (same as observed series labels)
    point_wh: np.ndarray
    low_wh: np.ndarray  # q10 (80% interval)
    high_wh: np.ndarray  # q90
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    source: str = "oracle"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_minutes": self.bucket_minutes,
            "generated_at": self.generated_at,
            "source": self.source,
            "horizon_steps": len(self.point_wh),
            "buckets": [
                {
                    "time": label,
                    "forecast_wh": round(float(p), 3),
                    "low_q10_wh": round(float(lo), 3),
                    "high_q90_wh": round(float(hi), 3),
                }
                for label, p, lo, hi in zip(
                    self.labels, self.point_wh, self.low_wh, self.high_wh
                )
            ],
            "meta": self.meta,
        }


def forecast_energy(
    series: ProofSeries,
    horizon_steps: int = 8,
    interval: str = "p80",
    model: Optional[HoltTrend] = None,
) -> ForecastResult:
    """Fit Holt's trend on the observed buckets and forecast ahead.

    ``series`` must be a regular grid (call ``forward_fill`` if the proof
    cadence left gaps). Intervals: ``p80`` → q10/q90, ``p95`` → q2.5/q97.5.
    """
    y = np.maximum(np.asarray(series.values, dtype=float), 0.0)
    model = model or HoltTrend()
    model.fit(y)

    steps = max(int(horizon_steps), 1)
    point = np.maximum(model.forecast(steps), 0.0)
    z = _Z_95 if interval == "p95" else _Z_90
    half = model.interval(steps, quantile=z)
    low = np.maximum(point - half, 0.0)
    high = point + half

    last_start = series.starts[-1]
    step_sec = series.bucket_minutes * 60
    labels = [
        (last_start + dt.timedelta(seconds=step_sec * (i + 1))).isoformat(
            timespec="minutes"
        )
        for i in range(steps)
    ]

    meta: Dict[str, Any] = {
        "model": "holt-linear-trend",
        "alpha": round(model.alpha, 4),
        "beta": round(model.beta, 4),
        "residual_rmse_wh": round(model.residual_rmse, 4),
        "observed_buckets": int(len(y)),
        "total_observed_wh": round(float(np.sum(y)), 3),
        "interval": interval,
        "units": "Wh",
    }
    return ForecastResult(
        bucket_minutes=series.bucket_minutes,
        labels=labels,
        point_wh=point,
        low_wh=low,
        high_wh=high,
        source=series.source,
        meta=meta,
    )

