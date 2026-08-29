"""Market data model — "price × time" structures for forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from agent.digital_train.data import build_matrix
from agent.forecast.model import HoltTrend

#: Normal quantile for the 80% prediction interval (10% each tail).
_Z_90 = 1.2816


@dataclass
class PricePoint:
    """One timestamped market observation (prices in USD/kWh)."""

    ts: str
    prices: Dict[str, float] = field(default_factory=dict)


def price_matrix(points: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """Chronological price matrix from ``{ts, prices}`` dicts.

    Reuses the domain-agnostic ``build_matrix`` (same forward-fill semantics),
    normalizing the market shape (``prices``) to the training shape
    (``metrics``).
    """
    normalized = [
        {"ts": p["ts"], "metrics": dict(p.get("prices") or {})}
        for p in points
    ]
    return build_matrix(normalized)


def _forecast_column(values: np.ndarray, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Holt's trend on one price column; return (point, half-width)."""
    y = np.asarray(values, dtype=float)
    model = HoltTrend().fit(y)
    point = np.maximum(model.forecast(steps), 0.0)
    half = model.interval(steps, quantile=_Z_90)
    return point, half


def forecast_price_series(series: np.ndarray, steps: int = 1) -> np.ndarray:
    """Point forecast (Holt linear trend) per column for ``steps`` rows ahead.

    Replaces the former last-value placeholder with a real state-space trend
    model (see ``agent.forecast.model.HoltTrend``). The output shape matches
    the old contract: ``(steps, n_columns)``.
    """
    series = np.asarray(series, dtype=float)
    if series.shape[0] == 0:
        return series
    if series.shape[0] == 1:
        return np.tile(series[-1], (max(steps, 0), 1))
    cols = [series[:, d] for d in range(series.shape[1])]
    points = [_forecast_column(c, steps)[0] for c in cols]
    return np.column_stack(points) if points else np.zeros((0, 0))


def forecast_price_with_intervals(
    series: np.ndarray,
    steps: int = 1,
) -> Dict[str, Any]:
    """Holt forecast per column with 80% prediction intervals.

    Returns, per column: point path (``steps``), low/high band (``steps``),
    and the fitted residual RMSE. Structured for the hybrid AI oracle
    (``agent.signals``) so consumers get uncertainty, not just a point.
    """
    series = np.asarray(series, dtype=float)
    out: Dict[str, Any] = {}
    for d in range(series.shape[1]):
        values = series[:, d]
        model = HoltTrend().fit(np.asarray(values, dtype=float))
        steps_n = max(int(steps), 1)
        point = np.maximum(model.forecast(steps_n), 0.0)
        half = model.interval(steps_n, quantile=_Z_90)
        out[d] = {
            "point": [round(float(v), 6) for v in point],
            "low": [round(float(v), 6) for v in np.maximum(point - half, 0.0)],
            "high": [round(float(v), 6) for v in point + half],
            "rmse": round(float(model.residual_rmse), 6),
        }
    return out
