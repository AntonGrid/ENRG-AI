"""Market data model — "price × time" structures for forecasting."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from agent.digital_train.data import build_matrix


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


def forecast_price_series(series: np.ndarray, steps: int = 1) -> np.ndarray:
    """Naive next-step forecast per column: last observed value.

    A placeholder for the real price-forecast model — the Recommender accepts
    an explicit ``price_forecast`` anyway, so callers can plug a better
    predictor (e.g. the digital ``TimeSeriesModel``) without changing the API.
    """
    if series.shape[0] == 0:
        return series
    last = series[-1]
    return np.tile(last, (steps, 1))
