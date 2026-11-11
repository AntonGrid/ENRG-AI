"""What the book could have seen on a bar, and not one value it could not.

The hunter reads two numbers a bar: the *decline* it offered a buyer (:func:`declines`)
and the turnover left behind. That is the whole of its world — honest, and blind. A bar
that fell 4 % on dried volume looks exactly like a bar that fell 4 % inside a crash, and a
stop-loss cannot tell them apart either. This module gives a policy the rest of the picture,
computed the same way for every bar: from the closes, highs, lows and turnovers *up to and
including* that bar.

The rule that makes the file trustworthy is one line long: every series here is a function
of ``rows[:index + 1]``. No forward fill from the future, no z-score over the whole sample,
no percentile that already knows how the story ends. A feature that needed tomorrow's close
would be a look-ahead with a friendly name, and the walk-forward run would then measure
nothing at all.

What is measured, and why each one earns its place
==================================================

``ret_1``, ``ret_3``, ``ret_7``
    The *signed* return. The hunter cannot see it — a decline is non-negative by
    construction, because a pool's APY cannot be negative — so a bar that rose is a zero to
    the hunter and a fact to this file.
``decline_1``, ``decline_7``
    The hunter's own currency, kept so the policy can be read next to it.
``atr_pct``, ``vol_pct``
    The bar's normal size (ATR over a fortnight) and where today's size stands against its
    own last ninety bars. A 2 % stop is a risk on a quiet pair and a coin flip on ``zecusdt``
    (ATR 9 %); the policy has to see which one it is standing in.
``close_sma20``, ``close_sma200``
    Trend at two speeds — the difference the sweep already measured between a market above
    and below its own annual mean.
``turnover_z``, ``drain``
    Participation: how unusual this bar's turnover is, and how the last week's median
    compares with the week before it. ``drain`` below one is the hunter's exodus, in a
    number the policy can weigh.
``range_pct``, ``close_pos``
    The bar's shape. A close at the low of a wide bar is a different event from a close at
    the high of the same bar, and a stop placed inside the bar cares about exactly that.
``dd_252``
    Distance below the year's high — how much of the fall has already happened.
``rsi14``
    The one bounded oscillator here: a ratio of gains to losses rather than a raw return.

Nothing in this file decides anything. It describes, and :mod:`agent.backtest.policy`
learns what the description is worth.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from agent.backtest.data import Candle

__all__ = [
    "FEATURES",
    "POLICY_FEATURES",
    "PRICE_FEATURES",
    "bar_features",
    "enrich",
    "feature_matrix",
    "feature_vector",
    "series_features",
]

#: The columns every feature row carries, in the order a model reads them. The order is part
#: of the contract: weights are serialised against it, so a name inserted in the middle would
#: silently re-label a saved model. New features go last.
FEATURES = (
    "ret_1",
    "ret_3",
    "ret_7",
    "decline_1",
    "decline_7",
    "atr_pct",
    "vol_pct",
    "close_sma20",
    "close_sma200",
    "turnover_z",
    "drain",
    "range_pct",
    "close_pos",
    "dd_252",
    "rsi14",
)

#: The subset computable from prices alone — a feed that publishes a close but no turnover.
PRICE_FEATURES = tuple(name for name in FEATURES if name not in ("turnover_z", "drain"))

#: What a policy may read on top of :data:`FEATURES`: the hunter's own confidence, the
#: market regime the bar sits in, and how long the pair has been quiet. They are appended
#: rather than merged because two of them come from outside this file — the detector's
#: verdict and a benchmark's trend — and a feature that cannot be computed from the series
#: alone must not pretend that it can.
POLICY_FEATURES = FEATURES + ("confidence", "regime", "bars_since_panic")

#: The measurements behind ``atr_pct`` and ``vol_pct``. Not knobs: moving them changes what
#: the word means, so they are constants and not settings.
ATR_WINDOW = 14
VOL_WINDOW = 90
DRAIN_WINDOW = 7


def _finite(value: Any, default: float = 0.0) -> float:
    """A finite float, or the default — a warm-up row is not a zero."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0

def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) if variance > 0.0 else 0.0


def _true_ranges(rows: Sequence[Candle]) -> List[float]:
    """Wilder's true range per bar: the gap counts, not only the bar's own height."""
    ranges: List[float] = []
    for index, row in enumerate(rows):
        high = _finite(row.high)
        low = _finite(row.low)
        if index == 0:
            ranges.append(max(high - low, 0.0))
            continue
        previous = _finite(rows[index - 1].close)
        ranges.append(max(high - low, abs(high - previous), abs(low - previous)))
    return ranges




def series_features(
    rows: Sequence[Candle],
    *,
    atr_window: int = ATR_WINDOW,
    vol_window: int = VOL_WINDOW,
    drain_window: int = DRAIN_WINDOW,
) -> List[Dict[str, float]]:
    """One row of features per bar, each computed from the bars that precede it.

    The series is walked once, and every rolling window is a slice that *ends* at the bar
    being described. That is the whole look-ahead guarantee: bar ``i`` cannot read bar
    ``i + 1``, because no slice reaches it.
    """
    count = len(rows)
    if count == 0:
        return []
    closes = [_finite(row.close) for row in rows]
    highs = [_finite(row.high) for row in rows]
    lows = [_finite(row.low) for row in rows]
    volumes = [_finite(row.volume) for row in rows]
    ranges = _true_ranges(rows)
    rsi = _rsi(closes)

    atr_pct: List[float] = [0.0] * count
    for index in range(count):
        window = ranges[max(0, index - atr_window + 1) : index + 1]
        if window and closes[index] > 0.0:
            atr_pct[index] = 100.0 * _mean(window) / closes[index]

    out: List[Dict[str, float]] = []
    for index in range(count):
        close = closes[index]

        def ret(bars: int) -> float:
            if index - bars < 0 or closes[index - bars] <= 0.0 or close <= 0.0:
                return 0.0
            return 100.0 * (close / closes[index - bars] - 1.0)

        def sma(bars: int) -> float:
            window = closes[max(0, index - bars + 1) : index + 1]
            average = _mean(window)
            if len(window) < bars or close <= 0.0 or average <= 0.0:
                return 0.0
            return 100.0 * (close / average - 1.0)

        vol_values = atr_pct[max(0, index - vol_window + 1) : index + 1]
        below = sum(1 for value in vol_values if value <= atr_pct[index])
        vol_pct = below / len(vol_values) if vol_values else 0.5

        turnover = volumes[max(0, index - 19) : index + 1]
        spread = _std(turnover)
        turnover_z = ((volumes[index] - _median(turnover)) / spread) if spread > 0.0 else 0.0

        recent = volumes[max(0, index - drain_window + 1) : index + 1]
        before = volumes[max(0, index - 2 * drain_window + 1) : max(0, index - drain_window + 1)]
        drain = (_median(recent) / _median(before)) if _median(before) > 0.0 else 1.0

        height = highs[index] - lows[index]
        year = highs[max(0, index - 251) : index + 1]

        out.append(
            {
                "ret_1": _finite(ret(1)),
                "ret_3": _finite(ret(3)),
                "ret_7": _finite(ret(7)),
                "decline_1": max(0.0, -_finite(ret(1))),
                "decline_7": max(0.0, -_finite(ret(7))),
                "atr_pct": _finite(atr_pct[index]),
                "vol_pct": _finite(vol_pct, 0.5),
                "close_sma20": _finite(sma(20)),
                "close_sma200": _finite(sma(200)),
                "turnover_z": _finite(turnover_z),
                "drain": _finite(drain, 1.0),
                "range_pct": _finite(100.0 * height / close if close > 0.0 else 0.0),
                "close_pos": _finite(((close - lows[index]) / height) if height > 0.0 else 0.5),
                "dd_252": _finite(100.0 * (close / max(year) - 1.0) if year and max(year) > 0 else 0.0),
                "rsi14": _finite(rsi[index], 50.0),
            }
        )
    return out


def bar_features(rows: Sequence[Candle], index: int, **windows: Any) -> Dict[str, float]:
    """The features of one bar — ``series_features(rows[:index + 1])[-1]``, by contract.

    Implemented through the series rather than a second traversal: two implementations of
    the same number are two chances to disagree, and a policy trained on one and run on the
    other is a policy that quietly changed its mind.
    """
    if not 0 <= int(index) < len(rows):
        raise IndexError(f"bar {index} is outside a series of {len(rows)} bars")
    return series_features(rows[: int(index) + 1], **windows)[-1]


def feature_vector(
    features: Mapping[str, float], names: Sequence[str] = FEATURES
) -> List[float]:
    """The features as a vector in ``names`` order, missing values as ``0.0``."""
    return [_finite(features.get(name, 0.0)) for name in names]


def enrich(
    features: Mapping[str, float],
    *,
    confidence: float = 0.0,
    regime: float = 1.0,
    bars_since_panic: int = 0,
) -> Dict[str, float]:
    """A feature row plus the three readings the series alone cannot supply.

    ``regime`` is ``1.0`` above the benchmark's trend and ``0.0`` below it; ``0.5`` means
    the benchmark was not available and the policy should read it as "unknown", which after
    standardisation is the middle of the column and the least inventive answer available.
    """
    out = dict(features)
    out["confidence"] = _finite(confidence)
    out["regime"] = _finite(regime, 0.5)
    out["bars_since_panic"] = _finite(bars_since_panic)
    return out


def feature_matrix(
    rows: Sequence[Candle], names: Sequence[str] = FEATURES, **windows: Any
) -> List[List[float]]:
    """Every bar as a vector — the shape a model is fitted on."""
    return [feature_vector(row, names) for row in series_features(rows, **windows)]


def _rsi(closes: Sequence[float], window: int = 14) -> List[float]:
    """A bounded gain/loss ratio, ``50`` while the record is too short to say otherwise."""
    out: List[float] = [50.0] * len(closes)
    gains = [0.0] * len(closes)
    losses = [0.0] * len(closes)
    for index in range(1, len(closes)):
        change = _finite(closes[index]) - _finite(closes[index - 1])
        gains[index] = max(change, 0.0)
        losses[index] = max(-change, 0.0)
    for index in range(window, len(closes)):
        gain = _mean(gains[index - window + 1 : index + 1])
        loss = _mean(losses[index - window + 1 : index + 1])
        out[index] = 50.0 if gain + loss <= 0.0 else 100.0 * gain / (gain + loss)
    return out

