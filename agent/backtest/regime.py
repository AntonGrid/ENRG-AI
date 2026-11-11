"""The market's own weather, as one mask: above the benchmark's mean, or below it.

The sweep already measured the difference, and it was the loudest number in it: entries
taken while BTC traded above its 200-bar mean returned +0.28 % a trade (not significant),
entries taken below it returned −0.64 % (t = −3.20). That is not a property of the
detector — the detector never sees the benchmark — it is a property of the *market the bet is
placed in*, and a policy that cannot see the weather is a policy that will keep buying
umbrellas in a drought.

The mask is deliberately coarse. ``True`` above the mean, ``False`` below, ``True`` while
the mean is still warming up because refusing to trade the first 200 bars of a record is a
different decision than measuring one. A finer reading (slope, distance, realised volatility)
is a feature, and features belong in :mod:`agent.backtest.features` where the policy can
weigh them — this file answers one question so that the answer is checkable by eye.

The alignment matters and is not obvious: the benchmark is a *different* series with its own
calendar. A bar of ``solusdt`` is matched to the newest benchmark bar that closed at or
before it, never to the next one — a mask read from the future would turn a filter into a
look-ahead, and the whole walk-forward would inherit the mistake.
"""
from __future__ import annotations

import bisect
from typing import List, Optional, Sequence

from agent.backtest.data import Candle

__all__ = [
    "DEFAULT_REGIME_WINDOW",
    "align_regime",
    "sma_mask",
]

#: The benchmark's mean, in bars. Two hundred daily bars is the line the sweep measured.
DEFAULT_REGIME_WINDOW = 200


def sma_mask(closes: Sequence[float], window: int = DEFAULT_REGIME_WINDOW) -> List[bool]:
    """``True`` where the close stands at or above its own trailing mean.

    ``False`` on a series too short to have a mean at all would be a refusal to trade a
    young pair; ``True`` is the permissive reading and the one that keeps the mask's *only*
    job being "was the market above its mean", not "was there enough history".
    """
    size = max(int(window), 1)
    out: List[bool] = []
    total = 0.0
    for index, close in enumerate(closes):
        total += float(close)
        if index >= size:
            total -= float(closes[index - size])
        if index + 1 < size:
            out.append(True)
            continue
        mean = total / size
        out.append(float(close) >= mean)
    return out


def align_regime(
    rows: Sequence[Candle],
    benchmark: Sequence[Candle],
    window: int = DEFAULT_REGIME_WINDOW,
) -> List[bool]:
    """The benchmark's mask, read on the instrument's own clock.

    Each instrument bar takes the benchmark's verdict as of the *last benchmark bar that had
    already happened* — the same rule the engine applies to its own fills. A bar with no
    benchmark history behind it is ``True``: the mask must never be the reason a bar is
    silently dropped from a measurement, only the reason a bet is declined.
    """
    if not benchmark:
        return [True] * len(rows)
    stamps = [float(bar.at) for bar in benchmark]
    mask = sma_mask([float(bar.close) for bar in benchmark], window)
    out: List[bool] = []
    for row in rows:
        position = bisect.bisect_right(stamps, float(row.at)) - 1
        out.append(True if position < 0 else bool(mask[position]))
    return out
