"""What a bar was worth to a buyer, measured after the fees he would have paid.

A label is not a return. The return of the next three bars is a fact about the price; the
label this package trains on is the answer to a narrower question: *if the book had bought
at the next open and sold ``horizon`` bars later, would it have come out ahead of the venue's
own cut?* That is the only question a strategy may be fitted to, because it is the only one
whose answer changes sign when the fees change.

Three decisions are made here and each one is visible in the signature:

**The entry is the next open.** A decision made on bar ``i``'s close cannot be filled on
bar ``i``'s close — the engine already imposes that (:class:`agent.backtest.engine.Backtester`
queues the order for the next bar), so a label filled at the close would be training a model
on a fill that never happens.

**The exit is a close, ``horizon`` bars later.** Not a stop, not a take-profit: the label
answers "is this entry worth owning", not "did the exit rule work". The exit rule is
measured separately, by the backtest, and mixing the two would hide a bad exit inside a good
entry.

**Cost is subtracted, not ignored.** ``cost=0.004`` is the round trip the simulated venue
charges (two sides at 0.1 % fee and 0.1 % slippage). A model trained on gross returns learns
to buy moves that the venue keeps; the whole first sweep was that mistake measured in
dollars — 43 % of its loss was fees.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from agent.backtest.data import Candle

__all__ = [
    "DEFAULT_COST",
    "forward_return",
    "label_at",
    "labels_at",
    "net_forward_return",
]

#: Round-trip cost of the simulated venue, as a fraction: ``2 × (fee + slippage)``.
DEFAULT_COST = 0.004


def _close(rows: Sequence[Candle], index: int) -> float:
    return float(rows[index].close)


def _open(rows: Sequence[Candle], index: int) -> float:
    price = float(rows[index].open)
    return price if price > 0.0 else _close(rows, index)


def forward_return(
    rows: Sequence[Candle], index: int, horizon: int = 3, *, entry: str = "open"
) -> Optional[float]:
    """The next ``horizon`` bars as a percentage, or ``None`` when the record ends first.

    ``None`` and not zero: a bar whose future is not in the data is not a flat bar, and a
    zero label there would teach a model that the end of history is a coin that landed on
    its edge. The caller drops it (:func:`labels_at`), exactly as the detector drops a
    missing APY instead of filling it.
    """
    horizon = int(horizon)
    if horizon <= 0 or index < 0:
        return None
    entry_index = index + 1
    exit_index = index + horizon
    if entry_index >= len(rows) or exit_index >= len(rows):
        return None
    price_in = _open(rows, entry_index) if entry == "open" else _close(rows, index)
    price_out = _close(rows, exit_index)
    if price_in <= 0.0 or price_out <= 0.0:
        return None
    return 100.0 * (price_out / price_in - 1.0)


def net_forward_return(
    rows: Sequence[Candle], index: int, horizon: int = 3, *, cost: float = DEFAULT_COST
) -> Optional[float]:
    """The same return, in per cent, after the round trip the venue keeps."""
    gross = forward_return(rows, index, horizon)
    if gross is None:
        return None
    return gross - 100.0 * float(cost)


def label_at(
    rows: Sequence[Candle],
    index: int,
    horizon: int = 3,
    *,
    cost: float = DEFAULT_COST,
    threshold: float = 0.0,
) -> Optional[float]:
    """``1.0`` when the entry paid more than the cost, ``0.0`` when it did not, else ``None``.

    ``threshold`` moves the bar: on a market whose round trip is cheap, a label that only
    asks for "more than zero" is a label of a drift, not of an edge. Raising it to one ATR
    would ask for a move worth the risk — that is a choice for the caller, and the fact that
    it is a parameter is the point.
    """
    net = net_forward_return(rows, index, horizon, cost=cost)
    if net is None:
        return None
    return 1.0 if net > float(threshold) else 0.0


def labels_at(
    rows: Sequence[Candle],
    horizon: int = 3,
    *,
    cost: float = DEFAULT_COST,
    threshold: float = 0.0,
) -> List[Optional[float]]:
    """One label per bar, ``None`` where the future is not in the record."""
    return [label_at(rows, index, horizon, cost=cost, threshold=threshold) for index in range(len(rows))]
