"""What is allowed to reach the core, decided before the core is asked.

The hunter is the scanner and it stays the scanner: it reads two series — the decline a bar
offered a buyer and the turnover left behind — and says a word about them. Nothing in this
file re-reads its arithmetic, second-guesses its thresholds or competes with
:mod:`agent.backtest.policy`. What this file controls is the *input*: which bars are offered
to the scanner at all, and what a bar that is not worth judging is made to look like.

Why the input is the right place to work
========================================

A backtest of this strategy lost about 11 % over 579 trades, and the diagnosis was not that
the physics was wrong — it was that a large share of those trades were taken on bars where
the *reading itself* was an artifact:

* a bar with no turnover published no price, and a decline computed from it is the decline of
  an instrument that did not trade;
* a bar quoted in fractions of a cent is tick noise, and its 3 % move is a rounding, not a
  panic;
* a bar that gapped 25 % against a 3 % average true range is a listing, a halt, a vendor's
  correction — an event the hunter was never built to describe, and one the simulated venue
  cannot fill honestly either.

Feeding those bars to the detector is not a physics problem, and fixing it inside the
detector would be the wrong repair: the detector is also live, and a live loop that silently
ignored bars would be lying about what it saw. So the filter sits here, in front of the
scanner, and it is a *decision* an operator makes and a report states.

The mechanism: ``nan``, not zero
================================

There are only three things an unreadable bar can become, and two of them are lies.

* ``0.0`` says "this bar offered a buyer no decline" — a number the market never published,
  and one that would sit inside the resting median of every later prefix, quietly teaching
  the hunter that this instrument's normal is quieter than it is.
* Dropping the bar from the list would shift every later index, and an index is the only
  thing tying a verdict to a price.
* ``nan`` is what :func:`agent.backtest.data.declines` already answers while it warms up, and
  :func:`agent.physics.volatility_hunter._finite` already drops it. The record the detector
  sees is then *shorter and honest* rather than *longer and decorated* — the same rule this
  package keeps about a missing APY.

The filter therefore never rewrites a number. It withholds one, and the only other thing it
does is refuse to open a position on a bar it withheld.

What it costs
=============

A masked bar is a bar the detector has no reading for, so a run with a filter on can hold
fewer points than the same run with it off, and a record that was just long enough can become
just short. That is not a bug in the filter; it is the filter saying the record is thinner
than the bar count suggested, which is exactly what a ``THIN`` verdict means one level
deeper. :func:`report` states how many bars of each kind were withheld, so that a run can
never be read as "the strategy did nothing" when it was "the data said nothing".
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from agent.backtest.data import Candle

__all__ = [
    "Admission",
    "GAP",
    "STALE",
    "TICK",
    "THIN_BAR",
    "admit",
    "admission_from_args",
    "add_admission_arguments",
    "masked",
    "report",
]

#: The bar carried no turnover at all: nothing traded, so nothing was published.
STALE = "stale"
#: The bar's turnover is below the floor an operator set for a market worth reading.
THIN_BAR = "thin"
#: The bar's close is small enough that a basis point is a rounding error.
TICK = "tick"
#: The bar's move is a gap — many average true ranges in one bar, not a decline.
GAP = "gap"

#: Every reason :func:`admit` can give, in the order the rules are tested. The order is part of
#: the report: a zero-turnover bar is ``stale`` even if it is also thin, because the first
#: sentence is the one a reader needs.
REASONS = (STALE, THIN_BAR, TICK, GAP)


@dataclass(frozen=True)
class Admission:
    """The thresholds of the input filter, in one value, validated once.

    Supplying an :class:`Admission` at all turns on the two hygiene rules — :data:`STALE` and
    :data:`TICK` — because a bar with no turnover and a bar quoted at zero are not readings of
    a market, and no operator should have to ask for that. The other two rules are off until a
    number is set, so ``Admission()`` means "the hygiene rules and nothing else", and ``None``
    (the engine's default) means "every bar, exactly as published".
    """

    #: Quote turnover a bar must clear, in USDT. ``0`` disables the rule.
    min_turnover: float = 0.0
    #: The smallest close worth reading, in quote currency. ``0`` disables the rule.
    min_close: float = 0.0
    #: A move larger than this many average true ranges is a gap. ``0`` disables the rule.
    max_gap_atr: float = 0.0
    #: Bars behind the average true range ``max_gap_atr`` multiplies.
    atr_window: int = 14

    def __post_init__(self) -> None:
        if float(self.min_turnover) < 0.0:
            raise ValueError(f"min_turnover must not be negative, not {self.min_turnover!r}")
        if float(self.min_close) < 0.0:
            raise ValueError(f"min_close must not be negative, not {self.min_close!r}")
        if float(self.max_gap_atr) < 0.0:
            raise ValueError(f"max_gap_atr must not be negative, not {self.max_gap_atr!r}")
        if int(self.atr_window) < 1:
            raise ValueError(f"atr_window must be at least one bar, not {self.atr_window!r}")

    @property
    def is_open(self) -> bool:
        """True when no threshold was set — the two hygiene rules are all that is left."""
        return not (
            float(self.min_turnover) > 0.0
            or float(self.min_close) > 0.0
            or float(self.max_gap_atr) > 0.0
        )

    def rules(self) -> Dict[str, Any]:
        return {
            "min_turnover": round(float(self.min_turnover), 4),
            "min_close": round(float(self.min_close), 8),
            "max_gap_atr": round(float(self.max_gap_atr), 4),
            "atr_window": int(self.atr_window),
        }

    def as_dict(self) -> Dict[str, Any]:
        return self.rules()


# ── the filter ────────────────────────────────────────────────────────────


def admit(rows: Sequence[Candle], admission: Optional[Admission] = None) -> List[str]:
    """One reason per bar: ``""`` for a bar the core may read, otherwise why not.

    The rules are tested in the order of :data:`REASONS`, and the first one that fires is the
    answer — a bar can be stale, thin *and* in fractions of a cent, and the sentence a reader
    needs is the first true one, not the longest list.

    The average true range behind :data:`GAP` is measured on the bars *up to and including*
    this one, exactly like every feature in :mod:`agent.backtest.features`, so a filter can
    never see the bar it is judging the size of. Its window starts short: one true range is
    enough to answer, and a short window is a wide allowance rather than a pass, because a
    gap is only ever *rejected* by this rule.
    """
    reasons: List[str] = [""] * len(rows)
    if admission is None:
        return reasons
    floor = float(admission.min_turnover)
    penny = float(admission.min_close)
    gap_atr = float(admission.max_gap_atr)
    span = max(int(admission.atr_window), 1)
    true_ranges: deque = deque(maxlen=span)
    previous = float(rows[0].close) if rows else 0.0
    for index, row in enumerate(rows):
        close = float(row.close)
        volume = float(row.volume)
        if index:
            high = float(row.high)
            low = float(row.low)
            true_ranges.append(max(high, previous) - min(low, previous))
        if not math.isfinite(volume) or volume <= 0.0:
            reasons[index] = STALE
        elif floor > 0.0 and volume < floor:
            reasons[index] = THIN_BAR
        elif not math.isfinite(close) or close <= 0.0 or (penny > 0.0 and close < penny):
            reasons[index] = TICK
        elif gap_atr > 0.0 and index and previous > 0.0 and math.isfinite(close):
            atr = sum(true_ranges) / float(len(true_ranges)) if true_ranges else 0.0
            if atr > 0.0 and abs(close - previous) > gap_atr * atr:
                reasons[index] = GAP
        previous = close if math.isfinite(close) else previous
    return reasons


def masked(values: Sequence[float], reasons: Sequence[str]) -> List[float]:
    """``values`` with every withheld bar replaced by ``nan`` — a hole, not a zero.

    ``nan`` is the one value the detector already knows how to skip
    (:func:`agent.physics.volatility_hunter._finite`), and it keeps the list the same length,
    so an index still points at the bar it always pointed at.
    """
    if len(values) != len(reasons):
        raise ValueError(
            f"{len(values)} values against {len(reasons)} reasons: the mask has to be "
            "per bar or it is a mask on a different series"
        )
    return [
        float(value) if not reason else math.nan
        for value, reason in zip(values, reasons)
    ]


def report(
    rows: Sequence[Candle], admission: Optional[Admission] = None
) -> Dict[str, Any]:
    """What the filter did to this record: the counts, per reason, and one sentence.

    This is the number that keeps a quiet run honest. "No trades" and "no readable bars" look
    identical in a trade log and mean opposite things, so the count of withheld bars travels
    with every run that used a filter.
    """
    reasons = admit(rows, admission)
    counts: Dict[str, int] = {}
    for reason in reasons:
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    withheld = sum(counts.values())
    bars = len(rows)
    sentence = ""
    if withheld:
        parts = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
        sentence = (
            f"{withheld} of {bars} bars were withheld from the detector ({parts}); the "
            "verdicts were measured on what was left, which is the honest record and not the "
            "whole series"
        )
    return {
        "rules": None if admission is None else admission.rules(),
        "bars": int(bars),
        "admitted": int(bars - withheld),
        "withheld": int(withheld),
        "share": round(withheld / bars, 6) if bars else 0.0,
        "reasons": counts,
        "sentence": sentence,
    }


def admission_from_args(args: Any) -> Optional[Admission]:
    """Build the filter from a parsed command line, or ``None`` when nothing was asked for.

    Every entry point in this package — ``run``, ``sweep``, ``walkforward``, ``train`` — reads
    the same three flags through this function, so that "the filter is on" means the same
    thing in all four and one of them cannot quietly default to a different market.

    A command line that sets none of the three gets ``None`` rather than ``Admission()``, so a
    run without a filter is byte-for-byte the run that came before this module existed.
    """
    min_turnover = float(getattr(args, "min_turnover", 0.0) or 0.0)
    min_close = float(getattr(args, "min_close", 0.0) or 0.0)
    max_gap_atr = float(getattr(args, "max_gap_atr", 0.0) or 0.0)
    atr_window = int(getattr(args, "admission_atr_window", 14) or 14)
    if not (min_turnover > 0.0 or min_close > 0.0 or max_gap_atr > 0.0):
        return None
    return Admission(
        min_turnover=min_turnover,
        min_close=min_close,
        max_gap_atr=max_gap_atr,
        atr_window=atr_window,
    )


def add_admission_arguments(parser: Any) -> None:
    """Register the input-filter flags on a parser, identically in all four entry points.

    The flags are one group so that ``--help`` shows them as the single decision they are: a
    reader who sees ``--min-turnover`` three screens below ``--min-depth`` would reasonably
    think they are the same knob, and they are not — ``--min-depth`` is a verdict the hunter
    gives (``THIN``, and the position is closed), while these decide whether the hunter is
    asked at all.
    """
    group = parser.add_argument_group(
        "input filter", "what may reach the detector — nothing here changes the physics"
    )
    group.add_argument(
        "--min-turnover",
        type=float,
        default=0.0,
        help="quote turnover a bar must clear, USDT (e.g. 50000); 0 withholds nothing",
    )
    group.add_argument(
        "--min-close",
        type=float,
        default=0.0,
        help="the smallest close worth reading, in quote currency (e.g. 0.001); 0 is off",
    )
    group.add_argument(
        "--max-gap-atr",
        type=float,
        default=0.0,
        help="a move beyond this many average true ranges is a gap, not a decline; 0 is off",
    )
    group.add_argument(
        "--admission-atr-window",
        type=int,
        default=14,
        help="bars behind the average true range --max-gap-atr multiplies",
    )

