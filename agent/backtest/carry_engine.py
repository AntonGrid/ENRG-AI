"""The pair the funding history was collected for — and the gate it has to clear.

Why this file exists
====================

``carry.py`` answered one question and refused the next, on purpose: a contract can publish six
years of funding, report a positive median, and still pay nothing after the two taker legs, the
basis and the margin. That second question is this file. It prices one delta-neutral pair —
**long spot, short perp, paid the funding rate for holding both** — against a gate that was
written down before the first run:

    net APY      >= 12 %    of the capital the pair actually ties up, after every declared cost
    drawdown     <= 3 %     of that capital, on the funding-cash curve
    funding > 0  >= 60 %    of the periods the venue reported, not of the periods requested
    capacity     >= $50 000 deployable at <= 3x leverage

None of those numbers came from a table: they are the operator's terms, and the run either
meets them or says which one it missed. A backtest's job is to answer, never to succeed.

The arithmetic, in the order it happens
=======================================

A hedged pair ties up more capital than its notional, because the spot leg is bought outright
while the perp leg only posts margin. With notional ``N`` and ``L`` times leverage on the perp::

    capital = N + N / L          so     N = capital * L / (1 + L)

At the declared $50 000 and 3x, the pair carries $37 500 of notional: funding is earned on the
notional and the APY is quoted on the capital, and confusing the two overstates a carry book by
a third.

The position is opened at the first funding timestamp and closed at the last one, and the book
is paid for periods **after** the one it opened on: a pair that appears at the settlement
instant cannot claim the payment that settlement made. Each reported rate pays
``N * funding_rate`` — positive when longs pay shorts, which is the direction this book holds.
An unreported rate (``null`` or ``0E-18``) pays nothing and is counted, never filled: the
collector's rule that "unknown is not zero" survives into the pricing without a second
interpretation.

Costs are charged where they happen, both legs, both ways::

    leg     = N * (spot_taker + perp_taker + 2 * slippage)
    entry   = leg, exit = leg, round trip = 2 * leg

At the declared 0.2 % spot and 0.05 % perp list rates plus five bps of slippage a side, the
round trip is 70 bps of notional. It is not amortised away — it is subtracted, once, and the
entry cost sits in the equity curve from the first period, so it counts against the drawdown
too.

The two clocks, and why the declared drawdown is a lower bound
==============================================================

The drawdown the gate reads is measured on the **funding-cash curve**: capital, minus the entry
cost, plus every payment the venue published, minus the exit cost. It covers the whole history
the collector could reach, and it is blind to the basis by construction — the spot leg and the
perp leg move, and what the pair actually earns or loses between settlements is
``-N * (basis_t - basis_0)``.

That mark needs a spot bar and a perp bar for the same instant, and the perp kline feed serves
one page and no second one. So when ``--perp-csv`` and ``--spot-csv`` are given, the basis is
measured on the window those files cover and reported **beside** the funding curve, with its
own coverage share and its own drawdown — never folded into ``net_apy``, because a number that
mixes a six-year clock with a one-year one is not a number. A passing drawdown on the funding
curve is therefore a lower bound, and :data:`DOES_NOT_SHOW` says so in the hashed document.

What the verdict can say, and what it cannot
============================================

Each criterion is ``met``, ``missed``, or ``unmeasured`` — and the last one is not a soft "fail":
a history shorter than a year, fewer than :data:`MIN_PERIODS` held periods, or a venue that
published rates for less than :data:`REPORTED_FLOOR` of them cannot support an annualised claim,
and a book with no turnover reading cannot support a capacity claim. The overall verdict is
``pass`` only when every criterion is met, ``fail`` when any is missed, and ``unmeasured``
otherwise — three states, because "we could not ask" and "the answer is no" are different
answers, and a report that merged them would lie in whichever direction was convenient.

Measured on the first run
=========================

``BTC-USDT``: 6 643 published periods, 2020-10-21 .. 2026-11-13, mean funding 0.94 bps per 8 h,
positive in 83.0 % of them, one hole of two periods. A $50 000 book at 3x carries $37 500 of
notional, which earns 7.65 % a year after the round trip and draws down 0.78 % of capital on
the funding curve. It misses the 12 % gate — and that is the most useful sentence this file
prints, because the alternative was treating a positive median as a business.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.backtest.carry import (
    ALIGNMENT_TOLERANCE,
    DEFAULT_CACHE,
    DEFAULT_UNIVERSE,
    FundingSeries,
    fetch_funding_cached,
    read_universe,
)
from agent.backtest.criteria import Criteria, verify
from agent.backtest.data import CandleSeries, DEFAULT_TIMEOUT, HtxError, load_csv
from agent.backtest.run import EXIT_FEED, EXIT_SHORT, EXIT_USAGE

__all__ = [
    "ACCEPTANCE",
    "CARRY_GATE",
    "DEFAULT_OUT",
    "DOES_NOT_SHOW",
    "MET",
    "MIN_PERIODS",
    "MIN_YEARS",
    "MISSED",
    "PARTICIPATION",
    "PERP_FEE",
    "REPORTED_FLOOR",
    "SLIPPAGE",
    "SPOT_FEE",
    "UNMEASURED",
    "CarryRun",
    "Criterion",
    "Settings",
    "Verdict",
    "basis_marks",
    "criteria_of",
    "format_summary",
    "judge",
    "main",
    "simulate",
    "write_carry",
]

#: The declaration, written before this file was run and hashed into every report it writes.
#: ``max_leverage`` is a ceiling on the book, not a target; ``capacity_usdt`` is deployable
#: capital, the unit the operator named, so it is compared with the same unit that comes out of
#: :meth:`Settings.capacity_usdt` rather than with a notional.
CARRY_GATE: Dict[str, float] = {
    "net_apy": 0.12,
    "max_drawdown": 0.03,
    "positive_share": 0.60,
    "capacity_usdt": 50_000.0,
    "max_leverage": 3.0,
}

#: HTX's *list* prices, not a fee tier: a VIP-0 spot account pays 0.2 % a side and a VIP-0 swap
#: account 0.05 %. ``crosssection.py`` already names 0.002 as "the VIP-0 account"; the same
#: account is assumed on both legs here rather than a tier nobody reading this has.
SPOT_FEE = 0.002
PERP_FEE = 0.0005

#: Five bps a side, charged on entry and on exit, on both legs. A delta-neutral pair crosses
#: four spreads to round-trip (spot in, perp in, perp out, spot out) and this is each of them.
SLIPPAGE = 0.0005

#: How much of the contract's 24 h quote turnover a $50 000 book may be. One percent of a
#: contract that trades $684M a day is $6.8M of notional a day, which is generous for a pair
#: held across settlements; the number is declared so the capacity column is a rule and not a
#: feeling about liquidity.
PARTICIPATION = 0.01

#: What a history has to be before an annualised number means anything: a year of calendar, a
#: hundred periods, and rates on nine tenths of them. Below any of the three the APY and the
#: drawdown print as ``unmeasured`` — a caveat on the number would still be a number a reader
#: quotes without it.
MIN_YEARS = 1.0
MIN_PERIODS = 100
REPORTED_FLOOR = 0.90

#: Where the report lands unless ``--out`` says otherwise. Under ``data/`` with the collector's
#: own output: a report about a cache belongs beside the cache.
DEFAULT_OUT = "data/carry/engine.json"

#: What a passing run would have to satisfy, spelled out so that "pass" is not an opinion.
ACCEPTANCE: Tuple[str, ...] = (
    "the APY is on capital, not notional: the spot leg is bought outright and only the perp "
    "leg is leveraged, so net_apy divides by capital = notional * (1 + 1/leverage)",
    "every cost is charged, on both legs, both ways: the spot and perp taker fees and one "
    "slippage per leg at entry and again at exit, subtracted once rather than smoothed away",
    "an unreported rate pays nothing and is counted: a venue that published no rate for a "
    "period did not publish a zero, and the APY is only read when nine tenths of the periods "
    "carry a rate",
    "the drawdown is measured, not asserted: it is the largest peak-to-trough fall of the "
    "funding-cash curve in capital, with the entry cost already in it",
    "unmeasured is not a fail: a history shorter than a year or a contract with no turnover "
    "reading prints as unmeasured for the criterion it cannot support, and the overall "
    "verdict is unmeasured until every criterion has an answer",
)

#: The honest half, hashed with the gate: what even a perfect run has not shown.
DOES_NOT_SHOW: Tuple[str, ...] = (
    "that the short leg would have survived: no liquidation price, no mark-price path and no "
    "margin call is modelled, so a squeeze that takes the perp leg out is indistinguishable "
    "here from a quiet week — funding is the payment, not the risk",
    "the basis over the whole history: the perp kline feed serves one page and no second one, "
    "so the declared drawdown is the funding-cash curve and a lower bound; the marked window "
    "is reported beside it, on its own shorter clock, never inside net_apy",
    "anything about a managed book: this pair is opened once and closed once, so no "
    "rebalancing, no rolling the hedge, no sizing rule and no funding-driven exit is priced",
    "borrow, custody, transfer and margin interest: capital is assumed to sit on the venue "
    "with no cost to get there, stay there, or leave",
    "capacity as depth: one percent of 24 h quote turnover is a rule of thumb about a "
    "position, not a read of the order book, so a contract that screens as deep may still "
    "cost more than the declared slippage to enter",
    "a survivorship-free universe: the contracts priced here are the ones listed today, and "
    "the ones that were delisted — the failures a carry book is most likely to hold — are not",
    "any exchange risk: a funding history is not a promise that the venue pays out, and no "
    "number here says anything about the venue's solvency",
)


# ── the book, and what it has to answer to ─────────────────────────────────


@dataclass(frozen=True)
class Settings:
    """The book: how much capital, how much leverage, and what the two legs cost to move.

    ``notional`` and ``capital`` are deliberately different numbers and the difference is the
    whole point of a hedged pair: the spot leg is bought outright and only the perp leg posts
    margin, so ``N`` of notional occupies ``N + N / L`` of capital. Every rate this module
    prints is either on one or on the other, and each one says which.
    """

    capital: float = 50_000.0
    leverage: float = 3.0
    spot_fee: float = SPOT_FEE
    perp_fee: float = PERP_FEE
    slippage: float = SLIPPAGE
    participation: float = PARTICIPATION

    def __post_init__(self) -> None:
        if not float(self.capital) > 0.0:
            raise ValueError(f"capital {self.capital!r} cannot fund a pair")
        ceiling = float(CARRY_GATE["max_leverage"])
        if not 0.0 < float(self.leverage) <= ceiling:
            raise ValueError(
                f"leverage {self.leverage!r} is outside (0, {ceiling:g}]: the declared gate was "
                "written for a book at or below 3x, and a more levered pair is a different "
                "question with a different answer, not a better one"
            )
        for name in ("spot_fee", "perp_fee", "slippage"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 0.05:
                raise ValueError(f"{name} {value!r} is not a rate: it must lie in [0, 0.05)")
        if not 0.0 < float(self.participation) <= 0.10:
            raise ValueError(
                f"participation {self.participation!r} is not a share of a day: it must lie in "
                "(0, 0.10], because a book that is a tenth of a contract's turnover is the "
                "market in that contract and not a participant in it"
            )

    @property
    def notional(self) -> float:
        """The dollars on each leg: ``capital * L / (1 + L)``."""
        return float(self.capital) * float(self.leverage) / (1.0 + float(self.leverage))

    @property
    def capital_used(self) -> float:
        """The dollars the pair ties up — the spot leg plus the perp margin, i.e. ``capital``."""
        return self.notional + self.notional / float(self.leverage)

    @property
    def leg_cost(self) -> float:
        """One side of the round trip as a share of notional: both fees and both slippages."""
        return float(self.spot_fee) + float(self.perp_fee) + 2.0 * float(self.slippage)

    @property
    def round_trip_usdt(self) -> float:
        """What it costs to put the pair on and take it off again, in dollars."""
        return 2.0 * self.leg_cost * self.notional

    def capacity_usdt(self, turnover: float) -> Optional[float]:
        """Deployable capital at this leverage, given a contract's 24 h quote turnover.

        The chain is one line and worth keeping so: the book may be ``participation`` of a day's
        turnover in notional, and a notional of ``X`` needs ``X * (1 + 1/L)`` of capital. Thus
        ``capacity = turnover * participation * (1 + 1/L)``, in the same dollars as the gate's
        ``capacity_usdt``. ``None`` when no turnover was read — an unknown is not a zero.
        """
        if not float(turnover) > 0.0:
            return None
        share = float(self.participation) * (1.0 + 1.0 / float(self.leverage))
        return float(turnover) * share

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capital": float(self.capital),
            "leverage": float(self.leverage),
            "notional": float(self.notional),
            "capital_used": float(self.capital_used),
            "spot_fee": float(self.spot_fee),
            "perp_fee": float(self.perp_fee),
            "slippage": float(self.slippage),
            "leg_cost": float(self.leg_cost),
            "round_trip_usdt": float(self.round_trip_usdt),
            "participation": float(self.participation),
        }


#: The three answers a criterion can give. ``unmeasured`` is not a polite ``missed``: it says
#: the run could not ask, and the two must never be added up into one percentage.
MET = "met"
MISSED = "missed"
UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class Criterion:
    """One line of the gate: what was declared, what was measured, and which way it went."""

    name: str
    declared: float
    measured: Optional[float]
    unit: str
    verdict: str = UNMEASURED
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "declared": float(self.declared),
            "measured": None if self.measured is None else float(self.measured),
            "unit": self.unit,
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Verdict:
    """Every criterion at once, and the one word a reader is allowed to quote."""

    criteria: Tuple[Criterion, ...] = ()

    @property
    def overall(self) -> str:
        """``fail`` if anything is missed, ``unmeasured`` if anything could not be asked."""
        states = {item.verdict for item in self.criteria}
        if MISSED in states:
            return "fail"
        if not self.criteria or UNMEASURED in states:
            return UNMEASURED
        return "pass"

    @property
    def missed(self) -> List[str]:
        """The names of the criteria the run did not meet — empty when none was missed."""
        return [item.name for item in self.criteria if item.verdict == MISSED]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "missed": self.missed,
            "criteria": [item.as_dict() for item in self.criteria],
        }


# ── the basis, on its own clock ────────────────────────────────────────────


def basis_marks(
    perp: CandleSeries,
    spot: CandleSeries,
    *,
    tolerance: float = ALIGNMENT_TOLERANCE,
) -> List[Tuple[float, float]]:
    """The spot-perp spread at every bar open both feeds agree on, in bps, oldest first.

    Two-pointer nearest-neighbour, the same rule :func:`agent.backtest.carry.check_alignment`
    counts with, but handing out the marks themselves instead of an average of them: a drawdown
    is a path, and an average has no path to draw. A bar whose nearest partner sits further than
    ``tolerance`` seconds away is not marked — the same "unknown is not zero" the funding
    collector applies to a rate — and the coverage share printed beside the mark block is what
    tells a reader how much of the window this actually saw.
    """
    perp_rows = list(perp.rows)
    spot_rows = list(spot.rows)
    if not perp_rows or not spot_rows:
        return []
    marks: List[Tuple[float, float]] = []
    index = 0
    for row in perp_rows:
        while index + 1 < len(spot_rows) and abs(spot_rows[index + 1].at - row.at) <= abs(
            spot_rows[index].at - row.at
        ):
            index += 1
        if abs(float(row.at) - float(spot_rows[index].at)) > float(tolerance):
            continue
        spot_close = float(spot_rows[index].close)
        if spot_close <= 0.0 or float(row.close) <= 0.0:
            continue
        marks.append((float(row.at), (float(row.close) / spot_close - 1.0) * 1e4))
    return marks


# ── one book, held once ────────────────────────────────────────────────────


@dataclass
class CarryRun:
    """What one hedged pair would have paid, and what the gate says about it.

    Every money field is in USDT on the pair's own notional; every ratio is on capital. The
    distinction is not decoration: ``gross_usdt`` is ``notional * sum(rates)`` and ``net_apy``
    is ``net_usdt / capital / years``, so a reader who wants the notional return can have it and
    a reader comparing with the 12 % gate is comparing the quantity the gate names.
    """

    contract: str
    settings: Settings
    period_seconds: float = 0.0
    first: Optional[float] = None
    last: Optional[float] = None
    years: float = 0.0
    periods: int = 0
    paid_periods: int = 0
    unreported_periods: int = 0
    gap_periods: int = 0
    gross_usdt: float = 0.0
    entry_cost_usdt: float = 0.0
    exit_cost_usdt: float = 0.0
    net_usdt: float = 0.0
    net_apy: Optional[float] = None
    gross_apy: Optional[float] = None
    drawdown: Optional[float] = None
    positive_share: Optional[float] = None
    reported_share: Optional[float] = None
    turnover_usdt: float = 0.0
    capacity_usdt: Optional[float] = None
    marks: Optional[Dict[str, Any]] = None
    verdict: Verdict = field(default_factory=Verdict)
    warnings: List[str] = field(default_factory=list)

    @property
    def judgeable(self) -> bool:
        """Whether the history is long and well reported enough to annualise anything."""
        return (
            float(self.years) >= MIN_YEARS
            and int(self.periods) >= MIN_PERIODS
            and float(self.reported_share or 0.0) >= REPORTED_FLOOR
        )

    def readability(self) -> str:
        """Empty when the run is judgeable, else the sentence naming the floor it missed.

        The order matters for a reader: a two-period series is a data problem, a six-month
        series is a window problem, and a long series with gaps is a venue problem. Naming the
        first one that applies means the report never sends someone to fix the wrong thing.
        """
        if int(self.periods) < MIN_PERIODS:
            return (
                f"{int(self.periods)} held period(s) is fewer than the {MIN_PERIODS} this module "
                "requires before an annualised number is read"
            )
        if float(self.years) < MIN_YEARS:
            return (
                f"{float(self.years):.2f} year(s) of history is shorter than the {MIN_YEARS:g} "
                "year this module requires before an annualised number is read"
            )
        share = float(self.reported_share or 0.0)
        if share < REPORTED_FLOOR:
            return (
                f"{share * 100.0:.1f} % of the held periods carried a published rate, below the "
                f"{REPORTED_FLOOR * 100.0:g} % floor: the missing payments could have been "
                "anything, so an APY would be a lower bound wearing the name of a measurement"
            )
        return ""


    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "settings": self.settings.as_dict(),
            "window": {
                "period_seconds": float(self.period_seconds),
                "periods": int(self.periods),
                "paid_periods": int(self.paid_periods),
                "unreported_periods": int(self.unreported_periods),
                "gap_periods": int(self.gap_periods),
                "years": float(self.years),
                "first": self.first,
                "last": self.last,
                "reported_share": self.reported_share,
            },
            "money": {
                "gross_usdt": float(self.gross_usdt),
                "entry_cost_usdt": float(self.entry_cost_usdt),
                "exit_cost_usdt": float(self.exit_cost_usdt),
                "net_usdt": float(self.net_usdt),
                "gross_apy": self.gross_apy,
                "net_apy": self.net_apy,
            },
            "risk": {
                "drawdown": self.drawdown,
                "positive_share": self.positive_share,
            },
            "capacity": {
                "turnover_usdt": float(self.turnover_usdt),
                "capacity_usdt": self.capacity_usdt,
                "participation": float(self.settings.participation),
            },
            "marks": self.marks,
            "verdict": self.verdict.as_dict(),
            "warnings": list(self.warnings),
        }


# ── holding it ─────────────────────────────────────────────────────────────


def simulate(
    series: FundingSeries,
    *,
    settings: Optional[Settings] = None,
    turnover: float = 0.0,
    marks: Optional[Sequence[Tuple[float, float]]] = None,
    gate: Optional[Mapping[str, float]] = None,
) -> CarryRun:
    """Hold the pair across one funding history and report what it would have paid.

    One entry, one exit, and every published rate in between — the honest simplest version of
    the business, and the one whose arithmetic a reader can redo with a pencil. The pair is
    opened at the first record and earns from the second onward, because a book that appears at
    the settlement instant cannot claim what that settlement paid.

    ``marks`` is a sequence of ``(at, basis_bps)`` from :func:`basis_marks`; it is optional and
    it never touches ``net_apy``. ``turnover`` is the contract's 24 h quote turnover in USDT and
    a zero means "not read", which makes the capacity criterion unmeasured rather than failed.
    """
    book = settings or Settings()
    floors = dict(CARRY_GATE if gate is None else gate)
    records = list(series.records)
    run = CarryRun(
        contract=str(series.contract).strip().upper(),
        settings=book,
        period_seconds=float(series.step()),
        years=float(series.years),
        turnover_usdt=max(float(turnover), 0.0),
        warnings=list(series.warnings),
    )
    run.capacity_usdt = book.capacity_usdt(run.turnover_usdt)
    run.first = float(records[0].at) if records else None
    run.last = float(records[-1].at) if records else None
    if len(records) < 2:
        run.warnings.append(
            "the series holds fewer than two funding periods, so no pair could be opened and "
            "held across a payment"
        )
        run.verdict = judge(run, floors)
        return run

    held = records[1:]
    leg = book.notional * book.leg_cost
    run.entry_cost_usdt = leg
    run.exit_cost_usdt = leg
    run.periods = len(held)
    equity = float(book.capital) - leg
    peak = equity
    drawdown = 0.0
    gross = 0.0
    paid = 0
    for record in held:
        if record.funding_rate is None:
            continue
        payment = float(book.notional) * float(record.funding_rate)
        gross += payment
        equity += payment
        paid += 1
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    run.gross_usdt = gross
    run.paid_periods = paid
    run.unreported_periods = run.periods - paid
    run.gap_periods = sum(int(hole[2]) for hole in series.gaps())
    run.net_usdt = gross - leg - leg
    run.drawdown = drawdown / float(book.capital) if float(book.capital) > 0.0 else None
    run.reported_share = (paid / run.periods) if run.periods else None
    run.positive_share = series.positive_share
    if float(run.years) > 0.0:
        run.gross_apy = gross / float(book.capital) / float(run.years)
        run.net_apy = run.net_usdt / float(book.capital) / float(run.years)
    if run.gap_periods:
        run.warnings.append(
            f"{run.gap_periods} funding period(s) inside the window were never published: the "
            "book is credited nothing for them, so this revenue is what the venue recorded and "
            "not what a calendar says should have been paid"
        )
    if marks:
        run.marks = _mark_block(run, held, marks)
        share = run.marks.get("covered_share")
        if share is not None and float(share) < 1.0:
            run.warnings.append(
                f"the basis marks cover {float(share) * 100.0:.1f} % of the held periods: the "
                "marked block is a shorter clock and none of it is inside net_apy"
            )
    run.verdict = judge(run, floors)
    return run



def _mark_block(
    run: CarryRun,
    held: Sequence[Any],
    marks: Sequence[Tuple[float, float]],
) -> Dict[str, Any]:
    """The basis leg over the window the price files cover — beside the funding curve, not in it.

    A mark is applied by *carrying the last bar forward* to a funding timestamp, never by
    looking at the next one: the bar after a settlement is information the pair did not have
    when it was paid. That is the same no-lookahead rule the rest of the layer reads, and it is
    the reason the first covered period contributes no basis PnL — it is the reference. The
    carry-forward is bounded by one funding period, so a bar from six years ago is not allowed
    to stand in for a settlement it never saw.
    """
    ordered = sorted((float(at), float(bps)) for at, bps in marks)
    times = [at for at, _ in ordered]
    window = float(run.period_seconds)
    covered: List[Tuple[float, Optional[float]]] = []
    for record in held:
        index = bisect.bisect_right(times, float(record.at)) - 1
        if index < 0:
            continue
        if window > 0.0 and float(record.at) - ordered[index][0] > window:
            continue
        rate = None if record.funding_rate is None else float(record.funding_rate)
        covered.append((float(ordered[index][1]), rate))
    period_count = int(run.periods)
    block: Dict[str, Any] = {
        "marked_periods": len(covered),
        "covered_share": (len(covered) / period_count) if period_count else None,
        "basis_first_bps": covered[0][0] if covered else None,
        "basis_last_bps": covered[-1][0] if covered else None,
        "basis_drift_bps": None,
        "basis_pnl_usdt": None,
        "drawdown": None,
        "note": (
            "the price files share fewer than two funding periods with this history, so the "
            "basis is not reported at all rather than reported as flat"
        ),
    }
    if len(covered) < 2:
        return block
    notional = float(run.settings.notional)
    drift = covered[-1][0] - covered[0][0]
    equity = float(run.settings.capital) - float(run.entry_cost_usdt)
    peak = equity
    drawdown = 0.0
    for index in range(1, len(covered)):
        equity -= notional * (covered[index][0] - covered[index - 1][0]) / 1e4
        rate = covered[index][1]
        if rate is not None:
            equity += notional * rate
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    block["basis_drift_bps"] = drift
    block["basis_pnl_usdt"] = -notional * drift / 1e4
    block["drawdown"] = drawdown / float(run.settings.capital)
    block["note"] = (
        "the marks cover one page of price bars, which is shorter than the funding history: "
        "the drift, the PnL and the drawdown in this block are on a different clock and are "
        "not comparable with the declared gate"
    )
    return block


# ── what the gate says ─────────────────────────────────────────────────────


def judge(run: CarryRun, gate: Optional[Mapping[str, float]] = None) -> Verdict:
    """Compare one run with the declared gate, criterion by criterion.

    ``unmeasured`` is produced for a reason and the reason is printed: a reader has to be able
    to tell "the history is too short" from "the answer is no", because the first is a reason to
    fetch more and the second is a reason to stop.
    """
    floors = dict(CARRY_GATE if gate is None else gate)
    missing = sorted(name for name in CARRY_GATE if name not in floors)
    if missing:
        raise ValueError(
            f"the gate is missing {', '.join(missing)}: a criterion that was never declared "
            "cannot be met, and defaulting it here would be declaring it after the result"
        )
    unreadable = run.readability()
    criteria: List[Criterion] = []

    declared = float(floors["net_apy"])
    if run.net_apy is None or unreadable:
        criteria.append(
            Criterion(
                "net_apy",
                declared,
                run.net_apy,
                "fraction",
                UNMEASURED,
                unreadable or "the window is empty, so nothing could be annualised",
            )
        )
    else:
        criteria.append(
            Criterion(
                "net_apy",
                declared,
                float(run.net_apy),
                "fraction",
                MET if float(run.net_apy) >= declared else MISSED,
                f"{float(run.net_apy) * 100.0:.2f} % of capital a year against the declared "
                f"{declared * 100.0:.2f} %",
            )
        )

    ceiling = float(floors["max_drawdown"])
    if run.drawdown is None or unreadable:
        criteria.append(
            Criterion(
                "max_drawdown",
                ceiling,
                run.drawdown,
                "fraction",
                UNMEASURED,
                unreadable or "no funding curve was built, so it has no drawdown",
            )
        )
    else:
        criteria.append(
            Criterion(
                "max_drawdown",
                ceiling,
                float(run.drawdown),
                "fraction",
                MET if float(run.drawdown) <= ceiling else MISSED,
                f"{float(run.drawdown) * 100.0:.2f} % of capital against a "
                f"{ceiling * 100.0:.2f} % ceiling on the funding-cash curve",
            )
        )

    floor_share = float(floors["positive_share"])
    if run.positive_share is None or int(run.paid_periods) < MIN_PERIODS:
        criteria.append(
            Criterion(
                "positive_share",
                floor_share,
                run.positive_share,
                "fraction",
                UNMEASURED,
                f"{int(run.paid_periods)} reported period(s) is fewer than the {MIN_PERIODS} "
                "this module requires before a share of them is read",
            )
        )
    else:
        criteria.append(
            Criterion(
                "positive_share",
                floor_share,
                float(run.positive_share),
                "fraction",
                MET if float(run.positive_share) >= floor_share else MISSED,
                f"{float(run.positive_share) * 100.0:.1f} % of the reported periods paid the "
                f"short against a declared {floor_share * 100.0:g} %",
            )
        )

    floor_capacity = float(floors["capacity_usdt"])
    if run.capacity_usdt is None:
        criteria.append(
            Criterion(
                "capacity_usdt",
                floor_capacity,
                None,
                "USDT",
                UNMEASURED,
                "no turnover was read for this contract, and an unread market is not an "
                "empty one",
            )
        )
    else:
        capacity = float(run.capacity_usdt)
        enough = capacity >= floor_capacity and capacity >= float(run.settings.capital)
        reason = (
            f"{capacity:,.0f} USDT of capital deployable against the declared "
            f"{floor_capacity:,.0f} USDT floor"
        )
        if capacity < float(run.settings.capital):
            reason += (
                f", and less than the {float(run.settings.capital):,.0f} USDT this book "
                "actually deploys"
            )
        criteria.append(
            Criterion(
                "capacity_usdt",
                floor_capacity,
                capacity,
                "USDT",
                MET if enough else MISSED,
                reason,
            )
        )
    return Verdict(tuple(criteria))


# ── the declaration, the file, and the line a reader reads ──────────────────


def criteria_of(gate: Optional[Mapping[str, float]] = None) -> Criteria:
    """The declaration this report is written against, as the object the package hashes.

    The gate travels inside the hash, and so do :data:`ACCEPTANCE` and :data:`DOES_NOT_SHOW`:
    changing the ceiling after seeing the result, or dropping the sentence about liquidation,
    changes the fingerprint and makes the report visibly a new claim instead of a quiet edit.
    """
    floors = dict(CARRY_GATE if gate is None else gate)
    return Criteria(
        gate=floors,
        note=(
            "the carry gate: what a hedged pair has to earn, keep and absorb before the "
            "operator's capital goes into it — declared before the first run, and the hash "
            "beside every report is the hash of these terms"
        ),
        acceptance=ACCEPTANCE,
        does_not_show=DOES_NOT_SHOW,
    )


def write_carry(
    path: Any, run: CarryRun, *, criteria: Optional[Criteria] = None
) -> Dict[str, Path]:
    """Write the run and its declaration, then read the report back and verify the hash.

    Two files, one declaration: ``engine.json`` carries the numbers and the hash, and
    ``criteria.json`` beside it carries the same terms without the numbers, so the gate can be
    read by someone who has not seen the result. The bytes are sorted-key, two-space, one
    trailing newline, and carry no timestamp — two runs over the same cache produce the same
    file, which is what makes "the number changed" a fact rather than an impression.
    """
    declaration = criteria or criteria_of()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = declaration.document({"carry": run.as_dict()})
    target.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    criteria_path = declaration.write(target.with_name("criteria.json"))
    problem = verify(json.loads(target.read_text(encoding="utf-8")))
    if problem:
        raise HtxError(f"the carry report at {target} does not verify: {problem}")
    return {"json": target, "criteria": criteria_path}


def _pct(value: Optional[float]) -> str:
    """A ratio as a percentage, or the word this module uses instead of inventing a zero."""
    return "unmeasured" if value is None else f"{float(value) * 100.0:.2f} %"



def format_summary(run: CarryRun) -> str:
    """The run as the few lines a person reads before deciding to open the JSON."""
    book = run.settings
    lines = [
        f"{run.contract}: {float(run.years):.2f} year(s), {int(run.periods)} period(s) held, "
        f"{int(run.paid_periods)} paid, {int(run.unreported_periods)} unreported, "
        f"{int(run.gap_periods)} period(s) missing inside the window",
        f"   {book.notional:,.2f} USDT a leg of a {book.capital:,.2f} USDT book at "
        f"{book.leverage:g}x; the round trip costs {book.round_trip_usdt:,.2f} USDT",
        f"   funding {run.gross_usdt:,.2f} USDT gross, {run.net_usdt:,.2f} USDT net of it",
        f"   net APY {_pct(run.net_apy)} of capital (gross {_pct(run.gross_apy)}), drawdown "
        f"{_pct(run.drawdown)}, funding > 0 in {_pct(run.positive_share)} of the reported "
        "periods",
    ]
    if run.capacity_usdt is None:
        lines.append("   capacity: unmeasured, because no turnover was read for this contract")
    else:
        lines.append(
            f"   capacity {float(run.capacity_usdt):,.0f} USDT deployable at "
            f"{book.participation * 100.0:.2f} % of a {run.turnover_usdt:,.0f} USDT day"
        )
    lines.append(f"   verdict: {run.verdict.overall}")
    for item in run.verdict.criteria:
        lines.append(f"      {item.name:<16} {item.verdict:<11} {item.reason}")
    if run.marks:
        drift = run.marks.get("basis_drift_bps")
        drift_text = "unmeasured" if drift is None else f"{float(drift):.2f} bps"
        lines.append(
            f"   basis marks: {int(run.marks.get('marked_periods') or 0)} period(s), "
            f"{_pct(run.marks.get('covered_share'))} of the window, drift {drift_text}"
        )
    for note in run.warnings:
        lines.append(f"   note: {note}")
    return "\n".join(lines)


# ── the command line: one pair, one answer ─────────────────────────────────


def _turnover_of(universe: Mapping[str, Any], contract: str) -> float:
    """The 24 h quote turnover a snapshot recorded for one contract, or ``0.0`` when absent."""
    code = str(contract).strip().upper()
    for row in universe.get("ranked") or []:
        if isinstance(row, Mapping) and str(row.get("contract_code", "")).upper() == code:
            return float(row.get("turnover") or 0.0)
    return 0.0


def _parser() -> argparse.ArgumentParser:
    """The one verb this file has, and the settings that describe the book rather than the bar.

    The gate itself is deliberately *not* a flag: a ceiling that can be passed on the command
    line is a ceiling that gets moved after a result, and the whole point of hashing it is that
    moving it is visible.
    """
    parser = argparse.ArgumentParser(
        prog="python -m agent.backtest.carry_engine",
        description=(
            "Price one delta-neutral funding pair from the cached history and judge it against "
            "the declared gate. Reads the cache, writes a report, opens no position."
        ),
    )
    parser.add_argument("--contract", required=True, help="the swap contract, e.g. BTC-USDT")
    parser.add_argument(
        "--cache", default=DEFAULT_CACHE, help="the funding cache the collector wrote"
    )
    parser.add_argument(
        "--universe",
        default=DEFAULT_UNIVERSE,
        help="the turnover snapshot the capacity criterion reads",
    )
    parser.add_argument(
        "--turnover",
        type=float,
        default=None,
        help="24 h quote turnover in USDT, overriding the snapshot when one is not there",
    )
    parser.add_argument(
        "--capital", type=float, default=50_000.0, help="the dollars the pair ties up"
    )
    parser.add_argument(
        "--leverage", type=float, default=3.0, help="the perp leg's leverage, at most 3x"
    )
    parser.add_argument(
        "--participation",
        type=float,
        default=PARTICIPATION,
        help=f"the share of a day's turnover the book may be (default {PARTICIPATION:g})",
    )
    parser.add_argument("--perp-csv", default=None, help="cached perp bars, for the basis block")
    parser.add_argument("--spot-csv", default=None, help="cached spot bars, for the basis block")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where the report lands")
    parser.add_argument("--json", action="store_true", help="print the run to stdout as JSON")
    parser.add_argument("--quiet", action="store_true", help="write the files, print nothing")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="feed timeout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read the cache, price the pair, judge it, write it — and exit 0 whatever it found.

    The exit code describes the *question*, never the answer: 0 means the run was measured,
    3 means the funding feed could not be read, 4 means there was not enough history to price,
    and 2 means the settings cannot describe a book at all.
    """
    args = _parser().parse_args(argv)
    code = str(args.contract).strip().upper()
    if not code:
        print("carry-engine: --contract needs a contract code", file=sys.stderr)
        return EXIT_USAGE
    try:
        book = Settings(
            capital=float(args.capital),
            leverage=float(args.leverage),
            participation=float(args.participation),
        )
    except ValueError as exc:
        print(f"carry-engine: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if bool(args.perp_csv) != bool(args.spot_csv):
        print(
            "carry-engine: the basis block needs both --perp-csv and --spot-csv, because a "
            "spread needs two prices and one of them alone is a guess",
            file=sys.stderr,
        )
        return EXIT_USAGE
    try:
        series = fetch_funding_cached(
            code, cache=Path(args.cache), refresh=False, timeout=float(args.timeout)
        )
    except HtxError as exc:
        print(f"carry-engine: {exc}", file=sys.stderr)
        return EXIT_FEED
    if not series.records:
        print(f"carry-engine: {code} has no funding history in {args.cache}", file=sys.stderr)
        return EXIT_SHORT
    marks: Optional[List[Tuple[float, float]]] = None
    if args.perp_csv and args.spot_csv:
        marks = basis_marks(
            load_csv(args.perp_csv, symbol=code),
            load_csv(args.spot_csv, symbol=code),
        )
        if len(marks) < 2:
            print(
                f"carry-engine: the two price files share {len(marks)} bar open(s), which is "
                "not enough to mark a basis",
                file=sys.stderr,
            )
            return EXIT_SHORT
    turnover = (
        float(args.turnover)
        if args.turnover is not None
        else _turnover_of(read_universe(args.universe), code)
    )
    run = simulate(series, settings=book, turnover=turnover, marks=marks)
    written = write_carry(Path(args.out), run)
    if not args.quiet:
        print(format_summary(run))
        print(f"   wrote {written['json']} · {written['criteria']}")
    if args.json:
        print(json.dumps(run.as_dict(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

