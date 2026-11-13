"""The daily book: the same question asked of a hundred pairs at once, every day.

Why this file exists
====================

The hunter's book is one bet, repeated: a panic overshoots and a bar the detector calls
``PANIC`` tends to mean-revert. The measurement in this repo says the bet is *thin*, and it says
it with numbers rather than opinions: a four-hour book returns about +58 bps a trade at costs of
about 40 bps round-trip, on roughly 1.6 trades per pair per year, so a hundred pairs produce
about eight trades a month and the gate needs 155 of them before it can call the edge anything.
A book that trades twice a year per pair cannot pay a salary. It can pay a bonus, and only after
years of waiting.

There is a second question that the same detector's *conclusion* points at, and it is the one
that pays daily: if a hundred coins sit on one calendar, then on any given day some of them have
just fallen and some have just risen, and the difference between the two groups is a bet that
does not need a panic to happen. It needs breadth. That is this module.

What is actually bet
====================

Each rebalance, every symbol in the panel is scored by its own return over ``lookback`` bars,
the scores are ranked across symbols, and the book goes **long the top ``legs`` and short the
bottom ``legs``, with equal notional on each name**. Because both sides are sized off the same
equity, the book starts every rebalance **dollar-neutral**: half its gross is long and half is
short, so a coin-flip market moves both sides by the same amount and the net is roughly flat.
That is the property that makes it a daily business rather than a directional bet — the P&L is
the *spread* between the strongest and the weakest names, and a spread exists in an up market
and a down one.

``sign='reversal'`` (the default) buys the fallen and shorts the risen, which is the same
mean-reversion claim the hunter makes, expressed cross-sectionally instead of episodically.
``sign='momentum'`` is the opposite claim and is in the grid because a version of this file that
only tested the author's prior would be a preference with a CSV next to it. The grid is declared
here, printed whole, and never selected from: every row is a row.

The honesty structure
=====================

Three things in this file exist only to stop it from lying to itself, and each one is cheap:

* :data:`GATE` is declared **before** the first run and is hashed into ``criteria.json`` with
  :class:`~agent.backtest.criteria.Criteria`. A daily book is measured in *marks*, not trades, so
  its floor is per day (5 bps net ≈ 18 %/year) and not the walk-forward's 30 bps per trade — a
  different unit, not a looser gate. The neutrality requirement (``|beta| ≤ 0.2``) is a gate and
  not a comment, because a "neutral" book with a beta of 0.8 is a long book with extra steps.
* every config is scored on the whole window **and** on a held-back tail (``--split``), and the
  verdict count requires both. Splitting costs nothing here because the signal has no fitted
  parameters: the only thing that could be overfit is the choice of lookback and sign, and that
  choice is made visible by printing all of them rather than by pretending it was not made.
* costs are charged on **turnover**, per name, against the declared fee and slippage — so a
  config that churns its whole book every day pays for it in the same column it earns in. The
  rebalance period is a grid axis precisely because that trade-off is the whole question.

What this module does not do
============================

It does not fetch funding, borrow rates, or liquidations, because no code in this repo reads
them: a neutral book on spot needs a borrow to short, and the cost of that borrow is *declared
absent* here rather than assumed away. It does not model impact beyond a constant slippage per
fill (the same constant the single-pair engine uses, so the two books are comparable). It does
not touch ``trader_state.json`` and it cannot move a credit: it writes a table, and a table is a
question for a person to answer.

``python -m agent.backtest.crosssection --top 100 --period 1day`` is the run; ``--candles DIR``
is the same run against a frozen panel, which is how a result is replayed — and a frozen panel
measures its own bar width, because a directory of CSVs does not carry the period it was
downloaded at.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.backtest.criteria import Criteria, verify
from agent.backtest.data import DEFAULT_PERIOD, DEFAULT_TIMEOUT, HTX_ENDPOINT, HtxError
from agent.backtest.panel import Panel, PanelError, fetch_panel, load_panel, write_panel
from agent.backtest.power import DECLARED_ALPHA, DECLARED_POWER, resolution, spread_bps
from agent.backtest.run import EXIT_FEED, EXIT_SHORT, EXIT_USAGE
from agent.backtest.simulated_provider import FEE_RATE, FILL_SLIPPAGE

__all__ = [
    "ACCEPTANCE",
    "BasketReading",
    "BasketResult",
    "BasketSettings",
    "DEFAULT_CAPITAL",
    "DEFAULT_OUT",
    "DOES_NOT_SHOW",
    "FILLS",
    "GATE",
    "LOOKBACKS",
    "MAKER_FEE",
    "Mark",
    "REBALANCES",
    "SIGNS",
    "Window",
    "book",
    "configs",
    "format_table",
    "grid",
    "main",
    "read",
    "verdict_of",
    "write_marks",
]

#: The declared grid. Each axis is short on purpose: a hundred configs would find something in
#: noise with near-certainty, and these can be printed in a table a person actually reads.
LOOKBACKS: Tuple[int, ...] = (3, 7, 14, 30)
REBALANCES: Tuple[int, ...] = (1, 7)
SIGNS: Tuple[str, ...] = ("reversal", "momentum")

#: How an order becomes a position. ``taker`` crosses the spread at the bar's open and pays
#: slippage; ``maker`` posts a limit at the price the decision was made on and is filled only if
#: the fill bar trades *through* it, at that limit. They are not two fee schedules on the same
#: trade — they are two different trades, and the difference is *which* moves you end up holding.
FILLS: Tuple[str, ...] = ("taker", "maker")

#: The maker rate a passive fill pays. 0.02 % is HTX's *fee-tier* rate and not its list price:
#: a VIP-0 spot account pays 0.2 % on both sides, which is the same as :data:`FEE_RATE`, and a
#: maker run quoted at 2 bps is an assumption about the account, not about the venue.
MAKER_FEE = 0.0002

#: What a book starts with when nobody says otherwise — the single-pair engine's own default
#: capital, so the two books are read on the same notional.
DEFAULT_CAPITAL = 1000.0

#: The gate, declared before the first run and hashed into the report.
#:
#: The unit here is a **day**, not a trade, and the difference is worth spelling out: 5 bps a day
#: at 365 marks is 18.2 % a year on the book's own equity, after the declared fee and slippage. A
#: cross-sectional book that cannot clear that is not worth the borrow its shorts need; one that
#: clears it comfortably is a business. ``beta`` is a ceiling compared as ``|beta| <= 0.2``,
#: because it is the criterion that keeps the word "neutral" honest — a "neutral" book with a beta
#: of 0.8 is a long book with extra steps. ``max_drawdown`` is a ceiling too: 5 bps a day that
#: arrives with a 40 % hole is not a book anyone survives holding.
GATE: Dict[str, float] = {
    "edge_bps": 5.0,
    "t_stat": 2.0,
    "sharpe": 1.0,
    "max_drawdown": 0.20,
    "beta": 0.20,
}





#: What a reading of this book would still not have shown — hashed with the gate, so a passing
#: run cannot quietly drop the sentence that says what it did not prove.
DOES_NOT_SHOW: Tuple[str, ...] = (
    "that a cross-section of today's listed pairs is the cross-section of the past: the "
    "universe is survivorship-biased by construction, and the holes are visible but not fixed",
    "anything about borrowing the shorts: spot shorting needs a lender, and no funding, borrow "
    "or liquidation term is charged or modelled here",
    "anything about impact beyond a constant slippage per fill: a hundred-name book selling a "
    "thin alt is a market event the venue would price differently from a percentage",
    "anything about live slippage on rebalance day: every name here is assumed fillable at the "
    "bar's own open, which is exactly what a crowded rebalance is not",
    "the queue: a maker order fills here whenever the bar trades through its price, which is "
    "the fill a one-lot order gets and not the fill a book competing for the same touch gets — "
    "a maker run's whole saving is the spread it never crosses, and nothing here measures how "
    "much of a real queue it would have to wait behind to earn it",
    "that the grid was not chosen after the fact: every declared config is printed, and the "
    "multiple-comparison problem is shown rather than corrected",
    "an independent t: the ranking window is re-read every rebalance and overlaps itself, so "
    "the marks are not independent observations and the t-stat is inflated by up to the square "
    "root of the lookback — the table says by how much, and the gate does not correct for it",
    "a book that a real venue would fill at these prices: an open is a print, not a size, and a "
    "hundred-name rebalance every four hours is exactly the flow a thin alt's open does not "
    "absorb",
    "profit on a real account: a book that clears this gate on daily marks borrows, hedges and "
    "rebalances in a way no code here has ever placed an order for",
)

#: What a passing run would have to satisfy, spelled out so that "pass" is not an opinion.
ACCEPTANCE: Tuple[str, ...] = (
    "dollar-neutral at every rebalance: longs and shorts are sized off the same equity, so the "
    "net notional is zero to within a fee",
    "no lookahead: a decision for a fill at bar i reads closes up to bar i - latency and "
    "nothing later, and a name without a contiguous window is not scored",
    "the book is sized off the equity the decision could see — the marks of the decision bar, "
    "never the close of the bar the order is filled in",
    "costs on turnover: every unit bought or sold pays the declared fee and slippage, "
    "including the exit of a name that left the universe",
    "a hole is a hole: a symbol with no bar on a timestamp is not ranked, not filled, and not "
    "marked forward; a held name the venue stops quoting is closed at its last close",
    "held-back tail reported: every config is scored on the whole window and on the tail after "
    "--split, and a verdict needs both",
    "resolution printed: min_edge_bps sits beside every daily mean, and a book whose measured "
    "mean lies inside its own resolution band cannot pass, whatever the gate's floor says",
)


@dataclass(frozen=True)
class BasketSettings:
    """One book's declaration: what it ranks on, how many names, how often, and what it pays.

    The defaults are the *hypothesis*, not a tuned optimum: a seven-bar reversal, ten names a
    side, rebalanced weekly, marked every bar. The rebalance default is weekly and not daily on
    purpose — a book that re-targets twenty names every day pays the declared fee and slippage
    on its whole gross every day, which is 20 bps a day and more than any cross-sectional edge
    this file has measured. It is still marked *every* bar, so the equity curve answers "what
    did holding this book pay today" even on the days it did not trade.
    """

    lookback: int = 7
    sign: str = "reversal"
    legs: int = 10
    rebalance_bars: int = 7
    capital: float = DEFAULT_CAPITAL
    fee: float = FEE_RATE
    slippage: float = FILL_SLIPPAGE
    latency_bars: int = 1
    min_names: int = 8
    gross: float = 1.0
    max_weight: float = 0.05
    fill: str = "taker"
    maker_fee: float = MAKER_FEE

    def __post_init__(self) -> None:
        if int(self.lookback) < 1:
            raise ValueError(f"lookback {self.lookback!r} is not a window: it must be at least 1")
        if str(self.sign) not in SIGNS:
            raise ValueError(
                f"sign {self.sign!r} is not a claim this book can make; one of {list(SIGNS)} "
                "was expected"
            )
        if int(self.legs) < 1:
            raise ValueError(f"legs {self.legs!r} is not a book: it must be at least 1 a side")
        if int(self.rebalance_bars) < 1:
            raise ValueError(
                f"rebalance_bars {self.rebalance_bars!r} is not a cadence: it must be at least 1"
            )
        if float(self.capital) <= 0.0:
            raise ValueError(f"capital {self.capital!r} cannot fund a book")
        if not 0.0 <= float(self.fee) < 0.05:
            raise ValueError(f"fee {self.fee!r} is not a rate: it must lie in [0, 0.05)")
        if not 0.0 <= float(self.slippage) < 0.05:
            raise ValueError(f"slippage {self.slippage!r} is not a rate: it must lie in [0, 0.05)")
        if int(self.latency_bars) < 1:
            raise ValueError(
                "latency_bars must be at least 1: a book that fills on the bar it decided with "
                "is filling before it decided, and that is a lookahead with a friendly name"
            )
        if int(self.min_names) < 4:
            raise ValueError(
                f"min_names {self.min_names!r} is not a cross-section: two families of two are "
                "not a market, they are a pair of bets"
            )
        if not 0.0 < float(self.gross) <= 2.0:
            raise ValueError(
                f"gross {self.gross!r} is not an exposure: it must lie in (0, 2], and anything "
                "above 1 needs margin this book does not model"
            )
        if not 0.0 < float(self.max_weight) <= 0.5:
            raise ValueError(
                f"max_weight {self.max_weight!r} is not a limit: it must lie in (0, 0.5]"
            )
        if str(self.fill) not in FILLS:
            raise ValueError(
                f"fill {self.fill!r} is not a way to fill an order; one of {list(FILLS)} was "
                "expected, because a book that paid neither the taker's slippage nor the "
                "maker's missed fills would be paying nothing"
            )
        if not 0.0 <= float(self.maker_fee) < 0.05:
            raise ValueError(
                f"maker_fee {self.maker_fee!r} is not a rate: it must lie in [0, 0.05)"
            )

    @property
    def config(self) -> str:
        """The name a table prints — short enough for a column, precise enough to re-run."""
        name = f"{self.sign}-l{int(self.lookback)}-r{int(self.rebalance_bars)}"
        return name if str(self.fill) == "taker" else f"{name}-{self.fill}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "lookback": int(self.lookback),
            "sign": str(self.sign),
            "legs": int(self.legs),
            "rebalance_bars": int(self.rebalance_bars),
            "capital": float(self.capital),
            "fee": float(self.fee),
            "slippage": float(self.slippage),
            "latency_bars": int(self.latency_bars),
            "min_names": int(self.min_names),
            "gross": float(self.gross),
            "max_weight": float(self.max_weight),
            "fill": str(self.fill),
            "maker_fee": float(self.maker_fee),
        }


# ── the book itself ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mark:
    """One bar of the book's life: what it was worth, what it held, and what it paid.

    ``long`` and ``short`` are notional *magnitudes*, so a neutral book prints two equal numbers
    and a reader can see the neutrality instead of trusting a sentence about it. ``traded`` is
    the notional that changed hands to reach this mark, which is the only column that explains
    where the fees went.
    """

    at: float
    equity: float
    cash: float
    long: float
    short: float
    names: int
    traded: float
    fees: float

    @property
    def net(self) -> float:
        """Long minus short — the exposure to the market as a whole, in notional."""
        return float(self.long) - float(self.short)

    def as_row(self) -> Dict[str, Any]:
        return {
            "at": float(self.at),
            "equity": round(float(self.equity), 6),
            "cash": round(float(self.cash), 6),
            "long": round(float(self.long), 6),
            "short": round(float(self.short), 6),
            "net": round(float(self.net), 6),
            "names": int(self.names),
            "traded": round(float(self.traded), 6),
            "fees": round(float(self.fees), 6),
        }


@dataclass
class BasketResult:
    """A finished run: the marks, the weights it held, and the warnings it earned."""

    settings: BasketSettings
    panel: Panel
    marks: List[Mark] = field(default_factory=list)
    fills: int = 0
    unfilled: int = 0
    rebalances: int = 0
    traded: float = 0.0
    fees: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.marks)

    @property
    def equities(self) -> List[float]:
        return [float(mark.equity) for mark in self.marks]

    def returns(self) -> List[float]:
        """The book's own bar-to-bar returns — what every reading in this file is made of."""
        values = self.equities
        return [
            later / earlier - 1.0
            for earlier, later in zip(values, values[1:])
            if float(earlier) > 0.0
        ]

    def turnover(self) -> float:
        """Notional traded over the mean equity held — the number the fee column multiplies."""
        if not self.marks:
            return 0.0
        mean = statistics.fmean([abs(mark.equity) for mark in self.marks]) or 1.0
        return float(self.traded) / float(mean)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "settings": self.settings.as_dict(),
            "marks": len(self.marks),
            "rebalances": int(self.rebalances),
            "fills": int(self.fills),
            "unfilled": int(self.unfilled),
            "traded": round(float(self.traded), 6),
            "fees": round(float(self.fees), 6),
            "turnover": round(self.turnover(), 6),
            "panel": self.panel.as_dict(),
            "warnings": list(self.warnings),
        }


#: How many warnings of one kind are kept before the rest are dropped — a frozen universe with
#: fifty delistings would otherwise bury the table in the same sentence fifty times.
WARNING_CAP = 12


def _note(warnings: List[str], sentence: str) -> None:
    """Keep a warning, but not ten thousand copies of it."""
    if len(warnings) < WARNING_CAP:
        warnings.append(sentence)


def _mark_price(panel: Panel, symbol: str, at: float) -> Optional[float]:
    """A held name's price at this bar, or its last published close if the venue went quiet.

    The fallback is not a forward fill: it is the price the book would actually be carrying,
    and the very next step of :func:`book` closes the name, because a position in something the
    venue stopped quoting is not a position, it is a claim on a lawsuit.
    """
    price = panel.close(symbol, at)
    if price is not None:
        return float(price)
    last = panel.last_at(symbol, at)
    if last is None:
        return None
    return panel.close(symbol, last)


def _fill_price(price: float, delta: float, slippage: float) -> float:
    """What a fill actually gets: a buy pays up, a sell gives up, both by the same constant."""
    if delta > 0.0:
        return float(price) * (1.0 + float(slippage))
    return float(price) * (1.0 - float(slippage))


def _target_weights(scores: Dict[str, float], settings: BasketSettings) -> Dict[str, float]:
    """Long the top ``legs`` scores, short the bottom ``legs``, equal weight, dollar-neutral.

    Ties are broken by symbol name so that two runs of the same panel hold the same book: an
    order that depended on dict insertion would make the replay check meaningless in exactly
    the case — a hundred names with rounded scores — where it matters most.

    ``max_weight`` is the risk rule that keeps a thin universe honest: when fewer than ``legs``
    names a side are eligible the book holds *fewer names at the same weight* rather than the
    same number of names at a bigger one, so a panel that lost half its symbols shrinks the
    book instead of concentrating it.
    """
    if not scores:
        return {}
    ordered = sorted(scores.items(), key=lambda pair: (-float(pair[1]), pair[0]))
    per_side = min(int(settings.legs), len(ordered) // 2)
    if per_side < 1:
        return {}
    weight = min(float(settings.gross) / 2.0 / per_side, float(settings.max_weight))
    if weight <= 0.0:
        return {}
    out: Dict[str, float] = {}
    for name, _ in ordered[:per_side]:
        out[name] = +weight
    for name, _ in ordered[-per_side:]:
        out[name] = -weight
    return out


def book(panel: Panel, settings: Optional[BasketSettings] = None) -> BasketResult:
    """Run one market-neutral basket over a panel, marking every bar.

    The order of operations inside one bar *is* the contract, and it is written down here
    because every way of getting it wrong is a way of making money in a backtest that nobody
    could make in a market:

    1. **close what the venue stopped quoting** — before the decision, because a name with no
       bar today cannot be held today, and leaving it is a fact rather than an assumption;
    2. **decide** from closes up to ``i - latency``, never from the fill bar's own close;
    3. **fill** at the fill bar's *open*, with slippage against the book and the fee on the
       notional — or, in ``fill='maker'`` mode, **post** a limit at the decision's own close and
       be filled only if the bar trades through it, at that limit, at the maker rate and with
       no slippage;
    4. **mark** at the fill bar's close.

    The maker mode is not the taker mode with a smaller number in it. A taker buys what the bar
    opened at; a maker buys only if the bar comes to *him*, and the trades that come to him are
    exactly the ones already moving against him. Modelling the fee and not the fill would be the
    most flattering thing this file could do, so the fill is modelled, the missed ones are
    counted in :attr:`BasketResult.unfilled`, and the filled ones pay their own limit and never
    the better of it and the open.

    Steps 3 and 4 in that order are what make a bar's P&L a real holding period instead of the
    same close counted twice: a book bought at the open earns the open-to-close move and pays
    for the turn, and a book bought at the close would earn neither.
    """
    config = settings or BasketSettings()
    if not panel.ats:
        raise PanelError("this panel has no bars, so there is no calendar to trade on")
    first = int(config.lookback) + int(config.latency_bars)
    if first >= len(panel.ats):
        raise PanelError(
            f"the panel holds {len(panel.ats)} bars and this book needs {first} before its "
            "first fill (lookback plus latency); freeze a longer panel or shorten the window"
        )
    cash = float(config.capital)
    units: Dict[str, float] = {}
    result = BasketResult(settings=config, panel=panel)
    last_fill = -(10 ** 9)
    saw_negative_cash = False
    seen_thin = False

    def settle(name: str, target: float, price: float, fee_rate: float) -> Tuple[float, float]:
        """Move one name to ``target`` at ``price``, paying ``fee_rate`` on what moved."""
        nonlocal cash
        held = float(units.get(name, 0.0))
        delta = float(target) - held
        if delta == 0.0:
            return 0.0, 0.0
        notional = delta * float(price)
        fee = abs(notional) * float(fee_rate)
        cash -= notional + fee
        if abs(float(target)) <= 1e-15:
            units.pop(name, None)
        else:
            units[name] = float(target)
        return abs(notional), fee

    def trade(name: str, target: float, price: float) -> Tuple[float, float]:
        """Taker: cross the spread at the bar's open, paying slippage and the declared fee.

        The delisting exit goes through here in *both* modes, because an order for a name the
        venue stopped quoting is not an order that can rest: it is a position that has to be
        got out of at whatever the last print allows.
        """
        held = float(units.get(name, 0.0))
        fill = _fill_price(price, float(target) - held, float(config.slippage))
        return settle(name, target, fill, float(config.fee))

    def post(name: str, target: float, bar: Any, limit: float) -> Tuple[float, float, bool]:
        """Maker: rest at ``limit`` and be filled only if the bar trades through it.

        ``limit`` is the close the decision was made on — the price the book actually saw, and
        the price it is filled at. The reach test is *strict*: a bar whose low merely equals a
        buy limit proves the price was available, not that this book's place in the queue was
        reached, so the order needs the bar to trade below it. An order the bar never reaches is
        not filled *at all*, and that is the whole honest content of the maker idea: the cheaper
        fee is not a discount on the same trade, it is payment for standing in the queue and
        being handed exactly the trades that are going against you.

        The fill is at the limit and never at the better of the limit and the open. A resting
        bid that the market gaps through is filled at its own price — the improvement in a gap
        belongs to whoever crossed into it — so crediting the book ``min(open, limit)`` would
        hand it the gap for free on precisely the bars it was about to be wrong on.
        """
        delta = float(target) - float(units.get(name, 0.0))
        if delta == 0.0:
            return 0.0, 0.0, True
        if delta > 0.0:
            if float(bar.low) >= float(limit):
                return 0.0, 0.0, False
        else:
            if float(bar.high) <= float(limit):
                return 0.0, 0.0, False
        traded, fee = settle(name, target, float(limit), float(config.maker_fee))
        return traded, fee, True

    for index in range(first, len(panel.ats)):
        at = float(panel.ats[index])
        traded_here = 0.0
        fees_here = 0.0

        # 1. a held name with no bar today leaves at its last published price
        for name in sorted(units):
            if panel.bar(name, at) is not None:
                continue
            price = _mark_price(panel, name, at)
            if price is None:
                units.pop(name, None)
                _note(
                    result.warnings,
                    f"{name} stopped publishing bars and has no last price the book can use; "
                    "the position is dropped rather than marked at a guess",
                )
                continue
            traded, fee = trade(name, 0.0, float(price))
            traded_here += traded
            fees_here += fee
            if traded > 0.0:
                result.fills += 1
            _note(
                result.warnings,
                f"{name} had no bar at {int(at)} and was closed at its last close "
                f"{float(price):g}, paying {fee:.4f} to leave",
            )

        # 2. decide on closes up to i - latency, and 3. fill at this bar's open
        if index - last_fill >= int(config.rebalance_bars) or not units:
            decision_at = panel.at(index - int(config.latency_bars))
            # Size the book off the equity the decision could see: the marks of the *decision*
            # bar. This bar's close is the future the fill is about to be marked into, and
            # sizing an order sent at this bar's open off this bar's close pays the book for
            # knowing the bar it is trading in -- a leak of the same kind as trading on the
            # decision's own close, only quieter.
            equity_now = cash
            for name in sorted(units):
                price = _mark_price(
                    panel, name, float(decision_at) if decision_at is not None else at
                )
                if price is not None:
                    equity_now += units[name] * float(price)
            target_weights: Dict[str, float] = {}
            if decision_at is not None and equity_now > 0.0:
                sign = -1.0 if str(config.sign) == "reversal" else 1.0
                scores = {
                    name: sign * float(value)
                    for name, value in panel.returns_at(decision_at, int(config.lookback)).items()
                }
                if len(scores) < int(config.min_names):
                    if not seen_thin:
                        seen_thin = True
                        _note(
                            result.warnings,
                            f"{len(scores)} name(s) had a contiguous {int(config.lookback)}-bar "
                            f"window at {int(decision_at)} and min_names is "
                            f"{int(config.min_names)}; the book stands aside on such bars "
                            "rather than betting the cross-section on a handful of names",
                        )
                else:
                    target_weights = _target_weights(scores, config)
            maker = str(config.fill) == "maker"
            wanted: Dict[str, float] = {}
            limits: Dict[str, float] = {}
            for name, weight in sorted(target_weights.items()):
                # A taker sizes off the open it is about to cross; a maker sizes off the close it
                # is about to post at, because the open does not exist yet when the order is sent.
                seen = panel.close(name, decision_at) if decision_at is not None else None
                price = float(seen) if (maker and seen is not None) else panel.open(name, at)
                if price is None or float(price) <= 0.0:
                    _note(
                        result.warnings,
                        f"{name} was ranked for the book at {int(at)} and had no bar to fill "
                        "in; it was skipped rather than filled at a price from another day",
                    )
                    continue
                if maker:
                    limits[name] = float(price)
                wanted[name] = float(weight) * float(equity_now) / float(price)
            for name in sorted(units):
                wanted.setdefault(name, 0.0)
            if maker:
                # A name the ranking dropped is still sold, and it is sold the same way: at a
                # limit posted on the close the decision saw, and only if the bar reaches it.
                for name in sorted(wanted):
                    if name in limits:
                        continue
                    seen = panel.close(name, decision_at) if decision_at is not None else None
                    if seen is not None and float(seen) > 0.0:
                        limits[name] = float(seen)
            dust = 1e-9 * max(abs(equity_now), 1.0)
            for name in sorted(wanted):
                price = panel.open(name, at)
                bar = panel.bar(name, at)
                if price is None or price <= 0.0 or bar is None:
                    continue
                if abs(wanted[name] - float(units.get(name, 0.0))) * float(price) <= dust:
                    continue
                if maker:
                    limit = limits.get(name)
                    if limit is None:
                        result.unfilled += 1
                        continue
                    traded, fee, filled = post(name, wanted[name], bar, float(limit))
                    if not filled:
                        result.unfilled += 1
                else:
                    traded, fee = trade(name, wanted[name], float(price))
                traded_here += traded
                fees_here += fee
                if traded > 0.0:
                    result.fills += 1
            if target_weights:
                result.rebalances += 1
            last_fill = index

        # 4. mark at this bar's close
        equity = cash
        long_value = 0.0
        short_value = 0.0
        for name in sorted(units):
            price = _mark_price(panel, name, at)
            if price is None:
                continue
            value = units[name] * float(price)
            equity += value
            if value > 0.0:
                long_value += value
            else:
                short_value -= value
        result.marks.append(
            Mark(
                at=at,
                equity=equity,
                cash=cash,
                long=long_value,
                short=short_value,
                names=len(units),
                traded=traded_here,
                fees=fees_here,
            )
        )
        result.traded += traded_here
        result.fees += fees_here
        if cash < 0.0 and not saw_negative_cash:
            saw_negative_cash = True
            _note(
                result.warnings,
                "the book's cash went negative: this model charges no margin and no borrow, so "
                "a negative cash balance here is a loan it was never billed for",
            )
        if equity <= 0.0:
            _note(
                result.warnings,
                f"the book's equity reached {equity:.2f} at {int(at)} and it stopped there; a "
                "curve below zero is not a loss, it is a margin call nobody in this model sent",
            )
            break
    if str(config.fill) == "maker" and result.unfilled:
        _note(
            result.warnings,
            f"{result.unfilled} maker order(s) were never traded through on the fill bar, "
            f"against {result.fills} that were: the cheaper fee is paid for in cash that never "
            "got put to work, and that is a cost the fee column does not contain",
        )
    return result


# ── the readings ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Window:
    """One slice of a book's marks, read the way the gate reads it.

    ``days`` is the number of *calendar days* the window covers, not the number of marks: a
    4-hour panel marks six times a day, and a table that called six marks a day would make every
    per-day reading a lie by a factor of six. ``mean_bps``, ``sigma_bps`` and ``min_edge_bps``
    are all per day too, so two windows over two different bar widths are comparable; ``t_stat``
    is scale-free and stays on the marks, where the returns actually live.
    """

    days: int
    mean_bps: float
    t_stat: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    beta: float
    correlation: float
    roi: float
    sigma_bps: float
    min_edge_bps: float
    measurable: bool
    fees: float
    turnover: float
    names: float
    failures: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> Dict[str, Any]:
        return {
            "days": int(self.days),
            "mean_bps": float(self.mean_bps),
            "t_stat": float(self.t_stat),
            "sharpe": float(self.sharpe),
            "max_drawdown": float(self.max_drawdown),
            "hit_rate": float(self.hit_rate),
            "beta": float(self.beta),
            "correlation": float(self.correlation),
            "roi": float(self.roi),
            "sigma_bps": float(self.sigma_bps),
            "min_edge_bps": float(self.min_edge_bps),
            "measurable": bool(self.measurable),
            "fees": float(self.fees),
            "turnover": float(self.turnover),
            "names": float(self.names),
            "passed": bool(self.passed),
            "failures": list(self.failures),
        }


def _per_day_bps(bar_return: float, per_day: float) -> float:
    """A per-bar mean return expressed per day — by compounding, not by multiplying.

    ``per_day`` bars of ``r`` each is ``(1 + r) ** per_day - 1``, and not ``r * per_day``. At
    four bps a bar the two agree to the fourth decimal, so the simpler arithmetic would look
    identical on every run that passes; they differ exactly where a run is big enough to be
    doubted, which is where the reading has to be right. A bar that lost everything has no
    daily equivalent: ``-100 %`` is returned and the book's equity curve shows the rest.
    """
    value = float(bar_return)
    if not math.isfinite(value):
        return 0.0
    if value <= -1.0:
        return -1e4
    if float(per_day) <= 0.0:
        return float(1e4 * value)
    return float(1e4 * ((1.0 + value) ** float(per_day) - 1.0))


def _t_stat(values: Sequence[float]) -> float:
    """Mean over its own standard error — ``0`` when there is nothing to divide by."""
    count = len(values)
    if count < 2:
        return 0.0
    spread = statistics.stdev(float(value) for value in values)
    if spread <= 0.0:
        return 0.0
    return float(statistics.fmean(values)) / spread * math.sqrt(count)


def _sharpe(values: Sequence[float], marks_per_year: float) -> float:
    """Annualised Sharpe of daily marks — the same reading the single-pair walk-forward uses."""
    if len(values) < 3:
        return 0.0
    spread = statistics.stdev(float(value) for value in values)
    if spread <= 0.0:
        return 0.0
    return float(statistics.fmean(values)) / spread * math.sqrt(max(marks_per_year, 1.0))


def _drawdown(equities: Sequence[float]) -> float:
    peak = 0.0
    deepest = 0.0
    for value in equities:
        equity = float(value)
        if equity > peak:
            peak = equity
        if peak > 0.0:
            deepest = max(deepest, (peak - equity) / peak)
    return float(deepest)


def _beta(book: Sequence[float], market: Sequence[float]) -> Tuple[float, float]:
    """``(beta, correlation)`` of the book's returns against the equal-weight universe.

    This is the number that decides whether the word "neutral" in this file means anything. A
    dollar-neutral book that nevertheless moves with the market — because its shorts are illiquid
    alts that fall when everything falls — has a beta near one and is a directional bet with a
    second leg, and no amount of vocabulary changes that. ``(0, 0)`` when the market did not
    move: there is nothing to be correlated with.
    """
    pairs = [(float(a), float(b)) for a, b in zip(book, market)]
    if len(pairs) < 3:
        return 0.0, 0.0
    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    spread = sum((value - right_mean) ** 2 for value in right)
    if spread <= 0.0:
        return 0.0, 0.0
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in pairs) / len(pairs)
    variance = spread / len(pairs)
    beta = covariance / variance
    left_sd = math.sqrt(sum((value - left_mean) ** 2 for value in left) / len(left))
    right_sd = math.sqrt(variance)
    if left_sd <= 0.0 or right_sd <= 0.0:
        return float(beta), 0.0
    return float(beta), float(covariance / (left_sd * right_sd))


MARKET_CACHE = "_crosssection_market"


def _market_series(panel: Panel) -> Dict[float, float]:
    """The equal-weight universe's return at every bar, computed once per panel.

    Equal-weight and not cap-weight on purpose: this book has no notion of market capitalisation
    (the venue does not publish one), and the honest benchmark for a book that holds twenty alts
    in equal size is the twenty alts it could have held instead.

    It is cached on the panel because the grid reads every config three times (whole window,
    inside, outside) and the benchmark is the same series for all of them: recomputing it would
    multiply a hundred-symbol scan by forty-eight for a number that never changes.
    """
    cached = getattr(panel, MARKET_CACHE, None)
    if isinstance(cached, dict):
        return cached
    series: Dict[float, float] = {}
    for at in panel.ats:
        values = [float(value) for value in panel.returns_at(float(at), 1).values()]
        series[float(at)] = float(statistics.fmean(values)) if values else 0.0
    setattr(panel, MARKET_CACHE, series)
    return series


def _market_returns(panel: Panel, marks: Sequence[Mark]) -> List[float]:
    """The market's move between consecutive marks — the series the beta is read against.

    A pair of marks that is not one interval apart is a gap in the panel, and the market's move
    across a gap is not this book's holding period either: it is recorded as zero rather than
    guessed, so a hole cannot manufacture a beta.
    """
    series = _market_series(panel)
    step = float(panel.interval)
    out: List[float] = []
    for earlier, later in zip(marks, marks[1:]):
        if step > 0.0 and abs((float(later.at) - float(earlier.at)) - step) > step:
            out.append(0.0)
            continue
        out.append(float(series.get(float(later.at), 0.0)))
    return out



def read(
    result: BasketResult,
    *,
    start: int = 0,
    end: Optional[int] = None,
    gate: Dict[str, float] = GATE,
) -> Window:
    """Read one slice of a finished book against the gate, and say which checks failed.

    The failures are sentences and not booleans, because the useful output of a run like this is
    never "fail": it is *which* of the five things stopped it, since a book that fails on
    drawdown is a risk problem and one that fails on ``t`` is a sample problem, and those two
    want opposite next steps.
    """
    marks = list(result.marks)
    stop = len(marks) if end is None else max(0, min(int(end), len(marks)))
    begin = max(0, min(int(start), stop))
    window = marks[begin:stop]
    floor = float(gate.get("edge_bps", 0.0))
    if len(window) < 2:
        return Window(
            days=0,
            mean_bps=0.0,
            t_stat=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            hit_rate=0.0,
            beta=0.0,
            correlation=0.0,
            roi=0.0,
            sigma_bps=0.0,
            min_edge_bps=math.inf,
            measurable=False,
            fees=0.0,
            turnover=0.0,
            names=0.0,
            failures=("the window holds fewer than two marks, so it has no returns to read",),
        )
    equities = [float(mark.equity) for mark in window]
    returns = [
        later / earlier - 1.0
        for earlier, later in zip(equities, equities[1:])
        if float(earlier) > 0.0
    ]
    # The gate is written per *day*; the marks are per *bar*. On a daily panel those are the same
    # number, which is exactly why this hid: a 4-hour panel marks six times a day, so a mean of
    # 6.9 bps a bar is +41 bps a day, and reading one as the other mis-states the book by the
    # width of its own bar. The conversion happens once, here, and it is a compounding one.
    step = float(result.panel.interval)
    per_day = (86400.0 / step) if step > 0.0 else 1.0
    days = int(round(len(returns) / per_day)) if per_day > 0.0 else len(returns)
    sigma_bps = spread_bps(returns) * math.sqrt(per_day) if per_day > 0.0 else 0.0
    mean_bps = _per_day_bps(statistics.fmean(returns), per_day) if returns else 0.0
    per_year = 365.25 * per_day
    beta, correlation = _beta(returns, _market_returns(result.panel, window))
    drawdown = _drawdown(equities)
    # ``_t_stat`` is deliberately *not* converted: mean and standard error are both per bar, so
    # the ratio is the same number at any bar width. Only the count of independent bets changes,
    # and that is what ``days`` is for.
    heard = resolution(days, sigma_bps, floor, alpha=DECLARED_ALPHA, power=DECLARED_POWER)
    mean_equity = statistics.fmean([abs(value) for value in equities]) or 1.0
    traded = sum(float(mark.traded) for mark in window)
    failures: List[str] = []
    if mean_bps < floor:
        failures.append(f"edge {mean_bps:+.1f} bps/day < {floor:.0f} bps")
    if _t_stat(returns) < float(gate.get("t_stat", 0.0)):
        failures.append(f"t {_t_stat(returns):+.2f} < {float(gate.get('t_stat', 0.0)):.1f}")
    sharpe = _sharpe(returns, per_year)
    if sharpe < float(gate.get("sharpe", 0.0)):
        failures.append(f"sharpe {sharpe:.2f} < {float(gate.get('sharpe', 0.0)):.1f}")
    if drawdown > float(gate.get("max_drawdown", 1.0)):
        failures.append(
            f"drawdown {drawdown:.1%} > {float(gate.get('max_drawdown', 1.0)):.0%}"
        )
    if abs(beta) > float(gate.get("beta", 1.0)):
        failures.append(
            f"|beta| {abs(beta):.2f} > {float(gate.get('beta', 1.0)):.2f}: the book is not "
            "neutral to the universe it holds"
        )
    # The resolution question, stated exactly: a reading is *resolved* when the book's own
    # measured mean is at least as wide as the smallest edge its sample could have named 90 % of
    # the time. Below that, the mean is inside the noise band and the honest word is "unmeasured",
    # not "edge" — and above it the gate has been cleared by a margin the sample itself vouches
    # for, whatever the floor happened to be. This is deliberately *not* the looser rule "pass if
    # the mean clears the floor": a mean of 6 bps against a floor of 5, on a sample that resolves
    # 7, is a coin that landed well, and this file does not fund coins.
    resolved = math.isfinite(heard.min_edge_bps) and mean_bps >= float(heard.min_edge_bps)
    if not resolved:
        band = (
            "no dispersion, so nothing to resolve"
            if math.isinf(heard.min_edge_bps)
            else (
                f"{heard.min_edge_bps:.1f} bps at alpha {DECLARED_ALPHA:.2f} / power "
                f"{DECLARED_POWER:.0%}"
            )
        )
        failures.append(
            f"unresolvable: {days} day(s) at {sigma_bps:.0f} bps of daily spread "
            f"resolve {band}, and the measured {mean_bps:+.1f} bps/day is inside that band — "
            "raise the count, not the claim"
        )
    return Window(
        days=days,
        mean_bps=float(mean_bps),
        t_stat=float(_t_stat(returns)),
        sharpe=float(sharpe),
        max_drawdown=float(drawdown),
        hit_rate=(sum(1 for value in returns if value > 0.0) / len(returns)) if returns else 0.0,
        beta=float(beta),
        correlation=float(correlation),
        roi=float(equities[-1] / equities[0] - 1.0) if equities[0] > 0.0 else 0.0,
        sigma_bps=float(sigma_bps),
        min_edge_bps=float(heard.min_edge_bps),
        measurable=bool(resolved),
        fees=float(sum(float(mark.fees) for mark in window)),
        turnover=float(traded / mean_equity),
        names=float(statistics.fmean([float(mark.names) for mark in window])),
        failures=tuple(failures),
    )



@dataclass
class BasketReading:
    """One config's whole reading: the window, the held-back tail, and the verdict.

    The verdict needs both halves on purpose. A config that clears the gate over the whole window
    and loses money in the last third of it is a config that was described by its own history and
    then stopped being true — which is the most common way a cross-sectional backtest lies, and
    the cheapest one to catch.
    """

    config: str
    settings: BasketSettings
    full: Window
    inside: Window
    outside: Window
    rebalances: int = 0
    fills: int = 0
    unfilled: int = 0
    traded: float = 0.0
    fees: float = 0.0
    warnings: Tuple[str, ...] = ()
    failures: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "settings": self.settings.as_dict(),
            "rebalances": int(self.rebalances),
            "fills": int(self.fills),
            "unfilled": int(self.unfilled),
            "traded": float(self.traded),
            "fees": float(self.fees),
            "passed": bool(self.passed),
            "failures": list(self.failures),
            "full": self.full.as_dict(),
            "inside": self.inside.as_dict(),
            "outside": self.outside.as_dict(),
            "warnings": list(self.warnings),
        }

    def as_row(self) -> Dict[str, Any]:
        """The flat row a CSV wants — one line per config, both halves side by side."""
        return {
            "config": self.config,
            "sign": str(self.settings.sign),
            "lookback": int(self.settings.lookback),
            "rebalance_bars": int(self.settings.rebalance_bars),
            "fill": str(self.settings.fill),
            "fills": int(self.fills),
            "unfilled": int(self.unfilled),
            "days": int(self.full.days),
            "mean_bps": round(float(self.full.mean_bps), 4),
            "t_stat": round(float(self.full.t_stat), 4),
            "sharpe": round(float(self.full.sharpe), 4),
            "max_drawdown": round(float(self.full.max_drawdown), 6),
            "hit_rate": round(float(self.full.hit_rate), 4),
            "beta": round(float(self.full.beta), 4),
            "correlation": round(float(self.full.correlation), 4),
            "roi": round(float(self.full.roi), 6),
            "sigma_bps": round(float(self.full.sigma_bps), 4),
            "min_edge_bps": (
                ""
                if math.isinf(self.full.min_edge_bps)
                else round(float(self.full.min_edge_bps), 4)
            ),
            "fees": round(float(self.full.fees), 4),
            "turnover": round(float(self.full.turnover), 4),
            "names": round(float(self.full.names), 2),
            "inside_days": int(self.inside.days),
            "inside_bps": round(float(self.inside.mean_bps), 4),
            "inside_t": round(float(self.inside.t_stat), 4),
            "outside_days": int(self.outside.days),
            "outside_bps": round(float(self.outside.mean_bps), 4),
            "outside_t": round(float(self.outside.t_stat), 4),
            "outside_hit": round(float(self.outside.hit_rate), 4),
            "passed": bool(self.passed),
            "failures": " | ".join(self.failures),
        }



def configs(
    *,
    lookbacks: Sequence[int] = LOOKBACKS,
    rebalances: Sequence[int] = REBALANCES,
    signs: Sequence[str] = SIGNS,
    base: Optional[BasketSettings] = None,
) -> List[BasketSettings]:
    """The declared grid, in a fixed order — every config that will be printed, and no other.

    Nothing here is selected. The whole point of a declared grid is that the *reader* can see the
    rows and count how many look like edges, instead of the author showing them the one that does.
    """
    template = base or BasketSettings()
    out: List[BasketSettings] = []
    for sign in signs:
        for lookback in lookbacks:
            for rebalance in rebalances:
                out.append(
                    replace(
                        template,
                        sign=str(sign),
                        lookback=int(lookback),
                        rebalance_bars=int(rebalance),
                    )
                )
    return out


def _overlap_note(settings: "BasketSettings", window: "Window") -> str:
    """The sentence a t-stat needs whenever the ranking windows overlap themselves.

    ``Window.t_stat`` divides one mean by one standard error over the marks, so it counts every
    mark as an observation. A 14-bar ranking window rebalanced every bar is not 14 independent
    observations per 14 bars: it is one, restated. The adjustment here is the blunt one —
    ``t / sqrt(lookback)`` — and it is disclosed rather than applied to the gate, because a gate
    that quietly rewrote its own arithmetic after seeing the number would not be a gate.
    """
    lookback = int(settings.lookback)
    if lookback <= 1:
        return (
            "the ranking window is a single bar, so the marks are as independent as the bars "
            "are and no overlap adjustment applies"
        )
    raw = float(window.t_stat)
    scaled = raw / math.sqrt(float(lookback)) if math.isfinite(raw) else raw
    return (
        f"the {lookback}-bar ranking window is re-read every "
        f"{int(settings.rebalance_bars)} bar(s) and overlaps itself: the t of {raw:+.2f} counts "
        f"every mark as an independent observation, and if the windows were independent the "
        f"honest t would be nearer {scaled:+.2f} (t/sqrt({lookback})) — read the gate's t>=2 "
        "with that in hand"
    )


def grid(
    panel: Panel,
    *,
    lookbacks: Sequence[int] = LOOKBACKS,
    rebalances: Sequence[int] = REBALANCES,
    signs: Sequence[str] = SIGNS,
    base: Optional[BasketSettings] = None,
    split: float = 0.6,
    gate: Dict[str, float] = GATE,
) -> List[BasketReading]:
    """Run every declared config over one panel and read each on both halves of its window.

    The split is a fraction of the *marks*, so ``0.6`` means the first sixty percent of the
    trading days are described and the last forty are held back. It is not a walk-forward — the
    signal has no parameters to refit — and saying so matters: what the tail tests is whether the
    effect *persisted*, not whether a fitted model generalised.
    """
    if not 0.0 < float(split) < 1.0:
        raise ValueError(f"split {split!r} is not a fraction: it must lie in (0, 1)")
    readings: List[BasketReading] = []
    for settings in configs(
        lookbacks=lookbacks, rebalances=rebalances, signs=signs, base=base
    ):
        result = book(panel, settings)
        total = len(result.marks)
        cut = int(round(total * float(split))) if total else 0
        full = read(result, gate=gate)
        inside = read(result, start=0, end=cut, gate=gate)
        outside = read(result, start=cut, end=total, gate=gate)
        failures = list(full.failures)
        if outside.days < 30:
            failures.append(
                f"the held-back tail holds {outside.days} day(s): too few to confirm anything"
            )
        elif outside.mean_bps <= 0.0:
            failures.append(
                f"the held-back tail returned {outside.mean_bps:+.1f} bps/day: the effect did "
                "not persist past the split"
            )
        readings.append(
            BasketReading(
                config=settings.config,
                settings=settings,
                full=full,
                inside=inside,
                outside=outside,
                rebalances=int(result.rebalances),
                fills=int(result.fills),
                unfilled=int(result.unfilled),
                traded=float(result.traded),
                fees=float(result.fees),
                warnings=tuple(result.warnings) + (_overlap_note(settings, full),),
                failures=tuple(failures),
            )
        )
    return readings


def verdict_of(readings: Sequence[BasketReading]) -> str:
    """One sentence saying how many of the declared configs survived, and what that means.

    This sentence is the honest output of the whole module, so it is written to be hard to
    misread: the count, the denominator, and the reminder that the denominator is the number of
    tries rather than the number of edges.
    """
    if not readings:
        return "no config was run, so there is nothing to say about the book"
    passed = [reading for reading in readings if reading.passed]
    tried = len(readings)
    if not passed:
        return (
            f"0 of {tried} declared configs clear the gate and hold their sign out of sample. "
            "That is a result: this cross-section, at these costs, is not a daily business, and "
            "the honest move is to change the question rather than the threshold."
        )
    names = ", ".join(reading.config for reading in passed[:4])
    if len(passed) > 4:
        names += f", +{len(passed) - 4} more"
    return (
        f"{len(passed)} of {tried} declared configs clear the gate and hold their sign out of "
        f"sample ({names}). With {tried} tries, some pass is expected by chance alone: the "
        "survivors are a hypothesis to take to a fresh panel, not a book to fund."
    )



def _cells(reading: BasketReading) -> List[str]:
    """The table's row for one config — every column a reading that changed the decision."""
    full = reading.full
    drawn = "inf" if math.isinf(full.min_edge_bps) else f"{full.min_edge_bps:.0f}"
    return [
        reading.config,
        f"{full.days}",
        f"{full.mean_bps:+.1f}",
        f"{full.t_stat:+.2f}",
        f"{full.sharpe:.2f}",
        f"{full.max_drawdown:.1%}",
        f"{full.hit_rate:.0%}",
        f"{full.beta:+.2f}",
        f"{full.fees:.2f}",
        f"{full.turnover:.2f}",
        f"{full.names:.0f}",
        f"{drawn}",
        f"{reading.inside.mean_bps:+.1f}",
        f"{reading.outside.mean_bps:+.1f}",
        f"{reading.outside.t_stat:+.2f}",
        "pass" if reading.passed else "fail",
    ]


HEADER: Tuple[str, ...] = (
    "config",
    "days",
    "bps",
    "t",
    "sharpe",
    "dd",
    "hit",
    "beta",
    "fees$",
    "turn",
    "names",
    "res",
    "in bps",
    "out bps",
    "out t",
    "verdict",
)


def format_table(readings: Sequence[BasketReading]) -> str:
    """The table, aligned — the whole declared grid and nothing but it."""
    rows = [_cells(reading) for reading in readings]
    widths = [
        max(len(HEADER[column]), *(len(row[column]) for row in rows)) if rows else len(HEADER[column])
        for column in range(len(HEADER))
    ]
    lines = ["  ".join(cell.rjust(widths[i]) for i, cell in enumerate(HEADER))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _write_csv(path: Any, readings: Sequence[BasketReading], *, delimiter: str = ",") -> Path:
    """One row per config — the table, in a file a spreadsheet can argue with."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [reading.as_row() for reading in readings]
    fields = list(rows[0]) if rows else [
        "config",
        "mean_bps",
        "t_stat",
        "passed",
        "failures",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n", delimiter=delimiter
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


def write_marks(path: Any, result: BasketResult, *, delimiter: str = ",") -> Path:
    """The equity curve of one run — the file an operator opens when a number looks wrong."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(Mark(at=0.0, equity=0.0, cash=0.0, long=0.0, short=0.0, names=0, traded=0.0, fees=0.0).as_row())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n", delimiter=delimiter
        )
        writer.writeheader()
        for mark in result.marks:
            writer.writerow(mark.as_row())
    return target



# ── the command line ────────────────────────────────────────────────────────

#: Where the table, the JSON and the criteria land unless ``--out`` says otherwise.
DEFAULT_OUT = "data/crosssection"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.backtest.crosssection",
        description=(
            "Run the declared cross-sectional book over a universe: long the strongest names, "
            "short the weakest, dollar-neutral, marked every bar, read against a gate declared "
            "before the run."
        ),
    )
    parser.add_argument(
        "--candles",
        default="",
        help="a directory of <pair>.csv files to read instead of the venue (offline, replayable)",
    )
    parser.add_argument(
        "--symbols", default="", help="a comma-separated universe instead of --top"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="take the venue's top N pairs by 24h turnover; required unless --symbols or "
        "--candles is given",
    )
    parser.add_argument("--quote", default="usdt", help="the quote currency of the universe")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help=(
            "skip this many of the venue's top pairs by turnover, so --top 100 --skip 100 is "
            "ranks 101-200: a universe that is not the one a previous run already looked at"
        ),
    )
    parser.add_argument(
        "--until",
        type=float,
        default=0.0,
        help=(
            "end the window this many days before the venue's newest bar; the same universe "
            "read on an earlier period, which is the cheapest out-of-sample test there is"
        ),
    )
    parser.add_argument(
        "--period",
        default="",
        help=(
            "the bar; a frozen panel measures its own width when this is empty, while a venue "
            f"run uses {DEFAULT_PERIOD} — say it out loud when the file is not daily"
        ),
    )
    parser.add_argument(
        "--days",
        type=float,
        default=730.0,
        help="how far back to ask for; 1day over 2000 bars is 5.5 years, 4hour is 333 days",
    )
    parser.add_argument(
        "--lookback", type=int, nargs="+", default=list(LOOKBACKS), help="the ranking window(s)"
    )
    parser.add_argument(
        "--rebalance",
        type=int,
        nargs="+",
        default=list(REBALANCES),
        help="bars between rebalances (1 = daily, 7 = weekly)",
    )
    parser.add_argument(
        "--sign",
        nargs="+",
        default=list(SIGNS),
        choices=list(SIGNS),
        help="reversal buys the fallen, momentum buys the risen; both are printed by default",
    )
    parser.add_argument("--legs", type=int, default=10, help="names per side")
    parser.add_argument(
        "--gross", type=float, default=1.0, help="gross exposure as a share of equity"
    )
    parser.add_argument(
        "--max-weight", type=float, default=0.05, help="the most equity one name may take"
    )
    parser.add_argument(
        "--min-names",
        type=int,
        default=8,
        help="below this many ranked names the book stands aside",
    )
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--fee", type=float, default=FEE_RATE, help="fee per side per fill")
    parser.add_argument(
        "--slippage",
        type=float,
        default=FILL_SLIPPAGE,
        help="slippage per fill, against the book",
    )
    parser.add_argument(
        "--fill",
        default="taker",
        choices=list(FILLS),
        help=(
            "taker crosses the open and pays the fee plus slippage; maker posts a limit at the "
            "decision's close, is filled at that limit only if the bar trades through it, pays "
            "--maker-fee, and is not filled at all otherwise"
        ),
    )
    parser.add_argument(
        "--maker-fee",
        type=float,
        default=MAKER_FEE,
        help="the rate a passive fill pays; 0.0002 is a fee tier, 0.002 is a VIP-0 account",
    )
    parser.add_argument(
        "--latency", type=int, default=1, help="bars between the decision close and the fill"
    )
    parser.add_argument(
        "--split", type=float, default=0.6, help="the share of marks the book is described on"
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="where the table and the JSON land")
    parser.add_argument(
        "--freeze", default="", help="write the panel as <pair>.csv files here, for a replay"
    )
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="feed timeout")
    parser.add_argument("--endpoint", default=HTX_ENDPOINT, help="the kline endpoint")
    parser.add_argument("--quiet", action="store_true", help="write the files, print nothing")
    return parser



def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read a universe, run the declared grid over it, print the table — and mean the exit code.

    Exit 0 when the panel produced at least one readable window, *whether or not anything
    passed*, because "no cross-sectional edge at these costs" is an answer and a tool that
    exits non-zero for it would be a tool that only reports successes. 2 is a caller's mistake,
    3 a panel that could not be read, 4 a panel too short for the book to fill even once.
    """
    args = _parser().parse_args(argv)
    try:
        base = BasketSettings(
            lookback=int(args.lookback[0]),
            sign=str(args.sign[0]),
            legs=int(args.legs),
            rebalance_bars=int(args.rebalance[0]),
            capital=float(args.capital),
            fee=float(args.fee),
            slippage=float(args.slippage),
            latency_bars=int(args.latency),
            min_names=int(args.min_names),
            gross=float(args.gross),
            max_weight=float(args.max_weight),
            fill=str(args.fill),
            maker_fee=float(args.maker_fee),
        )
    except (ValueError, IndexError) as exc:
        print(f"crosssection: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if not 0.0 < float(args.split) < 1.0:
        print(f"crosssection: --split {args.split!r} is not a fraction", file=sys.stderr)
        return EXIT_USAGE

    failures: List[Tuple[str, str]] = []
    window_note = ""
    try:
        if str(args.candles).strip():
            panel = load_panel(
                args.candles, period=str(args.period), delimiter=str(args.delimiter)
            )
            if float(args.until) > 0.0:
                window_note = (
                    f"--until {float(args.until):g} was not applied: a frozen panel is already "
                    "the window it is, and cutting it here would make the replay of a run "
                    "disagree with the run"
                )
        else:
            symbols: List[str] = []
            if str(args.symbols).strip():
                symbols = [
                    name.strip().lower()
                    for name in str(args.symbols).split(",")
                    if name.strip()
                ]
            elif int(args.top) > 0:
                from agent.backtest.data import universe

                skip = max(0, int(args.skip))
                ranked = universe(
                    str(args.quote), skip + int(args.top), timeout=float(args.timeout)
                )
                symbols = [pair.symbol for pair in ranked[skip:]]
                if skip > 0:
                    window_note = (
                        f"ranks {skip + 1}-{skip + len(symbols)} of the {str(args.quote).upper()} "
                        "board by 24h turnover: a universe chosen before this run read anything"
                    )
            if not symbols:
                print(
                    "crosssection: name a universe with --symbols, --top N or --candles DIR — "
                    "a cross-section needs more than one pair",
                    file=sys.stderr,
                )
                return EXIT_USAGE
            panel, failures = fetch_panel(
                symbols,
                period=str(args.period).strip() or DEFAULT_PERIOD,
                days=float(args.days),
                until=max(float(args.until), 0.0),
                timeout=float(args.timeout),
                endpoint=str(args.endpoint),
            )
            if float(args.until) > 0.0:
                window_note = (
                    f"this window ends {float(args.until):g} day(s) before the venue's newest "
                    f"bar{'; ' + window_note if window_note else ''}"
                )
    except (PanelError, HtxError) as exc:
        print(f"crosssection: {exc}", file=sys.stderr)
        return EXIT_FEED

    if not args.quiet:
        for name, why in failures[:8]:
            print(f"   ! {name}: {why}")
        if len(failures) > 8:
            print(f"   ! and {len(failures) - 8} more pair(s) that could not be read")
        shape = panel.as_dict()
        cost = (
            f"maker {float(args.maker_fee) * 1e4:.1f} bps"
            if str(args.fill) == "maker"
            else f"fee {float(args.fee) * 1e4:.1f} + slip {float(args.slippage) * 1e4:.1f} bps"
        )
        print(
            f"── crosssection · {shape['symbols']} symbols · {shape['period']} · "
            f"{shape['bars']} bars · {shape['coverage']:.1%} filled · "
            f"{int(args.legs)} a side · gross {float(args.gross):.2f} · "
            f"${float(args.capital):,.2f} · {str(args.fill)} ({cost})"
        )
        if window_note:
            print(f"   window: {window_note}")
        if panel.interval and abs(float(panel.interval) - 86400.0) > 1.0:
            print(
                f"   {86400.0 / float(panel.interval):.0f} bar(s) a day: the marks are per bar "
                "and every bps in the table is per day, because a 5 bps floor per bar is a "
                f"{5.0 * 86400.0 / float(panel.interval):.0f} bps floor per day"
            )
        print(
            f"   gate: {GATE['edge_bps']:.0f} bps/day · t {GATE['t_stat']:.1f} · "
            f"sharpe {GATE['sharpe']:.1f} · dd <= {GATE['max_drawdown']:.0%} · "
            f"|beta| <= {GATE['beta']:.2f} · split {float(args.split):.0%}/"
            f"{1.0 - float(args.split):.0%}"
        )
        # The panel's own warnings are the ones a reader cannot get anywhere else: which
        # symbols arrived short, and what an ``--until`` cut does and does not prove.
        for line in list(panel.warnings)[:4]:
            print(f"   ! {line}")
        if len(panel.warnings) > 4:
            print(f"   ! and {len(panel.warnings) - 4} more panel warning(s), in the JSON")

    try:
        readings = grid(
            panel,
            lookbacks=[int(value) for value in args.lookback],
            rebalances=[int(value) for value in args.rebalance],
            signs=[str(value) for value in args.sign],
            base=base,
            split=float(args.split),
        )
    except PanelError as exc:
        print(f"crosssection: {exc}", file=sys.stderr)
        return EXIT_SHORT
    except ValueError as exc:
        print(f"crosssection: {exc}", file=sys.stderr)
        return EXIT_USAGE
    out = Path(args.out)
    written_csv = _write_csv(out / "crosssection.csv", readings, delimiter=str(args.delimiter))
    criteria = Criteria(
        gate=GATE,
        acceptance=ACCEPTANCE,
        does_not_show=DOES_NOT_SHOW,
        note=(
            "Declared before this run. The unit is a mark and not a trade: the floor is 5 bps a "
            "day net of the declared fee and slippage, which is 18 % a year on the book's own "
            "equity."
        ),
    )
    document = criteria.document(
        {
            "panel": panel.as_dict(),
            "window": {
                "until_days": max(float(args.until), 0.0),
                "skip": max(0, int(args.skip)),
                "top": int(args.top),
                "quote": str(args.quote),
                "days": float(args.days),
                "note": window_note,
            },
            "grid": {
                "lookbacks": [int(value) for value in args.lookback],
                "rebalances": [int(value) for value in args.rebalance],
                "signs": [str(value) for value in args.sign],
                "split": float(args.split),
                "base": base.as_dict(),
            },
            "readings": [reading.as_dict() for reading in readings],
            "verdict": verdict_of(readings),
            "failures": [{"symbol": name, "why": why} for name, why in failures],
        }
    )
    written_json = out / "crosssection.json"
    written_json.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Read the file back and check the hash it carries. This is the one file whose whole point
    # is that it cannot be edited after the fact, so a run that wrote a document the declaration
    # no longer matches is a run that should say so instead of printing a table over it.
    problem = verify(json.loads(written_json.read_text(encoding="utf-8")))
    if problem:
        print(f"crosssection: the report does not verify: {problem}", file=sys.stderr)
        return EXIT_FEED
    written_criteria = criteria.write(out / "criteria.json")
    if str(args.freeze).strip():
        write_panel(panel, args.freeze)
    if not args.quiet:
        print()
        print(format_table(readings))
        print(f"   {verdict_of(readings)}")
        seen: List[str] = []
        for reading in readings:
            for line in reading.warnings:
                if line not in seen:
                    seen.append(line)
        # The last warning of every reading is always the one about how to read the number on
        # it (the overlap of the ranking window), so those lines are never cut by the cap: a
        # table whose t-stat came with its own caveat trimmed would be worse than no table.
        tails = [reading.warnings[-1] for reading in readings if reading.warnings]
        rest = [line for line in seen if line not in tails]
        for line in rest[:6]:
            print(f"   ! {line}")
        if len(rest) > 6:
            print(f"   ! and {len(rest) - 6} more warning(s), in the JSON")
        for line in list(dict.fromkeys(tails)):
            print(f"   ! {line}")
        print(
            f"   wrote {written_csv} · {written_json} · {written_criteria} "
            f"· criteria {criteria.fingerprint()[:12]}"
        )
    if not any(reading.full.days > 0 for reading in readings):
        print("crosssection: not one config produced a readable window", file=sys.stderr)
        return EXIT_SHORT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

