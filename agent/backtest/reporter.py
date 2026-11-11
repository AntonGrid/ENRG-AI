"""What the run added up to: five numbers an operator asks for, and the one that shames them.

The brief asks for ROI, maximum drawdown, win rate, the number of trades and the mean
holding time. All five are here, computed from the equity curve and the fill ledger and
from nothing else — and so is a sixth, because a backtest without it is a sales pitch:
**buy-and-hold**. If a strategy that trades eleven times in six months returns 4 % while the
instrument returned 40 %, the honest summary is not "ROI +4 %", it is "it lost to holding
by 36 points, and here is what the exits cost".

The distinctions this file refuses to blur
==========================================

**Closed trades are not all trades.** A position still open on the last bar is marked at the
last close in the equity curve (so the ROI is honest) and is *excluded* from the win rate,
because a winner that has not been sold is an opinion. ``open_trades`` says how many.

**Drawdown is measured on the book, not on the price.** It is the largest fall of
``cash + units × close`` from its own running peak, in per cent of that peak, and its date is
reported — a 6 % drawdown in March and a 6 % drawdown in the last three days are different
facts about a strategy.

**Holding time is in bars and in seconds.** The bars are the unit the decision was made in;
the seconds come from the series' own measured interval, so a report cannot claim days for a
4-hour chart by assuming 86400 anywhere.

Both files are written for a reader, not for a program: the JSON is complete (settings,
warnings, verdict counts, every trade and the whole curve), and the CSV is the trades table
with an ISO timestamp per side, which is the shape Excel opens without an argument.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from agent.backtest.engine import BacktestResult, Trade
from agent.physics.volatility_hunter import VERDICTS

__all__ = [
    "DEFAULT_EQUITY_CSV",
    "DEFAULT_JSON",
    "DEFAULT_TRADES_CSV",
    "Metrics",
    "Report",
    "metrics_of",
    "stamp",
    "write_report",
]

#: The three files a run writes, beside each other, in ``--out``.
DEFAULT_JSON = "backtest_report.json"
DEFAULT_TRADES_CSV = "backtest_trades.csv"
DEFAULT_EQUITY_CSV = "backtest_equity.csv"

#: The exits a trade can have, in the order the report lists them.
EXIT_KINDS = ("take-profit", "convicted", "stop")


@dataclass(frozen=True)
class Metrics:
    """The readings: what the run paid, how far it fell, and how often it was right.

    Every field is a number an operator can check against the CSV in the same directory,
    which is the only test of a metric worth having: ``trades`` must equal the rows of the
    trades file, ``fees_paid`` their fee column, and ``final_equity`` the last row of the
    equity file.
    """

    capital: float = 0.0
    final_equity: float = 0.0
    pnl: float = 0.0
    roi: float = 0.0
    buy_hold_roi: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_at: float = 0.0
    trades: int = 0
    closed: int = 0
    open_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_hold_bars: float = 0.0
    avg_hold_seconds: float = 0.0
    exposure: float = 0.0
    fees_paid: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    exits: Dict[str, int] = field(default_factory=dict)
    verdicts: Dict[str, int] = field(default_factory=dict)
    bars: int = 0
    judged_bars: int = 0
    interval_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capital": round(float(self.capital), 6),
            "final_equity": round(float(self.final_equity), 6),
            "pnl": round(float(self.pnl), 6),
            "roi": round(float(self.roi), 6),
            "buy_hold_roi": round(float(self.buy_hold_roi), 6),
            "max_drawdown": round(float(self.max_drawdown), 6),
            "max_drawdown_at": float(self.max_drawdown_at),
            "trades": int(self.trades),
            "closed": int(self.closed),
            "open_trades": int(self.open_trades),
            "wins": int(self.wins),
            "losses": int(self.losses),
            "breakeven": int(self.breakeven),
            "win_rate": round(float(self.win_rate), 6),
            "profit_factor": (
                None
                if math.isinf(float(self.profit_factor))
                else round(float(self.profit_factor), 6)
            ),
            "avg_hold_bars": round(float(self.avg_hold_bars), 4),
            "avg_hold_seconds": round(float(self.avg_hold_seconds), 2),
            "exposure": round(float(self.exposure), 6),
            "fees_paid": round(float(self.fees_paid), 6),
            "best_trade": round(float(self.best_trade), 6),
            "worst_trade": round(float(self.worst_trade), 6),
            "exits": dict(self.exits),
            "verdicts": dict(self.verdicts),
            "bars": int(self.bars),
            "judged_bars": int(self.judged_bars),
            "interval_seconds": int(self.interval_seconds) if self.interval_seconds else 0,
        }


def drawdown(curve: List[Dict[str, Any]]) -> tuple:
    """``(deepest, at)`` — the largest fall of the book from its own running peak.

    Measured on the equity curve and not on the price, because a strategy that sits in cash
    through a crash has not drawn down at all: it missed the fall *and* the rebound, and
    that is a fact about the strategy the strategy's own curve is the only honest witness of.
    """
    peak = 0.0
    deepest = 0.0
    at = 0.0
    for row in curve:
        equity = float(row.get("equity", 0.0))
        if equity > peak:
            peak = equity
        if peak > 0.0:
            depth = (peak - equity) / peak
            if depth > deepest:
                deepest = depth
                at = float(row.get("at", 0.0))
    return deepest, at


def verdict_counts(result: BacktestResult) -> Dict[str, int]:
    """Every core verdict, in the core's own order, plus anything a future one adds."""
    counts = {verdict: int(result.verdicts.get(verdict, 0)) for verdict in VERDICTS}
    for verdict, count in sorted(result.verdicts.items()):
        if verdict not in counts and verdict:
            counts[verdict] = int(count)
    return counts


def metrics_of(result: BacktestResult) -> Metrics:
    """Read the facts once and answer everything the report prints.

    Nothing here re-derives a fill or a price: the equity rows and the trades are the
    ledger the engine produced, and a metric that disagreed with them would be a metric
    computed from something else.
    """
    settings = result.settings
    curve = result.equity
    closed = result.closed
    capital = float(settings.capital)
    final = float(result.final_equity)
    deepest, deepest_at = drawdown(curve)
    wins = [trade for trade in closed if trade.pnl > 0.0]
    losses = [trade for trade in closed if trade.pnl < 0.0]
    breakeven = len(closed) - len(wins) - len(losses)
    gained = sum(trade.pnl for trade in closed if trade.pnl > 0.0)
    lost = -sum(trade.pnl for trade in closed if trade.pnl < 0.0)
    if lost > 0.0:
        profit_factor = gained / lost
    else:
        profit_factor = math.inf if gained > 0.0 else 0.0
    holds = [trade.bars for trade in closed]
    interval = float(result.series.interval or 0.0)
    first_price = float(curve[0]["price"]) if curve else 0.0
    last_price = float(curve[-1]["price"]) if curve else 0.0
    buy_hold = (last_price / first_price - 1.0) if first_price > 0.0 else 0.0
    counts = verdict_counts(result)
    decided = sum(count for verdict, count in counts.items() if verdict != "UNMEASURED")
    exits = {kind: 0 for kind in EXIT_KINDS}
    for trade in closed:
        exits[trade.exit_reason or "verdict"] = exits.get(trade.exit_reason or "verdict", 0) + 1
    return Metrics(
        capital=capital,
        final_equity=final,
        pnl=final - capital,
        roi=(final / capital - 1.0) if capital > 0.0 else 0.0,
        buy_hold_roi=float(buy_hold),
        max_drawdown=float(deepest),
        max_drawdown_at=float(deepest_at),
        trades=len(result.trades),
        closed=len(closed),
        open_trades=len(result.open_trades),
        wins=len(wins),
        losses=len(losses),
        breakeven=int(breakeven),
        win_rate=(len(wins) / len(closed)) if closed else 0.0,
        profit_factor=float(profit_factor),
        avg_hold_bars=(sum(holds) / len(holds)) if holds else 0.0,
        avg_hold_seconds=(sum(holds) / len(holds) * interval) if holds and interval else 0.0,
        exposure=(
            sum(1 for row in curve if row.get("exposure")) / len(curve) if curve else 0.0
        ),
        fees_paid=sum(trade.fees for trade in result.trades),
        best_trade=max((trade.pnl for trade in closed), default=0.0),
        worst_trade=min((trade.pnl for trade in closed), default=0.0),
        exits=exits,
        verdicts=counts,
        bars=len(result.series.rows),
        judged_bars=int(decided),
        interval_seconds=interval,
    )


def stamp(at: float) -> str:
    """An ISO-8601 UTC stamp, or an empty string — a JSON report has no ``NaN`` dates."""
    if not at or not math.isfinite(float(at)):
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(at)))


#: The sentence every report carries, because the one thing a reader of a backtest must know
#: is what the numbers were measured on. It is the module docstring of ``data`` compressed to
#: two lines, and it is duplicated rather than imported so that the JSON is readable alone.
READING = (
    "yield = max(0, -bar return) in per cent per bar; depth = quote turnover. "
    "PANIC therefore reads as 'the instrument pays far more than its own rest while "
    "turnover drained', and CALM/GROWTH as 'the rate is back at its rest'. "
    "No lookahead: every verdict is measured on the prefix that ends on its own bar, and "
    "every fill happens at the next bar's open with slippage and fees."
)


@dataclass
class Report:
    """A finished run, with its readings and its three files.

    The metrics are computed once, in :meth:`__post_init__`, so that ``report.metrics`` and
    the file that was written cannot disagree — a report whose summary is recomputed at
    print time is a report whose summary can drift from what was saved.
    """

    result: BacktestResult
    metrics: Metrics = field(init=False)
    generated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.metrics = metrics_of(self.result)

    # -- what it says ------------------------------------------------------- #

    def as_dict(self) -> Dict[str, Any]:
        """The whole run as one JSON document: settings, facts, readings, warnings."""
        first, last = self.result.series.span()
        return {
            "tool": "enrg-ai backtest",
            "generated_at": float(self.generated_at),
            "reading": READING,
            "window": {
                "first": float(first),
                "last": float(last),
                "first_stamp": stamp(first),
                "last_stamp": stamp(last),
            },
            "settings": self.result.settings.as_dict(),
            "series": self.result.series.as_dict(),
            "venue": dict(self.result.venue),
            "metrics": self.metrics.as_dict(),
            "trades": [trade.as_dict() for trade in self.result.trades],
            "equity": list(self.result.equity),
            "verdicts": dict(self.metrics.verdicts),
            "decisions": list(self.result.decisions),
            "refusals": list(self.result.refusals),
            "warnings": list(self.result.warnings),
        }

    def summary_text(self) -> str:
        """The operator's block: what was run, what it paid, and what it cost to find out."""
        metrics = self.metrics
        series = self.result.series
        first, last = series.span()
        lines = [
            f"── backtest · {series.symbol or 'instrument'} {series.period} · "
            f"{metrics.bars} bars · {stamp(first)[:10]} → {stamp(last)[:10]}",
            f"   capital ${metrics.capital:,.2f} → ${metrics.final_equity:,.2f} · "
            f"ROI {metrics.roi:+.2%} · buy & hold {metrics.buy_hold_roi:+.2%}",
            f"   max drawdown {metrics.max_drawdown:.2%} · "
            f"win rate {metrics.win_rate:.1%} ({metrics.wins}/{metrics.closed}) · "
            f"profit factor "
            + (
                "n/a"
                if math.isinf(metrics.profit_factor)
                else f"{metrics.profit_factor:.2f}"
            ),
            f"   trades {metrics.closed} closed + {metrics.open_trades} open · "
            f"mean hold {metrics.avg_hold_bars:.1f} bars "
            f"({metrics.avg_hold_seconds / 86400.0:.1f} d) · "
            f"fees ${metrics.fees_paid:.2f} · exposure {metrics.exposure:.0%}",
        ]
        exits = ", ".join(
            f"{kind} {count}" for kind, count in metrics.exits.items() if count
        )
        if exits:
            lines.append(f"   exits {exits}")
        verdicts = ", ".join(
            f"{verdict} {count}" for verdict, count in metrics.verdicts.items() if count
        )
        lines.append(f"   verdicts {verdicts or 'none'} · judged {metrics.judged_bars} bars")
        for warning in self.result.warnings:
            lines.append(f"   warn  {warning}")
        return "\n".join(lines)

    # -- what it writes ----------------------------------------------------- #

    def write_json(self, path: Any) -> Path:
        """The whole report as JSON, written atomically enough for a reader to trust it.

        The document is built first and written second, so a run that dies mid-write leaves
        a truncated file rather than a file that looks complete and is not. ``NaN`` is
        already gone by here (``_round`` and the reporters keep it out), which matters
        because ``json`` writes ``NaN`` happily and Excel's importer does not.
        """
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            json.dumps(self.as_dict(), indent=2, allow_nan=False, sort_keys=False),
            encoding="utf-8",
        )
        return file

    def write_trades_csv(self, path: Any, *, delimiter: str = ",") -> Path:
        """The trades table: one row per round trip, both sides, and what it paid."""
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "entry_time",
            "entry_bar",
            "entry_price",
            "entry_usd",
            "units",
            "entry_verdict",
            "entry_confidence",
            "exit_time",
            "exit_bar",
            "exit_price",
            "exit_usd",
            "exit_reason",
            "exit_verdict",
            "bars",
            "pnl",
            "pnl_pct",
            "fees",
            "open",
        ]
        with file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
            writer.writerow(header)
            for trade in self.result.trades:
                writer.writerow(_trade_row(trade))
        return file

    def write_equity_csv(self, path: Any, *, delimiter: str = ",") -> Path:
        """The book bar by bar — the curve a drawdown is measured on."""
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        header = ["bar", "time", "price", "cash", "units", "equity", "exposure", "verdict"]
        with file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
            writer.writerow(header)
            for row in self.result.equity:
                writer.writerow(
                    [
                        int(row["bar"]),
                        stamp(float(row["at"])),
                        row["price"],
                        row["cash"],
                        row["units"],
                        row["equity"],
                        int(bool(row.get("exposure"))),
                        row.get("verdict", ""),
                    ]
                )
        return file

    def write(
        self,
        directory: Any,
        *,
        delimiter: str = ",",
        json_name: str = DEFAULT_JSON,
        trades_name: str = DEFAULT_TRADES_CSV,
        equity_name: str = DEFAULT_EQUITY_CSV,
    ) -> Dict[str, Path]:
        """Write all three files into ``directory`` and answer where they went."""
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        return {
            "json": self.write_json(folder / json_name),
            "trades": self.write_trades_csv(folder / trades_name, delimiter=delimiter),
            "equity": self.write_equity_csv(folder / equity_name, delimiter=delimiter),
        }


def _trade_row(trade: Trade) -> List[Any]:
    """One trade as a CSV row, in the order :meth:`Report.write_trades_csv` declares."""
    return [
        stamp(trade.entry_at),
        int(trade.entry_bar),
        round(float(trade.entry_price), 8),
        round(float(trade.entry_usd), 4),
        round(float(trade.units), 10),
        trade.entry_verdict,
        round(float(trade.entry_confidence), 4),
        stamp(trade.exit_at),
        int(trade.exit_bar),
        round(float(trade.exit_price), 8),
        round(float(trade.exit_usd), 4),
        trade.exit_reason,
        trade.exit_verdict,
        int(trade.bars),
        round(float(trade.pnl), 6),
        round(float(trade.pnl_pct), 8),
        round(float(trade.fees), 6),
        int(bool(trade.open)),
    ]


def write_report(
    result: BacktestResult,
    directory: Any,
    *,
    delimiter: str = ",",
    json_name: str = DEFAULT_JSON,
    trades_name: str = DEFAULT_TRADES_CSV,
    equity_name: str = DEFAULT_EQUITY_CSV,
) -> Report:
    """Build the report and write its three files — the call the CLI makes."""
    report = Report(result)
    report.write(
        directory,
        delimiter=delimiter,
        json_name=json_name,
        trades_name=trades_name,
        equity_name=equity_name,
    )
    return report
