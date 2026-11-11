"""``python -m agent.backtest.sweep`` — one run is an anecdote, twenty is a sample.

A single pair over a single window answers "what would this have paid", and that is a
sentence about one instrument in one regime. The question worth money is the next one up:
*does the thesis hold across the market?* So this file runs the whole engine once per pair
and prints one table, because the only way to read twenty backtests is side by side::

    # the twenty pairs the venue itself traded most, on daily bars for 5½ years
    python -m agent.backtest.sweep --top 20 --period 1day --days 2000 --capital 1000

    # the same, on a list an operator chose
    python -m agent.backtest.sweep --symbols btcusdt,ethusdt,solusdt --days 365

Which pairs are "the top twenty" is not decided here — it is read from the venue's own
ticker endpoint and ranked by 24-hour quote turnover (:func:`agent.backtest.data.universe`),
so the sample is the market's opinion rather than the operator's. It is still one biased
sample and the docstring of that function says how.

What the table is for
=====================

Three columns are the argument and the rest are the audit trail:

* ``roi`` — what the strategy paid;
* ``buy & hold`` — what doing nothing paid over the same bars;
* ``edge`` — the difference, which is the only number that says whether the hunter *did*
  anything. A pair where the strategy lost 5 % while the coin lost 40 % is a pair where the
  thesis worked; a pair where the strategy made 8 % while the coin made 60 % is a pair where
  it merely stood in the way less. Sorting by ``roi`` alone would confuse the two.

Rows that could not be measured keep their place in the table with the sentence that
explains why — a sweep that silently drops the pairs it failed on is a sweep that reports
its own successes as the market.

The cost of the sweep
=====================

One kline request and one full replay per pair, and the replay is the expensive half: every
bar is assessed against everything before it. Twenty pairs of two thousand daily bars is
twenty to forty minutes of single-core work, which is why :func:`sweep` reports each row as
it lands instead of holding the table until the end, and why ``--quiet`` exists for a run
that is being appended to a log.

No key, no RPC, no signature, no ``trader_state.json`` — the same wall as the rest of the
package, and the same one transport (:func:`agent.backtest.data.fetch_klines`) behind it.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent.backtest.data import (
    DEFAULT_PERIOD,
    DEFAULT_TIMEOUT,
    HTX_ENDPOINT,
    HTX_ENDPOINTS,
    HTX_TICKERS_ENDPOINTS,
    PERIOD_SECONDS,
    CandleSeries,
    HtxError,
    fetch_klines,
    universe,
)
from agent.backtest.engine import Backtester, Settings
from agent.backtest.reporter import Report, stamp
from agent.backtest.run import EXIT_FEED, EXIT_SHORT, EXIT_USAGE

__all__ = [
    "DEFAULT_OUT",
    "HEADERS",
    "SweepRow",
    "format_table",
    "main",
    "sweep",
    "totals",
    "totals_text",
]

#: Where the sweep writes unless ``--out`` says otherwise: one directory per pair under it,
#: plus ``sweep.csv``. Under ``data/`` because that is this repository's scratch space, and
#: never anywhere near the state file the live loop owns.
DEFAULT_OUT = "data/backtest/sweep"

#: Column headers, in order. ``edge`` is ``roi`` minus ``buy & hold`` — see the docstring.
HEADERS = (
    "symbol",
    "bars",
    "window",
    "trades",
    "roi",
    "buy & hold",
    "edge",
    "drawdown",
    "win",
    "fees",
)


@dataclass
class SweepRow:
    """One pair's run, reduced to what a table can hold — or the reason there is no row.

    A row is never dropped for being bad: an unmeasured pair is a row with ``error`` set and
    every reading at zero, so that the count of pairs in the table is the count of pairs that
    were asked about. That is the difference between a sample and a highlight reel.

    ``trades`` counts the positions that were *closed*, so it is the number of realized
    round-trips; ``open_trades`` is what was still held at the last bar, and its mark-to-market
    is already inside ``roi``. A row with ``0`` closed and ``1`` open therefore shows ``0+1``:
    a book that bought the last panic and never got its exit.
    """

    symbol: str
    bars: int = 0
    first_stamp: str = ""
    last_stamp: str = ""
    trades: int = 0
    open_trades: int = 0
    roi: float = 0.0
    buy_hold_roi: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    fees: float = 0.0
    final_equity: float = 0.0
    exits: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    out: str = ""
    error: str = ""

    @property
    def measured(self) -> bool:
        """Whether the row is a reading or an apology."""
        return not self.error

    @property
    def edge(self) -> float:
        """What the strategy did that holding the coin did not — in the same units."""
        return float(self.roi) - float(self.buy_hold_roi)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bars": int(self.bars),
            "first": self.first_stamp,
            "last": self.last_stamp,
            "trades": int(self.trades),
            "open_trades": int(self.open_trades),
            "roi": float(self.roi),
            "buy_hold_roi": float(self.buy_hold_roi),
            "edge": float(self.edge),
            "max_drawdown": float(self.max_drawdown),
            "win_rate": float(self.win_rate),
            "fees": float(self.fees),
            "final_equity": float(self.final_equity),
            "exits": dict(self.exits),
            "warnings": list(self.warnings),
            "out": self.out,
            "error": self.error,
        }


def sweep(
    symbols: Sequence[str],
    settings: Settings,
    *,
    period: str = DEFAULT_PERIOD,
    days: float = 180.0,
    out_root: Any = DEFAULT_OUT,
    loader: Optional[Callable[..., CandleSeries]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    endpoint: str = HTX_ENDPOINT,
    on_row: Optional[Callable[[SweepRow], None]] = None,
) -> List[SweepRow]:
    """Run one backtest per symbol and return their rows in the order asked for.

    The failures are the interesting part of the contract. A dead feed for one pair does not
    end the sweep — it becomes a row with ``error`` set, and the sweep continues, because a
    twenty-pair run that dies on the ninth pair answers nothing while a twenty-row table with
    one apology answers nineteen questions. The only way to get an exception out of here is
    to pass a settings object that cannot mean anything, and that is a programmer's mistake
    rather than a market's.

    ``loader`` is the candles: :func:`agent.backtest.data.fetch_klines` unless a caller says
    otherwise, which is how the tests hand over a dictionary per symbol and never open a
    socket. ``on_row`` is called as each row lands, so a long sweep can be watched.
    """
    fetch = loader or fetch_klines
    root = Path(out_root)
    rows: List[SweepRow] = []
    for name in symbols:
        symbol = str(name).strip().lower()
        if not symbol:
            continue
        row = SweepRow(symbol=symbol)
        try:
            series = fetch(
                symbol,
                period=str(period),
                days=float(days),
                timeout=float(timeout),
                endpoint=str(endpoint),
            )
            if len(series) <= int(settings.min_points):
                raise HtxError(
                    f"{len(series)} bars of {series.period} is not enough for a verdict — "
                    f"the detector needs more than {int(settings.min_points)}; this pair "
                    "was listed too recently for the window, or the window is too short"
                )
            report = Report(Backtester(series, settings).run())
            written = report.write(root / symbol)
            row.bars = len(series)
            row.first_stamp = stamp(series.span()[0])[:10]
            row.last_stamp = stamp(series.span()[1])[:10]
            row.trades = report.metrics.closed
            row.open_trades = report.metrics.open_trades
            row.roi = report.metrics.roi
            row.buy_hold_roi = report.metrics.buy_hold_roi
            row.max_drawdown = report.metrics.max_drawdown
            row.win_rate = report.metrics.win_rate
            row.fees = report.metrics.fees_paid
            row.final_equity = report.metrics.final_equity
            row.exits = dict(report.metrics.exits)
            row.warnings = list(report.result.warnings)
            row.out = str(written["json"].parent)
        except (HtxError, ValueError) as exc:
            # A judgement, not a crash: the sentence is the row.
            row.error = str(exc)
        rows.append(row)
        if on_row is not None:
            on_row(row)
    return rows


def _cells(row: SweepRow) -> List[str]:
    """One row's ten columns, already formatted — every number carries its own unit.

    A row that could not be measured fills every reading with a dash and keeps its sentence
    for the foot of the table: a failed pair belongs in the sample, but its explanation must
    not be allowed to widen a column to the length of a paragraph.
    """
    if row.error:
        return [row.symbol, "—", "—", "—", "—", "—", "—", "—", "—", "—"]
    return [
        row.symbol,
        f"{row.bars}",
        f"{row.first_stamp}→{row.last_stamp}",
        f"{row.trades}+{row.open_trades}" if row.open_trades else f"{row.trades}",
        f"{row.roi:+.2%}",
        f"{row.buy_hold_roi:+.2%}",
        f"{row.edge:+.2%}",
        f"{row.max_drawdown:.2%}",
        f"{row.win_rate:.0%}",
        f"${row.fees:,.2f}",
    ]


def format_table(rows: Sequence[SweepRow]) -> str:
    """The sweep as text: every pair on one line, columns aligned to their widest cell.

    Left-aligned are the two things a reader scans (the pair and its window); right-aligned
    is every number, because a column of ratios is read by its decimal points and not by its
    first digit — a table that pads numbers on the left is a table that hides a minus sign.
    """
    if not rows:
        return "no pairs were measured"
    body = [_cells(row) for row in rows]
    widths = [
        max(len(HEADERS[index]), *(len(cells[index]) for cells in body))
        for index in range(len(HEADERS))
    ]
    left = {0, 2}
    lines = [
        "  ".join(
            HEADERS[index].ljust(widths[index])
            if index in left
            else HEADERS[index].rjust(widths[index])
            for index in range(len(HEADERS))
        ),
        "  ".join("─" * width for width in widths),
    ]
    for cells in body:
        lines.append(
            "  ".join(
                cells[index].ljust(widths[index])
                if index in left
                else cells[index].rjust(widths[index])
                for index in range(len(HEADERS))
            ).rstrip()
        )
    for row in rows:
        if row.error:
            lines.append(f"   ! {row.symbol} — {row.error}")
    return "\n".join(lines)


def _median(values: Sequence[float]) -> float:
    """The middle of a non-empty sorted list, or the mean of the two middles."""
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return (float(values[middle - 1]) + float(values[middle])) / 2.0


def totals(rows: Sequence[SweepRow]) -> Dict[str, Any]:
    """What the twenty rows say together: the mean, the median, and who beat holding.

    Mean *and* median because a sweep of ratios is exactly the distribution where one pair
    can move the average by tens of per cent, and a single number that can be moved by one
    instrument is not a finding. ``beat_buy_hold`` counts the rows whose ``edge`` is positive:
    that count, more than any average, is the answer to "does the thesis hold".
    """
    measured = [row for row in rows if row.measured]
    if not measured:
        return {"measured": 0, "failed": len(rows)}
    rois = sorted(row.roi for row in measured)
    edges = sorted(row.edge for row in measured)
    return {
        "measured": len(measured),
        "failed": len(rows) - len(measured),
        "mean_roi": sum(rois) / len(rois),
        "median_roi": _median(rois),
        "mean_buy_hold": sum(row.buy_hold_roi for row in measured) / len(measured),
        "mean_edge": sum(edges) / len(edges),
        "median_edge": _median(edges),
        "beat_buy_hold": sum(1 for row in measured if row.edge > 0.0),
        "trades": sum(row.trades for row in measured),
        "fees": sum(row.fees for row in measured),
    }


def totals_text(rows: Sequence[SweepRow]) -> str:
    """The one line under the table that a reader is allowed to quote out of context."""
    reading = totals(rows)
    if not reading.get("measured"):
        return f"no pair could be measured ({reading.get('failed', 0)} asked)"
    return (
        f"{reading['measured']} pairs measured"
        + (f", {reading['failed']} failed" if reading["failed"] else "")
        + f" · mean ROI {reading['mean_roi']:+.2%} · median {reading['median_roi']:+.2%} · "
        f"mean buy & hold {reading['mean_buy_hold']:+.2%} · "
        f"mean edge {reading['mean_edge']:+.2%} · median edge "
        f"{reading['median_edge']:+.2%} · "
        f"beat buy & hold {reading['beat_buy_hold']}/{reading['measured']} · "
        f"{reading['trades']} trades · ${reading['fees']:,.2f} in fees"
    )


def _write_csv(path: Path, rows: Sequence[SweepRow], *, delimiter: str = ",") -> Path:
    """The same table as a file, with the raw fractions and the apology column kept.

    Fractions rather than percentages in the file: a spreadsheet can format ``-0.0212`` and
    cannot un-format ``"-2.12%"``, and a report that requires a text parser to become a chart
    is a report nobody charts.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerow(
            [
                "symbol",
                "bars",
                "first",
                "last",
                "trades",
                "open_trades",
                "roi",
                "buy_hold_roi",
                "edge",
                "max_drawdown",
                "win_rate",
                "fees",
                "final_equity",
                "exits",
                "out",
                "error",
                "warnings",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.symbol,
                    row.bars,
                    row.first_stamp,
                    row.last_stamp,
                    row.trades,
                    row.open_trades,
                    round(row.roi, 8),
                    round(row.buy_hold_roi, 8),
                    round(row.edge, 8),
                    round(row.max_drawdown, 8),
                    round(row.win_rate, 8),
                    round(row.fees, 8),
                    round(row.final_equity, 8),
                    ";".join(f"{kind}:{count}" for kind, count in row.exits.items() if count),
                    row.out,
                    row.error,
                    " | ".join(row.warnings),
                ]
            )
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.backtest.sweep",
        description=(
            "Run the same backtest across many pairs and print one table: what the strategy "
            "paid, what holding paid, and the difference. Signs nothing."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="how many pairs, ranked by the venue's own 24h turnover (default 20)",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="a comma-separated list instead of --top, e.g. btcusdt,ethusdt,solusdt",
    )
    parser.add_argument("--quote", default="usdt", help="the quote currency of the universe")
    parser.add_argument(
        "--exchange",
        default="htx",
        choices=("htx", "huobi"),
        help="which of the venue's two hosts to read",
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        choices=sorted(PERIOD_SECONDS),
        help=f"the bar (default {DEFAULT_PERIOD}); 1day over --days 2000 is 5½ years",
    )
    parser.add_argument("--days", type=float, default=180.0, help="the window, in days")
    parser.add_argument("--capital", type=float, default=1000.0, help="the book, per pair")
    parser.add_argument("--share", type=float, default=1.0, help="the fraction staked per trade")
    parser.add_argument("--stop", type=float, default=0.02, help="the stop, as a fraction")
    parser.add_argument("--fee", type=float, default=0.001, help="the fee per side")
    parser.add_argument("--slippage", type=float, default=0.001, help="the slippage per side")
    parser.add_argument(
        "--latency-bars", type=int, default=1, help="bars between a decision and its fill"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.0, help="the hunter confidence floor to enter"
    )
    parser.add_argument(
        "--entries",
        default="panic",
        choices=("panic", "organic", "both"),
        help="which verdicts may open a position",
    )
    parser.add_argument("--window", type=int, default=1, help="the detector's smoothing window")
    parser.add_argument("--min-points", type=int, default=64, help="bars before a verdict")
    parser.add_argument(
        "--min-depth", type=float, default=0.0, help="turnover floor for the THIN arm; 0 is off"
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=0,
        help="bars of silence between an exit and the next entry; 30 is the hunter's own",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, help="the directory the per-pair reports land under"
    )
    parser.add_argument("--delimiter", default=",", help="the CSV separator")
    parser.add_argument("--quiet", action="store_true", help="write the files, print nothing")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="feed timeout")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Read a universe, run it pair by pair, print the table — and mean the exit code.

    Exit 0 when the table holds at least one reading (a sweep is allowed to find losses),
    2 for a caller's mistake, 3 when the universe itself could not be read or when no pair
    could be measured for any reason other than a short window, and 4 when every pair was
    too young for the detector to speak.
    """
    args = _parser().parse_args(argv)
    try:
        settings = Settings(
            capital=float(args.capital),
            share=float(args.share),
            stop=float(args.stop),
            fee=float(args.fee),
            slippage=float(args.slippage),
            latency_bars=int(args.latency_bars),
            confidence=float(args.confidence),
            entries=str(args.entries),
            window=int(args.window),
            min_points=int(args.min_points),
            min_depth=float(args.min_depth),
            cooldown_bars=int(args.cooldown_bars),
        )
    except ValueError as exc:
        print(f"sweep: {exc}", file=sys.stderr)
        return EXIT_USAGE

    symbols: List[str] = []
    if str(args.symbols).strip():
        symbols = [name.strip().lower() for name in str(args.symbols).split(",") if name.strip()]
        if not symbols:
            print("sweep: --symbols named no pair; write them as btcusdt,ethusdt", file=sys.stderr)
            return EXIT_USAGE
    else:
        try:
            listed = universe(
                str(args.quote),
                int(args.top),
                timeout=float(args.timeout),
                endpoint=HTX_TICKERS_ENDPOINTS[1 if str(args.exchange) == "huobi" else 0],
            )
        except HtxError as exc:
            print(f"sweep: {exc}", file=sys.stderr)
            return EXIT_FEED
        symbols = [pair.symbol for pair in listed]

    if not args.quiet:
        print(
            f"── sweep · {len(symbols)} pairs · {args.period} · {args.days:g} days asked · "
            f"${float(args.capital):,.2f} each · entries {args.entries}"
        )

    def announce(row: SweepRow) -> None:
        """One line as each pair lands: a long sweep must show that it is still moving."""
        if args.quiet:
            return
        if row.error:
            print(f"   ! {row.symbol:<12} {row.error}")
        else:
            print(
                f"   · {row.symbol:<12} ROI {row.roi:+.2%} · buy & hold "
                f"{row.buy_hold_roi:+.2%} · {row.trades} trades"
            )

    rows = sweep(
        symbols,
        settings,
        period=str(args.period),
        days=float(args.days),
        out_root=Path(args.out),
        timeout=float(args.timeout),
        endpoint=HTX_ENDPOINTS[1 if str(args.exchange) == "huobi" else 0],
        on_row=announce,
    )
    written = _write_csv(
        Path(args.out) / "sweep.csv", rows, delimiter=str(args.delimiter)
    )
    if not args.quiet:
        print()
        print(format_table(rows))
        print(f"   {totals_text(rows)}")
        print(f"   wrote {written}")
    if not totals(rows).get("measured"):
        print(
            "sweep: not one pair could be measured — the table above says why for each",
            file=sys.stderr,
        )
        short = all("not enough for a verdict" in row.error for row in rows)
        return EXIT_SHORT if short else EXIT_FEED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
