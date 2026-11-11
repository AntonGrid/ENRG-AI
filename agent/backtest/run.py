"""``python -m agent.backtest.run`` — the machine time, on a command line.

One call, one report, and nothing to configure that has a default worth trusting blindly::

    python -m agent.backtest.run --exchange htx --symbol btcusdt --days 180 --capital 1000

That reads a public price history, walks it bar by bar through the same hunter the live loop
uses, and writes ``backtest_report.json`` plus two CSVs into ``--out``. It prints the summary
and exits 0 whatever the strategy earned — a backtest's job is to answer, not to succeed —
and it exits non-zero only when the *question* could not be asked: a dead feed (3), a
configuration that cannot mean anything (2), a window too short for the detector to speak
(4).

What it will never do
=====================

No key, no RPC, no signature, no ``trader_state.json``. The only network call in the whole
package is the one this file makes through :func:`agent.backtest.data.fetch_klines`, and
``--csv`` removes even that: a chart on disk is a backtest that can run on a plane, and the
file format is the one an operator can edit in a spreadsheet.

The window, and the one trap in it
==================================

``--days`` is converted to bars at ``--period`` and cut to what the venue will serve in one
answer (2000 bars). The cut is never silent: it lands in ``warnings``, in the report and in
the printed summary, because a backtest that quietly measured 333 days when it was asked for
730 is a backtest that answered a different question.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from agent.backtest.data import (
    DEFAULT_PERIOD,
    DEFAULT_TIMEOUT,
    HTX_ENDPOINTS,
    PERIOD_SECONDS,
    CandleSeries,
    HtxError,
    fetch_klines,
    load_csv,
)
from agent.backtest.engine import ENTRY_MODES, Backtester, Settings
from agent.backtest.reporter import Report
from agent.backtest.simulated_provider import FEE_RATE, FILL_SLIPPAGE, LATENCY_BARS
from agent.physics.portfolio import MIN_RECORD

__all__ = ["main"]

#: Where the three files land unless ``--out`` says otherwise. Under ``data/`` because that
#: is the directory the rest of this repository already treats as scratch — and *never* the
#: state file the live loop owns.
DEFAULT_OUT = "data/backtest"

#: Exit codes, the same convention the rest of the layer uses: 2 is a caller's mistake,
#: 3 is a feed that did not answer, 4 is a window that cannot support a verdict.
EXIT_USAGE = 2
EXIT_FEED = 3
EXIT_SHORT = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.backtest.run",
        description=(
            "Replay a public price history through the hunter and report what the strategy "
            "would have paid. Reads candles, writes a report, signs nothing."
        ),
    )
    parser.add_argument(
        "--exchange",
        default="htx",
        choices=("htx", "huobi", "csv"),
        help="where the candles come from: the HTX kline endpoint, its mirror, or a CSV",
    )
    parser.add_argument("--symbol", default="btcusdt", help="the pair, e.g. btcusdt")
    parser.add_argument(
        "--period",
        default=None,
        choices=sorted(PERIOD_SECONDS),
        help=f"the bar (default {DEFAULT_PERIOD}; a CSV infers it when omitted)",
    )
    parser.add_argument("--days", type=float, default=180.0, help="the window, in days")
    parser.add_argument("--csv", default="", help="a chart on disk: timestamp,o,h,l,c,volume")
    parser.add_argument("--capital", type=float, default=1000.0, help="the starting book, USDT")
    parser.add_argument(
        "--share", type=float, default=1.0, help="the share of cash one entry may spend"
    )
    parser.add_argument("--stop", type=float, default=0.02, help="the stop below the fill")
    parser.add_argument("--fee", type=float, default=FEE_RATE, help="the fee per side")
    parser.add_argument("--slippage", type=float, default=FILL_SLIPPAGE, help="per fill")
    parser.add_argument(
        "--latency-bars",
        type=int,
        default=LATENCY_BARS,
        help="bars between a decision and its fill (the network delay, in bars)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.0,
        help="the floor a PANIC must reach to be bought; 0 buys every one",
    )
    parser.add_argument(
        "--entries",
        default="panic",
        choices=ENTRY_MODES,
        help="panic: the hunt only; organic: carry on an honest record; both",
    )
    parser.add_argument("--window", type=int, default=1, help="bars the decline is measured over")
    parser.add_argument(
        "--min-points",
        type=int,
        default=MIN_RECORD,
        help=f"bars of history before the detector speaks (core floor {MIN_RECORD})",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="resting-turnover floor for the THIN arm, USDT; 0 disables the arm",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=0,
        help="bars of silence between an exit and the next entry; 30 is the hunter's own",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="the directory the report lands in")
    parser.add_argument(
        "--delimiter", default=",", help="the CSV separator; ';' opens better in some locales"
    )
    parser.add_argument(
        "--decisions",
        action="store_true",
        help="keep one row per bar in the JSON (large, and only sometimes wanted)",
    )
    parser.add_argument("--json", action="store_true", help="print the JSON report to stdout")
    parser.add_argument("--quiet", action="store_true", help="write the files, print nothing")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="feed timeout")
    return parser


def _series(args: argparse.Namespace) -> CandleSeries:
    """The candles: a file when one was named, the venue otherwise."""
    if args.csv or str(args.exchange) == "csv":
        if not args.csv:
            raise HtxError("--exchange csv needs --csv <path> to say which file")
        return load_csv(args.csv, symbol=args.symbol, period=args.period or "")
    endpoint = HTX_ENDPOINTS[1] if str(args.exchange) == "huobi" else HTX_ENDPOINTS[0]
    return fetch_klines(
        args.symbol,
        period=args.period or DEFAULT_PERIOD,
        days=float(args.days),
        timeout=float(args.timeout),
        endpoint=endpoint,
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Parse, read, run, write, print — and answer with an exit code that means something."""
    args = _parser().parse_args(argv)
    try:
        series = _series(args)
    except HtxError as exc:
        print(f"backtest: {exc}", file=sys.stderr)
        return EXIT_FEED
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
            log_decisions=bool(args.decisions),
        )
    except ValueError as exc:
        print(f"backtest: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if len(series) <= int(settings.min_points):
        print(
            f"backtest: {len(series)} bars of {series.period} is not enough for a verdict — "
            f"the detector needs more than {int(settings.min_points)}. Ask for more --days, "
            "a finer --period, or lower --min-points deliberately.",
            file=sys.stderr,
        )
        return EXIT_SHORT
    try:
        result = Backtester(series, settings).run()
    except ValueError as exc:
        print(f"backtest: {exc}", file=sys.stderr)
        return EXIT_USAGE
    report = Report(result)
    written = report.write(Path(args.out), delimiter=str(args.delimiter))
    if not args.quiet:
        print(report.summary_text())
        print(f"   wrote {written['json']} · {written['trades']} · {written['equity']}")
    if args.json:
        import json

        print(json.dumps(report.as_dict(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
