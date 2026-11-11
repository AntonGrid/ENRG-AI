"""``python -m agent.backtest.null`` — was that the signal, or was that the market?

Every number in :mod:`agent.backtest.walkforward` answers "what did the strategy pay". None of
them answers "how often does a book that makes the *same number of bets*, at the same times,
with the same exits, pay at least as much" — and that is the only question that separates an
edge from a lucky arrangement of the same lottery tickets. A 5 % return over a year in which
buy-and-hold made 60 % is not an edge; a 5 % return in a year of random entries that lost 20 %
might be. The sweep's own table was thirty rows of exactly this ambiguity.

What the null is, precisely
===========================

The coin replaces the verdict and **nothing else**. Same series, same venue, same fees, same
slippage, same latency, same stops and take-profits, same exit rules — one line of the engine
differs (:attr:`agent.backtest.engine.Settings.random_entry_rate`), and it is the line that
says which bar to buy. That matters more than the number of shuffles: a null that also
relaxed the exits would be measuring two differences and attributing both to the signal.

The rate is matched, not invented. ``rate`` defaults to the observed book's own closed trades
divided by its bars, so the null makes about as many bets as the strategy did. Matching the
count is what keeps the comparison to *which* bets rather than *how many*.

Reading the answer
==================

``p_value`` is the share of null books that did at least as well as the observed one, with the
``+1`` in both halves so that it can never be zero from a finite sample: twenty runs and a
strategy that beat all of them is ``1/21 ≈ 0.048``, not "significant". That is the honest
resolution of twenty runs, and the dataclass says so rather than letting the number imply
more than it can. For a p below 0.01 the caller asks for a hundred runs and waits.

Cost, stated up front
=====================

``runs`` full replays. The detector is the expensive part of a replay and the null cannot
skip it — the exits are verdict-driven, so a null book needs verdicts exactly as the strategy
does. Twenty runs over two thousand daily bars is an hour of single-core work; ``--days`` and
``--runs`` are the two knobs, and a shorter window with more runs is usually the better trade
because the question is about the shape of noise, not the length of the record.

No key, no RPC, no state file — the same wall as the rest of the package.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.backtest.data import (
    DEFAULT_PERIOD,
    DEFAULT_TIMEOUT,
    HTX_ENDPOINTS,
    PERIOD_SECONDS,
    CandleSeries,
    HtxError,
    fetch_klines,
)
from agent.backtest.engine import BacktestResult, Backtester, Settings
from agent.backtest.regime import align_regime
from agent.backtest.run import EXIT_FEED, EXIT_SHORT, EXIT_USAGE

__all__ = [
    "DEFAULT_RUNS",
    "STATISTICS",
    "NullResult",
    "main",
    "null_test",
    "percentile",
    "statistic_of",
]

#: How many null books to replay unless told otherwise. Twenty is not a p-value machine; it is
#: enough to see whether the observed book sits in the middle of the noise or outside it, and
#: the resolution is stated in :class:`NullResult` rather than implied.
DEFAULT_RUNS = 20

#: The statistics a null may be run on. ``avg_bps`` is the default because it is the one the
#: walk-forward gate is written in and the one that does not depend on how much of the book a
#: single position happened to be.
STATISTICS = ("avg_bps", "roi", "edge")


@dataclass(frozen=True)
class NullResult:
    """The observed book, a distribution of coin-flipped ones, and where the first sits.

    ``values`` is kept whole: a reader may want the histogram, and a summary that could not be
    checked against its own numbers is a summary a reader has to trust.
    """

    statistic: str
    observed: float
    runs: int
    rate: float
    mean: float
    std: float
    best: float
    worst: float
    p_value: float
    values: Tuple[float, ...] = ()

    @property
    def resolution(self) -> float:
        """The smallest p this many runs can express — ``1/(runs+1)``, stated, not hidden."""
        return 1.0 / (int(self.runs) + 1.0)

    @property
    def beats_every_run(self) -> bool:
        return bool(self.values) and float(self.observed) > max(self.values)

    def describe(self) -> str:
        """One sentence an operator can paste into a report, with the caveat attached."""
        if self.beats_every_run:
            return (
                f"{self.statistic} {self.observed:+.1f} against {self.runs} random books "
                f"(best {self.best:+.1f}, mean {self.mean:+.1f}): better than every one, "
                f"p ≤ {self.p_value:.3f} — the floor this many runs can measure, not zero"
            )
        return (
            f"{self.statistic} {self.observed:+.1f} against {self.runs} random books "
            f"(best {self.best:+.1f}, mean {self.mean:+.1f}): p = {self.p_value:.3f} — "
            f"{'inside' if self.p_value > 0.1 else 'near the edge of'} the noise, so the "
            "signal has not been shown to matter"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "statistic": str(self.statistic),
            "observed": round(float(self.observed), 6),
            "runs": int(self.runs),
            "rate": round(float(self.rate), 8),
            "mean": round(float(self.mean), 6),
            "std": round(float(self.std), 6),
            "best": round(float(self.best), 6),
            "worst": round(float(self.worst), 6),
            "p_value": round(float(self.p_value), 6),
            "resolution": round(float(self.resolution), 6),
            "values": [round(float(value), 6) for value in self.values],
        }


def percentile(values: Sequence[float], share: float) -> float:
    """The ``share`` quantile of ``values`` by linear interpolation — no numpy, no surprise.

    Written here rather than imported because the null is the one place in this package where
    a reader must be able to check the arithmetic on the back of an envelope: ten sorted
    numbers and a position between them.
    """
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    span = min(max(float(share), 0.0), 1.0) * (len(ordered) - 1)
    low = int(math.floor(span))
    high = min(low + 1, len(ordered) - 1)
    weight = span - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight



# ── the two readings a null is made of ─────────────────────────────────────


def statistic_of(result: BacktestResult, statistic: str = "avg_bps") -> float:
    """One number out of a finished run, in the units the statistic names.

    ``avg_bps`` is the mean closed trade after costs, in basis points; ``roi`` is the book's
    growth; ``edge`` is that growth minus buy-and-hold over the same bars. All three are
    computed from the ledger and the curve and from nothing else, so a null and an observation
    measured with the same name are the same kind of number.
    """
    name = str(statistic)
    if name not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, not {statistic!r}")
    closed = [float(trade.pnl_pct) for trade in result.closed]
    if name == "avg_bps":
        return 1e4 * float(statistics.fmean(closed)) if closed else 0.0
    capital = float(result.settings.capital)
    roi = float(result.final_equity) / capital - 1.0 if capital > 0.0 else 0.0
    if name == "roi":
        return roi
    rows = result.series.rows
    first = float(rows[0].close) if rows else 0.0
    last = float(rows[-1].close) if rows else 0.0
    hold = (last / first - 1.0) if first > 0.0 else 0.0
    return roi - hold


def trade_rate(result: BacktestResult) -> float:
    """Closed trades per bar — the frequency the null is matched to.

    Clamped to ``[0, 1]`` because the engine's rate is a per-bar probability and a strategy
    that traded more than once a bar (it cannot, but a future exit rule might allow it) would
    otherwise ask the coin for a probability above one.
    """
    bars = max(len(result.series.rows), 1)
    return min(max(len(result.closed) / float(bars), 0.0), 1.0)


def null_test(
    series: CandleSeries,
    *,
    settings: Optional[Settings] = None,
    benchmark: Optional[CandleSeries] = None,
    policy: Any = None,
    runs: int = DEFAULT_RUNS,
    statistic: str = "avg_bps",
    seed: int = 0,
    rate: Optional[float] = None,
    on_run: Optional[Callable[[int, float], None]] = None,
) -> NullResult:
    """Measure the strategy, then measure ``runs`` books that differ only in which bar they buy.

    The observed run is made first and with the policy attached, because that is the thing
    being defended; the null runs are made without it, because a coin does not consult a
    model and pretending otherwise would leave the model in the comparison twice. Everything
    else — series, fees, stops, latency, exits — is passed through unchanged.

    ``rate`` overrides the matched frequency. It exists for one honest use: asking "what if
    the coin bet half as often", which isolates whether an apparent edge was really a
    difference in exposure rather than a difference in selection.
    """
    config = settings or Settings()
    if not 0 <= int(runs) <= 1000:
        raise ValueError(f"runs must be between 0 and 1000, not {runs!r}")
    mask = (
        align_regime(series.rows, benchmark.rows)
        if benchmark is not None and len(benchmark.rows) > 0
        else None
    )
    observed_run = Backtester(series, config, policy=policy, regime=mask).run()
    observed = statistic_of(observed_run, statistic)
    frequency = trade_rate(observed_run) if rate is None else float(rate)
    if not 0.0 <= frequency <= 1.0:
        raise ValueError(f"rate must be a probability, not {frequency!r}")
    values: List[float] = []
    for index in range(int(runs)):
        flipped = replace(
            config, random_entry_rate=frequency, random_seed=int(seed) + int(index)
        )
        value = statistic_of(Backtester(series, flipped, regime=mask).run(), statistic)
        values.append(float(value))
        if on_run is not None:
            on_run(int(index), float(value))
    beaten = sum(1 for value in values if value >= observed)
    return NullResult(
        statistic=str(statistic),
        observed=float(observed),
        runs=int(runs),
        rate=float(frequency),
        mean=float(statistics.fmean(values)) if values else 0.0,
        std=float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        best=float(max(values)) if values else 0.0,
        worst=float(min(values)) if values else 0.0,
        p_value=(1.0 + beaten) / (len(values) + 1.0),
        values=tuple(values),
    )



# ── the command line ───────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.backtest.null",
        description=(
            "Replay a pair against books that buy the same number of times at random, and say "
            "whether the strategy is outside the noise. Reads candles, writes a table, signs "
            "nothing."
        ),
    )
    parser.add_argument("--symbol", default="btcusdt", help="the pair, e.g. btcusdt")
    parser.add_argument("--exchange", default="htx", choices=("htx", "huobi"), help="which endpoint")
    parser.add_argument(
        "--period", default=DEFAULT_PERIOD, choices=sorted(PERIOD_SECONDS), help="the bar"
    )
    parser.add_argument("--days", type=float, default=365.0, help="how far back to read, in days")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="how many random books")
    parser.add_argument(
        "--statistic", default="avg_bps", choices=STATISTICS, help="what is compared"
    )
    parser.add_argument("--seed", type=int, default=0, help="the first coin; runs use seed+i")
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="bets per bar for the null; default is the observed book's own rate",
    )
    parser.add_argument("--benchmark", default="", help="a pair whose trend gates entries")
    parser.add_argument("--capital", type=float, default=1000.0, help="the starting book, USDT")
    parser.add_argument("--share", type=float, default=1.0, help="the share of cash one entry spends")
    parser.add_argument("--stop", type=float, default=0.02, help="the fixed stop below the fill")
    parser.add_argument("--confidence", type=float, default=0.0, help="the floor a PANIC must reach")
    parser.add_argument("--fee", type=float, default=0.001, help="the fee per side")
    parser.add_argument("--slippage", type=float, default=0.001, help="slippage per fill")
    parser.add_argument("--latency-bars", type=int, default=1, help="bars from decision to fill")
    parser.add_argument("--cooldown-bars", type=int, default=0, help="bars of silence after an exit")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per request")
    parser.add_argument("--json", default="", help="write the whole distribution here")
    parser.add_argument("--quiet", action="store_true", help="print nothing but the sentence")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read one pair, replay the strategy and ``--runs`` random books, and print the sentence.

    Exit 0 whenever the null could be measured, *including* when the strategy turned out to be
    inside the noise — that is the answer the tool exists to give. 2 is a caller's mistake, 3 a
    feed that did not answer, 4 a record too short for the detector to speak on.
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
            cooldown_bars=int(args.cooldown_bars),
        )
    except ValueError as exc:
        print(f"null: {exc}", file=sys.stderr)
        return EXIT_USAGE

    endpoint = HTX_ENDPOINTS[1 if str(args.exchange) == "huobi" else 0]
    try:
        series = fetch_klines(
            str(args.symbol),
            period=str(args.period),
            days=float(args.days),
            timeout=float(args.timeout),
            endpoint=endpoint,
        )
        benchmark = (
            fetch_klines(
                str(args.benchmark),
                period=str(args.period),
                days=float(args.days),
                timeout=float(args.timeout),
                endpoint=endpoint,
            )
            if str(args.benchmark).strip()
            else None
        )
    except HtxError as exc:
        print(f"null: {exc}", file=sys.stderr)
        return EXIT_FEED

    def announce(index: int, value: float) -> None:
        if not args.quiet:
            print(f"   · run {index + 1:>3}  {args.statistic} {value:+.1f}")

    if not args.quiet:
        print(
            f"── null · {args.symbol} · {args.period} · {len(series.rows)} bars · "
            f"{int(args.runs)} random books · comparing {args.statistic}"
        )
    try:
        result = null_test(
            series,
            settings=settings,
            benchmark=benchmark,
            runs=int(args.runs),
            statistic=str(args.statistic),
            seed=int(args.seed),
            rate=args.rate,
            on_run=announce,
        )
    except ValueError as exc:
        print(f"null: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not result.values and result.observed == 0.0:
        print("null: the strategy never traded, so there is nothing to compare", file=sys.stderr)
        return EXIT_SHORT
    print(f"   {result.describe()}")
    if str(args.json).strip():
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        import json

        target.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"   wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

