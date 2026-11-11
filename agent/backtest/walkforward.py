"""``python -m agent.backtest.walkforward`` — the only number that cannot be fitted.

A backtest over the whole record answers "what would this have paid", and every model fitted
on that record answers the same question with the answer already in its hands. The sweep was
that mistake at the level of a threshold; a fifteen-weight policy fitted on five years of bars
and tested on the same five years is the identical mistake with a nicer report. So this file
does the one thing that makes the policy's number worth reading: it fits on the past and
measures on the future, and it never lets the two overlap.

How the split is drawn
======================

Expanding, never sliding. The first test year is trained on everything before it, the second
on everything before *it* — so the model always has more history than the trade it is about
to make, which is the only configuration that exists in production. A sliding window would
answer "what would a model with two years of memory have done", and nobody deploys that.

The test window is measured on the **book**, not on the trades: the reading is the growth of
``cash + units × close`` from the bar before the window to its last bar. Measuring the mean of
the closed trades in the window would flatter a strategy that simply stopped trading, and the
whole point of the exercise is to catch exactly that.

The four ablations, and what each one is *for*
==============================================

``baseline``
    The detector alone. This is the thing that lost 11 % in the first sweep, and it is here
    so that every other row has something to be better *than*.
``policy``
    The detector plus the learned veto, and nothing else changed. If this row does not beat
    ``baseline``, the features carry no information about which panics pay and the honest move
    is to stop — not to add a stop-loss and call the combination a discovery.
``policy+atr``
    The veto plus a stop measured in the instrument's own volatility and a fixed fraction of
    equity at risk. The sweep's 374 stops on the entry bar were a 2 % stop inside a 5 % bar:
    a risk rule that fires before the position can be right is not a risk rule.
``policy+atr+regime``
    The same, entered only while the benchmark is above its own mean. This is where the
    sweep's one statistically loud fact (risk-off, t = −3.20) is spent as a filter instead of
    being admired in a table.

What "it worked" means
======================

:data:`GATE` — 30 basis points a trade after costs, ``t ≥ 2``, profit factor ``≥ 1.3``,
Sharpe ``≥ 1`` — and :func:`summary` says which of the four failed, per ablation. A run that
fails the gate is a **result**: ``null.py`` exists to answer the next question ("was that even
better than a coin"), and the honest output of this whole exercise may well be "no alpha,
here is the evidence".

Everything here reads prices it is given and writes a table. No key, no RPC, no state file.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import math
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agent.backtest.admission import (
    add_admission_arguments,
    admission_from_args,
    admit,
    masked,
)
from agent.backtest.criteria import declared
from agent.backtest.data import (
    DEFAULT_PERIOD,
    DEFAULT_TIMEOUT,
    HTX_ENDPOINTS,
    HTX_TICKERS_ENDPOINTS,
    PERIOD_SECONDS,
    CandleSeries,
    HtxError,
    declines,
    fetch_klines,
    load_csv,
    universe,
    write_csv,
)
from agent.backtest.engine import BacktestResult, Backtester, Settings, Trade
from agent.backtest.exits import (
    add_exit_arguments,
    describe_exits,
    exits_from_args,
)
from agent.backtest.features import POLICY_FEATURES, enrich, series_features
from agent.backtest.labels import DEFAULT_COST, label_at, net_forward_return
from agent.backtest.policy import DEFAULT_L2, LogisticPolicy
from agent.backtest.power import (
    DECLARED_ALPHA,
    DECLARED_POWER,
    Resolution,
    resolution,
    spread_bps,
)
from agent.backtest.regime import DEFAULT_REGIME_WINDOW, align_regime
from agent.backtest.run import EXIT_FEED, EXIT_SHORT, EXIT_USAGE
from agent.physics.portfolio import MIN_RECORD
from agent.physics.volatility_hunter import PANIC, PanicSignal

__all__ = [
    "ABLATIONS",
    "DEFAULT_OUT",
    "GATE",
    "Ablation",
    "Fold",
    "Reading",
    "Summary",
    "expanding_folds",
    "format_readings",
    "format_summary",
    "pinned_ablations",
    "gate_of",
    "main",
    "panel",
    "summary",
    "verdict_series",
    "walk_forward",
]

#: Where the walk-forward writes its table. Beside the sweep, for the same reason: scratch
#: space, never the directory the live loop's state file lives in.
DEFAULT_OUT = "data/walkforward"

#: The bar a strategy must clear to be called an edge rather than a story. Each number is a
#: decision, and each one is the threshold under which the effort is not worth the risk it
#: claims to manage:
#:
#: * ``edge_bps`` — 30 basis points a trade, which is roughly three times a maker round trip
#:   and about a tenth of a retail taker round trip. Below that the strategy is paying the
#:   venue and calling the remainder a return.
#: * ``t_stat`` — 2, the usual line, and deliberately not adjusted for the four ablations:
#:   the four are nested hypotheses tested on the same windows, so a reader who wants a
#:   Bonferroni-corrected line should read this as 0.05/4 and expect a t near 2.5.
#: * ``profit_factor`` — 1.3, which is what a strategy with a 40 % win rate and a 2R payoff
#:   looks like before the fees eat the tail.
#: * ``sharpe`` — 1 on the test windows' equity, annualised. A daily strategy under 1 is a
#:   strategy whose drawdowns need a better story than its returns.
GATE = {
    "edge_bps": 30.0,
    "t_stat": 2.0,
    "profit_factor": 1.3,
    "sharpe": 1.0,
}


@dataclass(frozen=True)
class Fold:
    """One train/test split, as bar indices — half-open, so a bar is in exactly one side.

    ``train_start`` is always zero: the window expands, never slides, and a field that could
    only ever hold one value would be a field that invites someone to slide it.
    """

    name: str
    test_start: int
    test_end: int

    @property
    def train_bars(self) -> int:
        return int(self.test_start)

    @property
    def test_bars(self) -> int:
        return int(self.test_end) - int(self.test_start)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fold": str(self.name),
            "train_bars": self.train_bars,
            "test_start": int(self.test_start),
            "test_end": int(self.test_end),
            "test_bars": self.test_bars,
        }


@dataclass(frozen=True)
class Sample:
    """One bar the detector would have bought, with what it looked like and what it paid.

    ``label`` is ``None`` when the future is not in the record — the last ``horizon`` bars —
    and those rows are dropped at fit time rather than zeroed, because a zero label at the
    edge of history is not a flat bar, it is a bar whose answer nobody knows.
    """

    bar: int
    case: Dict[str, float]
    net: Optional[float]
    label: Optional[float]


@dataclass(frozen=True)
class Ablation:
    """One hypothesis, as the settings it changes and whether it is allowed to use the model."""

    name: str
    use_policy: bool = False
    require_regime: bool = False
    overrides: Mapping[str, Any] = field(default_factory=dict)


#: The four rows of the table, in the order they are read: each one adds exactly one idea to
#: the row above it, so a difference between two adjacent rows is the worth of that idea.
ABLATIONS: Tuple[Ablation, ...] = (
    Ablation("baseline"),
    Ablation("policy", use_policy=True),
    Ablation(
        "policy+atr",
        use_policy=True,
        overrides={"stop_mode": "atr", "atr_multiple": 2.0, "risk_per_trade": 0.02},
    ),
    Ablation(
        "policy+atr+regime",
        use_policy=True,
        require_regime=True,
        overrides={"stop_mode": "atr", "atr_multiple": 2.0, "risk_per_trade": 0.02},
    ),
)


def pinned_ablations(overrides: Mapping[str, Any]) -> Tuple[Ablation, ...]:
    """The ablations with an operator's explicit exit flags over their own overrides.

    Without this, ``--atr-multiple 3`` would print a row called ``policy+atr`` that ran at
    2.0 — the ablation's own value quietly winning over the operator's. An explicit flag
    wins, because a knob that does not reach the row it names is a lie in a table.
    """
    if not overrides:
        return ABLATIONS
    return tuple(
        replace(ablation, overrides={**dict(ablation.overrides), **dict(overrides)})
        for ablation in ABLATIONS
    )


@dataclass(frozen=True)
class Reading:
    """One ablation on one fold: the book's growth, the trades behind it, and their spread.

    ``returns`` is kept as the raw list of per-trade returns, because every aggregate that
    matters here — the mean, its ``t``, the profit factor — is a statement about *those*
    numbers, and a pool across folds that had only the means would be a mean of means.
    """

    ablation: str
    fold: str
    train_samples: int
    trades: int
    roi: float
    buy_hold: float
    edge: float
    win_rate: float
    profit_factor: float
    sharpe: float
    t_stat: float
    stops: int = 0
    targets: int = 0
    time_exits: int = 0
    returns: Tuple[float, ...] = ()
    note: str = ""

    @property
    def avg_bps(self) -> float:
        """The mean trade, in basis points — the number the gate is written in."""
        if not self.returns:
            return 0.0
        return 1e4 * float(statistics.fmean(self.returns))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ablation": self.ablation,
            "fold": self.fold,
            "train_samples": int(self.train_samples),
            "trades": int(self.trades),
            "roi": round(float(self.roi), 8),
            "buy_hold": round(float(self.buy_hold), 8),
            "edge": round(float(self.edge), 8),
            "avg_bps": round(self.avg_bps, 4),
            "win_rate": round(float(self.win_rate), 6),
            "profit_factor": round(float(self.profit_factor), 6),
            "sharpe": round(float(self.sharpe), 4),
            "t_stat": round(float(self.t_stat), 4),
            "stops": int(self.stops),
            "targets": int(self.targets),
            "time_exits": int(self.time_exits),
            "note": self.note,
        }


@dataclass(frozen=True)
class Summary:
    """An ablation over every fold, pooled — and the gate's verdict, in words.

    Pooling is the honest aggregate here because the folds are consecutive slices of the same
    out-of-sample period: the trades do not overlap, so the mean of them is a mean of
    independent bets, and the ``t`` over that pool is the number the gate asks for.
    """

    ablation: str
    folds: int
    trades: int
    edge_bps: float
    t_stat: float
    profit_factor: float
    sharpe: float
    roi: float
    passed: bool
    failures: Tuple[str, ...] = ()
    #: The edge floor this row was judged against — carried, because a caller may pass
    #: its own gate and the resolution must be read against the same one.
    gate_bps: float = GATE["edge_bps"]
    #: The per-trade spread the pooled edge was read with, in basis points.
    sigma_bps: float = 0.0
    #: The smallest edge these many trades can name at :data:`DECLARED_ALPHA` and
    #: :data:`DECLARED_POWER` — ``inf`` when there were too few trades to have one.
    min_edge_bps: float = math.inf
    alpha: float = DECLARED_ALPHA
    power: float = DECLARED_POWER

    @property
    def resolution(self) -> Resolution:
        """This book's reading resolution beside the edge it was judged against."""
        return Resolution(
            trades=int(self.trades),
            sigma_bps=float(self.sigma_bps),
            gate_bps=float(self.gate_bps),
            min_edge_bps=float(self.min_edge_bps),
            alpha=float(self.alpha),
            power=float(self.power),
        )

    @property
    def resolvable(self) -> bool:
        """Could this book have seen an edge the size of the gate at all?"""
        return self.resolution.measurable

    @property
    def verdict(self) -> str:
        """``pass``, ``fail`` — or ``unmeasured`` when the book could not see the gate.

        The third word is the honest one and the reason :mod:`agent.backtest.power` exists:
        a book too small to resolve thirty basis points has produced no evidence in either
        direction, and calling that ``fail`` would be an accusation dressed as a result.
        """
        if not self.resolvable:
            return "unmeasured"
        return "pass" if self.passed else "fail"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ablation": self.ablation,
            "folds": int(self.folds),
            "trades": int(self.trades),
            "edge_bps": round(float(self.edge_bps), 4),
            "t_stat": round(float(self.t_stat), 4),
            "profit_factor": round(float(self.profit_factor), 6),
            "sharpe": round(float(self.sharpe), 4),
            "roi": round(float(self.roi), 8),
            "passed": bool(self.passed),
            "verdict": self.verdict,
            "failures": list(self.failures),
            "resolution": self.resolution.as_dict(),
        }


# ── the splits ─────────────────────────────────────────────────────────────


def _year_at(at: float) -> int:
    """The UTC calendar year of a timestamp — the calendar a fold is named after."""
    return int(time.gmtime(float(at)).tm_year)


def _first_at_or_after(stamps: Sequence[float], moment: float) -> Optional[int]:
    """The index of the first bar at or after ``moment`` — ``None`` when the record ends first."""
    for index, stamp in enumerate(stamps):
        if float(stamp) >= float(moment):
            return index
    return None


def expanding_folds(
    rows: Sequence[Any],
    *,
    years: Optional[Sequence[int]] = None,
    splits: int = 3,
    min_train: int = MIN_RECORD + 1,
) -> List[Fold]:
    """Consecutive calendar years as test windows, each trained on everything before it.

    ``years`` names them explicitly; omitted, the last ``splits`` years *present in the
    record* are used, and the first year is never a test window because it is where the
    training has to come from. A year too short to hold two bars is skipped rather than
    reported as a fold with nothing in it, and a year whose training block is shorter than
    the detector's own record floor is skipped for the same reason: there is no model to
    test there, only a promise.

    The test window ends at the first bar of the next year and not at "the last bar of this
    year", which is the same index — the half-open convention is kept so that a bar cannot
    be counted in two folds and a fold cannot be measured on a bar its training saw.
    """
    stamps = [float(row.at) for row in rows]
    if years is None:
        present = sorted({_year_at(at) for at in stamps})
        years = present[1:][-max(int(splits), 1):] if len(present) > 1 else []
    out: List[Fold] = []
    for year in years:
        start = _first_at_or_after(stamps, float(calendar.timegm((int(year), 1, 1, 0, 0, 0, 0, 0, 0))))
        if start is None:
            continue
        end = _first_at_or_after(
            stamps, float(calendar.timegm((int(year) + 1, 1, 1, 0, 0, 0, 0, 0, 0)))
        )
        if end is None:
            end = len(rows)
        if end - start < 2:
            continue
        if start < max(int(min_train), 2):
            continue
        out.append(Fold(str(int(year)), int(start), int(end)))
    return out


# ── the panel: what the detector would have bought, and what it paid ───────


def verdict_series(
    series: CandleSeries, settings: Optional[Settings] = None
) -> List[PanicSignal]:
    """The detector's word on every bar, measured on the prefix that ends there.

    The same call the engine makes, on the same slices, and for the same reason: a panel
    assembled from a detector that could see further than the bar would train a policy on
    verdicts the live loop can never produce.
    """
    config = settings or Settings()
    rows = series.rows
    yields = declines(rows, window=config.window)
    volumes = [float(row.volume) for row in rows]
    if config.admission is not None:
        # The same filter the engine applies, on the same bars: a panel trained on verdicts
        # the engine would never have produced is a policy fitted to a different strategy.
        admitted = admit(rows, config.admission)
        yields = masked(yields, admitted)
        volumes = masked(volumes, admitted)
    detector = config.detector()
    name = series.symbol or "instrument"
    return [
        detector.measure(yields[: index + 1], volumes[: index + 1], name=name)
        for index in range(len(rows))
    ]


def panel(
    series: CandleSeries,
    settings: Optional[Settings] = None,
    *,
    benchmark: Optional[CandleSeries] = None,
    horizon: int = 3,
    cost: float = DEFAULT_COST,
    threshold: float = 0.0,
    signals: Optional[Sequence[PanicSignal]] = None,
) -> List[Sample]:
    """Every bar the detector would have bought under ``settings``, described and labelled.

    The filter mirrors :meth:`agent.backtest.engine.Backtester.wants_entry` minus the policy:
    a ``PANIC`` above the confidence floor. That is deliberate — the model is trained on the
    population it is asked to veto, and training it on all bars would teach it to predict a
    market instead of a decision.

    ``signals`` lets a caller pass an already-measured series (the CLI measures once and uses
    the panel for every fold), and it is checked against the bars so that a panel cannot be
    silently built on a different series than the one it describes.
    """
    config = settings or Settings()
    rows = series.rows
    if len(signals or []) not in (0, len(rows)):
        raise ValueError(
            f"the verdict series has {len(signals or [])} bars against the series' {len(rows)}"
        )
    words = list(signals) if signals else verdict_series(series, config)
    features = series_features(rows, atr_window=int(config.atr_window))
    mask = (
        align_regime(rows, benchmark.rows)
        if benchmark is not None and len(benchmark.rows) > 0
        else None
    )
    floor = float(config.confidence)
    admitted = admit(rows, config.admission)
    out: List[Sample] = []
    bars_since_panic = 0
    for index, signal in enumerate(words):
        bars_since_panic = 0 if signal.verdict == PANIC else bars_since_panic + 1
        if admitted[index]:
            # A withheld bar is not a bar the engine could have bought, so it is not a
            # training row either — the panel and the book have to agree on what happened.
            continue
        if signal.verdict != PANIC or float(signal.confidence) < floor:
            continue
        regime = 0.5 if mask is None else (1.0 if mask[index] else 0.0)
        out.append(
            Sample(
                bar=int(index),
                case=enrich(
                    features[index],
                    confidence=float(signal.confidence),
                    regime=regime,
                    bars_since_panic=bars_since_panic,
                ),
                net=net_forward_return(rows, index, horizon, cost=cost),
                label=label_at(rows, index, horizon, cost=cost, threshold=threshold),
            )
        )
    return out



# ── the readings ───────────────────────────────────────────────────────────


def _profit_factor(returns: Sequence[float]) -> float:
    """Gross wins over gross losses; ``inf`` when nothing lost, and that is a fact, not a bug."""
    wins = sum(float(value) for value in returns if float(value) > 0.0)
    losses = -sum(float(value) for value in returns if float(value) < 0.0)
    if losses <= 0.0:
        return math.inf if wins > 0.0 else 0.0
    return wins / losses


def _t_stat(values: Sequence[float]) -> float:
    """Mean over its own standard error — ``0`` when there is nothing to divide by.

    Zero and not infinity: two trades that agree perfectly are not evidence of an edge, they
    are two trades. The gate wants a sample that could have disagreed.
    """
    count = len(values)
    if count < 2:
        return 0.0
    spread = statistics.stdev(float(value) for value in values)
    if spread <= 0.0:
        return 0.0
    return float(statistics.fmean(values)) / spread * math.sqrt(count)


def _sharpe(equities: Sequence[float], bars_per_year: float) -> float:
    """Annualised Sharpe of the book's own bar-to-bar returns, over the window only.

    Measured on equity and not on trades because the drawdown that matters is the one an
    operator watching the screen would have lived through, and a strategy that trades twice a
    year has an equity curve with a hundred flat bars in it and a trade ledger with none.
    """
    values = [float(value) for value in equities]
    if len(values) < 3:
        return 0.0
    returns = [
        later / earlier - 1.0
        for earlier, later in zip(values, values[1:])
        if float(earlier) > 0.0
    ]
    if len(returns) < 2:
        return 0.0
    spread = statistics.stdev(returns)
    if spread <= 0.0:
        return 0.0
    return float(statistics.fmean(returns)) / spread * math.sqrt(max(bars_per_year, 1.0))



def _window(
    result: BacktestResult,
    fold: Fold,
    ablation: str,
    train_samples: int,
    bars_per_year: float,
    *,
    note: str = "",
) -> Reading:
    """The fold's slice of a finished run, read off the book and the ledger.

    Both sides come from the same run, so the ROI and the trades cannot disagree: the ROI is
    the equity curve from the bar before the window to its last bar, and the trades are the
    ones *opened* inside it. A position carried in from the training window is therefore in
    the ROI (it was real money) and not in the trade statistics (its decision was not made
    under test) — the asymmetry is stated here because either half alone would flatter.
    """
    rows = result.series.rows
    equity = result.equity
    start = max(int(fold.test_start), 1)
    end = min(int(fold.test_end), len(equity))
    if end <= start or start >= len(rows):
        return Reading(
            ablation=ablation,
            fold=fold.name,
            train_samples=int(train_samples),
            trades=0,
            roi=0.0,
            buy_hold=0.0,
            edge=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe=0.0,
            t_stat=0.0,
            note=note or "the fold holds no bars to measure",
        )
    base = float(equity[start - 1]["equity"])
    final = float(equity[end - 1]["equity"])
    roi = (final / base - 1.0) if base > 0.0 else 0.0
    price_in = float(rows[start - 1].close)
    price_out = float(rows[end - 1].close)
    buy_hold = (price_out / price_in - 1.0) if price_in > 0.0 else 0.0
    trades = [
        trade
        for trade in result.trades
        if not trade.open and start <= int(trade.entry_bar) < end
    ]
    returns = tuple(float(trade.pnl_pct) for trade in trades)
    return Reading(
        ablation=ablation,
        fold=fold.name,
        train_samples=int(train_samples),
        trades=len(returns),
        roi=float(roi),
        buy_hold=float(buy_hold),
        edge=float(roi - buy_hold),
        win_rate=(sum(1 for value in returns if value > 0.0) / len(returns))
        if returns
        else 0.0,
        profit_factor=_profit_factor(returns),
        sharpe=_sharpe(
            [float(row["equity"]) for row in equity[start - 1 : end]], bars_per_year
        ),
        t_stat=_t_stat(returns),
        stops=sum(1 for trade in trades if trade.exit_reason == "stop"),
        targets=sum(1 for trade in trades if trade.exit_reason == "target"),
        time_exits=sum(1 for trade in trades if trade.exit_reason == "time"),
        returns=returns,
        note=note,
    )



def walk_forward(
    series: CandleSeries,
    *,
    settings: Optional[Settings] = None,
    benchmark: Optional[CandleSeries] = None,
    folds: Optional[Sequence[Fold]] = None,
    ablations: Sequence[Ablation] = ABLATIONS,
    horizon: int = 3,
    cost: float = DEFAULT_COST,
    threshold: float = 0.0,
    l2: float = DEFAULT_L2,
    regime_window: int = DEFAULT_REGIME_WINDOW,
    features: Sequence[str] = POLICY_FEATURES,
    on_reading: Optional[Callable[[Reading], None]] = None,
) -> List[Reading]:
    """Fit on the past, measure on the future, once per fold and per ablation.

    One panel and one benchmark mask are built for the whole series and then *sliced*, not
    rebuilt: the detector's prefixes are the expensive part, and the splits are a choice about
    which slice is which rather than a second measurement. The policy is refitted inside every
    fold on that fold's training slice alone — a policy fitted once on all the panics and then
    judged on them is the in-sample illusion this file exists to refuse.
    """
    config = settings or Settings()
    rows = series.rows
    splits = list(folds) if folds is not None else expanding_folds(rows)
    if not splits:
        return []
    samples = panel(
        series,
        config,
        benchmark=benchmark,
        horizon=horizon,
        cost=cost,
        threshold=threshold,
    )
    mask = (
        align_regime(rows, benchmark.rows, window=int(regime_window))
        if benchmark is not None and len(benchmark.rows) > 0
        else None
    )
    interval = float(series.interval)
    bars_per_year = (365.25 * 86400.0 / interval) if interval > 0.0 else 365.25
    readings: List[Reading] = []
    for fold in splits:
        train = [
            sample
            for sample in samples
            if int(sample.bar) < int(fold.test_start) and sample.label is not None
        ]
        policy = None
        fit_refusal = ""
        if train:
            try:
                policy = LogisticPolicy.fit(
                    ((sample.case, sample.label) for sample in train),
                    features=features,
                    l2=float(l2),
                    horizon=int(horizon),
                    cost=float(cost),
                )
            except ValueError as exc:
                # A training block whose panics all paid (or all lost) cannot teach a
                # ranking, and the honest answer is to run the row without a model and say
                # so — not to invent a second fold with a friendlier sample.
                fit_refusal = f"the fit refused the training block: {exc}"
        for ablation in ablations:
            note = ""
            if ablation.use_policy and policy is None:
                note = (
                    fit_refusal
                    or "no labelled training panics: this row is the baseline by another name"
                )
            if ablation.require_regime and mask is None:
                note = f"{note} no benchmark: the regime filter had nothing to filter".strip()
            config_here = replace(
                config,
                log_decisions=False,
                require_risk_on=bool(ablation.require_regime),
                **dict(ablation.overrides),
            )
            result = Backtester(
                series,
                config_here,
                policy=policy if ablation.use_policy else None,
                regime=mask,
            ).run()
            reading = _window(
                result,
                fold,
                ablation.name,
                len(train),
                bars_per_year,
                note=note,
            )
            readings.append(reading)
            if on_reading is not None:
                on_reading(reading)
    return readings




def summary(
    readings: Sequence[Reading], *, gate: Mapping[str, float] = GATE
) -> List[Summary]:
    """The readings pooled per ablation, and the gate's verdict for each one.

    The folds are consecutive and their trades do not overlap, so the pool is a mean of
    independent bets rather than a mean of means; the ROI compounds across folds for the same
    reason a book does. Sharpe is averaged and not pooled — it is a property of a curve, and
    pooling two curves would invent a third that nobody held.
    """
    order: List[str] = []
    grouped: Dict[str, List[Reading]] = {}
    for reading in readings:
        if reading.ablation not in grouped:
            grouped[reading.ablation] = []
            order.append(reading.ablation)
        grouped[reading.ablation].append(reading)
    out: List[Summary] = []
    for name in order:
        group = grouped[name]
        pooled = [value for reading in group for value in reading.returns]
        edge_bps = 1e4 * float(statistics.fmean(pooled)) if pooled else 0.0
        t_stat = _t_stat(pooled)
        factor = _profit_factor(pooled)
        sigma_bps = spread_bps(pooled)
        read = resolution(
            len(pooled),
            sigma_bps,
            float(gate["edge_bps"]),
            alpha=DECLARED_ALPHA,
            power=DECLARED_POWER,
        )
        sharpe = (
            float(statistics.fmean([reading.sharpe for reading in group])) if group else 0.0
        )
        roi = 1.0
        for reading in group:
            roi *= 1.0 + float(reading.roi)
        failures: List[str] = []
        if edge_bps < float(gate["edge_bps"]):
            failures.append(f"edge {edge_bps:+.1f} bps < {float(gate['edge_bps']):.0f} bps")
        if t_stat < float(gate["t_stat"]):
            failures.append(f"t {t_stat:+.2f} < {float(gate['t_stat']):.1f}")
        if factor < float(gate["profit_factor"]):
            failures.append(f"profit factor {factor:.2f} < {float(gate['profit_factor']):.1f}")
        if sharpe < float(gate["sharpe"]):
            failures.append(f"sharpe {sharpe:.2f} < {float(gate['sharpe']):.1f}")
        if not read.measurable:
            heard = (
                "no dispersion"
                if math.isinf(read.min_edge_bps)
                else f"only {read.min_edge_bps:.0f} bps"
            )
            failures.append(
                f"unresolvable: {len(pooled)} trade(s) at {sigma_bps:.0f} bps of spread "
                f"resolve {heard}, so the gate's {float(gate['edge_bps']):.0f} bps cannot "
                "be settled either way — raise the count, not the claim"
            )
        out.append(
            Summary(
                ablation=name,
                folds=len(group),
                trades=len(pooled),
                edge_bps=float(edge_bps),
                t_stat=float(t_stat),
                profit_factor=float(factor),
                sharpe=float(sharpe),
                roi=float(roi - 1.0),
                passed=not failures,
                failures=tuple(failures),
                gate_bps=float(gate["edge_bps"]),
                sigma_bps=float(sigma_bps),
                min_edge_bps=float(read.min_edge_bps),
                alpha=float(DECLARED_ALPHA),
                power=float(DECLARED_POWER),
            )
        )
    return out


def gate_of(
    readings: Sequence[Reading],
    *,
    ablation: Optional[str] = None,
    gate: Mapping[str, float] = GATE,
) -> Summary:
    """The verdict for one ablation (the last by default) — the shape a caller wants to print."""
    rows = summary(readings, gate=gate)
    if not rows:
        raise ValueError(
            "no readings to judge: a walk-forward that measured nothing has no gate"
        )
    if ablation is None:
        return rows[-1]
    for row in rows:
        if row.ablation == ablation:
            return row
    raise ValueError(f"{ablation!r} is not among the ablations that were measured")


# ── the table ──────────────────────────────────────────────────────────────

#: The columns of the printed per-fold table, in the order they are read: the split, the
#: hypothesis, then what it did. ``bps`` is the mean trade after costs, and it sits beside
#: ``edge`` because one is per bet and the other is per window — a strategy can win a window
#: on a single lucky trade and still lose money per bet.
READING_HEADERS = (
    "fold",
    "ablation",
    "train",
    "trades",
    "roi",
    "buy & hold",
    "edge",
    "bps",
    "win",
    "pf",
    "sharpe",
    "t",
)


def format_readings(readings: Sequence[Reading]) -> str:
    """The per-fold table: one line per ablation per fold, and the notes under it."""
    widths = (6, 20, 6, 6, 9, 9, 9, 8, 6, 6, 7, 6)
    lines = ["  ".join(header.ljust(width) for header, width in zip(READING_HEADERS, widths))]
    lines.append("  ".join("-" * width for width in widths))
    for reading in readings:
        factor = reading.profit_factor
        cells = (
            reading.fold,
            reading.ablation,
            str(reading.train_samples),
            str(reading.trades),
            f"{reading.roi:+.2%}",
            f"{reading.buy_hold:+.2%}",
            f"{reading.edge:+.2%}",
            f"{reading.avg_bps:+.0f}",
            f"{reading.win_rate:.0%}",
            "inf" if math.isinf(factor) else f"{factor:.2f}",
            f"{reading.sharpe:+.2f}",
            f"{reading.t_stat:+.2f}",
        )
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(cells, widths)))
        if reading.note:
            lines.append(f"      ! {reading.note}")
    return "\n".join(lines)


def format_summary(summaries: Sequence[Summary]) -> str:
    """The pooled table and, under each row that failed, the sentence saying which gate failed."""
    widths = (20, 6, 7, 10, 7, 7, 8, 9, 8, 11)
    headers = (
        "ablation",
        "folds",
        "trades",
        "bps/trade",
        "t",
        "pf",
        "sharpe",
        "roi",
        "± bps",
        "gate",
    )
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    for row in summaries:
        cells = (
            row.ablation,
            str(row.folds),
            str(row.trades),
            f"{row.edge_bps:+.1f}",
            f"{row.t_stat:+.2f}",
            "inf" if math.isinf(row.profit_factor) else f"{row.profit_factor:.2f}",
            f"{row.sharpe:+.2f}",
            f"{row.roi:+.2%}",
            "inf" if math.isinf(row.min_edge_bps) else f"{row.min_edge_bps:.0f}",
            row.verdict,
        )
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(cells, widths)))
        for failure in row.failures:
            lines.append(f"      ! {failure}")
    return "\n".join(lines)



# ── the command line ───────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.backtest.walkforward",
        description=(
            "Fit the policy on the past, measure it on the future, and say whether the edge "
            "clears the gate. Reads candles, writes a table, signs nothing."
        ),
    )
    parser.add_argument("--symbols", default="", help="pairs separated by commas, e.g. btcusdt,ethusdt")
    parser.add_argument(
        "--top", type=int, default=20, help="pairs to take from the venue's own turnover ranking"
    )
    parser.add_argument("--quote", default="usdt", help="the quote currency of the ranking")
    parser.add_argument("--exchange", default="htx", choices=("htx", "huobi"), help="which endpoint")
    parser.add_argument(
        "--period", default=DEFAULT_PERIOD, choices=sorted(PERIOD_SECONDS), help="the bar"
    )
    parser.add_argument(
        "--days", type=float, default=2000.0, help="how far back to read, in days (2000 bars max)"
    )
    parser.add_argument("--benchmark", default="btcusdt", help="the regime's benchmark; '' disables")
    parser.add_argument("--regime-window", type=int, default=DEFAULT_REGIME_WINDOW, help="bars in the benchmark's mean")
    parser.add_argument("--capital", type=float, default=1000.0, help="the starting book, USDT")
    parser.add_argument("--share", type=float, default=1.0, help="the share of cash one entry may spend")
    parser.add_argument("--stop", type=float, default=0.02, help="the fixed stop below the fill")
    parser.add_argument(
        "--confidence", type=float, default=0.0, help="the floor a PANIC must reach to be bought"
    )
    parser.add_argument("--fee", type=float, default=0.001, help="the fee per side")
    parser.add_argument("--slippage", type=float, default=0.001, help="slippage per fill")
    parser.add_argument("--latency-bars", type=int, default=1, help="bars between a decision and its fill")
    parser.add_argument("--cooldown-bars", type=int, default=0, help="bars of silence after an exit")
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="turnover floor for the hunter's THIN arm, USDT; 0 is off",
    )
    parser.add_argument("--horizon", type=int, default=3, help="bars the label looks ahead")
    parser.add_argument("--threshold", type=float, default=0.0, help="the net return a label must beat, in per cent")
    parser.add_argument("--cost", type=float, default=DEFAULT_COST, help="the round trip the label pays")
    parser.add_argument("--l2", type=float, default=DEFAULT_L2, help="the policy's ridge")
    parser.add_argument("--splits", type=int, default=3, help="how many test years to take")
    parser.add_argument(
        "--folds", default="", help="test years separated by commas, e.g. 2024,2025,2026"
    )
    parser.add_argument("--min-train", type=int, default=MIN_RECORD + 1, help="bars a training block must hold")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per request")
    parser.add_argument("--out", default=DEFAULT_OUT, help="where the table is written")
    parser.add_argument(
        "--candles",
        default="",
        help=(
            "read <pair>.csv from this directory instead of the venue, so a run can be "
            "reproduced bar for bar; the file is the window, so --days does not apply, and "
            "the directory's hash is written beside the table"
        ),
    )
    parser.add_argument(
        "--save-candles",
        default="",
        help="write the bars this run read into this directory, for --candles to replay",
    )
    parser.add_argument("--delimiter", default=",", help="the CSV separator")
    parser.add_argument("--quiet", action="store_true", help="write the files, print nothing")
    add_admission_arguments(parser)
    add_exit_arguments(parser)
    return parser


def _series(
    symbol: str,
    *,
    period: str,
    days: float,
    timeout: float,
    endpoint: str,
    candles: Optional[Path] = None,
) -> CandleSeries:
    """One pair's candles: from a frozen file when one is named, from the venue otherwise.

    ``candles`` is what makes a run reproducible. The venue serves a moving window — one new
    bar closes and every fold boundary shifts — so two runs of the same command are only the
    same experiment if both read the same bars. Naming a directory freezes that choice, and
    the directory's hash travels with the table (:func:`_candles_fingerprint`).
    """
    if candles is not None:
        return load_csv(Path(candles) / f"{symbol}.csv", symbol=symbol, period=str(period))
    return fetch_klines(
        symbol,
        period=str(period),
        days=float(days),
        timeout=float(timeout),
        endpoint=endpoint,
    )


def _freeze(series: CandleSeries, directory: Path) -> Path:
    """Write one series where ``--candles`` will look for it, and say where that was."""
    return write_csv(series, Path(directory) / f"{series.symbol}.csv")


def _candles_fingerprint(paths: Sequence[Path]) -> str:
    """A short hash of the frozen files, so two machines can prove they read the same bars.

    The names go into the hash with the bytes: two directories that hold the same candles
    under different names are not the same input, because the pair list is part of the run.
    """
    digest = hashlib.sha256()
    for path in sorted(Path(path).resolve() for path in paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"absent")
        digest.update(b"\0")
    return digest.hexdigest()


def _write_csv(path: Path, readings: Sequence[Reading], summaries: Sequence[Summary]) -> Path:
    """Two blocks in one file: the per-fold readings, then the pooled verdict per ablation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(READING_HEADERS))
        for reading in readings:
            writer.writerow(
                [
                    reading.fold,
                    reading.ablation,
                    reading.train_samples,
                    reading.trades,
                    f"{reading.roi:.8f}",
                    f"{reading.buy_hold:.8f}",
                    f"{reading.edge:.8f}",
                    f"{reading.avg_bps:.4f}",
                    f"{reading.win_rate:.6f}",
                    "inf" if math.isinf(reading.profit_factor) else f"{reading.profit_factor:.6f}",
                    f"{reading.sharpe:.4f}",
                    f"{reading.t_stat:.4f}",
                ]
            )
        writer.writerow([])
        writer.writerow(
            [
                "ablation",
                "folds",
                "trades",
                "bps/trade",
                "t",
                "pf",
                "sharpe",
                "roi",
                "min_edge_bps",
                "gate",
                "failures",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.ablation,
                    row.folds,
                    row.trades,
                    f"{row.edge_bps:.4f}",
                    f"{row.t_stat:.4f}",
                    "inf" if math.isinf(row.profit_factor) else f"{row.profit_factor:.6f}",
                    f"{row.sharpe:.4f}",
                    f"{row.roi:.8f}",
                    "inf" if math.isinf(row.min_edge_bps) else f"{row.min_edge_bps:.4f}",
                    row.verdict,
                    "; ".join(row.failures),
                ]
            )
    return path



def main(argv: Optional[Sequence[str]] = None) -> int:
    """Read a universe, walk it forward pair by pair, print the pooled verdict and mean it.

    Exit 0 when at least one pair produced at least one measured fold — *whether or not the
    gate passed*, because "no alpha" is an answer and a tool that exits non-zero for it would
    be a tool that only reports successes. 2 is a caller's mistake, 3 a feed that did not
    answer, 4 a record too young for a single fold.
    """
    args = _parser().parse_args(argv)
    exits = exits_from_args(args)
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
            min_depth=float(args.min_depth),
            min_points=int(args.min_train),
            admission=admission_from_args(args),
            **exits,
        )
    except ValueError as exc:
        print(f"walkforward: {exc}", file=sys.stderr)
        return EXIT_USAGE

    symbols: List[str] = []
    if str(args.symbols).strip():
        symbols = [name.strip().lower() for name in str(args.symbols).split(",") if name.strip()]
    elif str(args.candles).strip():
        # A frozen directory *is* a universe: the pairs in it are the pairs that were read,
        # and asking the venue for a ranking would make the run depend on today's turnover.
        symbols = sorted(path.stem.lower() for path in Path(args.candles).glob("*.csv"))
        if not symbols:
            print(
                f"walkforward: {args.candles} holds no <pair>.csv; pass --symbols or --save-candles",
                file=sys.stderr,
            )
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
            print(f"walkforward: {exc}", file=sys.stderr)
            return EXIT_FEED
        symbols = [pair.symbol for pair in listed]
    if not symbols:
        print("walkforward: the universe came back empty; pass --symbols", file=sys.stderr)
        return EXIT_USAGE

    years: Optional[List[int]] = None
    if str(args.folds).strip():
        try:
            years = [int(part) for part in str(args.folds).split(",") if part.strip()]
        except ValueError:
            print("walkforward: --folds takes years, e.g. 2024,2025,2026", file=sys.stderr)
            return EXIT_USAGE

    if not args.quiet and exits:
        print(
            f"── walkforward · {len(symbols)} pairs · {args.period} · {describe_exits(exits)}"
        )
    endpoint = HTX_ENDPOINTS[1 if str(args.exchange) == "huobi" else 0]
    frozen = Path(args.candles) if str(args.candles).strip() else None
    saving = Path(args.save_candles) if str(args.save_candles).strip() else None
    frozen_paths: List[Path] = []
    benchmark: Optional[CandleSeries] = None
    if str(args.benchmark).strip():
        try:
            benchmark = _series(
                str(args.benchmark).strip().lower(),
                period=str(args.period),
                days=float(args.days),
                timeout=float(args.timeout),
                endpoint=endpoint,
                candles=frozen,
            )
        except HtxError as exc:
            # The regime filter is an ablation, not the thesis: a dead benchmark narrows the
            # table by one row and must not stop the other three from being measured.
            print(f"walkforward: no benchmark ({exc}); the regime row will be noted", file=sys.stderr)
        else:
            if saving is not None:
                _freeze(benchmark, saving)
            if frozen is not None and (frozen / f"{benchmark.symbol}.csv") not in frozen_paths:
                frozen_paths.append(frozen / f"{benchmark.symbol}.csv")

    if not args.quiet:
        print(
            f"── walk-forward · {len(symbols)} pairs · {args.period} · {float(args.days):g} days asked · "
            f"benchmark {args.benchmark or 'none'} · gate {GATE['edge_bps']:.0f} bps / "
            f"t {GATE['t_stat']:.0f} / pf {GATE['profit_factor']:.1f} / sharpe {GATE['sharpe']:.0f}"
        )

    all_readings: List[Reading] = []
    measured = 0
    for symbol in symbols:
        try:
            series = _series(
                symbol,
                period=str(args.period),
                days=float(args.days),
                timeout=float(args.timeout),
                endpoint=endpoint,
                candles=frozen,
            )
        except HtxError as exc:
            print(f"   ! {symbol:<12} {exc}", file=sys.stderr)
            continue
        if saving is not None:
            _freeze(series, saving)
        if frozen is not None and (frozen / f"{series.symbol}.csv") not in frozen_paths:
            frozen_paths.append(frozen / f"{series.symbol}.csv")
        try:
            folds = expanding_folds(
                series.rows,
                years=years,
                splits=int(args.splits),
                min_train=int(args.min_train),
            )
            readings = walk_forward(
                series,
                settings=settings,
                ablations=pinned_ablations(exits),
                benchmark=benchmark,
                folds=folds,
                horizon=int(args.horizon),
                cost=float(args.cost),
                threshold=float(args.threshold),
                l2=float(args.l2),
                regime_window=int(args.regime_window),
            )
        except (ValueError, HtxError) as exc:
            print(f"   ! {symbol:<12} {exc}", file=sys.stderr)
            continue
        if not readings:
            if not args.quiet:
                print(f"   ! {symbol:<12} no fold: {len(series.rows)} bars, "
                      f"{int(args.min_train)} needed before the first test year")
            continue
        measured += 1
        all_readings.extend(readings)
        if not args.quiet:
            print()
            print(f"   {symbol} · {len(series.rows)} bars · folds "
                  f"{', '.join(fold.name for fold in folds)}")
            print(format_readings(readings))

    if not measured:
        print("walkforward: not one pair could be measured — the lines above say why", file=sys.stderr)
        return EXIT_SHORT

    summaries = summary(all_readings)
    written = _write_csv(Path(args.out) / "walkforward.csv", all_readings, summaries)
    # The gate is written down beside the numbers it judges, with the hash that proves it was
    # the one declared before the run: a table that prints a verdict without the criteria it
    # was read against is a table nobody can audit later, and this repository has one of those.
    criteria_written = declared().write(
        Path(args.out) / "criteria.json",
        run={
            "readings": len(all_readings),
            "pairs": measured,
            "period": str(args.period),
            "days": float(args.days),
            "exits": dict(exits),
            "candles": (
                {
                    "dir": str(frozen),
                    "sha256": _candles_fingerprint(frozen_paths),
                    "files": sorted(path.name for path in frozen_paths),
                }
                if frozen is not None
                else {"dir": "", "sha256": "", "files": []}
            ),
            "summaries": [row.as_dict() for row in summaries],
        },
    )
    if not args.quiet:
        print()
        print("── pooled over every pair and fold")
        print(format_summary(summaries))
        print(f"   wrote {written}")
        print(f"   wrote {criteria_written} · criteria {declared().fingerprint()[:12]}")
    best = gate_of(all_readings)
    if not args.quiet:
        if best.verdict == "pass":
            print(
                f"   {best.ablation}: the gate passed — {best.edge_bps:+.1f} bps a "
                f"trade at t {best.t_stat:.2f}"
            )
        elif best.verdict == "unmeasured":
            print(f"   {best.ablation}: unmeasured — {best.resolution.describe()}")
        else:
            print(f"   {best.ablation}: no alpha — {'; '.join(best.failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

