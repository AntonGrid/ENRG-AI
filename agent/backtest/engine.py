"""The engine: one verdict per bar, and the four rules that make a book out of them.

This is the file that turns history into a question with an answer. It does four things and
nothing else: it asks :class:`agent.physics.volatility_hunter.PanicDetector` what each bar
of the prefix says, buys when the answer is a panic it trusts, sells when the answer changes
its mind, and marks the book at every close so that a drawdown is a measurement rather than
an opinion.

The three verdicts, in the live loop's own vocabulary
=====================================================

The live cycle already has this table (:data:`agent.trade.loop.EXIT_VERDICTS` and its
siblings), and a backtest that used a different one would be a backtest of a different
strategy. It is repeated here rather than imported, for the same reason the book repeats the
word ``PANIC``: ``agent.trade.loop`` reaches for a venue and a key, and this package must be
importable without either. A test holds the two tables together, so the repetition cannot
drift.

* **Enter** on ``PANIC`` — both arms of the hunter, above the confidence floor.
* **Enter** on ``CALM``/``GROWTH`` only when the operator asks for it (``--entries
  organic``): that is the carry trade of the live loop — an honest record bought at rest —
  and it is off by default because on a chart it would be "buy whenever the price is quiet",
  which is a different hypothesis wearing the same word.
* **Leave** on ``DISHONEST``/``FORMULA``/``THIN`` — the conviction set: the record was
  never what it claimed, so being in it was a mistake and the exit is the correction.
* **Leave** on ``CALM``/``GROWTH`` — the panic passed (or the rate normalized). This is the
  take-profit of the live loop, and it is where most of this strategy's money is made.
* **Stay** on ``UNMEASURED``. Blindness is not a verdict: a bar the detector cannot judge
  is a bar on which nothing should be done, in either direction.

The stop-loss sits beside all of it, at ``-2 %`` by default, and it is the *venue's* order
(:class:`agent.backtest.simulated_provider.SimulatedProvider` executes it inside the bar
that crosses it) rather than the engine's decision. That split is deliberate: a stop that a
decision loop has to notice is a stop that fires late, and this package should not be able
to give itself that advantage.

No lookahead, and the test that proves it
=========================================

Every call to the detector is handed ``yields[:t + 1]`` — the prefix that ends on the bar
being decided, and nothing after it. That is the whole honesty of a backtest, and it is one
line of code that a refactor can break silently, so ``tests/test_backtest_engine.py`` wraps
the detector, records the length of every prefix it is given, and asserts that the sequence
is ``1, 2, 3, …`` with no gap and no jump.

The position size is the cash on hand
=====================================

``--capital 1000`` with ``--share 1.0`` means the whole account goes into one position, and
the live loop's answer to "how much" is a fraction of a wallet — not the same number, but the
same idea: one position at a time, sized against what is actually there. A refusal to buy a
second position is the venue's, and the engine never tops up, because a strategy whose edge
depends on averaging into a falling market is a different strategy and should be tested as
one.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agent.backtest.data import CandleSeries, climbs, declines
from agent.backtest.admission import Admission, admit, masked, report as admission_report
from agent.backtest.features import enrich, series_features
from agent.backtest.simulated_provider import (
    FEE_RATE,
    FILL_SLIPPAGE,
    LATENCY_BARS,
    SimulatedProvider,
)
from agent.physics.portfolio import MIN_RECORD
from agent.physics.risk_assessment import HONEST
from agent.physics.volatility_hunter import (
    CALM,
    DISHONEST,
    FORMULA,
    GROWTH,
    PANIC,
    THIN,
    PanicDetector,
    PanicSignal,
)

__all__ = [
    "ENTRY_VERDICTS",
    "EXIT_VERDICTS",
    "TAKE_PROFIT_VERDICTS",
    "BacktestResult",
    "Backtester",
    "Settings",
    "Trade",
    "run_backtest",
]

#: Verdicts after which a position must not be held: the record was never the thing the
#: trade was bought on, so the exit is a correction and not a profit-taking.
EXIT_VERDICTS = frozenset({DISHONEST, FORMULA, THIN})

#: Verdicts that mean the dislocation is over — the rate is back at its own rest (``CALM``)
#: or it rose with the depth rather than against it (``GROWTH``). The take-profit set.
TAKE_PROFIT_VERDICTS = frozenset({CALM, GROWTH})

#: Every verdict that may open a position under some setting of ``Settings.entries``. A
#: name here is not a permission: :meth:`Backtester.wants_entry` decides, and ``UNMEASURED``
#: is deliberately absent from all three sets.
ENTRY_VERDICTS = frozenset({PANIC}) | TAKE_PROFIT_VERDICTS

#: The modes ``Settings.entries`` accepts, and what each of them means.
ENTRY_MODES = ("panic", "organic", "both")

#: The two ways a position can be held. Not a mode of the *rules*: the same rules read a
#: series for a long book and its mirror for a short one, so a difference between the two
#: books is a difference in the market rather than in the code that measured it.
SIDES = ("long", "short")

#: How ``Settings.stop`` is turned into a price: a fixed fraction, or a multiple of the ATR.
STOP_MODES = ("fixed", "atr")


@dataclass(frozen=True)
class Settings:
    """Everything an operator may move, in one value, validated once.

    Every field is a decision and not a constant, for the reason the hunter's thresholds are
    fields: a reader must be able to move one number, re-run, and see exactly what it
    changed. ``__post_init__`` refuses the ones that would make a run meaningless — a
    negative stop, a share above one, a window of zero bars — with a sentence, because a
    backtest that silently reinterpreted a typo is worse than one that refuses to start.
    """

    #: The starting book, in USDT.
    capital: float = 1000.0
    #: The share of the cash on hand that one entry may spend, ``0 … 1``.
    share: float = 1.0
    #: The stop-loss distance below the fill, as a fraction. ``0`` disables the stop.
    stop: float = 0.02
    #: How that distance is measured: ``fixed`` (the fraction above) or ``atr`` (a multiple
    #: of the bar's own true range, never below ``stop``). The sweep's 374 stops on the entry
    #: bar are the argument for the second mode: a 2 % stop on a pair whose daily ATR is 5 %
    #: is not a risk limit, it is a coin flip with a fee attached.
    stop_mode: str = "fixed"
    #: Bars behind the average true range that ``atr`` mode multiplies.
    atr_window: int = 14
    #: The multiple of the ATR that becomes the stop distance.
    atr_multiple: float = 2.0
    #: The fraction of equity risked on one trade. ``0`` disables it and leaves sizing to
    #: ``share``; otherwise the stake is ``cash × risk_per_trade / stop_distance``, capped by
    #: ``share``. This is the difference between "all the cash" and "two per cent of the book
    #: at the stop", and it is what makes a 5 % ATR pair and a 2 % ATR pair comparable bets.
    risk_per_trade: float = 0.0
    #: Take-profit as a multiple of the stop distance (``R``). ``0`` disables it.
    take_profit_r: float = 0.0
    #: Trailing stop as a multiple of ``R``. ``0`` disables it.
    trail_r: float = 0.0
    #: Bars after which the position is closed at the next open. ``0`` disables it.
    time_stop_bars: int = 0
    #: Enter only on bars whose regime mask is true. Off unless a mask is supplied.
    require_risk_on: bool = False
    #: The rate at which a random bar is bought regardless of verdict — the null test of
    #: :mod:`agent.backtest.null`, and off by default because a random entry is not a strategy.
    random_entry_rate: float = 0.0
    random_seed: int = 0
    #: Per-side fee and per-fill slippage, as fractions.
    fee: float = FEE_RATE
    slippage: float = FILL_SLIPPAGE
    #: Bars between a decision and its fill — the network delay, in the only clock here.
    latency_bars: int = LATENCY_BARS
    #: The confidence floor a ``PANIC`` must reach to be bought. ``0`` buys every one.
    confidence: float = 0.0
    #: Which verdicts may open a position: ``panic``, ``organic`` or ``both``.
    entries: str = "panic"
    #: Which side the book trades: ``long`` buys the decline, ``short`` sells the rise.
    #:
    #: The detector is a *downside* instrument: it asks how much a bar fell, and its whole
    #: vocabulary — ``PANIC``, ``convicted`` — describes a dislocation closing upwards. A short
    #: book therefore hands it :func:`agent.backtest.data.climbs` instead of ``declines``: the
    #: same detector, reading the same question about the other direction. Nothing in the
    #: detector changes, and the venue's ledger holds a negative position, so the exits, the
    #: fees and the stop arithmetic are the code that was already tested.
    direction: str = "long"
    #: Bars the decline is measured over: ``1`` is the per-bar rate.
    window: int = 1
    #: Bars of history the detector needs before it will speak. The core's floor by default.
    min_points: int = MIN_RECORD
    #: The resting-turnover floor for the ``THIN`` arm, in USDT. ``0`` disables it.
    min_depth: Optional[float] = 0.0
    #: Which bars may reach the detector at all: the input filter of
    #: :mod:`agent.backtest.admission`. ``None`` feeds every bar as published.
    admission: Optional[Admission] = None
    #: Bars of silence between an exit and the next entry decision. ``0`` is the strategy as
    #: written — a stop can be followed by a re-entry on the very next bar — and 30 is the
    #: hunter's own :data:`agent.physics.volatility_hunter.PANIC_COOLDOWN`, which makes one
    #: episode one bet instead of one bet per bar.
    cooldown_bars: int = 0
    #: Keep one row per bar in the report. Off by default: the trades are the report.
    log_decisions: bool = False

    def __post_init__(self) -> None:
        if not (float(self.capital) > 0.0):
            raise ValueError(f"capital must be positive, not {self.capital!r}")
        if not 0.0 <= float(self.share) <= 1.0:
            raise ValueError(f"share must be between 0 and 1, not {self.share!r}")
        if not 0.0 <= float(self.stop) < 1.0:
            raise ValueError(f"stop must be in [0, 1), not {self.stop!r}")
        if not 0.0 <= float(self.fee) < 0.5:
            raise ValueError(f"fee must be in [0, 0.5), not {self.fee!r}")
        if not 0.0 <= float(self.slippage) < 0.5:
            raise ValueError(f"slippage must be in [0, 0.5), not {self.slippage!r}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], not {self.confidence!r}")
        if int(self.latency_bars) < 0:
            raise ValueError(f"latency_bars must not be negative, not {self.latency_bars!r}")
        if int(self.window) < 1:
            raise ValueError(f"window must be at least one bar, not {self.window!r}")
        if int(self.min_points) < 2:
            raise ValueError(f"min_points must be at least 2, not {self.min_points!r}")
        if int(self.cooldown_bars) < 0:
            raise ValueError(f"cooldown_bars must not be negative, not {self.cooldown_bars!r}")
        if str(self.entries) not in ENTRY_MODES:
            raise ValueError(f"entries must be one of {ENTRY_MODES}, not {self.entries!r}")
        if str(self.direction) not in SIDES:
            raise ValueError(f"direction must be one of {SIDES}, not {self.direction!r}")
        if str(self.stop_mode) not in STOP_MODES:
            raise ValueError(f"stop_mode must be one of {STOP_MODES}, not {self.stop_mode!r}")
        if int(self.atr_window) < 1:
            raise ValueError(f"atr_window must be at least one bar, not {self.atr_window!r}")
        if not float(self.atr_multiple) > 0.0:
            raise ValueError(f"atr_multiple must be positive, not {self.atr_multiple!r}")
        if not 0.0 <= float(self.risk_per_trade) <= 1.0:
            raise ValueError(
                f"risk_per_trade must be between 0 and 1, not {self.risk_per_trade!r}"
            )
        if float(self.take_profit_r) < 0.0:
            raise ValueError(f"take_profit_r must not be negative, not {self.take_profit_r!r}")
        if float(self.trail_r) < 0.0:
            raise ValueError(f"trail_r must not be negative, not {self.trail_r!r}")
        if int(self.time_stop_bars) < 0:
            raise ValueError(
                f"time_stop_bars must not be negative, not {self.time_stop_bars!r}"
            )
        if not 0.0 <= float(self.random_entry_rate) <= 1.0:
            raise ValueError(
                f"random_entry_rate must be in [0, 1], not {self.random_entry_rate!r}"
            )

    def detector(self) -> PanicDetector:
        """The hunter's two arms, with this run's record floor and depth floor.

        Built here and not stored, because :class:`PanicDetector` is a value: two runs with
        the same settings must be able to compare their detectors, and a shared mutable one
        would make the second run depend on the first.
        """
        return PanicDetector(min_points=int(self.min_points), min_depth=self.min_depth)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "capital": round(float(self.capital), 4),
            "share": round(float(self.share), 4),
            "stop": round(float(self.stop), 6),
            "stop_mode": str(self.stop_mode),
            "atr_window": int(self.atr_window),
            "atr_multiple": round(float(self.atr_multiple), 4),
            "risk_per_trade": round(float(self.risk_per_trade), 6),
            "take_profit_r": round(float(self.take_profit_r), 4),
            "trail_r": round(float(self.trail_r), 4),
            "time_stop_bars": int(self.time_stop_bars),
            "require_risk_on": bool(self.require_risk_on),
            "random_entry_rate": round(float(self.random_entry_rate), 6),
            "fee": round(float(self.fee), 6),
            "slippage": round(float(self.slippage), 6),
            "latency_bars": int(self.latency_bars),
            "confidence": round(float(self.confidence), 4),
            "entries": str(self.entries),
            "direction": str(self.direction),
            "window": int(self.window),
            "min_points": int(self.min_points),
            "min_depth": None if self.min_depth is None else round(float(self.min_depth), 2),
            "admission": None if self.admission is None else self.admission.rules(),
            "cooldown_bars": int(self.cooldown_bars),
            "log_decisions": bool(self.log_decisions),
        }


@dataclass
class Trade:
    """One round trip: what was opened, on whose verdict, and what it paid.

    The entry side is the *decision* (the verdict and its confidence) and the fill; the exit
    side is the fill and the reason it happened — ``stop``, a conviction, or the panic
    passing. A trade still open on the last bar has no exit fields, and the reporter counts
    it as unrealized rather than pretending it was sold.

    :attr:`side` is which way the position was held, and :attr:`units` is always the
    *magnitude*: the direction lives in one place so that a short's profit is the entry's
    proceeds minus the cover's cost, not a second sign hidden in two other fields.
    """

    entry_bar: int
    entry_at: float
    entry_price: float
    entry_usd: float
    units: float
    side: str = "long"
    entry_verdict: str = ""
    entry_confidence: float = 0.0
    entry_reason: str = ""
    exit_bar: int = -1
    exit_at: float = 0.0
    exit_price: float = 0.0
    exit_usd: float = 0.0
    exit_reason: str = ""
    exit_verdict: str = ""
    fees: float = 0.0
    stop_pct: float = 0.0
    policy_score: float = 0.0
    regime: float = 1.0
    mfe: float = 0.0
    mae: float = 0.0

    @property
    def open(self) -> bool:
        return self.exit_bar < 0

    @property
    def bars(self) -> int:
        """Bars held: the difference of the two clocks, and ``0`` while still open."""
        return 0 if self.open else int(self.exit_bar) - int(self.entry_bar)

    @property
    def pnl(self) -> float:
        """Profit in USDT, after both fees — the venue's own numbers, subtracted.

        The two sides of the ledger run in opposite directions: a long pays ``entry_usd`` and
        receives ``exit_usd``, a short receives the sale's proceeds and pays the cover. One
        subtraction with a sign taken from :attr:`side`, so that a profitable long and a
        profitable short are both positive numbers here and nowhere else has to know which.
        """
        if self.open:
            return 0.0
        if str(self.side) == "short":
            return float(self.entry_usd) - float(self.exit_usd)
        return float(self.exit_usd) - float(self.entry_usd)

    @property
    def pnl_pct(self) -> float:
        """Profit as a fraction of what was paid, after fees."""
        if self.open or float(self.entry_usd) <= 0.0:
            return 0.0
        return float(self.pnl) / float(self.entry_usd)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "entry_bar": int(self.entry_bar),
            "entry_at": float(self.entry_at),
            "side": str(self.side),
            "entry_price": round(float(self.entry_price), 8),
            "entry_usd": round(float(self.entry_usd), 4),
            "units": round(float(self.units), 10),
            "entry_verdict": self.entry_verdict,
            "entry_confidence": round(float(self.entry_confidence), 4),
            "entry_reason": self.entry_reason,
            "exit_bar": int(self.exit_bar),
            "exit_at": float(self.exit_at),
            "exit_price": round(float(self.exit_price), 8),
            "exit_usd": round(float(self.exit_usd), 4),
            "exit_reason": self.exit_reason,
            "exit_verdict": self.exit_verdict,
            "bars": int(self.bars),
            "pnl": round(float(self.pnl), 6),
            "pnl_pct": round(float(self.pnl_pct), 8),
            "fees": round(float(self.fees), 6),
            "stop_pct": round(float(self.stop_pct), 8),
            "policy_score": round(float(self.policy_score), 6),
            "regime": round(float(self.regime), 4),
            "mfe": round(float(self.mfe), 8),
            "mae": round(float(self.mae), 8),
            "open": bool(self.open),
        }


@dataclass
class BacktestResult:
    """A run, as facts: the trades, the equity curve and the verdicts behind both.

    It holds no metrics. ROI, drawdown, win rate and holding time are *readings* of these
    facts, they belong to :mod:`agent.backtest.reporter`, and keeping them out of here is
    what lets a test assert on a fill without a metric standing between it and the ledger.
    """

    settings: Settings
    series: CandleSeries
    venue: Dict[str, Any]
    trades: List[Trade] = field(default_factory=list)
    equity: List[Dict[str, Any]] = field(default_factory=list)
    verdicts: Dict[str, int] = field(default_factory=dict)
    refusals: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    #: What the input filter withheld from the detector, per reason — empty when no filter.
    admissions: Dict[str, Any] = field(default_factory=dict)

    @property
    def closed(self) -> List[Trade]:
        """The trades that were actually sold — the only ones a win rate may count."""
        return [trade for trade in self.trades if not trade.open]

    @property
    def open_trades(self) -> List[Trade]:
        return [trade for trade in self.trades if trade.open]

    @property
    def final_equity(self) -> float:
        return float(self.equity[-1]["equity"]) if self.equity else float(self.settings.capital)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "settings": self.settings.as_dict(),
            "series": self.series.as_dict(),
            "venue": dict(self.venue),
            "bars": len(self.series.rows),
            "trades": [trade.as_dict() for trade in self.trades],
            "equity": list(self.equity),
            "verdicts": dict(self.verdicts),
            "refusals": list(self.refusals),
            "decisions": list(self.decisions),
            "warnings": list(self.warnings),
            "admissions": dict(self.admissions),
            "final_equity": round(float(self.final_equity), 6),
        }


class Backtester:
    """One series, one book, one clock — and the four rules, in the order they are read.

    The detector is injected rather than constructed inside :meth:`run` so that a test can
    hand it a wrapper that records what it was asked; it defaults to
    :meth:`Settings.detector`, which is the hunter with the core's own floors.
    """

    def __init__(
        self,
        series: CandleSeries,
        settings: Optional[Settings] = None,
        detector: Optional[PanicDetector] = None,
        policy: Optional[Any] = None,
        *,
        policy_threshold: Optional[float] = None,
        regime: Optional[Sequence[bool]] = None,
    ) -> None:
        self.series = series
        self.settings = settings or Settings()
        self.detector = detector or self.settings.detector()
        #: Anything with a ``score(features) -> float`` method: a
        #: :class:`agent.backtest.policy.LogisticPolicy`, or a stand-in a test writes in
        #: five lines. Duck-typed on purpose — the engine must not depend on numpy.
        self.policy = policy
        self.policy_threshold = (
            float(policy_threshold)
            if policy_threshold is not None
            else float(getattr(policy, "threshold", 0.5))
        )
        #: One flag per bar: ``True`` above the benchmark's trend. ``None`` means the caller
        #: did not supply a regime, and a filter with nothing behind it filters nothing.
        self.regime = list(regime) if regime is not None else None
        if self.regime is not None and len(self.regime) != len(series.rows):
            raise ValueError(
                f"the regime mask has {len(self.regime)} bars against the series' "
                f"{len(series.rows)}: a mask that does not line up cannot be used"
            )

    # -- the rules, as one line each ---------------------------------------- #

    def wants_entry(
        self, signal: PanicSignal, features: Optional[Mapping[str, float]] = None
    ) -> bool:
        """Whether this verdict may open a position, given the mode, the floor and policy.

        ``PANIC`` is the hunt and always allowed above the floor. ``CALM``/``GROWTH`` are
        allowed only in the ``organic``/``both`` modes, and only when the record arm passed
        (``HONEST``) — which is what the brief called "ORGANIC": the hunter has no verdict of
        that name, its word for the same finding is the assessment's, and using the core's
        word instead of inventing a second one is the whole point of not touching it.

        The policy is a *veto*, not a second opinion: the detector still decides what is
        interesting, and the model decides whether it is worth the fee. A model that could
        buy without a verdict would be a second strategy wearing this one's name.
        """
        if signal.verdict == PANIC:
            allowed = float(signal.confidence) >= float(self.settings.confidence)
        elif str(self.settings.entries) in ("organic", "both") and signal.verdict in (
            TAKE_PROFIT_VERDICTS
        ):
            allowed = signal.resting_verdict == HONEST
        else:
            allowed = False
        if not allowed:
            return False
        if self.policy is not None:
            return float(self.policy.score(features or {})) >= self.policy_threshold
        return True

    def exit_reason(self, signal: PanicSignal, entry_verdict: str) -> Optional[str]:
        """``"convicted"``, ``"take-profit"`` or ``None`` — the live loop's table, verbatim.

        The second leg needs the *entry* verdict: a dislocation that has closed is a sale
        only for a position that was bought on one. A position bought for carry (``CALM``)
        has no dislocation to have passed, and selling it on the next calm bar would be a
        churn that pays the venue and measures nothing.
        """
        if signal.verdict in EXIT_VERDICTS:
            return "convicted"
        if signal.verdict in TAKE_PROFIT_VERDICTS and str(entry_verdict).upper() == PANIC:
            return "take-profit"
        return None

    # -- the run ------------------------------------------------------------ #

    def run(self) -> BacktestResult:
        """Walk the series once, decide on every bar, and return the facts.

        The order inside a bar is the order a venue imposes: the fills that happened at this
        bar's open are absorbed first, then the verdict is measured on the prefix that ends
        here, and only then is a new order placed — which will settle at the *next* bar's
        open. A decision can therefore never be filled at a price it was allowed to see, and
        the equity row for the bar is marked after everything the bar did.
        """
        settings = self.settings
        rows = self.series.rows
        if len(rows) < 3:
            raise ValueError(
                f"{len(rows)} bars is not a backtest: at least three are needed to have a "
                "past, a decision and a fill"
            )
        # The series the detector reads: what the bar offered a *buyer* for a long book, and
        # what it offered a *seller* for a short one. The detector is not told which — it is
        # the same question about the same bar, and the mirror is the deliberate choice of
        # which side's question to ask.
        mirror = climbs if str(settings.direction) == "short" else declines
        yields = mirror(rows, window=settings.window)
        volumes = [float(row.volume) for row in rows]
        # The input filter, applied once and in front of the core: a bar the filter withheld
        # is handed over as ``nan`` — which the detector drops, exactly as it drops a warm-up
        # bar — and never as a zero, because a zero is a reading the market did not publish.
        admitted = admit(rows, settings.admission)
        if settings.admission is not None:
            yields = masked(yields, admitted)
            volumes = masked(volumes, admitted)
        symbol = self.series.symbol or "instrument"
        # The features are computed once for the whole series and then *read* bar by bar.
        # The alternative — recomputing the suffix on every bar — would be the same numbers
        # at a quadratic price, and a policy that could not run on a year of data would be a
        # policy nobody deploys.
        features_rows = None
        if self.policy is not None or str(settings.stop_mode) == "atr":
            features_rows = series_features(rows, atr_window=int(settings.atr_window))
        coin = random.Random(int(settings.random_seed))
        bars_since_panic = 0
        venue = SimulatedProvider(
            series=self.series,
            cash=float(settings.capital),
            slippage=float(settings.slippage),
            fee=float(settings.fee),
            latency_bars=int(settings.latency_bars),
            address=f"sim:{symbol}",
        )
        trades: List[Trade] = []
        equity: List[Dict[str, Any]] = []
        verdicts: Counter = Counter()
        decisions: List[Dict[str, Any]] = []
        warnings: List[str] = list(self.series.warnings)
        current: Optional[Trade] = None
        entry_decision: Optional[Dict[str, Any]] = None
        exit_decision: Optional[Dict[str, Any]] = None
        quiet_until = -1
        cursor = 0
        for t in range(len(rows)):
            if t:
                venue.to_bar(t)
            # 1. what the venue did on this bar: a settlement at the open, or a stop.
            for fill in venue.fills[cursor:]:
                cursor += 1
                if fill.side in ("buy", "short"):
                    decision = entry_decision or {}
                    current = Trade(
                        entry_bar=fill.bar,
                        entry_at=fill.at,
                        entry_price=fill.price,
                        entry_usd=fill.usd,
                        units=fill.units,
                        side="short" if fill.side == "short" else "long",
                        entry_verdict=str(decision.get("verdict", "")),
                        entry_confidence=float(decision.get("confidence", 0.0)),
                        entry_reason=str(decision.get("reason", "")),
                        fees=fill.fee,
                        stop_pct=float(decision.get("stop_pct", 0.0)),
                        policy_score=float(decision.get("policy_score", 0.0)),
                        regime=float(decision.get("regime", 0.5)),
                    )
                    trades.append(current)
                    entry_decision = None
                elif current is not None:
                    current.exit_bar = int(fill.bar)
                    current.exit_at = float(fill.at)
                    current.exit_price = float(fill.price)
                    current.exit_usd = float(fill.usd)
                    current.fees += fill.fee
                    current.mfe = float(venue.last_mfe)
                    current.mae = float(venue.last_mae)
                    if fill.reason == "stop":
                        # The venue's own order: no verdict is attached to it, because the
                        # stop is not a reading of the market — it is a rule that a position
                        # is not held past a loss.
                        current.exit_reason = "stop"
                        current.exit_verdict = ""
                    elif fill.reason == "target":
                        current.exit_reason = "target"
                        current.exit_verdict = ""
                    elif fill.reason == "time":
                        current.exit_reason = "time"
                        current.exit_verdict = ""
                    else:
                        decision = exit_decision or {}
                        current.exit_reason = str(decision.get("reason", fill.reason))
                        current.exit_verdict = str(decision.get("verdict", ""))
                    current = None
                    exit_decision = None
                    quiet_until = int(fill.bar) + int(settings.cooldown_bars)
            # 2. the verdict, on the prefix that ends on this bar and nothing after it.
            signal = self.detector.measure(
                yields[: t + 1],
                volumes[: t + 1],
                name=symbol,
            )
            verdicts[signal.verdict] += 1
            bars_since_panic = 0 if signal.verdict == PANIC else bars_since_panic + 1
            regime_here = (
                1.0
                if self.regime is not None and bool(self.regime[t])
                else (0.0 if self.regime is not None else 0.5)
            )
            case: Dict[str, float] = {}
            if features_rows is not None:
                case = enrich(
                    features_rows[t],
                    confidence=float(signal.confidence),
                    regime=regime_here,
                    bars_since_panic=bars_since_panic,
                )
            decision = {
                "bar": int(t),
                "at": float(rows[t].at),
                "price": round(float(rows[t].close), 8),
                "verdict": signal.verdict,
                "confidence": round(float(signal.confidence), 4),
                "resting_verdict": signal.resting_verdict,
                "reason": signal.reason,
                "regime": regime_here,
                "policy_score": (
                    round(float(self.policy.score(case)), 6) if self.policy is not None else 0.0
                ),
            }
            if settings.log_decisions:
                decisions.append(decision)
            # 3. the rules, on what the book actually holds.
            if venue.busy:
                pass
            elif venue.holding and current is not None:
                reason = self.exit_reason(signal, current.entry_verdict)
                if reason is not None and venue.withdraw(
                    symbol, abs(float(venue.units)), reason=reason
                ):
                    exit_decision = dict(decision, reason=reason)
            elif not venue.holding:
                wanted = self.wants_entry(signal, case) if case else self.wants_entry(signal)
                if float(settings.random_entry_rate) > 0.0:
                    # The null test: the coin replaces the verdict, and everything else —
                    # the stops, the fees, the exits — stays exactly as it is. A random
                    # entry measured against *this* book is the only honest answer to "how
                    # much of the result was the signal", because the alternative book is
                    # identical in every respect except the one under test.
                    wanted = coin.random() < float(settings.random_entry_rate)
                if wanted and settings.require_risk_on and self.regime is not None:
                    wanted = bool(self.regime[t])
                if wanted and admitted[t]:
                    # A bar the filter withheld is a bar the detector was not asked about;
                    # buying it would be acting on a reading nobody made. The position, if
                    # any, is still held and still exited on the verdicts of readable bars.
                    wanted = False
                if int(t) > int(quiet_until) and wanted:
                    stop_pct = float(settings.stop)
                    atr_here = (
                        float(features_rows[t]["atr_pct"]) if features_rows is not None else 0.0
                    )
                    if str(settings.stop_mode) == "atr" and atr_here > 0.0:
                        # The ATR is in per cent of the close; the stop is a fraction of it.
                        stop_pct = max(
                            stop_pct, float(settings.atr_multiple) * atr_here / 100.0
                        )
                    stake = float(venue.cash) * float(settings.share)
                    if float(settings.risk_per_trade) > 0.0 and stop_pct > 0.0:
                        stake = min(
                            stake,
                            float(venue.cash) * float(settings.risk_per_trade) / stop_pct,
                        )
                    if venue.execute_stake(
                        symbol,
                        stake,
                        stop_pct=stop_pct,
                        take_profit_pct=float(settings.take_profit_r) * stop_pct,
                        trail_pct=float(settings.trail_r) * stop_pct,
                        max_bars=int(settings.time_stop_bars),
                        reason="open",
                        direction=str(settings.direction),
                    ):
                        entry_decision = dict(decision, stop_pct=stop_pct)
            # 4. the book, marked at this bar's close.
            equity.append(
                {
                    "bar": int(t),
                    "at": float(rows[t].at),
                    "price": round(float(rows[t].close), 8),
                    "cash": round(float(venue.cash), 6),
                    "units": round(float(venue.units), 10),
                    "equity": round(float(venue.equity), 6),
                    "exposure": bool(venue.units != 0.0),
                    "verdict": signal.verdict,
                }
            )
        judged = sum(
            count
            for verdict, count in verdicts.items()
            if verdict and verdict != "UNMEASURED"
        )
        admissions = (
            admission_report(rows, settings.admission)
            if settings.admission is not None
            else {}
        )
        if admissions.get("sentence"):
            warnings.append(str(admissions["sentence"]))
        if judged == 0:
            warnings.append(
                f"the detector never spoke: {len(rows)} bars against a record floor of "
                f"{int(settings.min_points)} — the report below measures a strategy that was "
                "never given a decision to make"
            )
        if current is not None:
            warnings.append(
                f"one position was still open on the last bar ({current.units:.6f} units "
                f"bought at {current.entry_price:.6f}); the equity curve marks it at the "
                "last close and the win rate does not count it"
            )
        return BacktestResult(
            settings=settings,
            series=self.series,
            venue=venue.as_dict(),
            trades=trades,
            equity=equity,
            verdicts=dict(verdicts),
            refusals=[refusal.as_dict() for refusal in venue.refusals],
            decisions=decisions,
            warnings=warnings,
            admissions=admissions,
        )


def run_backtest(
    series: CandleSeries,
    policy: Optional[Any] = None,
    regime: Optional[Sequence[bool]] = None,
    **settings: Any,
) -> BacktestResult:
    """``Backtester(series, Settings(**settings), policy=…, regime=…).run()`` — the one-liner."""
    return Backtester(series, Settings(**settings), policy=policy, regime=regime).run()
