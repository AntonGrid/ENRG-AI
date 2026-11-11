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

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.backtest.data import CandleSeries, declines
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
    #: Per-side fee and per-fill slippage, as fractions.
    fee: float = FEE_RATE
    slippage: float = FILL_SLIPPAGE
    #: Bars between a decision and its fill — the network delay, in the only clock here.
    latency_bars: int = LATENCY_BARS
    #: The confidence floor a ``PANIC`` must reach to be bought. ``0`` buys every one.
    confidence: float = 0.0
    #: Which verdicts may open a position: ``panic``, ``organic`` or ``both``.
    entries: str = "panic"
    #: Bars the decline is measured over: ``1`` is the per-bar rate.
    window: int = 1
    #: Bars of history the detector needs before it will speak. The core's floor by default.
    min_points: int = MIN_RECORD
    #: The resting-turnover floor for the ``THIN`` arm, in USDT. ``0`` disables it.
    min_depth: Optional[float] = 0.0
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
            "fee": round(float(self.fee), 6),
            "slippage": round(float(self.slippage), 6),
            "latency_bars": int(self.latency_bars),
            "confidence": round(float(self.confidence), 4),
            "entries": str(self.entries),
            "window": int(self.window),
            "min_points": int(self.min_points),
            "min_depth": None if self.min_depth is None else round(float(self.min_depth), 2),
            "cooldown_bars": int(self.cooldown_bars),
            "log_decisions": bool(self.log_decisions),
        }


@dataclass
class Trade:
    """One round trip: what was bought, on whose verdict, and what it paid.

    The entry side is the *decision* (the verdict and its confidence) and the fill; the exit
    side is the fill and the reason it happened — ``stop``, a conviction, or the panic
    passing. A trade still open on the last bar has no exit fields, and the reporter counts
    it as unrealized rather than pretending it was sold.
    """

    entry_bar: int
    entry_at: float
    entry_price: float
    entry_usd: float
    units: float
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

    @property
    def open(self) -> bool:
        return self.exit_bar < 0

    @property
    def bars(self) -> int:
        """Bars held: the difference of the two clocks, and ``0`` while still open."""
        return 0 if self.open else int(self.exit_bar) - int(self.entry_bar)

    @property
    def pnl(self) -> float:
        """Profit in USDT, after both fees — the venue's own numbers, subtracted."""
        if self.open:
            return 0.0
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
    ) -> None:
        self.series = series
        self.settings = settings or Settings()
        self.detector = detector or self.settings.detector()

    # -- the rules, as one line each ---------------------------------------- #

    def wants_entry(self, signal: PanicSignal) -> bool:
        """Whether this verdict may open a position, given the mode and the confidence floor.

        ``PANIC`` is the hunt and always allowed above the floor. ``CALM``/``GROWTH`` are
        allowed only in the ``organic``/``both`` modes, and only when the record arm passed
        (``HONEST``) — which is what the brief called "ORGANIC": the hunter has no verdict of
        that name, its word for the same finding is the assessment's, and using the core's
        word instead of inventing a second one is the whole point of not touching it.
        """
        if signal.verdict == PANIC:
            return float(signal.confidence) >= float(self.settings.confidence)
        if str(self.settings.entries) in ("organic", "both") and signal.verdict in (
            TAKE_PROFIT_VERDICTS
        ):
            return signal.resting_verdict == HONEST
        return False

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
        yields = declines(rows, window=settings.window)
        volumes = [float(row.volume) for row in rows]
        symbol = self.series.symbol or "instrument"
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
                if fill.side == "buy":
                    decision = entry_decision or {}
                    current = Trade(
                        entry_bar=fill.bar,
                        entry_at=fill.at,
                        entry_price=fill.price,
                        entry_usd=fill.usd,
                        units=fill.units,
                        entry_verdict=str(decision.get("verdict", "")),
                        entry_confidence=float(decision.get("confidence", 0.0)),
                        entry_reason=str(decision.get("reason", "")),
                        fees=fill.fee,
                    )
                    trades.append(current)
                    entry_decision = None
                elif current is not None:
                    current.exit_bar = int(fill.bar)
                    current.exit_at = float(fill.at)
                    current.exit_price = float(fill.price)
                    current.exit_usd = float(fill.usd)
                    current.fees += fill.fee
                    if fill.reason == "stop":
                        # The venue's own order: no verdict is attached to it, because the
                        # stop is not a reading of the market — it is a rule that a position
                        # is not held past a loss.
                        current.exit_reason = "stop"
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
            decision = {
                "bar": int(t),
                "at": float(rows[t].at),
                "price": round(float(rows[t].close), 8),
                "verdict": signal.verdict,
                "confidence": round(float(signal.confidence), 4),
                "resting_verdict": signal.resting_verdict,
                "reason": signal.reason,
            }
            if settings.log_decisions:
                decisions.append(decision)
            # 3. the rules, on what the book actually holds.
            if venue.busy:
                pass
            elif venue.holding and current is not None:
                reason = self.exit_reason(signal, current.entry_verdict)
                if reason is not None and venue.withdraw(
                    symbol, float(venue.units), reason=reason
                ):
                    exit_decision = dict(decision, reason=reason)
            elif not venue.holding:
                if (
                    int(t) > int(quiet_until)
                    and self.wants_entry(signal)
                    and venue.execute_stake(
                        symbol,
                        float(venue.cash) * float(settings.share),
                        stop_pct=float(settings.stop),
                        reason="open",
                    )
                ):
                    entry_decision = dict(decision)
            # 4. the book, marked at this bar's close.
            equity.append(
                {
                    "bar": int(t),
                    "at": float(rows[t].at),
                    "price": round(float(rows[t].close), 8),
                    "cash": round(float(venue.cash), 6),
                    "units": round(float(venue.units), 10),
                    "equity": round(float(venue.equity), 6),
                    "exposure": bool(venue.units > 0.0),
                    "verdict": signal.verdict,
                }
            )
        judged = sum(
            count
            for verdict, count in verdicts.items()
            if verdict and verdict != "UNMEASURED"
        )
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
        )


def run_backtest(series: CandleSeries, **settings: Any) -> BacktestResult:
    """``Backtester(series, Settings(**settings)).run()`` — the one-liner a notebook uses."""
    return Backtester(series, Settings(**settings)).run()
