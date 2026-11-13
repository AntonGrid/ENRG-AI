"""Many pairs on one calendar — because a cross-section is a question about a *day*.

Everything else in this package reads one pair: the engine asks the detector what a bar says
about *that symbol*, the walk-forward pools one book's trades, and the sweep prints one row per
pair. That shape can answer "would this have paid" and it structurally cannot answer the other
question — *which of these hundred coins is the one to be long today* — because the answer to
that is a comparison, and a comparison needs one calendar with many columns.

This module is that calendar and nothing else. It holds each symbol's bars under a union of
timestamps, and it is deliberately strict about three things that decide whether a
cross-sectional backtest measures a market or an artefact:

**A missing bar is a hole and never a fill.** A symbol that was not yet listed, has been
delisted, or simply has no bar at a timestamp did not trade at that timestamp, and carrying its
last close forward would invent a price it never had. :meth:`Panel.ret` therefore refuses a
window that is not contiguous, and the caller decides what a hole means — for a rank it means
"not a candidate today", for a position it means "you cannot hold what the venue stopped
quoting", and both of those are true statements rather than a filled-in ``nan``.

**Survivorship is in the holes.** A universe assembled today is the set of pairs that *survived*
to today, and the ones that died are exactly the ones that would have hurt. Nothing in a candle
file can undo that, so this module does the next best thing: it keeps the hole visible
(:meth:`Panel.missing`, :meth:`Panel.as_dict` counts them) instead of letting a delisting look
like a long-running position that was quietly marked to a stale price.

**One interval, or it is not one market.** Two bars of different width in one panel would make
a cross-section compare a four-hour move with a daily one. :class:`Panel` refuses that with a
sentence naming both intervals rather than averaging them into something that is neither.

The panel is built from :class:`~agent.backtest.data.CandleSeries` — from the venue, from CSVs,
or from a hand-written list in a test — and it is written back as one CSV per symbol, so a run
over a hundred pairs can be frozen and replayed exactly as the single-pair runs already are.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agent.backtest.data import (
    DEFAULT_PERIOD,
    DEFAULT_TIMEOUT,
    HTX_ENDPOINT,
    PERIOD_SECONDS,
    Candle,
    CandleSeries,
    HtxError,
    fetch_klines,
    load_csv,
    write_csv,
)

__all__ = [
    "GAP_TOLERANCE",
    "Panel",
    "PanelError",
    "build_panel",
    "fetch_panel",
    "load_panel",
    "write_panel",
]

#: How far a symbol's bar may sit from where the calendar says it should be, as a fraction of
#: one interval, before the bar counts as a *hole* rather than a step. A venue that publishes
#: its bars a second late is still publishing them; a symbol that skipped a day is not.
GAP_TOLERANCE = 0.01


class PanelError(HtxError):
    """A set of series that cannot be read as one market, with the sentence that says why."""


@dataclass
class Panel:
    """Many symbols' bars under one union of timestamps, with every hole left visible.

    ``ats`` is the sorted union of every bar time in the panel, and it is the clock: a bar in
    the panel is a moment the market as a whole is read, even where some symbols are absent.
    ``bars[symbol][at]`` is the bar itself, so a caller who needs the open (the fill) gets it
    from the same object as the caller who needs the close (the mark).

    The per-symbol lists are kept beside the dicts because a *window* is a question about a
    symbol's own history — "did this bar's close come three of *its* bars after that one" —
    and answering it off the union calendar would call a symbol's own listing gap a market
    event.
    """

    period: str = DEFAULT_PERIOD
    step: float = 0.0
    symbols: List[str] = field(default_factory=list)
    ats: List[float] = field(default_factory=list)
    bars: Dict[str, Dict[float, Candle]] = field(default_factory=dict)
    ats_of: Dict[str, List[float]] = field(default_factory=dict)
    closes_of: Dict[str, List[float]] = field(default_factory=dict)
    index_of: Dict[str, Dict[float, int]] = field(default_factory=dict)
    positions: Dict[float, int] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # -- shape ------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.ats)

    @property
    def interval(self) -> float:
        """Seconds between bars — declared by the caller, else measured from the calendar.

        Never *assumed*: an undeclared width is read off the finest gap the panel contains, and
        a declared one is checked against it, so neither path can quietly invent a clock.
        """
        return float(self.step)

    def index(self, at: float) -> int:
        """The calendar position of a timestamp, or ``-1`` when it is not a bar time."""
        return int(self.positions.get(float(at), -1))

    def at(self, index: int) -> Optional[float]:
        """The timestamp at a calendar position, or ``None`` outside the panel."""
        if 0 <= int(index) < len(self.ats):
            return float(self.ats[int(index)])
        return None

    # -- one symbol -------------------------------------------------------- #

    def bar(self, symbol: str, at: float) -> Optional[Candle]:
        """The bar, or ``None`` — the hole is not filled with the last price."""
        return self.bars.get(str(symbol), {}).get(float(at))

    def close(self, symbol: str, at: float) -> Optional[float]:
        candle = self.bar(symbol, at)
        return None if candle is None else float(candle.close)

    def open(self, symbol: str, at: float) -> Optional[float]:
        candle = self.bar(symbol, at)
        return None if candle is None else float(candle.open)

    def missing(self, at: float) -> List[str]:
        """The symbols with no bar at this timestamp, in panel order."""
        return [symbol for symbol in self.symbols if float(at) not in self.bars.get(symbol, {})]

    def present(self, at: float) -> List[str]:
        """The symbols with a bar at this timestamp, in panel order."""
        return [symbol for symbol in self.symbols if float(at) in self.bars.get(symbol, {})]


    def last_at(self, symbol: str, at: float) -> Optional[float]:
        """The newest bar time at or before ``at`` — for marking a name the venue dropped.

        A price from the past is a *fact* ("this is what it last traded at") and a price from
        the future would be a look-ahead; this returns the first and refuses the second.
        """
        name = str(symbol)
        if float(at) in self.index_of.get(name, {}):
            return float(at)
        best: Optional[float] = None
        for value in self.ats_of.get(name, []):
            if float(value) > float(at):
                break
            best = float(value)
        return best

    def ret(self, symbol: str, at: float, lookback: int) -> Optional[float]:
        """The symbol's own return over ``lookback`` bars ending at ``at``, or ``None``.

        ``None`` is the honest answer in three cases, and each one of them would otherwise
        become a number that looks like a return:

        * the symbol has no bar at ``at`` (it was not trading today);
        * its record is shorter than ``lookback`` bars (a name listed last week has no
          fourteen-bar history, and scoring it as a zero would put it in the middle of the
          ranking, where it does not belong);
        * the bar ``lookback`` steps back is not ``lookback`` intervals back in time, which is
          what a listing gap or a venue outage looks like — the window spans a hole, and a
          window that spans a hole is not a return over that window.
        """
        name = str(symbol)
        steps = int(lookback)
        if steps < 1:
            return None
        positions = self.index_of.get(name, {})
        index = positions.get(float(at))
        if index is None or index < steps:
            return None
        times = self.ats_of[name]
        closes = self.closes_of[name]
        earlier_at = float(times[index - steps])
        step = self.interval
        if step > 0.0:
            span = float(at) - earlier_at
            if abs(span - steps * step) > max(GAP_TOLERANCE * step, 1.0) * steps:
                return None
        base = float(closes[index - steps])
        if not (base > 0.0):
            return None
        return float(closes[index]) / base - 1.0

    # -- the whole panel --------------------------------------------------- #

    def returns_at(self, at: float, lookback: int = 1) -> Dict[str, float]:
        """Every symbol's return over ``lookback`` bars ending at ``at``, holes dropped."""
        out: Dict[str, float] = {}
        for symbol in self.symbols:
            value = self.ret(symbol, at, lookback)
            if value is not None and math.isfinite(value):
                out[symbol] = float(value)
        return out

    def as_dict(self) -> Dict[str, Any]:
        """The panel's shape as numbers — what a report prints instead of the bars."""
        first = float(self.ats[0]) if self.ats else 0.0
        last = float(self.ats[-1]) if self.ats else 0.0
        slots = len(self.ats) * max(len(self.symbols), 1)
        filled = sum(len(self.bars.get(symbol, {})) for symbol in self.symbols)
        return {
            "period": self.period,
            "symbols": len(self.symbols),
            "bars": len(self.ats),
            "first": first,
            "last": last,
            "interval_seconds": int(self.interval) if self.interval else 0,
            "filled": filled,
            "holes": max(slots - filled, 0),
            "coverage": round(filled / slots, 6) if slots else 0.0,
            "sources": dict(sorted(self.sources.items())),
            "warnings": list(self.warnings),
        }

    def __repr__(self) -> str:  # pragma: no cover - a debugging aid, not a reading
        return f"Panel({len(self.symbols)} symbols, {len(self.ats)} bars, {self.period})"


def build_panel(
    series: Iterable[CandleSeries],
    *,
    period: str = "",
) -> Panel:
    """One panel from many series, refusing the inputs that would make it not one market.

    The refusals are the interesting half. Two intervals in one panel would let a rank compare
    a four-hour move with a daily one; a duplicate symbol would put the same price in the
    cross-section twice and so give that name two votes; an empty panel would let a caller run
    a book over nothing and read the resulting zero as a finding.

    The bar width comes from ``period`` when the caller declares one, and is otherwise the
    *finest* gap in the panel — a hole can only widen a gap, so the finest gap in a correctly
    sampled record is the venue's own step. Declared and measured are then compared the way the
    distinction actually needs: with a declared width a gap that is a whole number of bars is a
    *hole* and anything else is a different clock, while without one every symbol has to agree
    on its finest gap, because nothing in the panel could tell a hole from a clock.
    """
    panel = Panel(period=str(period) or DEFAULT_PERIOD)
    widths: Dict[str, float] = {}
    for item in series:
        name = str(getattr(item, "symbol", "") or "").strip().lower()
        if not name:
            raise PanelError(
                "a series in this panel has no symbol, so its bars have nowhere to go"
            )
        if name in panel.bars:
            raise PanelError(
                f"{name} appears twice in this panel; the same price twice in one "
                "cross-section is the same name voted twice"
            )
        rows = list(getattr(item, "rows", []) or [])
        if not rows:
            continue
        panel.symbols.append(name)
        panel.bars[name] = {float(row.at): row for row in rows}
        panel.ats_of[name] = sorted(float(row.at) for row in rows)
        panel.index_of[name] = {at: number for number, at in enumerate(panel.ats_of[name])}
        panel.closes_of[name] = [float(panel.bars[name][at].close) for at in panel.ats_of[name]]
        panel.sources[name] = str(getattr(item, "source", "") or "")
        panel.warnings.extend(
            f"{name}: {line}" for line in list(getattr(item, "warnings", []) or [])
        )
        gaps = [b - a for a, b in zip(panel.ats_of[name], panel.ats_of[name][1:]) if b > a]
        # The *finest* gap and not the median: a hole can only widen a gap, so the finest gap
        # in a correctly sampled record is the venue's bar width. A median would let a record
        # that skips every other bar redefine the clock it is read against.
        if gaps:
            widths[name] = min(gaps)
    if not panel.symbols:
        raise PanelError("no series in this panel held a single bar")
    declared = _declared_step(period)
    if declared > 0.0:
        # The width is the caller's declaration, so a gap that is a whole number of bars is a
        # hole and anything else is a different clock.
        panel.step = declared
        offenders = [
            (name, width)
            for name, width in sorted(widths.items())
            if width > 0.0 and not _same_clock(width, declared)
        ]
        why = (
            "this panel mixes bar widths, so a cross-section would compare a move over one "
            "width with a move over another: "
        )
    else:
        # Nothing was declared, so the panel has only itself to measure against and cannot
        # tell a hole from a clock: every symbol must therefore agree on its finest gap.
        finest = sorted({round(width, 6) for width in widths.values() if width > 0.0})
        panel.step = float(finest[0]) if finest else 0.0
        offenders = [
            (name, width)
            for name, width in sorted(widths.items())
            if width > 0.0 and abs(width - panel.step) > GAP_TOLERANCE * panel.step
        ]
        why = (
            "this panel mixes bar widths and no bar width was declared to tell a hole from a "
            "clock, so the same move would be compared over two widths: "
        )
    if offenders:
        everyone = ", ".join(f"{name} at {width:g}s" for name, width in sorted(widths.items()))
        raise PanelError(f"{why}{everyone}")
    panel.period = str(period) or _period_of(panel.step)
    panel.ats = sorted({at for name in panel.symbols for at in panel.ats_of[name]})
    panel.positions = {at: number for number, at in enumerate(panel.ats)}
    return panel


def _declared_step(period: str) -> float:
    """The bar width the caller declared, or ``0.0`` when the calendar has to be measured."""
    name = str(period or "").strip()
    if not name:
        return 0.0
    value = PERIOD_SECONDS.get(name)
    return float(value) if value else 0.0


def _same_clock(width: float, step: float) -> bool:
    """Whether a symbol's finest gap is a whole number of panel bars.

    A hole is a *multiple* of the bar width and a different clock is not, which is exactly the
    distinction the refusal needs: a daily record missing Tuesday belongs in a daily panel (its
    windows will refuse to span the gap), while a four-hour record in that panel would compare
    incomparable moves and is refused.
    """
    if step <= 0.0 or width <= 0.0:
        return True
    ratio = float(width) / float(step)
    whole = round(ratio)
    return abs(ratio - whole) <= GAP_TOLERANCE * max(abs(whole), 1.0)



def _period_of(seconds: float) -> str:
    """The venue's own name for a bar width, or the default when it is not one of them."""
    for name, value in PERIOD_SECONDS.items():
        if abs(float(value) - float(seconds)) < 1e-6:
            return name
    return DEFAULT_PERIOD



def load_panel(
    directory: Any,
    *,
    period: str = "",
    delimiter: str = ",",
) -> Panel:
    """A panel from a directory of per-symbol CSVs — the freeze, read back.

    The file name is the contract: ``btcusdt.csv`` is ``btcusdt``. That is the convention
    :func:`agent.backtest.data.load_csv` already uses when it takes a symbol from the file
    stem, and it means a frozen panel is a directory an operator can open and edit rather than
    an archive that needs this module to read it.
    """
    root = Path(directory)
    if not root.is_dir():
        raise PanelError(f"{root} is not a directory of per-symbol CSVs")
    files = sorted(path for path in root.glob("*.csv") if not path.name.startswith("."))
    if not files:
        raise PanelError(f"{root} holds no *.csv files; a panel needs one per symbol")
    series = [
        load_csv(path, symbol=path.stem.lower(), period=str(period), delimiter=delimiter)
        for path in files
    ]
    return build_panel(series, period=str(period))


def write_panel(panel: Panel, directory: Any, *, delimiter: str = ",") -> List[Path]:
    """One CSV per symbol — the inverse of :func:`load_panel`, exact enough to replay.

    ``write_csv`` is reused rather than re-implemented so a frozen panel and a frozen single
    pair are the same file format: a reader who can open one can open the other, and the
    replay check (same bytes, same table) applies to both without a second convention.
    """
    root = Path(directory)
    written: List[Path] = []
    for symbol in panel.symbols:
        rows = [panel.bars[symbol][at] for at in panel.ats_of[symbol]]
        written.append(
            write_csv(
                CandleSeries(symbol=symbol, period=panel.period, rows=rows),
                root / f"{symbol}.csv",
                delimiter=delimiter,
            )
        )
    return written


def fetch_panel(
    symbols: Sequence[str],
    *,
    period: str = DEFAULT_PERIOD,
    days: float = 365.0,
    until: float = 0.0,
    loader: Optional[Callable[..., CandleSeries]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    endpoint: str = HTX_ENDPOINT,
) -> Tuple[Panel, List[Tuple[str, str]]]:
    """Download a universe and return ``(panel, failures)`` — a dead pair is a sentence.

    The contract is the sweep's: a feed that did not answer for one symbol does not end the
    run, because a hundred-pair panel that dies on the ninth pair answers nothing while a
    ninety-nine-column panel with one apology answers ninety-nine questions. What it must not
    do is *silently* drop the pair, so every failure comes back as ``(symbol, sentence)`` and
    the caller prints it.

    ``until`` asks for a window that *ended* that many days ago, by downloading the extra days
    and cutting the newest bars off the series before the panel is built. This is the only
    cheap way to test a config on a period that did not exist when it was chosen, which is what
    "out of sample" has to mean if it is to mean anything: the same hundred names, the same
    widths, and a window the reader had no say in.

    The venue's own ceiling applies per symbol (:data:`agent.backtest.data.MAX_BARS`), which is
    the honest reason a four-hour panel cannot reach back years: the panel is as long as the
    shortest thing in it, and a symbol listed last month joins the calendar late rather than
    shortening it for everyone.
    """
    fetch = loader or fetch_klines
    series: List[CandleSeries] = []
    failures: List[Tuple[str, str]] = []
    for name in symbols:
        symbol = str(name).strip().lower()
        if not symbol:
            continue
        try:
            item = fetch(
                symbol,
                period=str(period),
                days=float(days) + max(float(until), 0.0),
                timeout=float(timeout),
                endpoint=str(endpoint),
            )
        except (HtxError, ValueError) as exc:
            failures.append((symbol, str(exc)))
            continue
        if not getattr(item, "rows", None):
            failures.append((symbol, "the feed answered with no bars"))
            continue
        series.append(item)
    if not series:
        raise PanelError(
            "not one symbol in the universe could be read"
            + (f"; the first failure was: {failures[0][1]}" if failures else "")
        )
    cut = float(until)
    if cut > 0.0:
        newest = max(float(item.rows[-1].at) for item in series)
        cutoff = newest - cut * 86400.0
        kept: List[CandleSeries] = []
        for item in series:
            rows = [row for row in item.rows if float(row.at) <= cutoff]
            if not rows:
                failures.append(
                    (item.symbol, f"nothing was left of it {cut:.0f} day(s) before the newest bar")
                )
                continue
            # The panel's calendar, indices and close lists are all derived in ``build_panel``,
            # so the window is cut on the *series* and rebuilt, never edited into a panel that
            # nothing had a chance to re-derive.
            kept.append(replace(item, rows=rows))
        if not kept:
            raise PanelError(
                f"--until {cut:g} days before the newest bar left no bar in any of the "
                f"{len(series)} symbol(s) that were read"
            )
        series = kept
    panel = build_panel(series, period=str(period))
    if cut > 0.0:
        panel.warnings.append(
            f"the window was cut to end {cut:g} day(s) before the venue's newest bar: this is "
            "the same universe read on an earlier period, which is the cheapest out-of-sample "
            "test that exists, and it is still not a period nobody had seen"
        )
    return panel, failures

