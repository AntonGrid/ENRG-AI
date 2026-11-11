"""History as a series of candles, and the one conversion the hunter cannot do itself.

The hunter was written for a *pool*, not for a chart. Its two inputs are an APY series
(per cent a year, the rate a pool publishes) and a depth series (dollars of TVL), and its
verdict is a sentence about the two: *the rate stands far above its own rest while depth
left*. A candle feed publishes neither of those things, so this file has to answer one
question before any backtest can begin, and it is better answered in prose than hidden in
a comprehension.

What a candle offers a long position
====================================

A pool pays a holder in APY. A chart pays a holder in exactly one currency: a lower price.
The rate a bar offered a buyer is therefore the *decline* of that bar in per cent — a bar
that fell 2.5 % offered 2.5, a bar that rose offered nothing, zero. :func:`declines` is
that function, and it is the only invented quantity in this package.

Two alternatives were considered and both are worse, which is why the rejected one is
written down here:

**Annualising the bar.** ``(1 + r) ** 365 - 1`` on a daily crypto return produces numbers in
the thousands of per cent. ``risk_assessment`` compares every claim against
:data:`agent.physics.risk_assessment.PLAUSIBLE_APY` — 35 % for a vault, 400 % for an
experiment — so every record would be convicted of promising more than it may, and a
backtest that is refused by its own claim ceiling measures nothing. The per-bar numbers are
also in the range the venue's APYs live in, which is what makes the assessment meaningful
rather than decorative.

**Using the signed return.** A pool's APY cannot be negative, and ``to_returns`` maps every
non-positive growth to ``-1.0``: half a chart of down bars would arrive at the assessment
as a record of total losses, ``implied_apy`` would be zero, and every honest instrument
would be convicted as a ghost. The decline is non-negative by construction for the same
reason a pool's APY is.

So the mapping is stated once, here, and every report prints it: a rate of 0 means the bar
paid nothing (it rose), and a rate of 3.0 means the bar paid three per cent (it fell three).
A ``PANIC`` verdict on this feed therefore reads as *"this instrument suddenly pays far more
than its own rest while participation drained"* — a capitulation on drying volume — and a
``GROWTH`` verdict reads as *"the rate rose and the depth rose with it"*, which on a chart
is the same fall on rising volume. That is not a defect of the mapping; it is the hunter's
thesis, read correctly on a different market.

The depth series, and why the floor is zero
===========================================

``tvl`` becomes quote turnover (``vol`` on HTX — base volume times price would be the same
number, and the fallback does exactly that when a feed publishes only one of the two). A
coin's turnover is not a pool's TVL, and comparing the two in dollars would refuse every
pair honestly measured in different units, so the engine passes ``min_depth=0.0`` by
default and says so: the ``THIN`` arm is off unless an operator asks for it. The exodus
arm, which is the one the verdict actually needs, works unchanged — a median of the last
fortnight against the median before it.

The feed, and what it will not do
=================================

``https://api.htx.com/market/history/kline`` answers one page: the newest
:data:`MAX_BARS` bars, newest first, and it ignores ``from``/``to`` (measured against the
live endpoint while this file was written). So there is no pagination here to get wrong:
the window is cut to what the venue will serve, the cut is reported in ``warnings``, and a
request for more history than that is answered with a shorter series and a sentence rather
than with a loop that looks like it fetched six months and in fact fetched the same page
six times.

Failure is a value, as everywhere in this layer: :class:`HtxError` carries a sentence, the
transport is a parameter (:func:`fetch_klines` takes ``fetch``), and the only code in this
package that can open a socket is the default transport at the bottom of this file — which
is why every test in ``tests/test_backtest_*.py`` runs offline by passing a dictionary.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


__all__ = [
    "DEFAULT_PERIOD",
    "DEFAULT_TIMEOUT",
    "HTX_ENDPOINT",
    "HTX_ENDPOINTS",
    "MAX_BARS",
    "PERIOD_SECONDS",
    "USER_AGENT",
    "Candle",
    "CandleSeries",
    "HtxError",
    "days_to_bars",
    "declines",
    "fetch_klines",
    "interval_of",
    "load_csv",
    "parse_klines",
    "period_name",
]

#: The public kline endpoint, and the mirror the same venue answers on. Both are read-only
#: and unauthenticated: no key, no account, no order — a price history is public data.
HTX_ENDPOINTS = (
    "https://api.htx.com/market/history/kline",
    "https://api.huobi.pro/market/history/kline",
)
HTX_ENDPOINT = HTX_ENDPOINTS[0]

#: The same host is identified to the venue as the rest of this layer identifies itself, so
#: that a misbehaving caller can be found in the venue's logs. It is not a secret.
USER_AGENT = "enrg-ai-backtest/0.1 (+https://github.com/AntonGrid/ENRG-AI)"

#: The bars the venue will serve in one answer, measured rather than assumed. A request for
#: more is a request for the last :data:`MAX_BARS` bars, whatever the window asked for.
MAX_BARS = 2000

#: Seconds per bar for every period the venue names. The values are the venue's own names,
#: not a convenience: a typo in ``--period`` has to be an error and not an empty series.
PERIOD_SECONDS: Dict[str, int] = {
    "1min": 60,
    "5min": 300,
    "15min": 900,
    "30min": 1800,
    "60min": 3600,
    "4hour": 14400,
    "1day": 86400,
    "1week": 604800,
    "1mon": 2592000,
}

#: The default bar. Four hours rather than a day on purpose: the detector will not speak
#: before :data:`agent.physics.portfolio.MIN_RECORD` bars (64), and on daily bars a
#: six-month backtest spends a third of its history warming up. Four-hour bars turn the
#: same six months into 1080 measurements with ten days of warm-up.
DEFAULT_PERIOD = "4hour"

#: How long the default transport waits. A public price feed that cannot answer in twenty
#: seconds is a feed with something wrong with it, and a backtest is not a scraper: there
#: is no retry loop here, because a retry would turn a dead feed into a slow one.
DEFAULT_TIMEOUT = 20.0


class HtxError(RuntimeError):
    """A feed that cannot be read, with the sentence an operator needs.

    Every failure in this file is this exception and never a bare ``URLError`` or a
    ``KeyError`` three frames down: a backtest run is a thing a person starts and walks
    away from, and the line it prints on failure is the whole diagnosis they will get.
    """


# ── the candle, and the series it belongs to ────────────────────────────────


@dataclass(frozen=True)
class Candle:
    """One bar as the venue published it: a time, four prices, and a turnover.

    ``volume`` is **quote** turnover (USDT), not base units, because that is the series the
    hunter's depth arm can compare between two windows; the parser derives it from ``vol``
    and falls back to ``amount × close`` when a feed publishes only base volume. ``trades``
    is kept because it is free and because a report that says how thin a bar was is worth
    more than one that does not.
    """

    at: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trades: int = 0

    @property
    def stamp(self) -> str:
        """The bar's opening time as an ISO-8601 UTC string — what a report prints."""
        return datetime.fromtimestamp(float(self.at), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def as_row(self) -> Dict[str, Any]:
        return {
            "at": float(self.at),
            "stamp": self.stamp,
            "open": round(float(self.open), 8),
            "high": round(float(self.high), 8),
            "low": round(float(self.low), 8),
            "close": round(float(self.close), 8),
            "volume": round(float(self.volume), 4),
        }


@dataclass
class CandleSeries:
    """A symbol's bars, oldest first, plus where they came from and what was cut.

    Oldest first is not a detail: every prefix the engine hands to the detector is
    ``rows[:t + 1]``, and a series in the venue's own (newest-first) order would let the
    detector read the future through an off-by-one nobody would ever see in a number.
    The parser reverses once, here, and the rest of the package never thinks about order.
    """

    symbol: str = ""
    period: str = DEFAULT_PERIOD
    rows: List[Candle] = field(default_factory=list)
    source: str = ""
    warnings: List[str] = field(default_factory=list)

    # -- shape ------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterable[Candle]:
        return iter(self.rows)

    def __getitem__(self, index: Any) -> Any:
        return self.rows[index]

    @property
    def closes(self) -> List[float]:
        return [row.close for row in self.rows]

    @property
    def volumes(self) -> List[float]:
        """Quote turnover per bar — the depth series, in the hunter's own units."""
        return [row.volume for row in self.rows]

    @property
    def interval(self) -> float:
        """Seconds between bars, measured from the series and never assumed."""
        return interval_of(self.rows) or float(PERIOD_SECONDS.get(self.period, 0) or 0)

    def span(self) -> Tuple[float, float]:
        """``(first, last)`` bar time; ``(0, 0)`` on an empty series."""
        if not self.rows:
            return 0.0, 0.0
        return float(self.rows[0].at), float(self.rows[-1].at)

    def as_dict(self) -> Dict[str, Any]:
        first, last = self.span()
        return {
            "symbol": self.symbol,
            "period": self.period,
            "bars": len(self.rows),
            "first": first,
            "last": last,
            "interval_seconds": int(self.interval) if self.interval else 0,
            "source": self.source,
            "warnings": list(self.warnings),
        }


def interval_of(rows: Sequence[Candle]) -> float:
    """The median gap between consecutive bars — a real measurement, not a lookup.

    Median and not mean because a feed with one duplicated timestamp would otherwise
    report a bar length that is half of the true one, and every holding time in the report
    is expressed in these units.
    """
    if len(rows) < 2:
        return 0.0
    gaps = [float(rows[index + 1].at) - float(rows[index].at) for index in range(len(rows) - 1)]
    gaps = [gap for gap in gaps if gap > 0.0]
    if not gaps:
        return 0.0
    gaps.sort()
    return float(gaps[len(gaps) // 2])


def period_name(seconds: float) -> str:
    """The venue's name for a bar length, or ``"<n>s"`` when it has none.

    Used when a CSV says nothing about its period: the report then prints what the file
    actually is (``4hour``) instead of the default it would otherwise have inherited.
    """
    if seconds <= 0.0:
        return ""
    for name, value in PERIOD_SECONDS.items():
        if abs(float(value) - float(seconds)) < 1.0:
            return name
    return f"{int(round(seconds))}s"


def days_to_bars(days: float, interval: float) -> int:
    """How many bars of ``interval`` seconds a window of ``days`` needs. Zero if unknown."""
    if interval <= 0.0:
        return 0
    return max(1, int(math.ceil(float(days) * 86400.0 / float(interval))))


# ── the conversion: a candle as the rate it offered a buyer ──────────────────


def declines(rows: Sequence[Candle], *, window: int = 1) -> List[float]:
    """One value per row: the decline the bar offered, in per cent, ``nan`` while warming up.

    ``window`` is how many bars the decline is measured over — ``1`` means "this bar
    against the previous close", which is the per-bar rate; a longer window smooths the
    series into a trailing rate and is what an operator reaches for when a 4-hour chart is
    too noisy for a 14-bar comparison.

    The first ``window`` rows answer ``nan`` and not ``0.0``. Zero would be a *number* the
    market never published, and it would sit in the resting median of every early prefix;
    ``nan`` is dropped by ``_finite`` inside the detector, so the record it sees is shorter
    and honest rather than longer and decorated — the same rule
    :func:`agent.physics.defillama_adapter.clean_series` keeps about a missing APY.
    """
    span = max(int(window), 1)
    values: List[float] = []
    for index, row in enumerate(rows):
        if index < span:
            values.append(math.nan)
            continue
        past = float(rows[index - span].close)
        now = float(row.close)
        if not math.isfinite(past) or not math.isfinite(now) or past <= 0.0:
            values.append(math.nan)
            continue
        change = now / past - 1.0
        # A bar that rose paid a buyer nothing; a bar that fell paid it the fall.
        values.append(max(0.0, -change * 100.0))
    return values


# ── reading the venue's answer ──────────────────────────────────────────────


def _number(value: Any, *, default: float = 0.0) -> float:
    """A finite float, or the default — a feed's ``None`` is not a zero-price bar."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def parse_klines(
    payload: Any,
    *,
    symbol: str = "",
    period: str = DEFAULT_PERIOD,
    source: str = "",
) -> CandleSeries:
    """The venue's JSON as a :class:`CandleSeries`, oldest first, or an :class:`HtxError`.

    The document is ``{"status": "ok", "ch": ..., "data": [{"id", "open", "high", "low",
    "close", "amount", "vol", "count"}, ...]}`` with the newest bar first. Every way it can
    be wrong is answered here with a sentence: a status that is not ``ok``, a ``data`` that
    is not a list, a row without a price, a series that is empty.
    """
    if not isinstance(payload, Mapping):
        raise HtxError(
            "the HTX endpoint answered something that is not a JSON object; "
            "nothing can be read from it"
        )
    status = str(payload.get("status") or "")
    if status and status != "ok":
        message = str(payload.get("err-msg") or payload.get("err_msg") or "")
        code = payload.get("err-code") or payload.get("err_code") or ""
        raise HtxError(
            f"the HTX endpoint refused the request ({status}"
            f"{f' {code}' if code else ''}){f': {message}' if message else ''}"
        )
    data = payload.get("data")
    if not isinstance(data, list):
        raise HtxError("the HTX answer carries no `data` list, so there are no candles in it")
    rows: List[Candle] = []
    dropped = 0
    for raw in data:
        if not isinstance(raw, Mapping):
            dropped += 1
            continue
        at = _number(raw.get("id", raw.get("t", raw.get("ts"))))
        close = _number(raw.get("close"))
        if at <= 0.0 or close <= 0.0:
            dropped += 1
            continue
        # `vol` is quote turnover and `amount` is base volume; when a feed carries only the
        # second, the first is derived from it and the price. A bar with neither is a bar
        # whose depth nobody published — it gets 0.0 and the exodus arm will say so.
        volume = _number(raw.get("vol"))
        if volume <= 0.0:
            volume = _number(raw.get("amount")) * close
        rows.append(
            Candle(
                at=float(at),
                open=_number(raw.get("open"), default=close),
                high=_number(raw.get("high"), default=close),
                low=_number(raw.get("low"), default=close),
                close=close,
                volume=volume,
                trades=int(_number(raw.get("count"))),
            )
        )
    if not rows:
        raise HtxError(
            "the HTX answer contains no readable candle: every row was missing a time or a "
            "price"
        )
    rows.sort(key=lambda row: row.at)
    unique: List[Candle] = []
    for row in rows:
        if unique and row.at <= unique[-1].at:
            continue
        unique.append(row)
    warnings: List[str] = []
    if dropped:
        warnings.append(f"{dropped} rows of the answer were unreadable and were dropped")
    return CandleSeries(
        symbol=symbol,
        period=period,
        rows=unique,
        source=source or f"htx {symbol or '?'} {period}",
        warnings=warnings,
    )


def fetch_klines(
    symbol: str,
    *,
    period: str = DEFAULT_PERIOD,
    days: float = 180.0,
    fetch: Optional[Callable[[str, float], Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    endpoint: str = HTX_ENDPOINT,
) -> CandleSeries:
    """The newest bars for ``symbol``, as far back as the venue will serve them.

    ``fetch`` is the transport — ``(url, timeout) -> JSON`` — and it is a parameter for the
    same reason the DeFiLlama scanner's is a field: a test must be able to answer with a
    dictionary, and a reader must be able to see that the network lives in exactly one
    place. It defaults to :func:`_http_get` at the bottom of this file.

    The window is ``days`` at ``period``, cut to :data:`MAX_BARS`, and the cut is reported:
    a caller who asked for two years of 4-hour bars receives 2000 bars (333 days) *and* a
    warning that says so, which is the difference between a short backtest and a backtest
    that silently tested something else.
    """
    interval = float(PERIOD_SECONDS.get(str(period), 0) or 0)
    if interval <= 0.0:
        raise HtxError(
            f"{period!r} is not a period this venue publishes; "
            f"one of {sorted(PERIOD_SECONDS)} was expected"
        )
    wanted = days_to_bars(days, interval)
    size = int(min(max(wanted, 2), MAX_BARS))
    url = f"{endpoint}?symbol={str(symbol).lower()}&period={period}&size={size}"
    transport = fetch or _http_get
    series = parse_klines(
        transport(url, float(timeout)),
        symbol=str(symbol).lower(),
        period=str(period),
        source=f"htx {str(symbol).lower()} {period} {size} bars",
    )
    if len(series.rows) > wanted:
        series.rows = series.rows[-wanted:]
    if wanted > MAX_BARS:
        series.warnings.append(
            f"{wanted} bars of {period} were asked for and the venue serves at most "
            f"{MAX_BARS} in one answer, so the window is {len(series.rows)} bars "
            f"({len(series.rows) * interval / 86400.0:.1f} days) instead of {days:g}"
        )
    elif len(series.rows) < wanted:
        series.warnings.append(
            f"{wanted} bars of {period} were asked for and {len(series.rows)} came back: "
            "the symbol has not been listed for the whole window, or the venue is serving "
            "less than it holds"
        )
    return series


# ── a chart from a file: the format an operator can edit ────────────────────


#: Column names accepted for each field, lower-cased. The contract in the brief is
#: ``timestamp, open, high, low, close, volume``; the rest are here because a chart
#: exported from an exchange's own CSV is not going to use our spelling of "time".
CSV_ALIASES: Dict[str, Tuple[str, ...]] = {
    "at": ("timestamp", "time", "date", "datetime", "open_time", "opentime", "at", "id"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "price"),
    "volume": ("volume", "vol", "quote_volume", "quotevolume", "amount"),
}


def _column_of(header: Sequence[str], field_name: str) -> Optional[int]:
    """The index of the first column named like ``field_name``, or ``None``."""
    wanted = CSV_ALIASES[field_name]
    for index, name in enumerate(header):
        if str(name).strip().lower() in wanted:
            return index
    return None


def _timestamp(value: str) -> float:
    """Seconds since the epoch from whatever a CSV calls a time — or ``nan``.

    Four shapes are accepted because four are common: seconds, milliseconds, an ISO
    timestamp with a ``Z``, and one with an offset. ``nan`` answers "not a time", and the
    caller turns that into an error that names the line.
    """
    text = str(value).strip()
    if not text:
        return math.nan
    try:
        number = float(text)
    except ValueError:
        number = math.nan
    if math.isfinite(number):
        # Milliseconds are the giveaway: 1e11 seconds is the year 5138, and every feed
        # that means milliseconds is above that by six orders of magnitude.
        return float(number) / 1000.0 if abs(number) > 1.0e11 else float(number)
    cleaned = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return math.nan
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


def load_csv(
    path: Any,
    *,
    symbol: str = "",
    period: str = "",
    delimiter: str = ",",
) -> CandleSeries:
    """Candles from a CSV on disk — the offline path, and the editable one.

    The format is the one the brief names (``timestamp, open, high, low, close, volume``),
    with a header row and one bar per line. A missing column is an error that names it, a
    line that is not a bar is an error that names the line, and rows are sorted and
    deduplicated, because a file assembled by hand is in whatever order it was assembled.

    ``period`` is inferred from the median gap when the caller does not say, so a report
    never prints "4hour" over a file that is daily.
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise HtxError(f"cannot read {file}: {exc}") from exc
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise HtxError(f"{file} is empty; a backtest needs at least one bar in it")
    reader = csv.reader(lines, delimiter=delimiter)
    header = next(reader)
    at_col = _column_of(header, "at")
    close_col = _column_of(header, "close")
    if at_col is None or close_col is None:
        missing = "timestamp" if at_col is None else "close"
        raise HtxError(
            f"{file} has no {missing} column; the first line is {header!r} and the format "
            "is 'timestamp, open, high, low, close, volume'"
        )
    open_col = _column_of(header, "open")
    high_col = _column_of(header, "high")
    low_col = _column_of(header, "low")
    volume_col = _column_of(header, "volume")
    rows: List[Candle] = []
    for number, record in enumerate(reader, start=2):
        if len(record) <= max(at_col, close_col):
            raise HtxError(f"{file}:{number} has fewer columns than the header promises")
        at = _timestamp(record[at_col])
        close = _number(record[close_col])
        if not math.isfinite(at) or close <= 0.0:
            raise HtxError(
                f"{file}:{number} is not a bar: timestamp {record[at_col]!r} and close "
                f"{record[close_col]!r} do not make one"
            )
        rows.append(
            Candle(
                at=float(at),
                open=_number(record[open_col], default=close) if open_col is not None else close,
                high=_number(record[high_col], default=close) if high_col is not None else close,
                low=_number(record[low_col], default=close) if low_col is not None else close,
                close=float(close),
                volume=_number(record[volume_col]) if volume_col is not None else 0.0,
            )
        )
    if not rows:
        raise HtxError(f"{file} has a header and no bars; a backtest needs at least one")
    rows.sort(key=lambda row: row.at)
    unique: List[Candle] = []
    dropped = 0
    for row in rows:
        if unique and row.at <= unique[-1].at:
            dropped += 1
            continue
        unique.append(row)
    interval = interval_of(unique)
    warnings = [f"{dropped} duplicate rows were dropped"] if dropped else []
    return CandleSeries(
        symbol=symbol or file.stem,
        period=period or period_name(interval) or DEFAULT_PERIOD,
        rows=unique,
        source=f"csv {file}",
        warnings=warnings,
    )


# ── the only socket in the package ─────────────────────────────────────────


def _http_get(url: str, timeout: float = DEFAULT_TIMEOUT) -> Any:
    """The default transport: one GET, a timeout, and a sentence when it fails.

    ``urllib`` from the standard library rather than an HTTP client, because this is the
    one module of the package that must work on a box where nothing but the standard
    library is installed — a backtest is a thing an operator runs on a laptop, and
    ``httpx`` is a dependency of the *service*, not of its history.

    Everything a dead feed raises — DNS, a refused connection, a timeout, a 429 — is an
    ``OSError`` somewhere in the chain, and all of it becomes :class:`HtxError` here.
    """
    from urllib.request import Request, urlopen

    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urlopen(request, timeout=float(timeout)) as response:  # noqa: S310 - https
            raw = response.read()
    except OSError as exc:
        raise HtxError(
            f"the HTX endpoint did not answer ({exc}); the backtest needs a price history "
            "and will not invent one — pass --csv to run from a file instead"
        ) from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise HtxError(
            f"the HTX endpoint answered {len(raw)} bytes that are not JSON"
        ) from exc
