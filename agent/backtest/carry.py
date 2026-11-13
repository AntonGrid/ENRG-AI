"""Funding as the one payment a delta-neutral book collects, and the coverage that admits it.

Why this file exists
====================

The cross-section is closed, and it is closed with numbers rather than opinions: the reversal
effect lived in one cell of the grid (a four-hour, top-100 book) and vanished when the same
contracts were read on another interval, another rank band or one hour of shift. The operator's
answer was to stop predicting price and start collecting the fee for liquidity: **short the
perp, long the spot, and be paid the funding rate for holding the pair**. The arithmetic of
that business is ``money = edge * capital * time``, and the edge is not a forecast — it is the
published rate, three times a day.

This module is the first step and deliberately only that step: **it collects the funding
history and reports what the venue actually published.** It does not compute an APY, a net
carry, or a Sharpe of the carry book, and it cannot: a funding series with a hole is a fact
about the data, and an average that silently bridges the hole would be a fact about this file.

What the venue publishes, measured while this file was written
=============================================================

* ``swap_historical_funding_rate`` answers ``{"status": "ok", "data": {"total_page",
  "current_page", "total_size", "data": [...]}}``, newest page first (``page_index=1``), and a
  record is ``funding_time`` (epoch **milliseconds**), ``funding_rate``, ``avg_premium_index``,
  ``realized_rate``, ``contract_code``.
* ``page_size=200`` is **silently clamped to 100 rows**. ``total_page`` is computed against the
  clamped width (BTC-USDT: ``total_size=6643``, ``total_page=67``), so a walk that trusts the
  requested width over ``total_page`` stops at page 34 and calls half a history complete. The
  walk here is driven by ``total_page`` and checked against ``total_size``.
* ``realized_rate`` was ``null`` in **every** record sampled (BTC, DOGE, QNTX). The rate that
  was actually applied is therefore not published on this endpoint, and this file uses
  ``funding_rate`` — the rate the venue announced — and says so in :data:`DOES_NOT_SHOW`.
* ``funding_rate`` is **not always a number**: ``null`` and HTX's scientific zero ``"0E-18"``
  both mean *the venue did not report a rate*, not *the rate was zero*. At the time of writing,
  ``swap_batch_funding_rate`` returned 425 contracts, of which **242 carried ``"0E-18"`` and 4
  carried ``null``** — 57.9 % of the market. Reading those as zero would turn "unknown" into
  "no carry" for most of the universe, so :func:`_rate` maps both spellings to ``None`` and the
  coverage report counts them separately. A literal ``"0.000000000000000000"`` *is* a reported
  zero and stays one; the two spellings are distinguishable in the JSON, which is why this
  rule is possible at all.
* The funding cadence is **per contract**, not a global 8 hours: ``swap_contract_info`` listed
  418 live USDT swaps, of which 311 settle every 8 h, 92 every 4 h and 15 every hour. A gap of
  "three periods" therefore means three different durations on three different contracts, and
  :class:`FundingSeries` measures holes against the contract's own ``settlement_period``.

The second half: what does not work here, and why
=================================================

**Perp klines cannot be paginated.** ``linear-swap-ex/market/history/kline`` ignores ``from``
and ``to`` (a request for 2020-09 .. 2021-01 answered with the newest 2000 bars, 2025-12-15 ..
2026-11-13) and refuses ``size=5000`` with an empty ``data`` list. So this module fetches one
page of at most :data:`KLINE_SIZE` bars, exactly like the spot feed in
:mod:`agent.backtest.data`, and reports the cut rather than looping over a window the venue
never honoured.

**``vol`` on the swap feed is not quote turnover.** Measured on a BTC-USDT 4-hour bar:
``amount=2001.488``, ``vol=2001488`` and ``trade_turnover=1.383708186328E8`` at a price near
69 100. ``vol`` is ``amount * 1000`` — a count, not dollars — while ``trade_turnover`` is the
dollar figure. :func:`parse_swap_klines` reads ``trade_turnover`` and derives it from
``amount * close`` only when it is absent; :func:`agent.backtest.data.parse_klines` would have
recorded a thousand times the base volume as if it were turnover, which is the kind of unit
error this repository has already paid for once.

**Basis is *averaged*, not sampled.** ``avg_premium_index`` is the premium averaged over the
funding interval, so it is a fair proxy for the basis that period and not a spot-perp spread at
a timestamp. Any later calculation that calls it "the basis at time t" is wrong by
construction, and this file prints it as bps *per period* for that reason.

**Liquidations are not available.** ``swap_liquidation_orders`` answers ``code 1347`` on every
window, so the forced-flow branch is not started. Nothing here can say whether a short leg would
have survived a squeeze, and :data:`DOES_NOT_SHOW` says it.

**Coverage says nothing about returns.** Not one number in this report is a rate of return: a
contract with 99 % reported funding and a positive median can still have paid nothing after the
two taker legs, the basis, and the margin. That measurement is the next session's job, against
kill criteria declared before it runs — 12 % net APY, drawdown <= 3 %, funding > 0 at least 60 %
of the time, capacity >= $50k at <= 3x leverage. This session's product is the roster of
contracts on which that measurement is even possible, with every hole left visible.

What the cache is, and why it is hashed
=======================================

``fetch_funding`` writes ``data/funding/<CONTRACT>/funding.csv`` and a ``manifest.json`` beside
it. The manifest records what the venue claimed (``total_size``, ``total_page``), what was
written, the page width it actually served, the fetch time, and the SHA-256 of the CSV. A second
call with a valid manifest makes **no request at all**, which is what makes a coverage report
reproducible: the numbers came from bytes, and the bytes have a name.

``data_sha256`` is the hash of the manifest *without* ``fetched_at``, so a refetch that brings
identical records hashes identically while the provenance time moves. A report that cannot be
reproduced bit-for-bit is not a measurement, and the tests check both halves of that sentence.

Nothing in this module is a signal, and the criterion it prints is not a profitability criterion:
:data:`COVERAGE_GATE` is the declared floor for "this contract's history is worth measuring", and
it was chosen before the first fetch, not after the first table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.backtest import data as _data
from agent.backtest.criteria import Criteria, verify
from agent.backtest.data import (
    DEFAULT_TIMEOUT,
    Candle,
    CandleSeries,
    HtxError,
    fetch_klines,
)

#: The package keeps exactly one socket, at the bottom of :mod:`agent.backtest.data`; every
#: fetcher here takes it as a ``fetch=`` parameter and defaults to that one, so a test can hand
#: in a dictionary-answering callable and no test ever opens a connection.
from agent.backtest.data import _http_get as _default_transport
from agent.backtest.run import EXIT_FEED, EXIT_SHORT, EXIT_USAGE

__all__ = [
    "ALIGNMENT_TOLERANCE",
    "CONTRACT_INFO_ENDPOINT",
    "COVERAGE_ACCEPTANCE",
    "COVERAGE_GATE",
    "DEFAULT_CACHE",
    "DEFAULT_KLINES",
    "DEFAULT_OUT",
    "DEFAULT_UNIVERSE",
    "DOES_NOT_SHOW",
    "FUNDING_ENDPOINT",
    "FUNDING_PERIOD_SECONDS",
    "GAP_TOLERANCE",
    "KLINE_SIZE",
    "POLITE_DELAY",
    "SERVED_PAGE_CAP",
    "SWAP_KLINE_ENDPOINT",
    "SWAP_TICKERS_ENDPOINT",
    "Alignment",
    "Contract",
    "Coverage",
    "CoverageReport",
    "FundingPage",
    "FundingRecord",
    "FundingSeries",
    "build_manifest",
    "cached_contracts",
    "check_alignment",
    "coverage_criteria",
    "coverage_of",
    "fetch_contracts",
    "fetch_funding",
    "fetch_funding_cached",
    "fetch_swap_klines",
    "fetch_swap_klines_cached",
    "fetch_turnover",
    "format_table",
    "format_universe",
    "funding_dir",
    "load_funding",
    "load_manifest",
    "main",
    "manifest_hash",
    "parse_contracts",
    "parse_funding",
    "parse_swap_klines",
    "parse_turnover",
    "period_name",
    "period_seconds_from",
    "rank_contracts",
    "read_universe",
    "report_of",
    "summary_line",
    "write_coverage",
    "write_funding",
    "write_universe",
]

#: The endpoints this module reads. All public, all unauthenticated, all read-only: a funding
#: history is published data, and nothing here signs anything.
FUNDING_ENDPOINT = "https://api.hbdm.com/linear-swap-api/v1/swap_historical_funding_rate"
CONTRACT_INFO_ENDPOINT = "https://api.hbdm.com/linear-swap-api/v1/swap_contract_info"
SWAP_KLINE_ENDPOINT = "https://api.hbdm.com/linear-swap-ex/market/history/kline"
SWAP_TICKERS_ENDPOINT = "https://api.hbdm.com/linear-swap-ex/market/detail/batch_merged"

#: The venue's funding cadences, keyed by the ``settlement_period`` field of
#: ``swap_contract_info`` (hours, as a string). Measured over the 418 live USDT swaps: 311 at
#: 8 h, 92 at 4 h, 15 at 1 h — a single global 8 h would mis-measure a quarter of the universe.
FUNDING_PERIOD_SECONDS: Dict[str, float] = {
    "1": 3600.0,
    "4": 14400.0,
    "8": 28800.0,
}

#: The funding step this file falls back to when the contract's own cadence is unknown. It is
#: the venue's most common one (311 of 418), and a series collected with it says so in warnings.
DEFAULT_PERIOD_SECONDS = 28800.0

#: The page width this module asks for, and the width the venue actually serves. Measured:
#: ``page_size=200`` answers 100 rows, and ``total_page`` is computed against 100 — so the walk
#: is driven by ``total_page`` and the clamp is reported rather than trusted.
DEFAULT_PAGE_SIZE = 200
SERVED_PAGE_CAP = 100

#: Seconds between two page requests. The venue publishes no rate limit for this endpoint and
#: this module has no desire to discover one; six hundred contracts at 67 pages each is 40 000
#: requests, and a polite crawler is the only kind that finishes.
POLITE_DELAY = 0.15

#: How far a funding timestamp may sit from its grid position, as a fraction of one period,
#: before the step counts as a hole. Funding stamps land on exact grid points, so this is only
#: there to keep a venue that rounds to the second from looking like a missing payment.
GAP_TOLERANCE = 0.01

#: How far a perp bar and a spot bar may sit apart before the join counts them as different
#: timestamps, in seconds. Both feeds label a 4-hour bar by its open, so the honest number is 0;
#: the tolerance exists so that a one-second rounding is reported as a note, not as a hole.
ALIGNMENT_TOLERANCE = 1.0

#: The most bars the perp kline endpoint serves in one answer. Measured: ``size=5000`` answers an
#: empty ``data`` list, and ``from``/``to`` are ignored, so there is no second page to ask for.
KLINE_SIZE = 2000

#: Where the funding cache and the coverage report land. ``data/`` is this repository's scratch
#: directory (gitignored except ``data/frozen``), so a cache never becomes a source file.
DEFAULT_CACHE = "data/funding"
DEFAULT_OUT = "data/carry/coverage.json"
#: The universe snapshot ``--contracts`` and ``--fetch-top`` write, and ``--report`` reads back.
#: It is a snapshot, not a live feed: a report that re-read the venue would not be reproducible.
DEFAULT_UNIVERSE = "data/carry/universe.json"
#: Where ``--klines`` keeps one page of perp bars per contract, beside the funding cache.
DEFAULT_KLINES = "data/carry/klines"

#: The files inside one contract's cache directory, and inside the universe snapshot.
FUNDING_FILE = "funding.csv"
FUNDING_MANIFEST = "manifest.json"
CONTRACTS_FILE = "contracts.json"
UNIVERSE_FILE = "universe.json"

#: The CSV's columns, in the order :func:`write_funding` writes them. A rate the venue did not
#: report is an *empty cell*, never a ``0``: the file has to be able to say "nothing published".
FUNDING_COLUMNS: Tuple[str, ...] = ("at", "funding_rate", "premium_index", "realized_rate")

#: The manifest's schema version. Bumped when a field changes meaning, so a stale cache is
#: refetched instead of being read as if it were written by this code.
MANIFEST_SCHEMA = 1

#: What makes a contract's funding history *measurable* in the next session, declared before
#: the first fetch and hashed into ``criteria.json``. These are floors on the data, not on
#: profitability: two years, 95 % of the periods with a rate the venue actually reported, and no
#: hole longer than three funding periods. A contract that fails them is not "a weaker edge"; it
#: is one this repository cannot price at all.
COVERAGE_GATE: Dict[str, float] = {
    "min_days": 730.0,
    "min_reported_share": 0.95,
    "max_gap_periods": 3.0,
}

#: What a *usable* row had to satisfy, spelled out so that "usable" is not an opinion.
COVERAGE_ACCEPTANCE: Tuple[str, ...] = (
    "the venue claimed a total and the walk read it: len(records) is checked against "
    "``total_size``, and a shortfall is printed as a hole in the history, not filled",
    "a period with no rate is not a zero: ``null`` and HTX's scientific zero ``0E-18`` are "
    "carried as unreported, counted separately, and excluded from every average",
    "holes are measured against the contract's own settlement period, taken from "
    "``swap_contract_info`` and cross-checked against the observed step",
    "the report is reproducible from the cache: a valid manifest means zero requests, and the "
    "CSV's SHA-256 is recorded beside the numbers it produced",
    "the criterion is hashed before the fetch and read back from the file, so a floor cannot be "
    "lowered after the table was seen",
)

#: The honest half: what this coverage report has not shown, hashed with the gate so a passing
#: run cannot quietly drop the sentence that says what it did not prove.
DOES_NOT_SHOW: Tuple[str, ...] = (
    "anything about returns: no APY, no net carry, no Sharpe, and no dollar of profit is "
    "claimed here — coverage is the roster, not the result",
    "what the venue actually credited: ``realized_rate`` is null in every record sampled, so "
    "``funding_rate`` (the announced rate) stands in for the paid one and may differ",
    "that a period with no published rate paid nothing: ``null`` and ``0E-18`` are carried as "
    "unreported and excluded from averages, which is not the same as a zero",
    "anything about the basis beyond ``avg_premium_index``: it is a premium averaged over the "
    "funding interval, not a spot-perp spread at a timestamp",
    "anything about costs: the two taker legs, the borrow, the margin and the liquidation of a "
    "short leg are next session's measurement and are not charged here",
    "anything about delisted contracts: ``swap_contract_info`` lists what is live today, so a "
    "contract that died is a hole in this universe that no row admits to",
    "anything outside one kline page: the perp feed ignores ``from``/``to`` and serves at most "
    "2000 bars, so the spot-perp join is only checked on that window",
    "capacity: no order book depth is read, so a contract that reports a beautiful funding rate "
    "may be one that cannot absorb $50k without moving",
    "that a usable history is a tradable one: this file says the data is there, never that the "
    "trade would have survived its own margin call",
)


# ── reading the venue's words without inventing any ─────────────────────────

#: HTX's spelling of "no rate": a scientific zero such as ``'0E-18'``. A plain
#: ``'0.000000000000000000'`` is *not* matched — that is a reported zero, and the venue really
#: does publish both spellings for different contracts.
_UNREPORTED = re.compile(r"^[+-]?0(?:\.0*)?[eE][+-]?\d+$")

#: Strings that mean the field was absent rather than zero. Kept separate from the pattern above
#: because a JSON ``null`` arrives as ``None`` while a sloppy feed may send the word.
_ABSENT = frozenset({"", "null", "none", "nan"})


def _number(value: Any, default: float = 0.0) -> float:
    """A float from whatever the venue sent, or ``default`` when it sent nothing usable.

    Deliberately *not* the same as :func:`_rate`: a missing contract status or a missing
    timestamp is a zero here (and the caller decides what that means), while a missing funding
    rate is ``None`` and stays ``None`` all the way to the report.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _rate(value: Any) -> Optional[float]:
    """The rate the venue reported, or ``None`` when it reported none — which is not a zero.

    ``null`` and HTX's scientific zero (``"0E-18"``, measured on 242 of 425 live contracts at
    the time of writing) both mean *no rate was published for this period*. A literal
    ``"0.000000000000000000"`` is a zero the venue chose to publish and is returned as ``0.0``.

    The one thing this cannot recover is a *numeric* JSON zero (``0`` rather than ``"0E-18"``):
    ``json.loads`` has already forgotten which spelling it read. The endpoint sends funding
    rates as strings, so the distinction holds on this feed — and :data:`DOES_NOT_SHOW` keeps
    the caveat visible for any feed that sends numbers.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _ABSENT or _UNREPORTED.match(text):
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> float:
    """Epoch **seconds** from the venue's epoch milliseconds, or ``0.0`` when unreadable.

    ``funding_time`` is ``"1794585600000"`` — milliseconds, as a string. Klines publish seconds.
    The threshold sits at ``1e11`` because seconds and milliseconds are six orders of magnitude
    apart and no plausible timestamp is near the boundary.
    """
    number = _number(value)
    if number <= 0.0:
        return 0.0
    return number / 1000.0 if number > 1e11 else number


def _utc(at: Optional[float]) -> str:
    """An ISO-8601 UTC date for a timestamp, or an empty string for a missing one."""
    if at is None or not math.isfinite(float(at)) or float(at) <= 0.0:
        return ""
    return datetime.fromtimestamp(float(at), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _page_index(url: str) -> int:
    """The ``page_index`` a URL asks for, or ``1`` — used to prove the walk asked once each."""
    from urllib.parse import parse_qs, urlparse

    values = parse_qs(urlparse(url).query).get("page_index") or ["1"]
    try:
        return int(values[0])
    except (TypeError, ValueError):
        return 1


def _quantile(values: Sequence[float], fraction: float) -> float:
    """A linearly interpolated quantile of an already-sorted, non-empty list.

    Written out rather than imported because ``statistics.quantiles`` refuses a sample of one and
    this report must be able to say "95th percentile of three records is the largest of them"
    instead of raising on a short history.
    """
    if not values:
        raise ValueError("a quantile of no values is not a number")
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * min(max(float(fraction), 0.0), 1.0)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(values[low])
    weight = position - low
    return float(values[low]) * (1.0 - weight) + float(values[high]) * weight


def period_name(seconds: float) -> str:
    """The venue's own name for a funding cadence, so a table never prints ``28800.0``."""
    for name, value in (("8hour", 28800.0), ("4hour", 14400.0), ("1hour", 3600.0)):
        if abs(float(seconds) - value) < 1.0:
            return name
    return f"{float(seconds):g}s"


def period_seconds_from(settlement_period: Any) -> float:
    """The funding step a contract declares, in seconds, or ``0.0`` when it declares neither 1/4/8.

    The venue publishes the cadence per contract and the three values above cover all 418 live
    USDT swaps today. A contract that answers something else is not silently assumed to be 8 h:
    it gets ``0.0``, the series derives its step from the data, and the report says so.
    """
    return FUNDING_PERIOD_SECONDS.get(str(settlement_period).strip(), 0.0)


@dataclass(frozen=True)
class Contract:
    """One live USDT perpetual, with the cadence the venue declares for it.

    ``settlement_period`` is the venue's string (``"8"``, ``"4"``, ``"1"``) and is kept as sent
    because it is evidence; :attr:`period_seconds` is the same fact in the unit the rest of this
    module measures time in, and is ``0.0`` when the venue answered something unexpected.
    ``turnover`` is the 24-hour quote turnover from ``batch_merged``, in dollars, and is ``0.0``
    when the contract was read from ``swap_contract_info`` alone.
    """

    contract_code: str
    settlement_period: str = ""
    period_seconds: float = 0.0
    create_date: str = ""
    turnover: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_code": self.contract_code,
            "settlement_period": self.settlement_period,
            "period_seconds": float(self.period_seconds),
            "create_date": self.create_date,
            "turnover": float(self.turnover),
        }


@dataclass(frozen=True)
class FundingRecord:
    """One funding payment as the venue announced it — or the announcement that it announced none.

    ``funding_rate`` is ``None`` when the venue published ``null`` or ``"0E-18"``: no rate was
    reported for that period. ``premium_index`` is the premium *averaged over the interval*, not
    a snapshot of the spot-perp spread, and is ``None`` under the same rule. ``realized_rate`` is
    carried because the venue sends it, and it has been ``null`` in every record sampled: the
    placeholder is kept so that a feed which starts publishing it becomes visible rather than
    ignored.
    """

    at: float
    funding_rate: Optional[float] = None
    premium_index: Optional[float] = None
    realized_rate: Optional[float] = None

    @property
    def reported(self) -> bool:
        """Whether the venue published a rate for this period."""
        return self.funding_rate is not None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "at": float(self.at),
            "at_utc": _utc(self.at),
            "funding_rate": self.funding_rate,
            "premium_index": self.premium_index,
            "realized_rate": self.realized_rate,
            "reported": self.reported,
        }


@dataclass(frozen=True)
class FundingPage:
    """One answer from the funding endpoint: the totals it claimed and the rows it carried.

    ``total_size`` and ``total_page`` are the venue's own words and are the only trustworthy
    description of how long the history is — the requested page width is not, because the venue
    clamps it. ``served`` is how many rows actually arrived, which is how a clamp is detected.
    """

    total_size: int = 0
    total_page: int = 0
    served: int = 0
    records: Tuple[FundingRecord, ...] = ()
    unreadable: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_size": int(self.total_size),
            "total_page": int(self.total_page),
            "served": int(self.served),
            "unreadable": int(self.unreadable),
            "records": len(self.records),
        }


@dataclass
class FundingSeries:
    """One contract's funding history, with every hole left where the venue left it.

    The series is the unit the rest of the session reads: it knows the cadence it was collected
    under, counts what the venue reported against what it did not, and measures holes in
    *periods* — the only unit in which "a three-day gap" means the same thing on an 8-hour
    contract as on an hourly one.

    Nothing here interpolates, forward-fills, or drops a record for being uninteresting. A
    missing period is missing: :meth:`gaps` reports where, how long, and how many payments it
    swallowed, and every average is computed over the records that exist.
    """

    contract: str
    period_seconds: float = 0.0
    records: List[FundingRecord] = field(default_factory=list)
    source: str = ""
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def first(self) -> Optional[float]:
        """The oldest funding timestamp, or ``None`` for an empty series."""
        return float(self.records[0].at) if self.records else None

    @property
    def last(self) -> Optional[float]:
        """The newest funding timestamp, or ``None`` for an empty series."""
        return float(self.records[-1].at) if self.records else None

    @property
    def reported(self) -> int:
        """How many records carry a rate the venue actually published."""
        return sum(1 for record in self.records if record.reported)

    @property
    def unreported(self) -> int:
        """How many records carry ``null`` or ``0E-18`` — which is not a count of zeros."""
        return len(self.records) - self.reported

    @property
    def reported_share(self) -> Optional[float]:
        """The share of records with a published rate, or ``None`` for an empty series."""
        return (self.reported / len(self.records)) if self.records else None

    def rates(self) -> List[float]:
        """The published rates in the order they were paid, with unreported periods omitted."""
        return [
            float(record.funding_rate)
            for record in self.records
            if record.funding_rate is not None
        ]

    @property
    def positive_share(self) -> Optional[float]:
        """The share of *reported* rates above zero — the "funding > 0 % of the time" column.

        The denominator is the reported count, not the record count: a contract on which the
        venue reported nothing half the time cannot claim it was positive for all of it.
        """
        rates = self.rates()
        if not rates:
            return None
        return sum(1 for rate in rates if rate > 0.0) / len(rates)

    @property
    def mean_bps(self) -> Optional[float]:
        """The mean published rate per funding period, in bps."""
        rates = self.rates()
        return statistics.fmean(rates) * 1e4 if rates else None

    @property
    def median_bps(self) -> Optional[float]:
        """The median published rate per funding period, in bps."""
        rates = self.rates()
        return statistics.median(rates) * 1e4 if rates else None

    @property
    def p95_bps(self) -> Optional[float]:
        """The 95th percentile of the published rate per period, in bps — the fat tail."""
        rates = sorted(self.rates())
        return _quantile(rates, 0.95) * 1e4 if rates else None

    @property
    def mean_abs_premium_bps(self) -> Optional[float]:
        """The mean absolute ``avg_premium_index``, in bps — a proxy for the basis, averaged.

        It is *not* the spot-perp spread at a moment: the venue averages the premium over the
        funding interval, and :data:`DOES_NOT_SHOW` says what that costs a later calculation.
        """
        premiums = [
            abs(float(record.premium_index))
            for record in self.records
            if record.premium_index is not None
        ]
        return statistics.fmean(premiums) * 1e4 if premiums else None

    @property
    def span_days(self) -> float:
        """The distance between the first and the last record, in days."""
        if self.first is None or self.last is None:
            return 0.0
        return max(float(self.last) - float(self.first), 0.0) / 86400.0

    @property
    def years(self) -> float:
        """The span in 365-day years — a description of the window, never an annualisation."""
        return self.span_days / 365.0

    def observed_step(self) -> Optional[float]:
        """The median distance between consecutive records, in seconds, or ``None``.

        The venue's cadence is declared per contract and this is the measurement that checks it:
        a series whose observed step disagrees with its declared ``settlement_period`` is a
        contract the venue changed the rules on, and the collector says so in warnings.
        """
        if len(self.records) < 2:
            return None
        diffs = [
            float(self.records[index + 1].at) - float(self.records[index].at)
            for index in range(len(self.records) - 1)
        ]
        diffs = [diff for diff in diffs if diff > 0.0]
        return statistics.median(diffs) if diffs else None

    def step(self) -> float:
        """The funding step this series is measured against: declared first, observed second."""
        observed = self.observed_step()
        if float(self.period_seconds) > 0.0:
            return float(self.period_seconds)
        return float(observed) if observed else 0.0

    def gaps(self) -> List[Tuple[float, float, int]]:
        """Every hole as ``(before_at, after_at, missing_periods)``, measured in periods.

        A hole exists when two consecutive records are further apart than one step (plus
        :data:`GAP_TOLERANCE`). ``missing_periods`` is the number of funding payments the venue
        did not publish inside it — a fact about the data, not a value to fill.
        """
        step = self.step()
        if step <= 0.0 or len(self.records) < 2:
            return []
        holes: List[Tuple[float, float, int]] = []
        for index in range(len(self.records) - 1):
            before = float(self.records[index].at)
            after = float(self.records[index + 1].at)
            distance = after - before
            if distance <= step * (1.0 + GAP_TOLERANCE):
                continue
            missing = int(round(distance / step)) - 1
            if missing >= 1:
                holes.append((before, after, missing))
        return holes

    @property
    def gap_count(self) -> int:
        """How many separate holes the history has."""
        return len(self.gaps())

    @property
    def max_gap_periods(self) -> int:
        """The longest hole, in funding periods — the number the coverage gate reads."""
        return max((hole[2] for hole in self.gaps()), default=0)

    def as_dict(self) -> Dict[str, Any]:
        holes = sorted(self.gaps(), key=lambda hole: -hole[2])[:5]
        return {
            "contract": self.contract,
            "period": period_name(self.step()),
            "period_seconds": float(self.step()),
            "declared_period_seconds": float(self.period_seconds),
            "records": len(self.records),
            "reported": self.reported,
            "unreported": self.unreported,
            "reported_share": self.reported_share,
            "positive_share": self.positive_share,
            "mean_bps": self.mean_bps,
            "median_bps": self.median_bps,
            "p95_bps": self.p95_bps,
            "mean_abs_premium_bps": self.mean_abs_premium_bps,
            "first": self.first,
            "last": self.last,
            "first_utc": _utc(self.first),
            "last_utc": _utc(self.last),
            "span_days": self.span_days,
            "years": self.years,
            "gap_count": self.gap_count,
            "max_gap_periods": self.max_gap_periods,
            "largest_gaps": [
                {
                    "before_utc": _utc(hole[0]),
                    "after_utc": _utc(hole[1]),
                    "missing_periods": int(hole[2]),
                }
                for hole in holes
            ],
            "source": self.source,
            "warnings": list(self.warnings),
        }

# ── reading the venue's answers ────────────────────────────────────────────


def _answer(payload: Any, what: str) -> Any:
    """The ``data`` of an HTX answer, or :class:`HtxError` quoting the venue's own refusal.

    Every endpoint this module reads answers ``{"status": "ok", "data": ...}`` when it is happy
    and ``{"status": "error", "err-code": ..., "err-msg": ...}`` when it is not. A failure is
    raised rather than returned as an empty list, because "the venue said no" and "the venue has
    nothing" are different facts and only one of them is a coverage problem.
    """
    if not isinstance(payload, Mapping):
        raise HtxError(f"{what}: the venue answered {type(payload).__name__}, not an object")
    status = str(payload.get("status", "") or "").strip().lower()
    if status and status != "ok":
        detail = (
            payload.get("err-msg")
            or payload.get("err_msg")
            or payload.get("message")
            or payload.get("code")
            or "no detail"
        )
        raise HtxError(f"{what}: the venue answered status {status!r}: {detail}")
    if "data" not in payload:
        raise HtxError(f"{what}: the answer carries no data field at all")
    return payload["data"]


def parse_contracts(
    payload: Any, *, turnover: Optional[Mapping[str, float]] = None
) -> List[Contract]:
    """The live swap contracts ``swap_contract_info`` listed, oldest code first by name.

    The list is the *universe*, and it is a snapshot of what is trading now: a contract that was
    delisted is absent rather than present-with-a-hole, which :data:`DOES_NOT_SHOW` admits.
    ``turnover`` is merged in when the caller has it, because the ranking the operator sees has
    to be by dollars traded rather than by the alphabet.
    """
    rows = _answer(payload, "swap_contract_info")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise HtxError(f"swap_contract_info: expected a list of contracts, got {type(rows).__name__}")
    found: Dict[str, Contract] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        code = str(row.get("contract_code", "") or "").strip().upper()
        if not code:
            continue
        settlement = str(row.get("settlement_period", "") or "").strip()
        found[code] = Contract(
            contract_code=code,
            settlement_period=settlement,
            period_seconds=period_seconds_from(settlement),
            create_date=str(row.get("create_date", "") or "").strip(),
            turnover=float((turnover or {}).get(code, 0.0)),
        )
    return [found[code] for code in sorted(found)]


def parse_turnover(payload: Any) -> Dict[str, float]:
    """24-hour quote turnover per contract from ``batch_merged``, in dollars.

    The ticker's ``trade_turnover`` is the dollar figure and that is the one read; ``vol`` is
    ``amount * 1000`` on this feed (measured on BTC-USDT: ``amount=2001.488``, ``vol=2001488``,
    ``trade_turnover=1.38E8``), so using it for a ranking would rank a thousand-fold lie.

    The list arrives under ``ticks``, not ``data`` — measured on the live endpoint, which answers
    ``{"status": "ok", "ticks": [...]}`` and an empty ``data`` list beside it. Reading ``data``
    here left every turnover at ``0.0`` and made the ranking alphabetical while looking like a
    measurement, so the key is named rather than assumed.
    """
    if isinstance(payload, Mapping) and "ticks" in payload:
        payload = {"status": payload.get("status", "ok"), "data": payload["ticks"]}
    ticks = _answer(payload, "batch_merged")
    if not isinstance(ticks, Sequence) or isinstance(ticks, (str, bytes)):
        raise HtxError(f"batch_merged: expected a list of tickers, got {type(ticks).__name__}")
    turnover: Dict[str, float] = {}
    for tick in ticks:
        if not isinstance(tick, Mapping):
            continue
        code = str(tick.get("contract_code", "") or "").strip().upper()
        if not code:
            continue
        turnover[code] = _number(tick.get("trade_turnover"))
    return turnover


def parse_funding(payload: Any, contract: str) -> FundingPage:
    """One page of ``swap_historical_funding_rate`` as records plus the totals it claimed.

    The venue serves the newest page first and clamps the requested width, so the totals in the
    answer — not the width asked for — are what :func:`fetch_funding` walks against. A row whose
    ``funding_time`` is unreadable, or which belongs to a different contract, is counted in
    ``unreadable`` and dropped: a page is never allowed to pass for longer than it is.
    """
    data = _answer(payload, f"funding {contract}")
    if not isinstance(data, Mapping):
        raise HtxError(f"funding {contract}: expected an object of pages, got {type(data).__name__}")
    rows = data.get("data") or []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise HtxError(f"funding {contract}: the page is not a list of records")
    wanted = str(contract).strip().upper()
    records: List[FundingRecord] = []
    unreadable = 0
    for row in rows:
        if not isinstance(row, Mapping):
            unreadable += 1
            continue
        code = str(row.get("contract_code", "") or "").strip().upper()
        at = _timestamp(row.get("funding_time"))
        if at <= 0.0 or (code and code != wanted):
            unreadable += 1
            continue
        records.append(
            FundingRecord(
                at=at,
                funding_rate=_rate(row.get("funding_rate")),
                premium_index=_rate(row.get("avg_premium_index")),
                realized_rate=_rate(row.get("realized_rate")),
            )
        )
    return FundingPage(
        total_size=int(_number(data.get("total_size"))),
        total_page=int(_number(data.get("total_page"))),
        served=len(rows),
        records=tuple(records),
        unreadable=unreadable,
    )



def fetch_contracts(
    *,
    fetch: Optional[Callable[[str, float], Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    endpoint: str = CONTRACT_INFO_ENDPOINT,
    turnover_endpoint: str = SWAP_TICKERS_ENDPOINT,
) -> List[Contract]:
    """The live USDT swaps, with their cadence and their 24-hour turnover, in one snapshot.

    Two requests, and the second one is allowed to fail: the cadence is what coverage *needs*,
    the turnover only decides which contracts a bounded fetch visits first. A ticker endpoint
    that refuses its answer leaves every ``turnover`` at ``0.0`` and the ranking falls back to
    the alphabet, which is a worse order but not a missing universe.
    """
    transport = fetch or _default_transport
    contracts = parse_contracts(transport(endpoint, float(timeout)))
    try:
        turnover = parse_turnover(transport(turnover_endpoint, float(timeout)))
    except HtxError:
        turnover = {}
    return [
        replace(item, turnover=float(turnover.get(item.contract_code, 0.0))) for item in contracts
    ]


def fetch_turnover(
    *,
    fetch: Optional[Callable[[str, float], Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    endpoint: str = SWAP_TICKERS_ENDPOINT,
) -> Dict[str, float]:
    """24-hour quote turnover per contract, in dollars, straight from ``batch_merged``."""
    transport = fetch or _default_transport
    return parse_turnover(transport(endpoint, float(timeout)))


def rank_contracts(contracts: Sequence[Contract]) -> List[Contract]:
    """The universe by dollars traded, largest first, with the code breaking ties.

    Ranking is not a strategy: it answers "which contracts can absorb a position at all", which
    is the one question a bounded crawl has to answer before it can decide what to fetch. The
    turnover is quoted in USDT, the same unit the next session's ``capacity >= $50k`` floor
    reads, so no conversion is being hidden here.
    """
    return sorted(contracts, key=lambda item: (-float(item.turnover), item.contract_code))


def funding_dir(cache: Any, contract: str) -> Path:
    """The one directory a contract's funding cache lives in."""
    code = str(contract).strip().upper()
    if not code:
        raise HtxError("funding: a contract code is required to name a cache directory")
    return Path(cache) / code


def fetch_funding(
    contract: str,
    *,
    period_seconds: float = 0.0,
    page_size: int = DEFAULT_PAGE_SIZE,
    fetch: Optional[Callable[[str, float], Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    delay: float = POLITE_DELAY,
    endpoint: str = FUNDING_ENDPOINT,
) -> FundingSeries:
    """Walk every page the venue claims, newest first, and return the history oldest first.

    The walk is driven by ``total_page``, not by the requested width, because HTX clamps
    ``page_size=200`` to 100 rows and computes ``total_page`` against 100. A walk that trusted its
    own request would stop at page 34 of 67 and call half of BTC's history complete — the exact
    failure this loop is written against.

    Duplicates are dropped by ``funding_time``, a page that repeats rows already read stops the
    walk with a warning rather than looping, and the record count is compared against the
    ``total_size`` the venue claimed. A shortfall is reported as a hole; it is never filled.
    """
    code = str(contract).strip().upper()
    if not code:
        raise HtxError("funding: a contract code is required")
    # A caller who did not say how often this contract settles still gets hole detection, but the
    # step is the venue's most common one and the series says so: measuring a 1-hour contract
    # against an 8-hour step would turn eight real payments into one invented hole.
    declared = float(period_seconds)
    assumed = declared <= 0.0
    if assumed:
        declared = DEFAULT_PERIOD_SECONDS
    transport = fetch or _default_transport
    records: Dict[float, FundingRecord] = {}
    warnings: List[str] = []
    total_size = 0
    total_page = 1
    served = 0
    unreadable = 0
    page = 1
    while True:
        url = (
            f"{endpoint}?contract_code={code}"
            f"&page_index={int(page)}&page_size={int(page_size)}"
        )
        answer = parse_funding(transport(url, float(timeout)), code)
        if page == 1:
            total_size = max(answer.total_size, 0)
            served = answer.served
            total_page = answer.total_page
            if total_page <= 0 and total_size > 0 and served > 0:
                total_page = int(math.ceil(total_size / float(served)))
                warnings.append(
                    f"the venue claimed {total_size} record(s) and no page count, so the walk "
                    f"derived {total_page} page(s) of {served} row(s) from the two"
                )
            total_page = max(total_page, 1)
            if served and served < int(page_size):
                warnings.append(
                    f"page_size={int(page_size)} was clamped to {served} row(s) — the venue "
                    f"serves at most {SERVED_PAGE_CAP} — which is why the walk follows "
                    "total_page rather than the requested width"
                )
            if not answer.records and total_size <= 0:
                warnings.append(f"the venue published no funding history for {code}")
                break
        unreadable += answer.unreadable
        fresh = 0
        for record in answer.records:
            if record.at in records:
                continue
            records[record.at] = record
            fresh += 1
        if page > 1 and answer.records and not fresh:
            warnings.append(
                f"page {page} of {code} repeated rows that were already read, so the walk "
                f"stopped at {len(records)} record(s) instead of looping the venue"
            )
            break
        if page >= total_page:
            break
        if delay > 0.0:
            time.sleep(float(delay))
        page += 1
    ordered = [records[at] for at in sorted(records)]
    if total_size > len(ordered):
        warnings.append(
            f"the venue claimed {total_size} funding record(s) and {len(ordered)} came back: "
            "the shortfall is a hole in the history, not a rounded total"
        )
    if unreadable:
        warnings.append(
            f"{unreadable} funding row(s) carried no readable timestamp or another contract's "
            "code and were dropped rather than guessed at"
        )
    if assumed:
        warnings.append(
            f"the contract's own settlement period was not supplied, so holes were measured "
            f"against the venue's most common {period_name(declared)} step"
        )
    series = FundingSeries(
        contract=code,
        period_seconds=declared,
        records=ordered,
        source=f"htx {code} funding {total_page} page(s) x {served} row(s)",
        warnings=warnings,
    )
    observed = series.observed_step()
    if observed:
        drift = abs(float(observed) - declared)
        if drift > declared * GAP_TOLERANCE:
            series.warnings.append(
                f"the venue declared a {period_name(declared)} cadence and the records sit "
                f"{float(observed):g}s apart: holes are measured against the declared period, "
                "and the contract's funding period may have changed mid-history"
            )
    return series


# ── the cache: bytes with a name, so a report can be reproduced ────────────


def build_manifest(
    series: FundingSeries, *, sha256: str, fetched_at: str = ""
) -> Dict[str, Any]:
    """The provenance of one CSV: what the venue claimed, what was written, and its hash.

    ``fetched_at`` is the only field that moves between two fetches of identical data, and it is
    excluded from ``data_sha256`` on purpose: "these are the same records" and "these were
    fetched at the same moment" are different claims, and a test needs to be able to assert the
    first one after a refetch.
    """
    manifest: Dict[str, Any] = {
        "schema": int(MANIFEST_SCHEMA),
        "contract": series.contract,
        "records": len(series.records),
        "reported": series.reported,
        "unreported": series.unreported,
        "first": series.first,
        "last": series.last,
        "first_utc": _utc(series.first),
        "last_utc": _utc(series.last),
        "period": period_name(series.step()),
        "period_seconds": float(series.step()),
        "declared_period_seconds": float(series.period_seconds),
        "gaps": series.gap_count,
        "max_gap_periods": series.max_gap_periods,
        "file": FUNDING_FILE,
        "sha256": str(sha256),
        "source": series.source,
        "warnings": list(series.warnings),
        "fetched_at": str(fetched_at) or _utc(time.time()),
    }
    manifest["data_sha256"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """SHA-256 of the manifest minus its provenance time — identical bytes give the same hash.

    The canonical form is the same one :mod:`agent.backtest.criteria` uses for its own documents
    (sorted keys, no spaces), so the two hashes in a report can be compared by eye and by script.
    """
    payload = {
        str(key): value
        for key, value in manifest.items()
        if key not in ("fetched_at", "data_sha256")
    }
    canonical = json.dumps(
        _data_plain(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_manifest(path: Any) -> Dict[str, Any]:
    """A manifest read from disk, or :class:`HtxError` saying which file is unreadable."""
    file = Path(path)
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HtxError(f"the funding manifest at {file} cannot be read ({exc})") from exc
    if not isinstance(payload, Mapping):
        raise HtxError(f"the funding manifest at {file} is not an object")
    return {str(key): value for key, value in payload.items()}


def _rate_cell(value: Optional[float]) -> str:
    """One rate as the CSV spells it: ``repr`` for a reported value, an empty cell for none."""
    return "" if value is None else repr(float(value))


def write_funding(
    folder: Any, series: FundingSeries, *, fetched_at: str = ""
) -> Dict[str, Any]:
    """Write ``funding.csv`` and ``manifest.json`` for one series, atomically, and return the manifest.

    Why ``repr`` rather than a format string: it is the shortest text that parses back to the same
    double, so :func:`load_funding` returns the numbers the venue published and not the ones that
    survived four decimal places. Why the temp-then-rename: a crawl that is interrupted halfway
    through a page must leave either the old cache or the new one, never a truncated CSV whose
    hash no longer describes it.
    """
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    csv_path = target / FUNDING_FILE
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(FUNDING_COLUMNS))
        for record in series.records:
            writer.writerow(
                [
                    int(record.at) if float(record.at).is_integer() else repr(float(record.at)),
                    _rate_cell(record.funding_rate),
                    _rate_cell(record.premium_index),
                    _rate_cell(record.realized_rate),
                ]
            )
    temporary.replace(csv_path)
    sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = build_manifest(series, sha256=sha256, fetched_at=fetched_at)
    manifest_path = target / FUNDING_MANIFEST
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    manifest_temporary.replace(manifest_path)
    return manifest


def load_funding(path: Any, *, contract: str = "") -> FundingSeries:
    """Read a CSV back into a series, exactly as :func:`write_funding` wrote it.

    An empty cell stays ``None`` — the round trip is the whole point, because a cache that turned
    "the venue published nothing" into ``0.0`` would silently manufacture an eight-hourly zero
    for half the universe. Rows are sorted and de-duplicated by timestamp, and anything the file
    cannot explain is counted in a warning rather than dropped in silence.
    """
    file = Path(path)
    if not file.is_file():
        raise HtxError(f"the funding cache at {file} does not exist")
    code = str(contract).strip().upper() or file.parent.name.upper()
    records: Dict[float, FundingRecord] = {}
    unreadable = 0
    repeated = 0
    with file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in FUNDING_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise HtxError(
                f"the funding cache at {file} has no {', '.join(missing)} column: it was not "
                "written by agent.backtest.carry"
            )
        for row in reader:
            at = _timestamp(row.get("at"))
            if at <= 0.0:
                unreadable += 1
                continue
            if at in records:
                repeated += 1
            records[at] = FundingRecord(
                at=at,
                funding_rate=_rate(row.get("funding_rate")),
                premium_index=_rate(row.get("premium_index")),
                realized_rate=_rate(row.get("realized_rate")),
            )
    warnings: List[str] = []
    if unreadable:
        warnings.append(f"{unreadable} cached row(s) had no readable timestamp and were dropped")
    if repeated:
        warnings.append(f"{repeated} cached row(s) repeated a timestamp and the later one was kept")
    return FundingSeries(
        contract=code,
        records=[records[at] for at in sorted(records)],
        source=f"csv {file}",
        warnings=warnings,
    )


def _manifest_problem(manifest: Mapping[str, Any], csv_path: Path, contract: str) -> str:
    """Why the cache beside this manifest must not be trusted, or an empty string when it can be."""
    if int(_number(manifest.get("schema"))) != MANIFEST_SCHEMA:
        return (
            f"the manifest names schema {manifest.get('schema')!r} and this code reads "
            f"{MANIFEST_SCHEMA}"
        )
    if str(manifest.get("contract", "") or "").strip().upper() != contract:
        return (
            f"the manifest names {manifest.get('contract')!r} and the directory is {contract}"
        )
    recorded = str(manifest.get("sha256", "") or "").strip().lower()
    if not recorded:
        return "the manifest records no sha256, so the bytes it describes are not identified"
    actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if actual != recorded:
        return (
            f"the CSV hashes to {actual[:12]} and the manifest recorded {recorded[:12]}: the "
            "bytes changed after they were described"
        )
    if str(manifest.get("data_sha256", "") or "") and manifest_hash(manifest) != str(
        manifest.get("data_sha256")
    ):
        return "the manifest's own data_sha256 does not match the fields it hashes"
    return ""


def fetch_funding_cached(
    contract: str,
    *,
    cache: Any = DEFAULT_CACHE,
    refresh: bool = False,
    period_seconds: float = 0.0,
    fetch: Optional[Callable[[str, float], Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    delay: float = POLITE_DELAY,
) -> FundingSeries:
    """A funding history from the cache when its manifest still describes the bytes, else fetched.

    This is the function a report reads, and it is deliberately the only one: a second call with a
    valid cache performs **no request at all**, and the series it returns is read back from the CSV
    rather than kept in memory, so the numbers a report prints are provably the numbers on disk.

    A cache is reused only if its manifest names the same contract, carries this schema, and its
    ``sha256`` matches the file. Anything else is refetched and the reason is appended to the new
    series' warnings — a stale cache is reported, never silently consulted or silently deleted.
    """
    code = str(contract).strip().upper()
    if not code:
        raise HtxError("funding: a contract code is required")
    folder = funding_dir(cache, code)
    csv_path = folder / FUNDING_FILE
    manifest_path = folder / FUNDING_MANIFEST
    stale = ""
    if not refresh and csv_path.is_file() and manifest_path.is_file():
        try:
            manifest = load_manifest(manifest_path)
        except HtxError as exc:
            stale = str(exc)
        else:
            stale = _manifest_problem(manifest, csv_path, code)
            if not stale:
                series = load_funding(csv_path, contract=code)
                series.period_seconds = float(
                    manifest.get("period_seconds") or period_seconds or 0.0
                )
                series.source = (
                    f"cache {csv_path} (sha256 {str(manifest.get('sha256'))[:12]})"
                )
                series.warnings.extend(str(item) for item in manifest.get("warnings") or [])
                return series
    series = fetch_funding(
        code,
        period_seconds=float(period_seconds),
        fetch=fetch,
        timeout=float(timeout),
        delay=float(delay),
    )
    if stale:
        series.warnings.append(f"the cache at {folder} was not reused: {stale}")
    write_funding(folder, series)
    sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    cached = load_funding(csv_path, contract=code)
    cached.period_seconds = float(series.period_seconds)
    cached.source = f"cache {csv_path} (sha256 {sha256[:12]})"
    cached.warnings = list(series.warnings)
    return cached


# ── the perp bars, and the join that says whether a hedge was even on a grid ──


def parse_swap_klines(payload: Any, contract: str, *, period: str = "4hour") -> CandleSeries:
    """One answer from the perp kline endpoint, oldest first, with ``trade_turnover`` as volume.

    ``vol`` on this feed is ``amount * 1000`` — measured on BTC-USDT 4-hour: ``amount=2001.488``,
    ``vol=2001488``, ``trade_turnover=1.383708186328E8`` near 69 100 — so it is a count and not
    dollars. :class:`agent.backtest.data.Candle` means *quote* turnover by ``volume``, and the
    only field that answers that on this endpoint is ``trade_turnover``. Where it is missing the
    parser derives ``amount * close`` and says so, rather than recording a thousand-fold number.
    """
    rows = _answer(payload, f"swap klines {contract}")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise HtxError(f"swap klines {contract}: expected a list of bars")
    code = str(contract).strip().upper()
    candles: Dict[float, Candle] = {}
    derived = 0
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        at = _timestamp(row.get("id"))
        if at <= 0.0:
            continue
        close = _number(row.get("close"))
        turnover = _number(row.get("trade_turnover"))
        if turnover <= 0.0:
            derived += 1
            turnover = _number(row.get("amount")) * close
        candles[at] = Candle(
            at=at,
            open=_number(row.get("open")),
            high=_number(row.get("high")),
            low=_number(row.get("low")),
            close=close,
            volume=turnover,
            trades=int(_number(row.get("count"))),
        )
    warnings: List[str] = []
    if derived:
        warnings.append(
            f"{derived} bar(s) carried no trade_turnover and the volume was derived as "
            "amount * close; the base volume in ``vol`` is not quote turnover on this feed"
        )
    return CandleSeries(
        symbol=code,
        period=str(period),
        rows=[candles[at] for at in sorted(candles)],
        source=f"htx swap klines {code} {period}",
        warnings=warnings,
    )


def fetch_swap_klines(
    contract: str,
    *,
    period: str = "4hour",
    size: int = KLINE_SIZE,
    fetch: Optional[Callable[[str, float], Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    endpoint: str = SWAP_KLINE_ENDPOINT,
) -> CandleSeries:
    """One page of perp bars — and only one, because the venue offers no second page.

    Measured: ``from``/``to`` are ignored (a request for 2020-09 .. 2021-01 answered with the
    newest bars) and ``size=5000`` answers an empty ``data`` list, so the honest way to ask for
    more history is to say that it cannot be had. The cut is a warning on the series, never a
    silent loop over a window the venue never honoured.
    """
    code = str(contract).strip().upper()
    if not code:
        raise HtxError("swap klines: a contract code is required")
    transport = fetch or _default_transport
    url = f"{endpoint}?contract_code={code}&period={period}&size={int(size)}"
    series = parse_swap_klines(transport(url, float(timeout)), code, period=str(period))
    if not series.rows:
        raise HtxError(
            f"the perp kline endpoint served no bar for {code} at {period}; it refuses "
            f"size={int(size)} and ignores from/to, so a larger window cannot be requested"
        )
    if len(series.rows) < int(size):
        series.warnings.append(
            f"{int(size)} bar(s) of {period} were asked for and {len(series.rows)} came back: "
            "the perp feed has no second page, so this is the whole window it will serve"
        )
    return series


def fetch_swap_klines_cached(
    contract: str,
    *,
    cache: Any = "data/carry/klines",
    refresh: bool = False,
    period: str = "4hour",
    size: int = KLINE_SIZE,
    fetch: Optional[Callable[[str, float], Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> CandleSeries:
    """The perp bars from a CSV when one exists, else fetched and written — same contract as funding."""
    code = str(contract).strip().upper()
    folder = funding_dir(cache, code)
    file = folder / f"klines-{period}.csv"
    if not refresh and file.is_file():
        series = _data.load_csv(file, symbol=code, period=str(period))
        series.source = f"cache {file}"
        return series
    series = fetch_swap_klines(
        code, period=str(period), size=int(size), fetch=fetch, timeout=float(timeout)
    )
    _data.write_csv(series, file)
    series.source = f"cache {file}"
    return series


@dataclass(frozen=True)
class Alignment:
    """How well a perp series and a spot series share a clock — and what the gap was where they do.

    ``joined`` counts the perp bars whose nearest spot bar sits within
    :data:`ALIGNMENT_TOLERANCE`; the rest are counted on the side they came from. ``worst_offset``
    is the largest distance among the joined ones, in seconds, so "they are on the same grid" is
    a measured claim rather than an assumption.

    The ``basis`` columns are a **sampled** spot-perp spread at a bar open. They are not the
    funding feed's ``avg_premium_index``, which is a premium *averaged over the funding interval*:
    the two are different measurements of different things, and a report that confused them would
    overstate how executable the basis was at any moment.
    """

    contract: str = ""
    perp_bars: int = 0
    spot_bars: int = 0
    joined: int = 0
    perp_only: int = 0
    spot_only: int = 0
    worst_offset_seconds: float = 0.0
    basis_mean_bps: Optional[float] = None
    basis_median_bps: Optional[float] = None
    basis_p95_bps: Optional[float] = None
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "perp_bars": int(self.perp_bars),
            "spot_bars": int(self.spot_bars),
            "joined": int(self.joined),
            "perp_only": int(self.perp_only),
            "spot_only": int(self.spot_only),
            "worst_offset_seconds": float(self.worst_offset_seconds),
            "basis_mean_bps": self.basis_mean_bps,
            "basis_median_bps": self.basis_median_bps,
            "basis_p95_bps": self.basis_p95_bps,
            "note": self.note,
        }


def check_alignment(
    perp: CandleSeries,
    spot: CandleSeries,
    *,
    contract: str = "",
    tolerance: float = ALIGNMENT_TOLERANCE,
) -> Alignment:
    """Join the two series on their bar opens and report what the join cost.

    Nearest-neighbour, two-pointer, one pass over each series: both feeds label a bar by its open,
    so the honest answer is usually an offset of zero, and the tolerance exists only so that a
    one-second rounding is a note rather than a hole. Spot bars that no perp bar came near are
    counted on the spot side, which is what catches a pair whose two markets failed at different
    moments.
    """
    code = str(contract or perp.symbol).strip().upper()
    perp_rows = list(perp.rows)
    spot_rows = list(spot.rows)
    if not perp_rows or not spot_rows:
        return Alignment(
            contract=code,
            perp_bars=len(perp_rows),
            spot_bars=len(spot_rows),
            joined=0,
            perp_only=len(perp_rows),
            spot_only=len(spot_rows),
            note="one of the two series is empty, so nothing could be joined",
        )
    joined = 0
    perp_only = 0
    worst = 0.0
    used: set = set()
    basis: List[float] = []
    index = 0
    for row in perp_rows:
        while index + 1 < len(spot_rows) and abs(spot_rows[index + 1].at - row.at) <= abs(
            spot_rows[index].at - row.at
        ):
            index += 1
        offset = float(row.at) - float(spot_rows[index].at)
        if abs(offset) <= float(tolerance):
            joined += 1
            used.add(index)
            worst = max(worst, abs(offset))
            spot_close = float(spot_rows[index].close)
            if spot_close > 0.0 and float(row.close) > 0.0:
                basis.append((float(row.close) / spot_close - 1.0) * 1e4)
        else:
            perp_only += 1
    ordered = sorted(basis)
    return Alignment(
        contract=code,
        perp_bars=len(perp_rows),
        spot_bars=len(spot_rows),
        joined=joined,
        perp_only=perp_only,
        spot_only=len(spot_rows) - len(used),
        worst_offset_seconds=worst,
        basis_mean_bps=statistics.fmean(ordered) if ordered else None,
        basis_median_bps=statistics.median(ordered) if ordered else None,
        basis_p95_bps=_quantile(ordered, 0.95) if ordered else None,
        note=(
            "the basis here is a sampled spot-perp spread at a bar open; the funding feed's "
            "avg_premium_index is a premium averaged over the interval and is a different "
            "measurement"
        ),
    )


def write_universe(contracts: Sequence[Contract], folder: Any) -> Path:
    """Snapshot the live universe, ranked by turnover, so a report can name what it measured.

    The snapshot is written once and read back by ``--report``; nothing about a coverage report
    should depend on a request made at printing time. Both the ranked view (``universe.json``,
    the one a person reads) and the raw list (``contracts.json``, what the venue said) are kept,
    because "which contracts are there" and "which are worth fetching first" are different
    questions and only the second one is a sort.
    """
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    ranked = rank_contracts(contracts)
    payload: Dict[str, Any] = {
        "read_utc": _utc(time.time()),
        "contracts": len(ranked),
        "turnover_unit": "USDT per 24 h, from batch_merged.trade_turnover",
        "ranked": [item.as_dict() for item in ranked],
    }
    universe_path = target / UNIVERSE_FILE
    universe_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    raw_path = target / CONTRACTS_FILE
    raw_path.write_text(
        json.dumps(
            {"contracts": [item.as_dict() for item in contracts]},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return universe_path


def read_universe(path: Any) -> Dict[str, Any]:
    """The universe snapshot, or ``{}`` when no snapshot exists yet — never a live request."""
    file = Path(path)
    if not file.is_file():
        return {}
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def format_universe(contracts: Sequence[Contract]) -> str:
    """The universe as a table: the code, the cadence the venue declared, and the turnover."""
    header = ("contract", "settles every", "24h turnover USDT", "listed")
    rows = [list(header)]
    for item in rank_contracts(contracts):
        rows.append(
            [
                item.contract_code,
                period_name(item.period_seconds) if item.period_seconds else "unknown",
                f"{item.turnover:,.0f}",
                item.create_date or "-",
            ]
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(rows[0], widths)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows[1:]:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    return "\n".join(lines)


def cached_contracts(cache: Any) -> List[str]:
    """The contracts whose funding cache holds a CSV, alphabetically — the report's input set."""
    folder = Path(cache)
    if not folder.is_dir():
        return []
    return sorted(
        item.name for item in folder.iterdir() if (item / FUNDING_FILE).is_file()
    )


def _cache_is_fresh(cache: Any, contract: str) -> bool:
    """True when a contract's funding cache is complete and matches its own manifest.

    This is the question the CLI asks before deciding whether a ``--fetch`` needs the venue at
    all: with a valid manifest the cadence, the rows and the hash are already on disk, so opening
    a socket to learn them again would turn an offline repeat into a network dependency. Anything
    doubtful — a missing file, an unreadable manifest, a CSV whose bytes are not the ones the
    manifest hashed — answers ``False``, which puts the run back on the fetch path where the
    doubt is resolved by refetching.
    """
    code = str(contract).strip().upper()
    if not code:
        return False
    folder = funding_dir(cache, code)
    csv_path = folder / FUNDING_FILE
    manifest_path = folder / FUNDING_MANIFEST
    if not (csv_path.is_file() and manifest_path.is_file()):
        return False
    try:
        manifest = load_manifest(manifest_path)
    except HtxError:
        return False
    return not _manifest_problem(manifest, csv_path, code)


def summary_line(series: FundingSeries) -> str:
    """One line per fetched contract: what arrived, how long, and what is missing."""
    span = f"{series.span_days:.0f} d" if series.records else "no records"
    holes = series.gap_count
    return (
        f"{series.contract}: {len(series.records)} record(s) over {span} "
        f"({len(series.rates())} reported, {series.unreported} unreported, {holes} hole(s), "
        f"longest {series.max_gap_periods} period(s))"
    )


def _spot_symbol(contract: str) -> str:
    """The spot pair a swap contract hedges against: ``BTC-USDT`` is ``btcusdt``."""
    return str(contract).replace("-", "").replace("_", "").strip().lower()


# ── the product of this session: what the data can and cannot support ──────


@dataclass(frozen=True)
class Coverage:
    """One contract's line in the coverage report — the numbers, and why it may be unusable.

    Every field is a measurement of the *history*, never of a return: ``records`` is how many
    funding announcements exist, ``reported`` how many carry a rate, ``max_gap_periods`` the
    longest stretch the venue published nothing for, and the bps columns are the published rate
    per funding period. ``failures`` is filled by :func:`coverage_of` against
    :data:`COVERAGE_GATE` and is what makes :attr:`usable` a decision that can be re-read rather
    than a feeling.
    """

    contract: str
    period_seconds: float
    records: int
    reported: int
    positive: int
    first_utc: str
    last_utc: str
    span_days: float
    gap_count: int
    max_gap_periods: int
    positive_share: Optional[float]
    mean_bps: Optional[float]
    median_bps: Optional[float]
    p95_bps: Optional[float]
    mean_abs_premium_bps: Optional[float]
    data_sha256: str = ""
    failures: Tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        """Whether the history clears every declared floor — no floor, no row in next session."""
        return not self.failures

    @property
    def years(self) -> float:
        """The covered span in 365-day years, for the table's ``years`` column."""
        return self.span_days / 365.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract": self.contract,
            "period": period_name(self.period_seconds),
            "period_seconds": float(self.period_seconds),
            "records": int(self.records),
            "reported": int(self.reported),
            "unreported": int(self.records - self.reported),
            "reported_share": (self.reported / self.records) if self.records else None,
            "positive_share": self.positive_share,
            "mean_bps": self.mean_bps,
            "median_bps": self.median_bps,
            "p95_bps": self.p95_bps,
            "mean_abs_premium_bps": self.mean_abs_premium_bps,
            "first_utc": self.first_utc,
            "last_utc": self.last_utc,
            "span_days": float(self.span_days),
            "years": self.years,
            "gap_count": int(self.gap_count),
            "max_gap_periods": int(self.max_gap_periods),
            "data_sha256": self.data_sha256,
            "usable": self.usable,
            "failures": list(self.failures),
        }

    def cells(self) -> List[str]:
        """The report row: one column per number that changed a decision, and nothing else."""
        share = "-" if self.reported == 0 else f"{self.reported / self.records:.0%}"
        positive = "-" if self.positive_share is None else f"{self.positive_share:.0%}"
        return [
            self.contract,
            period_name(self.period_seconds),
            f"{self.records}",
            share,
            f"{self.years:.1f}",
            f"{self.max_gap_periods}",
            f"{self.gap_count}",
            positive,
            _number_text(self.mean_bps),
            _number_text(self.median_bps),
            _number_text(self.p95_bps),
            _number_text(self.mean_abs_premium_bps),
            "usable" if self.usable else "unusable",
        ]


def _number_text(value: Optional[float]) -> str:
    """A bps number as the table prints it, with ``-`` for "the venue told us nothing"."""
    return "-" if value is None else f"{value:+.1f}"


def _data_plain(value: Any) -> Any:
    """The value as JSON will accept it: non-finite floats keep their own spelling.

    The report is written with ``allow_nan=False`` so that a corrupted number fails the writer
    instead of producing a file no other tool can read; a legitimate ``inf`` (a percentile of an
    absurd rate) is spelled rather than clamped, for the same reason
    :func:`agent.backtest.criteria._jsonable` does it.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(key): _data_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_data_plain(item) for item in value]
    return value


def coverage_of(
    series: FundingSeries, *, data_sha256: str = "", gate: Optional[Mapping[str, float]] = None
) -> Coverage:
    """Measure one series against the declared floors, and name every floor it misses.

    The failures are sentences rather than a boolean because a reader of the JSON must be able
    to see *which* condition a contract missed. Nothing here is a judgement about profitability:
    a contract that fails is one whose history cannot be priced, not one whose carry is thin.
    """
    floors = dict(COVERAGE_GATE if gate is None else gate)
    min_days = float(floors.get("min_days", 0.0))
    min_share = float(floors.get("min_reported_share", 0.0))
    max_gap = float(floors.get("max_gap_periods", 0.0))
    failures: List[str] = []
    if series.records and series.span_days < min_days:
        failures.append(
            f"the history spans {series.span_days:.0f} day(s) and the declared floor is "
            f"{min_days:.0f}"
        )
    share = series.reported_share
    if not series.records:
        failures.append("the series holds no record at all")
    elif share is not None and share < min_share:
        failures.append(
            f"the venue reported a rate for {share:.1%} of the periods and the declared floor "
            f"is {min_share:.0%}"
        )
    if series.max_gap_periods > max_gap:
        failures.append(
            f"the longest hole is {series.max_gap_periods} funding period(s) and the declared "
            f"ceiling is {max_gap:g}"
        )
    return Coverage(
        contract=series.contract,
        period_seconds=series.step(),
        records=len(series.records),
        reported=series.reported,
        positive=sum(1 for rate in series.rates() if rate > 0.0),
        first_utc=_utc(series.first),
        last_utc=_utc(series.last),
        span_days=series.span_days,
        gap_count=series.gap_count,
        max_gap_periods=series.max_gap_periods,
        positive_share=series.positive_share,
        mean_bps=series.mean_bps,
        median_bps=series.median_bps,
        p95_bps=series.p95_bps,
        mean_abs_premium_bps=series.mean_abs_premium_bps,
        data_sha256=str(data_sha256),
        failures=tuple(failures),
    )


@dataclass
class CoverageReport:
    """The roster: which contracts have a funding history worth measuring, and why not.

    This is the session's entire output. It is not a ranking of opportunities and it carries no
    return column, because a contract that reaches the top of a coverage table has proven only
    that the venue published its funding for long enough to be priced later. ``gate`` is copied
    into the report so that a reader can see the floor the rows were measured against without
    opening ``criteria.json``; the hash of that criterion lives beside the numbers it produced.
    """

    coverages: List[Coverage] = field(default_factory=list)
    universe: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    gate: Dict[str, float] = field(default_factory=lambda: dict(COVERAGE_GATE))
    criteria_sha256: str = ""

    def __len__(self) -> int:
        return len(self.coverages)

    @property
    def usable(self) -> List[Coverage]:
        """The rows that clear the gate, longest history first — the next session's input."""
        return sorted(
            (row for row in self.coverages if row.usable),
            key=lambda row: (-row.span_days, row.contract),
        )

    @property
    def unusable(self) -> List[Coverage]:
        """The rows that do not, with the reason each one fails carried on the row itself."""
        return [row for row in self.coverages if not row.usable]

    @property
    def longest_days(self) -> float:
        """The longest history in the report, in days — one number for "how much time is there"."""
        return max((row.span_days for row in self.coverages), default=0.0)

    def totals(self) -> Dict[str, Any]:
        """Sums and maxima over the rows, and explicitly nothing that averages rates across them.

        Pooling a 1-hour contract's rates with an 8-hour contract's would produce a number that
        is neither one's rate per day nor per period, so the totals stop at counts: they answer
        "how much data is there", never "what did it pay".
        """
        reported = sum(row.reported for row in self.coverages)
        positive = sum(row.positive for row in self.coverages)
        return {
            "contracts": len(self.coverages),
            "usable": len(self.usable),
            "unusable": len(self.unusable),
            "records": sum(row.records for row in self.coverages),
            "reported": reported,
            "unreported": sum(row.records - row.reported for row in self.coverages),
            "reported_share": (reported / sum(row.records for row in self.coverages))
            if self.coverages and sum(row.records for row in self.coverages)
            else None,
            "positive": positive,
            "positive_share": (positive / reported) if reported else None,
            "gaps": sum(row.gap_count for row in self.coverages),
            "max_gap_periods": max((row.max_gap_periods for row in self.coverages), default=0),
            "longest_days": self.longest_days,
            "longest_years": self.longest_days / 365.0,
            "rates_averaged_across_contracts": False,
        }

    def verdict(self) -> str:
        """One sentence that says what the table proved and, in the same breath, what it did not."""
        totals = self.totals()
        if not self.coverages:
            return (
                "Nothing was measured: no contract reached this report, so this file has no "
                "opinion about any funding history."
            )
        share = totals["positive_share"]
        positive = "no reported rate" if share is None else f"{share:.1%} of the reported periods"
        return (
            f"{totals['contracts']} contract(s) read; {totals['usable']} clear the declared floor. "
            f"The longest history is {totals['longest_years']:.1f} year(s), the venue published a "
            f"rate for {totals['reported']} of {totals['records']} periods, funding was above zero "
            f"in {positive}, and {totals['gaps']} hole(s) remain (longest "
            f"{totals['max_gap_periods']} period(s)). This is coverage, not carry: no APY, no net "
            "return, and no dollar of profit is claimed by this report."
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gate": dict(sorted(self.gate.items())),
            "acceptance": list(COVERAGE_ACCEPTANCE),
            "does_not_show": list(DOES_NOT_SHOW),
            "criteria_sha256": self.criteria_sha256,
            "universe": _data_plain(self.universe),
            "totals": self.totals(),
            "verdict": self.verdict(),
            "coverages": [row.as_dict() for row in self.coverages],
            "warnings": list(self.warnings),
        }


#: The table's columns, in the order the numbers are read: what the contract is, what the venue
#: published, how much of it, how long, what is missing, and what the rate was per period.
TABLE_HEADER: Tuple[str, ...] = (
    "contract",
    "period",
    "records",
    "reported",
    "years",
    "max gap",
    "gaps",
    "funding>0",
    "mean bps",
    "median bps",
    "p95 bps",
    "|prem| bps",
    "status",
)


def format_table(report: CoverageReport, *, limit: Optional[int] = None) -> str:
    """The report as a person reads it: usable rows first, then the rows that failed and why.

    ``limit`` cuts the table, never the report: the JSON keeps every row, because a caller who
    asked for the top twenty contracts must still be able to read why the other forty were not
    in them. The closing lines repeat the verdict and the criteria hash, so a printed table
    carries the same caveat as the file it came from.
    """
    ordered = report.usable + report.unusable
    if limit is not None:
        ordered = ordered[: max(int(limit), 0)]
    rows: List[List[str]] = [list(TABLE_HEADER)] + [row.cells() for row in ordered]
    widths = [max(len(row[index]) for row in rows) for index in range(len(TABLE_HEADER))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(rows[0], widths)).rstrip()]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows[1:]:
        lines.append("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    for row in report.unusable:
        if row in ordered:
            lines.append(f"  {row.contract}: {'; '.join(row.failures)}")
    lines.append("")
    lines.append(report.verdict())
    for warning in report.warnings:
        lines.append(f"note: {warning}")
    if report.criteria_sha256:
        lines.append(f"criteria sha256 {report.criteria_sha256} (the floor was declared, not fitted)")
    lines.append("bps are per funding period; not one number here is a return.")
    return "\n".join(lines)


def coverage_criteria(gate: Optional[Mapping[str, float]] = None) -> Criteria:
    """The declaration this report is written against, as a :class:`Criteria`.

    The same object the rest of the layer hashes: the floors in ``gate``, the acceptance
    sentences, and :data:`DOES_NOT_SHOW`. Building it here rather than inside
    :func:`write_coverage` is what lets the report print its hash *and* write the identical
    ``criteria.json`` beside it — two files, one declaration, one hash.
    """
    return Criteria(
        gate=dict(COVERAGE_GATE if gate is None else gate),
        note=(
            "the coverage floor: how much *published* funding history a contract needs before "
            "the next session may price it — declared before the first fetch and never a "
            "profitability criterion"
        ),
        acceptance=COVERAGE_ACCEPTANCE,
        does_not_show=DOES_NOT_SHOW,
    )


def report_of(
    series: Sequence[FundingSeries],
    *,
    hashes: Optional[Mapping[str, str]] = None,
    universe: Optional[Mapping[str, Any]] = None,
    gate: Optional[Mapping[str, float]] = None,
    criteria: Optional[Criteria] = None,
    warnings: Sequence[str] = (),
) -> CoverageReport:
    """Measure a batch of series into one report, with the criterion's hash on the cover.

    ``hashes`` maps a contract to the SHA-256 of the CSV its series was read from. It is passed
    in rather than computed here because the hash belongs to the *bytes*, and a series built in
    a test has no bytes: a row whose hash is empty says the numbers were not read from a file,
    which is exactly what a reader needs to know before quoting them.
    """
    floors = dict(COVERAGE_GATE if gate is None else gate)
    declaration = criteria or coverage_criteria(floors)
    measured = [
        coverage_of(item, data_sha256=(hashes or {}).get(item.contract, ""), gate=floors)
        for item in series
    ]
    measured.sort(key=lambda row: row.contract)
    return CoverageReport(
        coverages=measured,
        universe=dict(universe or {}),
        warnings=[str(item) for item in warnings],
        gate=floors,
        criteria_sha256=declaration.fingerprint(),
    )


def write_coverage(
    path: Any, report: CoverageReport, *, criteria: Optional[Criteria] = None
) -> Dict[str, Path]:
    """Write the report and its declaration, then read the report back and verify the hash.

    The verify step is not ceremony: it is the one thing that proves the file on disk is the
    file this process meant to write, and it costs one read. ``criteria.json`` sits beside the
    report so that the floor can be read without the numbers and the numbers without the prose.

    The bytes are written here rather than through :meth:`Criteria.write` only because the
    coverage block has to carry the same hash, and the call is otherwise the same one: sorted
    keys, two-space indent, no infinities, one trailing newline. Two runs over the same cache
    therefore produce the same file, which is what lets a test compare them byte for byte.
    """
    declaration = criteria or coverage_criteria(report.gate)
    target = Path(path)
    report.criteria_sha256 = declaration.fingerprint()
    document = declaration.document({"coverage": report.as_dict()})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    criteria_path = declaration.write(target.with_name("criteria.json"))
    problem = verify(json.loads(target.read_text(encoding="utf-8")))
    if problem:
        raise HtxError(f"the coverage report at {target} does not verify: {problem}")
    return {"json": target, "criteria": criteria_path}


# ── the command line: fetch, measure, or check one join ────────────────────


def _parser() -> argparse.ArgumentParser:
    """The verbs this file has. None of them computes a return, and the help says so."""
    parser = argparse.ArgumentParser(
        prog="python -m agent.backtest.carry",
        description=(
            "Collect HTX USDT funding histories, cache them beside a hash, and report what "
            "coverage the data actually has. No APY and no signal: this file measures the roster."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  --contracts                   the live USDT universe, ranked by 24h turnover\n"
            "  --fetch BTC-USDT              cache one contract's funding history\n"
            "  --fetch-top 60 --klines       cache the sixty largest, funding and one bar page\n"
            "  --report --top 60             measure the cache into data/carry/coverage.json\n"
            "  --join BTC-USDT               are the perp bars and the spot bars on one grid?\n"
        ),
    )
    parser.add_argument(
        "--contracts",
        action="store_true",
        help="print the live USDT perp universe, ranked by 24h turnover, and snapshot it",
    )
    parser.add_argument(
        "--fetch",
        action="append",
        default=[],
        metavar="CONTRACT",
        help="cache this contract's funding history (repeatable), e.g. --fetch BTC-USDT",
    )
    parser.add_argument(
        "--fetch-top",
        type=int,
        default=0,
        metavar="N",
        help="cache the N contracts with the largest 24h turnover, in that order",
    )
    parser.add_argument(
        "--klines",
        action="store_true",
        help="with --fetch/--fetch-top, also cache one page of perp bars per contract",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="measure every cached contract and write the coverage report and its criteria",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="with --report, keep the N longest histories in the report",
    )
    parser.add_argument(
        "--join",
        default="",
        metavar="CONTRACT",
        help="join one contract's perp bars against its spot pair and report the offset",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, metavar="PATH", help=f"the coverage report ({DEFAULT_OUT})"
    )
    parser.add_argument(
        "--cache", default=DEFAULT_CACHE, metavar="DIR", help=f"the funding cache ({DEFAULT_CACHE})"
    )
    parser.add_argument(
        "--klines-cache",
        default=DEFAULT_KLINES,
        metavar="DIR",
        help=f"where --klines keeps bars ({DEFAULT_KLINES})",
    )
    parser.add_argument(
        "--universe",
        default=DEFAULT_UNIVERSE,
        metavar="PATH",
        help=f"the universe snapshot ({DEFAULT_UNIVERSE}), written by --contracts, read by --report",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="refetch even when a valid cache exists"
    )
    parser.add_argument("--quiet", action="store_true", help="write the files, print nothing")
    parser.add_argument("--json", action="store_true", help="also print the measurement as JSON")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help="per-request timeout",
    )
    return parser



def _join(args: argparse.Namespace, klines_cache: Path) -> int:
    """Fetch one pair's perp bars and its spot pair, and print what the join measured.

    The spot leg comes from :func:`agent.backtest.data.fetch_klines`, which is the package's one
    spot reader and is capped at :data:`agent.backtest.data.MAX_BARS` exactly like the perp page —
    so the two sides are always compared over the window both feeds will actually serve.
    """
    code = str(args.join).strip().upper()
    perp = fetch_swap_klines_cached(
        code, cache=klines_cache, refresh=bool(args.refresh), timeout=float(args.timeout)
    )
    days = (len(perp.rows) * perp.interval) / 86400.0 if perp.rows else 2.0
    spot = fetch_klines(
        _spot_symbol(code),
        period=perp.period or "4hour",
        days=max(days, 2.0),
        timeout=float(args.timeout),
    )
    alignment = check_alignment(perp, spot, contract=code)
    if not args.quiet:
        print(
            f"{alignment.contract}: {alignment.joined} of {alignment.perp_bars} perp bar(s) "
            f"joined with {alignment.spot_bars} spot bar(s), worst offset "
            f"{alignment.worst_offset_seconds:g}s; perp-only {alignment.perp_only}, "
            f"spot-only {alignment.spot_only}"
        )
        print(
            f"   basis at bar open: mean {_number_text(alignment.basis_mean_bps)} bps, "
            f"median {_number_text(alignment.basis_median_bps)} bps, "
            f"p95 {_number_text(alignment.basis_p95_bps)} bps"
        )
        print(f"   note: {alignment.note}")
    if args.json:
        print(json.dumps(alignment.as_dict(), indent=2, allow_nan=False))
    return 0


def _report(args: argparse.Namespace, cache: Path) -> int:
    """Measure what the cache holds, write the report, and never touch a network.

    Every series is read through :func:`fetch_funding_cached` with ``refresh=False``: a valid
    manifest makes that call request-free, and the hash that reaches the report is the CSV's own.
    A cache with nothing in it exits :data:`EXIT_SHORT` rather than writing an empty report that
    would read as "measured, and nothing qualified".
    """
    codes = cached_contracts(cache)
    if not codes:
        print(
            f"carry: the cache at {cache} holds no {FUNDING_FILE} to measure — "
            "run --fetch or --fetch-top first",
            file=sys.stderr,
        )
        return EXIT_SHORT
    series_list: List[FundingSeries] = []
    hashes: Dict[str, str] = {}
    warnings: List[str] = []
    for code in codes:
        series = fetch_funding_cached(code, cache=cache, refresh=False, timeout=float(args.timeout))
        series_list.append(series)
        manifest_path = funding_dir(cache, code) / FUNDING_MANIFEST
        if manifest_path.is_file():
            hashes[code] = str(load_manifest(manifest_path).get("sha256") or "")
        for note in series.warnings:
            line = f"{code}: {note}"
            if line not in warnings:
                warnings.append(line)
    report = report_of(
        series_list,
        hashes=hashes,
        universe=read_universe(args.universe),
        warnings=warnings,
    )
    if args.top > 0:
        longest = sorted(report.coverages, key=lambda row: (-row.span_days, row.contract))
        keep = {row.contract for row in longest[: int(args.top)]}
        report.coverages = [row for row in report.coverages if row.contract in keep]
    written = write_coverage(args.out, report)
    if not args.quiet:
        print(format_table(report, limit=int(args.top) if args.top > 0 else None))
        print(f"   wrote {written['json']} · {written['criteria']}")
    if args.json:
        print(json.dumps(json.loads(Path(written["json"]).read_text(encoding="utf-8")), indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Fetch, snapshot, measure, or join — with :mod:`agent.backtest.run`'s exit codes.

    ``EXIT_FEED`` (3) when the venue refused an answer, ``EXIT_USAGE`` (2) when the request did
    not name anything to do, ``EXIT_SHORT`` (4) when there was nothing to measure. A report that
    measured nothing is a short answer and not a zero — that difference is the whole point of a
    coverage table, so it is an exit code as well as a sentence.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if not (args.contracts or args.fetch or args.fetch_top or args.report or args.join):
        parser.print_help()
        return EXIT_USAGE
    cache = Path(args.cache)
    klines_cache = Path(args.klines_cache)
    try:
        universe: List[Contract] = []
        known: Dict[str, Contract] = {}
        # A ``--fetch`` whose every contract already has a valid manifest is answered from disk:
        # the cadence it would ask the venue for is in that manifest, and a repeat run should not
        # need a socket to repeat what it already has. ``--refresh``, ``--fetch-top`` and
        # ``--contracts`` all want something only the venue can say, so those still go out.
        offline = bool(args.fetch) and not (
            args.contracts or args.fetch_top or args.refresh
        ) and all(_cache_is_fresh(cache, name) for name in args.fetch)
        if not offline and (args.contracts or args.fetch_top or args.fetch):
            universe = fetch_contracts(timeout=float(args.timeout))
            known = {item.contract_code: item for item in universe}
            if args.contracts and not args.quiet:
                print(format_universe(universe))
            if args.contracts or args.fetch_top:
                snapshot = write_universe(universe, Path(args.universe).parent)
                if not args.quiet:
                    print(f"   wrote {snapshot}")
        targets: List[Contract] = []
        for name in args.fetch:
            code = str(name).strip().upper()
            if not code:
                print("carry: --fetch needs a contract code", file=sys.stderr)
                return EXIT_USAGE
            targets.append(known.get(code, Contract(contract_code=code)))
        if args.fetch_top > 0:
            targets.extend(rank_contracts(universe)[: int(args.fetch_top)])
        seen: set = set()
        for item in targets:
            if item.contract_code in seen:
                continue
            seen.add(item.contract_code)
            series = fetch_funding_cached(
                item.contract_code,
                cache=cache,
                refresh=bool(args.refresh),
                period_seconds=float(item.period_seconds),
                timeout=float(args.timeout),
            )
            if args.klines:
                bars = fetch_swap_klines_cached(
                    item.contract_code,
                    cache=klines_cache,
                    refresh=bool(args.refresh),
                    timeout=float(args.timeout),
                )
                if not args.quiet:
                    print(f"   {item.contract_code}: {len(bars)} perp bar(s) of {bars.period}")
            if not args.quiet:
                print(summary_line(series))
                for note in series.warnings:
                    print(f"   note: {note}")
            if args.json:
                print(json.dumps(series.as_dict(), indent=2, allow_nan=False))
        if args.join:
            return _join(args, klines_cache)
        if args.report:
            return _report(args, cache)
        if not targets:
            print("carry: nothing to measure; see --help", file=sys.stderr)
            return EXIT_SHORT
        return 0
    except HtxError as exc:
        print(f"carry: {exc}", file=sys.stderr)
        return EXIT_FEED


if __name__ == "__main__":
    raise SystemExit(main())

