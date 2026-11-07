"""The wild market, read through the same two gates: DeFiLlama as a venue.

Every number the funnel has judged so far was *drawn* — ``ProtocolMarket.mock`` for the
seven declared protocols and :mod:`agent.physics.market_generator` for the two hundred.
Drawn numbers are how a threshold is calibrated (the gap between honest and junk is
measured, not assumed), and they are also the limit of the claim: a generator cannot
surprise you, and a gate that has only ever seen its own market has only ever been tested
against its own assumptions. This module hands the same gates a market nobody designed.

What the venue publishes, and what this file has to do about it
==============================================================

``https://yields.llama.fi/pools`` is one document containing every yield pool the index
knows — 1.4 MB, some fifteen thousand rows — each carrying ``chain``, ``project``,
``symbol``, ``tvlUsd``, ``apy`` and a ``pool`` uuid. ``https://yields.llama.fi/chart/{uuid}``
is that pool's daily history. Three properties of the feed decide the shape of this
adapter, and each one is a place a careless adapter would have invented a number:

**A pool has no address.** DeFiLlama identifies a pool by a uuid. The internal
:class:`agent.physics.portfolio.Protocol` needs a 20-byte address, so one is derived with
:func:`agent.physics.portfolio.sim_address` — the same SHA-256 of the name the simulated
market uses. It is *provably not a wallet*, it is stable across runs, and nothing in this
file can move value.

**The history is a series of APYs, not of returns.** A record in this layer is a per-step
rate, because that is what compounds (:func:`agent.physics.portfolio.annualise` inverts
it). The conversion used here is the exact inverse — ``(1 + apy)^(1/365) - 1`` — and it is
declared rather than hidden, because it is the only place where the venue's units become
ours. The shape survives it: the coefficient of variation of the per-step rates is exactly
that of the *log* returns, and differs from ``CV`` of the APY series only by the factor
compounding introduces — measured, four percent at an eight-percent yield — so Statement
#87 is reading the same curve either way.

**APY is empty in places.** ``None`` and ``NaN`` arrive in the middle of otherwise good
series, mostly for young pools. Nothing here fills a gap: a filled point is a number the
market did not publish, and Statement #87 would then be measuring the decorator. Gaps are
dropped, the count is reported, and a pool that is more than
:data:`MAX_MISSING_SHARE` empty is not published at all.

What the funnel does with it
============================

Nothing new. :class:`DiscoveryEngine` assembles the two gates that already exist —
:class:`agent.physics.discovery.Scanner` (mean, spread, ``CV``, turnover, microseconds per
pool) and :func:`agent.physics.risk_assessment.evaluate_opportunity` (``A²``, ``CV``, the
claim against the record) — and runs them over the real market exactly as over the drawn
one. The one rule this file adds is the *window* test the brief asks for: an Entrain
survey cuts a stream into fixations of :data:`agent.physics.sensorium.WINDOW` samples with
a hop of :data:`agent.physics.sensorium.STRIDE`, and below
:data:`agent.physics.curiosity.CURIOUS_MIN_WINDOWS` fixations there is no survey to run.
A pool whose history is that short is ``UNMEASURED`` — not refused, not approved, waiting —
and the reason says how many points produced how many windows.

Failure is a value, not a traceback
===================================

A market feed is a remote service and it will be down. :meth:`DefiLlamaScanner.get_top_pools`
raises :class:`DefiLlamaUnavailable` with a sentence (never a bare ``requests`` traceback),
:meth:`DefiLlamaScanner.fetch_pool_history` returns an empty tuple for one bad pool so a
single dead uuid cannot cost the scan, and :meth:`DiscoveryEngine.run` answers a dead feed
with an empty funnel that carries the reason in ``error``. The CLI — :func:`main` — turns
that into exit code 3 and one line of prose.

It will also be *busy*, and that is a different answer. Measured on the first real
two-hundred-pool run of this file: at eight concurrent requests the host replied ``429`` for
125 of the 200 pools. A scanner that had recorded those as "no history" would have been
reporting its own rate limit as a property of the market — the exact failure mode where an
adapter quietly shrinks the world and the funnel reports a smaller, cleaner market. So a
``429`` is waited out and asked again (:data:`RETRIES`, doubling pauses, four requests in
flight at once) and, if the host keeps asking, that pool is skipped *by name* like any other
unreadable one.

A name is not an address, and the bridge has to say so
=====================================================

The funnel reads *pools*, and a pool is a uuid. A trading loop holds an *address book*, and
an address is a vault. Between the two there is exactly one seam, and it is a **name**:
:class:`FeedHistory` turns a book entry's name (``USDC``, ``Aave V3 USDC (Arbitrum)``) into
the daily APY series of the pool that carries it, so a venue whose own
``YieldDistributed`` logs are silent can still be measured against history the market
actually published. The match is stated and counted, never assumed: four degrees of
evidence (an exact symbol, the whole label, a symbol that carries the name, a name that
carries the symbol), ties broken by depth, and a name nothing carries answers a record with
no series *and* the reason. A uuid can be pinned by the operator instead, which removes the
guesswork entirely — :class:`FeedRecord` then says which pool answered, so a report can print
"priced from the feed" and never confuse it with "the vault paid it".

Nothing here signs, stakes or withdraws: there is no key, no ABI and no transaction in this
file, and :class:`DiscoveryEngine` deliberately has no method that could move a credit.
"""
from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agent.physics.curiosity import CURIOUS_MIN_WINDOWS
from agent.physics.discovery import (
    TURNOVER_FLOOR,
    WATCHLIST_LIMIT,
    Scanner,
    Watchlist,
)
from agent.physics.portfolio import (
    BUCKET_ACTIVE,
    BUCKET_EXPERIMENTAL,
    BUCKET_SAFE,
    MIN_RECORD,
    PortfolioError,
    Protocol,
    sim_address,
)
from agent.physics.risk_assessment import (
    HONEST,
    PLAUSIBLE_APY,
    UNMEASURED,
    Assessment,
    evaluate_opportunity,
)
from agent.physics.sensorium import STRIDE, WINDOW

__all__ = [
    "BACKOFF",
    "CACHE_DIR",
    "CACHE_TTL",
    "CHAIN_NAMES",
    "CHART_URL",
    "DEFAULT_TIMEOUT",
    "FeedHistory",
    "FeedRecord",
    "MAX_MISSING_SHARE",
    "MIN_HISTORY",
    "MIN_TVL_USD",
    "MODE_REAL",
    "MODE_SYNTHETIC",
    "MODES",
    "POOLS_LIMIT",
    "POOLS_URL",
    "RETRIES",
    "STEPS_PER_YEAR",
    "USER_AGENT",
    "VENUE_NAME",
    "WATCH_LIMIT",
    "WHOLE_CATALOGUE",
    "WORKERS",
    "DefiLlamaError",
    "DefiLlamaMarket",
    "DefiLlamaScanner",
    "DefiLlamaUnavailable",
    "DiscoveryEngine",
    "MarketFunnel",
    "PoolFrame",
    "PoolRow",
    "apy_to_rate",
    "bucket_for",
    "chain_name",
    "clean_series",
    "entrain_windows",
    "main",
    "to_returns",
]

# ── the endpoints ───────────────────────────────────────────────────────────

#: Everything the index lists, in one document. The ``pool`` uuid inside each row is what
#: :data:`CHART_URL` is keyed on.
POOLS_URL = "https://yields.llama.fi/pools"

#: One pool's daily history. ``{pool}`` is the ``pool`` field of a :class:`PoolRow`.
CHART_URL = "https://yields.llama.fi/chart/{pool}"

#: Seconds a single request may take. Thirty rather than five: the catalogue is 1.4 MB and
#: measured from this machine it arrives in ten to twenty seconds on a normal link, so a
#: short timeout would report a slow feed as a dead one. It is a *field* rather than a
#: constant inside a call, so an operator or a test can move it without patching code.
DEFAULT_TIMEOUT = 30.0

#: Pools a scan reads when nobody says otherwise — the brief's two hundred, top by TVL.
POOLS_LIMIT = 200

#: Requests in flight at once when a market's histories are fetched. Four, and it was
#: measured the hard way: at eight, the catalogue's own host answers ``429`` for most of a
#: two-hundred-pool scan (measured on this machine: 125 of 200 pools refused), and a scan
#: that outruns the host's patience is a scan that reports somebody else's rate limit as a
#: property of the market. Four is the width where a scan takes a couple of minutes and the
#: feed keeps answering.
WORKERS = 4

#: How many times a ``429`` is retried before the pool is given up on, and the pause
#: between attempts, in seconds (doubling: 2 s, then 4 s). The host is asking to be paced,
#: not refusing service, so the answer is a pause and the same question — and if it keeps
#: asking, that pool is skipped *by name* like any other unreadable one.
RETRIES = 2
BACKOFF = 2.0

#: How long a cached document is served before it is fetched again. Six hours: a daily
#: series does not change within an afternoon, and the notebook is re-run more often than
#: the pools re-price.
CACHE_TTL = 6 * 3600.0

#: Share of a series that may be missing before the pool is refused as unreadable. A
#: quarter, because a record with one gap in four is a record with a different sampling
#: rate rather than a record with a typo in it.
MAX_MISSING_SHARE = 0.25

#: Steps in the venue's year: DeFiLlama publishes a daily number, so one step is a day.
#: Declared here as well as in :mod:`agent.physics.portfolio` on purpose — the number
#: belongs to the *feed's* calendar, and an adapter that changed venues would have to say
#: so rather than inherit a constant that happens to match.
STEPS_PER_YEAR = 365

#: The chain ids this file can spell the way the *feed* spells them, so that a venue which
#: knows it is chain 42161 can ask for ``Arbitrum`` pools and nothing else. Four entries, not
#: a registry: these are the chains the trading venue ships a profile for, and an id that is
#: missing here is answered with ``""`` — "no filter" — rather than with a guess, because a
#: wrong chain filter would price a vault against a pool on somebody else's network.
CHAIN_NAMES: Dict[int, str] = {
    1: "Ethereum",
    137: "Polygon",
    8453: "Base",
    42161: "Arbitrum",
    11155111: "Sepolia",
}


def chain_name(chain_id: Any) -> str:
    """The feed's spelling of a chain id, or ``""`` when this file does not know it."""
    try:
        return CHAIN_NAMES.get(int(chain_id or 0), "")
    except (TypeError, ValueError):
        return ""

#: The fewest points a pool must publish before it is worth a :class:`Protocol` at all:
#: two, because a spread needs two numbers. The scanner has its own, higher floor
#: (:data:`agent.physics.discovery.MIN_SCANNED`) for anything it will vouch for, and the
#: survey has a much higher one; below two there is simply nothing to hand over, and the
#: pool is reported as skipped rather than padded.
MIN_HISTORY = 2

#: Liquidity, in dollars, a pool must hold to have an exit. The scanner's rule about
#: turnover is the same rule — "a protocol nobody trades is a protocol with no exit" — and
#: DeFiLlama publishes *depth* (``tvlUsd``) rather than volume. Depth is what an exit
#: needs, so it is handed to the gate as the turnover field and the floor is stated in
#: dollars rather than in credits: a million, because below that a tenth of the empire
#: cannot leave without moving the price.
MIN_TVL_USD = 1_000_000.0

#: The queue in front of the expensive reading on a real market. The core's own policy
#: (:data:`agent.physics.discovery.WATCHLIST_LIMIT`) unless an operator says otherwise, and
#: the report always prints what the bound cost as ``truncated``.
WATCH_LIMIT = WATCHLIST_LIMIT

#: The two ways this engine can be pointed at a market. ``real_defillama`` is the wild one;
#: ``synthetic`` is the drawn market, kept so that the same loop can be run twice and the
#: difference becomes a measurement instead of a claim.
MODE_REAL = "real_defillama"
MODE_SYNTHETIC = "synthetic"
MODES = (MODE_REAL, MODE_SYNTHETIC)

#: What the adapter calls itself in a log line, in a table and in its `User-Agent`.
VENUE_NAME = "defillama"
USER_AGENT = "enrg-ai/0.1 defillama-adapter"

#: Where fetched documents are kept between runs. Overridable by environment because a
#: container may want its cache on a volume; under ``.cache`` because that is gitignored
#: and a cache is not a deliverable.
CACHE_DIR = ".cache/defillama"


class DefiLlamaError(RuntimeError):
    """The feed could not be read, and the sentence says which part and why."""


class DefiLlamaUnavailable(DefiLlamaError):
    """The feed answered nothing, or nothing usable — a dead service, not a bad pool.

    ``status`` carries the HTTP code when there was one, because *"too many requests"* and
    *"the service is down"* need different answers: the first is waited out and asked
    again, the second is reported. Nothing else distinguishes them.
    """

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def throttled(self) -> bool:
        """True when the host asked to be paced rather than refused service."""
        return self.status == 429


# ── the units: the only place the venue's numbers become ours ───────────────


def apy_to_rate(apy_percent: float, steps_per_year: int = STEPS_PER_YEAR) -> float:
    """One published APY (in per cent) as the per-step rate that compounds to it.

    The exact inverse of :func:`agent.physics.portfolio.annualise`: ``(1 + apy)^(1/365) − 1``
    with ``apy`` as a fraction. Two edge cases are answered rather than raised, because a
    feed is allowed to publish them and a rule that crashes on a feed cannot be run on one:

    * ``apy = -100`` (a total loss, which DeFiLlama does publish) gives ``-1.0`` — every
      credit gone, which is exactly what :func:`agent.physics.portfolio.implied_apy`
      reports for such a record;
    * anything below ``-100`` is arithmetic without a meaning, and is clamped to the total
      loss rather than turned into a complex number.
    """
    growth = 1.0 + float(apy_percent) / 100.0
    if not math.isfinite(growth):
        return 0.0
    if growth <= 0.0:
        return -1.0
    return float(growth ** (1.0 / float(max(int(steps_per_year), 1))) - 1.0)


def to_returns(
    apy_percent: Sequence[float], steps_per_year: int = STEPS_PER_YEAR
) -> np.ndarray:
    """A published APY series as a per-step return record — what the funnel reads.

    Finite by construction: every non-finite input is dropped before the conversion, and
    the conversion itself cannot produce one. That is the promise this function keeps to
    the rest of the layer — :class:`agent.physics.portfolio.Protocol` refuses a record
    containing ``NaN``, and it is better to hand it a shorter honest record than a
    longer one it has to reject.
    """
    values = np.asarray(list(apy_percent), dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(0, dtype=float)
    return np.array([apy_to_rate(float(value), steps_per_year) for value in values])


def clean_series(values: Sequence[Any]) -> Tuple[List[float], int]:
    """``(published, missing)`` — the finite part of a series, in the venue's order.

    A missing APY is not a zero and is not the previous day's number: both would put a
    number into a record that the market never published, and Statement #87 would then be
    measuring the adapter. Gaps are therefore *dropped* and counted, and the count is what
    :data:`MAX_MISSING_SHARE` is compared against.
    """
    published: List[float] = []
    missing = 0
    for value in values:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            missing += 1
            continue
        if not math.isfinite(number):
            missing += 1
            continue
        published.append(number)
    return published, missing


def entrain_windows(points: int) -> int:
    """How many fixations an Entrain survey gets out of ``points`` samples.

    The geometry is the sensorium's and is *not* re-invented here: a fixation is
    :data:`agent.physics.sensorium.WINDOW` samples and consecutive fixations are
    :data:`agent.physics.sensorium.STRIDE` samples apart, so a stream of ``n`` samples
    tiles into ``1 + (n − WINDOW) // STRIDE`` windows and into none at all below one
    window. The cap the survey itself applies
    (:data:`agent.physics.curiosity.SURVEY_FIXATIONS`) is deliberately not applied: this
    number answers "is there a survey to run", not "how long is it", and a long record is
    not made short by the fact that the survey reads the first twenty-four fixations.
    """
    total = int(points)
    if total < int(WINDOW):
        return 0
    return 1 + (total - int(WINDOW)) // int(STRIDE)


# ── one pool, as the index published it ─────────────────────────────────────


def _number(value: Any) -> Optional[float]:
    """A published number, or ``None`` when the feed left the field empty.

    ``None`` and ``NaN`` are the same fact about a feed — *no number here* — and both are
    answered with ``None`` rather than with a zero, because a zero APY is a claim and a
    missing one is not.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class PoolRow:
    """One row of ``/pools``: what the index says about a pool, in our own units.

    Not a :class:`agent.physics.portfolio.Protocol` and deliberately so — this is what the
    venue *advertises* (depth, chain, the current APY), while a ``Protocol`` is what the
    funnel reads (a record and a claim). The history arrives separately, one request per
    pool, and only the pair of them can become a ``Protocol``.
    """

    uuid: str
    project: str
    chain: str
    symbol: str
    tvl_usd: float
    apy: float
    apy_base: Optional[float] = None
    apy_reward: Optional[float] = None
    mean_apy_30d: Optional[float] = None
    volume_usd_1d: Optional[float] = None
    stablecoin: bool = False
    exposure: str = ""
    pool_meta: str = ""

    @property
    def name(self) -> str:
        """The label a report prints: ``project symbol (chain)``."""
        return f"{self.project} {self.symbol} ({self.chain})"

    @classmethod
    def from_payload(cls, row: Mapping[str, Any]) -> "PoolRow":
        """Read one published row, tolerating every field the feed may leave out."""
        uuid = str(row.get("pool") or "").strip()
        project = str(row.get("project") or "unknown").strip()
        chain = str(row.get("chain") or "unknown").strip()
        symbol = str(row.get("symbol") or "?").strip()
        if not uuid:
            # A row without a uuid cannot be charted, so it cannot be judged. The reader
            # mints one from the name rather than dropping the row: it still counts as
            # something the venue published, and `fetch_pool_history` will answer nothing
            # for it, which is reported as a skip instead of silently shrinking the market.
            uuid = f"unkeyed:{project}:{symbol}:{chain}"
        return cls(
            uuid=uuid,
            project=project,
            chain=chain,
            symbol=symbol,
            tvl_usd=_number(row.get("tvlUsd")) or 0.0,
            apy=_number(row.get("apy")) or 0.0,
            apy_base=_number(row.get("apyBase")),
            apy_reward=_number(row.get("apyReward")),
            mean_apy_30d=_number(row.get("apyMean30d")),
            volume_usd_1d=_number(row.get("volumeUsd1d")),
            stablecoin=bool(row.get("stablecoin")),
            exposure=str(row.get("exposure") or ""),
            pool_meta=str(row.get("poolMeta") or ""),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "project": self.project,
            "chain": self.chain,
            "symbol": self.symbol,
            "tvl_usd": round(float(self.tvl_usd), 2),
            "apy": round(float(self.apy), 6),
            "apy_base": None if self.apy_base is None else round(self.apy_base, 6),
            "apy_reward": None if self.apy_reward is None else round(self.apy_reward, 6),
            "mean_apy_30d": None
            if self.mean_apy_30d is None
            else round(self.mean_apy_30d, 6),
            "volume_usd_1d": None
            if self.volume_usd_1d is None
            else round(self.volume_usd_1d, 2),
            "stablecoin": bool(self.stablecoin),
            "exposure": self.exposure,
        }


def bucket_for(apy_percent: float) -> str:
    """Which bucket a published APY belongs to — a *policy*, not a measurement.

    The bucket decides what a protocol is allowed to promise
    (:data:`agent.physics.risk_assessment.PLAUSIBLE_APY`), so it cannot be decided by how
    the pool behaves; the only thing the venue states that is comparable is the APY it
    advertises. A pool that pays what a vault may promise is judged as a vault; one that
    advertises four hundred percent is judged as an experiment, and the ceiling that comes
    with the bucket is printed in every verdict that uses it.
    """
    apy = float(apy_percent) / 100.0 if math.isfinite(float(apy_percent)) else 0.0
    if apy <= float(PLAUSIBLE_APY[BUCKET_SAFE]):
        return BUCKET_SAFE
    if apy <= float(PLAUSIBLE_APY[BUCKET_ACTIVE]):
        return BUCKET_ACTIVE
    return BUCKET_EXPERIMENTAL


def _unique_names(rows: Sequence[PoolRow]) -> List[str]:
    """A distinct name per pool, so that two pools cannot share one verdict.

    ``project symbol (chain)`` is what a human reads, and it is not unique: one chain can
    carry two Uniswap pools of the same pair at different fees. The first pool to ask for
    a name gets it; a second one gets six characters of its uuid appended, which is enough
    to separate them and short enough to stay in a table.
    """
    taken: Dict[str, int] = {}
    names: List[str] = []
    for row in rows:
        base = row.name
        taken[base] = taken.get(base, 0) + 1
        names.append(base if taken[base] == 1 else f"{base} #{row.uuid[:6]}")
    return names


@dataclass(frozen=True)
class PoolFrame:
    """One pool's *two* published series, aligned row by row.

    The funnel needs one number per day and reads it with
    :meth:`DefiLlamaScanner.fetch_pool_history`. A question about *liquidity leaving*
    needs two — the APY and the depth — and two series cleaned independently cannot be
    zipped afterwards: :func:`clean_series` drops a missing ``apy`` from one and a
    missing ``tvlUsd`` from the other, and day 41 of the first is then day 47 of the
    second. So the pairing happens **before** the cleaning, row by row, and a row that is
    missing either number is dropped from both. What is left is a shorter record of days
    the venue actually published both numbers for — which is the only kind of day a
    statement about "yield rose while depth fell" can be made on.

    ``missing`` counts the rows that were dropped, so a report can say how much of the
    history the pairing cost instead of quietly measuring a thinner series.
    """

    uuid: str = ""
    apy: Tuple[float, ...] = ()
    tvl: Tuple[float, ...] = ()
    missing: int = 0

    def __len__(self) -> int:
        return len(self.apy)

    def __bool__(self) -> bool:
        return bool(self.apy)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "points": len(self),
            "missing": int(self.missing),
        }


# ── a name, priced from the feed ────────────────────────────────────────────


def _words(value: object) -> str:
    """A name as comparable words: lower case, letters and digits only.

    The same normalisation the trading loop applies to a vault's name, repeated here on
    purpose: this file has to answer "which pool is this name" on its own, and a resolver
    that borrowed the loop's helper would not be importable by a notebook that has no loop.
    """
    return "".join(
        character for character in str(value or "").lower() if character.isalnum()
    )


def _pieces(value: object) -> frozenset:
    """A name as a set of words — ``USDC-WETH`` and ``WETH-USDC`` are the same pair.

    The second normalisation, and it exists because the venue spells pairs in its own order:
    ``symbol`` is what the project called it, and an operator writing ``USDC-WETH`` should not
    have to know that the feed wrote ``WETH-USDC``. Tokens of one character are dropped —
    they make every name match every other name.
    """
    text = str(value or "")
    return frozenset(
        token
        for token in "".join(
            character if character.isalnum() else " " for character in text.lower()
        ).split()
        if len(token) > 1
    )


def _rank(name: object, row: PoolRow) -> Optional[int]:
    """How well a catalogue row carries a name; ``None`` when it does not carry it at all.

    Five degrees of evidence, and the order is the whole rule — a lower number is a better
    match:

    ``0``  the symbol *is* the name (``USDC`` → the pool whose symbol is ``USDC``);
    ``1``  the whole label is the name (``project symbol (chain)``, normalised) — the way a
           careful operator spells an entry;
    ``2``  the symbol is the same *set of words* (``USDC-WETH`` → ``WETH-USDC``: the same
           pair, spelled the way the project spelled it);
    ``3``  the symbol carries the name (``USDC`` → ``USDC-WETH``: a pair that holds it);
    ``4``  the name carries the symbol (``Aave V3 USDC`` → ``USDC``: a vault named after its
           token, which is how most real vaults are named).

    Anything else is not a match, and is not turned into one: the caller reports the
    absence, because a nearest-neighbour guess here is a measurement of a different pool
    wearing this one's name.
    """
    wanted = _words(name)
    symbol = _words(row.symbol)
    if not wanted or not symbol:
        return None
    if symbol == wanted:
        return 0
    if _words(row.project) + symbol + _words(row.chain) == wanted:
        return 1
    words = _pieces(row.symbol)
    if len(words) > 1 and words == _pieces(name):
        return 2
    if wanted in symbol:
        return 3
    if symbol in wanted or _words(row.project) + symbol == wanted:
        return 4
    return None


@dataclass(frozen=True)
class FeedRecord:
    """A *name*, priced from the feed: the pool that answered, and the series it published.

    Deliberately not a :class:`agent.physics.portfolio.Protocol`. A ``Protocol`` asserts that
    an *address* published a record; here the address published nothing and the record is the
    feed's statement about a **name**. Keeping the two apart is what lets a report say
    "priced from the feed, not from this address's log" — and lets a loop refuse to sign into
    a vault whose own record it has never read, while still measuring a real market.

    ``matched`` counts the pools that carried the name as well as the one that won, because
    "the only pool called this" and "the deepest of nine called this" are different amounts
    of evidence and the report prints the difference. ``reason`` carries the feed's own
    sentence when there is no series.
    """

    name: str
    uuid: str = ""
    pool: str = ""
    symbol: str = ""
    project: str = ""
    chain: str = ""
    apy: Tuple[float, ...] = ()
    missing: int = 0
    matched: int = 0
    #: True when the operator's book named the uuid, so nothing was matched and nothing is
    #: claimed about the pool's label: a report says so rather than looking it up.
    pinned: bool = False
    reason: str = ""

    def __len__(self) -> int:
        return len(self.apy)

    def __bool__(self) -> bool:
        """True when there is enough of a series to be a record at all."""
        return len(self.apy) >= MIN_HISTORY

    @property
    def label(self) -> str:
        """What a report prints as the provenance: the pool, the chain, enough uuid to find."""
        if not self.uuid:
            return "the feed (no pool matched)"
        if self.pinned and not self.pool:
            return f"the feed pool [{self.uuid[:8]}] (pinned by the book)"
        who = f" {self.pool}" if self.pool else ""
        base = f"the feed{who} [{self.uuid[:6]}]"
        # How many pools carried the name is part of the answer: "the deepest of forty-one
        # called USDC" and "the only pool called this" are different amounts of evidence.
        return f"{base} (deepest of {self.matched})" if self.matched > 1 else base

    def rates(self, steps_per_year: int = STEPS_PER_YEAR) -> np.ndarray:
        """The APY series as per-step returns — the units the funnel's arithmetic is in."""
        return to_returns(self.apy, int(steps_per_year))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "uuid": self.uuid,
            "pool": self.pool,
            "chain": self.chain,
            "points": len(self.apy),
            "missing": int(self.missing),
            "matched": int(self.matched),
            "pinned": bool(self.pinned),
            "reason": self.reason,
        }


# ── the wire: one reader, two endpoints, and a cache ────────────────────────


def _requests_get(url: str, timeout: float) -> Any:
    """The default transport: one GET, a timeout, and no retries to hide a dead feed.

    ``requests`` is imported here rather than at module import because the rest of the
    layer must stay importable without it — a machine that only runs the declared market
    should not need an HTTP client installed to import ``agent.physics``. When it is
    missing, the sentence says how to fix it instead of raising ``ImportError`` from three
    frames down.
    """
    try:
        import requests  # deliberate lazy import, see docstring
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise DefiLlamaError(
            "the DeFiLlama adapter needs `requests`: pip install requests"
        ) from exc
    return requests.get(
        url,
        timeout=float(timeout),
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )


def _payload(response: Any) -> Mapping[str, Any]:
    """The JSON body of a response, or :class:`DefiLlamaUnavailable` with a sentence.

    Every way a response can be unusable is answered here — a status other than 200, a body
    that is not JSON, a JSON document without the ``data`` list the endpoint promises — so
    that no caller has to know what ``requests`` raises.
    """
    status = getattr(response, "status_code", 200)
    if int(status) != 200:
        raise DefiLlamaUnavailable(
            f"the feed answered HTTP {int(status)}; the market is not readable right now",
            int(status),
        )
    try:
        body = response.json()
    except Exception as exc:  # noqa: BLE001 - a body that is not JSON is one fact
        raise DefiLlamaUnavailable(f"the feed answered a body that is not JSON: {exc}")
    if not isinstance(body, Mapping):
        raise DefiLlamaUnavailable("the feed answered JSON that is not an object")
    if str(body.get("status", "success")) != "success":
        raise DefiLlamaUnavailable(
            f"the feed reported status {body.get('status')!r}: {body.get('message', '')}"
        )
    if not isinstance(body.get("data"), (list, tuple)):
        raise DefiLlamaUnavailable("the feed answered without a `data` list")
    return body


@dataclass
class DefiLlamaScanner:
    """The venue's wire: the catalogue, one chart per pool, and a local cache.

    Two public calls, and both are answerable on their own — that is why this class exists
    separately from :class:`DiscoveryEngine`. :meth:`get_top_pools` is *what is on offer*,
    sorted by depth; :meth:`fetch_pool_history` is *what one pool paid*, as the daily APY
    series the venue publishes. :meth:`market` composes the two into the shape the funnel
    already reads, and nothing above it knows there was a network.

    The cache is not an optimisation of the network; it is what makes a scan *repeatable*.
    A two-hundred-pool run costs two hundred requests, and the second run of the same
    notebook would otherwise measure a slightly different market and be unable to say
    whether a change in the funnel came from the price or from the pool list. Documents are
    kept for :data:`CACHE_TTL` seconds; a stale document is served rather than nothing when
    the feed is down, and the scanner records that it did so in :attr:`warnings`.
    """

    pools_url: str = POOLS_URL
    chart_url: str = CHART_URL
    timeout: float = DEFAULT_TIMEOUT
    steps_per_year: int = STEPS_PER_YEAR
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    cache_ttl: float = CACHE_TTL
    transport: Optional[Callable[[str, float], Any]] = None
    retries: int = RETRIES
    backoff: float = BACKOFF
    requests_made: int = 0
    cache_hits: int = 0
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            root = os.environ.get("ENRG_CACHE_DIR") or ".cache"
            self.cache_dir = Path(root) / "defillama"

    # -- the wire ---------------------------------------------------------- #

    def _get(self, url: str) -> Mapping[str, Any]:
        """One request, as a payload or as :class:`DefiLlamaUnavailable`.

        A ``429`` is waited out and asked again (:data:`RETRIES`, doubling pauses), because
        the host is pacing the scan rather than failing it — the first two-hundred-pool run
        of this adapter lost 125 pools to exactly that, and reporting a rate limit as
        "these pools publish nothing" would have been a lie about the market.
        """
        fetch = self.transport or _requests_get
        attempts = max(int(self.retries), 0) + 1
        for attempt in range(attempts):
            self.requests_made += 1
            try:
                response = fetch(url, float(self.timeout))
            except DefiLlamaError:
                raise
            except Exception as exc:  # noqa: BLE001 - one sentence for every socket failure
                raise DefiLlamaUnavailable(
                    f"the feed at {url.split('?')[0]} could not be reached ({exc}); "
                    "the market is not readable right now"
                ) from exc
            try:
                return _payload(response)
            except DefiLlamaUnavailable as exc:
                if not exc.throttled or attempt + 1 >= attempts:
                    raise
                self.warnings.append(
                    f"{url.split('/')[-1]}: the feed asked to be paced; waiting "
                    f"{self.backoff * (attempt + 1):.0f}s and asking again"
                )
                time.sleep(float(self.backoff) * (attempt + 1))
        raise DefiLlamaUnavailable(  # pragma: no cover - the loop always returns or raises
            f"the feed at {url.split('?')[0]} never answered"
        )

    def _path(self, key: str) -> Optional[Path]:
        if not self.use_cache or self.cache_dir is None:
            return None
        return Path(self.cache_dir) / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[Mapping[str, Any]]:
        """A cached document; ``stale`` is set when it is too old to trust by itself."""
        path = self._path(key)
        if path is None or not path.exists():
            return None
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(body, Mapping) or "data" not in body:
            return None
        fetched = float(body.get("fetched", 0.0) or 0.0)
        if (time.time() - fetched) <= float(self.cache_ttl):
            self.cache_hits += 1
            return body
        return {**body, "stale": True}

    def _write_cache(self, key: str, body: Mapping[str, Any]) -> None:
        path = self._path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            blob = json.dumps(
                {"fetched": time.time(), "data": body.get("data", [])},
                separators=(",", ":"),
            )
            temporary = path.with_suffix(".tmp")
            temporary.write_text(blob, encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            # A cache that cannot be written is a cache that is not used; it must never be
            # the reason a scan fails.
            return

    def _document(
        self, key: str, url: str, *, refresh: bool = False
    ) -> Mapping[str, Any]:
        """A document from the cache when it is fresh, from the wire when it is not.

        When the wire fails and a stale copy exists, the stale copy is served and noted: a
        scan on yesterday's numbers that *says so* is worth more than no scan, and the note
        is what keeps it honest.
        """
        cached = None if refresh else self._read_cache(key)
        if cached is not None and not cached.get("stale"):
            return cached
        try:
            body = self._get(url)
        except DefiLlamaUnavailable:
            if cached is not None:
                self.warnings.append(f"{key}: the feed is unreachable, served from cache")
                return cached
            raise
        self._write_cache(key, body)
        return body

    def clear_cache(self) -> int:
        """Delete everything this scanner cached; returns how many documents went."""
        if self.cache_dir is None or not Path(self.cache_dir).exists():
            return 0
        removed = 0
        for path in Path(self.cache_dir).glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
        return removed

    # -- what is on offer --------------------------------------------------- #

    def get_top_pools(
        self, limit: int = POOLS_LIMIT, *, refresh: bool = False
    ) -> List[PoolRow]:
        """The market's deepest pools, ``limit`` of them, by TVL — the brief's top-N.

        Sorted by depth and not by APY on purpose: the funnel's first question is what a
        reader could actually get into and out of, and a four-thousand-percent pool holding
        a hundred dollars is not part of the market's opportunity, it is part of its noise.
        The sort is deterministic (ties break on the name), so two runs over the same cached
        document produce the same list.

        Raises:
            DefiLlamaUnavailable: the catalogue could not be read. One sentence, no
                traceback — the caller decides whether a dead feed is an empty report or a
                non-zero exit code.
        """
        body = self._document("pools", self.pools_url, refresh=refresh)
        rows = [
            PoolRow.from_payload(row)
            for row in body.get("data", ())
            if isinstance(row, Mapping)
        ]
        rows.sort(key=lambda row: (-float(row.tvl_usd), row.name))
        return rows[: max(int(limit), 0)]

    def fetch_pool_history(
        self, uuid: str, *, field: str = "apy", refresh: bool = False
    ) -> Tuple[float, ...]:
        """One pool's published ``field`` series — the daily APY, gaps dropped.

        Returns an empty tuple, never an exception, for a pool the feed cannot answer: one
        unreadable uuid is one pool to skip, and it must not cost the other hundred and
        ninety-nine. What was wrong is appended to :attr:`warnings`, so the report can say
        how many pools the feed failed on rather than quietly shrinking the market.

        A series more than :data:`MAX_MISSING_SHARE` empty is refused as well: a record with
        a third of its points absent is a record sampled at a different rate than the one
        the statistics assume.
        """
        key = "chart-" + "".join(
            character if character.isalnum() else "_" for character in str(uuid)
        )
        url = self.chart_url.format(pool=uuid)
        try:
            body = self._document(key, url, refresh=refresh)
        except DefiLlamaUnavailable as exc:
            self.warnings.append(f"{uuid}: {exc}")
            return ()
        published, missing = clean_series(
            [row.get(field) for row in body.get("data", ()) if isinstance(row, Mapping)]
        )
        total = len(published) + missing
        if total == 0:
            self.warnings.append(f"{uuid}: the feed published no history")
            return ()
        if missing / total > float(MAX_MISSING_SHARE):
            self.warnings.append(
                f"{uuid}: {missing} of {total} points have no {field}; not published"
            )
            return ()
        return tuple(published)

    def fetch_pool_frame(self, uuid: str, *, refresh: bool = False) -> PoolFrame:
        """One pool's APY **and** depth, aligned — the frame a two-variable question needs.

        Same contract as :meth:`fetch_pool_history`: an empty frame and one line in
        :attr:`warnings` for a pool the feed cannot answer, never an exception. The two
        numbers come from the *same* cached document, so this method costs no second
        request — and :data:`MAX_MISSING_SHARE` applies to the paired rows, because a
        pool that publishes an APY but no depth on a third of its days is a series
        sampled at a different rate than the statistics assume, exactly as before.
        """
        key = "chart-" + "".join(
            character if character.isalnum() else "_" for character in str(uuid)
        )
        url = self.chart_url.format(pool=uuid)
        try:
            body = self._document(key, url, refresh=refresh)
        except DefiLlamaUnavailable as exc:
            self.warnings.append(f"{uuid}: {exc}")
            return PoolFrame(uuid=str(uuid))
        apy: List[float] = []
        tvl: List[float] = []
        missing = 0
        for row in body.get("data", ()):
            if not isinstance(row, Mapping):
                missing += 1
                continue
            rate = _number(row.get("apy"))
            depth = _number(row.get("tvlUsd"))
            if rate is None or depth is None:
                missing += 1
                continue
            apy.append(rate)
            tvl.append(depth)
        total = len(apy) + missing
        if total == 0:
            self.warnings.append(f"{uuid}: the feed published no history")
            return PoolFrame(uuid=str(uuid))
        if missing / total > float(MAX_MISSING_SHARE):
            self.warnings.append(
                f"{uuid}: {missing} of {total} days are missing an apy or a depth; "
                "not published"
            )
            return PoolFrame(uuid=str(uuid), missing=missing)
        return PoolFrame(
            uuid=str(uuid),
            apy=tuple(apy),
            tvl=tuple(tvl),
            missing=int(missing),
        )

    # -- the two composed: a market the funnel can read ---------------------- #

    def protocols(
        self,
        limit: int = POOLS_LIMIT,
        *,
        workers: Optional[int] = None,
        refresh: bool = False,
    ) -> Tuple[List[Protocol], Dict[str, PoolRow], List[Dict[str, Any]]]:
        """``(protocols, pools_by_name, skipped)`` — the market as internal objects.

        One request per pool, eight at a time, and every pool that cannot become a
        ``Protocol`` is *named in* ``skipped`` with the reason instead of disappearing: a
        market report that does not say how many pools it could not read is a report that
        pretends its own coverage is complete.
        """
        rows = self.get_top_pools(limit, refresh=refresh)
        names = _unique_names(rows)
        histories = self._histories(rows, names, workers=workers, refresh=refresh)
        protocols: List[Protocol] = []
        pools: Dict[str, PoolRow] = {}
        skipped: List[Dict[str, Any]] = []
        for name, row, series in histories:
            if not series:
                skipped.append({"name": name, "reason": "no history from the feed"})
                continue
            returns = to_returns(series, int(self.steps_per_year))
            if returns.size < MIN_HISTORY:
                skipped.append(
                    {"name": name, "reason": f"{returns.size} points of history"}
                )
                continue
            protocols.append(
                Protocol(
                    name=name,
                    bucket=bucket_for(row.apy),
                    claim_apy=float(row.apy) / 100.0,
                    returns=returns,
                    address=sim_address(name),
                    turnover=float(row.tvl_usd),
                )
            )
            pools[name] = row
        return protocols, pools, skipped

    def _histories(
        self,
        rows: Sequence[PoolRow],
        names: Sequence[str],
        *,
        workers: Optional[int] = None,
        refresh: bool = False,
    ) -> List[Tuple[str, PoolRow, Tuple[float, ...]]]:
        """Fetch every pool's history, in the catalogue's order.

        The pool is the unit of failure and the order is preserved: ``map`` keeps the input
        order, and :meth:`fetch_pool_history` answers a dead uuid with an empty tuple, so
        neither a slow pool nor a broken one can reorder or shorten the market.
        """
        pairs = list(zip(names, rows))

        def one(pair: Tuple[str, PoolRow]) -> Tuple[str, PoolRow, Tuple[float, ...]]:
            name, row = pair
            return name, row, self.fetch_pool_history(row.uuid, refresh=refresh)

        if not pairs:
            return []
        size = max(int(workers or WORKERS), 1)
        if size == 1 or len(pairs) == 1:
            return [one(pair) for pair in pairs]
        with ThreadPoolExecutor(max_workers=size) as pool:
            return list(pool.map(one, pairs))

    def market(
        self,
        limit: int = POOLS_LIMIT,
        *,
        workers: Optional[int] = None,
        refresh: bool = False,
    ) -> "DefiLlamaMarket":
        """The whole scan as one value: what the venue publishes, read once."""
        protocols, pools, skipped = self.protocols(
            limit, workers=workers, refresh=refresh
        )
        return DefiLlamaMarket(
            protocols=tuple(protocols),
            pools=pools,
            skipped=tuple(skipped),
            fetched_at=time.time(),
            warnings=tuple(self.warnings),
        )


#: How many catalogue rows a *name* is resolved against: all of them. The funnel's
#: two hundred is a budget for reading **histories** (one request per pool); a lookup costs
#: nothing per row, and truncating the catalogue to the two hundred deepest pools would make
#: a $36 M Arbitrum pool unfindable because Ethereum holds the top two hundred. This is a
#: number rather than ``len(rows)`` only because ``get_top_pools`` takes a bound.
WHOLE_CATALOGUE = 1_000_000


@dataclass
class FeedHistory:
    """A *name* → the pool it denotes, and that pool's published series.

    The one seam between an address book and the feed. A venue that reads a chain knows the
    address and the label; the feed knows the label and a uuid; this class answers the
    question "what does this label earn" and says how it answered. Three rules, and each one
    is a place an adapter could have invented a measurement:

    **A match is evidence, not a similarity.** :func:`_rank` orders five kinds of evidence,
    and only those five count — the last of them is the same *set* of words, which is how
    ``USDC-WETH`` finds the pool the project spelled ``WETH-USDC``. Among equals the
    *deepest* pool wins (``tvlUsd``), because a book entry that names a stablecoin means the
    pool with an exit; ``matched`` reports how many tied, so a report can print "the deepest
    of nine". A name nothing carries gets a :class:`FeedRecord` with no series and a
    sentence — never the nearest pool.

    **A chain is a filter, not a preference.** ``chain`` is spelled the way the feed spells
    it (``Arbitrum``); when set, only pools on that chain are candidates, and a name that
    exists only elsewhere is reported with the chains it *does* exist on. An empty ``chain``
    filters nothing, which is the honest answer for a venue whose chain id this file does not
    know.

    **A short or gappy series is refused by the scanner, not padded here.** Every point comes
    from ``/chart/{uuid}`` through :meth:`DefiLlamaScanner.fetch_pool_history`, so
    :data:`MAX_MISSING_SHARE` and :data:`MIN_HISTORY` apply exactly as they do to the funnel,
    and a pool the feed cannot chart is a record with a reason.
    """

    scanner: DefiLlamaScanner = field(default_factory=DefiLlamaScanner)
    chain: str = ""
    limit: int = WHOLE_CATALOGUE

    # -- finding ------------------------------------------------------------ #

    def candidates(
        self, name: str, *, refresh: bool = False
    ) -> Tuple[List[Tuple[int, PoolRow]], List[PoolRow]]:
        """``(ranked, everything)`` — the candidates for a name and the whole catalogue.

        The catalogue is read once and both lists come from it, so asking "did this name
        exist on another chain" costs no second request (a warm cache answers it for free).
        """
        rows = list(self.scanner.get_top_pools(int(self.limit), refresh=refresh))
        ranked: List[Tuple[int, PoolRow]] = []
        if _words(name):
            for row in rows:
                rank = _rank(name, row)
                if rank is not None:
                    ranked.append((rank, row))
        return ranked, rows

    def find(
        self, name: str, *, refresh: bool = False
    ) -> Tuple[Optional[PoolRow], int]:
        """The pool this name denotes, and how many pools tied for it on this chain."""
        ranked, _ = self.candidates(name, refresh=refresh)
        if self.chain:
            ranked = [
                (rank, row)
                for rank, row in ranked
                if _words(row.chain) == _words(self.chain)
            ]
        if not ranked:
            return None, 0
        best = min(rank for rank, _ in ranked)
        tied = [(rank, row) for rank, row in ranked if rank == best]
        # The deepest of the pools that carry the name: a book entry that names a token means
        # the pool with an exit, and the tie count travels with the answer.
        tied.sort(key=lambda item: -float(item[1].tvl_usd))
        return tied[0][1], len(tied)

    # -- pricing ------------------------------------------------------------ #

    def history(
        self,
        name: str,
        *,
        steps: Optional[int] = None,
        uuid: str = "",
        refresh: bool = False,
    ) -> FeedRecord:
        """The series this name is priced at — or a record that says why there is none.

        ``uuid`` pins the pool and removes the name match entirely: an operator who knows
        which pool a vault implements should say so rather than hope that a label resolves.
        A pinned uuid the feed cannot chart is *still* a record with no series and the feed's
        own reason, never a fall-back to the name match — a fallback would mean the
        operator's declaration had been quietly replaced by a similarity.
        """
        label = str(name or "")
        pinned = str(uuid or "").strip()
        row: Optional[PoolRow] = None
        matched = 0
        if not pinned:
            row, matched = self.find(label, refresh=refresh)
            pinned = str(row.uuid) if row is not None else ""
        if not pinned:
            return FeedRecord(
                name=label, matched=int(matched), reason=self._miss(label, refresh=refresh)
            )

        before = len(self.scanner.warnings)
        apy = self.scanner.fetch_pool_history(pinned, refresh=refresh)
        said = self.scanner.warnings[before:]
        if steps is not None and int(steps) > 0:
            apy = tuple(apy[-int(steps) :])
        return FeedRecord(
            name=label,
            uuid=pinned,
            pinned=row is None,
            pool=row.name if row is not None else "",
            symbol=row.symbol if row is not None else "",
            project=row.project if row is not None else "",
            chain=row.chain if row is not None else "",
            apy=tuple(apy),
            matched=int(matched),
            reason="" if apy else (said[-1] if said else "the feed published no history"),
        )

    def _miss(self, name: str, *, refresh: bool = False) -> str:
        """Why a name has no pool: nothing carries it, or nothing carries it *here*."""
        reason = "no pool in the catalogue carries this name"
        if not self.chain:
            return reason
        ranked, _ = self.candidates(name, refresh=refresh)
        elsewhere = [row for _, row in ranked if _words(row.chain) != _words(self.chain)]
        if not elsewhere:
            return reason
        chains = sorted({row.chain for row in elsewhere})
        return (
            f"{len(elsewhere)} pool(s) carry this name and none of them is on "
            f"{self.chain} ({', '.join(chains[:3])})"
        )


@dataclass(frozen=True)
class DefiLlamaMarket:
    """The real market, in the shape the funnel already reads: a *venue*, not a scanner.

    Everything the two gates ask of a venue is here and nothing else —
    ``get_protocol_list()``, ``protocol(name)``, ``step``, ``name`` — which is exactly the
    surface :class:`agent.physics.interfaces.IWeb3Provider` defines for a chain. That is the
    point: the funnel does not learn a new vocabulary to read the wild market, and
    ``Scanner().scan(market)`` works on it without a line of new code.

    What it deliberately cannot do is move money: there is no ``execute_stake`` here, no
    signer, no key. The venue publishes records; the book is what funds them, and the book
    is not in this file.
    """

    protocols: Tuple[Protocol, ...] = ()
    pools: Mapping[str, PoolRow] = field(default_factory=dict)
    skipped: Tuple[Dict[str, Any], ...] = ()
    fetched_at: float = 0.0
    warnings: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.protocols)

    @property
    def name(self) -> str:
        return VENUE_NAME

    @property
    def step(self) -> int:
        """The venue's clock: the longest record it publishes, in days.

        A venue's step is the frontier of what may be read, and for a daily feed that
        frontier is "as many days as the oldest pool has published". A shorter pool is not
        truncated by it — ``Protocol.record(step)`` returns what exists — so no pool is
        credited with history it does not have, and none is denied the history it does.
        """
        return max((int(protocol.returns.size) for protocol in self.protocols), default=0)

    def get_protocol_list(self) -> List[Protocol]:
        return list(self.protocols)

    def protocol(self, name: str) -> Protocol:
        for protocol in self.protocols:
            if protocol.name == str(name):
                return protocol
        raise PortfolioError(f"the venue has no pool called {name!r}")

    def pool(self, name: str) -> Optional[PoolRow]:
        """The advertised row behind a protocol — chain, project, depth, current APY."""
        return self.pools.get(str(name))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "venue": VENUE_NAME,
            "protocols": len(self.protocols),
            "step": self.step,
            "fetched_at": round(float(self.fetched_at), 3),
            "skipped": len(self.skipped),
            "warnings": len(self.warnings),
        }


# ── the funnel's report: what the market offered and what the gates did ─────


@dataclass(frozen=True)
class MarketFunnel:
    """One trip through both gates on a *real* market: the counts, and who was refused.

    The numbers are kept in the order the gates ran, so that a reader can subtract two of
    them and get the work of one gate rather than a mixture:

    ``scanned``      what the venue offered (the brief's two hundred, top by TVL)
    ``market``       what could be read as a record (a pool with no chart is not one)
    ``watched``      what level 1 kept — a mean, a spread and a liquidity floor's worth
    ``measured``     what level 2 could have an opinion about (the survey needs windows)
    ``unmeasured``   records too short for a survey: waiting, neither refused nor approved
    ``approved``     ``HONEST`` verdicts — everything else is a refusal with a reason

    ``refused`` and ``unmeasured`` are different facts and are never added together: one
    says the record is not what it claims, the other says nobody has read enough of it.
    """

    mode: str = MODE_REAL
    scanned: int = 0
    market: int = 0
    watchlist: Watchlist = field(default_factory=Watchlist)
    assessments: Mapping[str, Assessment] = field(default_factory=dict)
    pools: Mapping[str, PoolRow] = field(default_factory=dict)
    step: int = 0
    elapsed: float = 0.0
    venue: str = VENUE_NAME
    skipped: Tuple[Dict[str, Any], ...] = ()
    warnings: Tuple[str, ...] = ()
    error: str = ""

    # -- what survived ------------------------------------------------------ #

    def approved_names(self) -> List[str]:
        """The ``HONEST`` records, in the queue's own order (best record first)."""
        return [
            name
            for name in self.watchlist.names()
            if self.assessments.get(name) is not None
            and self.assessments[name].admissible
        ]

    @property
    def approved(self) -> int:
        return len(self.approved_names())

    @property
    def watched(self) -> int:
        return len(self.watchlist.kept)

    @property
    def measured(self) -> int:
        """What level 2 actually had an opinion about — the rest is waiting."""
        return sum(
            1 for item in self.assessments.values() if item.verdict != UNMEASURED
        )

    @property
    def unmeasured(self) -> int:
        return sum(
            1 for item in self.assessments.values() if item.verdict == UNMEASURED
        )

    def verdicts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.assessments.values():
            counts[item.verdict] = counts.get(item.verdict, 0) + 1
        return dict(sorted(counts.items()))

    def refused(self) -> List[Dict[str, Any]]:
        """Every read that did not end in ``HONEST`` and was not merely short."""
        return [
            self._row(item)
            for item in self.assessments.values()
            if item.verdict not in (HONEST, UNMEASURED)
        ]

    def unmeasured_rows(self) -> List[Dict[str, Any]]:
        return [
            self._row(item)
            for item in self.assessments.values()
            if item.verdict == UNMEASURED
        ]

    def _row(self, item: Assessment) -> Dict[str, Any]:
        """One verdict as a table row, with the venue's own facts glued on."""
        pool = self.pools.get(item.name)
        return {
            "name": item.name,
            "address": item.address,
            "chain": "" if pool is None else pool.chain,
            "project": "" if pool is None else pool.project,
            "tvl_usd": 0.0 if pool is None else round(pool.tvl_usd, 2),
            "bucket": item.bucket,
            "apy": round(float(item.claim_apy) * 100.0, 4),
            "implied_apy": round(float(item.implied_apy) * 100.0, 4),
            "verdict": item.verdict,
            "a2": None if math.isnan(item.a2) else round(float(item.a2), 6),
            "cv": None if math.isnan(item.cv) else round(float(item.cv), 6),
            "claim_z": None if math.isnan(item.claim_z) else round(item.claim_z, 3),
            "points": int(item.points),
            "windows": entrain_windows(int(item.points)),
            "reason": item.reason,
        }

    # -- the ordering a reader wants ---------------------------------------- #

    def top_rejected(self, limit: int = 3) -> List[Dict[str, Any]]:
        """The highest-paying pools the agent *refused*, and why — the brief's table.

        Ordered by the APY the venue advertises, because the point of the table is the
        inversion: the market's loudest promises are the ones the record refused. A pool
        that was merely too short to read is not here — refusing to have an opinion is not
        an opinion, and mixing the two would make the table say that ``UNMEASURED`` is a
        finding about a pool rather than about the length of its history.
        """
        rows = sorted(
            self.refused(),
            key=lambda row: (-float(row["apy"]), float(-row["tvl_usd"])),
        )
        return rows[: max(int(limit), 0)]

    def stages(self) -> List[Tuple[str, int]]:
        """The funnel as surviving widths, in the order a chart draws them."""
        return [
            ("market", int(self.scanned or self.market)),
            ("worth a look", self.watched),
            ("measured by entrain", self.measured),
            ("approved", self.approved),
        ]

    # -- the whole trip, as data -------------------------------------------- #

    def as_dict(self) -> Dict[str, Any]:
        """Everything above, JSON-able — what a log line, a table and a figure need."""
        return {
            "mode": self.mode,
            "venue": self.venue,
            "step": int(self.step),
            "scanned": int(self.scanned),
            "market": int(self.market),
            "skipped": [dict(row) for row in self.skipped],
            "warnings": list(self.warnings),
            "watchlist": self.watched,
            "refused_by_scanner": len(self.watchlist.dropped),
            "scanner_reasons": self.watchlist.reasons(),
            "truncated": int(self.watchlist.truncated()),
            "measured": self.measured,
            "unmeasured": self.unmeasured,
            "refused": len(self.refused()),
            "approved": self.approved,
            "approved_names": self.approved_names(),
            "verdicts": self.verdicts(),
            "stages": self.stages(),
            "refusals": self.refused(),
            "top_rejected": self.top_rejected(),
            "elapsed": round(float(self.elapsed), 3),
            "error": self.error,
        }

    def summary_text(self) -> str:
        """Five lines in the order of the funnel, because that is the argument."""
        if self.error:
            return (
                f"mode {self.mode}: the market could not be read — {self.error}\n"
                "  nothing was scanned, nothing was judged, nothing was approved"
            )
        reasons = ", ".join(
            f"{reason} {count}" for reason, count in self.watchlist.reasons().items()
        )
        verdicts = ", ".join(
            f"{verdict} {count}" for verdict, count in self.verdicts().items()
        )
        return "\n".join(
            [
                f"mode {self.mode}: the venue ({self.venue}) publishes "
                f"{int(self.scanned)} pools, {self.market} of them readable",
                f"  level 1 (scanner): kept {self.watched}, dropped "
                f"{len(self.watchlist.dropped)}" + (f" ({reasons})" if reasons else ""),
                f"  level 2 (entrain): judged {self.measured}, refused "
                f"{len(self.refused())}, unmeasured {self.unmeasured}"
                + (f" ({verdicts})" if verdicts else ""),
                f"  approved for investment: {self.approved} of {self.watched}",
            ]
        )


# ── the engine: the two gates, pointed at a market ──────────────────────────


@dataclass
class DiscoveryEngine:
    """Level 1 then level 2, on a market that was not drawn — or on one that was.

    ``mode`` is the whole switch and it changes only *where the records come from*:
    ``real_defillama`` reads DeFiLlama, ``synthetic`` reads the drawn market, and both go
    through the same :class:`agent.physics.discovery.Scanner` and the same
    :func:`agent.physics.risk_assessment.evaluate_opportunity`. Neither mode can approve
    anything by itself: the report says ``HONEST``, and only a portfolio's whitelist is
    permission to fund — this engine does not have one, on purpose, because the brief of
    this step is data, judgement and conclusions, with no transaction anywhere.

    What the engine *does* add to the core's loop is the one rule a real feed needs and a
    drawn one never did: a survey needs :data:`agent.physics.curiosity.CURIOUS_MIN_WINDOWS`
    fixations, and a daily series younger than about ten months cannot give four of them.
    Such a pool is reported ``UNMEASURED`` with the arithmetic in the reason, because
    "I have not read enough of this" and "this is a lie" are different sentences, and a
    funnel that confuses them refuses the young for being young.

    ``oracle`` is optional exactly as it is in the core: without a body the structural and
    mirror arms are blind and every assessment says so in ``details["measured"]``. That is
    what makes a two-hundred-pool scan affordable in a notebook — and the report prints how
    long the deep pass took, so the trade is visible rather than hidden.
    """

    mode: str = MODE_REAL
    scanner: Optional[DefiLlamaScanner] = None
    gate: Optional[Scanner] = None
    watch_limit: int = WATCH_LIMIT
    min_record: int = MIN_RECORD
    oracle: Any = None
    synthetic_steps: int = 720
    synthetic_seed: int = 7

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise DefiLlamaError(
                f"unknown mode {self.mode!r}: choose one of {', '.join(MODES)}"
            )
        if self.scanner is None:
            self.scanner = DefiLlamaScanner()
        if self.gate is None:
            # The liquidity floor is stated in the venue's own units: dollars of depth for
            # the real market, credits of turnover for the drawn one. One default for both
            # would be a unit error waiting to filter an entire market out.
            self.gate = Scanner(
                turnover_floor=MIN_TVL_USD if self.mode == MODE_REAL else TURNOVER_FLOOR
            )

    # -- where the records come from ---------------------------------------- #

    def venue(
        self,
        limit: int = POOLS_LIMIT,
        *,
        refresh: bool = False,
        workers: Optional[int] = None,
    ) -> Any:
        """The market this engine reads, as objects the funnel already understands.

        Raises:
            DefiLlamaUnavailable: only in ``real_defillama`` mode, and only as one
                sentence: :meth:`run` turns it into an empty report carrying the reason.
        """
        if self.mode == MODE_SYNTHETIC:
            from agent.physics.interfaces import MockWeb3Provider
            from agent.physics.market_generator import generate_market

            venue = MockWeb3Provider(
                generate_market(
                    count=int(limit),
                    steps=int(self.synthetic_steps),
                    seed=int(self.synthetic_seed),
                )
            )
            venue.advance(int(self.synthetic_steps))
            return venue
        if self.scanner is None:  # pragma: no cover - __post_init__ guarantees it
            raise DefiLlamaError("the engine has no scanner in real_defillama mode")
        return self.scanner.market(limit, workers=workers, refresh=refresh)

    # -- the trip ------------------------------------------------------------ #

    def run(
        self,
        limit: int = POOLS_LIMIT,
        *,
        refresh: bool = False,
        workers: Optional[int] = None,
    ) -> MarketFunnel:
        """Read the market once and report what each gate did with it.

        Never raises for a dead feed, a bad pool or a short record: those are three of the
        four things a real market does, and each one has an answer in the report — an
        ``error``, a ``skipped`` row, an ``UNMEASURED`` verdict.
        """
        started = time.perf_counter()
        try:
            venue = self.venue(limit, refresh=refresh, workers=workers)
        except DefiLlamaError as exc:
            return MarketFunnel(
                mode=self.mode,
                elapsed=time.perf_counter() - started,
                error=str(exc),
                warnings=tuple(getattr(self.scanner, "warnings", ()) or ()),
            )
        gate = replace(self.gate, limit=int(self.watch_limit))
        watch = gate.scan(venue)
        step = int(getattr(venue, "step", 0))
        assessments: Dict[str, Assessment] = {}
        for candidate in watch.kept:
            try:
                protocol = venue.protocol(candidate.name)
            except PortfolioError:  # pragma: no cover - the scanner read the same list
                continue
            item = evaluate_opportunity(
                protocol.as_opportunity(step),
                oracle=self.oracle,
                min_record=int(self.min_record),
            )
            windows = entrain_windows(int(item.points))
            if windows < int(CURIOUS_MIN_WINDOWS):
                item = replace(
                    item,
                    verdict=UNMEASURED,
                    reason=(
                        f"{int(item.points)} points of record are {windows} of the "
                        f"{int(CURIOUS_MIN_WINDOWS)} windows an Entrain survey needs; "
                        "there is nothing to judge yet"
                    ),
                )
            assessments[candidate.name] = item
        published = len(venue.get_protocol_list())
        return MarketFunnel(
            mode=self.mode,
            scanned=int(limit) if self.mode == MODE_REAL else published,
            market=published,
            watchlist=watch,
            assessments=assessments,
            pools=dict(getattr(venue, "pools", {}) or {}),
            step=step,
            elapsed=time.perf_counter() - started,
            venue=str(getattr(venue, "name", VENUE_NAME)),
            skipped=tuple(getattr(venue, "skipped", ()) or ()),
            warnings=tuple(getattr(venue, "warnings", ()) or ()),
        )


# ── the CLI: one scan, one table, and an exit code that means something ─────


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m agent.physics.defillama_adapter --limit 200 --top 3``.

    Exit codes, because a shell script cares: ``0`` the market was read (whatever the
    verdicts), ``3`` the feed could not be read, ``4`` the feed published no usable pool.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default=MODE_REAL)
    parser.add_argument("--limit", type=int, default=POOLS_LIMIT, help="pools to scan")
    parser.add_argument("--top", type=int, default=3, help="refusals to explain")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument(
        "--watch-limit", type=int, default=WATCH_LIMIT, help="queue in front of level 2"
    )
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    parser.add_argument("--no-cache", action="store_true", help="do not touch the cache")
    parser.add_argument("--output", default="", help="write the report as JSON here")
    args = parser.parse_args(argv)

    engine = DiscoveryEngine(
        mode=args.mode,
        scanner=DefiLlamaScanner(timeout=args.timeout, use_cache=not args.no_cache),
        watch_limit=args.watch_limit,
    )
    funnel = engine.run(args.limit, refresh=args.refresh, workers=args.workers)
    print(funnel.summary_text())
    if funnel.error:
        return 3
    if funnel.market == 0:
        print("the feed published no usable pool")
        return 4
    for row in funnel.top_rejected(args.top):
        print(
            f"  refused: {row['name']:<44} APY {row['apy']:7.2f}%  {row['verdict']:<19} "
            f"{row['reason']}"
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(funnel.as_dict(), handle, indent=2, ensure_ascii=False)
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the module is normally imported
    raise SystemExit(main())
