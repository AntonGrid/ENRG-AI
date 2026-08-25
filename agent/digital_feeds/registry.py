"""Unified digital feed registry.

Every feed (weather, finance, macro, news, blockchain, science) speaks the
same shape — a ``FeedResult`` with a ``domain``, a ``source`` and a flat
``metrics`` dict — so downstream training is domain-agnostic.

- ``collect``     — one point in time (real or offline).
- ``collect_series`` — a chronological series (offline: deterministic).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from agent.digital_feeds import blockchain, finance, macro, news, science, weather

#: Online fetchers (real public APIs/RPC/RSS).
FEED_SOURCES = {
    "weather": weather.fetch,
    "finance": finance.fetch,
    "macro": macro.fetch,
    "news": news.fetch,
    "blockchain": blockchain.fetch,
    "science": science.fetch,
}

#: Offline generators (deterministic synthetic series for tests/training).
OFFLINE_SOURCES = {
    "weather": weather.fetch_offline,
    "finance": finance.fetch_offline,
    "macro": macro.fetch_offline,
    "news": news.fetch_offline,
    "blockchain": blockchain.fetch_offline,
    "science": science.fetch_offline,
}

DEFAULT_FEEDS = list(FEED_SOURCES)


@dataclass
class FeedResult:
    """One normalized observation from any digital feed."""

    domain: str
    source: str
    ts: str
    metrics: Dict[str, float]
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "source": self.source,
            "ts": self.ts,
            "metrics": dict(self.metrics),
        }


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def collect(
    feeds: Optional[List[str]] = None,
    offline: bool = True,
    client: Optional[httpx.Client] = None,
    timeout: float = 6.0,
) -> List[FeedResult]:
    """Fetch one observation per feed.

    In online mode a network failure on any feed falls back to its offline
    generator (graceful degradation); offline mode never touches the network.
    """
    names = feeds or DEFAULT_FEEDS
    owns_client = client is None
    if owns_client:
        client = httpx.Client(timeout=timeout, follow_redirects=True)

    results: List[FeedResult] = []
    try:
        for name in names:
            if name not in FEED_SOURCES:
                continue
            if offline:
                point = OFFLINE_SOURCES[name]()
            else:
                try:
                    point = FEED_SOURCES[name](client)
                except Exception:
                    point = OFFLINE_SOURCES[name]()
            if point:
                results.append(FeedResult(**point))
    finally:
        if owns_client and client is not None:
            client.close()

    return results


def collect_series(
    feeds: Optional[List[str]] = None,
    points: int = 48,
    offline: bool = True,
    seed: int = 7,
    interval_sec: int = 3600,
) -> List[FeedResult]:
    """Build a chronological series of observations.

    Offline: ``points`` deterministic steps per feed (trend + seasonality +
    noise, seeded) with hourly timestamps. Online: ``points`` consecutive
    real fetches (used to bootstrap a live series).
    """
    names = feeds or DEFAULT_FEEDS
    base = dt.datetime.now(dt.timezone.utc)
    series: List[FeedResult] = []

    for i in range(points):
        ts = (base - dt.timedelta(seconds=interval_sec * (points - 1 - i))).isoformat(
            timespec="seconds"
        )
        for name in names:
            if name not in FEED_SOURCES:
                continue
            if offline:
                point = OFFLINE_SOURCES[name](step=i + seed, ts=ts)
            else:
                # Real network point; offline fallback keeps the series dense.
                try:
                    with httpx.Client(timeout=6.0, follow_redirects=True) as client:
                        point = FEED_SOURCES[name](client)
                    point["ts"] = ts
                except Exception:
                    point = OFFLINE_SOURCES[name](step=i + seed, ts=ts)
            if point:
                series.append(FeedResult(**point))

    return series
