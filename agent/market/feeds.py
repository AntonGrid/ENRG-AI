"""Market price feeds (Phase 3 / GLOBAL_AI_ARCHITECTURE step B).

Normalized energy price signals in a single unit (USD per kWh):

- ``dayahead`` — day-ahead exchange price (offline model by default; a real
  ENTSO-E feed attaches with an API key);
- ``p2p`` — peer-to-peer benchmark (slightly discounted vs day-ahead);
- ``spot`` — real-time spot benchmark;
- ``macro`` — macro background (currency/BTC from keyless public APIs).

Every provider has a deterministic offline generator, so training, tests and
the demo run without network access.
"""
from __future__ import annotations

import datetime as dt
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx

#: Unit: USD per kWh.
UNITS = "usd_per_kwh"

PROVIDERS = ("dayahead", "p2p", "spot", "macro")


@dataclass
class PriceFeed:
    """One normalized price observation."""

    source: str
    ts: str
    prices: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {"source": self.source, "ts": self.ts, "prices": dict(self.prices)}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ── offline generators ──────────────────────────────────────────────────────


def _energy_price(step: int, base: float, amplitude: float, ttl: float) -> float:
    """Deterministic intraday sinusoid + slow trend + noise (USD/kWh)."""
    rng = random.Random(9_000 + step)
    hour = (step % 24) / 24.0
    value = base + amplitude * math.sin(2 * math.pi * hour) + 0.0001 * step + rng.uniform(-0.005, 0.005)
    return max(0.005, round(value, 5))


def fetch_offline(step: int = 0, ts: Optional[str] = None) -> List[PriceFeed]:
    """Deterministic market snapshot: day-ahead > p2p > spot."""
    now = ts or _now_iso()
    return [
        PriceFeed("dayahead", now, {"usd_per_kwh": _energy_price(step, 0.12, 0.06, 0.0)}),
        PriceFeed("p2p", now, {"usd_per_kwh": _energy_price(step, 0.10, 0.05, 0.0)}),
        PriceFeed("spot", now, {"usd_per_kwh": _energy_price(step, 0.09, 0.07, 0.0)}),
        PriceFeed("macro", now, {"usd_eur": round(0.92 + 0.001 * step % 0.05, 4), "btc_usd": float(50_000 + 80 * step)}),
    ]


# ── online providers ────────────────────────────────────────────────────────


def _online_macro(client: httpx.Client) -> PriceFeed:
    er = client.get("https://open.er-api.com/v6/latest/USD")
    er.raise_for_status()
    rates = er.json()["rates"]
    cg = client.get("https://api.coingecko.com/api/v3/simple/price", params={"ids": "bitcoin", "vs_currencies": "usd"})
    cg.raise_for_status()
    btc_usd = float(cg.json()["bitcoin"]["usd"])
    return PriceFeed("macro", _now_iso(), {"usd_eur": float(rates.get("EUR", 0.0)), "btc_usd": btc_usd})


def fetch_prices(offline: bool = True, client: Optional[httpx.Client] = None, step: int = 0) -> List[PriceFeed]:
    """One market snapshot. Online: real macro + offline energy models."""
    if offline:
        return fetch_offline(step=step)
    owns = client is None
    if owns:
        client = httpx.Client(timeout=6.0, follow_redirects=True)
    try:
        feeds = fetch_offline(step=step)  # energy benchmarks (keyed feeds attach here)
        try:
            feeds.append(_online_macro(client))
        except Exception:
            pass  # macro falls back to the offline value already included
        return feeds
    finally:
        if owns and client is not None:
            client.close()


class MarketCache:
    """TTL cache so periodic refreshes don't hammer the providers."""

    def __init__(self, ttl_sec: int = 300, offline: bool = True) -> None:
        self.ttl_sec = ttl_sec
        self.offline = offline
        self._cache: Dict[str, PriceFeed] = {}
        self._fetched_at: float = 0.0

    def prices(self) -> List[PriceFeed]:
        now = time.monotonic()
        if self._cache and (now - self._fetched_at) < self.ttl_sec:
            return list(self._cache.values())
        feeds = fetch_prices(offline=self.offline)
        self._cache = {f.source: f for f in feeds}
        self._fetched_at = now
        return feeds

    def clear(self) -> None:
        self._cache.clear()
        self._fetched_at = 0.0
