"""Proof stream loading and bucketing (energy series).

The oracle persists every proof as ``{device_id, ts, energy_wh, nonce,
mint_tx, mint_status}`` (ADR-0010). This module turns that stream into a
regular fixed-frequency energy series (Wh per bucket) suitable for time-series
forecasting.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

#: Live oracle REST endpoint (ADR-0010 data bridge).
ORACLE_STATS_URL = "https://enrg-oracle.onrender.com/api/v1/proofs"

#: The oracle caps a single page; proofs are returned newest-first.
ORACLE_MAX_LIMIT = 1000


@dataclass
class Proof:
    """One verified energy proof as served by the oracle REST API."""

    device_id: str
    ts: int  # unix seconds
    energy_wh: float
    nonce: int
    mint_tx: Optional[str] = None
    mint_status: Optional[str] = None


@dataclass
class ProofSeries:
    """Regular energy series (Wh per bucket) aggregated from proofs."""

    bucket_minutes: int
    starts: List[dt.datetime]  # UTC, aligned to bucket boundaries
    values: np.ndarray  # Wh per bucket
    device_id: str
    source: str = "oracle"
    total_wh: float = field(init=False)
    raw_proof_count: int = 0

    def __post_init__(self) -> None:
        self.total_wh = float(np.sum(self.values))

    @property
    def labels(self) -> List[str]:
        """ISO-8601 (UTC, minutes) labels for each bucket start."""
        return [t.isoformat(timespec="minutes") for t in self.starts]


def fetch_oracle_proofs(
    client: Any = None,
    limit: int = ORACLE_MAX_LIMIT,
    url: str = ORACLE_STATS_URL,
) -> List[Proof]:
    """Pull recent proofs from the oracle REST API (chronological order).

    Accepts an optional ``httpx.Client`` so callers can reuse a connection.
    Raises on transport/HTTP errors — no silent fallback here (the caller
    decides what to do).
    """
    import httpx

    owns = client is None
    if owns:
        client = httpx.Client(timeout=20.0)
    try:
        resp = client.get(url, params={"limit": min(int(limit), ORACLE_MAX_LIMIT)})
        resp.raise_for_status()
        body = resp.json()
        rows = body.get("proofs") or []
        proofs = [
            Proof(
                device_id=r.get("device_id") or "",
                ts=int(r.get("ts") or 0),
                energy_wh=float(r.get("energy_wh") or 0),
                nonce=int(r.get("nonce") or 0),
                mint_tx=r.get("mint_tx"),
                mint_status=r.get("mint_status"),
            )
            for r in rows
        ]
    finally:
        if owns:
            client.close()
    # Newest-first from the API → chronological for modelling.
    proofs.sort(key=lambda p: p.ts)
    return proofs


def aggregate_proofs(
    proofs: List[Proof],
    bucket_minutes: int = 15,
) -> ProofSeries:
    """Bucket proofs into fixed windows and sum energy (Wh) per bucket.

    Bucket boundaries are absolute (aligned to ``bucket_minutes`` since the
    epoch), so buckets that produced zero proofs stay absent — the caller may
    forward-fill with zeros if a dense grid is required.
    """
    if not proofs:
        raise ValueError("no proofs to aggregate")
    step = int(bucket_minutes) * 60
    if step <= 0:
        raise ValueError("bucket_minutes must be positive")

    sums: Dict[int, float] = {}
    for p in proofs:
        idx = int(p.ts) // step
        sums[idx] = sums.get(idx, 0.0) + float(p.energy_wh)

    idxs = sorted(sums)
    starts = [
        dt.datetime.fromtimestamp(i * step, tz=dt.timezone.utc) for i in idxs
    ]
    values = np.array([sums[i] for i in idxs], dtype=float)
    return ProofSeries(
        bucket_minutes=int(bucket_minutes),
        starts=starts,
        values=values,
        device_id=proofs[0].device_id,
        raw_proof_count=len(proofs),
    )


def forward_fill(series: ProofSeries) -> ProofSeries:
    """Fill gaps with zeros so buckets are contiguous (every bucket present).

    Useful when the proof cadence is sparser than the bucket size (e.g. a
    device that goes offline mid-interval) and the forecaster needs a dense,
    equally spaced grid.
    """
    if len(series.starts) == 0:
        return series
    step = series.bucket_minutes * 60
    first = int(series.starts[0].timestamp()) // step
    last = int(series.starts[-1].timestamp()) // step
    start_ts = dt.datetime.fromtimestamp(first * step, tz=dt.timezone.utc)
    values_by_idx = {
        int(s.timestamp()) // step: v for s, v in zip(series.starts, series.values)
    }
    new_starts: List[dt.datetime] = []
    new_values: List[float] = []
    for i in range(first, last + 1):
        new_starts.append(
            dt.datetime.fromtimestamp(i * step, tz=dt.timezone.utc)
        )
        new_values.append(float(values_by_idx.get(i, 0.0)))
    return ProofSeries(
        bucket_minutes=series.bucket_minutes,
        starts=new_starts,
        values=np.array(new_values, dtype=float),
        device_id=series.device_id,
        source=series.source,
        raw_proof_count=series.raw_proof_count,
    )


def synthetic_solar_series(
    bucket_minutes: int = 15,
    n_buckets: int = 24,
    seed: int = 7,
) -> ProofSeries:
    """Deterministic solar-ish daily cycle (Wh per bucket) for tests/demos.

    Anchors the series *end* at the current bucket boundary, models a
    daylight window between 06:00–18:00 (local solar peak ~12:00), and adds
    gaussian noise. Deterministic per ``seed`` so tests never flake.
    """
    rng = np.random.default_rng(seed)
    now = dt.datetime.now(dt.timezone.utc)
    step_sec = bucket_minutes * 60
    end_idx = int(now.timestamp()) // step_sec
    starts: List[dt.datetime] = []
    values: List[float] = []
    for i in range(n_buckets):
        bucket_idx = end_idx - (n_buckets - 1 - i)
        t = bucket_idx * step_sec
        hour = dt.datetime.fromtimestamp(t, dt.timezone.utc).hour
        hour += dt.datetime.fromtimestamp(t, dt.timezone.utc).minute / 60.0
        daylight = max(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))  # 06:00–18:00
        base = 6.0 * daylight * (bucket_minutes / 15.0)  # ~24 Wh/15m at noon
        noise = rng.normal(0.0, max(0.3, 0.1 * base))
        starts.append(dt.datetime.fromtimestamp(t, dt.timezone.utc))
        values.append(max(0.0, base + noise))
    return ProofSeries(
        bucket_minutes=int(bucket_minutes),
        starts=starts,
        values=np.array(values, dtype=float),
        device_id="offline-sim",
        source="offline",
        raw_proof_count=len(values),
    )
