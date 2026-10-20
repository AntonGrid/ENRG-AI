"""Proof stream → park matrix: the only I/O in the physics layer.

The oracle serves proofs ``{device_id, ts, energy_wh, nonce, mint_tx,
mint_status}`` (ADR-0010). A park reads a rectangular matrix: one row per
producer, one column per bucket. A bucket with no proof is zero — for an
energy series "reported nothing" and "produced nothing" are the same column
here, and pretending otherwise would invent data.

Bucketing is delegated to ``agent.forecast.energy.aggregate_proofs`` so the
physics layer reads the stream exactly the way the forecaster does.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from agent.forecast.energy import Proof, aggregate_proofs


@dataclass
class ParkMatrix:
    """Producers × buckets, with the bucket timestamps that label the columns."""

    device_ids: List[str]
    starts: List[dt.datetime]
    energy: np.ndarray
    proof_count: int = 0

    @property
    def devices(self) -> int:
        return int(self.energy.shape[0])

    @property
    def buckets(self) -> int:
        return int(self.energy.shape[1])


def park_matrix(
    proofs: Sequence[Proof],
    bucket_minutes: int = 60,
    min_devices: int = 2,
) -> ParkMatrix:
    """Lay a proof stream out as one energy matrix (producers × buckets).

    Args:
        proofs: verified proofs, any order.
        bucket_minutes: width of one bucket; the park's clock divides it.
        min_devices: a park is a relation, so at least two producers.

    Raises:
        ValueError: fewer producers than ``min_devices``, or no proofs.
    """
    if not proofs:
        raise ValueError("no proofs to read")
    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    rows: Dict[str, List[Proof]] = {}
    for proof in proofs:
        rows.setdefault(proof.device_id, []).append(proof)
    if len(rows) < min_devices:
        raise ValueError(
            f"a park needs at least {min_devices} producers, got {len(rows)}"
        )

    step = int(bucket_minutes) * 60
    grid_by_device: Dict[str, tuple] = {}
    indices = []
    for device_id in sorted(rows):
        series = aggregate_proofs(rows[device_id], bucket_minutes)
        index = np.array([int(start.timestamp()) // step for start in series.starts])
        grid_by_device[device_id] = (index, np.asarray(series.values, dtype=float))
        indices.append(index)

    all_index = np.unique(np.concatenate(indices))
    first, last = int(all_index.min()), int(all_index.max())
    grid = np.arange(first, last + 1)
    energy = np.zeros((len(grid_by_device), grid.size))
    for row, device_id in enumerate(sorted(grid_by_device)):
        index, values = grid_by_device[device_id]
        energy[row, index - first] = values

    starts = [
        dt.datetime.fromtimestamp(int(bucket) * step, tz=dt.timezone.utc)
        for bucket in grid
    ]
    return ParkMatrix(
        device_ids=sorted(grid_by_device),
        starts=starts,
        energy=energy,
        proof_count=len(proofs),
    )
