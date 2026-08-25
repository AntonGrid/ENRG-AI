"""Reputation-weighted aggregation helpers (HFL).

Reputation (ERS) shapes how much a contribution counts: a trusted gateway's
sample weight is boosted, a new/flagged one keeps only a small floor so the
network can still bootstrap. Weights are applied as multipliers in
``agent.fed.aggregate.fed_avg(extra_weight=...)`` — the wire format stays
untouched, so a contributor cannot fake its weight.
"""
from __future__ import annotations

from typing import Dict, List, Optional

#: Minimum sample multiplier for brand-new (ERS == 0) contributors.
MIN_WEIGHT = 0.1


def reputation_weight(ers: float, alpha: float = 1.0) -> float:
    """Map an ERS score in ``[0, 1]`` to a sample multiplier in
    ``[MIN_WEIGHT, 1.0]``."""
    ers = max(0.0, min(1.0, float(ers)))
    return MIN_WEIGHT + (1.0 - MIN_WEIGHT) * (ers ** alpha)


def reputation_map(
    contributions: List[Dict],
    ers_map: Optional[Dict[str, float]] = None,
    alpha: float = 1.0,
) -> Optional[Dict[str, float]]:
    """Build ``{device_id: multiplier}`` from an optional ERS map.

    Returns ``None`` when no ERS data is provided (no-op for ``fed_avg``).
    Unknown devices default to ``ers=0`` (floor weight).
    """
    if not ers_map:
        return None
    return {
        c.get("device_id", ""): reputation_weight(
            ers_map.get(c.get("device_id"), 0.0), alpha=alpha
        )
        for c in contributions
    }
