"""ERS reputation economy (Phase 2, Proof-of-Intelligence).

Quality has a price (constitution C-6): after every aggregation round the
aggregator adjusts each contributor's ERS — accepted contributions gain,
outlier-rejected ones decay toward a floor. ERS then feeds back into the next
round as a sample-weight multiplier (`agent.hfl.weights.reputation_map`), so
trusted gateways shape the global model more — without changing the wire
format (a contributor cannot fake its own weight).

This closes the loop: contribution → FedAvg + MAD screening → ERS update →
weighted next round.
"""
from __future__ import annotations

from typing import Any, Dict

from agent.fed.aggregate import FedResult

#: ERS a brand-new contributor starts at.
DEFAULT_ERS = 0.5
#: Floor ERS after repeated rejections (keeps a sliver of voice for retry).
ERS_FLOOR = 0.1


def update_ers(
    ers_map: Dict[str, float],
    result: FedResult,
    *,
    gain: float = 0.05,
    decay: float = 0.5,
) -> Dict[str, float]:
    """Update ERS after one aggregation round.

    - **Accepted** contributions: ``ers += gain * (1 - ers)`` — soft rise to 1.
    - **Rejected as outliers** (``outlier_loss`` / ``outlier_weights``):
      ``ers = max(ERS_FLOOR, ers * decay)`` — quality penalty.
    - Signature-invalid or insufficient-samples rejections are **not** a
      quality judgement, so ERS is left untouched.

    Returns a new map; the input is not mutated.
    """
    ers = dict(ers_map)

    for contribution in result.accepted:
        device_id = contribution.get("device_id")
        if not device_id:
            continue
        current = float(ers.get(device_id, DEFAULT_ERS))
        ers[device_id] = round(current + gain * (1.0 - current), 4)

    for rejection in result.rejected:
        reason = str(rejection.get("reason", ""))
        if not reason.startswith("outlier_"):
            continue
        device_id = rejection.get("device_id")
        if not device_id:
            continue
        current = float(ers.get(device_id, DEFAULT_ERS))
        ers[device_id] = round(max(ERS_FLOOR, current * decay), 4)

    return ers


def summary(ers_map: Dict[str, float]) -> Dict[str, Any]:
    """Compact human-readable snapshot of the ERS map."""
    if not ers_map:
        return {"devices": 0, "mean_ers": 0.0, "min_ers": 0.0, "max_ers": 0.0}
    values = [float(v) for v in ers_map.values()]
    return {
        "devices": len(values),
        "mean_ers": round(sum(values) / len(values), 4),
        "min_ers": round(min(values), 4),
        "max_ers": round(max(values), 4),
    }
