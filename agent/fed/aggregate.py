"""Federated aggregation — FedAvg with signed contributions (Phase 2).

The aggregator (ENRG-AI) never sees raw device data — only signed weight
updates. It:

1. **Verifies** every contribution signature (rejects unsigned/tampered);
2. **Drops outliers** — contributions whose loss or any weight is far from
   the median (by ``z_threshold`` standard deviations);
3. **Averages** the surviving weights, weighted by the number of local
   samples (FedAvg).

Rejected contributions are returned with a ``reason`` so the aggregator can
adjust device reputation — the incentive for *quality* is reputation, not a
separate mint (SRC is minted only for verified energy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional

from agent.fed.protocol import verify_contribution


@dataclass
class FedResult:
    """Outcome of one aggregation round."""

    round: Optional[int]
    weights: List[float]
    loss: Optional[float]
    n_contributions: int
    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


def _outlier_flags(values: List[float], k: float = 3.0) -> List[bool]:
    """Robust MAD-based outlier flags (median absolute deviation).

    A value is an outlier when it deviates from the median by more than
    ``k * 1.4826 * mad``. Unlike a mean/std z-score this stays robust when a
    single large outlier inflates the variance (which is exactly the attack
    a malicious gateway would try).
    """
    med = median(values)
    mad = median([abs(v - med) for v in values])
    scale = 1.4826 * mad
    if scale == 0:
        # Degenerate case (all values identical): use a tiny relative scale
        # so a single far-away value is still caught.
        scale = 1.4826 * 1e-6 * (1.0 + abs(med))
    return [abs(v - med) > k * scale for v in values]


def _weights_dim(contributions: List[Dict[str, Any]]) -> int:
    return len(contributions[0]["weights"])


def fed_avg(
    contributions: List[Dict[str, Any]],
    *,
    verify: bool = True,
    min_samples: int = 1,
    z_threshold: float = 3.0,
) -> FedResult:
    """Aggregate signed contributions into a global model (FedAvg).

    Args:
        contributions: list of signed contribution dicts (each carrying
            ``public_key`` when ``verify`` is enabled).
        verify: require a valid Ed25519 signature per contribution.
        min_samples: drop contributions trained on fewer local samples.
        z_threshold: outlier cutoff in standard deviations (loss & weights).

    Returns:
        ``FedResult`` with the global ``weights`` (empty when nothing was
        accepted) and the per-contribution accept/reject detail.
    """
    if not contributions:
        return FedResult(round=None, weights=[], loss=None, n_contributions=0)

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    # 1. Signature verification (constitution: every contribution checkable).
    for contribution in contributions:
        if verify:
            public_key = contribution.get("public_key")
            if not isinstance(public_key, str):
                rejected.append({**contribution, "reason": "missing_public_key"})
                continue
            if not verify_contribution(public_key, contribution):
                rejected.append({**contribution, "reason": "signature_invalid"})
                continue
        if contribution.get("samples", 0) < min_samples:
            rejected.append({**contribution, "reason": "insufficient_samples"})
            continue
        accepted.append(contribution)

    # 2. Outlier removal (only when there is a meaningful population).
    if len(accepted) >= 3:
        dim = _weights_dim(accepted)
        loss_flags = _outlier_flags([c["loss"] for c in accepted], k=z_threshold)
        weight_flags = [
            _outlier_flags([c["weights"][j] for c in accepted], k=z_threshold)
            for j in range(dim)
        ]

        kept: List[Dict[str, Any]] = []
        for i, contribution in enumerate(accepted):
            if loss_flags[i]:
                rejected.append({**contribution, "reason": "outlier_loss"})
                continue
            if any(weight_flags[j][i] for j in range(dim)):
                rejected.append({**contribution, "reason": "outlier_weights"})
                continue
            kept.append(contribution)
        accepted = kept

    if not accepted:
        rnd = contributions[0].get("round")
        return FedResult(round=rnd, weights=[], loss=None, n_contributions=len(contributions), rejected=rejected)

    # 3. Sample-weighted average (FedAvg).
    dim = _weights_dim(accepted)
    total_samples = sum(c["samples"] for c in accepted)
    weights: List[float] = []
    for j in range(dim):
        weighted = sum(c["weights"][j] * c["samples"] for c in accepted)
        weights.append(round(weighted / total_samples, 6))

    global_loss = sum(c["loss"] * c["samples"] for c in accepted) / total_samples

    return FedResult(
        round=accepted[0].get("round"),
        weights=weights,
        loss=round(global_loss, 6),
        n_contributions=len(contributions),
        accepted=accepted,
        rejected=rejected,
    )
