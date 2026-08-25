"""Global aggregator (HFL level 3).

Consumes signed regional contributions (level 2), verifies region
signatures, drops outlier regions (MAD) and produces the global model via
reputation-weighted FedAvg. The global model is *read-only* — it proposes,
it never signs or acts (constitution).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.fed.aggregate import fed_avg
from agent.hfl.weights import reputation_map


@dataclass
class GlobalResult:
    """Outcome of one global aggregation round."""

    weights: List[float]
    loss: Optional[float]
    accepted_count: int
    rejected: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GlobalAggregator:
    """Aggregates regional contributions into the global model."""

    domain: str = "energy"
    version: str = "model_1.0"

    def aggregate(
        self,
        regional_contributions: List[Dict[str, Any]],
        *,
        ers_map: Optional[Dict[str, float]] = None,
        min_samples: int = 1,
        z_threshold: float = 3.0,
    ) -> GlobalResult:
        extra = reputation_map(regional_contributions, ers_map)
        result = fed_avg(
            regional_contributions,
            verify=True,
            min_samples=min_samples,
            z_threshold=z_threshold,
            extra_weight=extra,
        )
        return GlobalResult(
            weights=result.weights,
            loss=result.loss,
            accepted_count=result.accepted_count,
            rejected=result.rejected,
        )
