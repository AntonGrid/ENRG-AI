"""Regional aggregator (HFL level 2).

Collects signed gateway contributions (level 1), verifies signatures, drops
outliers (MAD), aggregates with reputation-weighted FedAvg — and emits a
*signed regional contribution* (level 2) that the global aggregator can
consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.fed.aggregate import FedResult, fed_avg
from agent.hfl.protocol import DEFAULT_VERSION, make_contribution, sign_regional
from agent.hfl.weights import reputation_map


@dataclass
class RegionResult:
    """Outcome of one regional aggregation round."""

    region: str
    weights: List[float]
    loss: Optional[float]
    accepted_count: int
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    #: Signed ``axis-fed/2`` level-2 contribution (None when nothing accepted).
    regional_contribution: Optional[Dict[str, Any]] = None


@dataclass
class RegionAggregator:
    """Aggregates gateway contributions for one region."""

    region: str
    domain: str = "energy"
    version: str = DEFAULT_VERSION
    secret_key: Optional[str] = None
    round_no: int = 0

    def aggregate(
        self,
        contributions: List[Dict[str, Any]],
        *,
        ers_map: Optional[Dict[str, float]] = None,
        min_samples: int = 1,
        z_threshold: float = 3.0,
    ) -> RegionResult:
        """Aggregate gateways → regional weights + signed level-2 contribution."""
        extra = reputation_map(contributions, ers_map)
        result: FedResult = fed_avg(
            contributions,
            verify=True,
            min_samples=min_samples,
            z_threshold=z_threshold,
            extra_weight=extra,
        )

        regional_contribution = None
        if result.weights:
            samples = sum(int(c["samples"]) for c in result.accepted)
            regional_contribution = make_contribution(
                level=2,
                domain=self.domain,
                region=self.region,
                device_id=f"region_{self.region}",
                weights=result.weights,
                samples=samples,
                loss=result.loss or 0.0,
                round_no=self.round_no,
                version=self.version,
            )
            if self.secret_key:
                regional_contribution = sign_regional(self.secret_key, regional_contribution)

        return RegionResult(
            region=self.region,
            weights=result.weights,
            loss=result.loss,
            accepted_count=result.accepted_count,
            rejected=result.rejected,
            regional_contribution=regional_contribution,
        )
