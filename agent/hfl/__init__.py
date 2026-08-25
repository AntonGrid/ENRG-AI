"""Hierarchical Federated Learning (HFL) — regions → global model.

Levels (see ``GLOBAL_AI_ARCHITECTURE.md`` §4):

- **L1 gateways** — train locally, sign contributions (``agent.fed``);
- **L2 regions** — ``RegionAggregator``: verify + MAD-outlier removal +
  reputation-weighted FedAvg → signed ``axis-fed/2`` regional contribution;
- **L3 global** — ``GlobalAggregator``: aggregate regional contributions into
  the read-only global model.

Modules: ``protocol`` (axis-fed/2), ``weights`` (ERS reputation),
``region``, ``global_aggregator``, ``simulate``.
"""

from agent.hfl.global_aggregator import GlobalAggregator, GlobalResult
from agent.hfl.protocol import SCHEMA_V2, make_contribution, sign_regional
from agent.hfl.region import RegionAggregator, RegionResult
from agent.hfl.weights import reputation_map, reputation_weight

__all__ = [
    "SCHEMA_V2",
    "GlobalAggregator",
    "GlobalResult",
    "RegionAggregator",
    "RegionResult",
    "make_contribution",
    "reputation_map",
    "reputation_weight",
    "sign_regional",
]
