"""Federated learning (Phase 2).

- ``protocol`` — signed contribution wire format (Ed25519, canonical JSON);
- ``local_train`` — light model trained on the gateway, raw data never leaves;
- ``aggregate`` — FedAvg with signature verification and outlier removal;
- ``simulate`` — N-gateway demo (``python -m agent.fed.simulate``).
"""

from agent.fed.aggregate import FedResult, fed_avg
from agent.fed.local_train import evaluate, predict, train_local
from agent.fed.protocol import (
    SCHEMA,
    canonical_contribution_message,
    generate_keypair,
    sign_contribution,
    verify_contribution,
)

__all__ = [
    "SCHEMA",
    "FedResult",
    "canonical_contribution_message",
    "evaluate",
    "fed_avg",
    "generate_keypair",
    "predict",
    "sign_contribution",
    "train_local",
    "verify_contribution",
]
