"""Federated learning (Phase 2).

- ``protocol`` — signed contribution wire format (Ed25519, canonical JSON);
- ``local_train`` — light model trained on the gateway, raw data never leaves;
- ``aggregate`` — FedAvg with signature verification and outlier removal;
- ``ers`` — reputation economy (quality → ERS, ERS → aggregation weight);
- ``ers_loop`` — AI anomaly signals → on-chain ``report_anomaly`` severity;
- ``digest`` — contribution digests & on-chain commitment contract (PoI);
- ``simulate`` — N-gateway demo (``python -m agent.fed.simulate``).
"""

from agent.fed.aggregate import FedResult, fed_avg
from agent.fed.digest import (
    build_commitment,
    contribution_digest,
    sign_commitment,
    verify_commitment,
)
from agent.fed.ers import update_ers
from agent.fed.ers_loop import anomaly_severity, apply_anomaly_penalty, ers_loop_step
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
    "build_commitment",
    "canonical_contribution_message",
    "contribution_digest",
    "evaluate",
    "fed_avg",
    "generate_keypair",
    "predict",
    "sign_commitment",
    "sign_contribution",
    "train_local",
    "update_ers",
    "anomaly_severity",
    "apply_anomaly_penalty",
    "ers_loop_step",
    "verify_commitment",
    "verify_contribution",
]
