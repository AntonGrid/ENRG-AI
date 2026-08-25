"""Simulation: N gateways → signed contributions → FedAvg → better forecast.

Run with ``python -m agent.fed.simulate``. A pool of honest gateways learns
a shared linear signal; one malicious gateway contributes outlier weights and
gets rejected; the global model generalizes better than the average local
model on a held-out test set.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from agent.fed.aggregate import fed_avg
from agent.fed.local_train import evaluate, train_local
from agent.fed.protocol import generate_keypair, sign_contribution

#: Shared hidden signal: y = 2.0 + 3.0 * x (+ local noise).
TRUE_WEIGHTS = [2.0, 3.0]
#: The malicious gateway learns a completely different (outlier) signal.
MALICIOUS_WEIGHTS = [-10.0, 50.0]


def _synth_rows(n: int, w_true: List[float], seed: int, noise: float = 0.15):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.random()
        label = w_true[0] + w_true[1] * x + rng.gauss(0.0, noise)
        rows.append({"features": [x], "label": label})
    return rows


def run_simulation(
    n_gateways: int = 5,
    samples_per_gateway: int = 120,
    include_malicious: bool = True,
    test_size: int = 200,
    seed: int = 7,
) -> Dict[str, Any]:
    """Run one federated round and compare global vs local generalization."""
    rng = random.Random(seed)
    test_rows = _synth_rows(test_size, TRUE_WEIGHTS, seed=seed + 1, noise=0.0)

    contributions: List[Dict[str, Any]] = []
    local_weights: List[List[float]] = []

    for i in range(n_gateways):
        malicious = include_malicious and i == n_gateways - 1
        w_true = MALICIOUS_WEIGHTS if malicious else TRUE_WEIGHTS
        rows = _synth_rows(samples_per_gateway, w_true, seed=seed + 10 + i)

        trained = train_local(rows, seed=seed + 20 + i)
        local_weights.append(trained["weights"])

        secret_b64, public_b64 = generate_keypair()
        contribution = {
            "schema": "axis-fed/1",
            "round": 1,
            "device_id": f"gateway_{i:02d}",
            "weights": trained["weights"],
            "samples": trained["samples"],
            "loss": trained["loss"],
            "nonce": i,
            "public_key": public_b64,
        }
        contributions.append(sign_contribution(secret_b64, contribution))

    result = fed_avg(contributions)

    local_avg_loss = sum(evaluate(w, test_rows) for w in local_weights) / len(local_weights)
    global_loss = evaluate(result.weights, test_rows)

    summary = {
        "n_gateways": n_gateways,
        "accepted": result.accepted_count,
        "rejected": [r["device_id"] + ":" + r["reason"] for r in result.rejected],
        "global_weights": result.weights,
        "global_loss_on_test": round(global_loss, 4),
        "local_avg_loss_on_test": round(local_avg_loss, 4),
        "improved": global_loss < local_avg_loss,
    }
    return summary


def main() -> None:
    print("=" * 62)
    print("Federated learning simulation (axis-fed/1)")
    print("=" * 62)
    summary = run_simulation()
    for key, value in summary.items():
        print(f"{key:28s}: {value}")
    print("-" * 62)
    verdict = (
        "✅ Global model generalizes better than the average local model."
        if summary["improved"]
        else "⚠️ Global model did not beat local models — tune the simulation."
    )
    print(verdict)
    print("Malicious gateway rejected:", summary["rejected"])


if __name__ == "__main__":
    main()
