"""Hierarchical federated simulation — gateways → regions → global model.

Run with ``python -m agent.hfl.simulate``. A pool of honest regions learns a
shared signal (each with a small regional shift); a malicious gateway is
rejected at its region, a malicious region is rejected globally; the global
model generalizes better than the average regional model.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from agent.fed.local_train import evaluate, train_local
from agent.fed.protocol import generate_keypair, sign_contribution
from agent.hfl.global_aggregator import GlobalAggregator
from agent.hfl.protocol import SCHEMA_V2
from agent.hfl.region import RegionAggregator

#: Shared hidden signal: y = 2.0 + 3.0 * x (+ local noise).
TRUE_WEIGHTS = [2.0, 3.0]
#: Malicious signal (gateways / whole regions trying to poison the model).
MALICIOUS_WEIGHTS = [-10.0, 50.0]


def _synth_rows(n: int, w_true: List[float], seed: int, noise: float = 0.15):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.random()
        label = w_true[0] + w_true[1] * x + rng.gauss(0.0, noise)
        rows.append({"features": [x], "label": label})
    return rows


def _gateway_contribution(
    *,
    device_id: str,
    region: str,
    w_true: List[float],
    samples: int,
    seed: int,
    round_no: int,
) -> Dict[str, Any]:
    secret, public = generate_keypair()
    trained = train_local(_synth_rows(samples, w_true, seed), seed=seed + 1000)
    contribution = {
        "schema": SCHEMA_V2,
        "round": round_no,
        "device_id": device_id,
        "weights": trained["weights"],
        "samples": trained["samples"],
        "loss": trained["loss"],
        "nonce": seed,
        "level": 1,
        "domain": "energy",
        "region": region,
        "version": "model_1.0",
        "public_key": public,
    }
    return sign_contribution(secret, contribution)


def run_simulation(
    n_regions: int = 5,
    gateways_per_region: int = 5,
    samples_per_gateway: int = 80,
    include_malicious_gateway: bool = True,
    include_malicious_region: bool = True,
    test_size: int = 200,
    seed: int = 7,
) -> Dict[str, Any]:
    """Run one hierarchical round; returns comparison metrics."""
    test_rows = _synth_rows(test_size, TRUE_WEIGHTS, seed=seed + 1, noise=0.0)

    regional_contributions: List[Dict[str, Any]] = []
    regional_weights: List[List[float]] = []
    local_weights: List[List[float]] = []

    for r in range(n_regions):
        region = f"eu-{r:02d}"
        # Centered regional shift around the global signal.
        shift = 0.15 * (r - (n_regions - 1) / 2)
        w_region = [TRUE_WEIGHTS[0] + shift, TRUE_WEIGHTS[1] - shift * 0.5]

        malicious_region = include_malicious_region and r == n_regions - 1
        secret_key, _ = generate_keypair()
        aggregator = RegionAggregator(region=region, domain="energy", secret_key=secret_key, round_no=1)

        gateway_contributions: List[Dict[str, Any]] = []
        for g in range(gateways_per_region):
            malicious_gw = (
                include_malicious_gateway and not malicious_region and g == gateways_per_region - 1
            )
            w_gw = MALICIOUS_WEIGHTS if malicious_gw else w_region
            gateway_contributions.append(
                _gateway_contribution(
                    device_id=f"gw_{region}_{g:02d}",
                    region=region,
                    w_true=w_gw,
                    samples=samples_per_gateway,
                    seed=seed + 10 * r + g,
                    round_no=1,
                )
            )
            local_weights.append(
                train_local(_synth_rows(samples_per_gateway, w_gw, seed + 10 * r + g), seed=seed + 1000 + 10 * r + g)["weights"]
            )

        if malicious_region:
            # Whole region is malicious: every gateway trains on the bad signal.
            gateway_contributions = [
                _gateway_contribution(
                    device_id=f"gw_{region}_{g:02d}",
                    region=region,
                    w_true=MALICIOUS_WEIGHTS,
                    samples=samples_per_gateway,
                    seed=seed + 10 * r + g,
                    round_no=1,
                )
                for g in range(gateways_per_region)
            ]

        region_result = aggregator.aggregate(gateway_contributions)
        regional_weights.append(region_result.weights)
        if region_result.regional_contribution is not None:
            regional_contributions.append(region_result.regional_contribution)

    global_result = GlobalAggregator(domain="energy").aggregate(regional_contributions)

    local_avg_loss = sum(evaluate(w, test_rows) for w in local_weights) / len(local_weights)
    regional_avg_loss = sum(evaluate(w, test_rows) for w in regional_weights) / len(regional_weights)
    global_loss = evaluate(global_result.weights, test_rows)

    return {
        "n_regions": n_regions,
        "gateways_per_region": gateways_per_region,
        "global_accepted": global_result.accepted_count,
        "global_rejected": [r["device_id"] + ":" + r["reason"] for r in global_result.rejected],
        "global_weights": global_result.weights,
        "global_loss_on_test": round(global_loss, 4),
        "regional_avg_loss_on_test": round(regional_avg_loss, 4),
        "local_avg_loss_on_test": round(local_avg_loss, 4),
        "improved_vs_regional": global_loss < regional_avg_loss,
        "improved_vs_local": global_loss < local_avg_loss,
    }


def main() -> None:
    print("=" * 62)
    print("Hierarchical federated simulation (axis-fed/2)")
    print("=" * 62)
    summary = run_simulation()
    for key, value in summary.items():
        print(f"{key:28s}: {value}")
    print("-" * 62)
    print("✅ Global model beats regional and local models." if summary["improved_vs_regional"] else "⚠️ Check the simulation parameters.")
    print("Rejected globally:", summary["global_rejected"])


if __name__ == "__main__":
    main()

