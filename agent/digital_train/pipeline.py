"""Autonomous digital self-training pipeline.

Runs headlessly: collect digital feeds → build the series matrix → fit the
domain-agnostic model → update the persisted state → (optionally) sign the
weights as a federated contribution (``agent.fed.protocol``).

CLI:
    python -m agent.digital_train.pipeline --once --offline --points 48
    python -m agent.digital_train.pipeline --loop --interval 3600 --state digital_state.json
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import numpy as np

from agent.digital_feeds import FeedResult, collect, collect_series
from agent.digital_train.data import build_matrix
from agent.digital_train.model import TimeSeriesModel
from agent.digital_train.state import DigitalState, load, save
from agent.fed.protocol import SCHEMA, public_key_from_secret, sign_contribution

#: Minimum number of series points before the model will train.
MIN_TRAIN_POINTS = 12

NODE_ID = "digital_node_0"


def build_contribution(
    model: TimeSeriesModel,
    columns: List[str],
    samples: int,
    loss: float,
    round_no: int,
) -> Dict[str, Any]:
    """Shape the trained weights as a federated contribution (axis-fed)."""
    return {
        "schema": SCHEMA,
        "round": round_no,
        "device_id": NODE_ID,
        "weights": model.flatten_weights(),
        "samples": samples,
        "loss": round(float(loss), 6),
        "nonce": round_no,
    }


def _top_correlations(corr: np.ndarray, columns: List[str], top_n: int = 3) -> List[Dict[str, float]]:
    pairs = []
    k = len(columns)
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append(
                {
                    "left": columns[i],
                    "right": columns[j],
                    "correlation": round(float(corr[i, j]), 3),
                }
            )
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return pairs[:top_n]


def run_once(
    state_path: str,
    feeds: Optional[List[str]] = None,
    offline: bool = True,
    points: int = 48,
    secret_key: Optional[str] = None,
    seed: int = 7,
) -> Dict[str, Any]:
    """One self-training cycle. Idempotent and safe to call on a schedule."""
    state = load(state_path)

    if offline:
        samples = collect_series(feeds=feeds, points=points, offline=True, seed=seed)
        training = [s.to_dict() for s in samples]
    else:
        samples = collect(feeds=feeds, offline=False)
        state.history.extend(s.to_dict() for s in samples)
        training = state.history

    matrix, columns = build_matrix(training)

    if matrix.shape[0] < MIN_TRAIN_POINTS or matrix.shape[1] == 0:
        save(state_path, state)
        return {
            "trained": False,
            "samples": int(matrix.shape[0]),
            "columns": columns,
            "reason": "not enough data yet — keep collecting",
        }

    model = TimeSeriesModel().fit(matrix, columns)

    # One-step-ahead quality (normalized, comparable across domains).
    mse = model.last_step_mse(matrix)

    anomalies = model.detect_anomaly(matrix)
    corr_pairs = _top_correlations(model.correlations(matrix), columns)

    metrics = {
        "samples": int(matrix.shape[0]),
        "columns": columns,
        "mse_last_step": round(mse, 6),
        "anomalies": anomalies,
        "top_correlations": corr_pairs,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    contribution = build_contribution(
        model, columns, samples=int(matrix.shape[0]), loss=mse, round_no=state.round
    )
    signed = secret_key is not None
    if signed:
        contribution["public_key"] = public_key_from_secret(secret_key)
        contribution = sign_contribution(secret_key, contribution)

    state.model = model.to_dict()
    state.round += 1
    state.metrics = metrics
    save(state_path, state)

    return {
        "trained": True,
        "round": state.round,
        "samples": int(matrix.shape[0]),
        "columns": columns,
        "mse_last_step": round(mse, 6),
        "anomalies": anomalies,
        "top_correlations": corr_pairs,
        "contribution_signed": signed,
        "contribution": contribution,
    }


if __name__ == "__main__":
    # Allow `python -m agent.digital_train.pipeline` (not just the package).
    from agent.digital_train.__main__ import main as _cli_main

    _cli_main()
