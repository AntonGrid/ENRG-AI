"""Local training on a gateway (Phase 2).

A gateway trains a *light* model on its own data (an ESP32 only runs the
inference; training lives on the gateway host — Raspberry Pi / farm PC).
The model is a simple linear regressor: ``y = w0 + w1*x1 + ...``. Gradients
never leave the device; only the trained ``weights`` + ``samples`` + ``loss``
are signed and sent to the aggregator (privacy: raw data stays local).
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

DEFAULT_LR = 0.05
DEFAULT_EPOCHS = 200


def predict(weights: Sequence[float], features: Sequence[float]) -> float:
    """Linear prediction: ``weights[0]`` is the bias term."""
    return weights[0] + sum(
        w * x for w, x in zip(weights[1:], features)
    )


def evaluate(weights: Sequence[float], rows: List[Dict[str, Any]]) -> float:
    """Mean squared error of ``weights`` over ``rows``."""
    if not rows:
        return 0.0
    errors = [
        predict(weights, row["features"]) - row["label"]
        for row in rows
    ]
    return sum(e * e for e in errors) / len(rows)


def train_local(
    rows: List[Dict[str, Any]],
    lr: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = 0,
) -> Dict[str, Any]:
    """Train a linear model on the gateway's local rows.

    Args:
        rows: list of ``{"features": [...], "label": float}``.
        lr: learning rate (plain SGD).
        epochs: number of passes over the local data.
        seed: RNG seed for reproducibility (weight init + row order).

    Returns:
        ``{"weights": [...], "samples": len(rows), "loss": mse}`` — the
        minimal payload that gets signed into a contribution.
    """
    if not rows:
        return {"weights": [], "samples": 0, "loss": 0.0}

    dim = len(rows[0]["features"]) + 1  # bias + features

    rng = random.Random(seed)
    weights = [rng.uniform(-0.5, 0.5) for _ in range(dim)]
    order = list(range(len(rows)))

    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            row = rows[i]
            x = [1.0] + list(row["features"])
            err = predict(weights, row["features"]) - row["label"]
            for j in range(dim):
                weights[j] -= lr * err * x[j]

    return {
        "weights": [round(w, 6) for w in weights],
        "samples": len(rows),
        "loss": round(evaluate(weights, rows), 6),
    }


def make_rows(
    features: List[List[float]],
    labels: List[float],
) -> List[Dict[str, Any]]:
    """Bundle parallel feature/label lists into the row shape ``train_local`` wants."""
    return [
        {"features": list(f), "label": float(y)}
        for f, y in zip(features, labels)
    ]
