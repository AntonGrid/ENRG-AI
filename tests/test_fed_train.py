"""Local gateway training tests (agent.fed.local_train)."""

import random

import pytest

from agent.fed.local_train import evaluate, make_rows, predict, train_local


def _linear_rows(n=60, w_true=(2.0, 3.0), seed=1):
    rng = random.Random(seed)
    features = []
    labels = []
    for _ in range(n):
        x = rng.random()
        features.append([x])
        labels.append(w_true[0] + w_true[1] * x)
    return make_rows(features, labels)


def test_predict_linear():
    # y = 2 + 3 * 1 = 5
    assert predict([2.0, 3.0], [1.0]) == pytest.approx(5.0)


def test_train_learns_linear_target():
    rows = _linear_rows()
    trained = train_local(rows, epochs=300, seed=0)
    assert trained["samples"] == len(rows)
    # A clean linear target must be learned almost perfectly.
    assert trained["loss"] < 0.01
    assert trained["weights"][0] == pytest.approx(2.0, abs=0.2)
    assert trained["weights"][1] == pytest.approx(3.0, abs=0.2)


def test_train_reports_samples_and_loss():
    trained = train_local(_linear_rows(n=10), epochs=10)
    assert trained["samples"] == 10
    assert 0.0 <= trained["loss"] < 1_000.0


def test_train_empty_rows_returns_empty():
    trained = train_local([])
    assert trained == {"weights": [], "samples": 0, "loss": 0.0}


def test_evaluate_mse():
    rows = [
        {"features": [1.0], "label": 5.0},
        {"features": [2.0], "label": 8.0},
    ]
    perfect = evaluate([2.0, 3.0], rows)
    assert perfect == pytest.approx(0.0)

    bad = evaluate([0.0, 0.0], rows)
    assert bad == pytest.approx((25.0 + 64.0) / 2.0)
