"""Digital training tests (agent.digital_train) — model, state, pipeline."""

import random

import numpy as np
import pytest

from agent.digital_train.data import build_matrix
from agent.digital_train.model import TimeSeriesModel
from agent.digital_train.pipeline import run_once
from agent.digital_train.state import load, save
from agent.fed.protocol import generate_keypair, verify_contribution


def _ar_series(n=120, a=0.8, b=2.0, noise=0.2, seed=3, outlier_at=None):
    """Autoregressive series y[t] = a*y[t-1] + b + noise."""
    rng = random.Random(seed)
    values = [b / (1 - a) + rng.gauss(0, noise)]
    for _ in range(1, n):
        y = a * values[-1] + b + rng.gauss(0, noise)
        values.append(y)
    if outlier_at is not None:
        values[outlier_at] *= 20.0
    return np.array(values).reshape(-1, 1)


# ── data ────────────────────────────────────────────────────────────────────


def test_build_matrix_shape_and_columns():
    samples = [
        {"ts": "2026-01-01T00:00:00Z", "metrics": {"a": 1.0, "b": 2.0}},
        {"ts": "2026-01-01T01:00:00Z", "metrics": {"a": 2.0}},  # b missing → ffill
        {"ts": "2026-01-01T02:00:00Z", "metrics": {"b": 5.0}},  # a missing → ffill
    ]
    matrix, columns = build_matrix(samples)
    assert columns == ["a", "b"]
    assert matrix.shape == (3, 2)
    assert matrix[0].tolist() == [1.0, 2.0]
    assert matrix[1].tolist() == [2.0, 2.0]  # b forward-filled
    assert matrix[2].tolist() == [2.0, 5.0]  # a forward-filled


def test_build_matrix_empty():
    matrix, columns = build_matrix([])
    assert matrix.shape == (0, 0)
    assert columns == []


# ── model ───────────────────────────────────────────────────────────────────


def test_model_learns_ar_trend():
    series = _ar_series(n=120)
    model = TimeSeriesModel(lag=3).fit(series, columns=["trend"])

    pred = model.predict_next(series[:-1])[0]
    actual = series[-1, 0]
    relative_error = abs(pred - actual) / max(abs(actual), 1e-9)
    assert relative_error < 0.15, f"pred={pred:.3f} actual={actual:.3f}"


def test_model_predicts_multiple_steps():
    series = _ar_series(n=120)
    model = TimeSeriesModel(lag=3).fit(series, columns=["trend"])
    forecast = model.predict(series, steps=3)
    assert forecast.shape == (3, 1)
    # A forecast must be finite and in a sane range.
    assert np.isfinite(forecast).all()
    assert np.abs(forecast - series[-1, 0]).mean() < 50.0


def test_model_detects_anomaly():
    clean = _ar_series(n=80)
    model = TimeSeriesModel(lag=3).fit(clean, columns=["trend"])
    assert model.detect_anomaly(clean) == {}

    spiked = _ar_series(n=80, outlier_at=79)
    anomalies = model.detect_anomaly(spiked)
    assert "trend" in anomalies


def test_model_cross_domain_correlation():
    t = np.linspace(0, 6 * np.pi, 100)
    two_domain = np.column_stack([np.sin(t), 2.0 * np.sin(t)])
    model = TimeSeriesModel(lag=2).fit(two_domain, columns=["sun", "shadow"])
    corr = model.correlations(two_domain)
    assert abs(corr[0, 1]) > 0.99


def test_model_serialization_roundtrip():
    series = _ar_series(n=60)
    model = TimeSeriesModel(lag=3).fit(series, columns=["trend"])
    restored = TimeSeriesModel.from_dict(model.to_dict())
    assert restored.columns == ["trend"]
    assert restored.flatten_weights() == model.flatten_weights()


def test_model_flatten_weights_shape():
    series = np.column_stack([_ar_series(n=60, seed=1)[:, 0], _ar_series(n=60, seed=2)[:, 0]])
    model = TimeSeriesModel(lag=2).fit(series, columns=["a", "b"])
    # per column: lag*K weights + 1 intercept → 2 * (2*2+1) = 10
    assert len(model.flatten_weights()) == 10


# ── state & pipeline ────────────────────────────────────────────────────────


def test_state_roundtrip(tmp_path):
    state_path = str(tmp_path / "state.json")
    result = run_once(state_path, offline=True, points=48)
    assert result["trained"] is True

    state = load(state_path)
    assert state.round == 1
    assert state.trained is True
    assert state.model is not None
    assert len(state.metrics["columns"]) >= 6  # six domains of metrics


def test_pipeline_offline_trains_sane_model(tmp_path):
    state_path = str(tmp_path / "digital.json")
    result = run_once(state_path, offline=True, points=48)
    assert result["trained"] is True
    assert result["samples"] == 48
    assert result["columns"]
    assert result["mse_last_step"] >= 0.0
    # Anomalies / correlations are computed, not random noise.
    assert isinstance(result["anomalies"], dict)
    assert isinstance(result["top_correlations"], list)
    assert np.isfinite(result["mse_last_step"])


def test_pipeline_not_enough_data(tmp_path):
    state_path = str(tmp_path / "tiny.json")
    result = run_once(state_path, offline=True, points=5)
    assert result["trained"] is False
    assert "not enough data" in result["reason"]


def test_pipeline_signed_contribution(tmp_path):
    state_path = str(tmp_path / "signed.json")
    secret_key, public_key = generate_keypair()
    result = run_once(state_path, offline=True, points=48, secret_key=secret_key)
    assert result["trained"] is True
    assert result["contribution_signed"] is True

    contribution = result["contribution"]
    assert contribution["public_key"] == public_key
    assert verify_contribution(public_key, contribution) is True


def test_pipeline_second_run_increments_round(tmp_path):
    state_path = str(tmp_path / "state.json")
    first = run_once(state_path, offline=True, points=48)
    second = run_once(state_path, offline=True, points=48)
    assert second["round"] == first["round"] + 1
    assert second["trained"] is True
