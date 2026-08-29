"""Forecast module tests — offline/deterministic paths (no network)."""

import datetime as dt

import numpy as np
import pytest

from agent.forecast import (
    HoltTrend,
    Proof,
    ProofSeries,
    aggregate_proofs,
    forecast_energy,
    synthetic_solar_series,
)
from agent.forecast.__main__ import build_series, main


def _proofs_for_buckets(bucket_minutes: int, wh_per_bucket) -> list[Proof]:
    """Synthetic proofs: one per bucket, each carrying ``wh`` energy."""
    step = bucket_minutes * 60
    base = 1_700_000_000  # fixed epoch for determinism
    proofs = []
    for i, wh in enumerate(wh_per_bucket):
        proofs.append(
            Proof(
                device_id="0xabc",
                ts=base + i * step,
                energy_wh=wh,
                nonce=i + 1,
            )
        )
    return proofs


# ── aggregation ─────────────────────────────────────────────────────────────


def test_aggregate_sums_wh_per_bucket():
    proofs = _proofs_for_buckets(15, [1.0, 1.0, 1.0])
    series = aggregate_proofs(proofs, bucket_minutes=15)
    assert series.bucket_minutes == 15
    assert len(series.values) == 3
    assert np.allclose(series.values, [1.0, 1.0, 1.0])
    assert series.total_wh == pytest.approx(3.0)
    assert series.raw_proof_count == 3


def test_aggregate_merges_proofs_in_same_bucket():
    step = 15 * 60
    base = 1_700_000_000
    proofs = [
        Proof("0xabc", ts=base, energy_wh=1, nonce=1),
        Proof("0xabc", ts=base + 60, energy_wh=1, nonce=2),  # same 15-min window
        Proof("0xabc", ts=base + step, energy_wh=2, nonce=3),
    ]
    series = aggregate_proofs(proofs, bucket_minutes=15)
    assert np.allclose(series.values, [2.0, 2.0])


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_proofs([], bucket_minutes=15)


# ── Holt model ──────────────────────────────────────────────────────────────


def test_holt_constant_series_forecasts_flat():
    model = HoltTrend()
    y = np.full(20, 10.0)
    model.fit(y)
    fc = model.forecast(5)
    assert np.allclose(fc, 10.0, atol=1e-6)
    assert model.residual_rmse < 1e-9


def test_holt_trend_series_continues_trend():
    model = HoltTrend()
    y = np.arange(1.0, 11.0) * 3.0  # linear: 3, 6, …, 30
    model.fit(y)
    fc = model.forecast(3)
    # A perfect linear trend should keep the slope (~3 per step).
    assert np.allclose(fc, [33.0, 36.0, 39.0], atol=1.0)


def test_holt_intervals_widen_with_horizon():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.5, 0.1, 40))
    model = HoltTrend()
    model.fit(y)
    half = model.interval(6)
    assert np.all(np.diff(half) > 0)  # strictly widening
    assert np.all(half > 0)


def test_forecast_energy_interval_ordering_and_meta():
    series = aggregate_proofs(_proofs_for_buckets(15, [1.0] * 6), 15)
    result = forecast_energy(series, horizon_steps=4, interval="p80")
    assert len(result.labels) == 4
    assert np.all(result.low_wh <= result.point_wh + 1e-9)
    assert np.all(result.point_wh <= result.high_wh + 1e-9)
    assert result.meta["model"] == "holt-linear-trend"
    assert result.meta["observed_buckets"] == 6
    assert result.meta["units"] == "Wh"
    payload = result.to_dict()
    assert payload["horizon_steps"] == 4
    assert len(payload["buckets"]) == 4


def test_forecast_energy_clips_lower_bound_at_zero():
    series = aggregate_proofs(_proofs_for_buckets(15, [0.0] * 6), 15)
    result = forecast_energy(series, horizon_steps=4)
    assert np.all(result.low_wh >= 0.0)


# ── CLI / offline path ──────────────────────────────────────────────────────


def test_offline_series_shape_and_positive():
    series = synthetic_solar_series(bucket_minutes=15, n_buckets=24)
    assert len(series.values) == 24
    assert np.all(series.values >= 0)
    assert series.source == "offline"


def test_build_series_offline():
    series = build_series(bucket_minutes=15, source="offline", limit=100)
    assert len(series.values) > 0


def test_cli_offline_json(tmp_path):
    out = tmp_path / "fc.json"
    rc = main(["--source", "offline", "--horizon", "4", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    import json

    payload = json.loads(out.read_text())
    assert payload["horizon_steps"] == 4
    assert payload["meta"]["model"] == "holt-linear-trend"
