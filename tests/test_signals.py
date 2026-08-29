"""Hybrid AI oracle signal tests (agent.signals) — offline/deterministic."""

import datetime as dt
import json

import numpy as np

from agent.forecast.energy import ProofSeries, synthetic_solar_series
from agent.signals import (
    SignalBundle,
    collect_all,
    generation_signals,
    main,
    market_signals,
)


def _series(values, bucket_minutes=15):
    step = bucket_minutes * 60
    base = int(dt.datetime(2026, 8, 29, 12, 0, tzinfo=dt.timezone.utc).timestamp())
    starts = [
        dt.datetime.fromtimestamp(base - (len(values) - 1 - i) * step, dt.timezone.utc)
        for i in range(len(values))
    ]
    return ProofSeries(
        bucket_minutes=bucket_minutes,
        starts=starts,
        values=np.array(values, dtype=float),
        device_id="test",
        source="offline",
    )


# ── generation signals ───────────────────────────────────────────────────────


def test_generation_signals_produce_forecast_with_bands():
    series = synthetic_solar_series(bucket_minutes=15, n_buckets=24)
    signals = generation_signals(series, horizon_steps=6)
    forecasts = [s for s in signals if s.kind == "generation_forecast"]
    assert len(forecasts) == 6
    for s in forecasts:
        assert s.unit == "Wh"
        assert s.interval_low <= s.value <= s.interval_high
        assert s.meta["step_ahead"] >= 1
        assert s.ts  # ISO label


def test_generation_anomaly_flags_spike():
    series = _series([5.0] * 6 + [25.0])  # last point is a clear spike
    signals = generation_signals(series, horizon_steps=2)
    anomalies = [s for s in signals if s.kind == "generation_anomaly"]
    assert len(anomalies) == 1
    assert anomalies[0].meta["observed_wh"] == 25.0


def test_generation_anomaly_absent_on_steady_series():
    series = _series([5.0] * 8)
    signals = generation_signals(series, horizon_steps=2)
    anomalies = [s for s in signals if s.kind == "generation_anomaly"]
    assert anomalies == []


# ── market signals ───────────────────────────────────────────────────────────


def test_market_signals_cover_all_sources():
    from agent.signals import _offline_market_points

    points = _offline_market_points(points_n=24)
    signals = market_signals(points, steps=4)
    sources = {s.meta["market_source"] for s in signals}
    assert sources == {"dayahead", "p2p", "spot"}
    for s in signals:
        assert s.unit == "usd_per_kwh"
        assert 0.001 < s.value < 1.0
        assert s.interval_low <= s.value <= s.interval_high
        assert len(s.meta["path"]) == 4


# ── full bundle / CLI ────────────────────────────────────────────────────────


def test_collect_all_offline_bundle():
    bundle = collect_all(source="offline", horizon_steps=6, market_steps=3)
    kinds = {s.kind for s in bundle.signals}
    assert "generation_forecast" in kinds
    assert "market_forecast" in kinds
    assert bundle.meta["observed_buckets"] == 24
    assert bundle.meta["generation_unit"] == "Wh"
    assert len(bundle.by_kind("market_forecast")) == 3


def test_bundle_serialization_round_trip():
    bundle = collect_all(source="offline", horizon_steps=4)
    payload = bundle.to_dict()
    assert set(payload) == {"generated_at", "meta", "signals"}
    assert json.dumps(payload)  # JSON-serializable without extra handling


def test_cli_offline_json(tmp_path):
    out = tmp_path / "signals.json"
    rc = main(["--source", "offline", "--horizon", "4", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["meta"]["source"] == "offline"
    assert payload["signals"]


# ── signature / verifiability (constitution C-3) ────────────────────────────


def test_bundle_sign_and_verify():
    import agent.fed.protocol as proto
    from agent.signals import sign_bundle, verify_bundle_signature

    secret, pub = proto.generate_keypair()
    bundle = collect_all(source="offline", horizon_steps=3)
    signed = sign_bundle(bundle, secret)
    assert signed["signature"]
    assert verify_bundle_signature(pub, signed)


def test_bundle_signature_detects_tampering():
    import agent.fed.protocol as proto
    from agent.signals import sign_bundle, verify_bundle_signature

    secret, pub = proto.generate_keypair()
    bundle = collect_all(source="offline", horizon_steps=3)
    signed = sign_bundle(bundle, secret)
    signed["message"]["meta"]["observed_buckets"] = 999  # tamper
    assert not verify_bundle_signature(pub, signed)


def test_online_fallback_when_oracle_unreachable(monkeypatch):
    from agent.signals import collect_all

    def _boom(*args, **kwargs):
        raise RuntimeError("oracle unreachable")

    monkeypatch.setattr("agent.signals.fetch_oracle_proofs", _boom)
    bundle = collect_all(source="online", horizon_steps=3)
    assert bundle.meta["source"] == "offline-fallback"
    assert bundle.meta["requested_source"] == "online"
    assert len(bundle.by_kind("generation_forecast")) == 3


def test_cli_sign_requires_key():
    from agent.signals import main

    assert main(["--source", "offline", "--sign"]) == 2


def test_cli_sign_produces_verifiable_attestation(monkeypatch, tmp_path):
    import agent.fed.protocol as proto
    from agent.signals import main, verify_bundle_signature

    secret, pub = proto.generate_keypair()
    monkeypatch.setenv("AXIS_AI_SIGNING_KEY", secret)
    out = tmp_path / "assessments.json"
    rc = main(
        [
            "--source",
            "offline",
            "--horizon",
            "3",
            "--sign",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["public_key"] == pub
    assert payload["signature"]
    assert verify_bundle_signature(payload["public_key"], payload)


def test_publish_script_roundtrip(monkeypatch, tmp_path):
    import agent.fed.protocol as proto
    from agent.signals import verify_bundle_signature
    from scripts.publish_signals import main as publish_main

    secret, pub = proto.generate_keypair()
    monkeypatch.setenv("AXIS_AI_SIGNING_KEY", secret)
    out = tmp_path / "ai" / "assessments.json"
    rc = publish_main(["--source", "offline", "--horizon", "3", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["public_key"] == pub
    assert verify_bundle_signature(pub, payload)
