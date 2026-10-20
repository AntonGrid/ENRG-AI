"""Coherence signals: shape, the opt-in policy gate, and the signed envelope.

The point of this file is that the physics enters the ecosystem through the
doors that already exist: the Policy Engine's duck-typed AI gate (no new flag,
no new reason code) and the oracle's signed ``SignalBundle`` (no new wire
format). If either door changes, this file says so.
"""

import json

import pytest

from agent.physics.park import observe_park
from agent.physics.signal import CoherenceSignal, coherence_bundle, park_signals
from agent.physics.synthetic import build_scenario


@pytest.fixture(scope="module")
def observation():
    scenario = build_scenario()
    return observe_park(scenario.energy, scenario.device_ids, scenario.starts)


# ── the signal ──────────────────────────────────────────────────────────────


def test_one_signal_per_producer(observation):
    signals = park_signals(observation)
    assert len(signals) == len(observation.device_ids)
    for signal in signals:
        assert isinstance(signal, CoherenceSignal)
        assert 0.0 <= signal.score <= 1.0
        assert signal.is_anomaly == (signal.score >= observation.config.threshold)
        assert isinstance(signal.details["signals"], list)


def test_the_flagged_producers_are_the_ones_that_changed(observation):
    flagged = {signal.device_id for signal in park_signals(observation) if signal.is_anomaly}
    assert flagged == {"dev-02", "dev-05", "dev-06"}


# ── the policy gate (Axis-core, opt-in, unchanged) ──────────────────────────


def test_policy_gate_is_off_by_default_and_opt_in(observation):
    engine = pytest.importorskip("axis_core.policy.engine")
    models = pytest.importorskip("axis_core.policy.models")
    signal = next(s for s in park_signals(observation) if s.device_id == "dev-05")

    off = engine.PolicyEngine.evaluate_ai_anomaly(
        policy=models.PolicyRegistry(), signal=signal
    )
    assert off.allowed is True

    on = engine.PolicyEngine.evaluate_ai_anomaly(
        policy=models.PolicyRegistry(enforce_ai_anomaly=True), signal=signal
    )
    assert on.allowed is False
    assert on.reason == "ai_anomaly_flagged"
    assert on.details["ai_signals"] == ["phase", "memory"]


def test_a_quiet_producer_passes_the_gate(observation):
    engine = pytest.importorskip("axis_core.policy.engine")
    models = pytest.importorskip("axis_core.policy.models")
    quiet = next(s for s in park_signals(observation) if s.device_id == "dev-07")
    decision = engine.PolicyEngine.evaluate_ai_anomaly(
        policy=models.PolicyRegistry(enforce_ai_anomaly=True), signal=quiet
    )
    assert decision.allowed is True


# ── the envelope the oracle already signs ───────────────────────────────────


def test_bundle_carries_the_park_and_the_physics(observation):
    payload = coherence_bundle(observation, source="offline").to_dict()
    assert set(payload) == {"generated_at", "meta", "signals"}
    assert payload["meta"]["physics"]["engine"] == "entrain"
    assert payload["meta"]["physics"]["source_file"] == "entrain/julia/kuramoto/basic.jl"
    assert payload["meta"]["park"]["producers"] == len(observation.device_ids)
    assert len(payload["meta"]["devices"]) == len(observation.device_ids)
    assert [signal["kind"] for signal in payload["signals"]] == ["coherence"] * 3
    json.dumps(payload)  # JSON-serializable without extra handling


def test_bundle_sign_and_verify_and_tamper(observation):
    from agent.fed.protocol import generate_keypair
    from agent.signals import sign_bundle, verify_bundle_signature

    secret, public = generate_keypair()
    signed = sign_bundle(coherence_bundle(observation), secret)
    assert verify_bundle_signature(public, signed)
    signed["message"]["meta"]["park"]["coherence_mean"] = 0.0
    assert not verify_bundle_signature(public, signed)


# ── the CLI ─────────────────────────────────────────────────────────────────


def test_cli_writes_a_bundle(tmp_path):
    from agent.physics.__main__ import main

    out = tmp_path / "coherence.json"
    assert main(["--source", "offline", "--scenario", "mixed", "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["meta"]["source"] == "offline"
    assert len(payload["signals"]) == 3


def test_cli_signs_and_the_signature_verifies(monkeypatch, tmp_path):
    from agent.fed.protocol import generate_keypair
    from agent.physics.__main__ import main
    from agent.signals import verify_bundle_signature

    secret, public = generate_keypair()
    monkeypatch.setenv("AXIS_AI_SIGNING_KEY", secret)
    out = tmp_path / "coherence.json"
    assert main(["--scenario", "mixed", "--sign", "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["public_key"] == public
    assert verify_bundle_signature(payload["public_key"], payload)


def test_cli_sign_requires_key(monkeypatch):
    from agent.physics.__main__ import main

    monkeypatch.delenv("AXIS_AI_SIGNING_KEY", raising=False)
    assert main(["--scenario", "mixed", "--sign"]) == 2


def test_cli_online_falls_back_rather_than_failing(monkeypatch, tmp_path):
    from agent.physics.__main__ import main

    def _boom(*args, **kwargs):
        raise RuntimeError("oracle unreachable")

    monkeypatch.setattr("agent.forecast.energy.fetch_oracle_proofs", _boom)
    out = tmp_path / "online.json"
    assert main(["--source", "online", "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["meta"]["source"] == "offline-fallback"
    assert payload["meta"]["warnings"]
