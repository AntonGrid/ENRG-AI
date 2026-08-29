"""ERS loop tests — AI anomaly signals → on-chain reputation severity."""

from agent.fed.ers_loop import (
    anomaly_severity,
    apply_anomaly_penalty,
    ers_loop_step,
)
from agent.signals import Signal


def _anomaly(residual_wh=5.0, threshold_wh=2.0):
    return Signal(
        kind="generation_anomaly",
        domain="pilot",
        ts="2026-08-29T12:00:00+00:00",
        source="oracle",
        value=residual_wh,
        meta={"residual_wh": residual_wh, "threshold_wh": threshold_wh},
    ).to_dict()


def test_no_anomalies_means_no_report():
    step = ers_loop_step([], current_score=800)
    assert step["severity"] == 0
    assert step["should_report"] is False
    assert step["score_after"] == step["score_before"]


def test_single_anomaly_produces_severity():
    step = ers_loop_step([_anomaly(residual_wh=5.0, threshold_wh=2.0)], current_score=800)
    # strength = min(2.0, 5/2) = 2.0 → severity = 1 + 2 = 3
    assert step["severity"] == 3
    assert step["should_report"] is True
    # 3 * 5% = 15% cut → 800 - 120 = 680
    assert step["score_after"] == 680


def test_many_anomalies_push_severity_up():
    signals = [_anomaly(residual_wh=2.5, threshold_wh=2.0) for _ in range(4)]
    severity = anomaly_severity(signals)
    assert severity >= 4
    assert severity <= 10


def test_penalty_mirrors_onchain_semantics():
    # Mirrors the Anchor unit tests: severity 1 → −5%, severity 10 → −50%.
    assert apply_anomaly_penalty(800, 1) == 760
    assert apply_anomaly_penalty(800, 10) == 400
    assert apply_anomaly_penalty(10, 10) == 5
