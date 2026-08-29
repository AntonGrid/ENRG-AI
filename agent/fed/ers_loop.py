"""ERS loop — AI anomaly signals → on-chain reputation (Phase 2 / ADR-0010 L3).

The on-chain side already exists in the ENRG program: ``report_anomaly``
(trusted-oracle-only) applies ``apply_anomaly_penalty(score, severity)`` —
severity 1..=10 cuts the score by 5% per level (severity 10 = −50%). This
module is the **off-chain decision helper**: it turns a window of AI anomaly
signals (``agent.signals``) into a severity and computes the resulting ERS,
so the oracle's collector job can decide whether to call ``report_anomaly``
(rate-limited, oracle-only — constitution C-4/ADR-0010).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

#: Mirrors on-chain constants.rs ERS_MAX_SCORE.
ERS_MAX_SCORE = 1000


def anomaly_severity(signals: List[Dict[str, Any]], max_severity: int = 10) -> int:
    """Map a window of ``generation_anomaly`` signals to severity 1..=10.

    Rules (advisory — the Policy Engine still decides whether to act):
    - 0 anomalies  → 0 (no report).
    - Severity grows with the number of anomalies and their strength
      (residual vs. threshold ratio from ``meta``).
    """
    anomalies = [s for s in signals if s.get("kind") == "generation_anomaly"]
    if not anomalies:
        return 0

    strength = 0.0
    for a in anomalies:
        meta = a.get("meta") or {}
        threshold = float(meta.get("threshold_wh") or 0.0)
        residual = abs(float(meta.get("residual_wh") or 0.0))
        if threshold > 0:
            strength = max(strength, min(2.0, residual / threshold))
    # strength ∈ [1..2] for a single strong anomaly; more anomalies push up.
    severity = 1 + int(round(strength)) + max(0, len(anomalies) - 1)
    return max(1, min(max_severity, severity))


def apply_anomaly_penalty(score: int, severity: int) -> int:
    """Mirror of the on-chain ``apply_anomaly_penalty`` (state/reputation.rs)."""
    sev = max(1, min(10, int(severity)))
    cut_percent = min(50, 5 * sev)
    return score - (score * cut_percent) // 100


def ers_loop_step(
    signals: List[Dict[str, Any]],
    current_score: int,
) -> Dict[str, Any]:
    """One ERS-loop decision from a window of AI signals.

    Returns:
        ``{"severity": int, "should_report": bool, "score_before": int,
           "score_after": int, "anomaly_count": int}``
    """
    severity = anomaly_severity(signals)
    score_before = max(0, min(ERS_MAX_SCORE, int(current_score)))
    should_report = severity > 0
    score_after = (
        apply_anomaly_penalty(score_before, severity) if should_report else score_before
    )
    return {
        "severity": severity,
        "should_report": should_report,
        "score_before": score_before,
        "score_after": score_after,
        "anomaly_count": sum(
            1 for s in signals if s.get("kind") == "generation_anomaly"
        ),
    }


def severity_from_bundle(bundle: Any) -> int:
    """Convenience: severity from a ``SignalBundle`` (agent.signals)."""
    return anomaly_severity([s.to_dict() for s in bundle.signals])
