"""Experiment model + canonicalization logic (DAO evolution)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Experiment:
    """One evolution experiment (a candidate parameter set under test)."""

    experiment_id: str
    title: str
    params: Dict[str, Any]  # candidate hyperparameters (e.g. Recommender)
    domain: str = "energy"
    metric: str = "net_reward_usd"
    status: str = "proposed"  # proposed|approved|rejected|tested|canonicalized|rolled_back
    baseline_value: Optional[float] = None
    candidate_value: Optional[float] = None

    def outcome(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "status": self.status,
            "metric": self.metric,
            "baseline": self.baseline_value,
            "candidate": self.candidate_value,
        }


def canonicalize(
    experiment: Experiment,
    baseline_value: float,
    candidate_value: float,
) -> Dict[str, Any]:
    """Decide whether the candidate replaces the baseline (A/B verdict).

    Returns the canonicalized parameters dict (to become the active config)
    or an empty dict when the candidate is rolled back.
    """
    if candidate_value > baseline_value:
        experiment.status = "canonicalized"
        experiment.baseline_value = baseline_value
        experiment.candidate_value = candidate_value
        return dict(experiment.params)
    experiment.status = "rolled_back"
    experiment.baseline_value = baseline_value
    experiment.candidate_value = candidate_value
    return {}
