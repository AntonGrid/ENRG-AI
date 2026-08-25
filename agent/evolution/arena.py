"""A/B arena — compares two recommender configurations on the closed-loop pilot."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from agent.pilot.sim import PilotConfig, PilotResult, run_pilot


def run_ab_test(
    baseline_params: Dict[str, Any],
    candidate_params: Dict[str, Any],
    pilot_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[PilotResult, PilotResult]:
    """Run the pilot with two parameter sets; returns (baseline, candidate).

    ``pilot_kwargs`` controls the arena size (default: small for speed).
    """
    kwargs = {"n_devices": 5, "hours": 48, "seed": 7}
    if pilot_kwargs:
        kwargs.update(pilot_kwargs)

    base_cfg = PilotConfig(**kwargs)
    base_cfg.recommend_params = dict(baseline_params or {})
    baseline = run_pilot(base_cfg)

    cand_cfg = PilotConfig(**kwargs)
    cand_cfg.recommend_params = dict(candidate_params or {})
    candidate = run_pilot(cand_cfg)

    return baseline, candidate
