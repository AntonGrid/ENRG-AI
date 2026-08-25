"""Evolution loop — DAO-gated A/B experiments that improve the live config.

One round (GLOBAL_AI_ARCHITECTURE §8.2):

    propose → DAO vote (stake, quorum) → approved?
        → A/B test on the pilot (baseline vs candidate)
        → canonicalize the winner or roll back
        → record everything in the append-only ledger

Run the demo: ``python -m agent.evolution.loop``.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from agent.evolution.arena import run_ab_test
from agent.evolution.dao import DAOGovernance
from agent.evolution.experiment import Experiment, canonicalize


def run_evolution_round(
    dao: DAOGovernance,
    *,
    title: str,
    candidate_params: Dict[str, Any],
    baseline_params: Optional[Dict[str, Any]] = None,
    proposer: str = "model",
    vote_map: Optional[Dict[str, bool]] = None,
    pilot_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run one full evolution round: DAO gate → A/B → canonicalize → record.

    ``vote_map``: ``{voter_id: approve}``; ``None`` means every registered
    voter approves (the demo default). Passing an empty dict keeps the
    proposal pending (no quorum).
    """
    baseline_params = baseline_params or {}
    proposal = dao.propose(title=title, params=candidate_params, proposer=proposer)

    if vote_map is None:
        vote_map = {voter_id: True for voter_id in dao.voters}
    for voter_id, approve in vote_map.items():
        if voter_id in dao.voters:
            dao.vote(proposal.proposal_id, voter_id, approve)

    approved = dao.tally(proposal.proposal_id)
    if not approved:
        dao.record({"event": "skipped_ab", "proposal": proposal.proposal_id, "reason": "not_approved"})
        return {"proposal": proposal.proposal_id, "approved": False, "status": proposal.status}

    baseline, candidate = run_ab_test(
        baseline_params=baseline_params,
        candidate_params=candidate_params,
        pilot_kwargs=pilot_kwargs,
    )

    experiment = Experiment(
        experiment_id=proposal.proposal_id,
        title=title,
        params=candidate_params,
    )
    winner = canonicalize(
        experiment,
        baseline_value=baseline.net_reward_usd,
        candidate_value=candidate.net_reward_usd,
    )

    dao.record(
        {
            "event": "ab_result",
            "proposal": proposal.proposal_id,
            "baseline_reward": baseline.net_reward_usd,
            "candidate_reward": candidate.net_reward_usd,
            "status": experiment.status,
        }
    )

    return {
        "proposal": proposal.proposal_id,
        "approved": True,
        "status": experiment.status,
        "baseline_reward": round(baseline.net_reward_usd, 3),
        "candidate_reward": round(candidate.net_reward_usd, 3),
        "winner_params": winner,
    }


def _demo() -> None:
    dao = DAOGovernance(quorum=0.5, majority=0.5)
    for i, stake in enumerate((10.0, 20.0, 30.0, 15.0, 25.0)):
        dao.add_voter(f"voter_{i}", stake)

    print("=== DAO evolution round 1: sell stored earlier (min_sell_storage_ratio 0.1) ===")
    result = run_evolution_round(
        dao,
        title="sell stored energy earlier",
        candidate_params={"min_sell_storage_ratio": 0.1},
        proposer="aggregator_alpha",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== DAO evolution round 2: aggressive sell premium (10x) — should roll back ===")
    result2 = run_evolution_round(
        dao,
        title="aggressive sell premium",
        candidate_params={"sell_premium": 10.0},
        proposer="aggregator_beta",
    )
    print(json.dumps(result2, ensure_ascii=False, indent=2))

    print("\n=== Append-only ledger (last 5 events) ===")
    for entry in dao.history[-5:]:
        print(" ", entry["event"], entry.get("proposal", ""), entry.get("status", ""))


if __name__ == "__main__":
    _demo()
