"""DAO evolution tests (agent.evolution)."""

import pytest

pytest.importorskip("axis_core")

from agent.evolution.arena import run_ab_test
from agent.evolution.dao import DAOGovernance
from agent.evolution.experiment import Experiment, canonicalize
from agent.evolution.loop import run_evolution_round


def _dao() -> DAOGovernance:
    dao = DAOGovernance(quorum=0.5, majority=0.5)
    for i, stake in enumerate((10.0, 20.0, 30.0, 40.0)):  # total 100
        dao.add_voter(f"voter_{i}", stake)
    return dao


# ── DAO governance ──────────────────────────────────────────────────────────


def test_dao_approves_with_quorum_and_majority():
    dao = _dao()
    proposal = dao.propose("experiment A", {"sell_premium": 1.05}, "model")
    for voter in dao.voters:
        dao.vote(proposal.proposal_id, voter, approve=True)
    assert dao.tally(proposal.proposal_id) is True
    assert proposal.status == "approved"


def test_dao_not_approved_without_quorum():
    dao = _dao()
    proposal = dao.propose("experiment B", {"sell_premium": 1.05}, "model")
    dao.vote(proposal.proposal_id, "voter_0", approve=True)  # 10 < 50 quorum
    assert dao.tally(proposal.proposal_id) is False
    assert proposal.status == "proposed"


def test_dao_rejects_with_majority_against():
    dao = _dao()
    proposal = dao.propose("experiment C", {"sell_premium": 1.05}, "model")
    for voter in ("voter_0", "voter_1", "voter_3"):  # 70 against
        dao.vote(proposal.proposal_id, voter, approve=False)
    dao.vote(proposal.proposal_id, "voter_2", approve=True)  # 30 for
    assert dao.tally(proposal.proposal_id) is False
    assert proposal.status == "rejected"


def test_dao_ledger_is_append_only():
    dao = _dao()
    proposal = dao.propose("x", {"a": 1}, "model")
    before = dao.history_size()
    dao.vote(proposal.proposal_id, "voter_0", approve=True)
    assert dao.history_size() == before + 1


# ── canonicalization ────────────────────────────────────────────────────────


def test_canonicalize_promotes_winner():
    experiment = Experiment(experiment_id="e1", title="t", params={"sell_premium": 1.05})
    winner = canonicalize(experiment, baseline_value=10.0, candidate_value=12.0)
    assert experiment.status == "canonicalized"
    assert winner == {"sell_premium": 1.05}


def test_canonicalize_rolls_back_loser():
    experiment = Experiment(experiment_id="e2", title="t", params={"sell_premium": 1.05})
    winner = canonicalize(experiment, baseline_value=12.0, candidate_value=10.0)
    assert experiment.status == "rolled_back"
    assert winner == {}


# ── A/B arena & evolution loop ──────────────────────────────────────────────


def test_ab_test_is_deterministic():
    baseline, candidate = run_ab_test({}, {})
    assert baseline.net_reward_usd == candidate.net_reward_usd


def test_evolution_round_rolls_back_worse_candidate():
    dao = _dao()
    result = run_evolution_round(
        dao,
        title="aggressive premium",
        candidate_params={"sell_premium": 10.0},
        proposer="model",
        pilot_kwargs={"n_devices": 5, "hours": 48, "seed": 7},
    )
    assert result["approved"] is True
    assert result["status"] == "rolled_back"
    assert result["winner_params"] == {}
    assert result["candidate_reward"] < result["baseline_reward"]


def test_evolution_round_requires_dao_approval():
    dao = _dao()  # nobody votes → no quorum → experiment not run
    result = run_evolution_round(
        dao,
        title="no votes",
        candidate_params={"sell_premium": 1.05},
        vote_map={},
        pilot_kwargs={"n_devices": 5, "hours": 48, "seed": 7},
    )
    assert result["approved"] is False
    assert result["status"] == "proposed"
