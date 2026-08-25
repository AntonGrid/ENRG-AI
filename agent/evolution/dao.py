"""DAO governance simulation — proposals, staked voting, quorum.

Implements the "evolution without a founder" loop (GLOBAL_AI_ARCHITECTURE
§8.2): a model/aggregator proposes an experiment (hyperparameters), the DAO
votes with stake, and only approved experiments are A/B-tested. Every step
is recorded in an append-only history ledger.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Voter:
    voter_id: str
    stake: float


@dataclass
class Proposal:
    proposal_id: str
    title: str
    params: Dict[str, Any]
    proposer: str
    status: str = "proposed"  # proposed | approved | rejected
    votes_for: float = 0.0
    votes_against: float = 0.0


class DAOGovernance:
    """Staked voting with quorum & majority thresholds."""

    def __init__(self, quorum: float = 0.5, majority: float = 0.5) -> None:
        self.quorum = quorum
        self.majority = majority
        self.voters: Dict[str, Voter] = {}
        self.proposals: Dict[str, Proposal] = {}
        self.history: List[Dict[str, Any]] = []  # append-only ledger
        self._seq = 0

    @property
    def total_stake(self) -> float:
        return sum(v.stake for v in self.voters.values())

    def add_voter(self, voter_id: str, stake: float) -> None:
        self.voters[voter_id] = Voter(voter_id=voter_id, stake=stake)

    def propose(self, title: str, params: Dict[str, Any], proposer: str) -> Proposal:
        self._seq += 1
        proposal = Proposal(
            proposal_id=f"prop_{self._seq}",
            title=title,
            params=dict(params),
            proposer=proposer,
        )
        self.proposals[proposal.proposal_id] = proposal
        self.record({"event": "proposed", "proposal": proposal.proposal_id, "title": title, "params": dict(params), "proposer": proposer, "ts": int(time.time())})
        return proposal

    def vote(self, proposal_id: str, voter_id: str, approve: bool) -> None:
        """Cast a staked vote (one vote per voter per proposal)."""
        proposal = self.proposals[proposal_id]
        voter = self.voters[voter_id]
        if approve:
            proposal.votes_for += voter.stake
        else:
            proposal.votes_against += voter.stake
        self.record({"event": "vote", "proposal": proposal_id, "voter": voter_id, "approve": approve, "stake": voter.stake, "ts": int(time.time())})

    def tally(self, proposal_id: str) -> bool:
        """Tally votes; returns True (approved) only with quorum + majority."""
        proposal = self.proposals[proposal_id]
        if proposal.status != "proposed":
            return proposal.status == "approved"

        total = self.total_stake
        cast = proposal.votes_for + proposal.votes_against
        if cast < self.quorum * total:
            return False  # not enough stake participated — stays proposed

        approved = proposal.votes_for > self.majority * cast
        proposal.status = "approved" if approved else "rejected"
        self.record({"event": "tally", "proposal": proposal_id, "approved": approved, "votes_for": proposal.votes_for, "votes_against": proposal.votes_against, "ts": int(time.time())})
        return approved

    def record(self, entry: Dict[str, Any]) -> None:
        self.history.append(entry)

    def history_size(self) -> int:
        return len(self.history)
