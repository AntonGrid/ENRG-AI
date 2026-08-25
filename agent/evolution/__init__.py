"""DAO-driven model evolution (GLOBAL_AI_ARCHITECTURE §8.2).

The system improves itself: experiments are proposed, gated by staked DAO
voting with quorum, A/B-tested on the closed-loop pilot, and canonicalized
into the live configuration — or rolled back. The full history is kept in an
append-only ledger (an experiment cannot be rewritten).

- ``dao``        — staked voting, quorum, majority, append-only ledger;
- ``experiment`` — experiment lifecycle + canonicalize (A/B verdict);
- ``arena``      — A/B test runner on the DePIN pilot;
- ``loop``       — one evolution round end-to-end (demo: ``python -m agent.evolution.loop``).
"""

from agent.evolution.arena import run_ab_test
from agent.evolution.dao import DAOGovernance, Proposal, Voter
from agent.evolution.experiment import Experiment, canonicalize

__all__ = [
    "DAOGovernance",
    "Experiment",
    "Proposal",
    "Voter",
    "canonicalize",
    "run_ab_test",
]
