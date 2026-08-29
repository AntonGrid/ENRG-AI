"""PoI leaderboard — per-round view of contributors, quality and ERS.

Runs a multi-round simulation: honest gateways contribute near the shared
signal, one malicious gateway keeps contributing outliers. Every round:

1. contributions are signed and FedAvg-aggregated with ERS-weighted samples;
2. outliers are rejected (MAD);
3. ERS is updated (accepted gain, rejected-outliers decay) — the next round's
   weights follow.

CLI:
    python -m agent.leaderboard --rounds 5 --gateways 5
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.fed.aggregate import FedResult, fed_avg
from agent.fed.digest import build_commitment, contribution_digest
from agent.fed.ers import update_ers
from agent.fed.protocol import generate_keypair, sign_contribution
from agent.hfl.weights import reputation_map

#: Shared hidden signal weights (the honest gateways learn around).
TRUE_WEIGHTS = [2.0, 3.0]
#: The malicious gateway's outlier weights.
MALICIOUS_WEIGHTS = [-10.0, 50.0]


@dataclass
class LeaderboardRow:
    """One contributor's standing after a round."""

    device_id: str
    samples: int
    accepted: int = 0
    rejected: int = 0
    last_loss: float = 0.0
    weight_multiplier: float = 1.0
    ers: float = 0.5
    malicious: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "malicious": self.malicious,
            "samples": self.samples,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "last_loss": round(self.last_loss, 4),
            "weight_multiplier": round(self.weight_multiplier, 3),
            "ers": round(self.ers, 4),
        }


@dataclass
class Leaderboard:
    """Snapshot of one aggregation round."""

    round_no: int
    rows: List[LeaderboardRow] = field(default_factory=list)
    n_contributions: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    global_loss: float = 0.0

    def sorted_rows(self) -> List[LeaderboardRow]:
        """Rank by ERS desc, then accepted desc."""
        return sorted(
            self.rows,
            key=lambda r: (r.ers, r.accepted),
            reverse=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_no,
            "n_contributions": self.n_contributions,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "global_loss": round(self.global_loss, 4),
            "rows": [r.to_dict() for r in self.sorted_rows()],
        }


def _make_contribution(
    device_id: str,
    weights: List[float],
    samples: int,
    loss: float,
    round_no: int,
) -> Dict[str, Any]:
    secret_b64, public_b64 = generate_keypair()
    contribution = {
        "schema": "axis-fed/1",
        "round": round_no,
        "device_id": device_id,
        "weights": weights,
        "samples": samples,
        "loss": round(loss, 6),
        "nonce": round_no * 1000,
        "public_key": public_b64,
    }
    return sign_contribution(secret_b64, contribution)

def _round_contributions(
    round_no: int,
    n_gateways: int,
    include_malicious: bool,
    seed: int,
) -> tuple:
    rng = random.Random(seed + round_no * 100)
    contributions: List[Dict[str, Any]] = []
    ers_labels: List[bool] = []  # True → malicious gateway
    for i in range(n_gateways):
        malicious = include_malicious and i == n_gateways - 1
        ers_labels.append(malicious)
        if malicious:
            weights = list(MALICIOUS_WEIGHTS)
            loss = 9.0
        else:
            weights = [w + rng.gauss(0.0, 0.02) for w in TRUE_WEIGHTS]
            loss = abs(rng.gauss(0.5, 0.05))
        contributions.append(
            _make_contribution(
                device_id=f"gateway_{i:02d}",
                weights=weights,
                samples=100 + i * 10,
                loss=loss,
                round_no=round_no,
            )
        )
    return contributions, ers_labels


def simulate(
    n_rounds: int = 5,
    n_gateways: int = 5,
    include_malicious: bool = True,
    seed: int = 7,
) -> List[Leaderboard]:
    """Run the ERS loop for ``n_rounds`` and return per-round leaderboards."""
    ers_map: Dict[str, float] = {}
    history: List[Leaderboard] = []

    for round_no in range(1, n_rounds + 1):
        contributions, malicious_flags = _round_contributions(
            round_no, n_gateways, include_malicious, seed
        )
        extra_weight = reputation_map(contributions, ers_map)
        result: FedResult = fed_avg(contributions, extra_weight=extra_weight)
        ers_map = update_ers(ers_map, result)

        rows: List[LeaderboardRow] = []
        for i, contribution in enumerate(contributions):
            device_id = contribution["device_id"]
            accepted = sum(1 for c in result.accepted if c["device_id"] == device_id)
            rejected = sum(1 for r in result.rejected if r["device_id"] == device_id)
            rows.append(
                LeaderboardRow(
                    device_id=device_id,
                    samples=contribution["samples"],
                    accepted=accepted,
                    rejected=rejected,
                    last_loss=contribution["loss"],
                    weight_multiplier=round(
                        (extra_weight or {}).get(device_id, 1.0), 3
                    ),
                    ers=ers_map.get(device_id, 0.5),
                    malicious=malicious_flags[i],
                )
            )

        history.append(
            Leaderboard(
                round_no=round_no,
                rows=rows,
                n_contributions=len(contributions),
                accepted_count=result.accepted_count,
                rejected_count=result.rejected_count,
                global_loss=result.loss or 0.0,
            )
        )

    return history



def print_leaderboard(history: List[Leaderboard]) -> None:
    print("Proof-of-Intelligence leaderboard (ERS economy, simulation)")
    print("=" * 74)
    for board in history:
        print(
            f"\nround {board.round_no} · contributions={board.n_contributions} "
            f"accepted={board.accepted_count} rejected={board.rejected_count} "
            f"global_loss={board.global_loss:.4f}"
        )
        header = (
            f"{'#':>2} {'gateway':<12} {'ers':>6} {'w':>6} {'accepted':>9} "
            f"{'rejected':>9} {'loss':>7}  note"
        )
        print(header)
        print("-" * len(header))
        for rank, row in enumerate(board.sorted_rows(), start=1):
            note = "⚡ malicious" if row.malicious else ""
            print(
                f"{rank:>2} {row.device_id:<12} {row.ers:>6.3f} "
                f"{row.weight_multiplier:>6.2f} {row.accepted:>9} "
                f"{row.rejected:>9} {row.last_loss:>7.3f}  {note}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--gateways", type=int, default=5)
    parser.add_argument("--no-malicious", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    history = simulate(
        n_rounds=args.rounds,
        n_gateways=args.gateways,
        include_malicious=not args.no_malicious,
        seed=args.seed,
    )
    print_leaderboard(history)

    last = history[-1]
    accepted = [r for r in last.rows if r.accepted]
    if accepted:
        device_id = accepted[0].device_id
        sample = {
            "schema": "axis-fed/1",
            "round": last.round_no,
            "device_id": device_id,
            "weights": [2.0, 3.0],
            "samples": 120,
            "loss": 0.5,
            "nonce": 1,
        }
        print("\ncommitment preview (off-chain contract)")
        print("=" * 74)
        print(f"device_id : {device_id}")
        print(f"digest    : {contribution_digest(sample)}")
        commit = build_commitment(
            round_no=last.round_no, device_id=device_id, contribution=sample
        )
        print(f"schema    : {commit['schema']} · on-chain write = oracle-only (ADR-0010)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
