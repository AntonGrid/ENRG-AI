"""The criteria a run is judged by, written down before it, and a hash to prove it.

A gate is only worth the paper it is printed on if it was chosen before the result was
seen. ``bench/criteria.json`` in ``entrain_infer`` exists for exactly that reason: the
file declares alpha, power and the acceptance checks, the report prints its hash, and any
tweak after the fact becomes a visibly different hash rather than a quiet edit.

This module is the trading twin. It bundles the alpha and power of
:mod:`agent.backtest.power`, the gate of :mod:`agent.backtest.walkforward`, the Monte-Carlo
seed of :mod:`agent.backtest.null` and the acceptance checks into one document with a
SHA-256 over its canonical JSON. A report that carries ``criteria_sha256`` can be checked
years later against the file that was declared; a report without it cannot, which is the
whole point.

The second half of the document is the part reports usually omit: :data:`DOES_NOT_SHOW`,
the list of things a passing run would still not have shown. It sits inside the hashed
payload, so the disclaimer cannot be dropped after the fact either.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.backtest.power import DECLARED_ALPHA, DECLARED_POWER

#: Bumped when the *meaning* of a criterion changes, so an old report is not compared with
#: a new rule under the same name.
VERSION = 1

#: The date the default criteria were declared. A run that declares its own says its own.
DECLARED_UTC = "2026-11-11"

#: The seed every simulation in this package draws from unless it is told otherwise — the
#: null book, the Monte-Carlo power check, the shuffled folds.
DECLARED_SEED = 20261111

NOTE = (
    "This file is written before the run. Changing it afterwards means a new run and a new "
    "report: the hash printed beside the result is the hash of these bytes, and a report "
    "whose hash does not match its criteria is not a result, it is a preference."
)

#: What a passing run would have to satisfy, spelled out so that "pass" is not an opinion.
ACCEPTANCE: Tuple[str, ...] = (
    "alpha measured: the null book's share of false winners must land in alpha * [0.5, 2.0] "
    "— outside it, the declared alpha is not the alpha the book is read at",
    "power at the edge: a book whose true mean equals min_edge_bps must clear the gate "
    "0.90 +/- 0.02 of the time — this is what power 0.9 means, and it is simulated, not "
    "asserted",
    "resolution printed: every pooled row prints min_edge_bps beside its edge, and a row "
    "whose min_edge_bps exceeds the gate cannot pass, whatever its measured edge says",
    "unmeasured is not disproved: fewer than two trades prints as no resolution rather than "
    "as a failure, because a book with no spread has not been read",
    "no lookahead: a fold's policy sees only bars before its boundary, and the admission "
    "filter masks an unusable bar with nan rather than filling it",
    "costs charged both ways: the label and the fill pay the same fee and slippage, and a "
    "trade that stops on its entry bar pays them anyway",
)

#: The honest half: what even a passing run has not shown.
DOES_NOT_SHOW: Tuple[str, ...] = (
    "that an edge is real: this prices whether the gate could be seen at this count; "
    "null.py is the test of whether the book beats coin-flipped bets with the same shape",
    "anything about a pair, venue or period the run did not read — a withheld bar is "
    "withheld, never simulated",
    "anything about live fills: fees and slippage are declared constants, not measured "
    "slippage on a real book",
    "anything about a book whose trades overlap in time: overlap breaks the independence "
    "the t test assumes, and the effective count is smaller than the printed one",
    "that the gate was not chosen after the fact: the hash makes a change visible, it does "
    "not make the choice wise",
    "profit: a gate this wide can be cleared by a book that loses money to fees on a "
    "different fee schedule, and clearing a statistical gate is not an edge in a market",
)


class CriteriaError(ValueError):
    """A criteria document is malformed, forged, or was edited after the run."""


class Criteria:
    """One declaration: the risk, the power, the gate, the seed, and what is not claimed."""

    def __init__(
        self,
        *,
        alpha: float = DECLARED_ALPHA,
        power: float = DECLARED_POWER,
        gate: Optional[Mapping[str, float]] = None,
        seed: int = DECLARED_SEED,
        declared_utc: str = DECLARED_UTC,
        note: str = NOTE,
        acceptance: Sequence[str] = ACCEPTANCE,
        does_not_show: Sequence[str] = DOES_NOT_SHOW,
        version: int = VERSION,
    ) -> None:
        if gate is None:
            # Imported here, not at module scope: this package's ``__init__`` imports this
            # module, and pulling ``walkforward`` in with it makes ``python -m
            # agent.backtest.walkforward`` warn that the module was already imported.
            from agent.backtest.walkforward import GATE as declared_gate

            gate = declared_gate
        if not 0.0 < float(alpha) < 1.0:
            raise CriteriaError(f"alpha {alpha!r} is not a risk: it must lie in (0, 1)")
        if not 0.0 < float(power) < 1.0:
            raise CriteriaError(f"power {power!r} is not a chance: it must lie in (0, 1)")
        self.alpha = float(alpha)
        self.power = float(power)
        self.gate: Dict[str, float] = {
            str(key): float(value) for key, value in dict(gate or {}).items()
        }
        self.seed = int(seed)
        self.declared_utc = str(declared_utc)
        self.note = str(note)
        self.acceptance: List[str] = [str(item) for item in acceptance]
        self.does_not_show: List[str] = [str(item) for item in does_not_show]
        self.version = int(version)

    @property
    def edge_bps(self) -> float:
        """The gate's edge floor, which is the number :mod:`power` reads the book against."""
        return float(self.gate.get("edge_bps", 0.0))

    def as_dict(self) -> Dict[str, Any]:
        """The declared payload — exactly the bytes that are hashed, in a JSON-safe form."""
        return {
            "version": int(self.version),
            "declared_before_run": True,
            "declared_utc": self.declared_utc,
            "note": self.note,
            "alpha": self.alpha,
            "alpha_meaning": (
                "the risk of calling a flat book a winner, per pooled verdict; two-sided"
            ),
            "target_power": self.power,
            "target_power_meaning": (
                "the chance of seeing an edge that sits exactly on min_edge_bps, at alpha"
            ),
            "gate": dict(sorted(self.gate.items())),
            "seed": int(self.seed),
            "acceptance": list(self.acceptance),
            "does_not_show": list(self.does_not_show),
        }

    def canonical(self) -> str:
        """The one JSON spelling of the payload: sorted keys, no spaces, no infinities."""
        return json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    def fingerprint(self) -> str:
        """SHA-256 of :meth:`canonical` — the value a report prints beside its numbers."""
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def document(self, run: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """A report header: the criteria, their hash, and whatever the run measured."""
        payload: Dict[str, Any] = {
            "criteria_sha256": self.fingerprint(),
            "criteria": self.as_dict(),
        }
        if run is not None:
            payload["run"] = dict(run)
        return payload

    def write(self, path: Any, run: Optional[Mapping[str, Any]] = None) -> Path:
        """Write the document as JSON with a trailing newline, and return where it went."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.document(run), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return target

    def __repr__(self) -> str:
        return (
            f"Criteria(alpha={self.alpha}, power={self.power}, gate={self.gate}, "
            f"seed={self.seed})"
        )


def verify(document: Mapping[str, Any]) -> str:
    """Empty string when the document's hash matches its criteria, else the reason it does not.

    The check is deliberately one-way: it can say "these numbers were not the ones declared"
    and it cannot say the declared ones were good. That is all a hash can do, and all a
    pre-registration needs.
    """
    if "criteria" not in document:
        return "no criteria block: the run was not read against anything declared"
    payload = document["criteria"]
    if not isinstance(payload, Mapping):
        return "the criteria block is not an object"
    declared = str(document.get("criteria_sha256", "")).strip().lower()
    if not declared:
        return "no criteria_sha256 beside the criteria: nothing proves they were not edited"
    canonical = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if actual != declared:
        return (
            f"criteria hash {actual[:12]} does not match the declared {declared[:12]}: "
            "the criteria were changed after the run"
        )
    return ""


def declared() -> Criteria:
    """The package's default declaration — the one every report in this repo prints."""
    return Criteria()


__all__ = [
    "ACCEPTANCE",
    "DECLARED_SEED",
    "DECLARED_UTC",
    "DOES_NOT_SHOW",
    "NOTE",
    "VERSION",
    "Criteria",
    "CriteriaError",
    "declared",
    "verify",
]
