"""Coherence signals: the physics in the AI oracle's own language.

The physics layer observes; the oracle speaks in signals. This translation is
deliberately thin:

* :class:`CoherenceSignal` carries exactly what
  ``PolicyEngine.evaluate_ai_anomaly`` duck-types (``is_anomaly``, ``score``,
  ``details``) plus the device it is about, so an operator's existing opt-in
  gate can use it without a new reason code and without touching the protocol;
* :func:`coherence_bundle` wraps a reading in the same ``SignalBundle`` that
  ``agent.signals`` signs and ships (``sign_bundle`` / ``verify_bundle_signature``
  work unchanged), so nothing new appears on the wire.

The signal is an observation, never a decision: the score says how far a
producer's day moved, and the policy (a human, a DAO, or the opt-in gate) says
what that means.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from agent.physics.park import COMPONENTS, ParkObservation
from agent.signals import Signal, SignalBundle

#: What this layer adds to the oracle's vocabulary: one kind, one unit.
KIND = "coherence"
UNIT = "score"

#: Where the maths comes from — printed in every bundle, so a reading can be
#: traced to the implementation it was ported from.
ENGINE = "entrain"
SOURCE_FILE = "entrain/julia/kuramoto/basic.jl"
PORTED = (
    "overlap",
    "coherence",
    "partial_coherence",
    "resonance",
    "spread",
    "chord",
    "wave",
    "coincidence",
    "memory_goal",
    "memory_step",
    "step_state",
)


@dataclass
class CoherenceSignal:
    """One producer's reading, shaped for a Policy Engine.

    ``details["signals"]`` lists the readings that cleared the threshold — the
    same field ``evaluate_ai_anomaly`` reports back as ``ai_signals``.
    """

    device_id: str
    score: float
    is_anomaly: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "score": round(float(self.score), 6),
            "is_anomaly": bool(self.is_anomaly),
            "details": self.details,
        }


def park_signals(observation: ParkObservation) -> List[CoherenceSignal]:
    """One signal per producer, in row order (not only the flagged ones).

    Keeping the quiet producers in the list is deliberate: a park keeper needs
    to see that a device was *read* and found in its place, not just that
    something fired.
    """
    park = observation.park_reading()
    out: List[CoherenceSignal] = []
    for row in observation.rows():
        out.append(
            CoherenceSignal(
                device_id=row["device_id"],
                score=float(row["score"]),
                is_anomaly=bool(row["is_anomaly"]),
                details={
                    "signals": list(row["reasons"]),
                    "components": row["components"],
                    "readings": row["readings"],
                    "coherence": park["coherence_mean"],
                    "anomaly_since": row["anomaly_since"],
                    "threshold": observation.config.threshold,
                    "engine": ENGINE,
                },
            )
        )
    return out


def coherence_bundle(
    observation: ParkObservation,
    source: str = "offline",
    only_anomalies: bool = True,
) -> SignalBundle:
    """A reading as a ``SignalBundle``: the same envelope as the oracle's.

    ``only_anomalies`` keeps the wire quiet (one signal per flagged producer);
    every producer's score still travels in ``meta["devices"]``, so no reading
    is lost, only unspoken.
    """
    ts = observation.starts[observation.window.stop - 1].isoformat(timespec="minutes")
    signals: List[Signal] = []
    for row in observation.rows():
        if only_anomalies and not row["is_anomaly"]:
            continue
        signals.append(
            Signal(
                kind=KIND,
                domain="pilot",
                ts=ts,
                source=source,
                value=float(row["score"]),
                unit=UNIT,
                meta={
                    "device_id": row["device_id"],
                    "is_anomaly": bool(row["is_anomaly"]),
                    "reasons": list(row["reasons"]),
                    "components": row["components"],
                    "readings": row["readings"],
                    "anomaly_since": row["anomaly_since"],
                },
            )
        )
    meta: Dict[str, Any] = {
        "source": source,
        "physics": {
            "engine": ENGINE,
            "source_file": SOURCE_FILE,
            "ported": list(PORTED),
            "components": list(COMPONENTS),
            "config": asdict(observation.config),
        },
        "park": observation.park_reading(),
        "devices": observation.rows(),
        "warnings": list(observation.warnings),
    }
    return SignalBundle(signals=signals, meta=meta)
