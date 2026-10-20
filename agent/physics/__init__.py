"""Physics layer of the AI oracle (ENRG-AI, L3): a park read as one medium.

Entrain's arithmetic (``oscillator``), the park reading built on it (``park``),
the proof-stream adapter (``series``), deterministic scenarios (``synthetic``)
and the translation into oracle signals (``signal``). The boundary with the
protocol is written down where it belongs — ``entrain/docs/ENTRAIN-BOUNDARY.md``:
Axis keeps provenance and cryptography; this layer adds an observation about
coherence and never replaces a signature.

Quick start::

    python -m agent.physics --source offline --scenario mixed --output coherence.json

The maths is a port, not an interpretation: every function in ``oscillator``
matches ``entrain/julia/kuramoto/basic.jl`` to the last bit (see
``tests/fixtures/entrain_reference.json``).
"""
from agent.physics.oscillator import (
    MEMORY_REFERENCE,
    chord,
    coincidence,
    coherence,
    memory_goal,
    memory_step,
    overlap,
    partial_coherence,
    resonance,
    spread,
    step_state,
    unison,
    wave,
)
from agent.physics.park import (
    COMPONENTS,
    ParkConfig,
    ParkObservation,
    demodulate,
    observe_park,
)
from agent.physics.series import ParkMatrix, park_matrix
from agent.physics.signal import (
    ENGINE,
    KIND,
    SOURCE_FILE,
    UNIT,
    CoherenceSignal,
    coherence_bundle,
    park_signals,
)
from agent.physics.synthetic import (
    CLEAN,
    CLOCK_OFFSET,
    LEVEL_SCALE,
    OUTAGE,
    PEAK_SHIFT,
    Injection,
    Scenario,
    build_scenario,
    clean_park,
)

__all__ = [
    "MEMORY_REFERENCE",
    "COMPONENTS",
    "ENGINE",
    "KIND",
    "SOURCE_FILE",
    "UNIT",
    "CLEAN",
    "OUTAGE",
    "PEAK_SHIFT",
    "CLOCK_OFFSET",
    "LEVEL_SCALE",
    "Injection",
    "Scenario",
    "ParkMatrix",
    "ParkConfig",
    "ParkObservation",
    "CoherenceSignal",
    "chord",
    "coincidence",
    "coherence",
    "coherence_bundle",
    "demodulate",
    "memory_goal",
    "memory_step",
    "observe_park",
    "overlap",
    "park_matrix",
    "park_signals",
    "partial_coherence",
    "resonance",
    "spread",
    "step_state",
    "unison",
    "wave",
    "build_scenario",
    "clean_park",
]
