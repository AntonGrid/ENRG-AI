"""Physics layer of the AI oracle (ENRG-AI, L3): a park read as one medium.

Entrain's arithmetic (``oscillator``), the park reading built on it (``park``),
the proof-stream adapter (``series``), deterministic scenarios (``synthetic``),
the translation into oracle signals (``signal``), Statement #87/#88 — the
sterility index and the smoothed λ* operator (``sterility``) — and the picture
Anton asked for (``plots``). The boundary with the protocol is written down
where it belongs — ``entrain/docs/ENTRAIN-BOUNDARY.md``: Axis keeps provenance
and cryptography; this layer adds an observation about coherence and never
replaces a signature.

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
from agent.physics.plots import DEFAULT_PATH, plot_depin_dynamics
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
from agent.physics.sterility import (
    FAILED,
    STERILE_A2,
    STERILE_CV,
    BanachResult,
    SterilitySignal,
    SterilityVerdict,
    anderson_darling,
    banach_iteration,
    smooth_max,
    sterility_index,
    variation,
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
    "FAILED",
    "STERILE_A2",
    "STERILE_CV",
    "DEFAULT_PATH",
    "Injection",
    "Scenario",
    "ParkMatrix",
    "ParkConfig",
    "ParkObservation",
    "CoherenceSignal",
    "SterilitySignal",
    "SterilityVerdict",
    "BanachResult",
    "anderson_darling",
    "banach_iteration",
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
    "plot_depin_dynamics",
    "resonance",
    "smooth_max",
    "spread",
    "step_state",
    "sterility_index",
    "unison",
    "variation",
    "wave",
    "build_scenario",
    "clean_park",
]
