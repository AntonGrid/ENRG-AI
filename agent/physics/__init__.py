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

The autopoietic layer sits on the same arithmetic and never re-derives any of it:

=========================  ==================================================
``genome``                 the core, reached through one door and fingerprinted
``sensorium``              organs of sense: world feeds and the four dreams
``homeostasis``            λ, ε, α and the mutations that keep them honest
``self_model``             the mirror: internal resonance, the ledger, the score
``organism``               the body that lives the three phases in order
``curiosity``              the appetite: which stream is worth asking about
``economy_sim``            the market: what a verdict is worth, and the rent
``worker``                 the body that bids, refuses, and remembers why
``portfolio``              capital: three buckets, a cap, a whitelist, a switch
``risk_assessment``        the verdict on somebody else's yield curve
``interfaces``             the seam: what a venue must answer, and the mock that does
``real_testnet_provider``  first contact: the same seam, answered by a real testnet
=========================  ==================================================

Quick start of the organism::

    python -m agent.physics.organism --offline --output report.json --plot phases.png

Quick start of the appetite (the same body, now asking questions)::

    python -m agent.physics.organism --offline --explore --curiosity-plot curiosity.png

Quick start of the worker (the same body, now earning its rent)::

    python -m agent.physics.worker --iterations 120 --plot economy.png

Quick start of the empire (the same body, now managing capital)::

    python -m agent.physics.portfolio --steps 720 --body --against --plot empire.png

Capital is handed its venue through ``interfaces.IWeb3Provider`` rather than reaching
for one, so the same run can be pointed at a different source with one line::

    provider = MockWeb3Provider()                  # the declared market
    empire = Empire.launch(body=body, provider=provider)

The same run against a chain, once an operator has a node and an address book (the key
and the endpoint arrive from the environment and from nowhere else)::

    provider = RealTestnetProvider.from_env()      # TESTNET_RPC_URL, TESTNET_PROTOCOLS
    empire = Empire.launch(body=body, provider=provider)
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
from agent.physics.genome import (
    GenomeError,
    GenomeFrozenError,
    describe as describe_genome,
    fingerprint as genome_fingerprint,
    verify as verify_genome,
)
from agent.physics.homeostasis import (
    Calibration,
    Homeostasis,
    Mutation,
    Parameters,
    calibrate,
    internal_resonance,
    mutate,
    separation,
)
from agent.physics.sensorium import (
    Perception,
    SensorError,
    Sensorium,
    dream,
    hear,
    listen_dataset,
    listen_finance,
)
from agent.physics.self_model import Ledger, SelfModel, reality_score
from agent.physics.curiosity import (
    ANOMALY_THRESHOLD,
    MAX_PROBES,
    PROBE_THRESHOLD,
    SURVEY_FIXATIONS,
    Certainty,
    Curiosity,
    Probe,
    Trait,
    UCB,
    lindley_gain,
    perturb,
    reducibility,
    structure_of,
)
from agent.physics.economy_sim import (
    CONTRACT_STREAMS,
    PENALTY,
    REWARD,
    SOLVABLE,
    START_BALANCE,
    Bankrupt,
    Contract,
    EconomyError,
    EconomySim,
    Market,
    Settlement,
)
from agent.physics.worker import (
    CERTAIN,
    SCHOOLING,
    Decision,
    EpisodicMemory,
    Experience,
    RealityOracle,
    Worker,
)
from agent.physics.plots import (
    plot_curiosity,
    plot_economy,
    plot_emergence,
    plot_empire,
    plot_funnel,
)
from agent.physics.discovery import (
    EXECUTION_LIMIT,
    SCAN_CV_MAX,
    TURNOVER_FLOOR,
    WATCHLIST_LIMIT,
    Candidate,
    DiscoveryReport,
    Scanner,
    Watchlist,
)
from agent.physics.market_generator import (
    DARK_SHARE,
    HONEST_SHARE,
    JUNK_SHARE,
    KIND_HONEST,
    KIND_JUNK,
    KIND_SCAM,
    KINDS,
    RUG_SHARE,
    SCAM_SHARE,
    MarketRecipe,
    generate_market,
    generate_protocols,
    kind_of,
    kinds_of,
)
from agent.physics.portfolio import (
    BUCKETS,
    CYCLE,
    MIN_RECORD,
    START_CAPITAL,
    TARGETS,
    Empire,
    NaiveInvestor,
    NotWhitelisted,
    Portfolio,
    PortfolioError,
    Protocol,
    ProtocolMarket,
    StakeRefused,
    WhiteList,
    annualise,
    implied_apy,
    per_step_rate,
    sim_address,
)
from agent.physics.interfaces import IWeb3Provider, MockWeb3Provider, NoVenueProvider
from agent.physics.real_testnet_provider import (
    DryRun,
    ProtocolBook,
    ProtocolEntry,
    RealTestnetProvider,
    TestnetConfig,
    TestnetError,
    TestnetNotConfigured,
    TestnetUnavailable,
)
from agent.physics.risk_assessment import (
    CONVICTIONS,
    HONEST,
    KILL_DRAWDOWN,
    KILL_VOLATILITY,
    PLAUSIBLE_APY,
    REALITY_FLOOR,
    REPEAT_R2,
    RESONANCE_FLOOR,
    SCAM_TRAP,
    SUSPECT,
    SYNTHETIC_COLLUSION,
    UNMEASURED,
    VERDICTS,
    Assessment,
    KillSwitch,
    Trigger,
    evaluate_opportunity,
    sterility_of,
    volatility_ratio,
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
    "Calibration",
    "GenomeError",
    "GenomeFrozenError",
    "Homeostasis",
    "Ledger",
    "Mutation",
    "Parameters",
    "Perception",
    "SelfModel",
    "SensorError",
    "Sensorium",
    "calibrate",
    "describe_genome",
    "dream",
    "genome_fingerprint",
    "hear",
    "internal_resonance",
    "listen_dataset",
    "listen_finance",
    "mutate",
    "plot_curiosity",
    "plot_emergence",
    "reality_score",
    "separation",
    "verify_genome",
    "ANOMALY_THRESHOLD",
    "MAX_PROBES",
    "PROBE_THRESHOLD",
    "SURVEY_FIXATIONS",
    "Certainty",
    "Curiosity",
    "Probe",
    "Trait",
    "UCB",
    "lindley_gain",
    "perturb",
    "reducibility",
    "structure_of",
    "CONTRACT_STREAMS",
    "PENALTY",
    "REWARD",
    "SOLVABLE",
    "START_BALANCE",
    "Bankrupt",
    "Contract",
    "EconomyError",
    "EconomySim",
    "Market",
    "Settlement",
    "CERTAIN",
    "SCHOOLING",
    "Decision",
    "EpisodicMemory",
    "Experience",
    "RealityOracle",
    "Worker",
    "plot_economy",
    "plot_empire",
    "BUCKETS",
    "CYCLE",
    "MIN_RECORD",
    "START_CAPITAL",
    "TARGETS",
    "Empire",
    "NaiveInvestor",
    "NotWhitelisted",
    "Portfolio",
    "PortfolioError",
    "Protocol",
    "ProtocolMarket",
    "StakeRefused",
    "WhiteList",
    "annualise",
    "implied_apy",
    "per_step_rate",
    "sim_address",
    "IWeb3Provider",
    "MockWeb3Provider",
    "NoVenueProvider",
    "RealTestnetProvider",
    "TestnetConfig",
    "TestnetError",
    "TestnetNotConfigured",
    "TestnetUnavailable",
    "ProtocolBook",
    "ProtocolEntry",
    "DryRun",
    "CONVICTIONS",
    "HONEST",
    "KILL_DRAWDOWN",
    "KILL_VOLATILITY",
    "PLAUSIBLE_APY",
    "REALITY_FLOOR",
    "REPEAT_R2",
    "RESONANCE_FLOOR",
    "SCAM_TRAP",
    "SUSPECT",
    "SYNTHETIC_COLLUSION",
    "UNMEASURED",
    "VERDICTS",
    "Assessment",
    "KillSwitch",
    "Trigger",
    "evaluate_opportunity",
    "sterility_of",
    "volatility_ratio",
    "EXECUTION_LIMIT",
    "SCAN_CV_MAX",
    "TURNOVER_FLOOR",
    "WATCHLIST_LIMIT",
    "Candidate",
    "DiscoveryReport",
    "Scanner",
    "Watchlist",
    "DARK_SHARE",
    "HONEST_SHARE",
    "JUNK_SHARE",
    "KIND_HONEST",
    "KIND_JUNK",
    "KIND_SCAM",
    "KINDS",
    "RUG_SHARE",
    "SCAM_SHARE",
    "MarketRecipe",
    "generate_market",
    "generate_protocols",
    "kind_of",
    "kinds_of",
    "plot_funnel",
]
