"""Closed-loop DePIN pilot (GLOBAL_AI_ARCHITECTURE step C).

Simulates N devices over H hours and demonstrates the full loop:

    device → forecast model → market price → Recommender (axis_core.ai)
        → Policy Engine (evaluate_trade) → action → reward (USD)
        → reward feeds ERS + the forecast model (retrained in-loop)

Compares the AI strategy with a blind baseline (no action).

Run: ``python -m agent.pilot.sim`` or ``python -m agent.pilot --help``.
"""
