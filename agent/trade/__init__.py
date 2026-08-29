"""Trade — closed economic loop (Phase 3).

- ``loop`` — signals → recommender → policy gate → simulated action → reward → ERS.
"""

from agent.trade.loop import TradeRun, TradeStep, simulate_rounds

__all__ = [
    "TradeRun",
    "TradeStep",
    "simulate_rounds",
]
