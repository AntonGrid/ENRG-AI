"""Trade — two loops, one word, and a wall between them.

- ``loop`` — the **live** cycle: the operator's vault book → the DeFiLlama market → the
  hunter's verdict → a size cut to a fraction of the wallet → a stake, or the log of the
  stake that would have been sent. It signs nothing unless ``DRY_RUN=0``.
- ``capital`` — the sizing policy in one file: ``min(MAX_TRADE_SIZE_USD, 20 % of the
  wallet)``, a gas reserve, and a refusal for everything else. No key, no network.
- ``simulation`` — the Phase-3 closed loop: watt-hours, ``axis_core``, no money.

``simulation`` is imported **lazily** (PEP 562) because it needs ``axis-core`` on the
path, and the trading path must not: a container that has an RPC endpoint and web3 but
no ``axis_core`` still imports this package, which is the whole point of the split.
"""

from agent.trade.capital import Sizing, TradeSize, size_trade

__all__ = [
    "Action",
    "Cycle",
    "Position",
    "Sizing",
    "TradeError",
    "TradeRun",
    "TradeSize",
    "TradeStep",
    "Trader",
    "simulate_rounds",
    "size_trade",
]

#: ``loop`` and ``simulation`` are both imported on first use, and for the same reason:
#: ``python -m agent.trade.loop`` must not find the module already imported by this file
#: (runpy warns about exactly that), and ``simulation`` drags ``axis_core`` in.
_LAZY = {
    "Action": "loop",
    "Cycle": "loop",
    "Position": "loop",
    "TradeError": "loop",
    "TradeRun": "simulation",
    "TradeStep": "simulation",
    "Trader": "loop",
    "simulate_rounds": "simulation",
}


def __getattr__(name: str):
    """Answer a name from ``loop`` or ``simulation`` without importing either eagerly."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module}"), name)


