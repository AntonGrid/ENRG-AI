"""Allow ``python -m agent.trade`` — the trading cycle, one cycle at a time.

The rehearsal that signs nothing::

    python -m agent.trade --once --market synthetic --dry-run

The same entry point the container runs is ``python -m agent.trade.loop``.
"""

from agent.trade.loop import main

if __name__ == "__main__":
    raise SystemExit(main())

