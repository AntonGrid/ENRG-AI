"""Allow ``python -m agent.backtest`` — the same call as ``python -m agent.backtest.run``.

The rehearsal that signs nothing::

    python -m agent.backtest --exchange htx --symbol btcusdt --days 180 --capital 1000

The module form the brief names is ``python -m agent.backtest.run``; both land in
:func:`agent.backtest.run.main`.
"""

from agent.backtest.run import main

if __name__ == "__main__":
    raise SystemExit(main())
