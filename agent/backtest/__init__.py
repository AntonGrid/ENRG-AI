"""Backtest — the same hunt, run against history instead of a wallet.

The live cycle (:mod:`agent.trade.loop`) asks the hunter what the market says *now* and
then either signs a stake or prints one. That is the only way to find out whether the
thesis is worth money, and it is also the worst way: every answer costs a transaction and
arrives one episode at a time. This package asks the same question of history, in
isolation, and answers with a number instead of a hope.

Four files, and the wall between them is the design:

- ``data`` — candles, from HTX's public kline endpoint or a CSV on disk, and the one
  declared conversion from a candle to the rate the hunter reads (a decline is a yield —
  see the module docstring, it is the only invented quantity in the package);
- ``simulated_provider`` — a venue that answers the same operations a real one does
  (:class:`agent.physics.interfaces.IWeb3Provider`), filled from the candles row by row,
  with slippage, fees and a settlement lag, and with *no* key, no RPC and no chain;
- ``engine`` — the decisions: the hunter's verdict per bar, a buy on ``PANIC`` above the
  confidence floor, a sale on a conviction or on the panic passing, and a stop-loss;
- ``reporter`` — the report: ROI, drawdown, win rate, trades, holding time, and the one
  benchmark that keeps an honest backtest honest (buy-and-hold).

``run`` is one of those runs on a command line, and ``sweep`` is the same run over the
venue's own top pairs: an anecdote is not a finding, so the table carries ROI, buy-and-hold
and the difference between them for every pair it measured.

What this package will never do
===============================

It does not import :mod:`agent.physics` for anything but measurement — the hunter and
``risk_assessment`` are read, never modified, and neither of them can move a credit. It has
no network path except :func:`agent.backtest.data.fetch_klines`, which takes the transport
as an argument so that a test can hand it a dictionary instead of an HTTP client. And it
never touches ``trader_state.json``: the live loop's memory is not a backtest's business,
and a backtest that wrote to it would be a backtest that could change what the money does.

The names below are the public ones; ``python -m agent.backtest.run`` is the entry point.
"""

from agent.backtest.data import (
    Candle,
    CandleSeries,
    HtxError,
    Pair,
    declines,
    fetch_klines,
    load_csv,
    universe,
)
from agent.backtest.engine import (
    ENTRY_VERDICTS,
    EXIT_VERDICTS,
    TAKE_PROFIT_VERDICTS,
    BacktestResult,
    Backtester,
    Settings,
    Trade,
)
from agent.backtest.reporter import Report, write_report
from agent.backtest.simulated_provider import Fill, SimBalance, SimulatedProvider

__all__ = [
    "ENTRY_VERDICTS",
    "EXIT_VERDICTS",
    "TAKE_PROFIT_VERDICTS",
    "BacktestResult",
    "Backtester",
    "Candle",
    "CandleSeries",
    "Fill",
    "HtxError",
    "Pair",
    "Report",
    "Settings",
    "SimBalance",
    "SimulatedProvider",
    "Trade",
    "declines",
    "fetch_klines",
    "load_csv",
    "universe",
    "write_report",
]
