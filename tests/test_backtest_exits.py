"""The exit mechanics: every knob reaches the engine, and each one moves the book.

The resolution layer says the lever on a verdict is ``n`` and ``sigma``. ``n`` is expensive
— 554 trades for a 30 bps gate — while ``sigma`` is set by how a position ends, which is a
decision rather than a discovery. So these tests are about that decision being real: that
all four entry points read the same flags, that a flag reaches the field it names (and not
the one beside it), that an explicit ``0`` is not silence, and that each knob visibly
changes the exits of one and the same chart.

The last test is the commercial one: a target clips the right tail, and the tail is what
makes ``sigma`` wide. It is asserted as a *narrowing* of the spread, not as a better ROI,
because narrowing the spread is what the knob is for and improving the ROI is what the next
fold is for.
"""
from __future__ import annotations

import argparse

import pytest

import agent.backtest.engine as E
from agent.backtest.data import Candle, CandleSeries
from agent.backtest.exits import (
    EXIT_FLAGS,
    add_exit_arguments,
    describe_exits,
    exits_from_args,
)
from agent.backtest.power import spread_bps
from agent.backtest.walkforward import ABLATIONS, pinned_ablations

START = 1_600_000_000.0
PERIOD = 14400.0

#: Everything an operator may say about an exit, in one command line.
FULL_GRID = (
    "--stop-mode",
    "atr",
    "--atr-window",
    "20",
    "--atr-multiple",
    "3",
    "--risk-per-trade",
    "0.01",
    "--take-profit-r",
    "2",
    "--trail-r",
    "1.5",
    "--time-stop-bars",
    "24",
)


def _chart(changes, volumes) -> CandleSeries:
    """A series from per-bar returns and turnovers — the engine only reads candles."""
    rows = []
    price = 100.0
    for index, (change, volume) in enumerate(zip(changes, volumes)):
        opening = price
        price = price * (1.0 + float(change))
        rows.append(
            Candle(
                at=START + index * PERIOD,
                open=opening,
                high=max(opening, price) * 1.0005,
                low=min(opening, price) * 0.9995,
                close=price,
                volume=float(volume),
            )
        )
    return CandleSeries(symbol="btcusdt", period="4hour", rows=rows, source="fixture")


#: A capitulation on draining turnover: eight bars of about two per cent, which the hunter
#: reads as a PANIC and the venue fills on the next open.
CRASH = (
    -0.019,
    -0.017,
    -0.021,
    -0.018,
    -0.020,
    -0.019,
    -0.017,
    -0.021,
    -0.018,
    -0.020,
    -0.019,
    -0.017,
    -0.021,
    -0.018,
)


def panic_chart() -> CandleSeries:
    """A quiet market, a capitulation, and a recovery heavy enough to read as CALM."""
    changes = [0.004 if index % 2 == 0 else -0.004 for index in range(40)]
    volumes = [1.0e6] * 40
    changes += list(CRASH)
    volumes += [4.0e5] * len(CRASH)
    changes += [0.012] * 12
    volumes += [7.0e5] * 12
    return _chart(changes, volumes)


def bounce_chart() -> CandleSeries:
    """A panic, a bounce on heavy turnover, then a slide — where a trail can be paid."""
    changes = [0.004 if index % 2 == 0 else -0.004 for index in range(40)]
    volumes = [1.0e6] * 40
    changes += list(CRASH)
    volumes += [4.0e5] * len(CRASH)
    changes += [0.02, 0.02, 0.02]
    volumes += [1.2e6] * 3
    changes += [-0.01] * 10
    volumes += [1.6e6] * 10
    return _chart(changes, volumes)


def _run(series=None, **overrides):
    """One engine run on the same record floor the engine's own tests use."""
    settings = E.Settings(min_points=20, **overrides)
    return E.Backtester(series or panic_chart(), settings).run()


# ── the command line ───────────────────────────────────────────────────────


def _parsers():
    from agent.backtest import run as runner
    from agent.backtest import sweep
    from agent.backtest import train as trainer
    from agent.backtest import walkforward

    return {
        "run": runner._parser(),
        "sweep": sweep._parser(),
        "train": trainer._parser(),
        "walkforward": walkforward._parser(),
    }


def test_a_command_line_that_says_nothing_moves_nothing():
    for name, parser in _parsers().items():
        assert exits_from_args(parser.parse_args([])) == {}, name


def test_every_flag_reaches_the_field_it_names():
    parser = argparse.ArgumentParser()
    add_exit_arguments(parser)
    overrides = exits_from_args(parser.parse_args(list(FULL_GRID)))
    assert overrides == {
        "stop_mode": "atr",
        "atr_window": 20,
        "atr_multiple": 3.0,
        "risk_per_trade": 0.01,
        "take_profit_r": 2.0,
        "trail_r": 1.5,
        "time_stop_bars": 24,
    }
    assert isinstance(overrides["atr_window"], int)
    assert isinstance(overrides["time_stop_bars"], int)


def test_the_four_entry_points_read_one_grid_the_same_way():
    seen = set()
    for name, parser in _parsers().items():
        overrides = exits_from_args(parser.parse_args(list(FULL_GRID)))
        assert set(overrides) == set(EXIT_FLAGS.values()), name
        seen.add(tuple(sorted(overrides.items())))
    assert len(seen) == 1


def test_an_explicit_zero_is_not_the_same_as_silence():
    parser = argparse.ArgumentParser()
    add_exit_arguments(parser)
    said = exits_from_args(parser.parse_args(["--take-profit-r", "0", "--trail-r", "0"]))
    assert said == {"take_profit_r": 0.0, "trail_r": 0.0}


def test_a_stop_mode_the_engine_does_not_have_is_refused_by_the_parser():
    parser = argparse.ArgumentParser()
    add_exit_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--stop-mode", "sideways"])


def test_the_flags_are_settings_the_engine_accepts():
    parser = argparse.ArgumentParser()
    add_exit_arguments(parser)
    settings = E.Settings(**exits_from_args(parser.parse_args(list(FULL_GRID))))
    assert settings.stop_mode == "atr"
    assert settings.atr_multiple == pytest.approx(3.0)
    assert settings.risk_per_trade == pytest.approx(0.01)
    assert settings.take_profit_r == pytest.approx(2.0)
    assert settings.trail_r == pytest.approx(1.5)
    assert settings.time_stop_bars == 24


def test_a_risk_the_book_cannot_hold_is_refused_with_a_sentence():
    with pytest.raises(ValueError, match="risk_per_trade"):
        E.Settings(risk_per_trade=2.0)
    with pytest.raises(ValueError, match="take_profit_r"):
        E.Settings(take_profit_r=-1.0)
    with pytest.raises(ValueError, match="time_stop_bars"):
        E.Settings(time_stop_bars=-1)


def test_describe_exits_says_the_defaults_and_then_each_knob():
    assert describe_exits({}).startswith("exits: the engine's defaults")
    sentence = describe_exits(
        {
            "stop_mode": "atr",
            "atr_multiple": 2.0,
            "risk_per_trade": 0.02,
            "take_profit_r": 2.0,
            "trail_r": 0.0,
            "time_stop_bars": 48,
        }
    )
    assert "stop atr" in sentence
    assert "2x ATR" in sentence
    assert "risk 2.00%" in sentence
    assert "2R target" in sentence
    assert "no trail" in sentence
    assert "48-bar clock" in sentence


def test_pinned_ablations_lays_the_flags_over_the_rows_own_values():
    assert pinned_ablations({}) is ABLATIONS
    pinned = pinned_ablations({"atr_multiple": 3.0, "take_profit_r": 1.0})
    assert len(pinned) == len(ABLATIONS)
    assert pinned[0].overrides == {"atr_multiple": 3.0, "take_profit_r": 1.0}
    # The row called "policy+atr" asked for 2.0; the operator asked for 3.0 and wins, because
    # a table whose row does not run what its name says is a table that lies.
    assert pinned[2].overrides["atr_multiple"] == pytest.approx(3.0)
    assert pinned[2].overrides["stop_mode"] == "atr"
    assert pinned[2].overrides["risk_per_trade"] == pytest.approx(0.02)


# ── what each knob does to one and the same chart ──────────────────────────


def test_risk_per_trade_sizes_the_stake_at_the_stop_not_at_the_cash():
    # One per cent risked at a two per cent stop is half the book: 1000 x 0.01 / 0.02.
    book = _run()
    sized = _run(risk_per_trade=0.01)
    assert book.trades and sized.trades
    assert book.trades[0].entry_usd == pytest.approx(1000.0, rel=1e-6)
    assert sized.trades[0].entry_usd == pytest.approx(500.0, rel=1e-6)
    # The trade's return is the market's, not the stake's: the sizing changes the book, not
    # the strategy, which is exactly why it cannot manufacture an edge.
    assert sized.trades[0].pnl_pct == pytest.approx(book.trades[0].pnl_pct, rel=1e-6)


def test_atr_mode_puts_the_stop_outside_the_bars_own_noise():
    fixed = [trade for trade in _run().trades if trade.exit_reason == "stop"]
    atr = [
        trade
        for trade in _run(stop_mode="atr", atr_multiple=2.0).trades
        if trade.exit_reason == "stop"
    ]
    assert fixed and atr, [trade.as_dict() for trade in _run().trades]
    assert abs(fixed[0].pnl_pct) == pytest.approx(0.0229, abs=0.002)
    # A 2x ATR stop on this chart is wider than two per cent, so a stop that fires loses
    # more than the fixed one would have — the point is that it fires far less often.
    assert abs(atr[0].pnl_pct) > abs(fixed[0].pnl_pct)
    assert len(atr) <= len(fixed)


def test_a_target_at_r_multiple_closes_early_and_in_profit():
    taken = [trade for trade in _run(take_profit_r=0.5).trades if trade.exit_reason == "target"]
    assert taken, [trade.as_dict() for trade in _run(take_profit_r=0.5).trades]
    assert all(trade.pnl > 0.0 for trade in taken)
    # Half of a two per cent stop is one per cent gross; two fees and a slippage eat a third.
    assert taken[0].pnl_pct == pytest.approx(0.007, abs=0.002)
    assert taken[0].pnl_pct < 0.01


def test_a_trail_locks_in_a_profit_where_a_stop_would_have_taken_a_loss():
    bare = [trade for trade in _run(bounce_chart()).trades if trade.exit_reason == "stop"]
    trailed = [
        trade
        for trade in _run(bounce_chart(), trail_r=1.0).trades
        if trade.exit_reason in ("stop", "trail")
    ]
    assert bare and all(trade.pnl < 0.0 for trade in bare)
    assert any(trade.pnl > 0.0 for trade in trailed), [
        trade.as_dict() for trade in trailed
    ]


def test_a_clock_closes_the_position_at_the_next_open():
    closed = [trade for trade in _run(time_stop_bars=1).trades if trade.exit_reason == "time"]
    assert closed, [trade.as_dict() for trade in _run(time_stop_bars=1).trades]
    # A one-bar clock plus the bar the order is filled on: the position lives one bar and
    # is closed on the next, which is why the count is two and not one.
    assert all(trade.bars == 2 for trade in closed)


def test_a_target_narrows_the_spread_the_verdict_is_read_at():
    # The reason these flags exist: the same chart, read at a narrower spread, can settle a
    # smaller edge at the same count. A knob that widens the spread is not an improvement
    # even if it raises the mean, because the mean is then a rumour the count cannot check.
    bare = spread_bps([trade.pnl_pct for trade in _run().trades])
    clipped = spread_bps([trade.pnl_pct for trade in _run(take_profit_r=0.5).trades])
    assert bare > 0.0 and clipped > 0.0
    assert clipped < bare
