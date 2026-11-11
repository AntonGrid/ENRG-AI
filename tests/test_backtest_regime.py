"""The regime mask, and the alignment that decides whether it is a filter or a look-ahead.

The mask is one bit per bar — was the benchmark above its own mean — and it is the sweep's one
statistically loud finding spent as a rule. The tests here are about the *alignment* more than
the average: the benchmark is a different instrument on a different calendar, and reading it
from the next bar rather than the last one turns a filter into a look-ahead that the whole
walk-forward would then inherit.
"""
from __future__ import annotations

from agent.backtest.data import Candle
from agent.backtest.regime import DEFAULT_REGIME_WINDOW, align_regime, sma_mask

START = 1_600_000_000.0
DAY = 86400.0


def series(closes, *, step: float = DAY, start: float = START):
    """A benchmark from closes, one bar per ``step`` seconds."""
    return [
        Candle(
            at=start + index * step,
            open=float(close),
            high=float(close) * 1.001,
            low=float(close) * 0.999,
            close=float(close),
            volume=1000.0,
        )
        for index, close in enumerate(closes)
    ]


def test_the_mask_is_true_above_the_mean_and_false_below_it():
    closes = [100.0 + index for index in range(20)] + [200.0 - index for index in range(20)]
    mask = sma_mask(closes, window=5)
    assert mask[-1] is False
    assert all(mask[index] for index in range(5, 18))
    assert mask[:5] == [True] * 5


def test_a_young_record_is_permitted_rather_than_refused():
    # Warming up is not evidence against a pair: the mask's only job is "was the market above
    # its mean", and a mask that said False for the first 200 bars would silently be a rule
    # about the length of the record.
    assert sma_mask([100.0, 99.0, 98.0], window=DEFAULT_REGIME_WINDOW) == [True, True, True]


def test_a_flat_market_sits_at_its_own_mean_and_is_allowed():
    assert sma_mask([100.0] * 10, window=3) == [True] * 10


def test_an_instrument_bar_reads_the_last_benchmark_bar_that_had_already_closed():
    rows = series([100.0, 100.0], step=DAY)
    # The benchmark falls hard *after* both instrument bars: a mask read from the future
    # would darken them, and this is the one test that catches that.
    benchmark = series([100.0, 100.0, 1.0, 1.0], step=DAY)
    mask = align_regime(rows, benchmark, window=2)
    assert mask == [True, True]


def test_a_benchmark_bar_that_closed_earlier_is_what_a_later_bar_reads():
    rows = series([100.0, 100.0, 100.0], step=DAY)
    benchmark = series([100.0, 100.0, 1.0], step=DAY)  # falls on the third bar only
    mask = align_regime(rows, benchmark, window=2)
    assert mask[0] is True and mask[1] is True
    assert mask[2] is False


def test_a_bar_before_the_benchmark_begins_is_allowed():
    rows = series([100.0, 100.0], start=START - 5 * DAY)
    benchmark = series([100.0, 100.0, 100.0])
    assert align_regime(rows, benchmark, window=2) == [True, True]


def test_a_benchmark_on_a_coarser_calendar_still_lines_up():
    # Four instrument bars per benchmark bar: each instrument bar must read the newest
    # benchmark bar that had closed, which is the same one four times.
    rows = series([100.0] * 9, step=DAY)
    benchmark = series([100.0, 100.0, 1.0], step=4 * DAY)
    mask = align_regime(rows, benchmark, window=2)
    assert mask[:4] == [True] * 4
    assert mask[4:8] == [True] * 4
    assert mask[8] is False, "the bar at the benchmark's third stamp reads the fall"


def test_no_benchmark_means_no_filter_and_not_a_refusal():
    rows = series([100.0, 100.0, 100.0])
    assert align_regime(rows, [], window=2) == [True, True, True]
