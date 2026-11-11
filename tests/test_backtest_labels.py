"""The label, which is the only place in the package where the cost is subtracted.

A model trained on gross returns learns to buy moves the venue keeps — the first sweep was
that mistake measured in dollars, with 43 % of its loss in fees. So the tests here are about
arithmetic that must not drift: the entry is the *next* open, the exit is a close ``horizon``
bars later, ``DEFAULT_COST`` comes off before the comparison, and a bar whose future is not in
the record has **no** label rather than a zero.
"""
from __future__ import annotations

import pytest

from agent.backtest.data import Candle
from agent.backtest.labels import (
    DEFAULT_COST,
    forward_return,
    label_at,
    labels_at,
    net_forward_return,
)

START = 1_600_000_000.0
PERIOD = 86400.0


def bars(opens, closes) -> list:
    """A series whose opens the caller controls — the only way to test the fill honestly."""
    return [
        Candle(
            at=START + index * PERIOD,
            open=float(open_),
            high=max(float(open_), float(close)) * 1.001,
            low=min(float(open_), float(close)) * 0.999,
            close=float(close),
            volume=1000.0,
        )
        for index, (open_, close) in enumerate(zip(opens, closes))
    ]


def test_the_fill_is_the_next_open_and_not_the_close_the_decision_was_made_on():
    rows = bars(opens=[100.0, 110.0, 121.0, 133.1], closes=[100.0, 110.0, 121.0, 133.1])
    # Bar 0 closes at 100; the fill is bar 1's open (110); the exit is bar 3's close (133.1).
    assert forward_return(rows, 0, horizon=3) == pytest.approx(100.0 * (133.1 / 110.0 - 1.0))


def test_the_exit_is_the_close_of_the_bar_horizon_bars_later():
    rows = bars(opens=[100.0] * 6, closes=[100.0, 100.0, 104.0, 100.0, 100.0, 100.0])
    assert forward_return(rows, 0, horizon=2) == pytest.approx(4.0)


def test_a_bar_without_a_future_has_no_label_rather_than_a_zero():
    rows = bars(opens=[100.0] * 5, closes=[100.0] * 5)
    assert forward_return(rows, 3, horizon=2) is None
    assert forward_return(rows, 4, horizon=1) is None
    assert labels_at(rows, horizon=2) == [0.0, 0.0, 0.0, None, None]


def test_the_cost_comes_off_before_the_comparison():
    rows = bars(opens=[100.0, 100.0], closes=[100.0, 100.2])
    gross = forward_return(rows, 0, horizon=1)
    assert gross == pytest.approx(0.2)
    assert net_forward_return(rows, 0, horizon=1) == pytest.approx(0.2 - 100.0 * DEFAULT_COST)
    # A 0.2 % move does not pay a 0.4 % round trip, and the label has to say so.
    assert label_at(rows, 0, horizon=1) == 0.0


def test_a_move_that_beats_the_cost_is_a_win():
    rows = bars(opens=[100.0, 100.0], closes=[100.0, 101.0])
    assert label_at(rows, 0, horizon=1) == 1.0


def test_the_boundary_is_strict_on_both_sides_of_the_cost():
    below = bars(opens=[100.0, 100.0], closes=[100.0, 100.0 * (1.0 + 0.9 * DEFAULT_COST)])
    above = bars(opens=[100.0, 100.0], closes=[100.0, 100.0 * (1.0 + 1.1 * DEFAULT_COST)])
    assert net_forward_return(below, 0, horizon=1) < 0.0
    assert label_at(below, 0, horizon=1) == 0.0
    assert net_forward_return(above, 0, horizon=1) > 0.0
    assert label_at(above, 0, horizon=1) == 1.0


def test_the_threshold_moves_the_bar_the_label_measures_against():
    rows = bars(opens=[100.0, 100.0], closes=[100.0, 101.0])
    assert label_at(rows, 0, horizon=1, threshold=2.0) == 0.0
    assert label_at(rows, 0, horizon=1, threshold=0.5) == 1.0


def test_a_zero_horizon_is_refused_instead_of_looking_at_the_same_bar_twice():
    rows = bars(opens=[100.0, 101.0], closes=[100.0, 101.0])
    assert forward_return(rows, 0, horizon=0) is None
    assert forward_return(rows, 0, horizon=-3) is None


def test_an_impossible_price_is_no_answer_rather_than_a_division():
    rows = bars(opens=[100.0, 0.0], closes=[100.0, 0.0])
    assert forward_return(rows, 0, horizon=1) is None
