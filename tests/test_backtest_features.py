"""The feature file, and the one property that makes it usable: it cannot see forward.

Every number in :mod:`agent.backtest.features` is a function of ``rows[:index + 1]``. That is
easy to write and easy to break — one ``max()`` over the whole sample, one z-score computed
after the fact, and the walk-forward would be measuring a model that already knew the answer.
So the first test here appends a violent future bar to a finished series and asserts that not
one feature of the past moved, which is a statement no implementation can satisfy by accident.

The rest are the ordinary ones a reader would want: the true range includes the gap, a flat
market has no trend and no RSI, the signed return is signed where the hunter's decline is not,
and the column order is a contract because weights are serialised against it.
"""
from __future__ import annotations

import math

from agent.backtest.data import Candle, CandleSeries
from agent.backtest.features import (
    FEATURES,
    POLICY_FEATURES,
    PRICE_FEATURES,
    bar_features,
    enrich,
    feature_matrix,
    feature_vector,
    series_features,
)

START = 1_600_000_000.0
PERIOD = 86400.0


def chart(closes, *, highs=None, lows=None, volumes=None) -> CandleSeries:
    """A series from closes, with the wick and the turnover supplied or inferred."""
    rows = []
    price = float(closes[0])
    for index, close in enumerate(closes):
        close = float(close)
        opening = price
        high = float(highs[index]) if highs else max(opening, close) * 1.001
        low = float(lows[index]) if lows else min(opening, close) * 0.999
        volume = float(volumes[index]) if volumes else 1000.0 + index
        rows.append(
            Candle(
                at=START + index * PERIOD,
                open=opening,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        price = close
    return CandleSeries(symbol="btcusdt", period="1day", rows=rows, source="fixture")


#: Forty bars of a market that wanders, so that the rolling windows have something to roll.
WANDER = [
    100.0,
    101.0,
    100.4,
    102.2,
    101.1,
    99.8,
    100.9,
    102.4,
    103.1,
    101.9,
    100.2,
    99.4,
    100.8,
    102.0,
    103.4,
    104.1,
    102.9,
    101.7,
    100.3,
    99.1,
    98.4,
    99.7,
    101.2,
    102.8,
    104.3,
    105.1,
    103.8,
    102.2,
    101.0,
    99.9,
    98.8,
    99.6,
    101.1,
    102.9,
    104.0,
    105.4,
    106.2,
    104.9,
    103.1,
    102.0,
]


def test_no_feature_of_the_past_moves_when_a_violent_future_bar_arrives():
    series = chart(WANDER)
    before = series_features(series.rows)
    future = Candle(
        at=START + len(WANDER) * PERIOD,
        open=102.0,
        high=180.0,
        low=40.0,
        close=60.0,
        volume=9_000_000.0,
    )
    after = series_features(list(series.rows) + [future])
    assert len(after) == len(before) + 1
    for index, row in enumerate(before):
        assert after[index] == row, f"bar {index} was rewritten by a bar that had not happened"


def test_bar_features_is_the_last_row_of_the_walk_that_stops_there():
    rows = chart(WANDER).rows
    walked = series_features(rows)
    for index in (0, 1, 7, len(rows) - 1):
        assert bar_features(rows, index) == walked[index]


def test_a_bar_outside_the_series_is_refused_rather_than_extrapolated():
    rows = chart(WANDER).rows
    for index in (-1, len(rows)):
        try:
            bar_features(rows, index)
        except IndexError as exc:
            assert "outside a series" in str(exc)
        else:  # pragma: no cover - the assertion below is the failure
            raise AssertionError(f"bar {index} was answered instead of refused")


def test_the_true_range_counts_the_gap_and_not_only_the_bar():
    # Two bars, each 2 wide, but the second opens 10 above the first's close: the gap is the
    # range a stop actually has to survive, and a bar-height ATR would miss it entirely.
    rows = [
        Candle(at=START, open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0),
        Candle(at=START + PERIOD, open=110.0, high=111.0, low=109.0, close=110.0, volume=1000.0),
    ]
    features = series_features(rows, atr_window=2)
    # TR of the second bar is 11 and not 10: the range runs from the previous *close* to the
    # new high, which is the gap a stop is exposed to.
    assert features[1]["atr_pct"] == 100.0 * ((2.0 + 11.0) / 2.0) / 110.0


def test_a_flat_market_has_no_trend_no_drawdown_and_no_rsi():
    series = chart([100.0] * 60, highs=[100.0] * 60, lows=[100.0] * 60)
    last = series_features(series.rows)[-1]
    assert last["close_sma20"] == 0.0
    assert last["close_sma200"] == 0.0
    assert last["dd_252"] == 0.0
    assert last["rsi14"] == 50.0
    assert last["ret_1"] == 0.0
    assert last["range_pct"] == 0.0
    assert last["atr_pct"] == 0.0


def test_the_signed_return_sees_a_rise_that_the_hunters_decline_cannot():
    up = series_features(chart([100.0, 101.0, 102.0]).rows)[-1]
    down = series_features(chart([102.0, 101.0, 100.0]).rows)[-1]
    assert up["ret_1"] > 0.0 and up["decline_1"] == 0.0
    assert down["ret_1"] < 0.0 and down["decline_1"] > 0.0


def test_every_feature_is_finite_on_a_degenerate_market():
    closes = [1e-6, 1e-6, 5e-7, 1e-6] + [2e-6] * 30
    series = chart(closes, volumes=[0.0] * 34)
    for row in series_features(series.rows):
        for name in FEATURES:
            assert name in row
            assert math.isfinite(row[name]), f"{name} is not finite: {row[name]!r}"


def test_the_close_position_says_where_in_the_bar_the_close_sat():
    rows = [
        Candle(at=START, open=100.0, high=110.0, low=90.0, close=109.0, volume=1.0),
        Candle(at=START + PERIOD, open=100.0, high=110.0, low=90.0, close=91.0, volume=1.0),
    ]
    features = series_features(rows)
    assert features[0]["close_pos"] > 0.9
    assert features[1]["close_pos"] < 0.1
    assert features[0]["range_pct"] == 100.0 * 20.0 / 109.0


def test_the_volatility_percentile_is_a_share_and_not_a_level():
    for row in series_features(chart(WANDER).rows):
        assert 0.0 <= row["vol_pct"] <= 1.0
        assert row["drain"] >= 0.0


def test_enrich_adds_exactly_the_three_readings_the_series_cannot_supply():
    base = series_features(chart(WANDER).rows)[-1]
    case = enrich(base, confidence=0.42, regime=0.5, bars_since_panic=3)
    assert set(case) == set(base) | {"confidence", "regime", "bars_since_panic"}
    assert case["confidence"] == 0.42
    assert case["regime"] == 0.5
    assert case["bars_since_panic"] == 3
    for name, value in base.items():
        assert case[name] == value


def test_the_column_order_is_the_contract_the_weights_are_written_against():
    assert FEATURES[:4] == ("ret_1", "ret_3", "ret_7", "decline_1")
    assert POLICY_FEATURES[: len(FEATURES)] == FEATURES
    assert POLICY_FEATURES[len(FEATURES) :] == ("confidence", "regime", "bars_since_panic")
    assert "turnover_z" not in PRICE_FEATURES and "atr_pct" in PRICE_FEATURES


def test_a_vector_reads_the_named_columns_and_calls_a_missing_one_zero():
    row = series_features(chart(WANDER).rows)[-1]
    vector = feature_vector(row)
    assert len(vector) == len(FEATURES)
    assert vector[FEATURES.index("atr_pct")] == row["atr_pct"]
    assert feature_vector({"atr_pct": 3.0}, names=("atr_pct", "ret_1")) == [3.0, 0.0]


def test_the_matrix_has_one_row_per_bar_and_matches_the_vectors():
    series = chart(WANDER)
    matrix = feature_matrix(series.rows)
    assert len(matrix) == len(series.rows)
    assert matrix[5] == feature_vector(series_features(series.rows)[5])

