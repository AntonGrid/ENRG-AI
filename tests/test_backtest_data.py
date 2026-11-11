"""The feed and the one conversion: what a candle means to a hunter built for pools.

These tests never open a socket. Every answer from the venue is a dictionary built in the
test, handed to :func:`agent.backtest.data.parse_klines` or to ``fetch_klines(fetch=...)`` —
which is the reason the transport is a parameter and not an import.

What is being proven, in the order the file proves it:

* **the order.** The venue answers newest first and every consumer downstream assumes
  oldest first, so the reversal is asserted on the parsed rows rather than trusted.
* **the units.** ``vol`` is quote turnover and ``amount`` is base volume; a bar with only
  the second gets the first derived from it and the price. Getting this wrong would make
  the exodus arm compare base units to dollars and read every panic as growth.
* **the failures.** A refused request, a document with no ``data``, a row with no price, a
  period the venue does not publish: each is an :class:`HtxError` with a sentence, never a
  ``KeyError`` three frames down and never an empty series that looks like a quiet market.
* **the mapping.** ``declines`` is the only invented quantity in the package, so it is
  pinned here: a bar that rose pays 0, a bar that fell 2.5 % pays 2.5, and the first bar of
  a prefix pays ``nan`` because it has nothing to be compared with.
"""
from __future__ import annotations

import math

import pytest

from agent.backtest import data as D

#: A venue answer with two bars, newest first, in the shape HTX actually returns — including
#: the scientific-notation turn that a JSON float becomes on the wire.
PAYLOAD = {
    "ch": "market.btcusdt.kline.4hour",
    "status": "ok",
    "ts": 1700000000000,
    "data": [
        {
            "id": 1700000000,
            "open": 101.0,
            "close": 103.0,
            "high": 104.0,
            "low": 100.5,
            "amount": 12.5,
            "vol": 1287.5,
            "count": 9,
        },
        {
            "id": 1699992800,
            "open": 100.0,
            "close": 101.0,
            "high": 101.5,
            "low": 99.5,
            "amount": 10.0,
            "vol": 1005.0,
            "count": 7,
        },
    ],
}


def _row(at: float, close: float, *, open_: float = 100.0, volume: float = 1000.0) -> dict:
    return {
        "id": at,
        "open": open_,
        "close": close,
        "high": max(open_, close) * 1.01,
        "low": min(open_, close) * 0.99,
        "amount": 1.0,
        "vol": volume,
        "count": 1,
    }


def test_the_venue_answer_is_read_oldest_first():
    series = D.parse_klines(PAYLOAD, symbol="btcusdt", period="4hour")

    assert [row.at for row in series] == [1699992800.0, 1700000000.0]
    assert series.closes == [101.0, 103.0]
    assert series.period == "4hour"
    assert series.interval == 7200.0


def test_the_depth_is_quote_turnover_and_falls_back_to_base_volume_times_price():
    payload = {
        "status": "ok",
        "data": [
            {"id": 1700000000, "open": 50.0, "close": 50.0, "high": 50.0, "low": 50.0},
            {
                "id": 1699992800,
                "open": 100.0,
                "close": 100.0,
                "high": 100.0,
                "low": 100.0,
                "amount": 20.0,
            },
        ],
    }

    series = D.parse_klines(payload, symbol="btcusdt")

    # Oldest first: the row with only base volume publishes 20 units × 100 = 2000 quoted,
    # and the row that published neither weight of volume gets 0.0 — a depth nobody
    # declared, which the exodus arm reports instead of silently treating as "intact".
    assert series.volumes == [2000.0, 0.0]
    assert series.closes == [100.0, 50.0]


def test_a_refused_request_is_a_sentence_and_not_a_key_error():
    with pytest.raises(D.HtxError) as refused:
        D.parse_klines(
            {"status": "error", "err-code": "invalid-parameter", "err-msg": "bad symbol"}
        )

    assert "invalid-parameter" in str(refused.value)
    assert "bad symbol" in str(refused.value)


def test_an_answer_without_a_data_list_is_an_error_not_an_empty_market():
    with pytest.raises(D.HtxError, match="no `data` list"):
        D.parse_klines({"status": "ok"})


def test_an_answer_that_is_not_an_object_is_an_error():
    with pytest.raises(D.HtxError, match="not a JSON object"):
        D.parse_klines([{"id": 1}])


def test_a_row_without_a_price_is_dropped_and_counted():
    payload = {
        "status": "ok",
        "data": [
            _row(1700000000, 100.0),
            {"id": 1699992800, "open": 1.0},
        ],
    }

    series = D.parse_klines(payload, symbol="btcusdt")

    assert len(series) == 1
    assert any("unreadable" in warning for warning in series.warnings)


def test_an_answer_with_no_readable_candle_is_an_error():
    with pytest.raises(D.HtxError, match="no readable candle"):
        D.parse_klines({"status": "ok", "data": [{"id": 0, "close": 0}]})


def test_the_fetch_asks_for_the_window_it_was_given_in_bars():
    asked = {}

    def fetch(url, timeout):
        asked["url"] = url
        asked["timeout"] = timeout
        return PAYLOAD

    series = D.fetch_klines("BTCUSDT", period="4hour", days=180, fetch=fetch, timeout=7.0)

    # 180 days of 4-hour bars is 1080 bars, and the symbol is asked for in lower case
    # because that is what the venue's own documentation uses.
    assert "symbol=btcusdt" in asked["url"]
    assert "period=4hour" in asked["url"]
    assert "size=1080" in asked["url"]
    assert asked["timeout"] == 7.0
    assert len(series) == 2
    assert series.symbol == "btcusdt"


def test_a_window_beyond_what_the_venue_serves_is_cut_and_said_so():
    def fetch(url, timeout):
        return {"status": "ok", "data": [PAYLOAD["data"][0]]}

    series = D.fetch_klines("btcusdt", period="4hour", days=800, fetch=fetch)

    assert any("2000" in warning for warning in series.warnings)
    assert any("instead of 800" in warning for warning in series.warnings)


def test_a_window_the_symbol_cannot_fill_is_reported_too():
    def fetch(url, timeout):
        return {"status": "ok", "data": [PAYLOAD["data"][0]]}

    series = D.fetch_klines("btcusdt", period="4hour", days=10, fetch=fetch)

    assert any("came back" in warning for warning in series.warnings)


def test_an_unknown_period_is_an_error_before_any_request_is_made():
    def fetch(url, timeout):  # pragma: no cover - it must not be called
        raise AssertionError("the transport must not be reached for an unknown period")

    with pytest.raises(D.HtxError, match="not a period"):
        D.fetch_klines("btcusdt", period="3hour", days=10, fetch=fetch)


def test_the_bars_the_window_asks_for_are_the_last_ones():
    payload = {
        "status": "ok",
        "data": [_row(1700000000 - index * 14400, 100.0) for index in range(5)],
    }

    series = D.fetch_klines(
        "btcusdt", period="4hour", days=1, fetch=lambda url, timeout: payload
    )

    # One day of 4-hour bars is six bars; five were published, so the answer is those five.
    assert len(series) == 5


# ── the conversion: a bar as the rate it offered a buyer ────────────────────


def _series(*closes: float, volume: float = 1000.0) -> D.CandleSeries:
    rows = [
        D.Candle(
            at=1_600_000_000.0 + index * 14400.0,
            open=closes[index - 1] if index else close,
            high=max(closes[index - 1] if index else close, close),
            low=min(closes[index - 1] if index else close, close),
            close=close,
            volume=volume,
        )
        for index, close in enumerate(closes)
    ]
    return D.CandleSeries(symbol="btcusdt", period="4hour", rows=rows)


def test_a_bar_that_fell_pays_the_fall_and_a_bar_that_rose_pays_nothing():
    values = D.declines(_series(100.0, 97.5, 100.0, 101.0))

    assert math.isnan(values[0])
    assert values[1] == pytest.approx(2.5)
    assert values[2] == 0.0
    assert values[3] == 0.0


def test_a_longer_window_measures_the_fall_over_that_window():
    values = D.declines(_series(100.0, 95.0, 90.0, 96.0), window=2)

    assert math.isnan(values[0]) and math.isnan(values[1])
    assert values[2] == pytest.approx(10.0)  # 100 → 90
    assert values[3] == pytest.approx(0.0)  # 95 → 96 is a rise


def test_the_warm_up_rows_answer_nan_and_not_zero():
    values = D.declines(_series(100.0, 99.0, 98.0), window=2)

    # A zero would be a rate the market never published, and it would sit in the resting
    # median of every early prefix; `nan` is dropped by the detector instead.
    assert math.isnan(values[0]) and math.isnan(values[1])
    assert values[2] == pytest.approx((1 - 98.0 / 100.0) * 100.0)


def test_a_bar_with_no_previous_close_has_no_rate():
    rows = [
        D.Candle(at=1.0, open=0.0, high=0.0, low=0.0, close=0.0),
        D.Candle(at=2.0, open=1.0, high=1.0, low=1.0, close=1.0),
    ]

    values = D.declines(rows)

    assert math.isnan(values[0])
    assert math.isnan(values[1])


# ── a chart from a file ────────────────────────────────────────────────────


def test_csv_is_read_in_the_format_the_brief_names(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text(
        "timestamp,open,high,low,close,volume\n"
        "1600007200,100,101,99,100.5,1000\n"
        "1599992800,99,100,98,99.5,900\n"
        "1600021600,101,102,100,101.5,1100\n"
        "1600021600,101,102,100,101.5,1100\n",
        encoding="utf-8",
    )

    series = D.load_csv(path)

    assert [row.close for row in series] == [99.5, 100.5, 101.5]
    assert series.period == "4hour"  # inferred from the gap, not inherited from the default
    assert series.symbol == "chart"
    assert any("duplicate" in warning for warning in series.warnings)


def test_csv_accepts_iso_timestamps_and_milliseconds(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text(
        "time,close,volume\n"
        "2020-09-13T00:00:00Z,100,1\n"
        "1600041600000,101,1\n",
        encoding="utf-8",
    )

    series = D.load_csv(path)

    # `2020-09-13T00:00:00Z` and 1600041600000 ms are consecutive days, and neither is
    # refused for being in the wrong spelling of "when".
    assert [row.at for row in series] == [1599955200.0, 1600041600.0]


def test_csv_without_a_close_column_names_the_column_it_wants(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("timestamp,open,high,low\n1600000000,1,1,1\n", encoding="utf-8")

    with pytest.raises(D.HtxError, match="no close column"):
        D.load_csv(path)


def test_csv_line_that_is_not_a_bar_names_the_line(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("timestamp,close\n1600000000,not-a-price\n", encoding="utf-8")

    with pytest.raises(D.HtxError, match=":2 is not a bar"):
        D.load_csv(path)


def test_csv_that_does_not_exist_is_a_sentence(tmp_path):
    with pytest.raises(D.HtxError, match="cannot read"):
        D.load_csv(tmp_path / "absent.csv")


def test_a_written_series_reads_back_as_the_same_floats(tmp_path):
    """A freeze that loses the last digit is a different experiment wearing the same name.

    ``--save-candles`` exists so a second machine can read the same bars, and the only thing
    that makes it worth doing is exactness: the floats must come back bit for bit, or the
    reproduction test is testing the writer's rounding.
    """
    rows = [
        D.Candle(at=1735689600.0, open=1e-05, high=0.000123456789, low=1.5e-08,
                 close=67651.23456789012, volume=12345.678901234567),
        D.Candle(at=1735704000.0, open=3.0, high=3.0, low=3.0, close=3.0, volume=0.0),
    ]
    series = D.CandleSeries(symbol="btcusdt", period="4hour", rows=rows)

    path = D.write_csv(series, tmp_path / "frozen" / "btcusdt.csv")
    back = D.load_csv(path, symbol="btcusdt", period="4hour")

    assert back.rows == rows
    assert back.period == "4hour"
    assert back.warnings == []
    # Epoch seconds with a header, so the file is also readable by anything else, and the
    # trade count travels with it because it is part of the bar the venue published.
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "timestamp,open,high,low,close,volume,trades"
    )


def test_an_empty_file_is_refused_rather_than_measured(tmp_path):
    path = tmp_path / "chart.csv"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(D.HtxError, match="is empty"):
        D.load_csv(path)


def test_a_period_name_is_measured_and_not_assumed():
    assert D.period_name(14400) == "4hour"
    assert D.period_name(86400) == "1day"
    assert D.period_name(0) == ""
    assert D.period_name(7) == "7s"
