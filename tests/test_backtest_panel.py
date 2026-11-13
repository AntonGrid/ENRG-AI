"""The panel: many pairs on one calendar, with every hole left visible.

A cross-sectional backtest is a comparison, and a comparison is only as honest as the calendar it
is made on. The interesting cases are therefore not the happy path but the holes: a symbol that
was not listed yet, a venue that skipped a bar, a name that stopped publishing, and two bar
widths that would make a four-hour move comparable with a daily one.

What this file proves:

* the union calendar is the clock, and a symbol absent from a timestamp is absent rather than
  carried forward from its last price;
* a return window that spans a hole is **not** a return over that window, and a record too short
  to hold the window has no score rather than a zero;
* one panel holds one bar width, one copy of each symbol, and at least one bar;
* a frozen panel round-trips through a directory exactly, and a universe that answers for some
  pairs and not others comes back with both the panel and the apology.
"""
from __future__ import annotations

import pytest

from agent.backtest.data import Candle, CandleSeries, HtxError
from agent.backtest.panel import (
    Panel,
    PanelError,
    build_panel,
    fetch_panel,
    load_panel,
    write_panel,
)

DAY = 86400.0


def chart(name, closes, *, start=0.0, step=DAY):
    """One symbol's candles: the open is the previous close, which is what a quiet venue does."""
    rows = []
    for index, close in enumerate(closes):
        opening = closes[index - 1] if index else close
        rows.append(
            Candle(
                at=start + index * step,
                open=float(opening),
                high=float(max(opening, close)),
                low=float(min(opening, close)),
                close=float(close),
                volume=1000.0,
            )
        )
    return CandleSeries(symbol=name, period="", rows=rows)


def two_symbols():
    return chart("aaa", [100.0, 110.0, 121.0]), chart("bbb", [100.0, 90.0, 81.0])


def test_a_panel_puts_every_symbol_on_one_union_calendar():
    panel = build_panel(two_symbols())

    assert panel.symbols == ["aaa", "bbb"]
    assert len(panel) == 3
    assert panel.ats == [0.0, DAY, 2 * DAY]
    assert panel.interval == DAY
    assert panel.as_dict()["coverage"] == 1.0


def test_a_missing_bar_is_a_hole_and_never_a_fill():
    early, late = chart("aaa", [100.0, 110.0, 121.0]), chart("bbb", [90.0, 81.0], start=DAY)
    panel = build_panel([early, late])

    assert len(panel) == 3
    assert panel.close("bbb", 0.0) is None
    assert panel.missing(0.0) == ["bbb"]
    assert panel.present(0.0) == ["aaa"]
    assert panel.missing(DAY) == []


def test_the_last_price_is_a_fact_from_the_past_and_never_from_the_future():
    panel = build_panel([chart("aaa", [100.0, 110.0, 121.0])])

    assert panel.last_at("aaa", 10 * DAY) == 2 * DAY
    assert panel.last_at("aaa", -DAY) is None


def test_a_window_that_spans_a_hole_is_not_a_return_over_that_window():
    gap = chart("aaa", [100.0, 110.0, 121.0])
    # Drop the middle bar: the record now jumps from day 0 to day 2.
    gap.rows = [gap.rows[0], gap.rows[2]]
    panel = build_panel([gap], period="1day")

    assert panel.interval == DAY
    assert panel.ret("aaa", 2 * DAY, 1) is None
    assert panel.ret("aaa", 2 * DAY, 2) is None


def test_without_a_declared_width_the_panel_measures_its_own_finest_gap():
    gap = chart("aaa", [100.0, 110.0, 121.0])
    gap.rows = [gap.rows[0], gap.rows[2]]
    panel = build_panel([gap])

    # Nothing was declared and there is one symbol, so the record *is* a two-day record as far
    # as the panel can tell; declaring the width is what turns the missing bar into a hole.
    assert panel.interval == 2 * DAY
    assert panel.as_dict()["interval_seconds"] == int(2 * DAY)


def test_a_record_too_short_for_the_window_has_no_score_rather_than_a_zero():
    panel = build_panel(two_symbols())

    assert panel.ret("aaa", DAY, 1) == pytest.approx(0.10)
    assert panel.ret("aaa", DAY, 2) is None
    assert panel.returns_at(DAY, 2) == {}


def test_a_hole_in_the_universe_is_counted_and_not_filled_in():
    panel = build_panel(
        [chart("aaa", [100.0, 101.0, 102.0]), chart("bbb", [50.0, 51.0], start=DAY)]
    )
    shape = panel.as_dict()

    assert shape["filled"] == 5
    assert shape["holes"] == 1
    assert shape["coverage"] == pytest.approx(5 / 6)


def test_two_bar_widths_are_not_one_market():
    with pytest.raises(PanelError) as failed:
        build_panel(
            [chart("aaa", [100.0, 101.0]), chart("bbb", [50.0, 51.0, 52.0], step=4 * 3600.0)]
        )

    assert "mixes bar widths" in str(failed.value)
    assert "aaa" in str(failed.value) and "bbb" in str(failed.value)


def test_a_declared_width_refuses_a_symbol_on_a_faster_clock_and_allows_a_hole():
    daily = chart("aaa", [100.0, 101.0, 102.0])
    holed = chart("bbb", [50.0, 51.0, 52.0])
    holed.rows = [holed.rows[0], holed.rows[2]]
    four_hourly = chart("ccc", [10.0, 11.0, 12.0, 13.0], step=4 * 3600.0)

    panel = build_panel([daily, holed], period="1day")
    assert panel.interval == DAY
    assert panel.missing(DAY) == ["bbb"]

    with pytest.raises(PanelError) as failed:
        build_panel([daily, four_hourly], period="1day")
    assert "mixes bar widths" in str(failed.value)


def test_the_same_symbol_twice_is_refused():
    with pytest.raises(PanelError) as failed:
        build_panel([chart("aaa", [100.0, 101.0]), chart("AAA", [99.0, 100.0])])

    assert "voted twice" in str(failed.value)


def test_a_symbol_with_no_bars_or_no_name_is_not_a_panel():
    with pytest.raises(PanelError):
        build_panel([CandleSeries(symbol="aaa", period="1day", rows=[])])
    with pytest.raises(PanelError):
        build_panel([chart("", [100.0, 101.0])])


def test_a_panel_round_trips_through_a_directory(tmp_path):
    panel = build_panel(two_symbols())
    written = write_panel(panel, tmp_path / "frozen")

    assert sorted(path.name for path in written) == ["aaa.csv", "bbb.csv"]
    back = load_panel(tmp_path / "frozen")

    assert back.symbols == panel.symbols
    assert back.ats == panel.ats
    for symbol in panel.symbols:
        for at in panel.ats_of[symbol]:
            assert back.close(symbol, at) == pytest.approx(panel.close(symbol, at))
    assert back.period == "1day"


def test_a_directory_with_no_csv_is_a_sentence_not_an_empty_panel(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(PanelError) as failed:
        load_panel(tmp_path / "empty")
    assert "no *.csv" in str(failed.value)

    with pytest.raises(PanelError):
        load_panel(tmp_path / "does-not-exist")


def test_a_universe_that_answers_for_some_pairs_comes_back_with_both_halves():
    def loader(symbol, period="1day", days=1.0, timeout=1.0, endpoint=""):
        if symbol == "deadusdt":
            raise HtxError("no bars for deadusdt")
        return chart(symbol, [100.0, 101.0, 102.0])

    panel, failures = fetch_panel(["aaausdt", "deadusdt", "bbbusdt"], loader=loader)

    assert panel.symbols == ["aaausdt", "bbbusdt"]
    assert failures == [("deadusdt", "no bars for deadusdt")]


def test_a_pair_that_answers_with_nothing_is_a_failure_and_not_a_column():
    def loader(symbol, period="1day", days=1.0, timeout=1.0, endpoint=""):
        if symbol == "emptyusdt":
            return CandleSeries(symbol="emptyusdt", period="1day", rows=[])
        return chart(symbol, [100.0, 101.0, 102.0])

    panel, failures = fetch_panel(["aaausdt", "emptyusdt"], loader=loader)

    assert panel.symbols == ["aaausdt"]
    assert failures == [("emptyusdt", "the feed answered with no bars")]


def test_a_universe_where_nothing_answered_refuses_rather_than_returning_nothing():
    def loader(symbol, period="1day", days=1.0, timeout=1.0, endpoint=""):
        raise HtxError(f"{symbol} is not a pair on this venue")

    with pytest.raises(PanelError) as failed:
        fetch_panel(["aaausdt", "bbbusdt"], loader=loader)

    assert "not one symbol" in str(failed.value)
    assert "aaausdt is not a pair" in str(failed.value)


def test_an_empty_panel_has_no_calendar_and_says_so():
    panel = Panel()
    assert len(panel) == 0
    assert panel.as_dict()["symbols"] == 0
    assert panel.index(0.0) == -1
    assert panel.at(0) is None

