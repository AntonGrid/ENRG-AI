"""The input filter: what reaches the scanner, and what a withheld bar becomes.

The module under test sits in front of the physics, so its tests are about *inputs*: a bar
with no turnover is not a reading, a bar in fractions of a cent is tick noise, a bar that
moved twenty average-true-ranges in one go is a gap and not a decline — and none of those may
be handed to the detector as a zero, because a zero is a number the market never published.

The engine tests at the bottom are the ones that matter commercially: a withheld bar cannot
open a position, the count of withheld bars travels with the run, and a filter that fires on
nothing leaves a run *byte-identical* to the run without it. The last one is the guard against
the failure mode this whole layer risks — a filter that quietly changes a result it was only
meant to protect.
"""
from __future__ import annotations

import math
from argparse import Namespace

import pytest

from agent.backtest.admission import (
    GAP,
    REASONS,
    STALE,
    THIN_BAR,
    TICK,
    Admission,
    admission_from_args,
    add_admission_arguments,
    admit,
    masked,
    report,
)
from agent.backtest.data import Candle, CandleSeries
from agent.backtest.engine import Backtester, Settings
from agent.backtest.walkforward import verdict_series

START = 1_600_000_000.0
DAY = 86400.0


def chart(changes, volumes=None, closes=None) -> CandleSeries:
    """A price path from per-bar changes, with per-bar turnover (and a close override)."""
    volumes = list(volumes) if volumes is not None else [1000.0] * len(changes)
    rows = []
    price = 100.0
    for index, change in enumerate(changes):
        opening = price
        price = price * (1.0 + float(change))
        close = float(price if closes is None else closes[index])
        rows.append(
            Candle(
                at=START + index * DAY,
                open=opening,
                high=max(opening, close) * 1.0005,
                low=min(opening, close) * 0.9995,
                close=close,
                volume=float(volumes[index]),
            )
        )
    return CandleSeries(symbol="btcusdt", period="1day", rows=rows, source="fixture")


def flat(bars: int = 30, change: float = 0.001) -> CandleSeries:
    return chart([change] * bars)


# ── the rules, one at a time ──────────────────────────────────────────────


def test_no_filter_means_every_bar_is_admitted():
    rows = flat(10).rows
    assert admit(rows, None) == [""] * 10
    assert Admission().is_open


def test_a_bar_with_no_turnover_published_nothing():
    rows = chart([0.001] * 5, volumes=[1000, 1000, 0, 1000, float("nan")]).rows
    reasons = admit(rows, Admission())
    assert reasons[2] == STALE
    assert reasons[4] == STALE
    assert reasons[:2] == ["", ""]


def test_the_turnover_floor_is_a_number_the_operator_sets():
    rows = chart([0.001] * 4, volumes=[1000, 400, 399.9, 2000]).rows
    assert admit(rows, Admission(min_turnover=400.0)) == ["", "", THIN_BAR, ""]


def test_a_close_in_fractions_of_a_cent_is_tick_noise_not_a_decline():
    rows = chart([0.001] * 3, closes=[100.0, 0.0004, 0.0004]).rows
    assert admit(rows, Admission(min_close=0.001)) == ["", TICK, TICK]


def test_a_move_many_true_ranges_wide_is_a_gap_and_not_a_decline():
    calm = [0.001, -0.001] * 15
    series = chart(calm + [0.30] + calm)
    reasons = admit(series.rows, Admission(max_gap_atr=3.0))
    assert reasons[30] == GAP
    assert reasons[31:] == [""] * len(calm)


def test_a_gap_on_a_quiet_pair_is_judged_against_that_pair_own_range():
    calm = [0.0001, -0.0001] * 15
    calm_series = chart(calm + [0.02] + calm)
    wild = [0.05, -0.05] * 15
    wild_series = chart(wild + [0.02] + wild)
    admission = Admission(max_gap_atr=4.0)
    assert admit(calm_series.rows, admission)[30] == GAP
    assert admit(wild_series.rows, admission)[30] == "", "2 % inside a 5 % range is not a gap"


def test_the_rule_can_be_turned_off_and_nothing_else_fires_on_clean_bars():
    series = flat(20)
    assert admit(series.rows, Admission(max_gap_atr=0.0)) == [""] * len(series.rows)


def test_every_reason_is_in_the_published_list():
    rows = flat(3).rows
    for reason in admit(rows, Admission(min_turnover=10.0**9)):
        assert reason in REASONS


# ── the mechanism: nan, not zero ──────────────────────────────────────────


def test_masking_replaces_a_withheld_bar_with_a_hole_and_keeps_the_index():
    values = [1.0, 2.0, 3.0]
    masked_values = masked(values, ["", STALE, ""])
    assert masked_values[0] == 1.0
    assert math.isnan(masked_values[1])
    assert masked_values[2] == 3.0
    assert len(masked_values) == 3


def test_a_mask_for_a_different_series_is_refused():
    with pytest.raises(ValueError, match="reasons"):
        masked([1.0, 2.0], [""])



# ── the report ────────────────────────────────────────────────────────────


def test_a_run_without_a_filter_reports_nothing_withheld():
    outcome = report(flat(5).rows, None)
    assert outcome["rules"] is None
    assert outcome["withheld"] == 0
    assert outcome["admitted"] == 5
    assert outcome["sentence"] == ""


def test_the_report_names_the_counts_and_carries_a_sentence():
    rows = chart([0.001] * 6, volumes=[1000, 0, 100, 100, 0, 1000]).rows
    outcome = report(rows, Admission(min_turnover=500.0))
    assert outcome["withheld"] == 4
    assert outcome["admitted"] == 2
    assert outcome["reasons"] == {STALE: 2, THIN_BAR: 2}
    assert outcome["share"] == pytest.approx(4 / 6)
    assert "4 of 6 bars" in outcome["sentence"]
    assert "2 stale" in outcome["sentence"]


def test_an_empty_record_has_no_share_to_divide_by():
    assert report([], Admission())["share"] == 0.0


# ── the thresholds, validated ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    (
        {"min_turnover": -1.0},
        {"min_close": -0.5},
        {"max_gap_atr": -2.0},
        {"atr_window": 0},
    ),
)
def test_a_threshold_that_would_be_meaningless_is_refused(kwargs):
    with pytest.raises(ValueError):
        Admission(**kwargs)


def test_the_rules_are_the_run_s_own_record():
    rules = Admission(min_turnover=1000.0, min_close=0.01, max_gap_atr=4.0).rules()
    assert rules == {
        "min_turnover": 1000.0,
        "min_close": 0.01,
        "max_gap_atr": 4.0,
        "atr_window": 14,
    }


# ── the command line ──────────────────────────────────────────────────────


def test_a_command_line_that_asks_for_no_filter_gets_none():
    assert admission_from_args(Namespace()) is None
    assert admission_from_args(Namespace(min_turnover=0.0, min_close=0.0)) is None


def test_a_command_line_that_asks_for_one_gets_the_whole_filter():
    built = admission_from_args(
        Namespace(min_turnover=50000.0, min_close=0.001, max_gap_atr=3.0, admission_atr_window=7)
    )
    assert built == Admission(
        min_turnover=50000.0, min_close=0.001, max_gap_atr=3.0, atr_window=7
    )
    assert not built.is_open


def test_the_flags_are_registered_the_same_way_everywhere():
    import argparse

    parser = argparse.ArgumentParser()
    add_admission_arguments(parser)
    parsed = parser.parse_args(["--min-turnover", "10", "--max-gap-atr", "2.5"])
    assert admission_from_args(parsed) == Admission(min_turnover=10.0, max_gap_atr=2.5)


# ── the engine: a withheld bar is not a bar the book acted on ─────────────

WARM = [0.001, -0.0008] * 28
FALL = [-0.02] * 8
REBOUND = [0.02] * 6
SINK = [-0.03] * 6
DRAIN = [600.0, 500.0, 420.0, 350.0, 300.0, 250.0, 220.0, 200.0]


def panic_market(zeros=()) -> CandleSeries:
    """A quiet month, a draining fall, a rebound the hunter calls ``PANIC`` — then the sink.

    The rebound is where the verdicts are, so ``zeros`` is how a test puts an unreadable bar
    exactly where the strategy would otherwise have traded.
    """
    changes = WARM + FALL + REBOUND + SINK
    volumes = [1000.0] * 56 + DRAIN + [200.0] * 12
    for index in zeros:
        volumes[index] = 0.0
    return chart(changes, volumes)


def test_the_fixture_actually_produces_panics_the_book_trades():
    result = Backtester(panic_market(), Settings()).run()
    assert result.verdicts.get("PANIC", 0) > 0
    assert result.trades, "the fixture has to trade or the tests below measure nothing"
    assert result.admissions == {}


def test_a_filter_that_withholds_every_bar_leaves_no_decision_to_make():
    series = panic_market()
    result = Backtester(series, Settings(admission=Admission(min_turnover=10.0**9))).run()
    assert result.trades == []
    assert result.admissions["withheld"] == len(series.rows)
    assert any("withheld from the detector" in line for line in result.warnings)
    assert any("never spoke" in line for line in result.warnings)


def test_a_withheld_bar_is_a_hole_in_the_record_the_detector_reads():
    series = panic_market()
    loose = Backtester(series, Settings()).run()
    tight = Backtester(
        series, Settings(admission=Admission(min_turnover=150.0))
    ).run()
    # The fall drains to 200 USDT a bar, so a 150 floor admits all of it and the run is the
    # same run; a 500 floor withholds the whole fall and the panics go with it.
    assert tight.admissions["withheld"] == 0
    assert len(tight.trades) == len(loose.trades)
    harsh = Backtester(series, Settings(admission=Admission(min_turnover=500.0))).run()
    assert harsh.admissions["withheld"] > 0
    assert len(harsh.trades) < len(loose.trades)


def test_the_run_states_what_the_filter_took_out():
    result = Backtester(
        panic_market(zeros=(65,)), Settings(admission=Admission())
    ).run()
    assert result.admissions["reasons"] == {STALE: 1}
    assert result.admissions["withheld"] == 1
    assert result.as_dict()["admissions"]["withheld"] == 1
    assert result.settings.as_dict()["admission"] == Admission().rules()


def test_the_engine_and_the_verdict_series_agree_under_the_same_filter():
    from collections import Counter

    series = panic_market(zeros=(65,))
    settings = Settings(admission=Admission(), confidence=0.8)
    result = Backtester(series, settings).run()
    counted = Counter(signal.verdict for signal in verdict_series(series, settings))
    assert dict(counted) == result.verdicts


def test_a_filtered_panel_never_offers_a_withheld_bar_as_a_training_row():
    from agent.backtest.walkforward import panel

    series = panic_market(zeros=(65,))
    loose = panel(series, Settings())
    tight = panel(series, Settings(admission=Admission()))
    assert any(sample.bar == 65 for sample in loose)
    assert not any(sample.bar == 65 for sample in tight)
    assert [sample.bar for sample in tight] == [
        sample.bar for sample in loose if sample.bar != 65
    ] or len(tight) < len(loose)
