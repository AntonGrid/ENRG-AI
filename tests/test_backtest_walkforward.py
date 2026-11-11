"""The walk-forward: expanding splits, an honest window, and a gate that can say no.

The fixture is a market of two identical panic episodes — a fall, a rebound that the hunter
calls ``PANIC``, and a second fall that takes the labels with it. That shape is chosen, not
found: it gives the training block *both* labels (four panics that paid, six that did not), so
a policy can actually be fitted inside the fold, and it puts the second episode strictly after
the split, so a run that peeked would light up here immediately.

What the file proves:

* the folds are half-open, consecutive and expanding, and a year too young to train is skipped
  rather than reported as a fold with nothing in it;
* the panel holds the bars the detector would have bought and nothing else, and the last bars
  of the record carry **no** label rather than a zero;
* the window reading is the book's growth and the trades opened inside it — a position carried
  in from training is in the ROI and not in the trade statistics;
* the pooled summary does the arithmetic the gate is written in, and names the failures.
"""
from __future__ import annotations

import calendar
import math

import pytest

from agent.backtest.data import Candle, CandleSeries
from agent.backtest.engine import Backtester, Settings
from agent.backtest.policy import LogisticPolicy
from agent.backtest.walkforward import (
    ABLATIONS,
    GATE,
    Ablation,
    Fold,
    Reading,
    expanding_folds,
    format_readings,
    format_summary,
    gate_of,
    panel,
    summary,
    verdict_series,
    walk_forward,
    _window,
)
from agent.physics.volatility_hunter import PANIC

START = 1_600_000_000.0
DAY = 86400.0


def chart(changes, volumes) -> CandleSeries:
    rows = []
    price = 100.0
    for index, (change, volume) in enumerate(zip(changes, volumes)):
        opening = price
        price = price * (1.0 + float(change))
        rows.append(
            Candle(
                at=START + index * DAY,
                open=opening,
                high=max(opening, price) * 1.001,
                low=min(opening, price) * 0.999,
                close=price,
                volume=float(volume),
            )
        )
    return CandleSeries(symbol="btcusdt", period="1day", rows=rows, source="fixture")


#: 56 quiet bars, then two panic episodes. The rebound is what the hunter calls ``PANIC`` (a
#: yield jump with the depth still drained), and the second fall is what makes some of those
#: panics pay and the rest lose — a fixture where every label was the same would make the fit
#: untestable and the note about it the only thing measured.
WARM = [0.001, -0.0008] * 28
FALL = [-0.02] * 8
REBOUND = [0.02] * 6
SINK = [-0.03] * 6
QUIET = [0.0005] * 40
DRAIN = [600.0, 500.0, 420.0, 350.0, 300.0, 250.0, 220.0, 200.0]

CHANGES = WARM + FALL + REBOUND + SINK + QUIET + FALL + REBOUND + SINK + [0.0005] * 20
VOLUMES = (
    [1000.0] * 56
    + DRAIN
    + [200.0] * 12
    + [200.0] * 40
    + DRAIN
    + [200.0] * 12
    + [200.0] * 20
)

#: The split: everything before bar 100 is the training block, the rest is the test window.
SPLIT = 100


def market() -> CandleSeries:
    return chart(CHANGES, VOLUMES)


def readings() -> list:
    """Four hand-built readings: two folds of a strategy that pays, and its baseline."""
    return [
        Reading("baseline", "2024", 10, 4, -0.05, 0.04, -0.09, 0.25, 0.4, -0.5, -0.8,
                returns=(-0.03, -0.02, 0.01, -0.01)),
        Reading("baseline", "2025", 12, 4, -0.04, 0.10, -0.14, 0.25, 0.3, -0.6, -0.7,
                returns=(-0.04, 0.02, -0.03, -0.01)),
        Reading("policy", "2024", 10, 3, 0.06, 0.04, 0.02, 0.67, 2.5, 1.4, 3.2,
                returns=(0.05, 0.03, -0.02)),
        Reading("policy", "2025", 12, 3, 0.05, 0.10, -0.05, 0.67, 2.0, 1.2, 2.9,
                returns=(0.04, 0.03, -0.02)),
    ]


def daily_year(year: int, days: int = 365) -> CandleSeries:
    """A flat year of daily bars, stamped from January 1st — the calendar a fold is named by."""
    first = float(calendar.timegm((year, 1, 1, 0, 0, 0, 0, 0, 0)))
    rows = [
        Candle(
            at=first + index * DAY,
            open=100.0,
            high=100.1,
            low=99.9,
            close=100.0,
            volume=1000.0,
        )
        for index in range(days)
    ]
    return CandleSeries(symbol="btcusdt", period="1day", rows=rows, source="fixture")


# ── the splits ─────────────────────────────────────────────────────────────


def test_folds_are_consecutive_half_open_and_expanding():
    rows = []
    for year in (2021, 2022, 2023, 2024, 2025):
        rows.extend(daily_year(year, 20).rows)
    series = CandleSeries(symbol="btcusdt", period="1day", rows=rows, source="fixture")
    folds = expanding_folds(series.rows, splits=3, min_train=10)
    assert [fold.name for fold in folds] == ["2023", "2024", "2025"]
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end == later.test_start, "a bar was counted in two folds"
        assert later.train_bars > earlier.train_bars, "the window did not expand"
    assert folds[-1].train_bars == folds[-1].test_start
    assert folds[0].test_bars == 20


def test_a_year_whose_training_block_is_too_young_is_skipped_not_reported():
    rows = []
    for year in (2024, 2025):
        rows.extend(daily_year(year, 10).rows)
    series = CandleSeries(symbol="btcusdt", period="1day", rows=rows, source="fixture")
    # Ten bars of 2024 cannot train a detector that needs sixty-four.
    assert expanding_folds(series.rows, splits=1, min_train=64) == []


def test_naming_the_years_is_honoured_and_a_missing_one_is_skipped():
    rows = []
    for year in (2022, 2023, 2024):
        rows.extend(daily_year(year, 30).rows)
    series = CandleSeries(symbol="btcusdt", period="1day", rows=rows, source="fixture")
    folds = expanding_folds(series.rows, years=[2023, 1999], min_train=10)
    assert [fold.name for fold in folds] == ["2023"]
    assert folds[0].test_bars == 30


def test_a_fold_reports_its_own_bounds():
    fold = Fold("2024", 400, 700)
    assert fold.train_bars == 400
    assert fold.test_bars == 300
    assert fold.as_dict()["test_end"] == 700


# ── the panel ─────────────────────────────────────────────────────────────


def test_the_verdict_series_answers_once_per_bar():
    series = market()
    words = verdict_series(series, Settings())
    assert len(words) == len(series.rows)
    assert any(word.verdict == PANIC for word in words)


def test_the_panel_holds_the_panics_the_detector_would_have_bought():
    series = market()
    words = verdict_series(series, Settings())
    samples = panel(series, Settings())
    panic_bars = [index for index, word in enumerate(words) if word.verdict == PANIC]
    assert [sample.bar for sample in samples] == panic_bars
    assert len(samples) == len(panic_bars) > 5


def test_a_confidence_floor_narrows_the_panel_to_the_bars_that_cleared_it():
    series = market()
    words = verdict_series(series, Settings())
    floor = 0.83
    wide = panel(series, Settings(confidence=floor))
    wanted = [
        index
        for index, word in enumerate(words)
        if word.verdict == PANIC and word.confidence >= floor
    ]
    assert [sample.bar for sample in wide] == wanted
    assert len(wide) < len(panel(series, Settings()))


def test_the_last_bars_of_the_record_have_no_label_rather_than_a_zero():
    # Truncated two bars after the last panic, so that panic's exit is genuinely absent.
    series = chart(CHANGES[:138], VOLUMES[:138])
    samples = panel(series, Settings(), horizon=3)
    assert samples, "the fixture produced no panics at all"
    assert samples[-1].bar == 136
    assert samples[-1].net is None
    assert samples[-1].label is None
    assert [sample.label for sample in samples[-2:]] == [None, None]
    assert all(sample.label is not None for sample in samples[:-2])


def test_every_case_carries_the_three_readings_the_policy_needs():
    samples = panel(market(), Settings())
    for sample in samples:
        for name in ("confidence", "regime", "bars_since_panic", "atr_pct", "ret_1"):
            assert name in sample.case
        assert 0.0 <= sample.case["confidence"] <= 1.0
        assert sample.case["regime"] in (0.0, 0.5, 1.0)


def test_a_panel_built_from_the_wrong_series_is_refused():
    series = market()
    with pytest.raises(ValueError, match="bars against the series"):
        panel(series, Settings(), signals=verdict_series(series, Settings())[:10])



# ── the window ────────────────────────────────────────────────────────────


def test_the_window_reads_the_book_and_only_the_trades_opened_inside_it():
    series = market()
    settings = Settings()
    result = Backtester(series, settings).run()
    fold = Fold("late", SPLIT, len(series.rows))
    reading = _window(result, fold, "baseline", 10, 365.25)
    equity = result.equity
    expected_roi = float(equity[fold.test_end - 1]["equity"]) / float(
        equity[fold.test_start - 1]["equity"]
    ) - 1.0
    assert reading.roi == pytest.approx(expected_roi)
    rows = series.rows
    assert reading.buy_hold == pytest.approx(
        float(rows[fold.test_end - 1].close) / float(rows[fold.test_start - 1].close) - 1.0
    )
    inside = [
        trade
        for trade in result.trades
        if not trade.open and fold.test_start <= trade.entry_bar < fold.test_end
    ]
    assert reading.trades == len(inside)
    assert reading.returns == tuple(float(trade.pnl_pct) for trade in inside)
    assert reading.stops == sum(1 for trade in inside if trade.exit_reason == "stop")
    assert reading.edge == pytest.approx(reading.roi - reading.buy_hold)
    assert reading.note == ""


def test_an_empty_window_is_a_reading_with_a_sentence_and_not_a_crash():
    series = market()
    result = Backtester(series, Settings()).run()
    reading = _window(result, Fold("nowhere", len(series.rows), len(series.rows) + 5), "x", 0, 365.25)
    assert reading.trades == 0
    assert reading.roi == 0.0
    assert "no bars" in reading.note


def test_the_mean_trade_in_basis_points_is_the_mean_of_the_returns():
    fold = Fold("2024", 10, 20)
    reading = Reading(
        "policy", "2024", 5, 3, 0.02, 0.0, 0.02, 0.67, 2.0, 1.0, 1.0,
        returns=(0.01, 0.002, -0.006),
    )
    assert reading.avg_bps == pytest.approx(1e4 * (0.01 + 0.002 - 0.006) / 3.0)
    assert Reading("x", "y", 0, 0, 0, 0, 0, 0, 0, 0, 0).avg_bps == 0.0


# ── the walk ──────────────────────────────────────────────────────────────


def test_a_walk_forward_measures_every_ablation_on_the_test_window_only():
    series = market()
    fold = Fold("late", SPLIT, len(series.rows))
    walked = walk_forward(series, folds=[fold])
    assert [reading.ablation for reading in walked] == [item.name for item in ABLATIONS]
    assert all(reading.fold == "late" for reading in walked)
    assert {reading.train_samples for reading in walked} == {walked[0].train_samples}
    assert walked[0].train_samples > 0
    for reading in walked:
        assert reading.trades <= len(series.rows) - SPLIT


def test_the_policy_is_fitted_on_the_panics_before_the_split_and_no_others():
    series = market()
    fold = Fold("late", SPLIT, len(series.rows))
    samples = [sample for sample in panel(series, Settings()) if sample.bar < SPLIT]
    labelled = [sample for sample in samples if sample.label is not None]
    walked = walk_forward(series, folds=[fold], ablations=[ABLATIONS[1]])
    assert walked[0].train_samples == len(labelled)
    assert {sample.label for sample in labelled} == {0.0, 1.0}


def test_two_identical_walks_are_two_identical_readings():
    series = market()
    fold = Fold("late", SPLIT, len(series.rows))
    first = walk_forward(series, folds=[fold], ablations=ABLATIONS[:2])
    second = walk_forward(series, folds=[fold], ablations=ABLATIONS[:2])
    assert [reading.as_dict() for reading in first] == [
        reading.as_dict() for reading in second
    ]


def test_a_training_block_that_cannot_teach_a_ranking_is_measured_without_a_model(
    monkeypatch,
):
    series = market()
    fold = Fold("late", SPLIT, len(series.rows))

    def refuse(cls, *args, **kwargs):
        raise ValueError("every one of 10 samples carries the same label")

    monkeypatch.setattr(LogisticPolicy, "fit", classmethod(refuse))
    walked = walk_forward(series, folds=[fold], ablations=ABLATIONS[:2])
    assert walked[1].ablation == "policy"
    assert "refused the training block" in walked[1].note
    assert walked[1].trades == walked[0].trades, "the row ran as the baseline, by design"


def test_no_folds_means_no_readings_rather_than_a_quiet_zero():
    assert walk_forward(market(), folds=[]) == []


def test_a_regime_ablation_without_a_benchmark_says_so():
    series = market()
    fold = Fold("late", SPLIT, len(series.rows))
    walked = walk_forward(series, folds=[fold], ablations=[ABLATIONS[3]])
    assert "no benchmark" in walked[0].note



def test_a_run_writes_the_criteria_it_read_the_table_against(monkeypatch, tmp_path):
    """The gate leaves with the numbers, hashed — a table nobody can audit is a claim."""
    import json

    from agent.backtest import walkforward
    from agent.backtest.criteria import declared, verify

    monkeypatch.setattr(walkforward, "_series", lambda *args, **kwargs: market())
    folder = tmp_path / "wf"
    code = walkforward.main(
        [
            "--symbols",
            "btcusdt",
            "--folds",
            "2021",
            "--min-train",
            "60",
            "--out",
            str(folder),
            "--stop-mode",
            "atr",
            "--take-profit-r",
            "1.5",
        ]
    )

    assert code == 0
    document = json.loads((folder / "criteria.json").read_text())
    assert verify(document) == "", "the written document must hash back to itself"
    assert document["criteria_sha256"] == declared().fingerprint()
    # The exits a table was produced with travel with it, or two tables are indistinguishable.
    assert document["run"]["exits"] == {"stop_mode": "atr", "take_profit_r": 1.5}
    assert {row["verdict"] for row in document["run"]["summaries"]} == {"unmeasured"}


def test_a_frozen_directory_reproduces_the_table_bar_for_bar(monkeypatch, tmp_path):
    """The point of ``--candles``: the same bars give the same table, on any machine.

    The venue serves a moving window, so two runs of the same command an hour apart are two
    experiments — one new bar closes and every fold boundary shifts. Freezing the bars first
    is what makes "the VPS reproduced it" a checkable sentence rather than a hopeful one, so
    the test asserts bytes, not agreement about conclusions.
    """
    import json

    from agent.backtest import walkforward
    from agent.backtest.data import write_csv

    series = market()
    frozen = tmp_path / "frozen"
    write_csv(series, frozen / "btcusdt.csv")

    live_out = tmp_path / "live"
    monkeypatch.setattr(walkforward, "_series", lambda *args, **kwargs: market())
    assert walkforward.main(
        ["--symbols", "btcusdt", "--folds", "2021", "--min-train", "60",
         "--out", str(live_out), "--quiet"]
    ) == 0
    monkeypatch.undo()

    replay_out = tmp_path / "replay"
    # No monkeypatch: this reads the file, and would reach for the venue if it did not.
    assert walkforward.main(
        ["--candles", str(frozen), "--folds", "2021", "--min-train", "60",
         "--out", str(replay_out), "--quiet"]
    ) == 0

    assert (replay_out / "walkforward.csv").read_bytes() == (
        live_out / "walkforward.csv"
    ).read_bytes()

    document = json.loads((replay_out / "criteria.json").read_text())
    candles = document["run"]["candles"]
    assert candles["files"] == ["btcusdt.csv"]
    assert len(candles["sha256"]) == 64

    # A different freeze is a different input, and the hash has to say so — otherwise it is
    # decoration that would agree with any comparison made in its name.
    write_csv(market(), frozen / "ethusdt.csv")
    assert walkforward.main(
        ["--candles", str(frozen), "--folds", "2021", "--min-train", "60",
         "--out", str(tmp_path / "two"), "--quiet"]
    ) == 0
    second = json.loads((tmp_path / "two" / "criteria.json").read_text())
    assert second["run"]["candles"]["sha256"] != candles["sha256"]


def test_saving_candles_writes_what_the_run_read(monkeypatch, tmp_path):
    """``--save-candles`` is how a table gets a receipt: the bars it was computed from."""
    import json

    from agent.backtest import walkforward

    monkeypatch.setattr(walkforward, "_series", lambda *args, **kwargs: market())
    folder = tmp_path / "saved"
    assert walkforward.main(
        ["--symbols", "btcusdt", "--folds", "2021", "--min-train", "60",
         "--out", str(tmp_path / "wf"), "--save-candles", str(folder), "--quiet"]
    ) == 0

    written = folder / "btcusdt.csv"
    assert written.is_file()
    document = json.loads((tmp_path / "wf" / "criteria.json").read_text())
    # Nothing was read from a frozen directory, so the run is not claimed to be replayable.
    assert document["run"]["candles"]["dir"] == ""
