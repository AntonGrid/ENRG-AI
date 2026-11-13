"""The cross-sectional book: neutral by construction, honest about what it cannot resolve.

The single-pair engine answers "would this trade have paid". This book answers a different
question — "which of these names should I be long *today*" — and the tests below are mostly
about the ways that question can be answered dishonestly:

* **neutrality is structural, not a claim.** A flat market and a market-wide gap must leave the
  book's equity where it was, with ``long == short`` at every mark. If that ever fails, the
  book is a directional bet with a second leg and the word "neutral" is decoration.
* **a decision cannot see the bar it fills on.** The whole panel is re-read truncated one bar
  earlier; every mark up to the cut must be byte-identical, so no future bar can have reached
  the past.
* **the fill is the open and the mark is the close,** which is what makes a bar's P&L a holding
  period instead of the same close counted twice. A book that filled at the close earns nothing
  on the crafted reversal below, and this file checks that it does earn.
* **costs are the declared rate on the notional that actually moved,** including the exit of a
  name the venue stopped quoting.
* **a mean inside its own noise band is not an edge.** The resolution rule is tested in both
  directions: a small mean on a wide spread is named unresolvable, and a clear one on a narrow
  spread is measurable.
"""
from __future__ import annotations

import csv
import json
import random

import pytest

from agent.backtest.criteria import verify
from agent.backtest.crosssection import (
    HEADER,
    BasketReading,
    BasketResult,
    BasketSettings,
    Mark,
    Window,
    _target_weights,
    book,
    configs,
    format_table,
    grid,
    main,
    read,
    verdict_of,
    write_marks,
)
from agent.backtest.data import Candle, CandleSeries
from agent.backtest.panel import Panel, PanelError, build_panel, load_panel, write_panel
from agent.backtest.run import EXIT_FEED, EXIT_SHORT, EXIT_USAGE
from agent.backtest.simulated_provider import FEE_RATE, FILL_SLIPPAGE

DAY = 86400.0


def chart(name, closes, *, start=0.0, step=DAY):
    """One symbol's candles, opened at the previous close — the fill is the open, the mark the close."""
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
    return CandleSeries(symbol=name, period="1day", rows=rows)


def panel_of(paths, *, period="1day") -> Panel:
    return build_panel([chart(name, values) for name, values in paths.items()], period=period)


def flat(symbols, bars, price=100.0):
    return {name: [price] * bars for name in symbols}


def walk(seed, symbols, bars, *, vol=0.01, plant=0.0, look=3, until=None):
    """A deterministic random walk, with an optional planted reversal that stops at ``until``.

    ``plant`` is the strength of the negative feedback: each new bar fades ``plant`` times the
    move over the previous ``look`` bars, which is exactly the effect a reversal book is
    looking for. The feedback stops at bar ``until``, so a panel can hold an effect that *dies*
    — the case the held-back tail exists to catch.
    """
    rng = random.Random(seed)
    paths = {}
    for name in symbols:
        prices = [100.0]
        for index in range(1, bars):
            shock = rng.gauss(0.0, vol)
            pull = 0.0
            if plant and index >= look and (until is None or index < until):
                past = prices[index - 1] / prices[index - look] - 1.0
                pull = -plant * past
            prices.append(max(prices[-1] * (1.0 + shock + pull), 1.0))
        paths[name] = prices
    return paths


NAMES = [f"s{index:02d}" for index in range(10)]


def test_a_flat_market_leaves_the_book_dollar_neutral_and_pays_only_the_fee():
    panel = panel_of(flat([f"s{index:02d}" for index in range(12)], 40))
    settings = BasketSettings(lookback=2, rebalance_bars=5, slippage=0.0)

    result = book(panel, settings)

    assert len(result.marks) == 40 - (settings.lookback + settings.latency_bars)
    assert all(mark.names == 12 for mark in result.marks)
    for mark in result.marks:
        assert mark.long == pytest.approx(mark.short)
        assert mark.net == pytest.approx(0.0, abs=1e-9)
    # Nothing moved, so the only thing that happened to the equity is the fee it paid.
    assert result.equities[-1] == pytest.approx(settings.capital - result.fees)
    assert result.fees > 0.0


def test_a_market_wide_gap_leaves_a_neutral_book_where_it_was():
    paths = flat([f"s{index:02d}" for index in range(12)], 30)
    for name in paths:
        paths[name] = [100.0] * 22 + [110.0] * 8
    settings = BasketSettings(lookback=2, rebalance_bars=5, fee=0.0, slippage=0.0)

    result = book(panel_of(paths), settings)
    gap = result.marks[22 - (settings.lookback + settings.latency_bars)]
    before = result.marks[22 - (settings.lookback + settings.latency_bars) - 1]

    assert gap.long == pytest.approx(gap.short)
    # The longs gained exactly what the shorts lost, because the sides are equally sized.
    assert gap.equity == pytest.approx(before.equity, rel=1e-9)
    assert result.equities[-1] == pytest.approx(settings.capital, rel=1e-9)



def test_the_ranking_longs_the_top_scores_and_shorts_the_bottom():
    settings = BasketSettings(legs=2, max_weight=0.5)

    weights = _target_weights({"aaa": 0.30, "bbb": 0.10, "ccc": -0.10, "ddd": -0.30}, settings)

    assert weights == {"aaa": 0.25, "bbb": 0.25, "ccc": -0.25, "ddd": -0.25}
    assert sum(weights.values()) == pytest.approx(0.0)
    assert sum(abs(value) for value in weights.values()) == pytest.approx(settings.gross)


def test_ties_are_broken_by_name_so_a_replay_holds_the_same_book():
    settings = BasketSettings(legs=1, max_weight=0.5)
    scores = {name: 0.0 for name in ("ddd", "aaa", "ccc", "bbb")}

    first = _target_weights(scores, settings)
    second = _target_weights(dict(reversed(list(scores.items()))), settings)

    assert first == second == {"aaa": 0.5, "ddd": -0.5}


def test_a_thin_cross_section_shrinks_the_book_instead_of_concentrating_it():
    settings = BasketSettings(legs=10, max_weight=0.05, gross=1.0)
    scores = {f"s{index}": float(index) for index in range(6)}

    weights = _target_weights(scores, settings)

    # Three a side because three is all the cross-section has, each at the 5 % ceiling rather
    # than at the 16.7 % that half the gross over three names would have needed.
    assert len(weights) == 6
    assert all(abs(value) == pytest.approx(0.05) for value in weights.values())
    assert sum(abs(value) for value in weights.values()) == pytest.approx(0.30)


def test_a_cross_section_of_one_name_is_no_book_at_all():
    assert _target_weights({"only": 0.5}, BasketSettings(legs=5)) == {}
    assert _target_weights({}, BasketSettings()) == {}
    # Two names *are* a cross-section for a one-a-side book: one long, one short.
    assert _target_weights({"a": 1.0, "b": 0.0}, BasketSettings(legs=1)) == {
        "a": 0.05,
        "b": -0.05,
    }


def reversal_market():
    """Six quiet names and two that move before the fill and back after it."""
    paths = flat([f"s{index:02d}" for index in range(6)], 14)
    paths["s00"] = [100.0, 103.0, 106.0, 109.0, 109.0, 100.0] + [100.0] * 8
    paths["s01"] = [100.0, 97.0, 94.0, 91.0, 91.0, 100.0] + [100.0] * 8
    return paths


def reversal_settings(**overrides):
    base = dict(
        lookback=3,
        legs=1,
        rebalance_bars=1,
        fee=0.0,
        slippage=0.0,
        latency_bars=1,
        min_names=4,
        max_weight=0.5,
    )
    base.update(overrides)
    return BasketSettings(**base)


def test_a_reversal_book_buys_the_fallen_name_and_fills_at_the_open():
    panel = panel_of(reversal_market())

    result = book(panel, reversal_settings(sign="reversal"))

    # The book shorted s00 near 109 and bought s01 near 91 at the open, then both closed at
    # 100: filling at the close instead of the open would have earned exactly nothing here.
    assert result.equities[-1] > reversal_settings().capital * 1.05
    assert result.marks[0].names == 2


def test_the_momentum_sign_is_the_same_book_held_the_other_way_round():
    panel = panel_of(reversal_market())

    reversal = book(panel, reversal_settings(sign="reversal")).equities[-1]
    momentum = book(panel, reversal_settings(sign="momentum")).equities[-1]

    assert reversal > reversal_settings().capital > momentum
    # Not exactly symmetric: the winning book is sized off a larger equity on every rebalance,
    # so it holds a little more than the losing one held. The sign and the scale are the claim.
    assert (reversal - reversal_settings().capital) == pytest.approx(
        reversal_settings().capital - momentum, rel=0.25
    )


def test_a_bar_cannot_reach_back_into_the_book_that_was_built_before_it():
    settings = reversal_settings(lookback=3, legs=3, rebalance_bars=1)
    paths = walk(11, NAMES, 60)

    full = book(panel_of(paths), settings)
    truncated = book(panel_of({name: values[:41] for name, values in paths.items()}), settings)

    # Every mark the short panel can possibly know is identical: nothing after bar 40 reached
    # bar 40, so there is no look-ahead to find.
    assert [mark.as_row() for mark in full.marks[: len(truncated.marks)]] == [
        mark.as_row() for mark in truncated.marks
    ]
    assert len(full.marks) == len(truncated.marks) + 19


def test_a_name_the_venue_stops_quoting_is_closed_at_its_last_close():
    paths = walk(3, NAMES, 80)
    paths["s00"] = paths["s00"][:50]  # the venue publishes nothing for s00 from bar 50 on
    settings = BasketSettings(lookback=3, legs=3, rebalance_bars=5, min_names=4, max_weight=0.5)

    result = book(panel_of(paths), settings)
    leaving = result.marks[50 - (settings.lookback + settings.latency_bars)]

    assert leaving.traded > 0.0 and leaving.fees > 0.0
    position = result.marks.index(leaving)
    assert result.marks[position - 1].names == 6
    assert leaving.names == 5
    assert any("s00" in line and "closed at its last close" in line for line in result.warnings)
    # The gap is a hole in the panel as well as a fact in the book.
    assert "s00" in result.panel.missing(result.panel.ats[50])
    assert all(mark.equity > 0.0 for mark in result.marks)


def test_a_weekly_book_trades_on_one_bar_in_seven_and_marks_every_bar():
    settings = BasketSettings(
        lookback=3, legs=3, rebalance_bars=7, min_names=4, max_weight=0.5
    )

    result = book(panel_of(walk(5, NAMES, 60)), settings)
    traded = [index for index, mark in enumerate(result.marks) if mark.traded > 0.0]

    assert traded[0] == 0
    assert all(step == 7 for step in (b - a for a, b in zip(traded, traded[1:])))
    assert result.rebalances == len(traded)
    assert len(result.marks) == 60 - 4


def test_the_fee_is_the_declared_rate_on_the_notional_that_changed_hands():
    settings = BasketSettings(lookback=3, legs=3, rebalance_bars=5, min_names=4, max_weight=0.5)

    result = book(panel_of(walk(9, NAMES, 70)), settings)

    assert result.traded > 0.0
    assert result.fees == pytest.approx(result.traded * FEE_RATE)
    assert result.turnover() == pytest.approx(result.traded / (
        sum(abs(mark.equity) for mark in result.marks) / len(result.marks)
    ))


def test_slippage_is_paid_against_the_book_on_both_sides():
    panel = panel_of(flat([f"s{index:02d}" for index in range(10)], 30))
    free = BasketSettings(lookback=2, fee=0.0, slippage=0.0, rebalance_bars=7)
    charged = BasketSettings(lookback=2, fee=0.0, slippage=FILL_SLIPPAGE, rebalance_bars=7)

    without = book(panel, free).equities[-1]
    with_slip = book(panel, charged).equities[-1]

    # Nothing moved, so the whole difference is what the book gave up crossing the spread.
    assert without == pytest.approx(free.capital)
    assert with_slip < without
    assert (without - with_slip) / without < 0.05


def test_a_book_that_cannot_fill_once_says_so_instead_of_printing_a_zero():
    panel = panel_of(flat(["aaa", "bbb", "ccc", "ddd"], 5))

    with pytest.raises(PanelError) as failed:
        book(panel, BasketSettings(lookback=7, latency_bars=1))

    assert "freeze a longer panel" in str(failed.value)


def test_a_panel_with_no_calendar_is_not_a_book():
    with pytest.raises(PanelError) as failed:
        book(Panel(), BasketSettings())

    assert "no calendar" in str(failed.value)


def test_the_settings_that_would_make_the_test_a_lookahead_are_refused():
    with pytest.raises(ValueError) as failed:
        BasketSettings(latency_bars=0)
    assert "lookahead" in str(failed.value)

    with pytest.raises(ValueError):
        BasketSettings(min_names=2)
    with pytest.raises(ValueError):
        BasketSettings(gross=3.0)
    with pytest.raises(ValueError):
        BasketSettings(lookback=0)
    with pytest.raises(ValueError):
        BasketSettings(sign="reversion")
    with pytest.raises(ValueError):
        BasketSettings(fee=0.5)


def test_the_config_name_is_the_hypothesis_in_one_word():
    settings = BasketSettings(lookback=14, sign="momentum", rebalance_bars=7)

    assert settings.config == "momentum-l14-r7"
    assert settings.as_dict()["config"] == settings.config
    assert settings.as_dict()["rebalance_bars"] == 7




def crafted(equities):
    """A finished run with a hand-written equity curve — a reading test needs exact returns."""
    panel = panel_of(flat(["aaa", "bbb", "ccc", "ddd"], len(equities)))
    marks = [
        Mark(
            at=float(panel.ats[index]),
            equity=float(value),
            cash=0.0,
            long=0.0,
            short=0.0,
            names=0,
            traded=0.0,
            fees=0.0,
        )
        for index, value in enumerate(equities)
    ]
    return BasketResult(settings=BasketSettings(), panel=panel, marks=marks)


def test_a_reading_is_the_book_s_own_returns_and_counts_the_fee_once():
    settings = BasketSettings(lookback=3, legs=3, rebalance_bars=5, min_names=4, max_weight=0.5)
    result = book(panel_of(walk(9, NAMES, 70)), settings)

    window = read(result)

    assert window.days == len(result.marks) - 1
    assert window.fees == pytest.approx(sum(mark.fees for mark in result.marks))
    assert window.fees == pytest.approx(result.fees)
    assert window.turnover == pytest.approx(result.turnover())
    assert window.names == pytest.approx(
        sum(mark.names for mark in result.marks) / len(result.marks)
    )
    assert window.roi == pytest.approx(result.equities[-1] / result.equities[0] - 1.0)
    assert 0.0 <= window.hit_rate <= 1.0
    assert window.sigma_bps >= 0.0
    assert window.as_dict()["days"] == window.days


def test_a_window_with_one_mark_has_a_sentence_rather_than_a_number():
    result = book(panel_of(walk(4, NAMES, 40)), BasketSettings(lookback=3, min_names=4))

    window = read(result, start=0, end=1)

    assert window.days == 0
    assert window.measurable is False
    assert window.min_edge_bps == float("inf")
    assert "fewer than two marks" in window.failures[0]
    assert window.passed is False


def test_a_mean_inside_its_own_noise_band_is_named_unresolvable():
    equities = [1000.0]
    for index in range(1, 40):
        equities.append(equities[-1] * (1.0 + (0.02 if index % 2 else -0.0195)))

    window = read(crafted(equities))

    assert window.mean_bps > 0.0
    assert window.min_edge_bps > window.mean_bps
    assert window.measurable is False
    assert any("unresolvable" in line for line in window.failures)
    assert "raise the count" in " ".join(window.failures)


def test_a_clear_mean_on_a_narrow_spread_is_measurable_and_passes():
    equities = [1000.0]
    for index in range(1, 60):
        noise = 0.0001 if index % 2 else -0.0001
        equities.append(equities[-1] * (1.0 + 0.0005 + noise))

    window = read(crafted(equities))

    assert window.mean_bps == pytest.approx(5.0, abs=0.2)
    assert window.min_edge_bps < window.mean_bps
    assert window.measurable is True
    assert window.passed is True, window.failures
    assert window.max_drawdown < 0.01


def test_the_benchmark_is_the_equal_weight_universe_and_a_book_that_holds_nothing_is_flat():
    paths = {"aaa": [100.0, 100.0, 200.0]}
    for name in ("bbb", "ccc", "ddd"):
        paths[name] = [100.0, 100.0, 100.0]
    panel = panel_of(paths)

    assert panel.returns_at(panel.ats[2], 1)["aaa"] == pytest.approx(1.0)
    # An equity curve that never moves has no beta against a universe that did.
    window = read(crafted([1000.0, 1000.0, 1000.0, 1000.0]))
    assert window.beta == pytest.approx(0.0)
    assert window.correlation == pytest.approx(0.0)


def test_the_marks_round_trip_to_a_csv(tmp_path):
    result = book(panel_of(walk(6, NAMES, 40)), BasketSettings(lookback=3, min_names=4))

    path = write_marks(tmp_path / "marks.csv", result)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(result.marks)
    assert list(rows[0]) == list(Mark(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0).as_row())
    assert float(rows[0]["equity"]) == pytest.approx(result.marks[0].equity, abs=1e-5)


def test_the_grid_is_the_declared_grid_in_a_fixed_order():
    declared = configs()

    assert len(declared) == 2 * 4 * 2
    assert len({settings.config for settings in declared}) == len(declared)
    assert declared[0].config == "reversal-l3-r1"
    assert declared[1].config == "reversal-l3-r7"
    assert declared[-1].config == "momentum-l30-r7"
    assert all(settings.legs == 10 for settings in declared)


def test_a_config_cannot_be_built_from_the_grid_that_the_book_would_refuse():
    with pytest.raises(ValueError):
        configs(signs=["reversion"])
    with pytest.raises(ValueError):
        configs(lookbacks=[0])


def test_an_edge_that_stops_before_the_split_does_not_pass():
    paths = walk(21, NAMES, 200, plant=0.5, look=3, until=120)

    readings = grid(
        panel_of(paths), lookbacks=[3], rebalances=[1], signs=["reversal"], split=0.6
    )
    reading = readings[0]

    assert reading.inside.mean_bps > 0.0
    assert reading.outside.mean_bps <= 0.0
    assert reading.passed is False
    assert any("did not persist" in line for line in reading.failures)


def test_a_tail_too_short_to_confirm_anything_says_so():
    readings = grid(
        panel_of(walk(2, NAMES, 40)), lookbacks=[3], rebalances=[7], signs=["reversal"]
    )

    reading = readings[0]

    assert reading.outside.days < 30
    assert any("too few to confirm" in line for line in reading.failures)


def test_the_planted_reversal_is_found_and_the_null_panel_is_refused():
    planted = grid(
        panel_of(walk(31, NAMES, 300, plant=0.5, look=3, vol=0.01)),
        lookbacks=[3],
        rebalances=[1],
        signs=["reversal"],
    )
    null = grid(
        panel_of(walk(31, NAMES, 300, vol=0.01)),
        lookbacks=[3],
        rebalances=[1],
        signs=["reversal"],
    )

    assert planted[0].full.mean_bps > 0.0
    assert planted[0].full.t_stat > 2.0
    assert planted[0].passed is True, planted[0].failures
    assert null[0].full.mean_bps < planted[0].full.mean_bps
    assert null[0].passed is False
    assert any("edge" in line for line in null[0].failures)


def test_an_effect_that_lives_one_bar_is_not_reachable_by_a_weekly_book():
    paths = walk(31, NAMES, 300, plant=0.5, look=3, vol=0.01)
    panel = panel_of(paths)

    daily = grid(panel, lookbacks=[3], rebalances=[1], signs=["reversal"])[0]
    weekly = grid(panel, lookbacks=[3], rebalances=[7], signs=["reversal"])[0]

    # The planted effect fades within one bar, so a book that re-ranks weekly is holding last
    # week's answer: same panel, same signal, no edge — and one seventh of the turnover.
    assert daily.passed is True, daily.failures
    assert weekly.full.mean_bps < 2.0
    assert weekly.passed is False
    assert weekly.full.turnover < daily.full.turnover


def window_of(*, mean_bps=0.0, failures=()):
    """A reading with hand-written numbers, for the sentences that read them."""
    return Window(
        days=60,
        mean_bps=float(mean_bps),
        t_stat=3.0,
        sharpe=1.5,
        max_drawdown=0.05,
        hit_rate=0.55,
        beta=0.01,
        correlation=0.02,
        roi=0.05,
        sigma_bps=50.0,
        min_edge_bps=3.0,
        measurable=True,
        fees=1.0,
        turnover=2.0,
        names=20.0,
        failures=tuple(failures),
    )


def reading_of(config, *, passed, failures=()):
    window = window_of(mean_bps=12.0, failures=() if passed else failures)
    return BasketReading(
        config=config,
        settings=BasketSettings(),
        full=window,
        inside=window,
        outside=window,
        failures=() if passed else tuple(failures),
    )


def test_the_verdict_counts_the_survivors_and_the_tries():
    nothing = verdict_of([])
    assert "nothing to say" in nothing

    refused = verdict_of(
        [reading_of("reversal-l3-r1", passed=False, failures=("t +1.00 < 2.0",))]
    )
    assert refused.startswith("0 of 1")
    assert "change the question rather than the threshold" in refused

    survivors = verdict_of(
        [
            reading_of("reversal-l3-r1", passed=True),
            reading_of("reversal-l3-r7", passed=True),
            reading_of("momentum-l3-r1", passed=False, failures=("t +1.00 < 2.0",)),
        ]
    )
    assert survivors.startswith("2 of 3")
    assert "reversal-l3-r1" in survivors and "momentum-l3-r1" not in survivors
    assert "hypothesis" in survivors and "not a book to fund" in survivors


def test_the_table_prints_the_whole_grid_and_a_verdict_column():
    readings = grid(
        panel_of(walk(8, NAMES, 90)),
        lookbacks=[3, 7],
        rebalances=[7],
        signs=["reversal", "momentum"],
    )

    table = format_table(readings)
    lines = table.splitlines()

    assert len(lines) == len(readings) + 2
    assert lines[0].split() == " ".join(HEADER).split()
    assert lines[1].startswith("-")
    assert "verdict" in HEADER
    assert all(len(line.split()) == len(HEADER) for line in lines[2:])
    assert {line.split()[-1] for line in lines[2:]} <= {"pass", "fail"}
    assert lines[2].split()[0] == readings[0].config


def test_the_grid_refuses_a_split_that_is_not_a_fraction():
    panel = panel_of(walk(12, NAMES, 60))

    with pytest.raises(ValueError):
        grid(panel, lookbacks=[3], rebalances=[7], signs=["reversal"], split=1.0)



# ── the command line ────────────────────────────────────────────────────────

ONE_CONFIG = [
    "--lookback",
    "3",
    "--rebalance",
    "7",
    "--sign",
    "reversal",
    "--legs",
    "3",
    "--min-names",
    "4",
]


def test_the_cli_writes_the_table_the_json_and_the_criteria_and_replays_identically(
    tmp_path, capsys
):
    panel = panel_of(walk(41, NAMES, 120))
    write_panel(panel, tmp_path / "candles")

    first = main(
        ["--candles", str(tmp_path / "candles"), "--out", str(tmp_path / "a")]
        + ONE_CONFIG
        + ["--freeze", str(tmp_path / "frozen")]
    )
    second = main(
        ["--candles", str(tmp_path / "candles"), "--out", str(tmp_path / "b"), "--quiet"]
        + ONE_CONFIG
    )

    assert (first, second) == (0, 0)
    table = (tmp_path / "a" / "crosssection.csv").read_text(encoding="utf-8")
    assert table == (tmp_path / "b" / "crosssection.csv").read_text(encoding="utf-8")
    rows = table.splitlines()
    assert len(rows) == 2  # the header and the one declared config, and nothing else
    assert rows[0].startswith("config,sign,lookback,rebalance_bars")
    assert rows[1].startswith("reversal-l3-r7")

    document = json.loads((tmp_path / "a" / "crosssection.json").read_text(encoding="utf-8"))
    assert verify(document) == ""
    assert verify(json.loads((tmp_path / "a" / "criteria.json").read_text(encoding="utf-8"))) == ""
    assert "of 1 declared config" in document["run"]["verdict"]
    assert document["run"]["grid"]["rebalances"] == [7]
    assert document["run"]["panel"]["symbols"] == len(NAMES)
    assert document["run"]["panel"]["bars"] == 120
    # The printed table is the whole declared grid, verdict column included.
    printed = capsys.readouterr().out
    assert "verdict" in printed
    assert "reversal-l3-r7" in printed
    assert "── crosssection · 10 symbols · 1day · 120 bars" in printed

    frozen = load_panel(tmp_path / "frozen")
    assert frozen.symbols == panel.symbols
    assert len(frozen) == len(panel)


def test_the_cli_means_its_exit_codes(tmp_path):
    write_panel(panel_of(flat(NAMES, 5)), tmp_path / "short")
    write_panel(panel_of(flat(NAMES, 40)), tmp_path / "fine")

    # No universe at all is the caller's mistake.
    assert main(["--out", str(tmp_path / "x")]) == EXIT_USAGE
    # A directory that is not there is a feed that could not be read.
    assert main(["--candles", str(tmp_path / "nowhere"), "--out", str(tmp_path / "x")]) == EXIT_FEED
    # A panel shorter than the book's own warm-up is neither of those: it is too short.
    assert (
        main(
            [
                "--candles",
                str(tmp_path / "short"),
                "--lookback",
                "7",
                "--min-names",
                "4",
                "--out",
                str(tmp_path / "x"),
                "--quiet",
            ]
        )
        == EXIT_SHORT
    )
    # A split that is not a fraction, and settings the book would refuse, are both usage.
    assert (
        main(
            ["--candles", str(tmp_path / "fine"), "--split", "1.5", "--out", str(tmp_path / "x")]
            + ONE_CONFIG
            + ["--quiet"]
        )
        == EXIT_USAGE
    )
    assert (
        main(["--symbols", "aaa,bbb", "--lookback", "0", "--out", str(tmp_path / "x")])
        == EXIT_USAGE
    )


def test_the_cli_reads_a_universe_from_the_symbols_flag(tmp_path, monkeypatch):
    """``--symbols`` goes through the panel fetcher; here the feed is a stand-in, not a venue."""
    from agent.backtest import crosssection as module

    calls = []

    def fake_fetch(symbols, **kwargs):
        calls.append(list(symbols))
        series = [chart(name, walk(51, [name], 60)[name]) for name in symbols]
        return build_panel(series, period="1day"), []

    monkeypatch.setattr(module, "fetch_panel", fake_fetch)

    code = main(
        ["--symbols", "AAA,bbb,ccc,ddd,eee,fff", "--out", str(tmp_path), "--quiet"] + ONE_CONFIG
    )

    assert code == 0
    # The universe is normalised where it enters, so the feed is asked for lower-case pairs
    # in the order the operator wrote them.
    assert calls == [["aaa", "bbb", "ccc", "ddd", "eee", "fff"]]

