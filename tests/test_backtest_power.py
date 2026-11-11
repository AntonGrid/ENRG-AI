"""The resolution of a trading claim: what this many trades can name, and what they cannot.

The module under test does not decide whether a strategy works; it decides whether the
book's own record could answer that question at all. So the tests are about the arithmetic
of that answer: the power of a zero edge is the declared risk and nothing else, the
resolution converges to ``(z + z) * sigma / sqrt(n)`` when the count grows, two trades
resolve almost nothing, and the inversion ("how many trades would it take") is exact rather
than approximate.

The Monte-Carlo test is the one that keeps the formula honest: it draws returns from a known
distribution, runs the same t test the gate runs, and checks that the measured rejection
rate agrees with :func:`power_at` — so a change to the noncentral t cannot pass unnoticed.

At the bottom the walk-forward is asked the only question that matters commercially: does a
report say "unresolvable" instead of "fail" when the book was too small to see its own gate.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from agent.backtest.power import (
    DECLARED_ALPHA,
    DECLARED_POWER,
    Resolution,
    ResolutionError,
    minimum_edge_bps,
    power_at,
    resolution,
    spread_bps,
    trades_needed,
)
from agent.backtest.walkforward import GATE, Reading, Summary, format_summary, summary

#: The sweep's own book on 2026-11-11: 19 closed trades at 217.3 bps of spread, which is the
#: measurement this module was written after.
SWEEP_TRADES = 19
SWEEP_SIGMA = 217.3


def test_the_power_of_a_zero_edge_is_the_declared_risk():
    assert power_at(0.0, SWEEP_TRADES, SWEEP_SIGMA) == pytest.approx(DECLARED_ALPHA, abs=1e-9)
    assert power_at(0.0, 500, 40.0, alpha=0.01) == pytest.approx(0.01, abs=1e-9)


def test_the_power_rises_with_the_edge_and_stops_at_one():
    seen = [power_at(edge, 20, 300.0) for edge in (0.0, 50.0, 150.0, 400.0, 4000.0)]
    assert all(later > earlier for earlier, later in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(1.0, abs=1e-6)


def test_the_limit_of_the_resolution_is_the_sum_of_the_two_normal_quantiles():
    # z(0.975) + z(0.9) = 1.959964 + 1.281552; the noncentral t must converge to it, and
    # to nothing else — this is the anchor that catches a wrong tail or a wrong sign.
    trades, sigma = 200_000, 1000.0
    heard = minimum_edge_bps(trades, sigma) * math.sqrt(trades) / sigma
    assert heard == pytest.approx(3.2415, abs=2e-3)


def test_the_normal_approximation_is_optimistic_at_a_backtest_sized_book():
    # 19 trades at 217 bps: 161.6 bps claimed by the table formula, 171.0 bps in fact.
    exact = minimum_edge_bps(SWEEP_TRADES, SWEEP_SIGMA)
    approximation = 3.2415 * SWEEP_SIGMA / math.sqrt(SWEEP_TRADES)
    assert exact == pytest.approx(171.0, abs=0.1)
    assert exact > approximation
    assert (exact - approximation) / exact < 0.07


def test_two_trades_resolve_almost_nothing():
    # The t with one degree of freedom is so wide that a two-trade book needs a twenty-sigma
    # edge before the power reaches nine in ten: said as a number, not as a shrug.
    heard = minimum_edge_bps(2, 100.0)
    assert heard > 10 * 100.0
    assert math.isfinite(heard)


def test_a_perfectly_repeatable_book_resolves_any_edge():
    assert minimum_edge_bps(10, 0.0) == 0.0
    assert power_at(0.0, 10, 0.0) == pytest.approx(DECLARED_ALPHA)
    assert power_at(1.0, 10, 0.0) == 1.0
    assert resolution(10, 0.0, 30.0).measurable is True


def test_the_resolution_falls_with_the_square_root_of_the_count():
    # Four times the count halves the resolution — but only from above: the t is wider than
    # the normal at small degrees of freedom, so a small book gets less than the square root
    # promises, never more. That direction is the point of the test.
    assert minimum_edge_bps(25, 200.0) / minimum_edge_bps(100, 200.0) == pytest.approx(
        2.0, abs=0.07
    )
    assert minimum_edge_bps(100, 200.0) / minimum_edge_bps(400, 200.0) == pytest.approx(
        2.0, abs=0.02
    )
    assert minimum_edge_bps(25, 200.0) > minimum_edge_bps(100, 200.0) > minimum_edge_bps(
        400, 200.0
    )


def test_trades_needed_inverts_the_resolution_exactly():
    edge, sigma = 30.0, SWEEP_SIGMA
    needed = trades_needed(edge, sigma)
    assert needed == pytest.approx(554.0)
    assert minimum_edge_bps(int(needed), sigma) <= edge
    assert minimum_edge_bps(int(needed) - 1, sigma) > edge


def test_spread_bps_is_the_sample_deviation():
    assert spread_bps([0.01, -0.01]) == pytest.approx(1e4 * 0.02 / math.sqrt(2), abs=1e-6)
    assert spread_bps([0.01]) == 0.0
    assert spread_bps([]) == 0.0


def test_one_trade_is_unmeasured_rather_than_disproved():
    heard = resolution(1, 0.0, 30.0)
    assert heard.trades == 1
    assert math.isinf(heard.min_edge_bps)
    assert heard.measurable is False
    assert "unmeasured rather than disproved" in heard.describe()


def test_the_measurable_flag_is_the_gate_against_the_resolution():
    assert resolution(SWEEP_TRADES, SWEEP_SIGMA, 30.0).measurable is False
    assert resolution(SWEEP_TRADES, SWEEP_SIGMA, 200.0).measurable is True
    assert resolution(600, SWEEP_SIGMA, 30.0).measurable is True


def test_needed_trades_is_what_the_gate_would_take():
    heard = resolution(SWEEP_TRADES, SWEEP_SIGMA, 30.0)
    assert heard.needed_trades == pytest.approx(554.0)
    assert resolution(600, SWEEP_SIGMA, 30.0).needed_trades == pytest.approx(600.0)
    assert math.isinf(resolution(SWEEP_TRADES, SWEEP_SIGMA, 0.0).needed_trades)


def test_the_sentence_names_the_count_the_spread_and_the_gate():
    heard = resolution(SWEEP_TRADES, SWEEP_SIGMA, 30.0)
    assert heard.describe() == (
        "19 trades at 217 bps of spread resolve 171 bps at alpha 0.05 / power 90%, which is "
        "outside the gate's 30 bps: 554 trades would be needed, so this many cannot say yes "
        "and cannot say no"
    )
    inside = resolution(SWEEP_TRADES, SWEEP_SIGMA, 200.0).describe()
    assert inside.endswith("the gate's 200 bps is inside that")


def test_the_dict_is_json_safe_where_the_dataclass_is_infinite():
    payload = resolution(1, 0.0, 30.0).as_dict()
    assert payload["min_edge_bps"] is None
    assert payload["needed_trades"] is None
    assert payload["measurable"] is False
    finite = resolution(SWEEP_TRADES, SWEEP_SIGMA, 30.0).as_dict()
    assert finite["min_edge_bps"] == pytest.approx(171.0, abs=0.1)
    assert finite["needed_trades"] == 554
    assert finite["alpha"] == pytest.approx(DECLARED_ALPHA)
    assert finite["power"] == pytest.approx(DECLARED_POWER)


def test_refusals_are_sentences_not_tracebacks():
    with pytest.raises(ResolutionError, match="no dispersion to read"):
        minimum_edge_bps(1, 100.0)
    with pytest.raises(ResolutionError, match="not a spread"):
        minimum_edge_bps(10, -1.0)
    with pytest.raises(ResolutionError, match="not a risk"):
        minimum_edge_bps(10, 100.0, alpha=0.0)
    with pytest.raises(ResolutionError, match="not a chance"):
        minimum_edge_bps(10, 100.0, power=1.0)
    with pytest.raises(ResolutionError, match="not an edge"):
        trades_needed(0.0, 100.0)
    with pytest.raises(ResolutionError, match="not a spread"):
        trades_needed(30.0, math.inf)
    with pytest.raises(ResolutionError, match="not a spread"):
        power_at(30.0, 10, -2.0)


def test_the_formula_agrees_with_a_simulation_of_the_test_it_models():
    # The one check that cannot drift: draw returns from a known mean and spread, run the
    # gate's own two-sided t, and count how often it fires. If the noncentral t is read
    # wrong — a tail, a sign, a degree of freedom — this number moves and the suite says so.
    from scipy import stats

    trades, sigma_bps = 20, 100.0
    edge_bps = minimum_edge_bps(trades, sigma_bps)
    expected = power_at(edge_bps, trades, sigma_bps)
    rng = np.random.default_rng(20261111)
    draws = 4000
    returns = rng.normal(edge_bps / 1e4, sigma_bps / 1e4, size=(draws, trades))
    means = returns.mean(axis=1)
    spreads = returns.std(axis=1, ddof=1)
    t_stats = means / (spreads / math.sqrt(trades))
    critical = float(stats.t.ppf(1.0 - DECLARED_ALPHA / 2.0, trades - 1))
    observed = float(np.mean(np.abs(t_stats) > critical))
    assert expected == pytest.approx(DECLARED_POWER, abs=0.01)
    assert observed == pytest.approx(expected, abs=0.03)


def _reading(returns, ablation="all", fold="2026", sharpe=0.0, profit_factor=1.0):
    """One fold's book, with the aggregates a summary would have written beside the returns."""
    values = tuple(float(value) for value in returns)
    mean = sum(values) / len(values) if values else 0.0
    return Reading(
        ablation=ablation,
        fold=fold,
        train_samples=100,
        trades=len(values),
        roi=mean,
        buy_hold=0.0,
        edge=mean,
        win_rate=0.5,
        profit_factor=float(profit_factor),
        sharpe=float(sharpe),
        t_stat=0.0,
        returns=values,
    )


def test_a_book_too_small_to_see_the_gate_is_reported_as_unresolvable():
    row = summary([_reading([0.01, -0.05, 0.03])])[0]
    assert row.resolvable is False
    assert any(failure.startswith("unresolvable:") for failure in row.failures)
    assert "3 trades at" in row.resolution.describe()
    assert "1633 bps" in row.resolution.describe()
    assert row.passed is False
    assert row.as_dict()["resolution"]["measurable"] is False


def test_a_book_wide_enough_to_see_the_gate_is_not_called_unresolvable():
    # 19 trades of +200 bps with 10 bps of spread: the gate is inside what they can name,
    # so whatever else fails, the resolution does not.
    row = summary([_reading([0.019, 0.021] * 9 + [0.020])])[0]
    assert row.trades == 19
    assert row.sigma_bps == pytest.approx(10.0, abs=0.5)
    assert row.resolvable is True
    assert not any("unresolvable" in failure for failure in row.failures)
    assert row.as_dict()["resolution"]["measurable"] is True


def test_the_printed_table_carries_the_resolution_beside_the_edge():
    table = format_summary(summary([_reading([0.01, -0.05, 0.03])]))
    assert "± bps" in table
    assert "unresolvable" in table
    assert f"{GATE['edge_bps']:.0f}" in table


def test_a_one_trade_fold_is_summarised_without_an_exception():
    row = summary([_reading([0.02])])[0]
    assert math.isinf(row.min_edge_bps)
    assert row.resolvable is False
    assert "no dispersion" in row.resolution.describe()
    assert "inf" in format_summary([row])


def test_a_small_book_is_called_unmeasured_rather_than_failed():
    row = summary([_reading([0.01, -0.05, 0.03])])[0]
    assert row.passed is False
    assert row.verdict == "unmeasured"
    assert row.as_dict()["verdict"] == "unmeasured"
    assert "unmeasured" in format_summary([row])


def test_a_book_that_can_see_the_gate_and_misses_it_is_called_a_failure():
    row = summary([_reading([0.0009, 0.0011] * 9 + [0.0010])])[0]
    assert row.resolvable is True
    assert row.verdict == "fail"
    assert any(failure.startswith("edge ") for failure in row.failures)


def test_a_wide_book_with_a_real_edge_passes_all_five_checks():
    # +200 bps a trade on 10 bps of spread, nineteen times: big enough to resolve the gate,
    # positive, consistent, and never losing — the one case where the third verdict is "pass".
    row = summary([_reading([0.019, 0.021] * 9 + [0.020], sharpe=1.5, profit_factor=2.0)])[0]
    assert row.failures == ()
    assert row.verdict == "pass"
    assert row.as_dict()["verdict"] == "pass"
