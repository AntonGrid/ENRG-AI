"""The friction: net alpha is what survives the trade, and it is arithmetic.

This is the test of a *cost model*, so it asserts the four things a cost model can get
wrong:

* **gas is a floor, slippage is a slope.** Gas costs the same on a ten-dollar stake as
  on a hundred-thousand-dollar one, so it is priced per transaction and refuses small
  trades outright; slippage scales with the size *relative to the depth*, so it refuses
  big trades in thin pools. Both are pinned to the cent, not to a range.
* **an unmeasured depth is not a bottomless pool.** ``tvl_usd=None`` leaves slippage
  unestimated and says so in ``measured_depth``; a *published* zero costs the whole
  trade. The difference between "deep" and "unmeasured" is the difference between a
  report and a lie, and it is the one thing a table cannot show.
* **a profit is a claim about money.** Both floors apply — :data:`MIN_NET_ALPHA_USD` and
  :data:`MIN_NET_ALPHA_PCT` — and there is a trade that passes each one alone and is
  still refused, because a cent on a million dollars and a hundred percent of a dollar
  are both rounding errors wearing the wrong unit.
* **the book asks before it pays, and pays when it takes.** A refused entry leaves the
  credits in cash and writes ``skip``; an accepted one credits the position
  ``placed − slippage − gas``, so ``equity`` is what survived the trade. The exit is
  never refused and never charged: the risk-reducing move cannot be held hostage by a
  gas estimate, or the kill switch would be the thing that cannot run.
"""
import json
import types

import numpy as np
import pytest

from agent.physics import portfolio as P
from agent.physics.transaction_cost_analyzer import (
    MAX_SLIPPAGE,
    MIN_NET_ALPHA_PCT,
    MIN_NET_ALPHA_USD,
    NO_TRADE,
    STABLE_SLIPPAGE_COEFFICIENT,
    UNPROFITABLE_AFTER_FEES,
    VOLATILE_SLIPPAGE_COEFFICIENT,
    CostModelError,
    CostReport,
    TransactionCostAnalyzer,
)


def venue(**overrides):
    """A model of a quiet Sepolia hour: two gwei, ninety-two thousand units, ETH at 3k."""
    facts = {"eth_price_usd": 3_000.0, "gas_used": 92_000, "gas_price_gwei": 2.0}
    facts.update(overrides)
    return TransactionCostAnalyzer(**facts)


@pytest.fixture()
def model():
    return venue()


def published(name, apy, *, bucket=P.BUCKET_SAFE, depth_usd=None, steps=365):
    """A protocol whose record pays exactly ``apy``, so ``implied_apy`` is that number."""
    per_step = P.per_step_rate(apy)
    return P.Protocol(
        name=name,
        bucket=bucket,
        claim_apy=apy,
        returns=np.full(steps, per_step),
        depth_usd=depth_usd,
    )


def book(*protocols, cash=1_000.0, friction=None):
    return P.Portfolio(
        whitelist=P.WhiteList.of(*protocols), cash=cash, friction=friction
    )


# ── gas: a floor, paid once, per transaction ────────────────────────────────


def test_gas_is_units_times_gwei_times_the_price_of_an_ether(model):
    assert model.gas_cost_wei == 92_000 * 2 * 10 ** 9
    assert model.estimate_gas_cost_usd() == pytest.approx(0.552)


def test_gas_does_not_scale_with_the_size_of_the_trade(model):
    small = model.assess(annual_apy=0.365, trade_size_usd=10.0, tvl_usd=1e6)
    large = model.assess(annual_apy=0.365, trade_size_usd=1e5, tvl_usd=1e6)
    assert small.gas_cost_usd == pytest.approx(large.gas_cost_usd)


def test_a_free_transaction_is_a_price_and_not_an_error():
    assert venue(gas_price_gwei=0.0).estimate_gas_cost_usd() == 0.0


def test_a_model_with_no_price_for_gas_is_not_a_model():
    with pytest.raises(CostModelError):
        venue(eth_price_usd=0.0)
    with pytest.raises(CostModelError):
        venue(gas_used=-1)
    with pytest.raises(CostModelError):
        venue(horizon_days=0.0)
    with pytest.raises(CostModelError):
        venue(minimum_pct=-0.1)


# ── slippage: a slope, priced against the depth that was published ──────────


def test_slippage_is_the_coefficient_times_the_relative_size(model):
    assert model.estimate_slippage(1_000.0, 250_000.0) == pytest.approx(
        STABLE_SLIPPAGE_COEFFICIENT * 1_000.0 / 250_000.0
    )
    assert model.estimate_slippage(2_000.0, 250_000.0) == pytest.approx(
        2.0 * model.estimate_slippage(1_000.0, 250_000.0)
    )
    assert model.estimate_slippage(1_000.0, 125_000.0) == pytest.approx(
        2.0 * model.estimate_slippage(1_000.0, 250_000.0)
    )


def test_a_volatile_pair_costs_more_than_a_stable_one(model):
    stable = model.estimate_slippage(1_000.0, 250_000.0, stable=True)
    volatile = model.estimate_slippage(1_000.0, 250_000.0, stable=False)
    assert volatile > stable
    assert volatile / stable == pytest.approx(
        VOLATILE_SLIPPAGE_COEFFICIENT / STABLE_SLIPPAGE_COEFFICIENT
    )


def test_slippage_is_capped_at_the_whole_trade(model):
    assert model.estimate_slippage(1e9, 100.0) == pytest.approx(MAX_SLIPPAGE)


def test_an_unmeasured_depth_is_not_a_bottomless_pool(model):
    assert model.estimate_slippage(1_000.0, None) is None
    report = model.assess(annual_apy=0.365, trade_size_usd=1_000.0, tvl_usd=None)
    assert report.measured_depth is False
    assert report.slippage_pct == 0.0 and report.slippage_usd == 0.0
    assert report.net_alpha_usd == pytest.approx(report.gross_alpha_usd - 0.552)


def test_a_published_depth_of_zero_costs_the_whole_trade(model):
    report = model.assess(annual_apy=0.365, trade_size_usd=1_000.0, tvl_usd=0.0)
    assert report.measured_depth is True
    assert report.slippage_pct == pytest.approx(MAX_SLIPPAGE)
    assert report.slippage_usd == pytest.approx(1_000.0)
    assert report.profitable is False
    assert report.reason == UNPROFITABLE_AFTER_FEES


# ── the arithmetic that decides ─────────────────────────────────────────────


def test_the_horizon_turns_an_annual_rate_into_what_the_hold_pays(model):
    assert model.horizon_days == 30.0
    assert model.horizon_return(0.365) == pytest.approx(0.03)
    assert venue(horizon_days=365.0).horizon_return(0.0512) == pytest.approx(0.0512)


def test_net_alpha_is_gross_minus_slippage_minus_gas(model):
    report = model.assess(annual_apy=0.365, trade_size_usd=1_000.0, tvl_usd=250_000.0)
    assert report.gross_alpha_pct == pytest.approx(0.03)
    assert report.gross_alpha_usd == pytest.approx(30.0)
    assert report.slippage_usd == pytest.approx(2.0)
    assert report.gas_cost_usd == pytest.approx(0.552)
    assert report.total_cost_usd == pytest.approx(2.552)
    assert report.net_alpha_usd == pytest.approx(30.0 - 2.552)
    assert report.net_alpha_pct == pytest.approx((30.0 - 2.552) / 1_000.0)
    assert report.profitable is True and report.reason == ""
    assert report.cost_share == pytest.approx(2.552 / 1_000.0)


def test_a_profit_must_clear_the_dollar_floor_as_well_as_the_percentage_one(model):
    # A hundred percent of one dollar is a percentage and not a profit.
    tiny = model.calculate_net_alpha(1.0, 1.0)
    assert tiny.net_alpha_usd == pytest.approx(1.0 - 0.552)
    assert tiny.net_alpha_pct > MIN_NET_ALPHA_PCT
    assert tiny.net_alpha_usd < MIN_NET_ALPHA_USD
    assert tiny.profitable is False and tiny.reason == UNPROFITABLE_AFTER_FEES
    # Ten dollars of alpha on ten million dollars is a profit and not a trade: the cost of
    # the risk dwarfs the cost of the gas, and the hurdle rate is why both floors exist.
    huge = model.calculate_net_alpha(1e-6, 1e7, tvl_usd=None)
    assert huge.net_alpha_usd > MIN_NET_ALPHA_USD
    assert huge.net_alpha_pct < MIN_NET_ALPHA_PCT
    assert huge.profitable is False and huge.reason == UNPROFITABLE_AFTER_FEES


def test_a_trade_that_loses_money_is_refused_with_a_reason(model):
    report = model.assess(annual_apy=0.01, trade_size_usd=10.0, tvl_usd=250_000.0)
    assert report.net_alpha_usd < 0.0
    assert report.refused is True
    assert report.reason == UNPROFITABLE_AFTER_FEES


def test_a_trade_of_no_size_is_not_a_refusal(model):
    report = model.calculate_net_alpha(0.03, 0.0, tvl_usd=250_000.0)
    assert report.trade_size_usd == 0.0
    assert report.reason == NO_TRADE
    assert report.profitable is False


def test_the_two_floors_are_the_declared_ones():
    assert MIN_NET_ALPHA_USD == 0.50
    assert MIN_NET_ALPHA_PCT == 0.001
    assert UNPROFITABLE_AFTER_FEES == "UNPROFITABLE_AFTER_FEES"


def test_a_report_is_json_and_a_line(model):
    report = model.assess(annual_apy=0.365, trade_size_usd=1_000.0, tvl_usd=250_000.0)
    assert json.loads(json.dumps(report.as_dict()))["net_alpha_usd"] == pytest.approx(
        round(30.0 - 2.552, 6)
    )
    assert "net +27.4480 $" in report.summary_text()
    assert UNPROFITABLE_AFTER_FEES not in report.summary_text()


def test_a_report_can_be_built_by_hand_and_keeps_the_order_of_the_question():
    empty = CostReport(trade_size_usd=0.0)
    assert empty.refused is True and empty.cost_share == 0.0


# ── the model from a receipt: a fact beats a parameter ──────────────────────


def test_a_receipt_prices_the_next_trade_at_what_the_last_one_cost():
    receipt = {"gasUsed": 120_000, "effectiveGasPrice": 3 * 10 ** 9}
    measured = TransactionCostAnalyzer.from_receipt(receipt, eth_price_usd=2_000.0)
    assert measured.gas_used == 120_000
    assert measured.gas_price_gwei == pytest.approx(3.0)
    assert measured.estimate_gas_cost_usd() == pytest.approx(0.72)


def test_a_receipt_is_read_in_either_shape_and_either_spelling():
    snake = types.SimpleNamespace(gas_used=100_000, effective_gas_price=10 ** 9)
    camel = {"gasUsed": 100_000, "effectiveGasPrice": 10 ** 9}
    first = TransactionCostAnalyzer.from_receipt(snake)
    second = TransactionCostAnalyzer.from_receipt(camel)
    assert (first.gas_used, first.gas_price_gwei) == (
        second.gas_used,
        second.gas_price_gwei,
    )


def test_a_receipt_with_no_gas_is_refused_rather_than_priced_at_zero():
    with pytest.raises(CostModelError):
        TransactionCostAnalyzer.from_receipt(
            {"gasUsed": 0, "effectiveGasPrice": 10 ** 9}
        )
    with pytest.raises(CostModelError):
        TransactionCostAnalyzer.from_receipt({})
    with pytest.raises(CostModelError):
        TransactionCostAnalyzer.from_receipt({"gasUsed": 100_000})


def test_a_measured_model_still_takes_the_operators_policy():
    measured = TransactionCostAnalyzer.from_receipt(
        {"gasUsed": 100_000, "effectiveGasPrice": 10 ** 9},
        eth_price_usd=2_000.0,
        horizon_days=7.0,
    )
    assert measured.horizon_days == 7.0
    assert measured.horizon_return(0.365) == pytest.approx(0.007)


# ── the book: it asks before it pays, and pays when it takes ────────────────


def test_a_book_without_a_model_is_the_book_it_always_was():
    deep = published("Deep", 0.365, depth_usd=1e6)
    wallet = book(deep)
    assert wallet.deposit(deep, 1_000.0) == 1_000.0
    assert wallet.value(deep.address) == 1_000.0
    assert wallet.cash == 0.0 and wallet.equity == 1_000.0
    assert wallet.cost_log == [] and wallet.friction_summary() is None
    assert deep.as_opportunity(0)["turnover"] is None  # depth is not smuggled in as flow


def test_an_entry_that_cannot_pay_its_gas_is_skipped_not_raised():
    deep = published("Deep", 0.365, depth_usd=1e6)
    wallet = book(deep, friction=venue())
    assert wallet.deposit(deep, 1.0) == 0.0  # a dollar of a 36.5 % pool cannot pay 0.55
    assert wallet.cash == 1_000.0 and wallet.equity == 1_000.0
    assert wallet.holdings == {}
    assert [event["action"] for event in wallet.events] == ["skip"]
    assert wallet.events[0]["reason"] == UNPROFITABLE_AFTER_FEES
    assert wallet.cost_log[0]["action"] == "skip"
    assert wallet.friction_summary()["refused"] == 1


def test_an_entry_that_clears_the_model_is_charged_what_it_costs():
    deep = published("Deep", 0.365, depth_usd=1e6)
    wallet = book(deep, friction=venue())
    assert wallet.deposit(deep, 500.0) == 500.0
    cost = wallet.cost_log[0]["total_cost_usd"]
    assert cost > 0.0
    assert wallet.value(deep.address) == pytest.approx(500.0 - cost)
    assert wallet.cash == pytest.approx(500.0)
    assert wallet.equity == pytest.approx(1_000.0 - cost)
    summary = wallet.friction_summary()
    assert summary["entries"] == 1 and summary["paid_usd"] == pytest.approx(cost)
    assert summary["horizon_days"] == 30.0 and summary["eth_price_usd"] == 3_000.0


def test_a_thin_pool_is_refused_and_a_deep_one_is_taken():
    thin = published("Thin", 0.365, bucket=P.BUCKET_EXPERIMENTAL, depth_usd=2_000.0)
    deep = published("Deep", 0.365, depth_usd=1e6)
    wallet = book(thin, deep, friction=venue())
    # A volatile pair: the coefficient is the volatile one, and a quarter of the pool is
    # half the trade before the gas is even counted.
    assert wallet.assess_entry(thin, 500.0).slippage_pct == pytest.approx(
        VOLATILE_SLIPPAGE_COEFFICIENT * 500.0 / 2_000.0
    )
    assert wallet.assess_entry(thin, 500.0).refused is True
    assert wallet.deposit(thin, 500.0) == 0.0
    assert wallet.assess_entry(deep, 500.0).refused is False
    assert wallet.deposit(deep, 500.0) == 500.0


def test_a_review_asks_what_a_trade_would_cost_and_records_nothing():
    deep = published("Deep", 0.365, depth_usd=1e6)
    wallet = book(deep, friction=venue())
    assert wallet.assess_entry(deep, 500.0).profitable is True
    assert wallet.cost_log == [] and wallet.events == []
    assert wallet.cash == 1_000.0


def test_the_cost_is_subtracted_from_the_measured_rate_and_not_the_claim():
    liar = P.Protocol(
        name="Liar",
        bucket=P.BUCKET_SAFE,
        claim_apy=5.0,
        returns=np.full(365, P.per_step_rate(0.04)),
        depth_usd=1e6,
    )
    wallet = book(liar, friction=venue())
    report = wallet.assess_entry(liar, 1_000.0)
    assert report.gross_alpha_pct == pytest.approx(
        wallet.friction.horizon_return(liar.implied_apy)
    )
    assert report.gross_alpha_usd < 5.0  # the claim, priced, would have been 411 $


def test_the_depth_is_read_from_the_published_field_and_not_from_turnover():
    quiet = P.Protocol(
        name="Quiet",
        bucket=P.BUCKET_SAFE,
        claim_apy=0.365,
        returns=np.full(365, P.per_step_rate(0.365)),
        turnover=1e6,
        depth_usd=2_000.0,
    )
    wallet = book(quiet, friction=venue())
    report = wallet.assess_entry(quiet, 500.0)
    assert report.measured_depth is True
    assert report.slippage_pct > 0.0  # thin depth, fat flow: flow does not pay the toll


def test_an_unpublished_depth_is_priced_as_unmeasured():
    hidden = published("Hidden", 0.365, depth_usd=None)
    wallet = book(hidden, friction=venue())
    report = wallet.assess_entry(hidden, 500.0)
    assert report.measured_depth is False and report.slippage_usd == 0.0
    assert wallet.deposit(hidden, 500.0) == 500.0


def test_the_exit_is_neither_refused_nor_charged():
    deep = published("Deep", 0.365, depth_usd=1e6)
    wallet = book(deep, friction=venue())
    wallet.deposit(deep, 500.0)
    held = wallet.value(deep.address)
    raised = wallet.withdraw(deep.address, held)
    assert raised == pytest.approx(held)  # the whole position, not a discounted one
    assert wallet.cash == pytest.approx(1_000.0 - wallet.cost_log[0]["total_cost_usd"])
    assert [row["action"] for row in wallet.cost_log] == ["deposit"]


def test_every_entry_in_the_cost_log_carries_the_report_it_came_from():
    deep = published("Deep", 0.365, depth_usd=1e6)
    wallet = book(deep, friction=venue())
    wallet.deposit(deep, 500.0, step=7)
    row = wallet.cost_log[0]
    for key in ("step", "action", "protocol", "address", "net_alpha_usd", "reason"):
        assert key in row
    assert row["protocol"] == "Deep" and row["step"] == 7


def test_a_friction_object_the_book_cannot_ask_is_a_bug():
    with pytest.raises(P.PortfolioError):
        P.Portfolio(whitelist=P.WhiteList(), cash=100.0, friction=object())


def test_the_empire_hands_its_friction_to_the_wallet():
    model = venue()
    empire = P.Empire.launch(capital=500.0, friction=model)
    assert empire.portfolio.friction is model
    assert P.Empire.launch(capital=500.0).portfolio.friction is None

