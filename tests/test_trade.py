"""Closed economic loop tests (Phase 3) — recommender + policy gates + reward."""

from axis_core.ai.recommender import recommend
from axis_core.policy.engine import PolicyEngine
from axis_core.policy.models import PolicyRegistry, ProducerState

from agent.trade import simulate_rounds


def test_recommender_overflow_sell():
    rec = recommend(
        forecast_wh=30.0,
        storage_wh=95.0,
        capacity_wh=100.0,
        price_now=0.10,
        price_forecast=0.10,
    )
    assert rec.action == "SELL"
    assert rec.volume_wh == 25.0  # available 125 - capacity 100


def test_policy_gate_not_authorized():
    rec = recommend(
        forecast_wh=1.0,
        storage_wh=0.0,
        capacity_wh=100.0,
        price_now=0.10,
        price_forecast=0.10,
    )
    decision = PolicyEngine.evaluate_trade(
        policy=None,
        recommendation=rec,
        producer=ProducerState(device_id="gw", trade_enabled=False),
    )
    assert decision.allowed is False
    assert decision.reason == "trade_not_authorized"


def test_policy_dao_gate_blocks_without_approval():
    rec = recommend(
        forecast_wh=1.0,
        storage_wh=0.0,
        capacity_wh=100.0,
        price_now=0.10,
        price_forecast=0.10,
    )
    decision = PolicyEngine.evaluate_trade(
        policy=PolicyRegistry(trade_requires_dao=True),
        recommendation=rec,
        producer=ProducerState(device_id="gw", trade_enabled=True),
        dao_approved=False,
    )
    assert decision.allowed is False
    assert decision.reason == "trade_dao_gated"


def test_policy_volume_limit():
    rec = recommend(
        forecast_wh=400.0,
        storage_wh=0.0,
        capacity_wh=1000.0,
        price_now=0.10,
        price_forecast=0.10,
    )
    # No overflow here, but check the limit path with a small cap.
    decision = PolicyEngine.evaluate_trade(
        policy=PolicyRegistry(enforce_trade_limits=True, trade_max_volume_wh=50),
        recommendation=rec,
        producer=ProducerState(device_id="gw", trade_enabled=True),
    )
    # HOLD has volume 0 → within limits → allowed.
    assert decision.allowed is True


def test_loop_simulate_shape():
    run = simulate_rounds(n_rounds=3, capacity_wh=500.0)
    assert len(run.steps) == 3
    assert len(run.experience) == 3
    assert run.total_revenue_usd >= 0.0
    for step in run.steps:
        assert step.reason in {
            "trade_not_authorized",
            "trade_volume_exceeded",
            "trade_dao_gated",
            "trade_allowed",
        }
        assert step.ers >= 0.0


def test_loop_trade_disabled_never_executes():
    run = simulate_rounds(
        n_rounds=2, capacity_wh=100.0, storage_init_wh=90.0, trade_enabled=False
    )
    for step in run.steps:
        assert step.allowed is False
        assert step.reason == "trade_not_authorized"
        assert step.revenue_usd == 0.0


def test_loop_serializable():
    run = simulate_rounds(n_rounds=2)
    import json

    assert json.dumps(run.to_dict())
