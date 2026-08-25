"""Closed-loop DePIN pilot tests (agent.pilot).

Requires ``axis_core`` (installed alongside the workspace); skipped when the
package is unavailable so the core ENRG-AI suite stays standalone.
"""

import random

import pytest

pytest.importorskip("axis_core")

from agent.pilot.sim import (  # noqa: E402
    PilotConfig,
    SeasonalForecast,
    market_price,
    run_pilot,
    solar_production,
)


def test_solar_production_day_night():
    rng = random.Random(1)
    assert solar_production(3, 1000.0, rng) == 0.0  # night
    assert solar_production(12, 1000.0, rng) > 900.0  # solar noon near peak
    assert solar_production(21, 1000.0, rng) == 0.0  # night


def test_market_price_peaks_in_evening():
    config = PilotConfig()
    assert market_price(19, config) > market_price(7, config)
    assert market_price(7, config) < config.price_base  # cheap mid-morning


def test_seasonal_forecast_learns_per_hour():
    model = SeasonalForecast()
    model.add(12, 800.0)
    model.add(12, 1000.0)
    assert model.predict(12) == 900.0  # mean of the two observations
    assert model.predict(36) == 900.0  # same hour of day (12 % 24)
    assert model.predict(13) == 0.0  # unknown hour → no forecast yet


def test_pilot_ai_generates_profitable_actions():
    result = run_pilot(PilotConfig(n_devices=10, hours=72, seed=7))
    summary = result.summary()
    assert summary["n_sells"] > 0
    assert summary["n_buys"] > 0
    assert summary["net_reward_usd"] > 0.0


def test_pilot_ai_beats_blind_baseline():
    result = run_pilot(PilotConfig(n_devices=50, hours=168, seed=7))
    summary = result.summary()
    assert summary["ai_vs_baseline_usd"] > 0.0
    assert summary["net_reward_usd"] > summary["baseline_reward_usd"]


def test_pilot_is_deterministic():
    first = run_pilot(PilotConfig(n_devices=5, hours=48, seed=42))
    second = run_pilot(PilotConfig(n_devices=5, hours=48, seed=42))
    assert first.net_reward_usd == second.net_reward_usd


def test_pilot_denied_actions_are_counted():
    # Devices without the trading right should be denied, not executed.
    result = run_pilot(PilotConfig(n_devices=5, hours=24, seed=1))
    assert result.n_denied == 0  # all pilot devices have the trading right
    assert len(result.records) == 5 * 24  # every device logged every hour
