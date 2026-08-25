"""Closed-loop DePIN pilot simulation (GLOBAL_AI_ARCHITECTURE step C).

Simulates N devices over H hours and closes the full loop:

    device (physical production + storage)
        → model (seasonal production forecast, retrained in-loop)
        → market price (``agent.market``-style cycle)
        → Recommender (``axis_core.ai.recommend``)
        → Policy Engine (``evaluate_trade``)
        → executed action → economic reward (USD)
        → reward feeds ERS + the forecast model for the next round

The AI strategy is compared against a "blind" baseline (no action) to show
that recommendations create measurable economic value.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

from axis_core.ai import recommend
from axis_core.policy.engine import PolicyEngine
from axis_core.policy.models import ProducerState

#: Storage fill ratio the Recommender treats as "low" (matches defaults).
MIN_STORAGE_RATIO = 0.2


@dataclass
class DeviceAgent:
    """One virtual device in the pilot."""

    device_id: str
    capacity_wh: float = 10_000.0
    storage_wh: float = 0.0
    ers: float = 0.5
    trade_enabled: bool = True
    dao_approved: bool = True


@dataclass
class PilotConfig:
    """Pilot scenario parameters (all deterministic via ``seed``)."""

    n_devices: int = 50
    hours: int = 168
    seed: int = 7
    capacity_wh: float = 2_000.0
    peak_production_wh: float = 1_000.0  # solar noon peak per hour
    price_base: float = 0.10  # USD/kWh
    price_amplitude: float = 0.05  # daily cycle amplitude
    retrain_every: int = 24  # hours between forecast-model retrains
    #: Overrides for ``axis_core.ai.recommend`` (used by A/B experiments).
    recommend_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionRecord:
    """One executed/gated action in the pilot."""

    hour: int
    device_id: str
    production_wh: float
    price: float
    action: str
    volume_wh: float
    reward_usd: float
    decision_reason: str
    storage_after_wh: float
    forecast_wh: float


@dataclass
class PilotResult:
    """Final metrics of one pilot run."""

    config: PilotConfig
    net_reward_usd: float
    revenue_usd: float
    cost_usd: float
    n_actions: int
    n_sells: int
    n_buys: int
    n_denied: int
    baseline_reward_usd: float
    records: List[ActionRecord] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "devices": self.config.n_devices,
            "hours": self.config.hours,
            "net_reward_usd": round(self.net_reward_usd, 2),
            "revenue_usd": round(self.revenue_usd, 2),
            "cost_usd": round(self.cost_usd, 2),
            "baseline_reward_usd": round(self.baseline_reward_usd, 2),
            "ai_vs_baseline_usd": round(self.net_reward_usd - self.baseline_reward_usd, 2),
            "n_actions": self.n_actions,
            "n_sells": self.n_sells,
            "n_buys": self.n_buys,
            "n_denied": self.n_denied,
        }


def solar_production(hour: int, peak: float, rng: random.Random) -> float:
    """Deterministic solar curve: zero at night, peak at solar noon."""
    daylight = 6 <= hour <= 18
    if not daylight:
        return 0.0
    wave = math.sin(math.pi * (hour - 6) / 12.0)  # 0 at 6h/18h, 1 at 12h
    noise = rng.uniform(0.9, 1.1)
    return max(0.0, peak * wave * noise)


def market_price(hour: int, config: PilotConfig) -> float:
    """Deterministic daily price cycle: cheapest mid-morning, peak at 19:00.

    The cosine peaks at 19:00 (evening demand) and bottoms at 07:00 — so
    overnight the price sits below its rolling average and the Recommender's
    BUY signal fires, while daytime production is sold near the peak.
    """
    wave = math.cos(2 * math.pi * (hour - 19) / 24.0)
    return max(0.01, config.price_base + config.price_amplitude * wave)


class SeasonalForecast:
    """In-loop production forecast: per-hour average of the previous days.

    Retrained on accumulated history every ``retrain_every`` hours — this is
    the "reward → learning" step: the model keeps improving with data.
    """

    def __init__(self) -> None:
        self._by_hour: Dict[int, List[float]] = {}

    def add(self, hour: int, production_wh: float) -> None:
        self._by_hour.setdefault(hour % 24, []).append(production_wh)

    def predict(self, hour: int) -> float:
        history = self._by_hour.get(hour % 24, [])
        if not history:
            return 0.0
        return sum(history) / len(history)


def _run_strategy(config: PilotConfig, use_ai: bool) -> PilotResult:
    """Run the pilot with (use_ai=True) or without (False) AI recommendations."""
    rng = random.Random(config.seed)
    devices = [
        DeviceAgent(device_id=f"dev_{i:03d}", capacity_wh=config.capacity_wh)
        for i in range(config.n_devices)
    ]
    forecast = SeasonalForecast()
    records: List[ActionRecord] = []

    revenue = 0.0
    cost = 0.0
    n_actions = 0
    n_sells = 0
    n_buys = 0
    n_denied = 0

    for hour in range(config.hours):
        price_now = market_price(hour, config)
        price_next = market_price(hour + 1, config)
        avg_price = sum(market_price(h, config) for h in range(max(0, hour - 23), hour + 1)) / min(24, hour + 1)

        for device in devices:
            production = solar_production(hour, config.peak_production_wh, rng)
            forecast_wh = forecast.predict(hour + 1)

            if use_ai:
                rec = recommend(
                    forecast_wh=forecast_wh,
                    storage_wh=device.storage_wh,
                    capacity_wh=device.capacity_wh,
                    price_now=price_now,
                    price_forecast=price_next,
                    avg_price=avg_price,
                    **config.recommend_params,
                )
                decision = PolicyEngine.evaluate_trade(
                    policy=None,
                    recommendation=rec,
                    producer=ProducerState(device_id=device.device_id, trade_enabled=device.trade_enabled),
                    dao_approved=device.dao_approved,
                )
                action = rec.action
                volume = rec.volume_wh
                reason = decision.reason
                allowed = decision.allowed
            else:
                # Blind baseline: do nothing, just absorb (lose) the energy.
                action = "HOLD"
                volume = 0.0
                reason = "baseline"
                allowed = True

            reward_usd = 0.0
            if allowed and action == "SELL" and volume > 0:
                sell_wh = min(volume, device.storage_wh + production)
                device.storage_wh = max(0.0, device.storage_wh + production - sell_wh)
                reward_usd = sell_wh / 1000.0 * price_now  # USD (Wh → kWh)
                revenue += reward_usd
                n_sells += 1
                n_actions += 1
            elif allowed and action == "BUY" and volume > 0:
                buy_wh = min(volume, device.capacity_wh - device.storage_wh)
                device.storage_wh += buy_wh
                reward_usd = -(buy_wh / 1000.0 * price_now)
                cost -= reward_usd  # cost += |reward|
                n_buys += 1
                n_actions += 1
            else:
                # HOLD / STORE / denied: production fills storage, overflow is lost.
                device.storage_wh = min(device.capacity_wh, device.storage_wh + production)
                if not allowed:
                    n_denied += 1

            if use_ai:
                device.ers = min(1.0, device.ers + 0.002 if allowed and action != "HOLD" else device.ers - 0.001)

            records.append(
                ActionRecord(
                    hour=hour,
                    device_id=device.device_id,
                    production_wh=round(production, 2),
                    price=round(price_now, 4),
                    action=action,
                    volume_wh=round(volume, 2),
                    reward_usd=round(reward_usd, 4),
                    decision_reason=reason,
                    storage_after_wh=round(device.storage_wh, 2),
                    forecast_wh=round(forecast_wh, 2),
                )
            )

            # Close the loop: the observed production becomes training data.
            forecast.add(hour, production)

    result = PilotResult(
        config=config,
        net_reward_usd=revenue - cost,
        revenue_usd=revenue,
        cost_usd=cost,
        n_actions=n_actions,
        n_sells=n_sells,
        n_buys=n_buys,
        n_denied=n_denied,
        baseline_reward_usd=0.0,
        records=records,
    )
    return result


def run_pilot(config: PilotConfig | None = None) -> PilotResult:
    """Run the AI-guided pilot and compare it with the blind baseline."""
    config = config or PilotConfig()
    ai = _run_strategy(config, use_ai=True)
    baseline = _run_strategy(config, use_ai=False)
    ai.baseline_reward_usd = baseline.net_reward_usd
    return ai


if __name__ == "__main__":
    result = run_pilot()
    for key, value in result.summary().items():
        print(f"{key:24s}: {value}")

