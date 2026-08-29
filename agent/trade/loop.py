"""Closed economic loop (Phase 3) — signals → recommendation → policy → action → reward.

The loop connects the layers that already exist:

1. **Signals** — ``agent.signals`` produces the generation forecast and the
   market price (USD/kWh).
2. **Recommender** — ``axis_core.ai.recommender`` ranks ``SELL / STORE / BUY /
   HOLD`` with confidence (a signal, never an action by itself — C-4).
3. **Policy Engine** — ``axis_core.policy.engine.PolicyEngine.evaluate_trade``
   gates the action: trading right, volume limits, DAO approval (ADR-0003).
4. **Reward** — an allowed SELL/BUY produces simulated revenue; a successful
   trade feeds ERS, and every decision is recorded as *experience* for future
   learning (GLOBAL_AI_ARCHITECTURE §7: the model learns economics).

Requires the ``axis-core`` reference implementation on ``PYTHONPATH``
(``pip install -e ../Axis-core``).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from axis_core.ai.recommender import recommend
from axis_core.policy.engine import PolicyEngine
from axis_core.policy.models import PolicyRegistry, ProducerState

from agent.signals import Signal, collect_all

#: ERS gain for an executed trade (soft rise toward 1).
ERS_TRADE_GAIN = 0.02
#: Expected price growth factor fed to the recommender's STORE check.
PRICE_RISE_FACTOR = 1.02


@dataclass
class TradeStep:
    """One loop iteration: decision + simulated result."""

    round_no: int
    action: str
    volume_wh: float
    confidence: float
    price_now: float
    allowed: bool
    reason: str
    revenue_usd: float = 0.0
    ers: float = 0.0
    storage_wh: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_no,
            "action": self.action,
            "volume_wh": round(self.volume_wh, 2),
            "confidence": round(self.confidence, 3),
            "price_now": round(self.price_now, 4),
            "allowed": self.allowed,
            "reason": self.reason,
            "revenue_usd": round(self.revenue_usd, 4),
            "ers": round(self.ers, 4),
            "storage_wh": round(self.storage_wh, 2),
        }


@dataclass
class TradeRun:
    """Result of a multi-round closed-loop simulation."""

    steps: List[TradeStep] = field(default_factory=list)
    ers: float = 0.5
    storage_wh: float = 0.0
    total_revenue_usd: float = 0.0
    experience: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ers": round(self.ers, 4),
            "storage_wh": round(self.storage_wh, 2),
            "total_revenue_usd": round(self.total_revenue_usd, 4),
            "steps": [s.to_dict() for s in self.steps],
            "experience_count": len(self.experience),
        }


def _market_price(signals: List[Signal]) -> float:
    for s in signals:
        if s.kind == "market_forecast" and s.meta.get("market_source") == "dayahead":
            return float(s.value)
    return 0.12


def _forecast_next(signals: List[Signal]) -> float:
    for s in signals:
        if s.kind == "generation_forecast":
            return float(s.value)
    return 0.0


def simulate_rounds(
    n_rounds: int = 4,
    *,
    capacity_wh: float = 1000.0,
    storage_init_wh: float = 0.0,
    ers_init: float = 0.5,
    trade_enabled: bool = True,
    trade_max_volume_wh: int = 0,
    trade_requires_dao: bool = False,
    dao_approved: bool = False,
    source: str = "offline",
) -> TradeRun:
    """Run the closed loop for ``n_rounds`` and return the trade history."""
    run = TradeRun(ers=ers_init, storage_wh=storage_init_wh)
    policy = PolicyRegistry(
        enforce_trade_limits=trade_max_volume_wh > 0,
        trade_max_volume_wh=trade_max_volume_wh,
        trade_requires_dao=trade_requires_dao,
    )
    producer = ProducerState(device_id="gateway_00", trade_enabled=trade_enabled)

    for round_no in range(1, n_rounds + 1):
        bundle = collect_all(source=source, horizon_steps=1, market_steps=1)
        signals = bundle.signals
        price_now = _market_price(signals)

        recommendation = recommend(
            forecast_wh=_forecast_next(signals),
            storage_wh=run.storage_wh,
            capacity_wh=capacity_wh,
            price_now=price_now,
            price_forecast=price_now * PRICE_RISE_FACTOR,
            avg_price=price_now,
        )

        decision = PolicyEngine.evaluate_trade(
            policy=policy,
            recommendation=recommendation,
            producer=producer,
            dao_approved=dao_approved,
        )

        allowed = bool(decision.allowed)
        volume = float(recommendation.volume_wh) if allowed else 0.0
        revenue = 0.0
        if allowed and recommendation.action == "SELL":
            revenue = (volume / 1000.0) * price_now  # Wh → kWh × USD/kWh
        if allowed and recommendation.action == "BUY":
            revenue = -((volume / 1000.0) * price_now)

        ers_delta = 0.0
        if allowed and recommendation.action in ("SELL", "BUY"):
            ers_delta = ERS_TRADE_GAIN * (1.0 - run.ers)
            run.ers = round(run.ers + ers_delta, 4)

        # Storage bookkeeping: generation fills, SELL drains, BUY tops up.
        if recommendation.action == "BUY" and allowed:
            run.storage_wh += volume
        if recommendation.action == "SELL" and allowed:
            run.storage_wh = max(0.0, run.storage_wh - volume)
        run.storage_wh = min(capacity_wh, run.storage_wh + _forecast_next(signals))
        run.storage_wh = max(0.0, run.storage_wh)

        run.total_revenue_usd += revenue
        run.experience.append(
            {
                "round": round_no,
                "action": recommendation.action,
                "allowed": allowed,
                "reason": decision.reason,
                "volume_wh": round(volume, 2),
                "price_now": round(price_now, 4),
                "revenue_usd": round(revenue, 4),
            }
        )
        run.steps.append(
            TradeStep(
                round_no=round_no,
                action=recommendation.action,
                volume_wh=volume,
                confidence=float(recommendation.confidence),
                price_now=price_now,
                allowed=allowed,
                reason=decision.reason,
                revenue_usd=revenue,
                ers=run.ers,
                storage_wh=run.storage_wh,
            )
        )

    return run


def print_run(run: TradeRun) -> None:
    print("Closed economic loop (Phase 3) — signals → policy → action → reward")
    print("=" * 86)
    header = (
        f"{'#':>3} {'action':<7} {'volume Wh':>10} {'conf':>6} {'price':>7} "
        f"{'allowed':>7} {'reason':<22} {'revenue $':>10} {'ers':>6}"
    )
    print(header)
    print("-" * len(header))
    for step in run.steps:
        print(
            f"{step.round_no:>3} {step.action:<7} {step.volume_wh:>10.1f} "
            f"{step.confidence:>6.2f} {step.price_now:>7.3f} "
            f"{str(step.allowed):>7} {step.reason:<22} {step.revenue_usd:>10.4f} "
            f"{step.ers:>6.3f}"
        )
    print("-" * len(header))
    print(
        f"total revenue ${run.total_revenue_usd:.4f} · ers {run.ers:.3f} · "
        f"storage {run.storage_wh:.1f} Wh · experience {len(run.experience)} events"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--capacity-wh", type=float, default=1000.0)
    parser.add_argument("--storage-init-wh", type=float, default=0.0)
    parser.add_argument("--ers-init", type=float, default=0.5)
    parser.add_argument("--max-volume-wh", type=int, default=0)
    parser.add_argument("--dao-gated", action="store_true")
    parser.add_argument("--source", choices=["offline", "online"], default="offline")
    args = parser.parse_args(argv)

    run = simulate_rounds(
        n_rounds=args.rounds,
        capacity_wh=args.capacity_wh,
        storage_init_wh=args.storage_init_wh,
        ers_init=args.ers_init,
        trade_max_volume_wh=args.max_volume_wh,
        trade_requires_dao=args.dao_gated,
        source=args.source,
    )
    print_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


