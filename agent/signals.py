"""Hybrid AI oracle — a constitutional signal layer (ENRG-AI).

Per the constitution (ADR-0003 / GLOBAL_AI_ARCHITECTURE C-1), the AI oracle is
a **source of signals, not decisions**. This module produces structured,
uncertainty-aware observations across the ecosystem:

- ``generation_forecast`` — energy output forecast from the proof stream
  (Wh per bucket, 80% prediction interval), built on ``agent.forecast``;
- ``generation_anomaly`` — last-observed point outside the model's MAD band
  (device behaving unlike its own history);
- ``market_forecast`` — USD/kWh price forecast (dayahead/p2p/spot) with
  80% intervals, built on ``agent.market``.

Everything is deterministic in ``offline`` mode (tests, demos, CI) and uses
live oracle data in ``online`` mode. Signals are plain data: any Policy Engine
(or a human) decides what to do with them.

CLI:
    python -m agent.signals --source offline --horizon 8
    python -m agent.signals --source online --horizon 6 --output signals.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional

import numpy as np

from agent.forecast.energy import (
    ProofSeries,
    aggregate_proofs,
    fetch_oracle_proofs,
    forward_fill,
    synthetic_solar_series,
)
from agent.forecast.model import HoltTrend, forecast_energy
from agent.market.feeds import fetch_offline
from agent.market.model import forecast_price_with_intervals, price_matrix

#: MAD z-threshold for the generation anomaly signal.
ANOMALY_Z = 3.0
#: Market energy sources we forecast (macro excluded — different unit).
MARKET_SOURCES = ("dayahead", "p2p", "spot")


@dataclass
class Signal:
    """One structured observation from the AI oracle."""

    kind: str  # generation_forecast | generation_anomaly | market_forecast
    domain: str
    ts: str  # ISO UTC
    source: str  # oracle | offline | market-offline
    value: float
    interval_low: float = 0.0
    interval_high: float = 0.0
    unit: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "domain": self.domain,
            "ts": self.ts,
            "source": self.source,
            "value": round(float(self.value), 6),
            "interval_low": round(float(self.interval_low), 6),
            "interval_high": round(float(self.interval_high), 6),
            "unit": self.unit,
            "meta": self.meta,
        }


@dataclass
class SignalBundle:
    """A timestamped collection of signals (one oracle "heartbeat")."""

    signals: List[Signal] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    meta: Dict[str, Any] = field(default_factory=dict)

    def by_kind(self, kind: str) -> List[Signal]:
        return [s for s in self.signals if s.kind == kind]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "meta": self.meta,
            "signals": [s.to_dict() for s in self.signals],
        }


def _anomaly_signal(series: ProofSeries, model: HoltTrend, z: float) -> Optional[Signal]:
    """MAD-based flag if the last observed point deviates from its own trend."""
    y = np.asarray(series.values, dtype=float)
    residuals = model.in_sample_residuals(y)
    if residuals.size < 3:
        return None
    mad = median([abs(float(v)) for v in residuals])
    scale = 1.4826 * mad if mad > 0 else 1.4826 * 1e-6
    last_res = float(residuals[-1])
    observed = float(y[-1])
    expected = observed - last_res
    if abs(last_res) <= z * scale:
        return None
    return Signal(
        kind="generation_anomaly",
        domain="pilot",
        ts=series.labels[-1],
        source=series.source,
        value=last_res,
        unit="Wh",
        meta={
            "observed_wh": round(observed, 3),
            "expected_wh": round(expected, 3),
            "threshold_wh": round(z * scale, 3),
            "residual_wh": round(last_res, 3),
            "z": z,
        },
    )


def generation_signals(
    series: ProofSeries,
    horizon_steps: int = 8,
    interval: str = "p80",
    anomaly_z: float = ANOMALY_Z,
) -> List[Signal]:
    """Forecast + anomaly signals for an energy proof series."""
    result = forecast_energy(series, horizon_steps=horizon_steps, interval=interval)
    signals: List[Signal] = [
        Signal(
            kind="generation_forecast",
            domain="pilot",
            ts=label,
            source=series.source,
            value=float(point),
            interval_low=float(lo),
            interval_high=float(hi),
            unit="Wh",
            meta={"bucket_minutes": series.bucket_minutes, "step_ahead": i + 1},
        )
        for i, (label, point, lo, hi) in enumerate(
            zip(result.labels, result.point_wh, result.low_wh, result.high_wh)
        )
    ]
    model = HoltTrend().fit(np.asarray(series.values, dtype=float))
    anomaly = _anomaly_signal(series, model, anomaly_z)
    if anomaly is not None:
        signals.append(anomaly)
    return signals


def market_signals(points: List[Dict[str, Any]], steps: int = 4) -> List[Signal]:
    """Price forecast signals (one per market source) from a price series."""
    matrix, columns = price_matrix(points)
    if matrix.shape[0] < 2:
        return []
    forecasts = forecast_price_with_intervals(matrix, steps=max(int(steps), 1))
    signals: List[Signal] = []
    for d, col in enumerate(columns):
        if col not in MARKET_SOURCES:
            continue
        spec = forecasts.get(d, {})
        points_path = spec.get("point", [])
        low_path = spec.get("low", [])
        high_path = spec.get("high", [])
        if not points_path:
            continue
        signals.append(
            Signal(
                kind="market_forecast",
                domain="market",
                ts=points[-1]["ts"],
                source="market-offline",
                value=float(points_path[0]),
                interval_low=float(low_path[0]),
                interval_high=float(high_path[0]),
                unit="usd_per_kwh",
                meta={
                    "market_source": col,
                    "path": [
                        {
                            "step": i + 1,
                            "point": round(float(p), 6),
                            "low": round(float(lo), 6),
                            "high": round(float(hi), 6),
                        }
                        for i, (p, lo, hi) in enumerate(
                            zip(points_path, low_path, high_path)
                        )
                    ],
                },
            )
        )
    return signals


def _offline_market_points(points_n: int = 24) -> List[Dict[str, Any]]:
    """Deterministic hourly market series (distinct timestamps per step).

    ``fetch_offline(step)`` prices depend on ``step``; the timestamps are
    synthesised here so ``price_matrix`` sees a real chronological grid.
    """
    points: List[Dict[str, Any]] = []
    now = dt.datetime.now(dt.timezone.utc)
    for step in range(points_n):
        feeds = fetch_offline(step=step)
        prices = {
            f.source: float(f.prices["usd_per_kwh"])
            for f in feeds
            if f.source in MARKET_SOURCES and "usd_per_kwh" in f.prices
        }
        ts = (now - dt.timedelta(hours=(points_n - 1 - step))).isoformat(
            timespec="minutes"
        )
        points.append({"ts": ts, "prices": prices})
    return points


def collect_all(
    source: str = "offline",
    bucket_minutes: int = 15,
    horizon_steps: int = 8,
    market_steps: int = 4,
    limit: int = 1000,
    anomaly_z: float = ANOMALY_Z,
) -> SignalBundle:
    """Collect a full AI-oracle heartbeat (generation + market signals).

    ``source="online"`` prefers the live oracle; if it is unreachable (cold
    Render instance, network) the bundle degrades to the deterministic solar
    series and records ``source="offline-fallback"`` — the AI oracle never
    comes back empty-handed, mirroring the landing's demo fallback.
    """
    actual_source = source
    if source == "online":
        try:
            proofs = fetch_oracle_proofs(limit=limit)
            if proofs:
                series = forward_fill(
                    aggregate_proofs(proofs, bucket_minutes)
                )
                actual_source = "oracle"
            else:
                series = synthetic_solar_series(bucket_minutes, n_buckets=24)
                actual_source = "offline-fallback"
        except Exception:  # noqa: BLE001 — degrade gracefully, never block
            series = synthetic_solar_series(bucket_minutes, n_buckets=24)
            actual_source = "offline-fallback"
    else:
        series = synthetic_solar_series(bucket_minutes, n_buckets=24)

    signals = generation_signals(
        series,
        horizon_steps=horizon_steps,
        interval="p80",
        anomaly_z=anomaly_z,
    )
    signals.extend(
        market_signals(_offline_market_points(points_n=24), steps=market_steps)
    )

    meta: Dict[str, Any] = {
        "source": actual_source,
        "requested_source": source,
        "bucket_minutes": bucket_minutes,
        "horizon_steps": horizon_steps,
        "generation_model": "holt-linear-trend",
        "generation_unit": "Wh",
        "market_unit": "usd_per_kwh",
        "anomaly_z": anomaly_z,
        "observed_generation_wh": round(series.total_wh, 3),
        "observed_buckets": int(len(series.values)),
    }
    return SignalBundle(signals=signals, meta=meta)


def print_bundle(bundle: SignalBundle) -> None:
    m = bundle.meta
    print(f"generated_at      : {bundle.generated_at}")
    print(f"source            : {m['source']}")
    print(
        f"generation        : {m['observed_generation_wh']} Wh over "
        f"{m['observed_buckets']} x {m['bucket_minutes']}-min buckets"
    )
    print(f"signal count      : {len(bundle.signals)}")
    print()
    print(
        f"{'kind':<22} {'time':<20} {'value':>10} {'low':>10} {'high':>10} {'unit'}"
    )
    print("-" * 78)
    for s in bundle.signals:
        if s.kind == "market_forecast":
            label = f"market_forecast[{s.meta.get('market_source')}]"
        else:
            label = s.kind
        print(
            f"{label:<22} {s.ts:<20} {s.value:>10.3f} "
            f"{s.interval_low:>10.3f} {s.interval_high:>10.3f} {s.unit}"
        )


def sign_bundle(
    bundle: SignalBundle,
    secret_key_b64: str,
) -> Dict[str, Any]:
    """Sign a SignalBundle with an Ed25519 key (constitution C-3).

    Returns ``{"message": <bundle dict>, "signature": <b64>}``. The signature
    covers the canonical JSON of the bundle (same canonicalization as
    ``agent.fed.protocol``), so a Policy Engine / aggregator can verify that a
    signal set really came from this node and was not tampered with.
    """
    import base64

    import nacl.signing

    from agent.fed.protocol import canonical_json_bytes

    message = canonical_json_bytes(bundle.to_dict())
    secret = nacl.signing.SigningKey(base64.b64decode(secret_key_b64))
    signature = secret.sign(message).signature
    return {
        "message": bundle.to_dict(),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def verify_bundle_signature(
    public_key_b64: str,
    signed: Dict[str, Any],
) -> bool:
    """Verify an Ed25519-signed SignalBundle. Returns False (never raises)."""
    import base64

    import nacl.exceptions
    import nacl.signing

    from agent.fed.protocol import canonical_json_bytes

    signature_b64 = signed.get("signature")
    message = signed.get("message")
    if not isinstance(signature_b64, str) or not isinstance(message, dict):
        return False
    try:
        pk_bytes = base64.b64decode(public_key_b64, validate=True)
        sig_bytes = base64.b64decode(signature_b64, validate=True)
    except ValueError:
        return False
    if len(pk_bytes) != 32 or len(sig_bytes) != 64:
        return False
    try:
        verifier = nacl.signing.VerifyKey(pk_bytes)
        verifier.verify(canonical_json_bytes(message), sig_bytes)
        return True
    except (ValueError, nacl.exceptions.BadSignatureError):
        return False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["offline", "online"], default="offline")
    parser.add_argument("--bucket-minutes", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--market-steps", type=int, default=4)
    parser.add_argument("--anomaly-z", type=float, default=ANOMALY_Z)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None, help="JSON file")
    args = parser.parse_args(argv)

    try:
        bundle = collect_all(
            source=args.source,
            bucket_minutes=args.bucket_minutes,
            horizon_steps=args.horizon,
            market_steps=args.market_steps,
            limit=args.limit,
            anomaly_z=args.anomaly_z,
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_bundle(bundle)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(bundle.to_dict(), fh, ensure_ascii=False, indent=2)
        print(f"\nwrote JSON → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

