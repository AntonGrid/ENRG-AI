"""CLI: read a park and write ``coherence.json``.

    python -m agent.physics --source offline --scenario mixed --output coherence.json
    python -m agent.physics --source online --bucket-minutes 60 --output coherence.json --sign

``--source online`` reads the live oracle (ADR-0010) and degrades to the
deterministic offline park with ``source="offline-fallback"`` when the oracle is
unreachable or reports a single producer — the same never-empty-handed rule as
``agent.signals``.

The JSON is a ``SignalBundle``: the same envelope the oracle already signs, so
``agent.signals.verify_bundle_signature`` verifies it unchanged. Coherence does
not need a new format to be trusted, only a new ``kind``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from agent.physics.park import ParkConfig, observe_park
from agent.physics.series import park_matrix
from agent.physics.signal import coherence_bundle
from agent.physics.synthetic import build_scenario


def _offline(args) -> Tuple[Any, List[str], List[Any], str, List[str]]:
    scenario = build_scenario(
        seed=args.seed,
        scenario=args.scenario,
        bucket_minutes=args.bucket_minutes,
    )
    energy = scenario.energy
    starts = scenario.starts
    if args.window and args.window < len(starts):
        energy = energy[:, -args.window :]
        starts = starts[-args.window :]
    return energy, scenario.device_ids, starts, "offline", []


def _online(args) -> Tuple[Any, List[str], List[Any], str, List[str]]:
    from agent.forecast.energy import fetch_oracle_proofs

    try:
        proofs = fetch_oracle_proofs(limit=args.limit)
        matrix = park_matrix(proofs, args.bucket_minutes)
    except Exception as exc:  # noqa: BLE001 — CLI boundary, never block
        energy, device_ids, starts, _, _ = _offline(args)
        return (
            energy,
            device_ids,
            starts,
            "offline-fallback",
            [f"oracle unavailable ({exc}); read the offline park instead"],
        )
    energy = matrix.energy
    starts = matrix.starts
    if args.window and args.window < len(starts):
        energy = energy[:, -args.window :]
        starts = starts[-args.window :]
    return energy, matrix.device_ids, starts, "oracle", []


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["offline", "online"], default="offline")
    parser.add_argument("--scenario", choices=["mixed", "clean"], default="mixed")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bucket-minutes", type=int, default=60)
    parser.add_argument("--window", type=int, default=0, help="read the last N buckets")
    parser.add_argument("--baseline-hours", type=float, default=96.0)
    parser.add_argument("--band", type=float, default=0.05, help="Pi, rad per hour")
    parser.add_argument("--memory-hours", type=float, default=12.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None, help="JSON file")
    parser.add_argument(
        "--sign",
        action="store_true",
        help="Ed25519-sign the bundle with AXIS_AI_SIGNING_KEY (base64 secret)",
    )
    args = parser.parse_args(argv)

    warnings: List[str] = []
    try:
        if args.source == "online":
            energy, device_ids, starts, source, warnings = _online(args)
        else:
            energy, device_ids, starts, source, warnings = _offline(args)
        config = ParkConfig(
            bucket_minutes=args.bucket_minutes,
            baseline_hours=args.baseline_hours,
            band_rad_per_hour=args.band,
            memory_hours=args.memory_hours,
            threshold=args.threshold,
        )
        observation = observe_park(energy, device_ids, starts, config)
        warnings = list(warnings) + list(observation.warnings)
        bundle = coherence_bundle(observation, source=source)
        bundle.meta["warnings"] = warnings
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.sign:
        from agent.fed.protocol import public_key_from_secret
        from agent.signals import sign_bundle

        secret = os.environ.get("AXIS_AI_SIGNING_KEY", "")
        if not secret:
            print(
                "error: --sign requires AXIS_AI_SIGNING_KEY (base64 Ed25519 seed) in env",
                file=sys.stderr,
            )
            return 2
        payload: Dict[str, Any] = sign_bundle(bundle, secret)
        payload["public_key"] = public_key_from_secret(secret)
    else:
        payload = bundle.to_dict()

    park = bundle.meta["park"]
    print(f"source            : {source}")
    print(
        f"park              : {park['producers']} producers, "
        f"{park['window_buckets']} x {args.bucket_minutes}-min buckets read"
    )
    print(
        f"coherence         : mean {park['coherence_mean']} "
        f"min {park['coherence_min']} silent {park['silent_fraction']}"
    )
    print()
    print(f"{'device':12} {'score':>6}  {'verdict':8} readings")
    print("-" * 78)
    for row in bundle.meta["devices"]:
        verdict = "ANOMALY" if row["is_anomaly"] else "ok"
        reasons = ",".join(row["reasons"]) or "-"
        print(f"{row['device_id']:12} {row['score']:>6.3f}  {verdict:8} {reasons}")
    if warnings:
        print()
        for warning in warnings:
            print(f"warning: {warning}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\nwrote {'signed ' if args.sign else ''}JSON → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
