"""CLI — live energy forecast from the oracle proof stream.

Usage:
    python -m agent.forecast                          # live oracle, 15-min buckets, 8 steps
    python -m agent.forecast --bucket-minutes 10 --horizon 12
    python -m agent.forecast --source offline --horizon 6 --output /tmp/forecast.json
    python -m agent.forecast --csv out.csv            # same CSV contract as the TimesFM skill

The ``offline`` source is a deterministic daily-cycle generator used for
tests and for running without network access.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from agent.forecast.energy import (
    Proof,
    ProofSeries,
    aggregate_proofs,
    fetch_oracle_proofs,
    synthetic_solar_series,
)
from agent.forecast.model import ForecastResult, forecast_energy


def build_series(
    bucket_minutes: int,
    source: str,
    limit: int,
) -> ProofSeries:
    if source == "offline":
        return synthetic_solar_series(bucket_minutes=bucket_minutes, n_buckets=24)
    proofs: List[Proof] = fetch_oracle_proofs(limit=limit)
    if not proofs:
        raise RuntimeError("oracle returned no proofs")
    return aggregate_proofs(proofs, bucket_minutes=bucket_minutes)


def print_table(series: ProofSeries, forecast: ForecastResult) -> None:
    print(f"source            : {forecast.source}")
    print(f"bucket            : {forecast.bucket_minutes} min")
    print(f"observed buckets  : {len(series.values)} ({len(series.values) * series.bucket_minutes} min)")
    print(f"total observed    : {series.total_wh:.1f} Wh")
    m = forecast.meta
    print(
        f"model             : {m['model']} (alpha={m['alpha']}, beta={m['beta']}, "
        f"rmse={m['residual_rmse_wh']} Wh)"
    )
    print(f"interval          : {m['interval']} (q10–q90)")
    print(f"forecast horizon  : {len(forecast.point_wh)} x {forecast.bucket_minutes} min")
    print()
    header = f"{'bucket start (UTC)':<20} {'Wh':>7} {'point':>8} {'q10':>8} {'q90':>8}"
    print(header)
    print("-" * len(header))
    # Observed tail (last 3 buckets) for context.
    for label, value in zip(series.labels[-3:], series.values[-3:]):
        print(f"{label:<20} {value:>7.1f} {'—':>8} {'—':>8} {'—':>8}")
    print(f"{'— forecast —':<20} {'':>7} {'':>8} {'':>8} {'':>8}")
    for label, point, lo, hi in zip(
        forecast.labels, forecast.point_wh, forecast.low_wh, forecast.high_wh
    ):
        print(f"{label:<20} {'':>7} {point:>8.1f} {lo:>8.1f} {hi:>8.1f}")


def write_csv(forecast: ForecastResult, path: str) -> None:
    """Write the forecast in the same CSV contract as the TimesFM skill."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "forecast_wh", "q10_wh", "q90_wh"])
        for label, point, lo, hi in zip(
            forecast.labels, forecast.point_wh, forecast.low_wh, forecast.high_wh
        ):
            writer.writerow([label, f"{point:.3f}", f"{lo:.3f}", f"{hi:.3f}"])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket-minutes", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--interval", choices=["p80", "p95"], default="p80")
    parser.add_argument("--source", choices=["oracle", "offline"], default="oracle")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None, help="JSON file")
    parser.add_argument("--csv", type=str, default=None, help="CSV file (TimesFM contract)")
    args = parser.parse_args(argv)

    try:
        series = build_series(args.bucket_minutes, args.source, args.limit)
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = forecast_energy(series, horizon_steps=args.horizon, interval=args.interval)
    print_table(series, result)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)
        print(f"\nwrote JSON → {args.output}")
    if args.csv:
        write_csv(result, args.csv)
        print(f"wrote CSV  → {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
