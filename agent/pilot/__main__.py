"""CLI for the closed-loop DePIN pilot.

Examples:
    python -m agent.pilot.sim --devices 50 --hours 168
    python -m agent.pilot.sim --devices 10 --hours 48 --capacity 2000 --peak 1000
"""
from __future__ import annotations

import argparse
import json

from agent.pilot.sim import PilotConfig, run_pilot


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="pilot")
    parser.add_argument("--devices", type=int, default=50)
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capacity", type=float, default=2000.0)
    parser.add_argument("--peak", type=float, default=1000.0)
    args = parser.parse_args(argv)

    config = PilotConfig(
        n_devices=args.devices,
        hours=args.hours,
        seed=args.seed,
        capacity_wh=args.capacity,
        peak_production_wh=args.peak,
    )
    result = run_pilot(config)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
