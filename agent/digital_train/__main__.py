"""CLI entry point for the digital self-training loop.

Examples:
    python -m agent.digital_train.pipeline --once --offline --points 48
    python -m agent.digital_train.pipeline --loop --interval 3600
"""
from __future__ import annotations

import argparse
import json
import time

from agent.digital_train.pipeline import run_once


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="digital_train")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    parser.add_argument("--loop", action="store_true", help="run in the background forever")
    parser.add_argument("--interval", type=int, default=3600, help="seconds between loops")
    parser.add_argument("--offline", action="store_true", help="use deterministic synthetic feeds")
    parser.add_argument("--points", type=int, default=48, help="series points per offline cycle")
    parser.add_argument("--state", default="digital_state.json", help="state file path")
    parser.add_argument("--feeds", nargs="*", help="feed domains (default: all)")
    parser.add_argument("--secret-key", default=None, help="Base64 Ed25519 seed for the contribution")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.once:
        result = run_once(
            state_path=args.state,
            feeds=args.feeds,
            offline=args.offline,
            points=args.points,
            secret_key=args.secret_key,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    if args.loop:
        while True:
            result = run_once(
                state_path=args.state,
                feeds=args.feeds,
                offline=args.offline,
                # offline: retrain on a fresh deterministic series each cycle;
                # online: collect one real point into the persistent history.
                points=args.points if args.offline else 1,
                secret_key=args.secret_key,
            )
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] trained={result['trained']}")
            time.sleep(args.interval)
        return

    print("Nothing to do: pass --once or --loop (see --help).")


if __name__ == "__main__":
    main()
