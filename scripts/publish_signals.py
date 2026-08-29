"""Publish a signed AI-oracle attestation to a static JSON file.

This is the "observability" output of the hybrid AI oracle (ADR-0010 Layer 2):
a periodically generated, Ed25519-signed bundle of signals that any frontend
(landing, PWA) or external verifier can consume as ``assessments.json``.

Usage (local):
    AXIS_AI_SIGNING_KEY=<base64-ed25519-seed> python scripts/publish_signals.py \\
        --source online --output /tmp/ai/assessments.json

Runs from CI on a schedule (see .github/workflows/signals.yml) and publishes
to the repository's GitHub Pages (branch ``gh-pages``).

The output schema is:
    {"message": <SignalBundle dict>, "signature": <b64>, "public_key": <b64>}

Verify with ``agent.signals.verify_bundle_signature(public_key, payload)``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["offline", "online"], default="online")
    parser.add_argument("--bucket-minutes", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--market-steps", type=int, default=4)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=str,
        default="docs/ai/assessments.json",
        help="JSON output path (default: docs/ai/assessments.json)",
    )
    args = parser.parse_args(argv)

    from agent.fed.protocol import public_key_from_secret
    from agent.signals import collect_all, sign_bundle

    secret = os.environ.get("AXIS_AI_SIGNING_KEY", "")
    if not secret:
        print(
            "error: AXIS_AI_SIGNING_KEY (base64 Ed25519 seed) is required",
            file=sys.stderr,
        )
        return 2

    bundle = collect_all(
        source=args.source,
        bucket_minutes=args.bucket_minutes,
        horizon_steps=args.horizon,
        market_steps=args.market_steps,
        limit=args.limit,
    )

    payload = sign_bundle(bundle, secret)
    payload["public_key"] = public_key_from_secret(secret)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    kinds = {}
    for s in bundle.signals:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    print(
        f"published {bundle.meta['source']} bundle → {out} "
        f"(signals={len(bundle.signals)} {kinds}, "
        f"generation={bundle.meta['observed_generation_wh']} Wh)"
    )
    print(f"public_key={payload['public_key']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
