"""Configure GitHub Pages + the AI signing secret for this repository.

Automates the two manual steps needed for the live AI-oracle attestation
(Layer 4, Phase 1):

1. Enable **GitHub Pages** served from the ``gh-pages`` branch (created);
2. Create the **AXIS_AI_SIGNING_KEY** Actions secret (Ed25519 seed, base64),
   encrypted with the repo's public key (PyNaCl sealed box).

Requires a GitHub token with ``repo`` scope:

    GITHUB_TOKEN=ghp_xxx .venv/bin/python scripts/setup_pages.py \
        [--key <base64-seed>]

If ``--key`` is omitted, a fresh keypair is generated and printed — save the
secret! The public key is safe to share; the landing widget shows it in the
signature strip.

After both succeed, the cron workflow (``.github/workflows/signals.yml``)
publishes ``docs/ai/assessments.json`` to ``gh-pages`` every 30 minutes, and
the landing widget switches to LIVE (``/ai/assessments.json``).
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from typing import Optional

import httpx
import nacl.bindings
import nacl.public

REPO = "AntonGrid/ENRG-AI"
API = "https://api.github.com"
SECRET_NAME = "AXIS_AI_SIGNING_KEY"


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def enable_pages(client: httpx.Client, token: str) -> None:
    resp = client.put(
        f"{API}/repos/{REPO}/pages",
        headers=_headers(token),
        json={"source": {"branch": "gh-pages", "path": "/"}},
    )
    if resp.status_code in (200, 201, 409):
        print(f"✅ GitHub Pages enabled → https://{REPO.lower().replace('/', '.')}/")
        return
    if resp.status_code == 404:
        print("ℹ️  Pages already configured or repo changed — skipping.")
        return
    raise RuntimeError(f"pages enable failed: {resp.status_code} {resp.text[:200]}")


def set_secret(client: httpx.Client, token: str, secret_value: str) -> None:
    pk_resp = client.get(
        f"{API}/repos/{REPO}/actions/secrets/public-key",
        headers=_headers(token),
    )
    pk_resp.raise_for_status()
    pk = pk_resp.json()
    key_id = pk["key_id"]
    public_key_raw = base64.b64decode(pk["key"])

    # libsodium sealed box: encrypt the secret to the repo's public key.
    sealed = nacl.bindings.crypto_box_seal(
        secret_value.encode("utf-8"),
        public_key_raw,
    )
    encrypted = base64.b64encode(sealed).decode("ascii")

    resp = client.put(
        f"{API}/repos/{REPO}/actions/secrets/{SECRET_NAME}",
        headers=_headers(token),
        json={"encrypted_value": encrypted, "key_id": key_id},
    )
    resp.raise_for_status()
    print(f"✅ Secret {SECRET_NAME} created (encrypted).")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=str, default=None, help="base64 Ed25519 seed")
    args = parser.parse_args(argv)

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print(
            "error: GITHUB_TOKEN required (Personal Access Token, repo scope)",
            file=sys.stderr,
        )
        return 2

    secret = args.key
    if not secret:
        from agent.fed.protocol import generate_keypair

        secret, public_key = generate_keypair()
        print(f"🔑 generated fresh keypair (save the SECRET!):")
        print(f"   secret = {secret}")
        print(f"   public = {public_key}")
    else:
        from agent.fed.protocol import public_key_from_secret

        print(
            f"🔑 using provided key; public = {public_key_from_secret(secret)}"
        )

    with httpx.Client(timeout=20.0) as client:
        enable_pages(client, token)
        set_secret(client, token, secret)

    print("\nNext: run the workflow once manually to publish the first attestation:")
    print("  https://github.com/AntonGrid/ENRG-AI/actions/workflows/signals.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
