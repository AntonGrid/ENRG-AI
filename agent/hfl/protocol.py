"""Hierarchical contribution protocol — ``axis-fed/2``.

Extends ``axis-fed/1`` with hierarchy metadata (``level``, ``domain``,
``region``, ``version``, ``quality``) so contributions can be routed and
aggregated at two levels:

- ``level == 1`` — a gateway contribution (same shape as ``agent.fed``);
- ``level == 2`` — a regional contribution signed by the *region's* key.

All of these fields are covered by the Ed25519 signature
(``agent.fed.protocol.SIGNED_FIELDS``). Signing/verification is shared with
the base protocol, so a level-2 contribution is verified exactly like a
level-1 one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.fed.protocol import public_key_from_secret, sign_contribution, verify_contribution

#: Hierarchical contribution schema (v2).
SCHEMA_V2 = "axis-fed/2"

DEFAULT_VERSION = "model_1.0"


def make_contribution(
    *,
    level: int,
    domain: str,
    region: str,
    device_id: str,
    weights: List[float],
    samples: int,
    loss: float,
    round_no: int,
    version: str = DEFAULT_VERSION,
    quality: Optional[Dict[str, Any]] = None,
    nonce: Optional[int] = None,
) -> Dict[str, Any]:
    """Build an unsigned ``axis-fed/2`` contribution dict."""
    contribution: Dict[str, Any] = {
        "schema": SCHEMA_V2,
        "round": round_no,
        "device_id": device_id,
        "weights": [round(float(w), 6) for w in weights],
        "samples": samples,
        "loss": round(float(loss), 6),
        "nonce": nonce if nonce is not None else round_no,
        "level": level,
        "domain": domain,
        "region": region,
        "version": version,
    }
    if quality is not None:
        contribution["quality"] = quality
    return contribution


def sign_regional(
    secret_key_b64: str,
    contribution: Dict[str, Any],
) -> Dict[str, Any]:
    """Sign a regional (level-2) contribution with the region's key."""
    signed = sign_contribution(secret_key_b64, contribution)
    signed["public_key"] = public_key_from_secret(secret_key_b64)
    return signed


__all__ = [
    "SCHEMA_V2",
    "DEFAULT_VERSION",
    "make_contribution",
    "public_key_from_secret",
    "sign_regional",
    "verify_contribution",
]
