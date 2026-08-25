"""Federated contribution protocol (Phase 2).

Each gateway signs its weight update with the device Ed25519 key — the same
pattern as proof-of-production (ADR-0001: the key never leaves the device;
the contribution is verifiable). The aggregator accepts only signed, valid
contributions; unsigned or tampered ones are rejected.

Canonical message (v1) — the contribution fields *excluding* ``signature``,
serialized with canonical JSON (keys sorted, no whitespace, floats that are
integers normalized), the same canonical form used by the Axis
proof-of-production path:

    {"device_id":..., "loss":..., "nonce":...,
     "round":..., "samples":..., "schema":"axis-fed/1", "weights":[...]}

Keys and signatures are Base64-encoded raw Ed25519 bytes (32-byte public
key, 64-byte signature) — the same encoding as the Axis device registry.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Dict, Optional

import nacl.exceptions
import nacl.signing

#: Contribution schema marker (bump when the wire format changes).
SCHEMA = "axis-fed/1"

#: Contribution fields covered by the device signature (canonical order).
SIGNED_FIELDS = (
    "schema",
    "round",
    "device_id",
    "weights",
    "samples",
    "loss",
    "nonce",
)


def _normalize_numbers(value: Any) -> Any:
    """Canonicalize numbers so ``2`` and ``2.0`` produce identical bytes."""
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {k: _normalize_numbers(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_numbers(v) for v in value]
    return value


def canonical_json_bytes(obj: Any) -> bytes:
    """Canonical JSON serialization (sort_keys, no whitespace, int floats)."""
    return json.dumps(
        _normalize_numbers(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_contribution_message(contribution: Dict[str, Any]) -> bytes:
    """The canonical bytes a gateway signs for a contribution."""
    body = {k: contribution[k] for k in SIGNED_FIELDS if k in contribution}
    return canonical_json_bytes(body)


def generate_keypair() -> tuple:
    """Generate a fresh Ed25519 device keypair.

    Returns ``(secret_key_b64, public_key_b64)``. The secret is the Base64
    seed (32 bytes); the public key is Base64 32-byte raw Ed25519.
    """
    secret = nacl.signing.SigningKey.generate()
    return (
        base64.b64encode(bytes(secret)).decode("ascii"),
        base64.b64encode(bytes(secret.verify_key)).decode("ascii"),
    )


def sign_contribution(
    secret_key_b64: str,
    contribution: Dict[str, Any],
) -> Dict[str, Any]:
    """Sign a contribution with the device key; returns a copy with ``signature``.

    ``contribution`` must contain the ``SIGNED_FIELDS``; any existing
    ``signature`` is replaced.
    """
    message = canonical_contribution_message(contribution)
    secret = nacl.signing.SigningKey(base64.b64decode(secret_key_b64))
    signature = secret.sign(message).signature
    signed = dict(contribution)
    signed["signature"] = base64.b64encode(signature).decode("ascii")
    return signed


def verify_contribution(
    public_key_b64: str,
    contribution: Dict[str, Any],
) -> bool:
    """Verify a signed contribution against the device public key.

    Returns ``False`` (never raises) for invalid Base64, wrong key/signature
    lengths, a missing ``signature``, or a bad signature.
    """
    signature_b64 = contribution.get("signature")
    if not isinstance(signature_b64, str):
        return False
    try:
        public_key_bytes = base64.b64decode(public_key_b64, validate=True)
        signature_bytes = base64.b64decode(signature_b64, validate=True)
    except ValueError:
        return False

    if len(public_key_bytes) != 32 or len(signature_bytes) != 64:
        return False

    try:
        public_key = nacl.signing.VerifyKey(public_key_bytes)
        public_key.verify(canonical_contribution_message(contribution), signature_bytes)
        return True
    except (ValueError, nacl.exceptions.BadSignatureError):
        return False


def is_valid_public_key_b64(public_key_b64: str) -> bool:
    """True for a strict Base64 32-byte Ed25519 public key."""
    try:
        raw = base64.b64decode(public_key_b64, validate=True)
    except ValueError:
        return False
    return len(raw) == 32
