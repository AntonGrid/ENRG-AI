"""Contribution digests & on-chain commitments (Phase 2, Proof-of-Intelligence).

Every federated contribution is reducible to a compact, deterministic digest
(SHA-256 over the canonical JSON, base58-encoded — Solana-friendly). Committing
that digest on-chain makes the history of "what was contributed when and by
whom" publicly verifiable without publishing the weights themselves: anyone can
re-check ``digest == contribution_digest(contribution)`` and verify the signed
commitment.

The actual on-chain write (a commitment PDA in the ENRG program) is the
oracle's job (ADR-0010: oracle-only writes); this module defines the
off-chain contract the oracle and verifiers share.
"""
from __future__ import annotations

import base64
import hashlib
from typing import Any, Dict

from agent.fed.protocol import canonical_json_bytes

#: Commitment schema marker (bump when the wire format changes).
COMMIT_SCHEMA = "axis-fed-commit/1"

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58_ALPHABET[r] + out
    pad = 0
    for b in data:
        if b == 0:
            pad += 1
        else:
            break
    return "1" * pad + out


def contribution_digest(contribution: Dict[str, Any]) -> str:
    """Deterministic SHA-256 digest (base58) of a contribution's canonical JSON.

    ``contribution`` should be the *signed* object (including ``signature``),
    or the unsigned body — as long as the caller always hashes the same shape
    it verified. The digest never reveals the weights.
    """
    message = canonical_json_bytes(contribution)
    return _b58encode(hashlib.sha256(message).digest())


def build_commitment(
    *,
    round_no: int,
    device_id: str,
    contribution: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the unsigned commitment body for a contribution digest."""
    return {
        "schema": COMMIT_SCHEMA,
        "round": round_no,
        "device_id": device_id,
        "digest": contribution_digest(contribution),
    }


def sign_commitment(commitment: Dict[str, Any], secret_key_b64: str) -> Dict[str, Any]:
    """Ed25519-sign a commitment body (returns a copy with ``signature``)."""
    import nacl.signing

    message = canonical_json_bytes(commitment)
    secret = nacl.signing.SigningKey(base64.b64decode(secret_key_b64))
    signature = secret.sign(message).signature
    signed = dict(commitment)
    signed["signature"] = base64.b64encode(signature).decode("ascii")
    return signed


def verify_commitment(public_key_b64: str, signed: Dict[str, Any]) -> bool:
    """Verify a signed commitment. Returns False (never raises)."""
    import nacl.exceptions
    import nacl.signing

    signature_b64 = signed.get("signature")
    if not isinstance(signature_b64, str):
        return False
    try:
        pk_bytes = base64.b64decode(public_key_b64, validate=True)
        sig_bytes = base64.b64decode(signature_b64, validate=True)
    except ValueError:
        return False
    if len(pk_bytes) != 32 or len(sig_bytes) != 64:
        return False
    body = {k: v for k, v in signed.items() if k != "signature"}
    try:
        verifier = nacl.signing.VerifyKey(pk_bytes)
        verifier.verify(canonical_json_bytes(body), sig_bytes)
        return True
    except (ValueError, nacl.exceptions.BadSignatureError):
        return False
