"""Federated contribution protocol tests (agent.fed.protocol)."""

import base64

import pytest

from agent.fed.protocol import (
    canonical_contribution_message,
    generate_keypair,
    is_valid_public_key_b64,
    sign_contribution,
    verify_contribution,
)


def _contribution(**overrides):
    base = {
        "schema": "axis-fed/1",
        "round": 3,
        "device_id": "gateway_00",
        "weights": [2.0, 3.0],
        "samples": 120,
        "loss": 0.041,
        "nonce": 42,
    }
    base.update(overrides)
    return base


def test_sign_verify_roundtrip():
    secret, public = generate_keypair()
    signed = sign_contribution(secret, _contribution())
    assert verify_contribution(public, signed) is True


def test_tampered_weights_fail_verification():
    secret, public = generate_keypair()
    signed = sign_contribution(secret, _contribution())
    signed["weights"] = [999.0, 3.0]
    assert verify_contribution(public, signed) is False


def test_tampered_loss_fails_verification():
    secret, public = generate_keypair()
    signed = sign_contribution(secret, _contribution())
    signed["loss"] = 0.0
    assert verify_contribution(public, signed) is False


def test_wrong_key_fails_verification():
    secret, _ = generate_keypair()
    _, other_public = generate_keypair()
    signed = sign_contribution(secret, _contribution())
    assert verify_contribution(other_public, signed) is False


def test_missing_signature_fails():
    _, public = generate_keypair()
    assert verify_contribution(public, _contribution()) is False


def test_invalid_base64_keys_fail():
    secret, _ = generate_keypair()
    signed = sign_contribution(secret, _contribution())
    assert verify_contribution("!!!not-base64!!!", signed) is False
    assert is_valid_public_key_b64("!!!not-base64!!!") is False


def test_wrong_length_public_key_fails():
    secret, _ = generate_keypair()
    signed = sign_contribution(secret, _contribution())
    short_key = base64.b64encode(b"short").decode("ascii")
    assert verify_contribution(short_key, signed) is False


def test_canonical_message_is_deterministic():
    a = canonical_contribution_message(_contribution())
    b = canonical_contribution_message(_contribution())
    assert a == b
    # Signature must not be part of the canonical message.
    secret, _ = generate_keypair()
    signed = sign_contribution(secret, _contribution())
    assert canonical_contribution_message(signed) == a


def test_float_integer_normalization_in_canonical():
    message = canonical_contribution_message(
        _contribution(weights=[2.0, 3.0], loss=1.0)
    )
    assert message == canonical_contribution_message(
        _contribution(weights=[2, 3], loss=1)
    )


def test_keypair_shape():
    secret, public = generate_keypair()
    assert len(base64.b64decode(secret)) == 32
    assert len(base64.b64decode(public)) == 32
    assert is_valid_public_key_b64(public) is True
