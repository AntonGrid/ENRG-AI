"""Federated aggregation tests (agent.fed.aggregate)."""

from agent.fed.aggregate import fed_avg
from agent.fed.protocol import generate_keypair, sign_contribution


def _signed(weights, samples=10, loss=0.05, nonce=0, round_no=1):
    secret, public = generate_keypair()
    contribution = {
        "schema": "axis-fed/1",
        "round": round_no,
        "device_id": f"g_{nonce}",
        "weights": weights,
        "samples": samples,
        "loss": loss,
        "nonce": nonce,
        "public_key": public,
    }
    return sign_contribution(secret, contribution)


def test_empty_contributions():
    result = fed_avg([])
    assert result.weights == []
    assert result.n_contributions == 0
    assert result.accepted_count == 0


def test_fed_avg_weighted_by_samples():
    a = _signed([0.0, 1.0], samples=1, nonce=0)
    b = _signed([2.0, 1.0], samples=3, nonce=1)
    result = fed_avg([a, b])
    assert result.accepted_count == 2
    assert result.weights[0] == 1.5  # (0*1 + 2*3) / 4
    assert result.weights[1] == 1.0


def test_unverified_contribution_is_rejected():
    secret, public = generate_keypair()
    signed = {
        "schema": "axis-fed/1",
        "round": 1,
        "device_id": "bad",
        "weights": [1.0, 1.0],
        "samples": 10,
        "loss": 0.1,
        "nonce": 0,
        "public_key": public,
    }
    result = fed_avg([signed])
    assert result.accepted_count == 0
    assert result.rejected[0]["reason"] == "signature_invalid"


def test_missing_public_key_is_rejected():
    contribution = {
        "schema": "axis-fed/1",
        "round": 1,
        "device_id": "bad",
        "weights": [1.0, 1.0],
        "samples": 10,
        "loss": 0.1,
        "nonce": 0,
    }
    result = fed_avg([contribution])
    assert result.accepted_count == 0
    assert result.rejected[0]["reason"] == "missing_public_key"


def test_insufficient_samples_is_rejected():
    good = [_signed([1.0, 1.0], samples=10, nonce=0)]
    result = fed_avg(good, min_samples=20)
    assert result.accepted_count == 0
    assert result.rejected[0]["reason"] == "insufficient_samples"


def test_outlier_weights_are_rejected():
    # Two honest gateways around [2, 3], one malicious at [500, 500].
    contributions = [
        _signed([2.0, 3.0], samples=100, loss=0.01, nonce=0),
        _signed([1.8, 3.2], samples=100, loss=0.01, nonce=1),
        _signed([500.0, 500.0], samples=100, loss=0.01, nonce=2),
    ]
    result = fed_avg(contributions)
    assert result.accepted_count == 2
    assert result.rejected[0]["reason"] == "outlier_weights"
    # Global weights close to the honest mean.
    assert result.weights[0] == 1.9
    assert result.weights[1] == 3.1


def test_outlier_loss_is_rejected():
    contributions = [
        _signed([2.0, 3.0], samples=100, loss=0.01, nonce=0),
        _signed([2.0, 3.0], samples=100, loss=0.02, nonce=1),
        _signed([2.0, 3.0], samples=100, loss=99.0, nonce=2),
    ]
    result = fed_avg(contributions)
    assert result.accepted_count == 2
    assert result.rejected[0]["reason"] == "outlier_loss"


def test_outlier_removal_requires_population():
    # With only 2 contributions there is no meaningful z-score population:
    # both survive even when far apart.
    result = fed_avg(
        [
            _signed([1.0, 1.0], samples=10, loss=0.01, nonce=0),
            _signed([100.0, 100.0], samples=10, loss=0.01, nonce=1),
        ]
    )
    assert result.accepted_count == 2


def test_verify_can_be_disabled():
    # Unverified raw contributions pass when verify=False (e.g. a trusted
    # internal aggregator feeding its own models).
    contribution = {
        "schema": "axis-fed/1",
        "round": 1,
        "device_id": "g_0",
        "weights": [2.0, 3.0],
        "samples": 10,
        "loss": 0.01,
        "nonce": 0,
    }
    result = fed_avg([contribution], verify=False)
    assert result.accepted_count == 1
