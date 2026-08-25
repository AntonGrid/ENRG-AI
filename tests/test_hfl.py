"""Hierarchical federated learning tests (agent.hfl)."""

import pytest

from agent.fed.aggregate import fed_avg
from agent.fed.protocol import generate_keypair, sign_contribution, verify_contribution
from agent.hfl.global_aggregator import GlobalAggregator
from agent.hfl.protocol import SCHEMA_V2, make_contribution, sign_regional
from agent.hfl.region import RegionAggregator
from agent.hfl.simulate import run_simulation
from agent.hfl.weights import reputation_map, reputation_weight


def _gateway(weights, samples=50, loss=0.05, device_id="gw_0", region="eu-0", level=1):
    secret, public = generate_keypair()
    contribution = {
        "schema": SCHEMA_V2,
        "round": 1,
        "device_id": device_id,
        "weights": list(weights),
        "samples": samples,
        "loss": loss,
        "nonce": 0,
        "level": level,
        "domain": "energy",
        "region": region,
        "version": "model_1.0",
        "public_key": public,
    }
    return sign_contribution(secret, contribution)


# ── protocol (axis-fed/2) ───────────────────────────────────────────────────


def test_axis_fed2_new_fields_are_signed():
    secret, public = generate_keypair()
    contribution = {
        "schema": SCHEMA_V2,
        "round": 1,
        "device_id": "gw_0",
        "weights": [1.0, 2.0],
        "samples": 10,
        "loss": 0.1,
        "nonce": 1,
        "level": 2,
        "domain": "climate",
        "region": "eu-7",
        "version": "model_2.0",
        "quality": {"val_loss": 0.2},
        "public_key": public,
    }
    signed = sign_contribution(secret, contribution)
    assert verify_contribution(public, signed) is True

    tampered = dict(signed)
    tampered["region"] = "us-1"
    assert verify_contribution(public, tampered) is False


def test_sign_regional_adds_public_key():
    secret, _ = generate_keypair()
    unsigned = make_contribution(
        level=2,
        domain="energy",
        region="eu-0",
        device_id="region_eu-0",
        weights=[2.0, 3.0],
        samples=100,
        loss=0.01,
        round_no=1,
    )
    signed = sign_regional(secret, unsigned)
    assert signed["level"] == 2
    assert signed["schema"] == SCHEMA_V2
    assert verify_contribution(signed["public_key"], signed) is True


# ── reputation weights ──────────────────────────────────────────────────────


def test_reputation_weight_range():
    assert reputation_weight(0.0) == 0.1
    assert reputation_weight(1.0) == 1.0
    assert 0.1 < reputation_weight(0.5) < 1.0


def test_reputation_map_defaults_unknown_devices():
    contributions = [_gateway(weights=[1, 1], device_id="a")]
    mapping = reputation_map(contributions, ers_map={"other": 1.0})
    assert mapping["a"] == pytest.approx(0.1)


def test_fed_avg_respects_extra_weight():
    a = _gateway(weights=[0.0, 1.0], samples=1, device_id="a")
    b = _gateway(weights=[2.0, 1.0], samples=1, device_id="b")
    result = fed_avg([a, b], extra_weight={"a": 0.1, "b": 1.0})
    assert result.weights[0] == pytest.approx(2.0 / 1.1)


# ── regional aggregation ────────────────────────────────────────────────────


def test_region_aggregates_gateways_into_signed_contribution():
    secret, public = generate_keypair()
    aggregator = RegionAggregator(region="eu-0", secret_key=secret, round_no=1)

    result = aggregator.aggregate(
        [
            _gateway([2.0, 3.0], device_id="g0", region="eu-0"),
            _gateway([2.1, 2.9], device_id="g1", region="eu-0"),
            _gateway([1.9, 3.1], device_id="g2", region="eu-0"),
        ]
    )
    assert result.accepted_count == 3
    assert result.weights[0] == pytest.approx(2.0, abs=0.05)
    assert result.weights[1] == pytest.approx(3.0, abs=0.05)

    regional = result.regional_contribution
    assert regional is not None
    assert regional["level"] == 2
    assert verify_contribution(public, regional) is True


def test_malicious_gateway_rejected_at_region_level():
    aggregator = RegionAggregator(region="eu-0")
    result = aggregator.aggregate(
        [
            _gateway([2.0, 3.0], device_id="g0", region="eu-0"),
            _gateway([2.1, 2.9], device_id="g1", region="eu-0"),
            _gateway([2.0, 3.0], device_id="g2", region="eu-0"),
            _gateway([500.0, 500.0], device_id="evil", region="eu-0"),
        ]
    )
    assert result.accepted_count == 3
    assert any("evil" in r["device_id"] for r in result.rejected)


def test_unsigned_gateway_rejected_at_region_level():
    aggregator = RegionAggregator(region="eu-0")
    bogus = {
        "schema": SCHEMA_V2,
        "round": 1,
        "device_id": "nokey",
        "weights": [2.0, 3.0],
        "samples": 50,
        "loss": 0.05,
        "nonce": 0,
        "level": 1,
        "domain": "energy",
        "region": "eu-0",
        "version": "model_1.0",
    }
    result = aggregator.aggregate([bogus])
    assert result.accepted_count == 0
    assert result.rejected[0]["reason"] == "missing_public_key"


# ── global aggregation ──────────────────────────────────────────────────────


def test_global_aggregates_regions():
    regionals = [
        sign_regional(
            generate_keypair()[0],
            make_contribution(
                level=2, domain="energy", region=f"eu-{i}", device_id=f"r{i}",
                weights=weights, samples=100, loss=0.01, round_no=1,
            ),
        )
        for i, weights in enumerate(([2.0, 3.0], [2.1, 2.9], [1.9, 3.1]))
    ]
    result = GlobalAggregator().aggregate(regionals)
    assert result.accepted_count == 3
    assert result.weights[0] == pytest.approx(2.0)
    assert result.weights[1] == pytest.approx(3.0)


def test_malicious_region_rejected_globally():
    regionals = []
    for i, weights in enumerate(([2.0, 3.0], [2.1, 2.9], [1.9, 3.1], [500.0, 500.0])):
        secret, _ = generate_keypair()
        regionals.append(
            sign_regional(
                secret,
                make_contribution(
                    level=2, domain="energy", region=f"r{i}", device_id=f"r{i}",
                    weights=weights, samples=100, loss=0.01, round_no=1,
                ),
            )
        )
    result = GlobalAggregator().aggregate(regionals)
    assert result.accepted_count == 3
    assert any(r["reason"] == "outlier_weights" for r in result.rejected)
    assert result.weights[0] == pytest.approx(2.0, abs=0.05)


# ── simulation ──────────────────────────────────────────────────────────────


def test_simulation_global_beats_regional_and_local():
    summary = run_simulation(
        n_regions=4,
        gateways_per_region=3,
        samples_per_gateway=40,
        include_malicious_gateway=True,
        include_malicious_region=True,
    )
    assert summary["improved_vs_regional"] is True
    assert summary["improved_vs_local"] is True
    assert summary["global_weights"][0] == pytest.approx(2.0, abs=0.15)
    assert summary["global_weights"][1] == pytest.approx(3.0, abs=0.15)

