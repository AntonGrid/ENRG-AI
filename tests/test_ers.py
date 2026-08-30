"""ERS reputation economy & commitment digest tests (Phase 2, PoI)."""

from agent.fed.aggregate import FedResult
from agent.fed.digest import (
    build_commitment,
    contribution_digest,
    sign_commitment,
    verify_commitment,
)
from agent.fed.ers import ERS_FLOOR, DEFAULT_ERS, update_ers


def _contribution(device_id, loss=0.5):
    return {
        "device_id": device_id,
        "samples": 100,
        "loss": loss,
        "weights": [1.0, 2.0],
    }


def test_ers_accepted_contributions_gain():
    result = FedResult(
        round=1,
        weights=[1.0, 2.0],
        loss=0.5,
        n_contributions=1,
        accepted=[_contribution("gw_a")],
    )
    ers = update_ers({}, result)
    # DEFAULT 0.5 + 0.05 * (1 - 0.5) = 0.525
    assert ers["gw_a"] == round(0.525, 4)


def test_ers_outlier_rejection_decays():
    result = FedResult(
        round=1,
        weights=[],
        loss=None,
        n_contributions=1,
        accepted=[],
        rejected=[{**_contribution("gw_bad"), "reason": "outlier_weights"}],
    )
    ers = update_ers({}, result)
    # DEFAULT 0.5 * 0.5 = 0.25
    assert ers["gw_bad"] == round(0.25, 4)


def test_ers_ignores_non_quality_rejections():
    result = FedResult(
        round=1,
        weights=[],
        loss=None,
        n_contributions=1,
        accepted=[],
        rejected=[{**_contribution("gw_err"), "reason": "signature_invalid"}],
    )
    ers = update_ers({}, result)
    assert "gw_err" not in ers


def test_ers_floor_is_respected():
    result = FedResult(
        round=1,
        weights=[],
        loss=None,
        n_contributions=1,
        accepted=[],
        rejected=[{**_contribution("gw_bad"), "reason": "outlier_loss"}],
    )
    ers = update_ers({"gw_bad": 0.001}, result)
    assert ers["gw_bad"] >= ERS_FLOOR


# ── contribution digests & commitments ──────────────────────────────────────


def test_contribution_digest_is_deterministic():
    a = {"schema": "axis-fed/1", "round": 1, "device_id": "gw", "weights": [1.0, 2.0]}
    assert contribution_digest(a) == contribution_digest(a)
    b = {**a, "weights": [1.0, 2.001]}
    assert contribution_digest(a) != contribution_digest(b)


def test_commitment_sign_and_verify():
    import agent.fed.protocol as proto

    secret, pub = proto.generate_keypair()
    sample = {"schema": "axis-fed/1", "round": 2, "device_id": "gw", "weights": [1.0]}
    commit = build_commitment(round_no=2, device_id="gw", contribution=sample)
    assert commit["schema"] == "axis-fed-commit/1"
    assert commit["digest"]
    signed = sign_commitment(commit, secret)
    assert verify_commitment(pub, signed)


def test_commitment_tamper_detected():
    import agent.fed.protocol as proto

    secret, pub = proto.generate_keypair()
    sample = {"schema": "axis-fed/1", "round": 2, "device_id": "gw", "weights": [1.0]}
    commit = build_commitment(round_no=2, device_id="gw", contribution=sample)
    signed = sign_commitment(commit, secret)
    signed["digest"] = "tampered"
    assert not verify_commitment(pub, signed)


def test_onchain_commit_message_layout():
    """P1-4 (audit 2026-08-30): the message a device signs for the on-chain
    commit_contribution PDA must match programs/enrg-mvp/src/state/poi.rs
    (b"enrg:poi:commit" || round(8 LE) || device_id(32) || digest(32))."""
    from agent.fed.digest import ONCHAIN_COMMIT_MSG_LEN, onchain_commit_message

    device = bytes(range(32))
    digest = bytes(range(32, 64))
    msg = onchain_commit_message(round_no=7, device_id_pubkey=device, digest_sha256=digest)
    assert len(msg) == ONCHAIN_COMMIT_MSG_LEN == 87
    assert msg[:15] == b"enrg:poi:commit"
    assert msg[15:23] == (7).to_bytes(8, "little")
    assert msg[23:55] == device
    assert msg[55:87] == digest
