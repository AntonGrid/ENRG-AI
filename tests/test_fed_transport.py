"""Real FL transport tests (P3-4, audit 2026-08-30).

Starts an FLServer on an ephemeral port, submits signed contributions over
HTTP, closes the round and verifies the aggregated global weights. No external
network is used (127.0.0.1 only).
"""
import httpx
import pytest

from agent.fed.protocol import generate_keypair, sign_contribution
from agent.fed.transport import FLServer, close_round, fetch_weights, submit_contribution


def _contribution(device_id: str, round_no: int, weights, loss=0.5, samples=100):
    return {
        "schema": "axis-fed/1",
        "round": round_no,
        "device_id": device_id,
        "weights": weights,
        "samples": samples,
        "loss": loss,
        "nonce": 1,
    }


@pytest.fixture()
def server():
    srv = FLServer(host="127.0.0.1", port=0)
    srv.start()
    yield srv
    srv.stop()


def test_submit_and_aggregate_over_http(server):
    sk1, pk1 = generate_keypair()
    sk2, pk2 = generate_keypair()

    c1 = sign_contribution(sk1, _contribution("gw_a", 1, [1.0, 2.0], loss=0.4, samples=50))
    c2 = sign_contribution(sk2, _contribution("gw_b", 1, [3.0, 2.0], loss=0.6, samples=50))

    url = f"http://127.0.0.1:{server.port}"
    assert submit_contribution(url, pk1, c1)["accepted"] is True
    assert submit_contribution(url, pk2, c2)["accepted"] is True

    gw = close_round(url, 1)
    assert gw["accepted_count"] == 2
    assert gw["weights"] == [2.0, 2.0]  # sample-weighted FedAvg of [1,3] and [2,2]

    fetched = fetch_weights(url, 1)
    assert fetched["round"] == 1
    assert fetched["weights"] == [2.0, 2.0]


def test_rejects_invalid_signature(server):
    sk, pk = generate_keypair()
    c = sign_contribution(sk, _contribution("gw_a", 2, [1.0]))
    c["weights"] = [999.0]  # invalidates the signature

    url = f"http://127.0.0.1:{server.port}"
    with pytest.raises(httpx.HTTPStatusError) as e:
        submit_contribution(url, pk, c)
    assert e.value.response.status_code == 403


def test_rejects_wrong_public_key(server):
    sk, _ = generate_keypair()
    _, other_pk = generate_keypair()
    c = sign_contribution(sk, _contribution("gw_a", 3, [1.0]))

    url = f"http://127.0.0.1:{server.port}"
    with pytest.raises(httpx.HTTPStatusError) as e:
        submit_contribution(url, other_pk, c)  # wrong key for this signature
    assert e.value.response.status_code == 403


def test_weights_404_before_close(server):
    url = f"http://127.0.0.1:{server.port}"
    with pytest.raises(httpx.HTTPStatusError) as e:
        fetch_weights(url, 99)
    assert e.value.response.status_code == 404


def test_health(server):
    url = f"http://127.0.0.1:{server.port}"
    resp = httpx.get(f"{url}/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
