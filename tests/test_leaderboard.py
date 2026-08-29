"""PoI leaderboard tests — the ERS loop punishes the malicious gateway."""

from agent.leaderboard import main, simulate


def test_simulate_punishes_malicious_gateway():
    history = simulate(n_rounds=5, n_gateways=5, include_malicious=True, seed=7)
    last = history[-1]
    rows = {r.device_id: r for r in last.rows}

    # Malicious gateway is rejected in every round and its ERS decays to floor.
    assert rows["gateway_04"].malicious
    assert rows["gateway_04"].accepted == 0  # never accepted in the last round
    assert rows["gateway_04"].ers <= 0.2  # decayed toward the floor
    assert rows["gateway_04"].weight_multiplier < 0.3  # nearly silenced

    # Honest gateways are accepted and their ERS rises well above floor.
    assert rows["gateway_00"].accepted == 1  # accepted in the last round
    assert rows["gateway_00"].ers >= 0.6


def test_simulate_without_malicious_mostly_accepted():
    # MAD screening can reject a *random* outlier among honest gateways too —
    # that is the point of quality screening; at most one is ever rejected.
    history = simulate(n_rounds=3, n_gateways=4, include_malicious=False, seed=7)
    for board in history:
        assert board.rejected_count <= 1
        assert board.accepted_count >= board.n_contributions - 1


def test_simulate_ers_feed_into_weight_multiplier():
    history = simulate(n_rounds=4, n_gateways=5, include_malicious=True, seed=7)
    last = history[-1]
    by_id = {r.device_id: r for r in last.rows}
    # Reputation weighting: the punished malicious gateway should carry less
    # sample weight than the trusted gateways in the final round.
    assert by_id["gateway_04"].weight_multiplier < by_id["gateway_00"].weight_multiplier


def test_leaderboard_serializable():
    history = simulate(n_rounds=2, n_gateways=3, include_malicious=False, seed=1)
    import json

    payload = [b.to_dict() for b in history]
    assert json.dumps(payload)
    assert payload[0]["rows"]


def test_leaderboard_cli_runs():
    assert main(["--rounds", "2", "--gateways", "3"]) == 0
