"""The policy: a probability, its two calibrating numbers, and the JSON between them.

The model is deliberately small, and so is the surface a test has to hold down: it scores a
mapping, it serialises losslessly, it learns a pattern that is *there*, and it refuses to be
fitted on a sample that could teach nothing (one class, or nobody home). The tests below are
written against a synthetic panel whose answer is known by construction, because "does it
learn" is only answerable if the label was planted.
"""
from __future__ import annotations

import json
import math

import pytest

from agent.backtest.features import FEATURES, POLICY_FEATURES
from agent.backtest.policy import DEFAULT_L2, LogisticPolicy, load_policy

START = 1_600_000_000.0


def learnable(count: int = 400, *, seed: int = 7):
    """A panel where the label *is* a threshold on ``ret_1``, plus noise that must be ignored.

    Deterministic without ``random``: a linear congruential walk, so the fixture repeats
    exactly across machines and a failure is a failure and not a coin. The two draws are
    independent on purpose — a fixture where ``atr_pct`` moved with ``ret_1`` would let a
    fitted model split its weight between them and make "the loudest weight is ``ret_1``" a
    claim about floating-point ordering rather than about learning.
    """
    state = int(seed)
    samples = []
    for _ in range(count):
        state = (1103515245 * state + 12345) % (2**31)
        momentum_noise = ((state >> 8) % 1000) / 1000.0 - 0.5
        state = (1103515245 * state + 12345) % (2**31)
        size_noise = ((state >> 8) % 1000) / 1000.0 - 0.5
        momentum = 3.0 * momentum_noise + 1.0
        case = {name: 0.0 for name in POLICY_FEATURES}
        case["ret_1"] = momentum
        case["atr_pct"] = 3.0 + size_noise
        case["confidence"] = 0.4 + size_noise / 10.0
        case["regime"] = 1.0
        samples.append((case, 1.0 if momentum > 1.0 else 0.0))
    return samples


def test_a_fitted_policy_puts_the_winners_above_the_losers():
    from agent.backtest.train import rank_auc

    policy = LogisticPolicy.fit(
        learnable(), features=POLICY_FEATURES, l2=0.1, horizon=3, cost=0.004
    )
    panel = learnable()
    auc = rank_auc(
        [policy.score(case) for case, _ in panel], [label for _, label in panel]
    )
    # Not 1.0: the label is a threshold on a continuous feature and a linear boundary cannot
    # land exactly on it, so a handful of bars either side are expected to swap. 0.9 on a
    # planted signal is learning; 0.5 would be the coin the whole exercise is testing against.
    assert auc > 0.9, f"a signal planted in `ret_1` scored an AUC of {auc:.3f}"


def test_the_weights_of_a_planted_signal_point_at_the_planted_feature():
    policy = LogisticPolicy.fit(learnable(), features=POLICY_FEATURES, l2=0.1)
    ranked = sorted(zip(policy.features, policy.weights), key=lambda kv: -abs(float(kv[1])))
    assert ranked[0][0] == "ret_1"
    assert float(ranked[0][1]) > 0.0


def test_the_score_is_a_probability_and_the_calibration_keeps_it_one():
    policy = LogisticPolicy.fit(learnable(), features=POLICY_FEATURES, l2=0.1)
    for case, _ in learnable(count=25):
        score = policy.score(case)
        assert 0.0 <= score <= 1.0
        assert math.isfinite(score)
    assert policy.temperature > 0.0


def test_the_calibration_brings_the_mean_score_back_to_the_base_rate():
    # Balanced learning inflates the raw score; the Platt step is what makes the *level*
    # honest, so the average probability over the panel is the frequency it was fitted on.
    policy = LogisticPolicy.fit(learnable(), features=POLICY_FEATURES, l2=0.1)
    scores = [policy.score(case) for case, _ in learnable()]
    mean = sum(scores) / len(scores)
    assert abs(mean - policy.base_rate) < 0.15, (
        f"mean p {mean:.2f} against a base rate of {policy.base_rate:.2f}: the calibration "
        "did not bring the probability back to a frequency"
    )
    assert policy.temperature != 1.0 or policy.calibration[1] != 0.0


def test_a_policy_survives_the_round_trip_through_json(tmp_path):
    policy = LogisticPolicy.fit(learnable(), features=POLICY_FEATURES, l2=0.1, threshold=0.4)
    path = policy.save(tmp_path / "policy.json")
    again = load_policy(path)
    assert again.features == policy.features
    assert again.threshold == policy.threshold
    assert again.to_dict() == policy.to_dict()
    for case, _ in learnable(count=20):
        # The document rounds to ten decimals, so the round trip is exact to that precision
        # and not to the bit — the difference is below any decision this model makes.
        assert again.score(case) == pytest.approx(policy.score(case), abs=1e-9)
        assert again.decision(case) == policy.decision(case)


def test_the_saved_document_is_the_numbers_a_reader_can_argue_with(tmp_path):
    policy = LogisticPolicy.fit(learnable(), features=POLICY_FEATURES, l2=0.1)
    raw = json.loads(policy.save(tmp_path / "p.json").read_text(encoding="utf-8"))
    assert raw["features"] == list(POLICY_FEATURES)
    assert len(raw["weights"]) == len(POLICY_FEATURES)
    assert len(raw["mean"]) == len(POLICY_FEATURES)
    assert raw["samples"] > 0
    assert 0.0 < raw["base_rate"] < 1.0
    assert isinstance(raw["calibration"], list) and len(raw["calibration"]) == 2


def test_a_policy_of_the_wrong_width_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="cannot hold"):
        LogisticPolicy(features=("ret_1", "ret_3"), weights=[1.0, 2.0, 3.0])


def test_fitting_on_one_class_is_refused_with_a_sentence():
    samples = [({name: 1.0 for name in POLICY_FEATURES}, 1.0) for _ in range(20)]
    with pytest.raises(ValueError, match="same label"):
        LogisticPolicy.fit(samples, features=POLICY_FEATURES)


def test_fitting_on_nothing_is_refused_with_a_sentence():
    with pytest.raises(ValueError, match="zero samples"):
        LogisticPolicy.fit([], features=POLICY_FEATURES)


def test_an_unlabelled_bar_is_dropped_and_not_counted_as_a_flat_one():
    samples = learnable(count=200) + [({name: 0.0 for name in POLICY_FEATURES}, None)] * 5
    policy = LogisticPolicy.fit(samples, features=POLICY_FEATURES, l2=0.1)
    assert policy.samples == 200


def test_the_contributions_add_up_to_the_logit_minus_the_bias():
    policy = LogisticPolicy.fit(learnable(), features=POLICY_FEATURES, l2=0.1)
    case = dict(learnable(count=1)[0][0])
    shares = policy.contributions(case)
    assert set(shares) == set(POLICY_FEATURES)
    assert sum(shares.values()) == pytest.approx(policy.logit(case) - policy.bias, abs=1e-9)


def test_describe_names_the_loudest_weights_and_a_bars_score():
    policy = LogisticPolicy.fit(learnable(), features=POLICY_FEATURES, l2=0.1)
    line = policy.describe()
    assert "ret_1" in line and "base rate" in line
    per_bar = policy.describe(learnable(count=1)[0][0])
    assert per_bar.startswith("p=")


def test_the_default_ridge_is_large_enough_to_be_the_thing_doing_the_work():
    # Fifteen correlated features on a thousand noisy labels: a small ridge would fit noise,
    # and the constant is asserted so that a change to it is a visible decision.
    assert DEFAULT_L2 >= 1.0
    assert len(FEATURES) == 15

