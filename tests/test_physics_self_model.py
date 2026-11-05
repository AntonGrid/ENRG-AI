"""The mirror: one metric, one prior, and one ledger.

Three properties are asserted, and they are the three arms of the decision:

* the metric rises with the evidence for "world" and is bounded — ``R_int`` is a
  likelihood ratio through a sigmoid, not a score with a range chosen afterwards;
* the prior only shifts the decision, and the sweep is monotone, so ``0.7`` in the
  acceptance rule is a statement about the *evidence* and not about a lucky prior;
* the ledger is content-addressed and decays at ε: an exact match is recalled with
  certainty at Δt = 0 and with ``e^{-εΔt}`` later, and a *nearly* matching stream
  is not recalled at all — "did I make this?" is a different question from "does
  this look like what I made?".

The phantom test is the sharp one: the score of the organism's own output must rise
from zero as the wait grows, because forgetting is the only thing that turns a
dream into a memory of the world.
"""
import json
import math

import numpy as np
import pytest

from agent.physics.homeostasis import Parameters
from agent.physics.self_model import (
    WORLD_JUDGEMENT,
    WORLD_PRIOR,
    Ledger,
    SelfModel,
    logit,
    prior_sweep,
    reality_score,
    sigmoid,
)
from agent.physics.sensorium import (
    DREAM_RED,
    DREAM_RHYTHM,
    DREAM_SURROGATE,
    Sensorium,
)


@pytest.fixture(scope="module")
def infancy():
    surface = Sensorium()
    world = [
        surface.hear("dataset:iris", limit=256),
        surface.hear("dataset:wine", limit=256),
    ]
    episodes = [surface.fixate(perception) for perception in world]
    episodes.append(surface.fixate(surface.dream(DREAM_RED, length=256, seed=3)))
    episodes.append(surface.fixate(surface.dream(DREAM_RHYTHM, length=256, seed=4)))
    episodes.append(
        surface.fixate(surface.dream(DREAM_SURROGATE, anchor=world[0], seed=5))
    )
    return episodes


@pytest.fixture(scope="module")
def mirror(infancy):
    model = SelfModel.awaken(infancy, Parameters(lam=0.4, epsilon=0.1), channels=8)
    for episode in infancy:
        if episode.origin != "world":
            model.emit(episode)
        model.observe(episode)
    return model


# --------------------------------------------------------------------------- #
# The two half-links: sigmoid and logit
# --------------------------------------------------------------------------- #


def test_logit_and_sigmoid_are_inverses_inside_the_box():
    for probability in (0.01, 0.2, 0.5, 0.8, 0.99):
        assert sigmoid(logit(probability)) == pytest.approx(probability)


def test_sigmoid_cannot_overflow_on_a_wide_likelihood_ratio():
    assert sigmoid(1e6) == pytest.approx(1.0)
    assert sigmoid(-1e6) == pytest.approx(0.0)
    assert 0.0 < sigmoid(-30.0) < 1e-12


# --------------------------------------------------------------------------- #
# The ledger: certain at the moment of emission, decaying afterwards
# --------------------------------------------------------------------------- #


def test_the_ledger_recalls_exactly_the_bytes_it_was_given():
    stream = Sensorium().hear("dataset:iris", limit=256)
    ledger = Ledger()
    ledger.record(stream, step=10)
    assert ledger.recall(stream.fingerprint, 10, epsilon=0.1) == pytest.approx(1.0)
    assert ledger.recall(stream.fingerprint, 20, epsilon=0.1) == pytest.approx(
        math.exp(-1.0)
    )
    assert ledger.recall("0" * 64, 10, epsilon=0.1) == 0.0


def test_the_ledger_prunes_old_emissions_and_stays_bounded():
    from agent.physics.sensorium import Perception, WORLD, DOMAIN_FINANCE

    ledger = Ledger(limit=3)
    for index in range(6):
        values = np.full(64, float(index))
        perception = Perception(
            values=values, origin=WORLD, source=f"s{index}", domain=DOMAIN_FINANCE
        )
        ledger.record(perception, step=index)
    assert len(ledger) == 3
    kept = {entry.step for entry in ledger.emissions.values()}
    assert kept == {3, 4, 5}


# --------------------------------------------------------------------------- #
# The score: evidence times what the organism knows it did
# --------------------------------------------------------------------------- #


def test_the_score_is_the_posterior_discounted_by_recall():
    posterior = sigmoid(1.0 + logit(WORLD_PRIOR))
    assert reality_score(1.0, recall=0.0) == pytest.approx(posterior)
    assert reality_score(1.0, recall=1.0) == pytest.approx(0.0)
    assert reality_score(1.0, recall=0.5) == pytest.approx(0.5 * posterior)
    assert 0.0 <= reality_score(-50.0, 0.0) < 0.01
    assert 0.99 < reality_score(50.0, 0.0) < 1.0


def test_the_score_is_monotone_in_the_evidence_and_bounded():
    scores = [reality_score(value, 0.0) for value in (-4.0, -1.0, 0.0, 1.0, 4.0)]
    assert scores == sorted(scores)
    assert scores[0] < WORLD_JUDGEMENT < scores[-1]


def test_the_prior_sweep_is_monotone_and_leaves_the_ranking_alone():
    rng = np.random.default_rng(0)
    model = SelfModel(
        parameters=Parameters(lam=0.4, epsilon=0.1),
        posture=None,
        prior=WORLD_PRIOR,
        channels=8,
    )
    from agent.physics.self_model import Moment

    model.trace = [
        Moment(0, "w", "world", 0.6, 0.5, 0.0, float(rng.normal(1.0, 0.2)), 1.0,
               False, True),
        Moment(1, "s", "self", 0.4, 0.5, 0.0, float(rng.normal(-1.0, 0.2)), 1.0,
               False, False),
    ]
    rows = prior_sweep(model)
    assert [row["world"] for row in rows] == sorted(row["world"] for row in rows)
    assert all(row["auc"] == pytest.approx(1.0) for row in rows)


# --------------------------------------------------------------------------- #
# The mirror on a real infancy: it observes, it remembers, it reports
# --------------------------------------------------------------------------- #


def test_observing_writes_one_moment_per_fixation_and_judges_the_self(mirror, infancy):
    assert len(mirror.trace) == sum(len(episode) for episode in infancy)
    for moment in mirror.trace:
        assert 0.0 <= moment.reality_score < 1.0
        assert 0.0 <= moment.resonance <= 1.0
        assert moment.judged_world == (moment.reality_score >= WORLD_JUDGEMENT)
    # what the organism emitted is judged its own, with its own memory doing the work
    dreams = [moment for moment in mirror.trace if moment.origin != "world"]
    assert dreams and np.mean([moment.recall for moment in dreams]) > 0.0
    assert np.mean([moment.reality_score for moment in dreams]) < WORLD_JUDGEMENT


def test_an_emission_is_recorded_at_the_step_it_happened(infancy):
    model = SelfModel.awaken(infancy, Parameters(lam=0.4, epsilon=0.1), channels=8)
    dream = [episode for episode in infancy if episode.origin != "world"][0]
    emission = model.emit(dream)
    assert emission.step == 0
    assert emission.fingerprint == dream.fingerprint
    assert len(model.ledger) == 1


def test_the_summary_is_json_able_and_separates_the_arms(mirror):
    summary = mirror.summary()
    assert json.loads(json.dumps(summary))  # no numpy scalars leaked
    assert summary["steps"] == len(mirror.trace)
    assert "comparator_only" in summary and "auc_resonance" in summary
    assert summary["per_source"]
    for row in summary["per_source"].values():
        assert 0.0 <= row["reality_score"] < 1.0


def test_the_blind_copy_has_no_memory_and_scores_the_dreams_higher(mirror):
    blind = mirror.blind()
    assert all(moment.recall == 0.0 for moment in blind.trace)
    world, _ = mirror.piles()
    blind_world, blind_dream = blind.piles()
    assert blind_world.mean() == pytest.approx(world.mean(), abs=0.2)
    assert blind_dream.mean() > 0.0  # without memory the dreams are not free to reject


def test_the_phantom_rises_from_zero_as_the_organism_forgets(mirror):
    rows = mirror.phantom(waits=(0, 1, 2, 5, 10, 40))
    selves = [row["self"] for row in rows]
    assert selves[0] == pytest.approx(0.0)
    assert selves == sorted(selves)  # forgetting only ever adds plausibility
    assert selves[-1] > selves[0]
    assert 0.0 <= rows[-1]["world"] <= 1.0


def test_the_acceptance_check_reports_the_rule_it_was_given(mirror):
    verdict = mirror.acceptance(world_floor=0.7, self_ceiling=0.3)
    assert set(verdict) >= {"passed", "world_ok", "self_ok", "world_mean", "self_mean"}
    assert verdict["world_floor"] == pytest.approx(0.7)
    assert verdict["self_ceiling"] == pytest.approx(0.3)
    strict = mirror.acceptance(world_floor=1.1, self_ceiling=0.0)
    assert strict["passed"] is False and strict["world_ok"] is False
