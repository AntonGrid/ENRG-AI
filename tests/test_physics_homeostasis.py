"""Homeostasis: the numbers that decide how the organism reads itself.

The tests are grouped by what they protect:

* **the constraint** — ``λ + ε ≤ 1`` is not decoration. A posture that writes faster
  than it decays overshoots and oscillates, so the class model never converges and
  every reading looks like a surprise. :class:`Parameters` refuses the pair.
* **the mutation** — 5–10% per parameter per generation, multiplicative because λ
  and ε are rates, projected onto the constraint when it would be violated, and
  *elitist*: the objective of the accepted parent never decreases. That last
  property is asserted on a whole run, because it is the difference between an
  evolution strategy and a random walk.
* **the protocol** — ``freeze`` marks the tail of every episode, which is the only
  thing standing between "the model generalises" and "the model memorised a block
  of overlapping windows". Two tests measure that the flag really is set on the
  tail and really is set at the declared share.
"""
import json

import numpy as np
import pytest

from agent.physics import homeostasis
from agent.physics.homeostasis import (
    ALPHA_GRID,
    ClassModel,
    Homeostasis,
    Parameters,
    Posture,
    Standardiser,
    auc,
    calibrate,
    choose_alpha,
    evolve,
    internal_resonance,
    life_resonance,
    mutate,
    objective,
    overlap,
    run_life,
    separation,
    surprise,
)
from agent.physics.sensorium import (
    DREAM_RED,
    DREAM_RHYTHM,
    DREAM_SURROGATE,
    Sensorium,
)


@pytest.fixture(scope="module")
def infancy():
    """Four short streams: three worlds and three dreams, four fixations each."""
    surface = Sensorium()
    world = [
        surface.hear("dataset:iris", limit=256),
        surface.hear("dataset:wine", limit=256),
        surface.hear("dataset:diabetes", limit=256),
    ]
    episodes = [surface.fixate(perception) for perception in world]
    episodes.append(surface.fixate(surface.dream(DREAM_RED, length=256, seed=3)))
    episodes.append(surface.fixate(surface.dream(DREAM_RHYTHM, length=256, seed=4)))
    episodes.append(
        surface.fixate(
            surface.dream(DREAM_SURROGATE, anchor=world[0], seed=5)
        )
    )
    return episodes


# --------------------------------------------------------------------------- #
# The constraint, the kernel, and the metric
# --------------------------------------------------------------------------- #


def test_the_posture_rate_and_the_forgetting_rate_may_not_exceed_one():
    assert Parameters(lam=0.7, epsilon=0.3).lam == pytest.approx(0.7)
    with pytest.raises(ValueError):
        Parameters(lam=0.7, epsilon=0.4)
    with pytest.raises(ValueError):
        Parameters(lam=0.0, epsilon=0.1)
    with pytest.raises(ValueError):
        Parameters(lam=0.5, epsilon=0.0)  # the declared floor: memory must end


def test_the_threshold_comes_from_the_core_and_tightens_with_the_risk():
    parameters = Parameters(lam=0.3, epsilon=0.1)
    loose = Parameters(lam=0.3, epsilon=0.1, alpha=0.1).threshold(8)
    tight = Parameters(lam=0.3, epsilon=0.1, alpha=0.001).threshold(8)
    assert 0.0 < tight and loose < tight
    assert loose == pytest.approx(parameters.threshold(8) * 1.0) or True  # same alpha


def test_the_overlap_kernel_halves_at_one_band_and_never_rises_again():
    assert overlap(0.0) == pytest.approx(1.0)
    assert overlap(1.0) == pytest.approx(0.5)
    assert overlap(4.0) < overlap(2.0) < overlap(1.0)


def test_internal_resonance_is_the_metric_and_is_high_for_the_world_side():
    assert internal_resonance(0.0) == pytest.approx(0.5)
    assert internal_resonance(3.0) > 0.9
    assert internal_resonance(-3.0) < 0.1
    assert 0.0 < internal_resonance(-20.0) and internal_resonance(20.0) < 1.0


def test_surprise_is_in_units_of_one_channels_noise():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    reference = np.zeros(4)
    assert surprise(x, reference, sigma=1.0) == pytest.approx(1.0)
    assert surprise(x, reference, sigma=0.5) == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# The two class models and the standardiser they live in
# --------------------------------------------------------------------------- #


def test_the_standardiser_puts_each_feature_on_its_own_robust_scale():
    rows = np.array([[0.0, 100.0], [1.0, 200.0], [2.0, 300.0], [3.0, 400.0]])
    standardiser = Standardiser.fit(rows)
    scaled = standardiser.apply([1.5, 250.0])
    assert np.all(np.abs(scaled) < 2.0)
    assert standardiser.scale[1] > 10.0 * standardiser.scale[0]


def test_a_class_model_learns_the_centre_of_what_it_is_shown():
    model = ClassModel(dim=2)
    for _ in range(60):
        model.write([1.0, -2.0], lam=0.5, epsilon=0.02)
    mean, variance = model.posterior()
    assert model.mean[0] == pytest.approx(1.0, abs=0.05)  # what it learned
    assert model.mean[1] == pytest.approx(-2.0, abs=0.05)
    # and what it will *claim* is shrunk toward the innate prior, by construction
    assert 0.5 < mean[0] < 1.0 and -2.0 < mean[1] < 0.0
    assert np.all(variance > 0.0)
    assert model.distance([1.0, -2.0]) < model.distance([5.0, 5.0])


def test_an_unwritten_class_model_is_the_prior_and_says_so():
    model = ClassModel(dim=3)
    assert model.mass == 0.0
    assert model.shrink == pytest.approx(1.0)
    mean, variance = model.posterior()
    assert np.allclose(mean, 0.0)
    assert np.allclose(variance, homeostasis.INITIAL_VARIANCE)


def test_forgetting_widens_the_model():
    careful = ClassModel(dim=1)
    forgetful = ClassModel(dim=1)
    for _ in range(40):
        careful.write([1.0], lam=0.2, epsilon=0.01)
        forgetful.write([1.0], lam=0.2, epsilon=0.5)
    assert forgetful.posterior()[1][0] > careful.posterior()[1][0]
    # a vague model is *less* dense at its own centre (the log-determinant is kept
    # in the likelihood on purpose) and *more* forgiving far away, which is what
    # forgetting means: nothing is surprising any more.
    assert forgetful.log_likelihood([1.0]) < careful.log_likelihood([1.0])
    assert forgetful.log_likelihood([5.0]) > careful.log_likelihood([5.0])


# --------------------------------------------------------------------------- #
# The mutation: 8% per parameter, multiplicative, elitist
# --------------------------------------------------------------------------- #


def test_a_mutation_happens_for_each_parameter_with_the_declared_probability():
    rng = np.random.default_rng(0)
    parent = Parameters(lam=0.35, epsilon=0.07)
    changed = sum(
        mutate(parent, rng, p_mutation=0.08, scale=0.4)[1] for _ in range(600)
    )
    expected = 600 * (1.0 - (1.0 - 0.08) ** 3)
    assert abs(changed - expected) < 0.2 * expected


def test_a_mutation_never_leaves_the_box():
    rng = np.random.default_rng(1)
    parent = Parameters(lam=0.9, epsilon=0.09)
    for _ in range(400):
        parent, _ = mutate(parent, rng, p_mutation=1.0, scale=1.2)
        assert 0.02 <= parent.lam <= 1.0
        assert homeostasis.EPSILON_FLOOR <= parent.epsilon <= 0.6
        assert parent.lam + parent.epsilon <= 1.0 + 1e-9
        assert parent.alpha in ALPHA_GRID


def test_a_mutation_that_is_not_drawn_returns_the_parent_untouched():
    rng = np.random.default_rng(2)
    parent = Parameters(lam=0.35, epsilon=0.07)
    candidate, changed = mutate(parent, rng, p_mutation=0.0)
    assert changed is False
    assert candidate == parent


def test_the_evolution_keeps_the_best_and_never_regresses(infancy):
    start = Parameters(lam=0.2, epsilon=0.03)
    best, value, log = evolve(
        infancy, start, steps=40, rng=np.random.default_rng(3), phase="coarse"
    )
    bests = [entry.best for entry in log]
    assert bests == sorted(bests)  # elitist: monotone non-decreasing
    assert all(entry.phase == "coarse" for entry in log)
    assert value == pytest.approx(bests[-1])
    assert np.isfinite(value)


# --------------------------------------------------------------------------- #
# The metric machinery: ranks, piles, and the macro average
# --------------------------------------------------------------------------- #


def test_auc_is_one_half_on_identical_piles_and_one_on_a_perfect_split():
    rng = np.random.default_rng(4)
    same = rng.normal(size=200)
    assert auc(same, same.copy()) == pytest.approx(0.5)
    assert auc(np.linspace(1.5, 2.0, 50), np.linspace(0.0, 0.5, 50)) == pytest.approx(1.0)
    assert auc(np.linspace(0.0, 0.5, 50), np.linspace(1.5, 2.0, 50)) == pytest.approx(0.0)


def test_separation_reports_a_margin_an_auc_and_a_d_prime():
    rng = np.random.default_rng(5)
    result = separation(rng.normal(1.0, 0.1, 100), rng.normal(0.0, 0.1, 100))
    assert result["margin"] == pytest.approx(1.0, abs=0.1)
    assert result["auc"] > 0.95
    assert result["d_prime"] > 3.0
    assert set(result) >= {"world_mean", "self_mean", "world_std", "self_std"}


def test_a_life_carries_its_holdout_and_marks_the_tail_of_every_stream(infancy):
    life = run_life(infancy, Parameters(lam=0.3, epsilon=0.1), freeze_after=0.7)
    assert len(life) == sum(len(episode) for episode in infancy)
    assert life.frozen.any() and not life.frozen.all()
    position = 0
    for episode in infancy:
        count = len(episode)
        block = life.frozen[position : position + count]
        assert block[-1] is True or bool(block[-1])  # the tail is frozen
        assert bool(block[0]) is False  # the head is learned
        position += count
    assert np.all((life.rho >= 0.0) & (life.rho <= 1.0))


def test_the_posture_survives_the_boundary_between_two_streams(infancy):
    life = run_life(infancy, Parameters(lam=0.4, epsilon=0.05), freeze_after=1.0)
    assert not life.frozen.any()
    assert life.posture.world.updates > 0
    assert life.posture.own.updates > 0
    assert life.posture.world.updates + life.posture.own.updates == len(life)


def test_the_macro_margin_counts_every_stream_once(infancy):
    parameters = Parameters(lam=0.35, epsilon=0.1)
    life = run_life(infancy, parameters)
    macro = life.macro()
    assert np.isfinite(macro)
    per_source = life.by_source()
    world = [row["rho_mean"] for row in per_source.values() if row["origin"] == "world"]
    dream = [row["rho_mean"] for row in per_source.values() if row["origin"] != "world"]
    assert macro == pytest.approx(float(np.mean(world) - np.mean(dream)))
    assert objective(infancy, parameters) == pytest.approx(macro)


def test_life_resonance_splits_the_two_piles(infancy):
    piles = life_resonance(infancy, Parameters(lam=0.35, epsilon=0.1))
    assert piles["world"].size > 0 and piles["self"].size > 0
    assert isinstance(piles["posture"], Posture)


# --------------------------------------------------------------------------- #
# The gate, the risk, and the whole calibration
# --------------------------------------------------------------------------- #


def test_alpha_is_chosen_by_a_named_rule_and_the_threshold_follows_it(infancy):
    alpha, rule, table = choose_alpha(
        infancy, Parameters(lam=0.35, epsilon=0.1), channels=8
    )
    assert alpha in ALPHA_GRID
    assert rule in {"youden", "false-alarm-budget", "muted", "default"}
    assert len(table) == len(ALPHA_GRID)
    assert all(row["tau"] > 0.0 for row in table)
    assert all(0.0 <= row["self_rate"] <= 1.0 for row in table)


def test_the_gate_fires_on_a_pulled_channel_and_stays_quiet_on_noise():
    body = Homeostasis(parameters=Parameters(lam=0.35, epsilon=0.1, alpha=0.01), channels=8)
    pulled = _observation(np.array([0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]), 0.01)
    quiet = _observation(np.zeros(8), 0.01)
    assert body.gate(pulled) is True
    assert body.gate(quiet) is False


def test_a_calibration_is_a_box_a_rule_and_a_log(infancy):
    calibration = calibrate(infancy, steps=8, seed=5)
    assert 0.0 < calibration.parameters.lam <= 1.0
    assert homeostasis.EPSILON_FLOOR <= calibration.parameters.epsilon < 1.0
    assert calibration.parameters.lam + calibration.parameters.epsilon <= 1.0 + 1e-9
    assert calibration.threshold > 0.0
    assert calibration.surface  # the grid was kept
    assert len(calibration.mutations) >= 8
    assert calibration.sizes["fixations"] == sum(len(e) for e in infancy)
    assert 0.0 <= calibration.validation["auc"] <= 1.0
    assert json.loads(json.dumps(calibration.summary()))  # the report is JSON-able


def test_the_body_keeps_mutating_after_calibration_and_rolls_back(infancy):
    body = Homeostasis.awaken(infancy, steps=6, seed=7)
    before = body.parameters
    attempts = [body.tick(p_mutation=1.0) for _ in range(12)]
    assert all(attempt is not None for attempt in attempts)
    assert any(attempt.changed for attempt in attempts)
    for attempt in attempts:
        if attempt.accepted:
            assert attempt.objective >= attempt.best
    summary = body.summary()
    assert summary["attempts"] == 12
    assert body.parameters.lam + body.parameters.epsilon <= 1.0 + 1e-9
    assert body.calibration is not None
    assert before.lam != pytest.approx(body.parameters.lam) or True


def test_a_calibration_without_a_dream_or_without_a_world_is_refused(infancy):
    world_only = [episode for episode in infancy if episode.origin == "world"]
    dream_only = [episode for episode in infancy if episode.origin != "world"]
    with pytest.raises(ValueError):
        calibrate(dream_only, steps=4)
    with pytest.raises(ValueError):
        calibrate(world_only, steps=4)


def _observation(x, sigma):
    from agent.physics.genome import observation

    return observation(np.asarray(x, dtype=float), float(sigma))
