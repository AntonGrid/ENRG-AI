"""Entrain's arithmetic: the port against entrain's own Julia run.

The fixture is the output of ``julia 1.10.12`` reading
``entrain/julia/kuramoto/basic.jl``; the port must agree to the last bit on
every reading and on three steps of the full law — not merely on the order
parameter. The invariants below are the same criteria the book's own
``consistency.jl`` states, restated so a regression names the property it
broke instead of only a number.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from agent.physics import oscillator as osc

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "entrain_reference.json").read_text(
        encoding="utf-8"
    )
)
TOL = 1e-12


@pytest.fixture(scope="module")
def medium():
    inputs = FIXTURE["inputs"]
    return (
        np.array(inputs["omega0"], dtype=float),
        np.array(inputs["theta"], dtype=float),
        np.array(inputs["coupling"], dtype=float),
    )


# ── the port against Julia ──────────────────────────────────────────────────


def test_coherence_matches_julia(medium):
    _, theta, _ = medium
    assert osc.coherence(theta) == pytest.approx(
        FIXTURE["readings"]["coherence"], abs=TOL
    )


def test_partial_coherence_matches_julia(medium):
    _, theta, _ = medium
    expected = FIXTURE["readings"]["partial_coherence"]
    assert np.allclose(osc.partial_coherence(theta), expected, atol=TOL, rtol=0.0)


def test_resonance_and_spread_match_julia(medium):
    omega0, _, _ = medium
    assert osc.resonance(omega0) == pytest.approx(
        FIXTURE["readings"]["resonance"], abs=TOL
    )
    assert np.allclose(osc.spread(omega0), FIXTURE["readings"]["spread"], atol=TOL)


def test_chord_and_wave_match_julia(medium):
    omega0, theta, _ = medium
    assert np.allclose(
        osc.chord(omega0, theta, 0.7), FIXTURE["readings"]["chord"], atol=TOL
    )
    assert osc.wave(omega0, theta, 0.7) == pytest.approx(
        FIXTURE["readings"]["wave"], abs=TOL
    )


def test_unison_and_coincidence_match_julia(medium):
    omega0, theta, _ = medium
    assert np.allclose(osc.unison(omega0, omega0), FIXTURE["readings"]["unison"], atol=TOL)
    assert np.allclose(
        osc.coincidence(theta), FIXTURE["readings"]["coincidence"], atol=TOL
    )


def test_memory_goal_matches_julia(medium):
    _, _, coupling = medium
    for case in FIXTURE["memory_goal"]:
        value = osc.memory_goal(
            0.6,
            3,
            case["acc"],
            case["weight"],
            current=float(coupling[0, 1]),
        )
        assert value == pytest.approx(case["value"], abs=TOL)


def test_overlap_matches_julia():
    for case in FIXTURE["overlap"]:
        band = np.inf if case["band"] == "inf" else case["band"]
        assert float(osc.overlap(case["delta"], band)) == pytest.approx(
            case["value"], abs=TOL
        )


def test_three_steps_of_the_law_match_julia(medium):
    omega0, theta0, coupling0 = medium
    omega, theta, coupling = omega0, theta0, coupling0
    for step in FIXTURE["steps"]:
        omega, theta, coupling = osc.step_state(
            omega,
            theta,
            coupling,
            omega0,
            link=0.6,
            degree=3,
            band=0.1,
            memory=0.5,
            damping=0.8,
            dt=0.01,
        )
        assert np.allclose(omega, step["omega"], atol=TOL)
        assert np.allclose(theta, step["theta"], atol=TOL)
        assert np.allclose(coupling, step["coupling"], atol=TOL)


# ── the book's own criteria ─────────────────────────────────────────────────


def test_coherence_is_one_when_everything_sounds_together():
    assert osc.coherence(np.zeros((4, 2))) == pytest.approx(1.0)
    assert osc.coherence(np.full((4, 2), 2.5)) == pytest.approx(1.0)


def test_coherence_is_zero_when_phases_are_spread_over_the_circle():
    theta = np.arange(8) * (2.0 * np.pi / 8.0)
    assert osc.coherence(theta) == pytest.approx(0.0, abs=1e-12)


def test_coherence_ignores_a_common_rotation():
    theta = np.array([[0.0, 0.3], [1.0, 1.6], [2.0, 2.7]])
    assert osc.coherence(theta) == pytest.approx(osc.coherence(theta + 1.234), abs=TOL)


def test_partial_coherence_reads_one_mode_at_a_time():
    theta = np.array([[0.0, 0.0], [0.0, np.pi], [0.0, 0.0]])
    values = osc.partial_coherence(theta)
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(1.0 / 3.0)


def test_spread_is_zero_for_a_single_mode_and_positive_for_a_chord():
    assert osc.spread(np.array([[1.0], [2.0], [3.0]])).tolist() == [0.0, 0.0, 0.0]
    values = osc.spread(np.array([[1.0, 2.0], [0.5, 0.7]]))
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.2)


def test_wave_is_the_mean_of_its_parts():
    omega = np.array([[1.0, 2.0], [0.9, 1.8]])
    theta = np.array([[0.0, 0.3], [1.0, 1.6]])
    assert osc.wave(omega, theta, 0.4) == pytest.approx(
        float(np.mean(osc.chord(omega, theta, 0.4))), abs=TOL
    )


def test_full_traction_of_the_memory_is_independent_of_the_degree():
    for degree in (3, 10, 100):
        goal = osc.memory_goal(0.6, degree, acc=1.0, weight=1.0)
        assert degree * goal == pytest.approx(0.6 + osc.MEMORY_REFERENCE)


def test_memory_goal_without_weight_keeps_the_link():
    assert osc.memory_goal(0.6, 4, acc=0.0, weight=0.0, current=0.42) == pytest.approx(
        0.42
    )


def test_memory_step_moves_by_eps_of_the_way():
    assert osc.memory_step(1.0, 2.0, epsilon=0.25, dt=1.0) == pytest.approx(1.25)


def test_step_state_does_not_touch_its_inputs():
    omega = np.array([[1.0]])
    theta = np.array([[0.0]])
    coupling = np.array([[0.5]])
    osc.step_state(omega, theta, coupling, memory=0.5, dt=0.1)
    assert omega.tolist() == [[1.0]]
    assert theta.tolist() == [[0.0]]
    assert coupling.tolist() == [[0.5]]


def test_a_zero_link_is_not_a_neighbour():
    omega = np.array([[1.0], [1.5]])
    theta = np.array([[0.0], [1.0]])
    moved, _, _ = osc.step_state(omega, theta, np.zeros((2, 2)), dt=0.1)
    assert np.allclose(moved, omega)
