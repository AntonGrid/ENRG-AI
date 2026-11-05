"""The organism: three phases, one body, and the rule it is judged by.

This is the integration test, and it is deliberately the one that can fail for a
scientific reason rather than a coding one. It asserts:

* **the three phases exist and are what they claim** — blind (no model, so the
  metric is a coin), learning, aware — and each of them hears both the world and
  the organism's own output;
* **the hard rule holds** — after the integration the mean score for real data is
  above 0.7 and for the organism's own simulations below 0.3. If a later change to
  the features, the prior or the ledger breaks that, this test fails and the
  notebook's claim is no longer true;
* **the honest floor is reported next to the result** — the same trace judged
  without the ledger (``without_ledger``), and the blind phase's metric-only AUC,
  so "how much of this is memory?" is a number in the report rather than a remark.

Everything runs on the ``--offline`` sensorium: sklearn's bundled recordings, no
network, no cache. That is the fallback the organism ships with, so the test needs
nothing an operator does not already have.
"""
import json

import numpy as np
import pytest

from agent.physics import organism as organism_module
from agent.physics.organism import Organism


@pytest.fixture(scope="module")
def hatched():
    """One body, hatched offline, living all three phases. Roughly three seconds."""
    body = Organism.hatch(offline=True, steps=60, seed=11)
    body.live(learning=40, aware=60, blind=30)
    return body


def test_hatching_hears_several_streams_and_records_what_it_missed(hatched):
    assert len(hatched.world) == len(organism_module.OFFLINE_SPECS)
    coverage = hatched.sensorium.coverage(hatched.world)
    assert coverage["ok"] is False  # offline cannot reach finance or physics
    assert any("blind on that side" in warning for warning in hatched.warnings)


def test_the_infancy_and_the_life_share_no_fixation(hatched):
    childhood = {
        (episode.source, reading.index)
        for episode in hatched.infancy
        for reading in episode.readings
    }
    later = [
        reading for episode in hatched.held_out for reading in episode.readings
    ]
    assert later  # something was held back for the life
    assert len(hatched.held_out) == len(hatched.world)
    assert len(later) == sum(len(episode) for episode in hatched.held_out)
    assert childhood  # and the childhood is not empty


def test_the_calibration_is_a_box_a_log_and_a_rule(hatched):
    calibration = hatched.homeostasis.calibration
    assert calibration is not None
    parameters = calibration.parameters
    assert 0.0 < parameters.lam <= 1.0
    assert parameters.lam + parameters.epsilon <= 1.0 + 1e-9
    assert parameters.alpha in {1e-1, 5e-2, 1e-2, 1e-3}
    assert calibration.threshold > 0.0
    assert calibration.alpha_rule in {"youden", "false-alarm-budget", "muted"}
    assert calibration.accepted_mutations + 1 <= calibration.attempted_mutations + 1


def test_the_three_phases_run_in_order_and_each_hears_both_sides(hatched):
    names = [phase.name for phase in hatched.phases]
    assert names == ["blind", "learning", "aware"]
    for phase in hatched.phases:
        rows = phase.summary()
        assert rows["world_fixations"] > 0, f"phase {rows['phase']} heard no world"
        assert rows["self_fixations"] > 0, f"phase {rows['phase']} made no dream"
    assert hatched.phases[0].last_step < hatched.phases[2].last_step


def test_the_blind_phase_has_no_metric_but_still_has_memory(hatched):
    blind = hatched.phases[0].summary()
    # no model means the two class models are identical, so the *metric* cannot
    # separate anything; what separates the piles is the ledger and only that
    assert blind["auc_resonance"] == pytest.approx(0.5, abs=0.02)
    assert blind["world_mean"] == pytest.approx(1.0 / (1.0 + np.exp(-0.0 - np.log(2.0))), abs=0.02)


def test_the_parameters_are_still_moving_while_the_body_lives(hatched):
    log = hatched.parameter_log
    assert log  # the body tried to change itself
    for row in log:
        assert set(row) >= {"step", "lam", "epsilon", "accepted", "objective"}
        assert row["lam"] + row["epsilon"] <= 1.0 + 1e-9


def test_the_hard_rule_holds_after_the_integration(hatched):
    verdict = hatched.acceptance()
    assert np.isfinite(verdict["world_mean"]) and np.isfinite(verdict["self_mean"])
    assert verdict["world_mean"] > 0.7, verdict
    assert verdict["self_mean"] < 0.3, verdict
    assert verdict["passed"] is True
    # and the same trace judged on statistics alone is reported next to it
    assert "without_ledger" in verdict
    assert 0.0 <= verdict["without_ledger"]["world_mean"] <= 1.0


def test_the_report_is_json_able_and_carries_the_genome(hatched):
    report = hatched.report()
    assert json.loads(json.dumps(report))  # no numpy scalar leaked into the report
    assert set(report["genome"]["fingerprint"]) == {"detector.py", "theorem.py"}
    assert report["coverage"]["domains"]
    assert report["phases"] and report["calibration"] and report["acceptance"]
    assert report["phantom"] and report["prior_sweep"]
    assert report["mirror"]["steps"] > 0


def test_the_report_can_be_written_and_the_phases_drawn(hatched, tmp_path):
    written = hatched.write(tmp_path / "report.json")
    assert written.is_file() and written.stat().st_size > 1000
    try:
        figure = hatched.plot(tmp_path / "phases.png")
    except ImportError:  # pragma: no cover - matplotlib is an optional extra
        pytest.skip("matplotlib is not installed")
    assert figure.is_file() and figure.stat().st_size > 1000


def test_a_body_that_hears_nothing_refuses_to_be_born():
    with pytest.raises(Exception):
        Organism.hatch(offline=False, specs=("file:/nonexistent/stream.csv",), steps=4)
