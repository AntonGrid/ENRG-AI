"""The experiment: the physics against the MAD rules on the same declared truth.

The criterion is stated before the run (as in the book's journal): a producer
is *caught* when its detector flags it at least once in the reading window, and
precision/recall are computed against the declared injections.

Two results are asserted on purpose, and neither is a compliment:

* the physics catches the outage, the peak shift and the timestamp lie, flags
  nobody in a quiet park, and cannot see a level change — a factor does not
  move a phase;
* the hourly spike rule fires at the morning ramp of a *clean* park (the rule
  compares the latest report with the median of a history that is mostly
  night), which is why the daily variant is measured next to it.

If someone later fixes the level rules, these characterisation tests fail — and
that is the point: the table in ``benchmark.format_table`` must be re-measured,
not quietly inherited.
"""

import numpy as np
import pytest

from agent.physics.benchmark import (
    coherence_flags,
    evaluate,
    format_table,
    spike_mad_daily_flags,
)
from agent.physics.synthetic import build_scenario

pytest.importorskip("axis_core.ai.anomaly")  # the comparison needs the verifier


@pytest.fixture(scope="module")
def result():
    return evaluate(build_scenario())


@pytest.fixture(scope="module")
def clean_result():
    return evaluate(build_scenario(scenario="clean"))


def test_the_scenario_declares_four_changes():
    scenario = build_scenario()
    assert sorted(scenario.truth.values()) == sorted(
        ["clean"] * 4 + ["outage", "peak_shift", "clock_offset", "level_scale"]
    )
    assert {injection.kind for injection in scenario.injections} == {
        "outage",
        "peak_shift",
        "clock_offset",
        "level_scale",
    }


def test_coherence_catches_the_phase_and_silence_changes(result):
    data = result["detectors"]["coherence"]
    assert data["caught"] == ["dev-02", "dev-05", "dev-06"]
    assert data["false_alarms"] == []
    assert data["precision"] == 1.0
    assert data["recall"] >= 0.5


def test_coherence_is_blind_to_a_level_change_by_construction(result):
    row = next(r for r in result["devices"] if r["truth"] == "level_scale")
    assert row["flags"]["coherence"] is False
    assert row["score"] < 0.5
    assert row["reasons"] == []


def test_the_level_rule_catches_a_level_change_worth_its_cut():
    """The comparison is not rigged: the same rule does see a level change."""
    scenario = build_scenario()
    scenario.energy[7, 120:] = scenario.energy[7, 120:] / 1.6 * 2.5
    run = spike_mad_daily_flags(scenario)
    assert run.flags["dev-07"] is True
    assert [device for device, flag in run.flags.items() if flag] == ["dev-07"]


def test_the_hourly_spike_rule_fires_on_a_clean_park(clean_result):
    data = clean_result["detectors"]["spike_mad"]
    assert len(data["false_alarms"]) == len(clean_result["devices"])
    assert data["caught"] == []


def test_the_daily_rule_is_quiet_when_nothing_happened(clean_result):
    assert clean_result["detectors"]["spike_mad_daily"]["false_alarms"] == []


def test_coherence_is_quiet_in_a_quiet_park(clean_result):
    assert clean_result["detectors"]["coherence"]["false_alarms"] == []
    assert clean_result["detectors"]["coherence"]["caught"] == []
    assert all(not row["flags"]["coherence"] for row in clean_result["devices"])
    assert clean_result["park"]["coherence_mean"] > 0.95


def test_the_verdict_holds_across_seeds():
    for seed in (7, 11, 23, 42):
        run, _ = coherence_flags(build_scenario(seed=seed))
        caught = [device for device, flag in run.flags.items() if flag]
        assert caught == ["dev-02", "dev-05", "dev-06"], f"seed {seed}"


def test_scores_are_deterministic():
    first, _ = coherence_flags(build_scenario(seed=11))
    second, _ = coherence_flags(build_scenario(seed=11))
    assert first.flags == second.flags


def test_the_table_names_every_detector(result):
    text = format_table(result)
    for name in ("coherence", "spike_mad", "spike_mad_daily", "trend_mad"):
        assert name in text
    assert "park coherence" in text


def test_trend_mad_is_reported_not_hidden(result):
    # The live AI anomaly signal is in the table with its own numbers: on this
    # model it fires on producers that were not touched, and the benchmark says
    # so instead of leaving it out.
    data = result["detectors"]["trend_mad"]
    assert data["caught"]
    assert data["false_alarms"]
