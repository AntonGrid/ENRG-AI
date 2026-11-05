"""The appetite: what a body asks about, and the arithmetic that makes it stop.

This is the test of a *decision*, not of a number, so it asserts the three things
a curiosity metric can get wrong:

* **the gate** — a stream that does not repeat itself is not called structured.
  Ten draws of white noise must all come back ``(0.0, 0, 1.0)``: without the exact
  null ``P(R²>r) = (1−r)^((n−2)/2)``, the maximum over the lags of an honest
  ``1/n`` looks like a discovery, and a body that believes it spends its life
  asking about noise. A repeat with a period must come back at the *fundamental*
  period, not at a multiple of it.
* **the product** — the score is the product of its four arms, the movement arm is
  a gate and not a ranking (a frozen feed is zeroed, a loud one saturates), and the
  budget of :data:`MAX_PROBES` questions is *arithmetically reachable*. That last
  one is a regression test with a history: with a fade of ``1/(1+looks)`` the third
  question needed a starting curiosity above 1.05 out of a score capped at 1, so
  :data:`KINDS` had a move in it that no organism could ever play.
* **the dialogue on a real body** — with the offline sensorium, a forgery and a
  dead sensor are surveyed beside the world. The forgery must be the only stream
  asked about, the questions must come in the order information is worth
  (``context``, ``next``, ``compare``), every answer must cost worth, and an answer
  must never be counted as new survey evidence about the stream — the window was
  drawn where the anomaly was, so counting it would double-count the very repeat
  that made the stream interesting.
"""

import json
import math

import numpy as np
import pytest

from agent.physics import sensorium as S
from agent.physics.curiosity import (
    GAIN_SCALE,
    KINDS,
    LOOK_COST,
    MAX_PROBES,
    PROBE_THRESHOLD,
    Curiosity,
    lindley_gain,
    reducibility,
    score,
    structure_of,
)
from agent.physics.organism import Organism


# -- the arithmetic -------------------------------------------------------- #

def test_the_score_is_the_product_of_its_arms_and_cannot_leave_the_unit_interval():
    assert score(1.0, 1.0, 0.0, 1.0) == 0.0           # no structure, nothing to ask
    assert score(0.0, 1.0, 8.0, 1.0) == 0.0           # a frozen feed is gated out
    assert score(1.0, 1.0, 8.0, 0.0) == 0.0           # nothing real, nothing to ask
    assert structure_of(0.0) == 0.0
    assert structure_of(1.0) < 25.0                   # floored, not infinite
    assert score(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0 - math.exp(-1.0 / GAIN_SCALE))
    assert score(9.0, 1.0, 9.0, 9.0) == pytest.approx(1.0 - math.exp(-9.0 / GAIN_SCALE))


def test_movement_is_a_gate_not_a_ranking():
    # A quiet stream is worth less than a loud one up to the survey's median, after
    # which the arm is clamped: this frame cannot tell noise from structure by
    # variance, so the arm must not pretend to rank them.
    quiet = score(0.25, 1.0, 0.9, 1.0)
    typical = score(1.0, 1.0, 0.9, 1.0)
    loud = score(4.0, 1.0, 0.9, 1.0)
    assert 0.0 < quiet < typical <= loud
    assert loud == pytest.approx(score(400.0, 1.0, 0.9, 1.0))   # saturates at 1.0


def test_the_budget_of_three_questions_is_arithmetically_reachable():
    # The regression: with a fade of 1/(1+looks) the third question needed a
    # starting curiosity above 1.05, so no organism could ever ask it.
    best = score(1.0, 1.0, 8.0, 1.0)
    assert best > 0.99
    for looks in range(MAX_PROBES):
        worth = score(1.0, 1.0, 8.0, 1.0, looks=looks)
        assert worth >= PROBE_THRESHOLD, f"question {looks + 1} could never be asked"
    assert score(1.0, 1.0, 8.0, 1.0, looks=1) == pytest.approx(best / (1.0 + LOOK_COST))
    assert score(1.0, 1.0, 8.0, 1.0, looks=2) == pytest.approx(best / (1.0 + 2 * LOOK_COST))


def test_the_reference_metric_is_stream_blind_where_the_score_is_not():
    # Lindley gain is a function of the same survey and cannot separate two streams
    # the body has looked at the same number of times: that is why the score is not
    # this number, and the notebook prints both.
    assert lindley_gain(1.0, 1.0) == lindley_gain(1.0, 1.0)
    assert lindley_gain(4.0, 1.0) > lindley_gain(1.0, 1.0)             # more learnable
    assert lindley_gain(1.0, 1.0, looks=4.0) < lindley_gain(1.0, 1.0)  # more looks
    assert score(1.0, 1.0, 0.0, 1.0) == 0.0


# -- the gate -------------------------------------------------------------- #

def test_white_noise_is_not_called_structure():
    for seed in range(10):
        rows = np.random.default_rng(seed).normal(size=(24, 7))
        assert reducibility(rows) == (0.0, 0, 1.0), f"seed {seed} invented structure"


def test_a_repeat_is_found_at_its_fundamental_period():
    block = np.random.default_rng(1).normal(size=(4, 7))
    rows = np.vstack([block] * 6)                      # 24 windows, period 4
    r2, lag, tail = reducibility(rows)
    assert lag == 4                                    # not 8, not 12, not 20
    assert r2 == pytest.approx(1.0)
    assert tail < 1e-90                                # an exact repeat is no accident


def test_a_trend_is_structure_and_is_reported_at_lag_one():
    # Autocorrelation that is real must be reported, or the gate is merely deaf.
    rng = np.random.default_rng(5)
    walk = np.zeros(24)
    for step in range(1, walk.size):
        walk[step] = 0.9 * walk[step - 1] + 0.3 * rng.normal()
    r2, lag, tail = reducibility(walk.reshape(-1, 1))
    assert lag == 1 and 0.3 < r2 < 0.9 and tail < 1e-3


def test_features_that_never_move_are_dropped_rather_than_divided_by():
    assert reducibility(np.zeros((24, 7))) == (0.0, 0, 1.0)
    rows = np.zeros((24, 7))
    rows[:, 3] = np.tile([1.0, 2.0, 3.0, 4.0], 6)      # one live feature, period 4
    r2, lag, _ = reducibility(rows)
    assert lag == 4 and r2 == pytest.approx(1.0)


def test_too_few_windows_is_a_shrug_and_not_a_guess():
    assert reducibility(np.zeros((3, 7))) == (0.0, 0, 1.0)
    assert reducibility([]) == (0.0, 0, 1.0)


def test_a_body_that_has_looked_at_nothing_asks_nothing():
    empty = Curiosity(posture=None, reference=1.0)
    assert empty.traits() == [] and len(empty) == 0
    assert empty.propose() is None
    assert empty.summary()["streams"] == 0
    assert json.loads(json.dumps(empty.summary()))["questions"] == []


# -- the dialogue on a real body ------------------------------------------- #

@pytest.fixture(scope="module")
def survey():
    """One body, one survey: the world, a forgery, and a sensor that has died.

    Everything the tests read is recorded here — the traits before the questions and
    after them — so no test depends on another test's order, and the numbers are the
    ones the notebook prints. Roughly two seconds on the offline sensorium.
    """
    body = Organism.hatch(offline=True, steps=20, seed=11)
    forger = S.counterfeit(body.world[0], block=256)
    dead = S.Perception(
        values=np.full(2048, 1.0),
        origin=S.WORLD,
        source="frozen",
        domain=body.world[0].domain,
        label="a frozen sensor",
    )
    before = {
        trait.stream: trait
        for trait in body.explore(extra=[forger, dead], ask=False)
    }
    questions = body.ask(answers=MAX_PROBES)
    forged = forger.source
    return {
        "before": before,
        "forged": forged,
        "questions": questions,
        "log": list(body.probe_log),
        "after": body.curiosity.trait(forged),
        "proposal": body.curiosity.propose(),
        "summary": body.curiosity.summary(),
        "report": body.report()["curiosity"],
        "text": body.summary_text(),
    }


def test_only_the_forgery_is_worth_asking_about(survey):
    asked = sorted(
        name for name, trait in survey["before"].items()
        if trait.curiosity >= PROBE_THRESHOLD
    )
    assert asked == [survey["forged"]]
    assert survey["before"][survey["forged"]].curiosity >= PROBE_THRESHOLD


def test_the_forgery_is_caught_by_its_repeat_and_by_nothing_else(survey):
    trait = survey["before"][survey["forged"]]
    assert trait.reducible > 0.99          # the block is re-served, exactly
    assert trait.lag == 4                  # 256 samples in 64-sample fixations
    assert trait.structure > 10.0          # nats: an exact repeat is enormous
    assert trait.reality > 0.5             # and its values are genuinely real
    assert not trait.blind


def test_a_dead_sensor_is_never_a_question(survey):
    dead = survey["before"]["frozen"]
    # A frozen feed is a flat line: its features are not *exactly* identical (the
    # standardiser still moves in the last bits), so the gate is "no movement to
    # speak of", and no structure at all means curiosity is exactly zero.
    assert dead.movement < 1e-9
    assert dead.structure == 0.0
    assert dead.curiosity == 0.0
    assert dead.looks == 0


def test_the_questions_are_spent_in_the_order_information_is_worth(survey):
    assert [probe.kind for probe in survey["questions"]] == list(KINDS)
    assert len(survey["questions"]) <= MAX_PROBES
    assert [entry["kind"] for entry in survey["log"]] == list(KINDS)
    assert all(entry["answered"] for entry in survey["log"])
    starts = [entry["start"] for entry in survey["log"]]
    assert len(set(starts)) == len(starts)   # three different places in the stream


def test_every_answer_costs_worth_and_the_dialogue_ends_by_arithmetic(survey):
    worth = [entry["curiosity"] for entry in survey["log"]]
    after = [entry["after"] for entry in survey["log"]]
    assert worth[0] >= PROBE_THRESHOLD
    assert all(later < earlier for earlier, later in zip(worth, after))
    assert survey["after"].curiosity < survey["before"][survey["forged"]].curiosity
    assert survey["after"].looks == MAX_PROBES
    assert survey["proposal"] is None        # three questions, then silence


def test_an_answer_is_evidence_about_a_place_not_about_the_stream(survey):
    # The probe's windows are drawn where the anomaly was. Counting them as survey
    # evidence would double-count the very repeat that raised the question, so the
    # number of windows the structure is computed from must not move.
    before = survey["before"][survey["forged"]]
    after = survey["after"]
    assert after.fixations == before.fixations
    assert after.reducible == pytest.approx(before.reducible)
    assert after.lag == before.lag


def test_the_summary_and_the_report_are_the_same_numbers_and_are_json(survey):
    assert survey["summary"] == survey["report"]
    assert survey["summary"]["streams"] == len(survey["before"])
    assert survey["summary"]["probes_asked"] == len(survey["questions"])
    assert survey["summary"]["reference"] > 0.0
    assert len(survey["summary"]["traits"]) == len(survey["before"])
    blob = json.dumps(survey["summary"])     # what a report, a table can hold
    assert len(blob) > 100


def test_the_prose_says_what_the_numbers_say(survey):
    # The organism prints its appetite; the forgery is named there, and the dead
    # sensor is not offered as something to ask about.
    block = survey["text"].split("CURIOSITY")[-1]
    assert survey["forged"] in block
    assert "asked" in block or "no question" in block
