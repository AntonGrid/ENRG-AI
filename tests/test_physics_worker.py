"""The worker: whether to bid, and the memory that teaches it which jobs to refuse.

This is the test of a *decision under uncertainty*, so it asserts the four things
such a decision can get wrong:

* **the certainty is measured, not asserted.** A contract with a re-served block
  must come back *named* with a stability of 1.0; a phase surrogate must come back
  with no suspect at all and a probability equal to chance. Those two cases are the
  whole reason the worker exists, and they are checked on the real sensorium.
* **the market's truth and the body's ability agree.** For every kind, the analysis
  must name a stream exactly when the market says the kind is solvable — a test of
  the market's honesty as much as of the detector's.
* **the arithmetic decides.** The bid is ``p >= penalty/(reward+penalty)``; a
  headline taken with a probability below the break-even, or a refusal above it,
  is a bug. The three rules of the brief are checked as three lanes.
* **memory is earned and pessimal.** A refusal is not evidence of incompetence, a
  single lucky guess must not reopen a hopeless kind, and the number the decision
  uses (``belief``) is always below the number the report prints (``trust``).
"""
import dataclasses
import json
import math

import numpy as np
import pytest

from agent.physics import economy_sim
from agent.physics import sensorium as S
from agent.physics.curiosity import UNSEEN_PLAYS, UCB
from agent.physics.economy_sim import (
    CONTRACT_STREAMS,
    KINDS,
    KIND_NOISE,
    KIND_REPEAT,
    KIND_SURROGATE,
    PENALTY,
    REWARD,
    Market,
)
from agent.physics.organism import Organism
from agent.physics.worker import (
    CERTAIN,
    LANE_EVIDENCE,
    LANE_REFUSAL,
    LANE_SCHOOLING,
    SCHOOLING,
    EpisodicMemory,
    Experience,
    Worker,
)


@pytest.fixture(scope="module")
def world():
    """One body and one market, shared: roughly two seconds on the offline sensorium."""
    body = Organism.hatch(offline=True, steps=20, seed=11)
    pool = S.listen_dataset("digits", limit=0)
    return {"body": body, "pool": pool, "market": Market(pool=pool.values, seed=7)}


def deal_of(market, kind, limit=200):
    """The first contract of a given kind, with its answer."""
    for index in range(int(limit)):
        contract, answer = market.deal(index)
        if contract.kind == kind:
            return contract, answer
    raise AssertionError(f"no {kind} contract in {limit} deals: the mix has changed")


def hire(world, *, seed=0, balance=50.0):
    return Worker.hire(
        world["body"], world["pool"], seed=seed, balance=balance, market=world["market"]
    )


def experience(**overrides):
    """A hand-made take, for the memory tests that need no sensorium at all."""
    fields = dict(
        iteration=1,
        index=0,
        kind=KIND_NOISE,
        lane=LANE_EVIDENCE,
        take=True,
        reason="",
        guess=0,
        answer=1,
        correct=False,
        credits=-PENALTY,
        balance=0.0,
        probability=0.3,
        trust=0.3,
        attention=1.0,
        breakeven=0.375,
        uncertainty=1.0,
        margin=0.0,
        saved=0.0,
    )
    fields.update(overrides)
    return Experience(**fields)


# -- the certainty is measured, not asserted -------------------------------- #

def test_a_repeat_is_named_and_a_surrogate_is_not(world):
    worker = hire(world)
    repeat, answer = deal_of(world["market"], KIND_REPEAT)
    analysis = worker.oracle.analyse(repeat)
    assert analysis.named and analysis.guess == answer
    assert analysis.certainty.stability == 1.0
    assert analysis.certainty.uncertainty == 0.0
    assert analysis.certainty.margin > 10.0
    assert analysis.certainty.probability > 0.99
    assert analysis.certainty.resonance >= 0.0

    surrogate, _ = deal_of(world["market"], KIND_SURROGATE)
    analysis = worker.oracle.analyse(surrogate)
    assert not analysis.named
    assert analysis.certainty.suspect is None
    assert analysis.certainty.uncertainty == 1.0
    assert analysis.certainty.probability == pytest.approx(1.0 / CONTRACT_STREAMS)


def test_the_market_declares_exactly_what_the_body_can_solve(world):
    """The four kinds, against the market's own ``SOLVABLE``: the heart of the file."""
    worker = hire(world)
    for kind in KINDS:
        contract, answer = deal_of(world["market"], kind)
        analysis = worker.oracle.analyse(contract)
        assert analysis.named == (kind in economy_sim.SOLVABLE), kind
        if analysis.named:
            assert analysis.guess == answer, kind



# -- the table: optimistic about what it has not tried ---------------------- #

def test_ucb_is_optimistic_and_the_optimism_decays():
    table = UCB(c=1.0)
    unseen = table.score("noise")
    assert unseen == pytest.approx(math.sqrt(math.log(2.0) / UNSEEN_PLAYS))
    table.observe("noise", 0.0)
    table.observe("noise", 0.0)
    assert table.count("noise") == 2
    assert table.mean("noise") == pytest.approx(0.0)
    assert table.score("noise") < unseen            # the bonus shrinks with plays

    for _ in range(20):
        table.observe("repeat", 1.0)
    assert table.best(["noise", "repeat"]) == "repeat"
    rows = table.table(["noise", "repeat"])
    assert [row["arm"] for row in rows] == ["repeat", "noise"]
    assert rows[0]["score"] > rows[1]["score"]
    json.dumps(table.as_dict())


def test_ucb_does_not_pretend_an_arm_is_worth_infinity():
    table = UCB()
    assert math.isfinite(table.score("never played"))
    assert table.best([]) is None
    assert table.mean("never played", default=0.25) == pytest.approx(0.25)


# -- memory: earned, smoothed, and pessimal --------------------------------- #

def test_memory_counts_only_what_was_taken_and_discounts_the_mean():
    memory = EpisodicMemory(prior=1.0 / CONTRACT_STREAMS)
    assert memory.trust(KIND_NOISE) == pytest.approx(1.0 / CONTRACT_STREAMS)
    assert memory.belief(KIND_NOISE) == 0.0            # nothing known, nothing bet

    memory.remember(experience())
    assert memory.plays(KIND_NOISE) == 1 and memory.wins(KIND_NOISE) == 0
    assert memory.trust(KIND_NOISE) < 1.0 / CONTRACT_STREAMS
    assert memory.belief(KIND_NOISE) < memory.trust(KIND_NOISE)

    # A refusal is not evidence of inability: it must not be counted as a take.
    memory.remember(experience(take=False, correct=False, credits=0.0))
    assert memory.plays(KIND_NOISE) == 1
    assert memory.trust(KIND_NOISE) == pytest.approx(
        (0.0 + 2.0 / CONTRACT_STREAMS) / 3.0
    )


def test_one_lucky_guess_can_flatter_the_mean_but_not_the_bound():
    memory = EpisodicMemory(prior=1.0 / CONTRACT_STREAMS)
    memory.remember(experience())
    memory.remember(experience(correct=True, credits=REWARD))
    assert memory.trust(KIND_NOISE) > 0.375            # the mean says "worth it"
    assert memory.belief(KIND_NOISE) < 0.375           # the bound says "no"
    assert memory.net(KIND_NOISE) == pytest.approx(REWARD - PENALTY)


def test_a_good_kind_keeps_its_trust_and_a_refusal_that_saved_is_counted():
    memory = EpisodicMemory(prior=1.0 / CONTRACT_STREAMS)
    for _ in range(12):
        memory.remember(experience(correct=True, credits=REWARD))
    assert memory.belief(KIND_NOISE) > 0.375
    assert memory.saved() == 0.0
    memory.remember(experience(take=False, guess=0, answer=0, credits=0.0))
    memory.remember(
        experience(take=False, guess=1, answer=0, credits=0.0, saved=PENALTY)
    )
    assert memory.saved() == pytest.approx(PENALTY)
    json.dumps(memory.as_dict())


# -- the three rules -------------------------------------------------------- #

def test_a_decided_repeat_is_taken_by_the_arithmetic(world):
    worker = hire(world)
    repeat, answer = deal_of(world["market"], KIND_REPEAT)
    decision = worker.should_take_contract(repeat)
    assert decision.take and decision.lane == LANE_EVIDENCE
    assert decision.named and decision.weight == 1.0
    assert decision.probability >= decision.breakeven
    assert decision.guess == answer
    assert "break-even" in decision.reason
    json.dumps(decision.as_dict())


def test_an_unstudied_kind_is_bought_once_and_the_lesson_is_affordable(world):
    worker = hire(world, balance=50.0)
    surrogate, _ = deal_of(world["market"], KIND_SURROGATE)
    decision = worker.should_take_contract(surrogate)
    assert decision.take and decision.lane == LANE_SCHOOLING
    assert decision.weight == 0.0                       # no evidence at all
    assert decision.probability < decision.breakeven    # a bet, and it says so
    assert decision.attention >= worker.explore_bar
    assert "lesson 1 of" in decision.reason


def test_evidence_can_trade_while_broke_and_a_guess_cannot(world):
    """A body with an answer may bid at any balance; a body without one may not."""
    broke = hire(world, balance=12.0)
    repeat, _ = deal_of(world["market"], KIND_REPEAT)
    decision = broke.should_take_contract(repeat)
    assert decision.take and decision.lane == LANE_EVIDENCE
    assert decision.balance < PENALTY                  # under the price of one fine
    assert decision.probability >= CERTAIN

    surrogate, _ = deal_of(world["market"], KIND_SURROGATE)
    decision = broke.should_take_contract(surrogate)
    assert not decision.take and decision.lane == LANE_REFUSAL
    assert "capital is kept" in decision.reason


def test_a_hopeless_kind_is_taught_twice_and_then_refused(world):
    worker = hire(world, balance=50.0)
    takes, refusals, last = 0, 0, None
    for _ in range(60):
        contract = worker.economy.next_contract()
        if contract.kind != KIND_SURROGATE:
            worker.economy.settle(contract, None)       # pass, without paying rent
            continue
        last = worker.work(contract)
        takes += int(last.take)
        refusals += int(last.refused)
        if refusals:
            break
    assert refusals == 1, "an unstudied kind was never refused"
    assert takes <= SCHOOLING, "more lessons were bought than the budget allows"
    assert last is not None and last.lane == LANE_REFUSAL
    assert any(
        phrase in last.reason
        for phrase in ("does not pay", "not worth a lesson", "capital is kept")
    )



# -- the work, and the books it keeps --------------------------------------- #

def test_work_settles_records_and_only_takes_move_the_table(world):
    worker = hire(world, balance=50.0)
    contract = worker.economy.next_contract()
    before = sum(worker.table.plays.values())
    item = worker.work(contract)
    assert worker.economy.settlements[-1].index == contract.index
    assert worker.memory.episodes[-1] is item
    assert sum(worker.table.plays.values()) == before + int(item.take)
    assert worker.economy.iteration == 1
    if item.refused:
        assert item.credits == 0.0
        assert item.answer == worker.economy.truth_of(contract.index)
    else:
        assert item.credits == (REWARD if item.correct else -PENALTY)


def test_a_refusal_is_priced_by_the_truth_that_was_hidden(world):
    """``saved`` is the counterfactual, and it must equal the fine and nothing else."""
    worker = hire(world, balance=200.0)
    last = None
    for _ in range(80):
        contract = worker.economy.next_contract()
        if contract.kind != KIND_SURROGATE:
            worker.economy.settle(contract, None)     # pass, without paying rent
            continue
        last = worker.work(contract)
        if last.refused:
            break
    assert last is not None and last.refused
    assert last.saved == pytest.approx(
        PENALTY if last.guess != last.answer else 0.0
    )
    saved = [item for item in worker.experiences if item.refused]
    assert worker.memory.saved() == pytest.approx(sum(item.saved for item in saved))


def test_the_life_keeps_its_books(world):
    worker = hire(world, balance=50.0)
    lived = worker.run(8)
    assert len(lived) == 8 == len(worker.experiences) == len(worker.decisions)
    assert worker.economy.summary()["offered"] == 8
    for item in lived:
        assert item.credits == pytest.approx(
            0.0 if item.refused else (REWARD if item.correct else -PENALTY)
        )
        assert item.balance >= 0.0 or worker.economy.died is not None
    report = worker.report()
    json.dumps(report)
    lanes = report["lanes"]
    assert lanes["evidence"] + lanes["schooling"] + lanes["refusal"] == 8
    assert len(report["log"]) == 8


def test_a_body_that_can_pay_nothing_and_earn_nothing_dies_of_rent(world):
    worker = hire(world, balance=9.0)
    worker.economy.rent = 10.0
    worker.economy.rent_period = 1
    lived = worker.run(5)
    assert len(lived) < 5
    assert worker.economy.died is not None and not worker.economy.alive
    assert worker.economy.balance < 0.0


def test_the_decision_does_not_depend_on_where_the_answer_sits(world):
    """The same streams must give the same decision whatever the hidden answer is."""
    worker = hire(world, balance=50.0)
    contract, _ = deal_of(world["market"], KIND_SURROGATE)
    twin = dataclasses.replace(contract, index=contract.index + 10_000)
    first = worker.should_take_contract(contract)
    second = worker.should_take_contract(twin)
    assert (first.take, first.lane, first.probability, first.guess) == (
        second.take,
        second.lane,
        second.probability,
        second.guess,
    )


def test_the_prose_says_what_the_numbers_say(world):
    worker = hire(world, balance=50.0)
    worker.run(6)
    text = worker.summary_text()
    summary = worker.economy.summary()
    assert f"{summary['taken']} taken" in text
    assert f"{summary['refused']} refused" in text
    assert "break-even" in text
    for kind in KINDS:
        if any(item.kind == kind for item in worker.experiences):
            assert kind in text
