"""The market: what a verdict is worth, what waiting costs, and who may know what.

Three things a simulated economy can get wrong, and this file asserts all three:

* **the hidden answer stays hidden.** A :class:`~agent.physics.economy_sim.Contract`
  has no field, no key and no label that says which stream is the fake; the truth
  lives in the environment and surfaces only in the settlement, after the verdict
  has been paid for. A leak here would make every result below meaningless, because
  the worker would be reading the answer rather than computing it.
* **the wallet is arithmetic.** A win is ``+reward``, a wrong answer ``−penalty``, a
  refusal exactly zero, and the rent arrives on its schedule whether or not the body
  found work. Death is one condition and only one: a negative balance.
* **the terms make refusal rational.** At 50/30 and three streams a blind guess is
  worth ``(50 − 60)/3 = −3.33`` and the break-even accuracy is ``0.375`` — both
  computed here from the constants, so a change to either number has to face this
  test. With Anton's original 50/20 the same guess is ``+3.33``, and then an agent
  that refuses is not careful, it is wrong; the test states that difference rather
  than leaving it in a docstring.
"""
import dataclasses
import json

import numpy as np
import pytest

from agent.physics import sensorium as S
from agent.physics.economy_sim import (
    BLOCK,
    CONTRACT_STREAMS,
    KINDS,
    KIND_NOISE,
    KIND_REPEAT,
    KIND_REPLAY,
    KIND_SURROGATE,
    PENALTY,
    REWARD,
    SOLVABLE,
    Bankrupt,
    Contract,
    EconomyError,
    EconomySim,
    Market,
)


@pytest.fixture(scope="module")
def pool():
    """The whole digits recording: long enough for hundreds of contracts."""
    return S.listen_dataset("digits", limit=0)


@pytest.fixture(scope="module")
def market(pool):
    return Market(pool=pool.values, seed=7)


def deal_of(market, kind, limit=200):
    """The first contract of a given kind, with its answer — for tests that need one."""
    for index in range(int(limit)):
        contract, answer = market.deal(index)
        if contract.kind == kind:
            return contract, answer
    raise AssertionError(f"no {kind} contract in {limit} deals: the mix has changed")


# -- the answer stays hidden ------------------------------------------------ #

def test_a_contract_carries_the_question_and_not_the_answer(market):
    for index in range(4):
        contract, answer = market.deal(index)
        names = {field.name for field in dataclasses.fields(contract)}
        words = ("answer", "truth", "fake", "synthetic")
        assert not any(word in name for name in names for word in words)
        blob = json.dumps(contract.as_dict())
        assert "answer" not in blob and "truth" not in blob
        assert all(stream.label == "" for stream in contract.streams)
        assert all(stream.is_world for stream in contract.streams)
        assert 0 <= answer < len(contract.streams) == CONTRACT_STREAMS


def test_the_market_deals_the_same_contract_twice_and_fresh_streams_between(market):
    first, answer_a = market.deal(3)
    again, answer_b = market.deal(3)
    assert answer_a == answer_b
    assert [s.fingerprint for s in first.streams] == [
        s.fingerprint for s in again.streams
    ]
    other, _ = market.deal(4)
    assert {s.fingerprint for s in other.streams}.isdisjoint(
        {s.fingerprint for s in first.streams}
    )


def test_the_forgery_is_where_the_answer_says_it_is(market):
    repeat, answer = deal_of(market, KIND_REPEAT)
    forged = np.asarray(repeat.streams[answer].values, dtype=float)
    block = int(market.block)
    assert np.array_equal(forged[block : 2 * block], forged[:block])
    for position, stream in enumerate(repeat.streams):
        if position != answer:
            assert not np.array_equal(
                np.asarray(stream.values)[block : 2 * block],
                np.asarray(stream.values)[:block],
            )

    replay, answer = deal_of(market, KIND_REPLAY)
    forged = np.asarray(replay.streams[answer].values, dtype=float)
    half = (forged.size // 2 // S.STRIDE) * S.STRIDE
    assert half > 0 and 2 * half <= forged.size
    assert np.array_equal(forged[half : 2 * half], forged[:half])


def test_the_unsolvable_kinds_are_not_smuggled_in_as_solvable(market):
    """The market's own declaration must match what it deals, kind by kind."""
    assert set(SOLVABLE) == {KIND_REPEAT, KIND_REPLAY}
    assert set(KINDS) == {KIND_REPEAT, KIND_REPLAY, KIND_SURROGATE, KIND_NOISE}
    seen = {market.deal(index)[0].kind for index in range(60)}
    assert seen == set(KINDS)



# -- the wallet is arithmetic ----------------------------------------------- #

def test_the_wallet_pays_fines_and_refuses_exactly(market):
    sim = EconomySim(market=market, balance=50.0, rent=0.0)
    passed = sim.settle(sim.next_contract(), None)       # a refusal, to open the books
    assert passed.refused and passed.credits == 0.0
    assert sim.balance == pytest.approx(50.0)

    contract = sim.next_contract()
    correct = sim.settle(contract, sim.truth_of(contract.index))
    assert correct.correct and correct.credits == pytest.approx(REWARD)

    contract = sim.next_contract()
    wrong = sim.settle(contract, (sim.truth_of(contract.index) + 1) % CONTRACT_STREAMS)
    assert not wrong.correct and wrong.credits == pytest.approx(-PENALTY)
    assert sim.balance == pytest.approx(50.0 + REWARD - PENALTY)


def test_rent_is_taken_on_the_schedule_and_nowhere_else(market):
    sim = EconomySim(market=market, balance=50.0, rent=10.0, rent_period=4)
    charged = []
    for _ in range(8):
        sim.settle(sim.next_contract(), None)
        entry = sim.charge_rent()
        if entry is not None:
            charged.append(entry.iteration)
    assert charged == [4, 8]
    assert sim.balance == pytest.approx(30.0)
    assert sim.net == pytest.approx(-20.0)


def test_a_negative_balance_is_death_and_nothing_else_is(market):
    sim = EconomySim(market=market, balance=25.0, rent=0.0)
    contract = sim.next_contract()
    wrong = (sim.truth_of(contract.index) + 1) % CONTRACT_STREAMS
    sim.settle(contract, wrong)
    assert sim.balance == pytest.approx(-5.0)
    assert not sim.alive and sim.died == 1
    with pytest.raises(Bankrupt):
        sim.next_contract()


def test_rent_alone_can_kill_a_body_that_never_bets(market):
    sim = EconomySim(market=market, balance=9.0, rent=10.0, rent_period=1)
    sim.settle(sim.next_contract(), None)
    entry = sim.charge_rent()
    assert entry is not None and entry.balance == pytest.approx(-1.0)
    assert not sim.alive and sim.died == 1


def test_a_body_may_be_wrong_while_it_can_still_pay(market):
    """Death is a balance, not a streak: three fines in a row from a rich bank."""
    sim = EconomySim(market=market, balance=200.0, rent=0.0)
    for _ in range(3):
        contract = sim.next_contract()
        sim.settle(contract, (sim.truth_of(contract.index) + 1) % CONTRACT_STREAMS)
    assert sim.balance == pytest.approx(200.0 - 3 * PENALTY)
    assert sim.alive and sim.died is None



# -- the terms make refusal rational ---------------------------------------- #

def test_a_blind_guess_does_not_pay_at_these_terms(market):
    sim = EconomySim(market=market)
    chance = 1.0 / CONTRACT_STREAMS
    edge = chance * sim.reward - (1.0 - chance) * sim.penalty
    assert edge < 0.0, "with reward 50 and penalty 20 a coin is worth +3.33 a job"
    assert sim.breakeven == pytest.approx(sim.penalty / (sim.reward + sim.penalty))
    assert sim.breakeven > chance
    assert sim.breakeven == pytest.approx(0.375)


def test_anton_s_numbers_would_have_made_guessing_profitable():
    """The honest reason the penalty is 30: at 20 the refusals would be irrational."""
    reward, penalty, chance = 50.0, 20.0, 1.0 / 3.0
    edge = chance * reward - (1.0 - chance) * penalty
    assert edge > 0.0
    assert edge == pytest.approx(10.0 / 3.0, rel=1e-6)
    assert penalty / (reward + penalty) < chance


# -- settlements the market does not have to accept ------------------------- #

def test_settling_a_contract_that_was_never_dealt_is_an_error(market):
    draft = Contract(index=999, kind=KIND_NOISE, streams=market.deal(0)[0].streams)
    sim = EconomySim(market=market)
    with pytest.raises(EconomyError):
        sim.settle(draft, 0)


def test_a_verdict_outside_the_contract_is_an_error(market):
    sim = EconomySim(market=market)
    contract = sim.next_contract()
    with pytest.raises(EconomyError):
        sim.settle(contract, CONTRACT_STREAMS)
    with pytest.raises(EconomyError):
        sim.settle(contract, -1)
    assert sim.balance == pytest.approx(50.0)


def test_a_pool_too_small_to_cut_is_refused(pool):
    with pytest.raises(EconomyError):
        Market(pool=np.asarray(pool.values[:100], dtype=float))
    with pytest.raises(EconomyError):
        Market(pool=pool.values, mix=((KIND_REPEAT, 0.0), (KIND_NOISE, 1.0)))
    with pytest.raises(EconomyError):
        Market(pool=pool.values, block=S.MIN_SAMPLES - 1)


# -- what a report reads ---------------------------------------------------- #

def test_the_summary_the_history_and_the_log_agree(market):
    sim = EconomySim(market=market, balance=50.0)
    for _ in range(10):
        contract = sim.next_contract()
        if contract.kind in SOLVABLE:
            sim.settle(contract, sim.truth_of(contract.index))
        else:
            sim.settle(contract, None)
        sim.charge_rent()
    summary = sim.summary()
    history = sim.history()
    assert summary["offered"] == len(sim.settlements) == 10
    assert summary["taken"] + summary["refused"] == 10
    assert summary["won"] <= summary["taken"]
    assert len(history["iteration"]) == len(history["balance"]) == len(sim.trace)
    assert history["balance"][-1] == pytest.approx(summary["balance"])
    assert summary["net"] == pytest.approx(summary["earned"] - summary["rent_paid"])
    json.dumps(summary)
    assert {row["kind"] for row in summary["kinds"]} <= set(KINDS)
    for row in summary["kinds"]:
        assert row["won"] <= row["taken"] <= row["offered"]


def test_the_published_terms_are_the_ones_the_notebook_quotes():
    assert (REWARD, PENALTY) == (50.0, 30.0)
    assert CONTRACT_STREAMS == 3
    assert BLOCK == 256
