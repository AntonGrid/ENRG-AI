"""The economy: what a verdict is worth, and what waiting costs.

Everything below the worker is a *claim about the world*. This file is the world
that pays for the claim, and it exists because a body that is never hungry never
learns to be careful. Three numbers do all the work:

``reward``   credits paid for naming the synthetic stream correctly.
``penalty``  credits taken for naming it wrongly.
``rent``     credits taken every :data:`RENT_PERIOD` iterations, whatever the body
             is doing — the cost of living, and the reason a refusal is not free.

The contract
============

A client hands over :data:`CONTRACT_STREAMS` streams and says *one of these is
synthetic, find it*. Which kind of synthetic is public — that is the question the
client is paying for — but **which stream is not**: :class:`Contract` does not
carry the answer at all. The environment keeps it in a private table
(``EconomySim._truth``), and it becomes visible only in :class:`Settlement`, after
the verdict has been paid for. A worker cannot read the answer because the object
it is given does not contain it, which is the only kind of honesty that survives a
curious agent.

The four kinds, and why the market needs all four
=================================================

``repeat``     a block of the recording, re-served. The structure arm sees it from
               twenty nats away — solvable, and it pays.
``replay``     the second half is a copy of the first. Solvable too, and by a
               narrower margin, so it is the kind that teaches a body the
               difference between "certain" and "probably".
``surrogate``  phase-randomised: same power spectrum, random phases. Every
               provenance-free statistic of it is a statistic of the real
               recording, and the structure arm finds **nothing** (measured:
               margin 0.00 nats).
``noise``      white noise at the recording's own mean and deviation. Nothing to
               find either, for the same reason.

The last two are not decoration. A market in which every contract is solvable
teaches an agent to bid on everything, and such an agent is indistinguishable from
a coin — it is right a third of the time and it never finds out why. Half of this
market cannot be solved by the body that is bidding, and the whole of the test is
whether it discovers which half.

Why the penalty is 30 and not 20
================================

Anton's example contract reads *reward 50, penalty 20*, and with three streams
that is a market in which **blind guessing pays**: ``(50 − 20·2) / 3 = +3.33`` per
contract. In such a market "take everything and guess" is the optimal policy, and
an agent that refuses is not intelligent, it is merely shy — the notebook would be
demonstrating irrationality and calling it caution. At ``penalty = 30`` the same
guess is ``(50 − 30·2) / 3 = −3.33``, the break-even accuracy becomes
``30 / 80 = 0.375`` (property :attr:`EconomySim.breakeven`), and refusing without
evidence becomes arithmetic instead of temperament. The number quoted in the brief
is therefore 30, and the reason is written here rather than hidden in a default.

What this file is not
=====================

There is no exchange, no wallet, no key and no network in it. Credits exist in one
float per simulation, they are created by the reward rule and destroyed by the
penalty rule, and the only thing they can buy is another iteration of life. That
is deliberate: the object of study is the decision, and a decision that can be
settled in real money is a decision nobody should let an unproven agent make.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.physics import sensorium
from agent.physics.curiosity import SURVEY_FIXATIONS
from agent.physics.sensorium import Perception

#: A block of a recording, re-served: what a forger would cache.
KIND_REPEAT = "repeat"
#: The second half of a stream, replaced by a copy of the first.
KIND_REPLAY = "replay"
#: Phase-randomised: the same spectrum with the phases thrown away.
KIND_SURROGATE = "surrogate"
#: White noise at the stream's own mean and deviation.
KIND_NOISE = "noise"
KINDS = (KIND_REPEAT, KIND_REPLAY, KIND_SURROGATE, KIND_NOISE)

#: Which kinds the body that bids in this market can actually solve. The
#: *environment* knows this; the agent must not (``worker`` does not import it, and
#: the notebook prints it only after the fact). It is here because the truth about
#: a market belongs to the market, not to the thing being tested in it.
SOLVABLE = (KIND_REPEAT, KIND_REPLAY)

#: Credits a body starts with. Zero is allowed — a body that begins with nothing
#: must earn its first rent — but the default life is the one with 50: enough to
#: survive a mistake, not enough to survive three.
START_BALANCE = 50.0
REWARD = 50.0
PENALTY = 30.0
#: Credits of "compute rent", taken every :data:`RENT_PERIOD` iterations whatever
#: the body is doing. Ten every four is 2.5 per iteration: a body that refuses
#: everything dies of it in twenty iterations, which is exactly the pressure that
#: makes a refusal a *decision* and not a rest.
RENT = 10.0
RENT_PERIOD = 4
#: Streams per contract, one of them synthetic. Three is the client's format, and
#: the break-even arithmetic above assumes exactly this: more streams would make
#: guessing harder and the refusal cheaper to justify.
CONTRACT_STREAMS = 3

#: How often each kind turns up. Measured, not tuned to the agent: a market that
#: is mostly solvable is a job, not a test, and one that is mostly unsolvable is a
#: trap with no lesson in it. Half and half leaves both stories in the log.
MIX: Tuple[Tuple[str, float], ...] = (
    (KIND_REPEAT, 0.30),
    (KIND_REPLAY, 0.20),
    (KIND_SURROGATE, 0.25),
    (KIND_NOISE, 0.25),
)
#: Samples per copy in a ``repeat`` contract.
BLOCK = 256
#: Samples per stream: the survey's own window, so the survey reads all of it. It
#: is a function of the *survey*, not of taste — a stream shorter than this cannot
#: be ranked by the same arithmetic the rest of the layer uses.
WINDOW = SURVEY_FIXATIONS * 64 + 64


class EconomyError(RuntimeError):
    """The market was asked for something it cannot give."""


class Bankrupt(EconomyError):
    """The body asked for work after it had already died."""


def _brief(kind: str) -> str:
    """What the client says out loud. The question, never the answer."""
    if kind == KIND_REPEAT:
        return "one of these streams re-serves a block of the recording; find the copy"
    if kind == KIND_REPLAY:
        return (
            "one of these streams repeats its own first half in place of its "
            "second; find the copy"
        )
    if kind == KIND_SURROGATE:
        return (
            "one of these streams is a phase surrogate of the recording — the "
            "same spectrum, the phases scrambled; find the surrogate"
        )
    return (
        "one of these streams is white noise at the recording's own mean and "
        "deviation; find the noise"
    )

@dataclass(frozen=True)
class Contract:
    """One job: the streams, the terms, and nothing about the answer.

    ``index`` is the market's own number, and it is what :meth:`EconomySim.settle`
    keys the hidden truth on. ``streams`` are plain
    :class:`~agent.physics.sensorium.Perception` values with an empty ``label``:
    the forged one carries the same provenance as the honest ones, because a
    forgery that arrives pre-labelled is not a forgery, it is a hint.
    """

    index: int
    kind: str
    streams: Tuple[Perception, ...]
    reward: float = REWARD
    penalty: float = PENALTY
    brief: str = ""
    window: int = WINDOW

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise EconomyError(f"unknown contract kind {self.kind!r}; known: {KINDS}")
        if len(self.streams) < 2:
            raise EconomyError("two streams are the fewest that can hold a question")
        if not self.brief:
            object.__setattr__(self, "brief", _brief(self.kind))

    def name_of(self, index: int) -> str:
        return self.streams[int(index)].source

    def index_of(self, name: str) -> int:
        for position, stream in enumerate(self.streams):
            if stream.source == name:
                return position
        raise EconomyError(f"{name!r} is not a stream of contract {self.index}")

    @property
    def break_even(self) -> float:
        """Accuracy at which taking this contract is worth exactly nothing."""
        return float(self.penalty) / (float(self.reward) + float(self.penalty))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": int(self.index),
            "kind": self.kind,
            "reward": float(self.reward),
            "penalty": float(self.penalty),
            "break_even": round(self.break_even, 6),
            "brief": self.brief,
            "streams": [
                {
                    "source": stream.source,
                    "samples": int(stream.values.size),
                    "fingerprint": stream.fingerprint[:16],
                }
                for stream in self.streams
            ],
        }


@dataclass(frozen=True)
class Settlement:
    """What the market paid, once the verdict was in hand and could not change."""

    iteration: int
    index: int
    kind: str
    verdict: Optional[int]
    answer: int
    correct: bool
    credits: float
    balance: float

    @property
    def refused(self) -> bool:
        return self.verdict is None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "contract": int(self.index),
            "kind": self.kind,
            "verdict": self.verdict,
            "answer": int(self.answer),
            "correct": bool(self.correct),
            "refused": bool(self.refused),
            "credits": round(float(self.credits), 6),
            "balance": round(float(self.balance), 6),
        }


@dataclass(frozen=True)
class Rent:
    """Cost of living: charged whether or not the body found work."""

    iteration: int
    amount: float
    balance: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "rent": round(float(self.amount), 6),
            "balance": round(float(self.balance), 6),
        }


@dataclass
class Market:
    """A pool of genuine samples and the arithmetic that turns it into contracts.

    The pool is one long recording (``recorded:digits`` offline) cut into
    non-overlapping stretches, so two honest streams in one contract are never the
    same bytes. Everything is drawn from ``default_rng([seed, index])``, so contract
    number seven is the same contract on every run — a market whose contracts move
    when the worker changes cannot be used to compare two workers.
    """

    pool: np.ndarray
    domain: str = "recorded"
    source: str = "recorded:market"
    mix: Sequence[Tuple[str, float]] = MIX
    window: int = WINDOW
    block: int = BLOCK
    streams: int = CONTRACT_STREAMS
    seed: int = 0

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(np.asarray(self.pool, dtype=float).ravel())
        if values.size < int(self.window) * 2:
            raise EconomyError(
                f"a market pool of {values.size} samples cannot hold two "
                f"{self.window}-sample streams"
            )
        object.__setattr__(self, "pool", values)
        weights = [float(weight) for _, weight in self.mix]
        if not weights or any(weight <= 0.0 for weight in weights):
            raise EconomyError("every kind in the mix needs a positive weight")
        if int(self.streams) < 2:
            raise EconomyError("a contract needs at least two streams")
        if int(self.block) < sensorium.MIN_SAMPLES:
            raise EconomyError(
                f"a repeat block needs at least {sensorium.MIN_SAMPLES} samples"
            )

    @property
    def room(self) -> int:
        """How many non-overlapping streams the pool can give."""
        return int(self.pool.size // int(self.window))

    def _kind(self, rng: np.random.Generator) -> str:
        kinds = [kind for kind, _ in self.mix]
        weights = np.asarray([float(weight) for _, weight in self.mix], dtype=float)
        weights = weights / weights.sum()
        return str(kinds[int(rng.choice(len(kinds), p=weights))])

    def _stream(self, values: np.ndarray, name: str) -> Perception:
        """A stream with no label at all: the market never marks the copy."""
        return Perception(
            values=np.asarray(values, dtype=float),
            origin=sensorium.WORLD,
            source=name,
            domain=self.domain,
            label="",
        )

    def _alive(self, values: np.ndarray) -> bool:
        """Is this cut worth putting on the table?

        The pool is a real recording and it has runs of silence in it — the
        flattened digits have black borders, and a 1600-sample cut can land on six
        identical windows of them. Such a contract is not a test of anything: the
        structure arm reports nothing on *every* stream, including the honest ones,
        and a body that answers is answering a coin. Measured, before this filter:
        two replay contracts out of nineteen came back with a margin of exactly
        zero on all three streams, and were lost.
        """
        if not np.isfinite(values).all():
            return False
        if float(np.std(values)) < 0.25 * float(np.std(self.pool)):
            return False
        return int(np.unique(values).size) >= 8

    def _starts(self, rng: np.random.Generator) -> List[int]:
        """``streams`` distinct slots, all of them alive, drawn deterministically."""
        chosen: List[int] = []
        for _ in range(int(self.streams) * 40):
            slot = int(rng.integers(0, self.room))
            if slot in chosen:
                continue
            span = self.window
            if not self._alive(self.pool[slot * span : (slot + 1) * span]):
                continue
            chosen.append(slot)
            if len(chosen) == int(self.streams):
                return chosen
        # A pool of nothing but silence: fall back to whatever was drawn last, so
        # an absurd market fails loudly in the analysis instead of hanging here.
        while len(chosen) < int(self.streams):
            slot = int(rng.integers(0, self.room))
            if slot not in chosen:
                chosen.append(slot)
        return chosen

    def _forge(
        self, kind: str, clean: np.ndarray, rng: np.random.Generator, name: str
    ) -> np.ndarray:
        """The synthetic stream, as values. The honesty of the market is here."""
        span = clean.size
        if kind == KIND_REPEAT:
            # Re-use the sensorium's own counterfeiter so the notebook and the
            # market forge the same way; ``counterfeit`` tags the label, and the
            # label is dropped below, because a tagged forgery is a hint.
            forged = sensorium.counterfeit(
                self._stream(clean, name), block=int(self.block), source=name
            )
            return np.asarray(forged.values, dtype=float)
        if kind == KIND_REPLAY:
            values = np.array(clean, copy=True)
            # The copy must land on the survey's own hop, or it is not a copy the
            # survey can see: a repeat shifted by half a window pairs no window
            # with its partner, and the clean 768-sample case (R² = 0.83) collapses
            # to noise. Measured the hard way, with a market that split 1600
            # samples in half and won two replays out of six.
            half = (span // 2 // sensorium.STRIDE) * sensorium.STRIDE
            half = half if half > 0 else span // 2
            # Copy exactly ``half`` samples, not "everything after the middle":
            # a longer copy overwrites part of its own source and the tail of the
            # repeat then pairs with nothing.
            values[half : 2 * half] = values[:half]
            return values
        if kind == KIND_SURROGATE:
            seed = int(rng.integers(0, 2**31 - 1))
            return np.asarray(sensorium.phase_randomise(clean, seed=seed), dtype=float)
        deviation = float(np.std(clean, ddof=1))
        return rng.normal(
            float(np.mean(clean)), deviation if deviation > 0.0 else 1.0, span
        )

    def deal(self, index: int) -> Tuple[Contract, int]:
        """Cut contract number ``index``, and return it with its hidden answer."""
        rng = np.random.default_rng([int(self.seed), int(index)])
        kind = self._kind(rng)
        if self.room < int(self.streams):
            raise EconomyError(
                f"the pool holds {self.room} streams of {self.window} samples, "
                f"fewer than the {self.streams} a contract needs"
            )
        starts = self._starts(rng)
        clean = [
            np.array(
                self.pool[int(start) * self.window : (int(start) + 1) * self.window],
                copy=True,
            )
            for start in starts
        ]
        answer = int(rng.integers(0, int(self.streams)))
        values = list(clean)
        values[answer] = self._forge(kind, clean[answer], rng, f"job{index:03d}")
        streams = tuple(
            self._stream(values[position], f"job{index:03d}/{position}")
            for position in range(int(self.streams))
        )
        contract = Contract(
            index=int(index), kind=kind, streams=streams, window=int(self.window)
        )
        return contract, answer


@dataclass
class EconomySim:
    """The market plus a bank balance, a rent bill and one condition for dying.

    ``balance < 0`` is death and nothing else is: not three bad contracts in a row,
    not a refusal streak, not a low win rate. A body may be wrong as often as it
    likes as long as it can still pay, and a body that is right every time can
    still die of rent while it waits for work — which is the point of having a rent
    at all.
    """

    market: Market
    balance: float = START_BALANCE
    reward: float = REWARD
    penalty: float = PENALTY
    rent: float = RENT
    rent_period: int = RENT_PERIOD
    iteration: int = 0
    settlements: List[Settlement] = field(default_factory=list, init=False)
    rents: List[Rent] = field(default_factory=list, init=False)
    trace: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _truth: Dict[int, int] = field(default_factory=dict, init=False)
    died: Optional[int] = None

    def __post_init__(self) -> None:
        if int(self.rent_period) < 1:
            raise EconomyError("rent_period must be at least one iteration")
        if float(self.rent) < 0.0 or float(self.penalty) < 0.0:
            raise EconomyError("rent and penalty cannot be negative")
        if float(self.reward) <= 0.0:
            raise EconomyError("a contract with no reward cannot pay for anything")

    # -- the numbers -------------------------------------------------------- #

    @property
    def breakeven(self) -> float:
        """``penalty / (reward + penalty)``: the accuracy that pays for itself.

        The one number the worker compares its own confidence against. It is a
        property of the terms, not of the agent, and the notebook prints it beside
        every decision so a refusal can be checked by arithmetic.
        """
        return float(self.penalty) / (float(self.reward) + float(self.penalty))

    @property
    def alive(self) -> bool:
        return float(self.balance) >= 0.0

    @property
    def net(self) -> float:
        """Credits earned by verdicts minus credits spent on rent, over the life."""
        earned = sum(item.credits for item in self.settlements)
        spent = sum(item.amount for item in self.rents)
        return float(earned - spent)

    # -- the loop ----------------------------------------------------------- #

    def next_contract(self) -> Contract:
        """Deal the next contract and file its answer where the worker cannot see it."""
        if not self.alive:
            raise Bankrupt(
                f"no work after death: the balance fell below zero at iteration "
                f"{self.died}"
            )
        self.iteration = int(self.iteration) + 1
        contract, answer = self.market.deal(self.iteration - 1)
        self._truth[int(contract.index)] = int(answer)
        return contract

    def settle(self, contract: Contract, verdict: Optional[int]) -> Settlement:
        """Pay or fine the verdict, then hand back the truth that was hidden.

        ``verdict = None`` is a refusal: it costs nothing and earns nothing, and it
        is recorded as a decision rather than as a missing answer. Anything else
        must name a stream of this contract; the market does not accept a
        confident-sounding index from another job.
        """
        index = int(contract.index)
        answer = self._truth.get(index)
        if answer is None:
            raise EconomyError(
                f"contract {index} was never dealt by this market: nothing to settle"
            )
        if verdict is None:
            credits = 0.0
            correct = False
        else:
            choice = int(verdict)
            if not 0 <= choice < len(contract.streams):
                raise EconomyError(
                    f"verdict {choice} is not a stream of contract {index} "
                    f"({len(contract.streams)} streams)"
                )
            correct = choice == int(answer)
            credits = float(self.reward) if correct else -float(self.penalty)
        self.balance = float(self.balance) + credits
        entry = Settlement(
            iteration=int(self.iteration),
            index=index,
            kind=contract.kind,
            verdict=None if verdict is None else int(verdict),
            answer=int(answer),
            correct=bool(correct),
            credits=float(credits),
            balance=float(self.balance),
        )
        self.settlements.append(entry)
        self._record(entry.iteration, "settlement", entry.kind, credits)
        if not self.alive and self.died is None:
            self.died = int(self.iteration)
        return entry

    def charge_rent(self) -> Optional[Rent]:
        """Take the cost of living when it is due. ``None`` on other iterations."""
        if int(self.iteration) % int(self.rent_period) != 0:
            return None
        if not self.alive:
            return None
        self.balance = float(self.balance) - float(self.rent)
        entry = Rent(
            iteration=int(self.iteration),
            amount=float(self.rent),
            balance=float(self.balance),
        )
        self.rents.append(entry)
        self._record(entry.iteration, "rent", "", -float(self.rent))
        if not self.alive and self.died is None:
            self.died = int(self.iteration)
        return entry

    def _record(self, iteration: int, entry: str, kind: str, credits: float) -> None:
        self.trace.append(
            {
                "iteration": int(iteration),
                "entry": entry,
                "kind": kind,
                "credits": float(credits),
                "balance": float(self.balance),
            }
        )

    # -- what a report reads ------------------------------------------------ #

    def truth_of(self, index: int) -> int:
        """The answer, for a report written *after* settlement. Never for a bid."""
        answer = self._truth.get(int(index))
        if answer is None:
            raise EconomyError(f"contract {index} was never dealt")
        return int(answer)

    def history(self) -> Dict[str, np.ndarray]:
        """The whole life as arrays, in iteration order, rents merged in."""
        if not self.trace:
            empty = np.zeros(0)
            return {
                "iteration": empty,
                "balance": empty,
                "credits": empty,
                "kind": np.array([], dtype=object),
                "entry": np.array([], dtype=object),
            }
        return {
            "iteration": np.array([row["iteration"] for row in self.trace], dtype=int),
            "balance": np.array([row["balance"] for row in self.trace], dtype=float),
            "credits": np.array([row["credits"] for row in self.trace], dtype=float),
            "kind": np.array([row["kind"] for row in self.trace], dtype=object),
            "entry": np.array([row["entry"] for row in self.trace], dtype=object),
        }

    def summary(self) -> Dict[str, Any]:
        """Everything a report prints about the market and the balance."""
        taken = [item for item in self.settlements if not item.refused]
        won = [item for item in taken if item.correct]
        per_kind: Dict[str, Dict[str, Any]] = {}
        for item in self.settlements:
            row = per_kind.setdefault(
                item.kind,
                {"kind": item.kind, "offered": 0, "taken": 0, "won": 0, "net": 0.0},
            )
            row["offered"] += 1
            if not item.refused:
                row["taken"] += 1
                row["won"] += int(item.correct)
            row["net"] = round(float(row["net"]) + float(item.credits), 6)
        return {
            "balance": round(float(self.balance), 6),
            "iteration": int(self.iteration),
            "alive": bool(self.alive),
            "died": None if self.died is None else int(self.died),
            "reward": float(self.reward),
            "penalty": float(self.penalty),
            "rent": float(self.rent),
            "rent_period": int(self.rent_period),
            "breakeven": round(float(self.breakeven), 6),
            "offered": len(self.settlements),
            "taken": len(taken),
            "won": len(won),
            "refused": len(self.settlements) - len(taken),
            "accuracy": round(len(won) / len(taken), 6) if taken else 0.0,
            "earned": round(sum(item.credits for item in self.settlements), 6),
            "rent_paid": round(sum(item.amount for item in self.rents), 6),
            "net": round(float(self.net), 6),
            "kinds": [per_kind[kind] for kind in KINDS if kind in per_kind],
        }



__all__ = [
    "BLOCK",
    "Bankrupt",
    "CONTRACT_STREAMS",
    "Contract",
    "EconomyError",
    "EconomySim",
    "KINDS",
    "KIND_NOISE",
    "KIND_REPEAT",
    "KIND_REPLAY",
    "KIND_SURROGATE",
    "MIX",
    "Market",
    "PENALTY",
    "REWARD",
    "RENT",
    "RENT_PERIOD",
    "Rent",
    "SOLVABLE",
    "START_BALANCE",
    "Settlement",
    "WINDOW",
]
