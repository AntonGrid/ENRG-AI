"""The worker: a body that bids for work, and the memory that teaches it to bid.

The organism can hear, the curiosity engine can rank, and the economy can pay.
What is missing is the thing that decides *whether to bid at all*, and that is the
whole of this file. It is written as one rule and two of its consequences:

.. math::
    p_{\\text{hat}} = w \\cdot P_{\\text{evidence}} + (1-w) \\cdot T_{\\text{memory}}

``P_evidence``
    what this contract's own survey says: the probability that the stream it names
    is the synthetic one (``Certainty.probability``, from the structure margin).

``w`` = ``Certainty.stability``
    how much that answer survives being re-measured — the epistemic weight. When
    the survey names a repeat, ``w = 1`` and the memory does not get a vote. When
    the survey names nothing, ``w = 0`` and the only thing left is what this kind
    of job has paid before.

``T_memory``
    the Laplace-smoothed win rate of the *takes* this body has already made of the
    same kind (``EpisodicMemory.trust``). Losses on one kind lower the confidence in
    all of it, which is exactly what Anton asked for: a body that has been burned by
    "noisy sine" contracts stops believing in them.

and then one comparison, which is arithmetic rather than character:

.. math::
    \\text{take} \\iff p_{\\text{hat}} \\ge
    \\frac{\\text{penalty}}{\\text{reward} + \\text{penalty}}

The right-hand side is :attr:`EconomySim.breakeven` — 0.375 in this market. A coin
scores 1/3 and is refused; a decided repeat scores 1.0 and is taken; a marginal
answer scores what it scores, and the same number is printed beside the decision in
the log so that any refusal can be checked by hand.

The third rule: a lesson
========================

There is one case the arithmetic above cannot decide, and it is the one Anton's
brief names as an *investment*: a kind of job the body has never tried. Its trust
is the chance level and its evidence is nothing, so ``p_hat`` is exactly 1/3 and
the rule refuses — forever. Since the alternative is a body that never finds out
that a quarter of the market is solvable, a bounded budget of **paid lessons** is
declared: at most :data:`SCHOOLING` takes per kind, only while the balance can
absorb the fine, and only while the UCB table still says the kind is worth
optimism. That budget is why the balance curve has a learning phase at all, and it
is spent — visibly, in the log — rather than hidden in a random seed.

The oracle is a *name*, not a new model
=======================================

:class:`RealityOracle` is the body's own mirror plus its own appetite, wired
together so a contract can be answered in one call. It does not decide: it hands
back the ranking, the certainty, and the name of the suspect. Everything below it
is the arithmetic that was proved in ``curiosity``, ``self_model`` and
``homeostasis``; nothing here re-derives any of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.physics.curiosity import (
    SURVEY_FIXATIONS,
    Certainty,
    Curiosity,
    Trait,
    UCB,
)
from agent.physics.economy_sim import Contract, EconomySim
from agent.physics.sensorium import Episode, Sensorium
from agent.physics.self_model import SelfModel

#: Paid takes of one kind before the body stops learning about it with money.
SCHOOLING = 2
#: Confidence above which a body bids even when the fine could end it. A body with
#: evidence can trade while broke; a body without evidence cannot, and the survival
#: of an under-capitalised agent rests on this one number being honest.
CERTAIN = 0.9
#: UCB score below which a kind is no longer worth a lesson.
EXPLORE_BAR = 0.5
#: Pseudo-takes at the chance rate used to smooth a kind's record. Two, so that one
#: lucky guess cannot promote a hopeless kind and one loss cannot bury a good one.
PSEUDO = 2.0

LANE_EVIDENCE = "evidence"
LANE_SCHOOLING = "schooling"
LANE_REFUSAL = "refusal"
LANES = (LANE_EVIDENCE, LANE_SCHOOLING, LANE_REFUSAL)


@dataclass(frozen=True)
class Analysis:
    """One contract, read: the ranking, the certainty, and the name of the suspect.

    ``guess`` is always present — a body that bids must name *some* stream — while
    ``named`` says whether the survey's answer is evidence or merely the least bad
    option. Keeping both is what lets the log separate "I knew" from "I guessed",
    which is the only difference that matters to a wallet.
    """

    index: int
    kind: str
    names: Tuple[str, ...]
    traits: Tuple[Trait, ...]
    certainty: Certainty

    @property
    def guess(self) -> int:
        """Index of the stream this body would name, evidence or not."""
        if not self.traits:
            return 0
        return int(self.names.index(self.traits[0].stream))

    @property
    def named(self) -> bool:
        """True when the survey named a stream and the margin survived its null."""
        return self.certainty.decisive

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract": int(self.index),
            "kind": self.kind,
            "guess": int(self.guess),
            "named": bool(self.named),
            "certainty": self.certainty.as_dict(),
            "traits": [trait.as_dict() for trait in self.traits],
        }


@dataclass
class RealityOracle:
    """The body's answer to *which of these is not the world*, priced.

    Three proven pieces in one call: the sensorium cuts the survey, the mirror
    judges every fixation (``learn=False`` — re-hearing must not move the world
    model it is judging), and the appetite ranks the streams and says how much of
    the ranking is arithmetic.
    """

    sensorium: Sensorium
    mirror: SelfModel
    engine: Curiosity
    survey: int = SURVEY_FIXATIONS
    seed: int = 0

    def analyse(self, contract: Contract) -> Analysis:
        """Survey every stream of a contract, then rank them by their own structure."""
        self.engine.reset()
        episodes: List[Episode] = []
        moments = []
        for stream in contract.streams:
            episode = self.sensorium.fixate(stream, limit=int(self.survey))
            episodes.append(episode)
            moments.append(self.mirror.observe(episode, learn=False))
        traits = self.engine.survey(episodes, moments=moments)
        certainty = self.engine.certainty(seed=int(self.seed))
        return Analysis(
            index=int(contract.index),
            kind=contract.kind,
            names=tuple(stream.source for stream in contract.streams),
            traits=tuple(traits),
            certainty=certainty,
        )


@dataclass(frozen=True)
class Experience:
    """One iteration of a working life, with every number the decision was made on.

    ``saved`` is the counterfactual and it is only knowable *after* the market has
    paid: for a refusal, it is the fine that would have been charged had the body
    answered with the stream it was leaning towards. It is not a claim about what
    would have happened — it is the truth table replayed, and it is the only honest
    way to say "this refusal saved 30 credits".
    """

    iteration: int
    index: int
    kind: str
    lane: str
    take: bool
    reason: str
    guess: int
    answer: int
    correct: bool
    credits: float
    balance: float
    probability: float
    trust: float
    attention: float
    breakeven: float
    uncertainty: float
    margin: float
    saved: float = 0.0

    @property
    def refused(self) -> bool:
        return not bool(self.take)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "contract": int(self.index),
            "kind": self.kind,
            "lane": self.lane,
            "take": bool(self.take),
            "reason": self.reason,
            "guess": int(self.guess),
            "answer": int(self.answer),
            "correct": bool(self.correct),
            "credits": round(float(self.credits), 6),
            "balance": round(float(self.balance), 6),
            "probability": round(float(self.probability), 6),
            "trust": round(float(self.trust), 6),
            "attention": round(float(self.attention), 6),
            "breakeven": round(float(self.breakeven), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "margin": round(float(self.margin), 6),
            "saved": round(float(self.saved), 6),
        }


@dataclass
class EpisodicMemory:
    """What this body has been paid, by kind, and what it therefore believes.

    Deliberately small and deliberately *earned*: only the contracts that were
    actually taken count as evidence about ability. A refusal is not a failure and
    must not be remembered as one — a body that counts its own caution as
    incompetence talks itself out of the only work it can do.

    ``prior`` is the chance rate of the market (:class:`Contract` with three streams
    → 1/3) and it is used as a Laplace pseudo-count, so an empty memory says
    "chance" instead of "certain" in either direction.
    """

    prior: float = 1.0 / 3.0
    episodes: List[Experience] = field(default_factory=list)

    def remember(self, experience: Experience) -> None:
        self.episodes.append(experience)

    def of_kind(self, kind: str) -> List[Experience]:
        return [item for item in self.episodes if item.kind == kind]

    def takes(self, kind: str) -> List[Experience]:
        return [item for item in self.of_kind(kind) if item.take]

    def plays(self, kind: str) -> int:
        """How many times this kind was actually taken, not merely offered."""
        return len(self.takes(kind))

    def wins(self, kind: str) -> int:
        return sum(1 for item in self.takes(kind) if item.correct)

    def net(self, kind: str) -> float:
        return float(sum(item.credits for item in self.of_kind(kind)))

    def trust(self, kind: str) -> float:
        """Smoothed win rate of a kind: ``(wins + PSEUDO·prior) / (plays + PSEUDO)``.

        With two pseudo-takes: one loss leaves 0.17, one win 0.50, two wins 0.75 —
        so a single lucky guess cannot promote a hopeless kind past the 0.375
        break-even, and a single loss cannot bury a good one below it. The mean is
        what a *report* prints; the decision uses :meth:`belief`, which is lower.
        """
        plays = float(self.plays(kind))
        wins = float(self.wins(kind))
        return float((wins + PSEUDO * float(self.prior)) / (plays + PSEUDO))

    def belief(self, kind: str) -> float:
        """Lower bound on the win rate: ``mean − sqrt(p(1−p)/n)``, floored at zero.

        The mean is not what a body should bet on, because a mean built from four
        observations is itself uncertain — and the direction of the error matters:
        over-estimating your own skill costs a fine, under-estimating it costs a
        job. So the decision uses the pessimistic side, one standard error wide.

        This is what stops a *lucky guess* from reopening a hopeless kind: after
        two lessons and one fluke win the mean is 0.42, which is above the 0.375
        break-even, but the bound is 0.17 — and it is the bound that decides. With
        forty-two of forty-two repeats the same bound is 0.94, so a kind that has
        earned its trust keeps it.
        """
        mean = float(self.trust(kind))
        spread = float(math.sqrt(max(mean * (1.0 - mean), 1e-12)))
        count = float(self.plays(kind)) + float(PSEUDO)
        return float(max(0.0, mean - spread / math.sqrt(count)))

    def saved(self) -> float:
        """Credits the body did not lose by refusing, counted after the fact."""
        return float(sum(item.saved for item in self.episodes if item.refused))

    def as_dict(self) -> Dict[str, Any]:
        kinds = sorted({item.kind for item in self.episodes})
        return {
            "prior": round(float(self.prior), 6),
            "episodes": len(self.episodes),
            "saved": round(float(self.saved()), 6),
            "kinds": [
                {
                    "kind": kind,
                    "offered": len(self.of_kind(kind)),
                    "taken": self.plays(kind),
                    "won": self.wins(kind),
                    "trust": round(float(self.trust(kind)), 6),
                    "net": round(float(self.net(kind)), 6),
                }
                for kind in kinds
            ],
        }


@dataclass(frozen=True)
class Decision:
    """Whether to bid, why, and every number the answer was computed from.

    Nothing here is a verdict of taste: ``probability`` is the blend in the module
    docstring, ``breakeven`` is the market's own arithmetic, and ``lane`` says which
    of the three rules was played. A refusal that cannot be explained in these
    numbers is a refusal that was not a decision.
    """

    iteration: int
    index: int
    kind: str
    take: bool
    lane: str
    reason: str
    probability: float
    evidence: float
    trust: float
    attention: float
    breakeven: float
    balance: float
    margin: float
    uncertainty: float
    weight: float
    guess: int
    named: bool

    @property
    def refused(self) -> bool:
        return not bool(self.take)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "contract": int(self.index),
            "kind": self.kind,
            "take": bool(self.take),
            "lane": self.lane,
            "reason": self.reason,
            "probability": round(float(self.probability), 6),
            "evidence": round(float(self.evidence), 6),
            "trust": round(float(self.trust), 6),
            "attention": round(float(self.attention), 6),
            "breakeven": round(float(self.breakeven), 6),
            "balance": round(float(self.balance), 6),
            "margin": round(float(self.margin), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "weight": round(float(self.weight), 6),
            "guess": int(self.guess),
            "named": bool(self.named),
        }


@dataclass
class Worker:
    """A body with a market, a mirror and a memory: the decision, and its life.

    One method decides (:meth:`should_take_contract`), one method works
    (:meth:`work`), one method lives (:meth:`run`). The split is not cosmetic: the
    notebook has to be able to ask *"why did you refuse that one?"* without paying
    for an iteration, and that is only possible if the decision is a value.
    """

    economy: EconomySim
    oracle: RealityOracle
    memory: EpisodicMemory
    table: UCB = field(default_factory=UCB)
    schooling: int = SCHOOLING
    certain: float = CERTAIN
    explore_bar: float = EXPLORE_BAR
    decisions: List[Decision] = field(default_factory=list, init=False)
    experiences: List[Experience] = field(default_factory=list, init=False)

    # -- hiring ------------------------------------------------------------- #

    @classmethod
    def hire(
        cls,
        body: Any,
        pool: Any,
        *,
        seed: int = 0,
        balance: Optional[float] = None,
        market: Any = None,
        schooling: int = SCHOOLING,
    ) -> "Worker":
        """Dress an :class:`~agent.physics.organism.Organism` as a worker.

        ``pool`` is any :class:`~agent.physics.sensorium.Perception` long enough to
        cut contracts from — in practice the whole of ``recorded:digits``. The
        mirror, the sensorium and the posture are the body's own; this constructor
        adds no model, only a wallet and a memory.
        """
        from agent.physics.economy_sim import Market  # local: flat import graph

        engine = Curiosity(body.mirror.posture, prior=body.mirror.prior)
        oracle = RealityOracle(
            sensorium=body.sensorium, mirror=body.mirror, engine=engine, seed=int(seed)
        )
        if market is None:
            market = Market(pool=np.asarray(pool.values, dtype=float), seed=int(seed))
        economy = EconomySim(
            market=market,
            **({} if balance is None else {"balance": float(balance)}),
        )
        prior = 1.0 / float(market.streams)
        return cls(
            economy=economy,
            oracle=oracle,
            memory=EpisodicMemory(prior=prior),
            schooling=int(schooling),
        )

    # -- the decision ------------------------------------------------------- #

    def should_take_contract(self, contract: Contract) -> Decision:
        """Bid or refuse, and say why in numbers that can be checked by hand.

        The three rules from the brief are the three branches below, in their order:

        1. the survey is silent and the body cannot afford to learn — **refuse**,
           and the reason is the balance, not a mood;
        2. the survey is silent but the kind is unstudied and the lesson is
           affordable — **take**, because information is the only thing on sale;
        3. the survey names a stream, or the kind's record does — **take**, and the
           comparison against :attr:`EconomySim.breakeven` is what says so.
        """
        analysis = self.oracle.analyse(contract)
        sure = analysis.certainty
        kind = contract.kind
        trust = self.memory.trust(kind)
        belief = self.memory.belief(kind)
        attention = self.table.score(kind)
        # ``stability`` is zero when the survey named nothing, so this is the
        # weight of evidence: 1 means "the arithmetic decides, memory has no vote".
        weight = float(sure.stability)
        probability = weight * float(sure.probability) + (1.0 - weight) * float(belief)
        bar = float(self.economy.breakeven)
        balance = float(self.economy.balance)
        fine = float(contract.penalty)
        if probability >= bar:
            if balance - fine >= 0.0 or probability >= float(self.certain):
                take, lane = True, LANE_EVIDENCE
                reason = (
                    f"p={probability:.2f} >= break-even {bar:.2f}: margin "
                    f"{sure.margin:.2f} nats, memory {trust:.2f}, "
                    f"balance {balance:.0f} can pay a fine of {fine:.0f}"
                )
            else:
                take, lane = False, LANE_REFUSAL
                reason = (
                    f"p={probability:.2f} clears {bar:.2f} but not {self.certain:.2f}: "
                    f"a fine of {fine:.0f} would end a balance of {balance:.0f}"
                )
        else:
            take, lane, reason = self._school(
                contract, balance, trust, attention, probability, bar
            )
        decision = Decision(
            iteration=int(self.economy.iteration),
            index=int(contract.index),
            kind=kind,
            take=bool(take),
            lane=lane,
            reason=reason,
            probability=float(probability),
            evidence=float(sure.probability),
            trust=float(trust),
            attention=float(attention),
            breakeven=float(bar),
            balance=float(balance),
            margin=float(sure.margin),
            uncertainty=float(sure.uncertainty),
            weight=float(weight),
            guess=int(analysis.guess),
            named=bool(analysis.named),
        )
        self.decisions.append(decision)
        return decision

    def _school(
        self,
        contract: Contract,
        balance: float,
        trust: float,
        attention: float,
        probability: float,
        bar: float,
    ) -> Tuple[bool, str, str]:
        """The investment rule: buy information only where it is still for sale."""
        kind = contract.kind
        plays = self.memory.plays(kind)
        if plays >= int(self.schooling):
            wins = self.memory.wins(kind)
            return False, LANE_REFUSAL, (
                f"no evidence (p={probability:.2f} < {bar:.2f}) and this kind has had "
                f"its {plays} lessons for {wins} wins: it does not pay"
            )
        if float(attention) < float(self.explore_bar):
            return False, LANE_REFUSAL, (
                f"no evidence (p={probability:.2f} < {bar:.2f}) and the table scores "
                f"this kind {attention:.2f} < {self.explore_bar:.2f}: "
                f"not worth a lesson"
            )
        if balance - float(contract.penalty) < 0.0:
            return False, LANE_REFUSAL, (
                f"no evidence (p={probability:.2f} < {bar:.2f}) and a lesson costs "
                f"{contract.penalty:.0f} against a balance of {balance:.0f}: "
                f"capital is kept"
            )
        return True, LANE_SCHOOLING, (
            f"no evidence (p={probability:.2f} < {bar:.2f}); lesson {plays + 1} of "
            f"{self.schooling} on {kind}, table {attention:.2f}, memory {trust:.2f}"
        )

    # -- the work ----------------------------------------------------------- #

    def work(self, contract: Contract) -> Experience:
        """Decide, answer or refuse, take the rent, and remember what happened."""
        decision = self.should_take_contract(contract)
        verdict = int(decision.guess) if decision.take else None
        settlement = self.economy.settle(contract, verdict)
        self.economy.charge_rent()
        if decision.take:
            self.table.observe(
                contract.kind, settlement.credits / float(contract.reward)
            )
        # Only now is the answer known, and only now may it be looked at: a refusal
        # that would have been wrong is a fine that was not paid, and that is the
        # one number the notebook draws as a saved credit.
        saved = 0.0
        if decision.refused and int(decision.guess) != int(settlement.answer):
            saved = float(contract.penalty)
        experience = Experience(
            iteration=int(settlement.iteration),
            index=int(contract.index),
            kind=contract.kind,
            lane=decision.lane,
            take=bool(decision.take),
            reason=decision.reason,
            guess=int(decision.guess),
            answer=int(settlement.answer),
            correct=bool(settlement.correct),
            credits=float(settlement.credits),
            balance=float(settlement.balance),
            probability=float(decision.probability),
            trust=float(decision.trust),
            attention=float(decision.attention),
            breakeven=float(decision.breakeven),
            uncertainty=float(decision.uncertainty),
            margin=float(decision.margin),
            saved=float(saved),
        )
        self.memory.remember(experience)
        self.experiences.append(experience)
        return experience

    def run(self, iterations: int = 100) -> List[Experience]:
        """Live ``iterations`` contracts, or until the balance goes negative."""
        lived: List[Experience] = []
        for _ in range(max(int(iterations), 0)):
            if not self.economy.alive:
                break
            lived.append(self.work(self.economy.next_contract()))
        return lived

    # -- what it can say ---------------------------------------------------- #

    def history(self) -> Dict[str, np.ndarray]:
        """The life as arrays, one row per iteration — what the plot draws."""
        empty = np.zeros(0)
        if not self.experiences:
            return {
                "iteration": empty,
                "balance": empty,
                "credits": empty,
                "probability": empty,
                "uncertainty": empty,
                "attention": empty,
                "take": np.zeros(0, dtype=bool),
                "correct": np.zeros(0, dtype=bool),
                "saved": empty,
                "kind": np.array([], dtype=object),
                "lane": np.array([], dtype=object),
            }
        return {
            "iteration": np.array([e.iteration for e in self.experiences], dtype=int),
            "balance": np.array([e.balance for e in self.experiences], dtype=float),
            "credits": np.array([e.credits for e in self.experiences], dtype=float),
            "probability": np.array(
                [e.probability for e in self.experiences], dtype=float
            ),
            "uncertainty": np.array(
                [e.uncertainty for e in self.experiences], dtype=float
            ),
            "attention": np.array([e.attention for e in self.experiences], dtype=float),
            "take": np.array([e.take for e in self.experiences], dtype=bool),
            "correct": np.array([e.correct for e in self.experiences], dtype=bool),
            "saved": np.array([e.saved for e in self.experiences], dtype=float),
            "kind": np.array([e.kind for e in self.experiences], dtype=object),
            "lane": np.array([e.lane for e in self.experiences], dtype=object),
        }

    def report(self) -> Dict[str, Any]:
        """Everything a report prints: the wallet, the memory, the table, the log."""
        return {
            "economy": self.economy.summary(),
            "memory": self.memory.as_dict(),
            "table": self.table.as_dict(),
            "schooling": int(self.schooling),
            "certain": float(self.certain),
            "explore_bar": float(self.explore_bar),
            "lanes": {
                lane: sum(1 for item in self.decisions if item.lane == lane)
                for lane in LANES
            },
            "log": [item.as_dict() for item in self.experiences],
        }

    def summary_text(self) -> str:
        """A short readable account, in the order a person would ask the questions."""
        economy = self.economy.summary()
        memory = self.memory.as_dict()
        lanes = {
            lane: sum(1 for item in self.decisions if item.lane == lane)
            for lane in LANES
        }
        lines = [
            f"life: {economy['iteration']} iterations, "
            f"balance {economy['balance']:.0f} "
            f"({'alive' if economy['alive'] else 'died at ' + str(economy['died'])})",
            f"work: {economy['taken']} taken, {economy['won']} won "
            f"({economy['accuracy']:.0%}), {economy['refused']} refused",
            f"money: earned {economy['earned']:.0f}, rent {economy['rent_paid']:.0f}, "
            f"net {economy['net']:.0f}",
            "lanes: " + ", ".join(f"{lane} {count}" for lane, count in lanes.items()),
            f"break-even: {economy['breakeven']:.3f} against a chance of "
            f"1/{self.economy.market.streams} = "
            f"{1.0 / self.economy.market.streams:.3f}",
            f"saved by refusing: {memory['saved']:.0f} credits",
        ]
        for row in memory["kinds"]:
            lines.append(
                f"  {row['kind']:<10} offered {row['offered']:>3}"
                f"  taken {row['taken']:>3}  won {row['won']:>3}"
                f"  trust {row['trust']:.2f}  net {row['net']:>6.0f}"
            )
        return "\n".join(lines)

    def plot(self, path: "str", dpi: int = 130) -> Any:
        """Draw the three panels Anton asked for, and return the path."""
        from agent.physics.plots import plot_economy

        return plot_economy(self, path, dpi=dpi)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m agent.physics.worker --iterations 120 --plot economy.png``."""
    import argparse

    from agent.physics import sensorium as S
    from agent.physics.organism import Organism

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--balance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default="", help="write the report as JSON here")
    parser.add_argument("--plot", default="", help="write the figure here")
    parser.add_argument("--offline", action="store_true", default=True)
    args = parser.parse_args(argv)

    body = Organism.hatch(offline=True, steps=120, seed=args.seed)
    pool = S.listen_dataset("digits", limit=0)
    worker = Worker.hire(body, pool, seed=args.seed, balance=args.balance)
    worker.run(args.iterations)
    print(worker.summary_text())
    if args.output:
        import json

        path = worker.economy.market
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(worker.report(), handle, indent=2, ensure_ascii=False)
        print(f"report written to {args.output} ({path.seed})")
    if args.plot:
        worker.plot(args.plot)
        print(f"figure written to {args.plot}")
    return 0 if worker.economy.alive else 1


__all__ = [
    "Analysis",
    "CERTAIN",
    "Decision",
    "EXPLORE_BAR",
    "EpisodicMemory",
    "Experience",
    "LANES",
    "LANE_EVIDENCE",
    "LANE_REFUSAL",
    "LANE_SCHOOLING",
    "PSEUDO",
    "RealityOracle",
    "SCHOOLING",
    "Worker",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

