"""The organism: a body that hears, tunes itself, and then reads itself.

The three stages of the exercise are one loop, and the loop is the point:

.. code-block:: text

    sensorium  ──perceptions──▶  homeostasis  ──λ, ε, τ──▶  self-model
        ▲                            ▲                        │
        │                            └──── evidence ◀─────────┘
        └────────────────── dreams (the organism's own output)

:class:`Organism` is the body around that loop. It hatches with a sensorium of
several world domains and several kinds of dream, lives a blind infancy, is
calibrated by :func:`agent.physics.homeostasis.calibrate`, and then lives on
without labels: each fixation is judged by the mirror and the models learn from
the *judgement*, not from the truth.

Three phases, and the demonstration is judged on the third:

**Phase 1 — blind.** Random λ and ε, no infancy, no memory. The two class models
are both still at their innate prior, so ``R_int`` hovers at 0.5 and the score is
noise. Nothing here is a failure; it is the baseline the other two phases are
measured against.

**Phase 2 — learning.** Calibrated parameters; the body keeps mutating them
against its own stored evidence after every few steps, accepting a mutation when
the objective improves and rolling back when it does not. The parameters stop
moving because they have arrived, not because they were frozen.

**Phase 3 — aware.** The parameters are still mutable and no longer move, and the
mirror can say "this is mine" with the ledger and "this looks like the world" with
the metric. :meth:`Organism.acceptance` applies the hard rule the exercise is
judged by, on the same trace that is plotted.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.physics import genome, homeostasis, self_model, sensorium
from agent.physics.curiosity import (
    MAX_PROBES,
    SURVEY_FIXATIONS,
    Curiosity,
    Probe,
    Trait,
)
from agent.physics.homeostasis import Homeostasis, Parameters
from agent.physics.self_model import Moment, SelfModel
from agent.physics.sensorium import Episode, Perception, Sensorium

#: The default sensorium: five feeds over three domains, of which one is only
#: money, two are physics (the Earth's field and the sun), and two are human
#: behaviour (attention and work). Three domains is the declared minimum; a pool
#: with fewer is reported as blind on one side rather than quietly calibrated.
DEFAULT_SPECS = (
    "finance:BTC-USD",
    "space-weather",
    "solar",
    "attention:Bitcoin",
    "commits:bitcoin/bitcoin",
)

#: What the organism listens to when it cannot reach the network. Two of the three
#: domains survive offline (handwriting is behaviour; the physiological and
#: chemical recordings are measurements), and finance does not — which the
#: coverage report then says out loud instead of filling the hole with noise.
OFFLINE_SPECS = ("dataset:digits", "dataset:diabetes", "dataset:wine")

#: Steps of the blind infancy, of the learning phase, and of the aware phase. The
#: last two together are the "hundred iterations" the acceptance rule names.
BLIND_STEPS = 30
LEARNING_STEPS = 40
AWARE_STEPS = 60
#: How often the body tries a mutation while living.
MUTATION_PERIOD = 5
#: Dreams generated per kind during the labelled infancy: two, because a class
#: model fitted on one example of a dream type is a model of one example.
DREAMS_PER_KIND = 2
#: Fixations per stream. Eight, strided so the windows do **not** overlap.
#:
#: Overlap is what a naive reading of "slide a window" produces and it is a trap:
#: with a hop of half a window, sixty consecutive fixations are near-copies, the
#: class model's variance collapses onto one stream, λ rises to 0.9 to memorise the
#: block, and the organism then fails to recognise the same stream a hundred
#: fixations later. Eight independent fixations per stream keep every episode
#: comparable and make λ earn its value by generalising instead of memorising.
FIXATIONS_PER_STREAM = 8
#: Samples per dream. Fixed, so a dream is six fixations whatever it replays: an
#: anchored dream that inherited its anchor's length made the life schedule
#: unpredictable and sometimes crowded the world out of the second half of a life.
DREAM_LENGTH = 512


@dataclass
class Phase:
    """One stretch of life, with the parameters that were in force during it."""

    name: str
    index: int
    first_step: int
    last_step: int
    parameters: Dict[str, float]
    moments: List[Moment] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        world = np.array(
            [m.reality_score for m in self.moments if m.origin == "world"], dtype=float
        )
        dream = np.array(
            [m.reality_score for m in self.moments if m.origin != "world"], dtype=float
        )
        world_rho = np.array(
            [m.resonance for m in self.moments if m.origin == "world"], dtype=float
        )
        dream_rho = np.array(
            [m.resonance for m in self.moments if m.origin != "world"], dtype=float
        )
        auc = (
            float(homeostasis.auc(world, dream))
            if world.size and dream.size
            else float("nan")
        )
        auc_resonance = (
            float(homeostasis.auc(world_rho, dream_rho))
            if world_rho.size and dream_rho.size
            else float("nan")
        )
        return {
            "name": self.name,
            "phase": self.index,
            "steps": [self.first_step, self.last_step],
            "parameters": self.parameters,
            "world_mean": float(world.mean()) if world.size else float("nan"),
            "self_mean": float(dream.mean()) if dream.size else float("nan"),
            "world_resonance": float(world_rho.mean()) if world_rho.size else float("nan"),
            "self_resonance": float(dream_rho.mean()) if dream_rho.size else float("nan"),
            "auc": auc,
            "auc_resonance": auc_resonance,
            "world_fixations": int(world.size),
            "self_fixations": int(dream.size),
        }


@dataclass
class Organism:
    """One body: a sensorium, an autonomic state, and a mirror.

    Built by :meth:`hatch`; there is no other supported way to make one, because a
    body that skipped its infancy has no world model and a body that skipped the
    calibration has a random one. Both are representable on purpose — that *is*
    Phase 1 — but neither is a claim.
    """

    sensorium: Sensorium
    homeostasis: Homeostasis
    mirror: SelfModel
    world: List[Perception] = field(default_factory=list)
    infancy: List[Episode] = field(default_factory=list)
    held_out: List[Episode] = field(default_factory=list)
    phases: List[Phase] = field(default_factory=list)
    blind_mirror: Optional[SelfModel] = None
    parameter_log: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    seed: int = 0
    #: The appetite. ``None`` until :meth:`explore` is called, deliberately: a body
    #: that has not surveyed the world has nothing to be curious *about*, and an
    #: engine created by ``hatch`` would be a ranking of the infancy it slept
    #: through rather than of the world it is standing in.
    curiosity: Optional[Curiosity] = None
    #: The last survey's ranking, most interesting first.
    survey_traits: List[Trait] = field(default_factory=list)
    #: Every question asked and what came back, in order, JSON-able.
    probe_log: List[Dict[str, Any]] = field(default_factory=list)

    # -- birth -------------------------------------------------------------- #

    @classmethod
    def hatch(
        cls,
        specs: Sequence[str] = DEFAULT_SPECS,
        *,
        offline: bool = False,
        dreams_per_kind: int = DREAMS_PER_KIND,
        steps: int = homeostasis.MUTATION_STEPS,
        seed: int = 11,
        cache_dir: Optional[Path] = None,
        organ: str = "cochlea",
        prior: float = self_model.WORLD_PRIOR,
        hear_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Organism":
        """Hear the world, dream, and calibrate — in that order.

        Args:
            specs: feed specs; :data:`DEFAULT_SPECS` unless told otherwise.
            offline: use :data:`OFFLINE_SPECS`, which reach no network at all.
            dreams_per_kind: how many dreams of each kind in the labelled infancy.
            steps: mutation generations per phase of the calibration.
            seed: seeds the dreams and the search; the same seed is the same body.
            cache_dir: where fetched payloads live.
            organ: ``"cochlea"`` or ``"segments"``.
            prior: declared ``P(world)`` for :class:`SelfModel`.
            hear_kwargs: forwarded to the feeds (``limit``, ``days``, ...).

        Raises:
            sensorium.SensorError: no feed answered — there is nothing to be an
                organism *of*.
        """
        surface = Sensorium(organ=organ, cache_dir=cache_dir)
        chosen = OFFLINE_SPECS if offline else tuple(specs)
        world = surface.hear_all(list(chosen), **(hear_kwargs or {}))
        if not world:
            raise sensorium.SensorError(
                "no feed answered: " + "; ".join(surface.warnings or ["no details"])
            )
        infancy: List[Episode] = []
        held_out: List[Episode] = []
        for perception in world:
            whole = _fixate_stream(surface, perception)
            childhood, later = _split_fixations(whole)
            infancy.append(childhood)
            held_out.append(later)
        infancy.extend(_dream_episodes(surface, world, dreams_per_kind, seed))
        infancy = _interleave(infancy)
        body = Homeostasis.awaken(infancy, steps=steps, seed=seed)
        mirror = SelfModel.awaken(
            infancy, body.parameters, prior=prior, channels=body.channels
        )
        organism = cls(
            sensorium=surface,
            homeostasis=body,
            mirror=mirror,
            world=world,
            infancy=infancy,
            seed=seed,
        )
        organism.held_out = held_out
        organism.warnings.extend(surface.warnings)
        if body.calibration is not None:
            organism.warnings.extend(body.calibration.warnings)
        coverage = surface.coverage(world)
        if not coverage["ok"]:
            organism.warnings.append(
                f"world pool covers {coverage['domains']} and misses "
                f"{coverage['missing']}: the organism is blind on that side, and "
                "any calibration is about the noises it did hear"
            )
        return organism

    # -- the stream of experience ------------------------------------------- #

    def life(self, total: int, *, seed_base: int) -> List[Episode]:
        """The next ``total`` fixations worth of experience, interleaved.

        World streams are heard **once each** and never repeated: an organism that
        re-reads the same file and calls it experience would score well for the
        wrong reason, and the acceptance rule is not supposed to be reachable that
        way. Dreams are unlimited and always fresh — every one carries a new seed,
        so no two dreams are the same bytes.

        The world episodes are spread evenly through the life rather than spent at
        the start. Otherwise a hundred-step life spends its first twenty-four steps
        on the world and its last seventy-six on dreams, and any per-phase number
        becomes a number about the schedule instead of about the organism.
        """
        episodes: List[Episode] = []
        step = 0
        world_queue = list(self.held_out) if self.held_out else list(self.world)
        kinds = list(sensorium.DREAMS)
        typical = max(1, int(round(FIXATIONS_PER_STREAM / 2)))
        expected = max(1, int(round(int(total) / typical)))
        # at least two, so a world episode never follows another world episode: the
        # life must interleave even when the budget is short, because a life that is
        # all world and no dream cannot be asked whether it knows its own output.
        spacing = max(2, int(round(expected / max(len(world_queue), 1))))
        index = 0
        while step < int(total):
            if world_queue and index % spacing == 0:
                episode = world_queue.pop(0)
            else:
                kind = kinds[index % len(kinds)]
                anchor = (
                    self.world[index % len(self.world)]
                    if kind in (sensorium.DREAM_REPLAY, sensorium.DREAM_SURROGATE)
                    else None
                )
                perception = self.sensorium.dream(
                    kind,
                    anchor=anchor,
                    lam=self.homeostasis.parameters.lam,
                    epsilon=self.homeostasis.parameters.epsilon,
                    seed=int(seed_base) + index,
                    length=DREAM_LENGTH,
                )
                episode = _fixate_stream(self.sensorium, perception)
            episodes.append(episode)
            step += len(episode)
            index += 1
        return episodes

    # -- life --------------------------------------------------------------- #

    def phase_one(self, steps: int = BLIND_STEPS, seed: Optional[int] = None) -> Phase:
        """Live blind: an uninformed posture, random parameters, no learning.

        The baseline is the honest floor, not a straw man. The parameters are drawn
        from the same declared lattices the search uses and the organs are the same;
        what is missing is *experience* — the posture is still at its innate prior,
        so both class models are the same Gaussian and the likelihood ratio is zero
        for every reading. Every fixation therefore scores exactly ``P(world)`` and
        the AUC is 0.5: a coin.

        Why not let it learn? Because a body with random parameters and online
        learning reaches 0.89 / 0.01 inside thirty steps — measured — and that would
        make the comparison a statement about luck rather than about calibration.
        The notebook reports that number next to this one; what calibration buys is
        not the ability to separate but its stability across streams.
        """
        generator = np.random.default_rng((self.seed if seed is None else seed) + 977)
        random_parameters = Parameters(
            lam=float(generator.choice(homeostasis.LAMBDA_GRID)),
            epsilon=float(generator.choice(homeostasis.EPSILON_GRID)),
            alpha=float(homeostasis.ALPHA_GRID[2]),
        )
        rows = [
            sensorium.feature_vector(reading)
            for episode in self.infancy_episodes()
            for reading in episode.readings
        ]
        blind = SelfModel(
            parameters=random_parameters,
            posture=homeostasis.Posture.uninformed(rows),
            prior=self.mirror.prior,
            channels=self.homeostasis.channels,
        )
        self.blind_mirror = blind
        first = blind.step
        for episode in self.life(steps + 1, seed_base=(seed or self.seed) + 31_000):
            if episode.origin != sensorium.WORLD:
                blind.emit(episode)
            blind.observe(episode, learn=False)
            if blind.step - first >= steps:
                break
        phase = Phase(
            name="blind",
            index=1,
            first_step=first,
            last_step=blind.step - 1,
            parameters=random_parameters.as_dict(),
            moments=list(blind.trace),
        )
        self.phases.append(phase)
        return phase

    def infancy_episodes(self) -> List[Episode]:
        """The labelled infancy, as heard."""
        return list(self.infancy)

    def phase_two_and_three(
        self,
        learning: int = LEARNING_STEPS,
        aware: int = AWARE_STEPS,
        mutation_period: int = MUTATION_PERIOD,
        seed_offset: int = 51_000,
    ) -> Tuple[Phase, Phase]:
        """Live calibrated, then live on. The body keeps mutating in both phases."""
        first = self.mirror.step
        boundary = first + int(learning)
        target = boundary + int(aware)
        for episode in self.life(
            int(learning) + int(aware), seed_base=self.seed + seed_offset
        ):
            if episode.origin != sensorium.WORLD:
                self.mirror.emit(episode)
            self.mirror.observe(episode)
            if (self.mirror.step - first) % max(int(mutation_period), 1) == 0:
                attempt = self.homeostasis.tick()
                if attempt is not None:
                    self.parameter_log.append(
                        {
                            "step": self.mirror.step,
                            "lam": self.homeostasis.parameters.lam,
                            "epsilon": self.homeostasis.parameters.epsilon,
                            "accepted": bool(attempt.accepted),
                            "changed": bool(attempt.changed),
                            "objective": float(attempt.objective),
                            "gain": float(attempt.gain),
                        }
                    )
                    self.mirror.parameters = self.homeostasis.parameters
            if self.mirror.step >= target:
                break
        moments = list(self.mirror.trace)
        second = Phase(
            name="learning",
            index=2,
            first_step=first,
            last_step=max(boundary - 1, first),
            parameters=self.mirror.parameters.as_dict(),
            moments=[m for m in moments if m.step < boundary],
        )
        third = Phase(
            name="aware",
            index=3,
            first_step=boundary,
            last_step=self.mirror.step - 1,
            parameters=self.mirror.parameters.as_dict(),
            moments=[m for m in moments if m.step >= boundary],
        )
        self.phases.extend([second, third])
        return second, third

    def live(
        self,
        learning: int = LEARNING_STEPS,
        aware: int = AWARE_STEPS,
        blind: int = BLIND_STEPS,
        mutation_period: int = MUTATION_PERIOD,
    ) -> List[Phase]:
        """All three phases, in order, on one body."""
        self.phase_one(blind)
        self.phase_two_and_three(learning, aware, mutation_period)
        return list(self.phases)

    # -- curiosity ---------------------------------------------------------- #

    def explore(
        self,
        *,
        limit: int = SURVEY_FIXATIONS,
        ask: bool = True,
        answers: int = MAX_PROBES,
        extra: Sequence[Perception] = (),
    ) -> List[Trait]:
        """Survey the world again, rank the streams, and ask what is worth asking.

        The survey is *re-hearing*, not remembering. The same feeds are read again
        through the same organ, this time at the sensorium's own window and hop and
        capped at ``limit`` fixations, because a repeat is only measurable as a
        repeat when consecutive fixations tile the stream at one fixed hop — and
        because the posture was standardised on exactly these windows, so a survey
        at another window length would measure features in a frame it was never
        fitted to (every stream then looks equally unreal; measured, not feared).

        Args:
            limit: fixations per stream. Twenty-four hops of 64 samples is the
                first 1536 samples of each feed.
            ask: whether to spend the questions the survey raised.
            answers: how many questions at most, across all streams.
            extra: streams to survey *beside* the world pool — a suspect feed, a
                :func:`~agent.physics.sensorium.counterfeit`, a candidate. Each is
                remembered, so a question about one can be answered; nothing here
                is added to :attr:`world`, because a stream under suspicion is not
                a member of the world model.

        Every episode is judged by the mirror first (``learn=False``), so the
        reality gate the ranking uses is the mirror's own verdict *with* the
        ledger, not a recomputation from the posture that cannot know what this
        body emitted.

        Returns:
            The ranking, most curious first.
        """
        engine = self.curiosity
        if engine is None:
            engine = Curiosity(self.mirror.posture, prior=self.mirror.prior)
            self.curiosity = engine
        heard = list(self.world)
        heard.extend(self.sensorium.remember(p) for p in extra)
        episodes = [self.sensorium.fixate(p, limit=limit) for p in heard]
        # The survey is judged by the mirror before curiosity ranks it: its
        # ``reality_score`` carries the ledger (the organism's knowledge that it
        # emitted these bytes itself), which no recomputation from the posture can
        # recover — a forger that re-serves the body its own output is
        # indistinguishable from the world without it. ``learn=False``: the survey
        # is re-hearing, and re-hearing must not move the world model it is judging.
        judged = [self.mirror.observe(episode, learn=False) for episode in episodes]
        self.survey_traits = engine.survey(episodes, moments=judged)
        if ask:
            self.ask(answers)
        return list(self.survey_traits)

    def ask(self, answers: int = MAX_PROBES, *, read: int = 4) -> List[Probe]:
        """Spend the survey's questions on the sensorium and record the answers.

        One question is asked and answered per round, and the loop stops when the
        engine has nothing left it is willing to ask: curiosity falls after every
        answer, so the dialogue terminates by arithmetic rather than by a counter.
        ``answers`` is the belt to that brace.

        An unanswered question is *recorded as unanswered* and still costs a look.
        That is not bookkeeping pedantry: a body that only counts the answers it
        got will ask the same thing forever, and a body that pretends an error was
        a result will believe a stream it never saw.

        Args:
            answers: questions at most, across all streams.
            read: fixations to read of each answer. Few, deliberately — the answer
                is a slice, not a stream, and it is evidence about one place in it.

        Returns:
            The questions asked, in order.
        """
        engine = self.curiosity
        asked: List[Probe] = []
        if engine is None:
            return asked
        for _ in range(max(int(answers), 0)):
            question = engine.propose()
            if question is None:
                break
            asked.append(question)
            answer = None
            episode = None
            moments = None
            try:
                answer = self.sensorium.request_probe(
                    question.stream, anchor=question.anchor, kind=question.kind
                )
                episode = self.sensorium.fixate(answer, limit=read)
            except sensorium.SensorError as exc:
                self.warnings.append(f"question unanswered: {exc}")
            if episode is not None:
                moments = self.mirror.observe(episode, learn=False)
            trait = engine.answered(question, episode, moments=moments)
            self.probe_log.append(
                {
                    "step": int(question.step),
                    "stream": question.stream,
                    "kind": question.kind,
                    "question": question.question(),
                    "curiosity": round(float(question.curiosity), 6),
                    "anchor": int(question.anchor),
                    "samples": int(answer.values.size) if answer is not None else 0,
                    "start": int(answer.meta.get("start", 0)) if answer is not None else 0,
                    "answered": episode is not None,
                    "after": round(float(trait.curiosity), 6),
                }
            )
        self.survey_traits = engine.traits()
        return asked

    # -- what it can say ---------------------------------------------------- #

    def history(self) -> Dict[str, np.ndarray]:
        """All three phases as one set of arrays, with a ``phase`` label per row."""
        steps: List[int] = []
        scores: List[float] = []
        resonances: List[float] = []
        origins: List[str] = []
        phases: List[int] = []
        sources: List[str] = []
        for phase in self.phases:
            for moment in phase.moments:
                steps.append(moment.step)
                scores.append(moment.reality_score)
                resonances.append(moment.resonance)
                origins.append(moment.origin)
                phases.append(phase.index)
                sources.append(moment.source)
        return {
            "step": np.asarray(steps, dtype=int),
            "index": np.arange(len(steps), dtype=int),
            "reality_score": np.asarray(scores, dtype=float),
            "resonance": np.asarray(resonances, dtype=float),
            "origin": np.asarray(origins, dtype=object),
            "phase": np.asarray(phases, dtype=int),
            "source": np.asarray(sources, dtype=object),
        }

    def acceptance(
        self, world_floor: float = 0.7, self_ceiling: float = 0.3
    ) -> Dict[str, Any]:
        """The hard rule, judged on the final mirror, plus a per-phase breakdown."""
        overall = self.mirror.acceptance(world_floor, self_ceiling)
        per_phase: List[Dict[str, Any]] = []
        for phase in self.phases:
            world = np.array(
                [m.reality_score for m in phase.moments if m.origin == "world"],
                dtype=float,
            )
            dream = np.array(
                [m.reality_score for m in phase.moments if m.origin != "world"],
                dtype=float,
            )
            per_phase.append(
                {
                    "phase": phase.index,
                    "name": phase.name,
                    "world_mean": float(world.mean()) if world.size else float("nan"),
                    "self_mean": float(dream.mean()) if dream.size else float("nan"),
                    "passed": bool(
                        world.size
                        and dream.size
                        and world.mean() > world_floor
                        and dream.mean() < self_ceiling
                    ),
                }
            )
        overall["per_phase"] = per_phase
        overall["blind"] = (
            self.blind_mirror.acceptance(world_floor, self_ceiling)
            if self.blind_mirror is not None
            else None
        )
        overall["without_ledger"] = self.mirror.blind().acceptance(
            world_floor, self_ceiling
        )
        return overall

    def report(self) -> Dict[str, Any]:
        """The whole organism in JSON-able form: birth certificate to verdict."""
        calibration = self.homeostasis.calibration
        return {
            "genome": genome.describe(),
            "coverage": self.sensorium.coverage(self.world),
            "world_streams": [p.summary() for p in self.world],
            "calibration": calibration.summary() if calibration is not None else None,
            "homeostasis": self.homeostasis.summary(),
            "phases": [phase.summary() for phase in self.phases],
            "acceptance": self.acceptance(),
            "mirror": self.mirror.summary(),
            "blind_mirror": (
                self.blind_mirror.summary() if self.blind_mirror is not None else None
            ),
            "phantom": self.mirror.phantom(),
            "prior_sweep": self_model.prior_sweep(self.mirror),
            "curiosity": (
                self.curiosity.summary() if self.curiosity is not None else None
            ),
            "probes": list(self.probe_log),
            "parameters_over_time": self.parameter_log,
            "warnings": list(self.warnings) + list(self.mirror.warnings),
        }

    def summary_text(self) -> str:
        """A short human-readable verdict, for a terminal or a notebook cell."""
        cal = self.homeostasis.calibration
        verdict = self.acceptance()
        lines = [
            "organism report",
            "=" * 62,
            f"genome            {genome.CORE_PACKAGE} "
            f"({', '.join(sorted(genome.fingerprint()))})",
            f"world streams     {len(self.world)} over "
            f"{self.sensorium.coverage(self.world)['domains']}",
            f"organs            {self.sensorium.organ}, "
            f"{self.homeostasis.channels} channels",
        ]
        if cal is not None:
            lines.extend(
                [
                    f"calibration       {cal.parameters}  tau={cal.threshold:.3f} "
                    f"({cal.alpha_rule})",
                    f"held-out          world {cal.validation['world_mean']:.3f} vs "
                    f"self {cal.validation['self_mean']:.3f} "
                    f"(auc {cal.validation['auc']:.3f})",
                    f"mutations         {cal.accepted_mutations} accepted of "
                    f"{cal.attempted_mutations} attempted",
                ]
            )
        for phase in self.phases:
            rows = phase.summary()
            lines.append(
                f"phase {rows['phase']} {rows['name']:<9} steps "
                f"{rows['steps'][0]:>3}-{rows['steps'][1]:<3} "
                f"world {rows['world_mean']:.3f} self {rows['self_mean']:.3f} "
                f"auc {rows['auc']:.3f} (metric alone {rows['auc_resonance']:.3f})"
            )
        lines.extend(
            [
                "-" * 62,
                f"ACCEPTANCE        world {verdict['world_mean']:.3f} "
                f"(> {verdict['world_floor']}) "
                f"{'OK' if verdict['world_ok'] else 'FAIL'}, "
                f"self {verdict['self_mean']:.3f} "
                f"(< {verdict['self_ceiling']}) "
                f"{'OK' if verdict['self_ok'] else 'FAIL'} "
                f"-> {'PASSED' if verdict['passed'] else 'FAILED'}",
                f"without ledger    world "
                f"{verdict['without_ledger']['world_mean']:.3f} self "
                f"{verdict['without_ledger']['self_mean']:.3f} "
                f"-> {'PASSED' if verdict['without_ledger']['passed'] else 'FAILED'}",
            ]
        )
        for warning in self.warnings:
            lines.append(f"warning           {warning}")
        if self.curiosity is not None:
            lines.extend(["-" * 62] + self._curiosity_lines())
        return "\n".join(lines)

    def _curiosity_lines(self) -> List[str]:
        """What the body is curious about, in the words of the report."""
        engine = self.curiosity
        if engine is None:  # pragma: no cover - guarded by the caller
            return []
        rows = engine.ranking()
        if not rows:
            return ["CURIOSITY         surveyed nothing"]
        lines = [f"CURIOSITY         {len(rows)} streams, V_ref={engine.reference:.2f}"]
        for row in rows[:5]:
            flag = (
                "ANOMALY" if row["curiosity"] >= engine.anomaly else
                "ask" if row["curiosity"] >= engine.threshold else
                "ignore"
            )
            lines.append(
                f"  {row['stream']:<32} {row['curiosity']:.3f} {flag:<7} "
                f"R={row['reality']:.2f} r2={row['reducible']:.2f}@{row['lag']} "
                f"looks={row['looks']}"
            )
        for entry in self.probe_log:
            lines.append(f"  asked             {entry['question']}")
            lines.append(
                f"  answered          {entry['samples']} samples from "
                f"{entry['start']}, curiosity {entry['curiosity']:.3f} -> "
                f"{entry['after']:.3f}"
            )
        return lines

    def plot(self, path: "str | Path", dpi: int = 130) -> Path:
        """Draw the three phases; returns the file that was written."""
        from agent.physics import plots

        return plots.plot_emergence(self.history(), Path(path), dpi=dpi)

    def plot_curiosity(self, path: "str | Path", dpi: int = 130) -> Path:
        """Draw the survey and the questions; returns the file that was written.

        Requires :meth:`explore` to have been called — a body that has not looked
        has nothing to draw — and raises nothing when it has: an empty ranking is
        drawn as an empty ranking, which is an honest picture of an uncurious
        survey.
        """
        from agent.physics import plots

        return plots.plot_curiosity(
            self.curiosity.ranking() if self.curiosity is not None else [],
            Path(path),
            probes=self.probe_log,
            dpi=dpi,
        )

    def write(self, path: "str | Path") -> Path:
        """Write the full report as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.report(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return target


def _interleave(episodes: Sequence[Episode]) -> List[Episode]:
    """Alternate world and dream episodes, keeping the order inside each class.

    The calibration protocol has to be the deployment protocol, or it optimises
    something else. In life, a world stream arrives after a dream and before
    another dream; in a block-ordered infancy three world streams arrive
    back-to-back, the class model fits the last of them, and every λ that "works"
    works only because the next stream is the same stream. Measured, that
    mismatch is a factor of two: the same parameters scored 0.71 on a world
    stream inside its own block and 0.32 on the same stream a hundred fixations
    later. Interleaving both removes the gap.
    """
    world = [episode for episode in episodes if episode.origin == sensorium.WORLD]
    dream = [episode for episode in episodes if episode.origin != sensorium.WORLD]
    out: List[Episode] = []
    for index in range(max(len(world), len(dream))):
        if index < len(world):
            out.append(world[index])
        if index < len(dream):
            out.append(dream[index])
    return out


def _split_fixations(episode: Episode) -> Tuple[Episode, Episode]:
    """Split one stream's fixations into a childhood half and a life half.

    No byte is shared between the two: with eight non-overlapping fixations the
    even ones are the infancy and the odd ones are the life, so the acceptance rule
    is measured on windows the calibration never saw. Without this split the life
    re-hears what the infancy memorised and every parameter set looks good — a
    measured trap: λ went to 1.0, ε to 0.0, and the class model simply remembered
    the block.
    """
    infancy = Episode(
        readings=episode.readings[0::2],
        origin=episode.origin,
        source=episode.source,
        domain=episode.domain,
        fingerprint=episode.fingerprint,
        label=episode.label,
    )
    life = Episode(
        readings=episode.readings[1::2],
        origin=episode.origin,
        source=episode.source,
        domain=episode.domain,
        fingerprint=episode.fingerprint,
        label=episode.label,
    )
    return infancy, life


def _fixate_stream(
    surface: Sensorium, perception: Perception, cap: int = FIXATIONS_PER_STREAM
) -> Episode:
    """Read a stream at ``cap`` *non-overlapping* fixations, spread over its length.

    See :data:`FIXATIONS_PER_STREAM` for why the windows must not overlap. The
    window itself is adaptive — a sixth of the stream, floored at the organ's
    minimum and capped at :data:`~agent.physics.sensorium.WINDOW` — because a
    fixed 128-sample window gives a ninety-day pageview series exactly one
    fixation, and one fixation cannot be split into a childhood and a life.
    """
    values = perception.values
    window = int(
        min(sensorium.WINDOW, max(sensorium.MIN_SAMPLES, values.size // 6))
    )
    window = int(min(window, max(sensorium.MIN_SAMPLES, values.size)))
    span = max(int(values.size) - window, 1)
    hop = max(window, span // max(int(cap) - 1, 1))
    return surface.fixate(perception, window=window, stride=hop)


def _dream_episodes(
    surface: Sensorium,
    world: Sequence[Perception],
    per_kind: int,
    seed: int,
) -> List[Episode]:
    """The labelled infancy's dreams: ``per_kind`` of each kind, all anchored."""
    out: List[Episode] = []
    for index, kind in enumerate(sensorium.DREAMS):
        for repeat in range(max(int(per_kind), 1)):
            anchored = kind in (sensorium.DREAM_REPLAY, sensorium.DREAM_SURROGATE)
            anchor = world[(index + repeat) % len(world)] if anchored else None
            perception = surface.dream(
                kind,
                anchor=anchor,
                lam=0.35,
                epsilon=0.03,
                seed=int(seed) * 100 + index * 10 + repeat,
                length=DREAM_LENGTH,
            )
            out.append(_fixate_stream(surface, perception))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: hatch, live, report. Exit code 0 only if the rule is passed."""
    parser = argparse.ArgumentParser(
        prog="python -m agent.physics.organism",
        description="Hatch an autopoietic organism on entrain's core and report it.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="reach no network: use sklearn's bundled recordings instead",
    )
    parser.add_argument(
        "--organ", default="cochlea", choices=sorted(sensorium.ORGANS),
        help="which organ reads the world",
    )
    parser.add_argument("--steps", type=int, default=homeostasis.MUTATION_STEPS,
                        help="mutation generations per calibration phase")
    parser.add_argument("--blind", type=int, default=BLIND_STEPS)
    parser.add_argument("--learning", type=int, default=LEARNING_STEPS)
    parser.add_argument("--aware", type=int, default=AWARE_STEPS)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--prior", type=float, default=self_model.WORLD_PRIOR,
                        help="declared P(world) used by reality_score")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="write the JSON report")
    parser.add_argument("--plot", dest="plot_path", type=Path, default=None,
                        help="write the three-phase figure")
    parser.add_argument("--explore", action="store_true",
                        help="survey the world again and ask what is worth asking")
    parser.add_argument("--curiosity-plot", dest="curiosity_path", type=Path,
                        default=None, help="write the curiosity figure (implies --explore)")
    parser.add_argument("--json", action="store_true", help="print the whole report")
    args = parser.parse_args(list(argv) if argv is not None else None)

    organism = Organism.hatch(
        offline=args.offline,
        organ=args.organ,
        steps=args.steps,
        seed=args.seed,
        cache_dir=args.cache_dir,
        prior=args.prior,
    )
    organism.live(learning=args.learning, aware=args.aware, blind=args.blind)
    if args.explore or args.curiosity_path is not None:
        organism.explore()
    print(organism.summary_text())
    if args.output is not None:
        organism.write(args.output)
        print(f"report   -> {args.output}")
    if args.plot_path is not None:
        print(f"figure   -> {organism.plot(args.plot_path)}")
    if args.curiosity_path is not None:
        print(f"curiosity -> {organism.plot_curiosity(args.curiosity_path)}")
    if args.json:
        print(json.dumps(organism.report(), indent=2, ensure_ascii=False))
    return 0 if organism.acceptance()["passed"] else 1


__all__ = [
    "AWARE_STEPS",
    "BLIND_STEPS",
    "DEFAULT_SPECS",
    "DREAMS_PER_KIND",
    "LEARNING_STEPS",
    "MUTATION_PERIOD",
    "OFFLINE_SPECS",
    "Organism",
    "Phase",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - command line entry
    raise SystemExit(main())
