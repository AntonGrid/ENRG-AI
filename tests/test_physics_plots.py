"""The DePIN picture: three panels, one PNG, and no invented data.

The plot is a reading, not a decoration: the counter keeps its 1 Wh steps, the
coherence is drawn as read, and the relief is compared with ``e^(−εt)`` so the
decay lifetime can be seen rather than asserted. What is tested here is exactly
that — the file appears, its bytes are a PNG, the axes refuse series that do not
share a time axis, and a trace without a known ε still gets a fitted one.
"""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from agent.physics.plots import DEFAULT_PATH, plot_depin_dynamics  # noqa: E402


def _day(points=101, memory=1e-3, level=0.02, step=6.0):
    """One synthetic day: counter steps, coherence breathes, relief decays."""
    times = np.arange(0.0, points * step, step)
    energy = np.minimum(np.arange(times.size) * 1.0, 47.0)   # 1 Wh per step
    rng = np.random.default_rng(3)
    coherence = 0.55 + 0.08 * np.sin(times / 60.0) + 0.01 * rng.normal(size=times.size)
    spread = 0.02 + 0.004 * np.abs(np.sin(times / 90.0))
    relief = level * np.exp(-memory * times) * (1.0 + 0.01 * rng.normal(size=times.size))
    return times, energy, coherence, spread, relief


def test_default_name_is_the_one_asked_for():
    assert DEFAULT_PATH == "depin_test_plot.png"


def test_plot_writes_a_png(tmp_path):
    target = tmp_path / "depin_test_plot.png"
    written = plot_depin_dynamics(*_day(), memory=1e-3, path=target)
    assert written == target
    assert target.exists()
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert target.stat().st_size > 10_000


def test_plot_fits_memory_when_it_is_unknown(tmp_path):
    # The same day, drawn without ε: the fit reads it back off the trace, and
    # the picture must still be produced (the slope is stated in the legend).
    target = tmp_path / "fitted.png"
    written = plot_depin_dynamics(*_day(), path=target)
    assert written.exists() and written.stat().st_size > 10_000


def test_plot_creates_missing_directories(tmp_path):
    target = tmp_path / "nested" / "frame.png"
    written = plot_depin_dynamics(*_day(), memory=1e-3, path=target)
    assert written.exists()


def test_plot_rejects_series_of_different_length(tmp_path):
    times, energy, coherence, spread, relief = _day()
    with pytest.raises(ValueError):
        plot_depin_dynamics(
            times, energy[:-1], coherence, spread, relief,
            memory=1e-3, path=tmp_path / "bad.png",
        )


def _ranking():
    """Two streams as the curiosity module prints them, in its own row format."""
    from agent.physics.curiosity import Trait

    return [
        Trait(
            stream="behavior:digits#counterfeit", source="behavior:digits",
            fixations=24, movement=3.7, reference=2.9, reducible=1.0, lag=4,
            structure=20.7, reality=0.94, curiosity=0.94,
        ).as_dict(),
        Trait(
            stream="physics:sunspots", source="physics:sunspots",
            fixations=2, movement=0.21, reference=2.9, reality=0.73,
            note="too short to judge",
        ).as_dict(),
    ]


def test_curiosity_plot_draws_the_ranking_and_the_dialogue(tmp_path):
    from agent.physics.plots import plot_curiosity

    probes = [
        {"kind": "context", "curiosity": 0.94, "after": 0.63, "samples": 192},
        {"kind": "next", "curiosity": 0.63, "after": 0.47, "samples": 192},
    ]
    target = tmp_path / "curiosity.png"
    written = plot_curiosity(_ranking(), target, probes=probes)
    assert written == target
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert target.stat().st_size > 10_000


def test_curiosity_plot_without_questions_still_says_something(tmp_path):
    # An uncurious survey is drawn as an uncurious survey: three panels, no dialogue.
    from agent.physics.plots import plot_curiosity

    written = plot_curiosity(_ranking(), tmp_path / "silent.png")
    assert written.exists() and written.stat().st_size > 10_000


def test_curiosity_plot_survives_an_empty_survey(tmp_path):
    from agent.physics.plots import plot_curiosity

    written = plot_curiosity([], tmp_path / "nothing.png")
    assert written.exists() and written.stat().st_size > 5_000


def _blind_life():
    """A worker with an oracle that names nothing: the cheapest life a plot can draw.

    No sensorium is touched, so this stays a test of the *drawing*: the market, the
    wallet and the log are real, only the survey is stubbed. With an oracle that
    names nothing the worker spends its schooling budget and then refuses, which is
    exactly the shape the figure has to survive — refusals, fines and a balance.
    """
    from agent.physics.curiosity import Certainty, UCB
    from agent.physics.economy_sim import EconomySim, Market
    from agent.physics.worker import Analysis, EpisodicMemory, Worker

    class _Blind:
        def analyse(self, contract):
            return Analysis(
                index=contract.index,
                kind=contract.kind,
                names=tuple(stream.source for stream in contract.streams),
                traits=(),
                certainty=Certainty(chance=1.0 / len(contract.streams)),
            )

    pool = np.random.default_rng(5).normal(size=6000)
    economy = EconomySim(market=Market(pool=pool, seed=2), balance=50.0)
    worker = Worker(
        economy=economy,
        oracle=_Blind(),
        memory=EpisodicMemory(prior=1.0 / economy.market.streams),
        table=UCB(),
    )
    worker.run(6)
    return worker


def test_economy_plot_draws_a_life_and_an_empty_one(tmp_path):
    from agent.physics.economy_sim import EconomySim, Market
    from agent.physics.plots import plot_economy
    from agent.physics.worker import EpisodicMemory, Worker
    from agent.physics.curiosity import Certainty, UCB

    life = _blind_life()
    target = tmp_path / "economy.png"
    written = plot_economy(life, target)
    assert written == target
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert target.stat().st_size > 20_000

    # A body that has not worked yet is drawn as one, in the same box.
    class _Blind:
        def analyse(self, contract):
            from agent.physics.worker import Analysis

            return Analysis(
                index=contract.index,
                kind=contract.kind,
                names=tuple(stream.source for stream in contract.streams),
                traits=(),
                certainty=Certainty(chance=1.0 / len(contract.streams)),
            )

    pool = np.random.default_rng(6).normal(size=6000)
    fresh = Worker(
        economy=EconomySim(market=Market(pool=pool, seed=3), balance=50.0),
        oracle=_Blind(),
        memory=EpisodicMemory(prior=1.0 / 3.0),
        table=UCB(),
    )
    empty = plot_economy(fresh, tmp_path / "nothing_earned.png")
    assert empty.exists() and empty.stat().st_size > 5_000


def test_empire_plot_draws_a_capital_a_trap_and_a_switch(tmp_path):
    """Four panels, two foils, and the two events the figure exists to show."""
    from agent.physics.plots import plot_empire
    from agent.physics.portfolio import (
        NAIVE_APE,
        NAIVE_SPLIT,
        Empire,
        NaiveInvestor,
        ProtocolMarket,
    )

    steps = 200
    empire = Empire.launch(market=ProtocolMarket.mock(steps=steps, seed=7), steps=steps)
    empire.run(steps)  # no body: the cheap run still has to be drawable
    foils = []
    for policy in (NAIVE_SPLIT, NAIVE_APE):
        investor = NaiveInvestor.launch(
            ProtocolMarket.mock(steps=steps, seed=7), policy=policy
        )
        investor.run(steps)
        foils.append((policy, investor))
    target = tmp_path / "empire.png"
    written = plot_empire(empire, target, foils=foils)
    assert written == target
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert target.stat().st_size > 40_000

    class _Empty:
        kill_switch = type("K", (), {"triggers": []})()

        def history(self):
            return {"step": np.zeros(0), "equity": np.zeros(0), "engaged": np.zeros(0),
                    "volatility": np.zeros(0)}

        market = type("M", (), {"events": []})()
        assessments = {}
        decisions = []

    empty = plot_empire(_Empty(), tmp_path / "no_empire.png")
    assert empty.exists() and empty.stat().st_size > 5_000

