"""Three readings of a DePIN day, drawn one under another.

Why a picture: Statement #87 answers with two numbers, and neither shows *when*
something happened. The plot does — the counter steps while the load runs, ``R``
moves at switch-on, and the relief melts afterwards against ``e^(−εt)``, so
"the channel decays by the memory lifetime" becomes a slope and not a claim.

The energy panel is a step, never a line: a PZEM counter moves in 1 Wh jumps,
and smoothing them would draw consumption that never happened.

``matplotlib`` is optional in this package (``pip install enrg-ai[plot]``); the
import lives inside the function so the statistics layer works without it.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from agent.physics.curiosity import (
    ANOMALY_THRESHOLD,
    PROBE_THRESHOLD,
    UCB,
    lindley_gain,
)
from agent.physics.sterility import sterility_index

#: Default file name: Anton asked for a plot he can open next to the log.
DEFAULT_PATH = "depin_test_plot.png"


def _fit_memory(times: np.ndarray, relief: np.ndarray) -> float:
    """ε from the trace itself: the slope of ``ln relief`` against time.

    Used only when the caller does not know the medium's memory. The fit runs on
    the positive readings — a zero has no logarithm, and inventing one would put
    a spike into the answer.
    """
    positive = relief > 0.0
    if int(positive.sum()) < 2:
        return 0.0
    slope = float(np.polyfit(times[positive], np.log(relief[positive]), 1)[0])
    return max(-slope, 0.0)


def plot_depin_dynamics(
    timestamps,
    energy_wh,
    R,
    share_spread,
    relief_trace,
    *,
    memory: Optional[float] = None,
    relief_level: Optional[float] = None,
    path: os.PathLike | str = DEFAULT_PATH,
    title: str = "DePIN dynamics: counter, coherence, relief",
    show: bool = False,
):
    """Draw counter, coherence and relief; save the figure to ``path``.

    Panels, top to bottom:

    1. ``energy_wh`` as a step (1 Wh per step, PZEM-like) — what the device
       reports and what a proof is minted for;
    2. ``R`` (coherence) together with ``share_spread`` on the right axis — the
       expected movement is at switch-on, where the park stops sounding together
       and then comes back;
    3. ``relief`` (solid, with its noise) against ``R₀·e^(−εt)`` (dashed) — the
       lived channel and the formula it is compared with. The panel also prints
       Statement #87's readings (``A²``, ``CV``) and its verdict, so the picture
       and the number are never read apart.

    ``memory`` — ε of the medium; when omitted it is estimated from the trace
    (the slope of ``ln relief``). ``relief_level`` — R₀; when omitted it is the
    first relief reading.

    Returns the path written. Raises ``ImportError`` with the install hint when
    ``matplotlib`` is missing — the statistics layer does not need it.
    """
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")          # headless: the file is the deliverable
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'enrg-ai[plot]' "
            "(or pip install matplotlib)"
        ) from exc

    times = np.asarray(timestamps, dtype=float).ravel()
    energy = np.asarray(energy_wh, dtype=float).ravel()
    coherence = np.asarray(R, dtype=float).ravel()
    spread = np.asarray(share_spread, dtype=float).ravel()
    relief = np.asarray(relief_trace, dtype=float).ravel()

    lengths = {
        "timestamps": times.size,
        "energy_wh": energy.size,
        "R": coherence.size,
        "share_spread": spread.size,
        "relief_trace": relief.size,
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"all series must share one time axis, got {lengths}")

    eps = float(memory) if memory is not None else _fit_memory(times, relief)
    level = (
        float(relief_level)
        if relief_level is not None
        else float(relief[0] if relief.size else 0.0)
    )

    figure, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)

    # 1. the counter, step by step (1 Wh, as PZEM writes it)
    axes[0].step(times, energy, where="post", color="#1f4e79", linewidth=1.6)
    axes[0].set_ylabel("energy, Wh (step 1 Wh)")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(title, loc="left")

    # 2. coherence and the spread of shares
    axes[1].plot(times, coherence, color="#b03a2e", linewidth=1.6, label="R (coherence)")
    axes[1].set_ylabel("R", color="#b03a2e")
    axes[1].tick_params(axis="y", labelcolor="#b03a2e")
    axes[1].grid(alpha=0.3)
    twin = axes[1].twinx()
    twin.plot(times, spread, linestyle="--", color="#5d6d7e", linewidth=1.2,
              label="share spread")
    twin.set_ylabel("share spread", color="#5d6d7e")
    twin.tick_params(axis="y", labelcolor="#5d6d7e")
    handles, labels = axes[1].get_legend_handles_labels()
    more, more_labels = twin.get_legend_handles_labels()
    axes[1].legend(handles + more, labels + more_labels, loc="upper right", fontsize=9)

    # 3. the relief against the theoretical decay
    denominator = level if level != 0.0 else 1.0
    axes[2].plot(times, relief / denominator, color="#196f3d", linewidth=1.4,
                 label="relief (lived)")
    axes[2].plot(times, np.exp(-eps * times), linestyle="--", color="#7d6608",
                 linewidth=1.4, label=f"e^(−εt), ε = {eps:.2e}")
    axes[2].set_ylabel("relief / R₀")
    axes[2].set_xlabel("time, s")
    axes[2].grid(alpha=0.3)

    verdict = sterility_index(times, relief, eps, level=level)
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].text(
        0.01,
        0.05,
        f"Statement #87: A² = {verdict.a2:.3f}, CV = {verdict.cv:.4f} "
        f"→ {verdict.verdict}",
        transform=axes[2].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#999999"},
    )

    figure.tight_layout()
    target = Path(path)
    if target.parent != Path("") and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=140, bbox_inches="tight")
    if show:  # pragma: no cover - interactive path
        plt.show()
    plt.close(figure)
    return target


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that survives short lives and empty arrays."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        return v
    span = max(1, min(int(window), v.size))
    if span == 1:
        return v.copy()
    kernel = np.ones(span, dtype=float) / span
    pad_left = span // 2
    padded = np.concatenate(
        [np.full(pad_left, v[0]), v, np.full(span - 1 - pad_left, v[-1])]
    )
    return np.convolve(padded, kernel, mode="valid")[: v.size]


def plot_emergence(
    history,
    path: os.PathLike | str = "emergent_consciousness.png",
    *,
    phantom=None,
    title: str = "Autopoietic organism: blind, learning, aware",
    world_floor: float = 0.7,
    self_ceiling: float = 0.3,
    show: bool = False,
    dpi: int = 130,
):
    """Draw the three phases: what the organism concluded about its own input.

    Four panels, in the order the argument is made:

    1. ``reality_score`` step by step, world against the organism's own output,
       with the two declared bands (world above ``world_floor``, self below
       ``self_ceiling``) and the phase boundaries marked;
    2. ``internal_resonance`` on the same axis — the metric underneath the score,
       before the prior and the ledger touch it;
    3. each phase's mean, so the three stages of the argument are a picture and
       not a paragraph;
    4. the phantom: what the organism's own output scores once it has forgotten
       emitting it, against the wait. The rising curve *is* forgetting, drawn.

    ``history`` is the dict from :meth:`agent.physics.organism.Organism.history`;
    ``phantom`` is :meth:`agent.physics.self_model.SelfModel.phantom`. Both are
    optional — a caller with only the trace gets panels 1-3.

    Returns the path written. Raises ``ImportError`` with the install hint when
    ``matplotlib`` is missing: the statistics never need it.
    """
    previous_backend = None
    try:
        import matplotlib

        # The backend is switched only while the file is written and then restored: a
        # function that leaves the process on Agg breaks every later inline plot in a
        # notebook, which is exactly where this figure is meant to be shown.
        previous_backend = matplotlib.get_backend()
        if not show:
            matplotlib.use("Agg")          # headless: the file is the deliverable
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'enrg-ai[plot]' "
            "(or pip install matplotlib)"
        ) from exc

    steps = np.asarray(history["step"], dtype=int)
    position = np.asarray(history.get("index", history["step"]), dtype=int)
    scores = np.asarray(history["reality_score"], dtype=float)
    resonances = np.asarray(history["resonance"], dtype=float)
    origin = np.asarray(history["origin"], dtype=object)
    phase = np.asarray(history["phase"], dtype=int)

    world = origin == "world"
    dream = ~world
    colours = {1: "#7f8c8d", 2: "#b9770e", 3: "#1e8449"}
    panels = 4 if phantom else 3
    figure, axes = plt.subplots(panels, 1, figsize=(11.0, 3.0 * panels))

    # 1. the score
    axes[0].axhspan(world_floor, 1.0, color="#eaf5ea", zorder=0)
    axes[0].axhspan(0.0, self_ceiling, color="#fdecea", zorder=0)
    axes[0].axhline(world_floor, color="#1e8449", linestyle=":", linewidth=1.0)
    axes[0].axhline(self_ceiling, color="#b03a2e", linestyle=":", linewidth=1.0)
    axes[0].scatter(position[world], scores[world], s=14, color="#1e8449",
                    alpha=0.55, label="world")
    axes[0].scatter(position[dream], scores[dream], s=14, color="#b03a2e",
                    alpha=0.55, label="the organism's own output")
    order = np.argsort(position)
    for mask, colour, name in (
        (world, "#145a32", "world (rolling)"),
        (dream, "#7b241c", "self (rolling)"),
    ):
        if int(mask.sum()) >= 2:
            axes[0].plot(position[order][mask[order]],
                         _rolling(scores[order][mask[order]], 5),
                         color=colour, linewidth=2.0, label=name)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("reality_score")
    axes[0].set_title(title, loc="left")
    axes[0].legend(loc="center right", fontsize=9)
    axes[0].grid(alpha=0.3)

    # 2. the metric underneath
    axes[1].axhline(0.5, color="#5d6d7e", linestyle="--", linewidth=1.0)
    axes[1].scatter(position[world], resonances[world], s=14, color="#1e8449", alpha=0.55)
    axes[1].scatter(position[dream], resonances[dream], s=14, color="#b03a2e", alpha=0.55)
    if int(world.sum()) >= 2:
        axes[1].plot(position[order][world[order]],
                     _rolling(resonances[order][world[order]], 5),
                     color="#145a32", linewidth=2.0)
    if int(dream.sum()) >= 2:
        axes[1].plot(position[order][dream[order]],
                     _rolling(resonances[order][dream[order]], 5),
                     color="#7b241c", linewidth=2.0)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("internal_resonance")
    axes[1].set_xlabel("step")
    axes[1].grid(alpha=0.3)

    # phase boundaries and labels on both time panels
    boundaries = [
        int(int(np.min(position[phase == index]))) for index in (2, 3) if (phase == index).any()
    ]
    for axis in axes[:2]:
        for boundary in boundaries:
            axis.axvline(boundary, color="#566573", linestyle="-.", linewidth=1.1)
        for index in (1, 2, 3):
            rows = position[phase == index]
            if rows.size:
                axis.text(float(rows.min()), 1.0, f" phase {index}", fontsize=8,
                          color=colours.get(index, "#566573"), va="top")

    # 3. the three phases as bars
    names: list = []
    world_means: list = []
    self_means: list = []
    for index, name in ((1, "1 blind"), (2, "2 learning"), (3, "3 aware")):
        rows = phase == index
        if not rows.any():
            continue
        w = scores[rows & world]
        s = scores[rows & dream]
        names.append(name)
        world_means.append(float(w.mean()) if w.size else float("nan"))
        self_means.append(float(s.mean()) if s.size else float("nan"))
    positions = np.arange(len(names))
    width = 0.38
    axes[2].bar(positions - width / 2, world_means, width, color="#1e8449",
                label="world", alpha=0.85)
    axes[2].bar(positions + width / 2, self_means, width, color="#b03a2e",
                label="self", alpha=0.85)
    axes[2].axhline(world_floor, color="#1e8449", linestyle=":", linewidth=1.0)
    axes[2].axhline(self_ceiling, color="#b03a2e", linestyle=":", linewidth=1.0)
    axes[2].set_xticks(positions, names)
    axes[2].set_ylim(0.0, 1.02)
    axes[2].set_ylabel("mean reality_score")
    axes[2].legend(loc="upper left", fontsize=9)
    axes[2].grid(alpha=0.3, axis="y")

    # 4. the phantom
    if phantom:
        waits = np.array([row["wait"] for row in phantom], dtype=float)
        selves = np.array([row["self"] for row in phantom], dtype=float)
        worlds = np.array([row["world"] for row in phantom], dtype=float)
        axes[3].plot(waits, selves, marker="o", color="#b03a2e",
                     label="own output, forgotten for Δt")
        axes[3].plot(waits, worlds, marker="s", linestyle="--", color="#1e8449",
                     label="world, forgotten for Δt")
        axes[3].axhline(world_floor, color="#1e8449", linestyle=":", linewidth=1.0)
        axes[3].axhline(self_ceiling, color="#b03a2e", linestyle=":", linewidth=1.0)
        axes[3].set_xlabel("wait Δt before re-hearing, steps")
        axes[3].set_ylabel("reality_score")
        axes[3].set_ylim(-0.02, 1.02)
        axes[3].legend(loc="center right", fontsize=9)
        axes[3].grid(alpha=0.3)

    figure.tight_layout()
    target = Path(path)
    if target.parent != Path("") and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    if show:  # pragma: no cover - interactive path
        plt.show()
    plt.close(figure)
    if not show and previous_backend:  # leave the process as it was found
        try:
            matplotlib.use(previous_backend)
        except Exception:  # pragma: no cover - a backend that cannot be restored
            pass
    return target




def _pyplot(show: bool):
    """Import ``pyplot``, switch to Agg when headless, and report the backend.

    The backend is changed only while a file is being written and is put back by the
    caller: a function that leaves the process on Agg breaks every later inline plot in
    a notebook, which is exactly where these figures are shown.
    """
    import matplotlib

    previous = matplotlib.get_backend()
    if not show:
        matplotlib.use("Agg")          # headless: the file is the deliverable
    import matplotlib.pyplot as plt
    return plt, previous


def plot_curiosity(
    ranking,
    path: os.PathLike | str = "curiosity.png",
    *,
    probes=None,
    threshold: float = PROBE_THRESHOLD,
    anomaly: float = ANOMALY_THRESHOLD,
    title: str = "Curiosity: what the survey found worth asking about",
    show: bool = False,
    dpi: int = 130,
):
    """Draw the appetite: what was ranked, why, and what was asked about it.

    ``ranking`` is :meth:`agent.physics.curiosity.Curiosity.ranking` — a list of
    plain dicts, one per stream — and ``probes`` is the question journal from
    :attr:`agent.physics.organism.Organism.probe_log`.

    The panels are the argument of the module, drawn:

    1. **curiosity per stream**, with the threshold where the body stops watching
       and asks, and the line where the stream is called an anomaly — so the ranking is
       a picture and not an ordering;
    2. **the survey's own two numbers**: nats of structure against reality, so it
       is visible that the streams with the most structure are not the most real ones
       and the truthful feed is not the interesting one;
    3. **the Lindley gain beside the score**: the textbook expected information
       gain computed from the same survey — it reads loudness (and how often the body
       has looked), so it puts the noisiest feed on top and the actual forgery in the
       middle. That is the reason this module does not use it, shown rather than
       asserted;
    4. **the dialogue**, when ``probes`` is given: curiosity before and after each
       question, so the decay that ends the interrogation is a slope.

    Returns the path written. Raises ``ImportError`` with the install hint when
    ``matplotlib`` is missing.
    """
    try:
        plt, previous_backend = _pyplot(show)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'enrg-ai[plot]' "
            "(or pip install matplotlib)"
        ) from exc

    rows = [row for row in (ranking or []) if row]
    panels = 4 if probes else 3
    figure, axes = plt.subplots(panels, 1, figsize=(11.0, 3.0 * panels))
    if not rows:
        axes[0].text(0.5, 0.5, "nothing surveyed", ha="center", va="center",
                     transform=axes[0].transAxes)
        for axis in axes[1:]:
            axis.axis("off")
    else:
        order = list(range(len(rows)))[::-1]      # most curious at the top
        names = [rows[i]["stream"] for i in order]
        scores = np.array([rows[i]["curiosity"] for i in order], dtype=float)
        reality = np.array([rows[i]["reality"] for i in order], dtype=float)
        structure = np.array([rows[i]["structure"] for i in order], dtype=float)

        # 1. the ranking
        bars = axes[0].barh(np.arange(len(names)), scores, color="#1f618d", alpha=0.8)
        for bar, score in zip(bars, scores):
            if score <= 0.0 and scores.max() > 0.0:
                continue                   # among non-zero bars a zero bar is a
                                          # statement and needs no ink
            axes[0].annotate(
                f"{score:.3f}", (max(score, 0.004), bar.get_y() + bar.get_height() / 2),
                fontsize=8, va="center", ha="left", xytext=(4, 0),
                textcoords="offset points",
            )
        axes[0].set_yticks(np.arange(len(names)))
        axes[0].set_yticklabels(names, fontsize=8)
        axes[0].axvline(threshold, color="#b9770e", linestyle="--", linewidth=1.0,
                        label=f"ask at {threshold:.2f}")
        axes[0].axvline(anomaly, color="#b03a2e", linestyle=":", linewidth=1.0,
                        label=f"anomaly at {anomaly:.2f}")
        axes[0].set_xlim(0.0, 1.02)
        axes[0].set_xlabel("curiosity")
        axes[0].set_title(title, loc="left")
        axes[0].legend(loc="lower right", fontsize=9)
        axes[0].grid(alpha=0.3, axis="x")

        # 2. structure against reality: the two arms, both visible at once
        colours = axes[1].scatter(structure, reality, s=42, c=scores,
                                  cmap="viridis", vmin=0.0, vmax=1.0,
                                  edgecolor="#1b2631", linewidth=0.5, zorder=3)
        quiet = [
            row for index, row in enumerate([rows[i] for i in order])
            if structure[index] < 1.0
        ]
        horizon = max(max(structure), 1.0)
        for index, row in enumerate([rows[i] for i in order]):
            if structure[index] < 1.0:
                continue                      # named together in the corner instead
            left = structure[index] > 0.5 * horizon
            axes[1].annotate(
                f"{row['stream'].split(':')[-1].replace('#', ' #')} @{row['lag']}"
                f" ({structure[index]:.1f} nats)",
                (structure[index], reality[index]), fontsize=7,
                xytext=(-6, 4) if left else (6, 4),
                textcoords="offset points", ha="right" if left else "left",
            )
        if quiet:
            # Every honest feed shares one point on this axis (no repeat, hence no
            # structure) and would print on top of the others; a list says the same
            # thing and can be read. Domains are dropped for width; the list goes on one
            # line when it fits and two when it does not.
            short = [row["stream"].split(":")[-1] for row in quiet]
            label = "structure 0, no repeat: " + ", ".join(short)
            if len(label) > 46:
                half = (len(short) + 1) // 2
                label = ("structure 0, no repeat: " + ", ".join(short[:half])
                         + "\n" + ", ".join(short[half:]))
            axes[1].text(
                0.01, 0.04, label,
                transform=axes[1].transAxes, ha="left", va="bottom", fontsize=7,
                color="#566573",
            )
        axes[1].set_xlabel("structure, nats")
        axes[1].set_ylabel("reality R")
        axes[1].set_ylim(-0.02, 1.02)
        if float(np.max(structure)) <= 0.0:
            # no structure anywhere: without a scale of its own the axis would invent
            # one around zero and put the points on a tie they did not measure. Say
            # plainly what the survey found.
            axes[1].set_xlim(-1.0, 1.0)
        else:
            axes[1].margins(x=0.08)
        axes[1].grid(alpha=0.3)
        figure.colorbar(colours, ax=axes[1], label="curiosity")
        axes[1].set_title(
            "the two arms apart: the most real stream is not the interesting one",
            loc="left", fontsize=10,
        )

        # 3. the reference metric, which cannot tell the streams apart
        gains = np.array(
            [lindley_gain(rows[i]["movement"], 1.0, rows[i]["looks"]) for i in order]
        )
        axes[2].scatter(gains, scores, s=42, color="#7d3c98", zorder=3)
        for index, name in enumerate(names):
            # a downward offset on every other point keeps labels apart where two
            # streams land on the same gain: the coincidence is the point, so the labels
            # must not hide it
            offset = (5, 4) if index % 2 == 0 else (5, -13)
            axes[2].annotate(name.split(":")[-1], (gains[index], scores[index]),
                             fontsize=7, xytext=offset, textcoords="offset points")
        axes[2].set_xlabel("Lindley gain in nats")
        axes[2].set_ylabel("curiosity")
        axes[2].set_ylim(-0.09, 1.02)
        axes[2].margins(x=0.12)
        axes[2].set_title(
            "the same survey through the textbook metric: it ranks the loudest feed",
            loc="left", fontsize=10,
        )
        axes[2].grid(alpha=0.3)

        # 4. the dialogue: why the interrogation stops
        if panels == 4 and probes:
            steps = np.arange(len(probes))
            before = [float(entry.get("curiosity", 0.0)) for entry in probes]
            after = [float(entry.get("after", entry.get("curiosity", 0.0)))
                     for entry in probes]
            axes[3].plot(steps, before, marker="o", color="#b9770e",
                         label="when asked")
            axes[3].plot(steps, after, marker="s", color="#1e8449",
                         label="after the answer")
            last = len(steps) - 1
            for step, entry in zip(steps, probes):
                # One line per question, alternating above and below the pair of lines:
                # six questions in a row stay readable, neither line is covered, and the
                # late labels sit clear of the threshold rule instead of on it. The
                # final label runs to the left, because to the right of it there is only
                # the edge of the axes.
                offset = (8, 6) if step % 2 == 0 else (8, -12)
                align = "left"
                if step == last:
                    offset, align = (-10, -26), "right"
                axes[3].annotate(
                    f"{entry.get('kind', '?')}, {entry.get('samples', 0)} samples",
                    (step, min(max(before[step], after[step]), 0.98)),
                    fontsize=7, xytext=offset, textcoords="offset points", ha=align,
                )
            axes[3].axhline(threshold, color="#b9770e", linestyle="--", linewidth=1.0)
            axes[3].set_ylim(-0.02, 1.02)
            axes[3].margins(x=0.15)
            axes[3].set_xticks(steps)
            axes[3].set_xlabel("question")
            axes[3].set_ylabel("curiosity")
            axes[3].set_title("the interrogation ends because it stops paying",
                              loc="left", fontsize=10)
            axes[3].legend(loc="upper right", fontsize=9)
            axes[3].grid(alpha=0.3)

    figure.tight_layout()
    target = Path(path)
    if target.parent != Path("") and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    if show:  # pragma: no cover - interactive path
        plt.show()
    plt.close(figure)
    if not show and previous_backend:  # leave the process as it was found
        try:
            import matplotlib

            matplotlib.use(previous_backend)
        except Exception:  # pragma: no cover - a backend that cannot be restored
            pass
    return target



def _economy_colours(kinds) -> dict:
    """One colour per contract kind, fixed, so two figures can be compared."""
    palette = ("#1e8449", "#1b4f72", "#b9770e", "#b03a2e", "#6c3483", "#7f8c8d")
    return {kind: palette[index % len(palette)] for index, kind in enumerate(kinds)}


def plot_empire(
    empire,
    path: os.PathLike | str = "digital_entrepreneur.png",
    *,
    foils=(),
    title: str = "Digital entrepreneur: capital, a trap refused, and a switch obeyed",
    show: bool = False,
    dpi: int = 130,
):
    """Draw one empire: the capital, the moment of truth, and the kill switch working.

    ``empire`` is a :class:`agent.physics.portfolio.Empire` that has already run and
    ``foils`` is a sequence of ``(label, investor)`` pairs — the policies that do not
    measure anything. Everything is duck-typed (``history()``, ``trace``, ``decisions``,
    ``assessments``, ``kill_switch``, ``market``), so this module never imports the
    portfolio and the portfolio imports this one only from inside ``Empire.plot``.

    The four panels, and what each is evidence for:

    1. **the capital** — the empire's equity against the two credulous policies and the
       line it started on. A curve that ends higher *and* never dips below the start is
       a curve that was managed, not merely lucky;
    2. **the moment of truth** — every assessment at one step, plotted as the claim
       against the record's own coefficient of variation, coloured by verdict. The trap
       is the point in the top-left corner: an enormous promise and a record with no
       noise in it. The shaded band is the sterility floor from Statement #87;
    3. **the switch** — equity again, with the engaged spans shaded and every trigger
       marked where it fired, so "risk off" is a period and not a claim;
    4. **the two measurements the switch reads** — the cycle drawdown against its
    fifteen
       percent and the market's volatility against its three-fold limit. Two arms, two
       lines, both of them measurable by a reader with the log open.

    Returns the path written. Raises ``ImportError`` with the install hint when
    ``matplotlib`` is missing.
    """
    try:
        plt, previous_backend = _pyplot(show)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'enrg-ai[plot]' "
            "(or pip install matplotlib)"
        ) from exc

    from agent.physics.risk_assessment import (
        KILL_DRAWDOWN,
        KILL_VOLATILITY,
        SCAM_TRAP,
        STERILE_CV,
        SYNTHETIC_COLLUSION,
        SUSPECT,
    )

    history = empire.history()
    figure, axes = plt.subplots(4, 1, figsize=(11.0, 16.0))
    if not len(history["step"]):
        axes[0].text(
            0.5, 0.5, "no empire was run", ha="center", va="center",
            transform=axes[0].transAxes,
        )
        for axis in axes[1:]:
            axis.axis("off")
        return _finish(figure, plt, path, dpi, show, previous_backend)

    step = np.asarray(history["step"], dtype=float)
    equity = np.asarray(history["equity"], dtype=float)
    engaged = np.asarray(history["engaged"], dtype=float) > 0.5
    volatility = np.asarray(history["volatility"], dtype=float)
    start = float(equity[0])
    triggers = list(getattr(empire.kill_switch, "triggers", []))
    rugs = [
        event
        for event in getattr(empire.market, "events", [])
        if event.get("action") == "rug"
    ]

    # 1. the capital, against the policies that only read the advertisement
    colours = ("#b9770e", "#b03a2e", "#6c3483")
    for index, (label, foil) in enumerate(foils):
        other = foil.history()
        if len(other["equity"]):
            axes[0].plot(
                other["step"], other["equity"], linewidth=1.4, alpha=0.85,
                color=colours[index % len(colours)], label=label,
            )
    axes[0].plot(step, equity, color="#1b4f72", linewidth=2.4, label="this empire")
    axes[0].axhline(start, color="#7f8c8d", linestyle=":", linewidth=1.2)
    axes[0].annotate(
        f" started at {start:.0f}", xy=(0.005, start),
        xycoords=("axes fraction", "data"), fontsize=8, color="#566573", va="bottom",
    )
    for event in rugs:
        axes[0].axvline(float(event["step"]), color="#b03a2e", linestyle="--",
                        linewidth=1.0, alpha=0.7)
        axes[0].annotate(
            f" {event['protocol']} stops paying",
            xy=(float(event["step"]), 0.92),
            xycoords=("data", "axes fraction"), fontsize=7, color="#922b21",
            rotation=90, va="top",
        )
    first = empire.decisions[0]["step"] if empire.decisions else None
    if first is not None:
        axes[0].axvline(float(first), color="#1e8449", linestyle="-.", linewidth=1.1)
        axes[0].annotate(
            "first review",
            xy=(float(first), 0.86), xycoords=("data", "axes fraction"),
            fontsize=7, color="#145a32", va="top",
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("credits (log)")
    axes[0].set_xlabel("step")
    axes[0].set_title(title, loc="left")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(loc="upper left", fontsize=8)

    # 2. the moment of truth: the claim against the noise in the record
    judged = _empire_snapshot(empire)
    verdict_colour = {
        SCAM_TRAP: "#b03a2e",
        SYNTHETIC_COLLUSION: "#b9770e",
        SUSPECT: "#6c3483",
        "HONEST": "#1e8449",
        "UNMEASURED": "#7f8c8d",
    }
    axes[1].axhspan(0.0, STERILE_CV, color="#f5b7b1", alpha=0.35, zorder=0)
    axes[1].annotate(
        f" Statement #87: CV below {STERILE_CV:.3f} is a formula, not a record",
        xy=(0.01, STERILE_CV), xycoords=("axes fraction", "data"),
        fontsize=8, color="#922b21", va="bottom",
    )
    for name, row in judged.items():
        noise = float(row["cv"]) if row.get("cv") is not None else STERILE_CV * 0.5
        promise = max(float(row["claim_apy"]), 1e-3)
        colour = verdict_colour.get(str(row.get("verdict")), "#7f8c8d")
        axes[1].scatter(
            noise, promise, s=70, color=colour, edgecolors="black",
            linewidths=0.6, zorder=3,
        )
        axes[1].annotate(
            f"{name}: {row.get('verdict')} @{row.get('step')}",
            xy=(noise, promise), fontsize=7, color=colour, va="center", ha="left",
            xytext=(7, 0), textcoords="offset points",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(4.0e-5, 300.0)  # room for the labels of the loudest records
    axes[1].set_xlabel("CV of the record (noise in what it paid)")
    axes[1].set_ylabel("claimed APY (log)")
    axes[1].set_title(
        "The moment of truth: a promise is only as good as the noise under it",
        loc="left",
    )
    axes[1].grid(alpha=0.3, which="both")

    # 3. the switch, as periods of engagement, and every trigger where it fired
    axes[2].plot(step, equity, color="#1b4f72", linewidth=2.0, label="equity")
    for span in _spans(step, engaged):
        axes[2].axvspan(span[0], span[1], color="#fdf2e9", zorder=0,
                        label="risk off: the switch is engaged")
    for event in triggers:
        if event.arm == "release":
            axes[2].axvline(float(event.step), color="#1e8449", linestyle=":",
                            linewidth=1.2)
        else:
            axes[2].axvline(float(event.step), color="#b03a2e", linewidth=1.6)
            axes[2].annotate(
                f"{event.arm} {event.drawdown:.0%}" if event.arm == "drawdown"
                else f"{event.arm} {event.volatility or 0.0:.1f}x",
                xy=(float(event.step), 0.28 if event.arm == "drawdown" else 0.12),
                xycoords=("data", "axes fraction"), fontsize=8, color="#922b21",
                ha="right", va="center",
            )
    axes[2].axhline(start, color="#7f8c8d", linestyle=":", linewidth=1.2)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("credits (log)")
    axes[2].set_xlabel("step")
    axes[2].set_title(
        "The kill switch: when the empire was told to hold nothing", loc="left"
    )
    axes[2].grid(alpha=0.3, which="both")
    handles, _ = axes[2].get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, _):
        seen.setdefault(label, handle)
    axes[2].legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)

    # 4. the two measurements the switch reads, against the lines it fires at
    cycle = int(getattr(empire, "review_every", 24) or 24)
    drawdown = np.zeros_like(equity)
    for index in range(equity.size):
        back = max(index - cycle, 0)
        drawdown[index] = 1.0 - equity[index] / max(equity[back], 1e-9)
    axes[3].plot(step, np.maximum(drawdown, 0.0) * 100.0, color="#b03a2e",
                 linewidth=1.6, label="cycle drawdown, %")
    axes[3].axhline(KILL_DRAWDOWN * 100.0, color="#922b21", linestyle="--",
                    linewidth=1.2, label=f"fires at {KILL_DRAWDOWN:.0%} per cycle")
    twin = axes[3].twinx()
    usable = volatility >= 0.0
    twin.plot(step[usable], np.minimum(volatility[usable], 50.0), color="#6c3483",
              linewidth=1.4, alpha=0.85, label="market volatility, x its history")
    twin.axhline(KILL_VOLATILITY, color="#4a235a", linestyle=":",
                 linewidth=1.2, label=f"fires at {KILL_VOLATILITY:.0f}x")
    twin.set_ylabel("volatility ratio")
    axes[3].set_ylabel("drawdown, %")
    axes[3].set_xlabel("step")
    axes[3].set_title("What the switch was reading", loc="left")
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="upper left", fontsize=8)
    twin.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    return _finish(figure, plt, path, dpi, show, previous_backend)


def _empire_snapshot(empire) -> dict:
    """One row per protocol: the *worst* verdict it ever earned, and the numbers for it.

    The last review would be the wrong picture — by then both traps are dead, and a dead
    protocol is refused for being dead, not for being a trap. What panel 2 has to show
    is the moment the measurement was made, so the snapshot keeps each protocol's most
    serious verdict over the whole life, with the numbers that produced it.
    """
    order = {"SCAM_TRAP": 4, "SYNTHETIC_COLLUSION": 3, "SUSPECT": 2, "HONEST": 1}
    worst: dict = {}
    for step in sorted(getattr(empire, "assessments", {}) or {}):
        for name, row in empire.assessments[step].items():
            rank = order.get(str(row.get("verdict")), 0)
            if name not in worst or rank > worst[name][0]:
                worst[name] = (rank, dict(row, step=step))
    return {name: value[1] for name, value in worst.items()}


def _spans(step: np.ndarray, mask: np.ndarray) -> list:
    """The ``(start, stop)`` pairs of a boolean mask — one run per engagement."""
    spans = []
    opening = None
    for index, flag in enumerate(mask):
        if flag and opening is None:
            opening = float(step[index])
        elif not flag and opening is not None:
            spans.append((opening, float(step[index])))
            opening = None
    if opening is not None:
        spans.append((opening, float(step[-1]) if step.size else opening))
    return spans



def plot_economy(
    life,
    path: os.PathLike | str = "economic_agent.png",
    *,
    title: str = "Economic agent: what a verdict was worth, and what a refusal saved",
    show: bool = False,
    dpi: int = 130,
):
    """Draw one working life: the wallet, the refusals, and the attention behind them.

    ``life`` is a :class:`agent.physics.worker.Worker` that has already run — it is
    duck-typed (``history()``, ``experiences``, ``economy``, ``table``), so this
    module never imports the worker and the worker imports this one only from
    inside :meth:`Worker.plot`. The import graph stays a tree.

    The five panels are the argument of the two modules underneath, drawn:

    1. **the balance**, with two policies the body could have played instead of
       thinking: refuse everything (rent only, a straight line into the floor) and guess
       every job (the expectation of a coin against these terms). A curve that beats
       both is a curve that was *earned* by the survey;
    2. **the refusals that saved credits** — the fine that was not paid, counted
       only where the answer the body was leaning towards would have been wrong. This is
       a counterfactual and it is drawn as one: hollow markers are the refusals where
       the same answer would have won, so the panel shows both the refusals that paid
       and the jobs that were simply not taken;
    3. **the confidence against the break-even** — every decision as a dot, coloured
       by which rule played it, with the market's own ``penalty/(reward+penalty)``
       as a line. A point above the line that was refused, or below it that was taken,
       is a bug, and it is visible as one;
    4. **the UCB table over time**, replayed from the log with the same arithmetic,
       so the optimism about a kind is a curve that shrinks as the kind is learned;
    5. **the attention share** — the rolling share of *takes* per kind, which is
       where "it stopped spending its time on noise" becomes a picture.

    Returns the path written. Raises ``ImportError`` with the install hint when
    ``matplotlib`` is missing.
    """
    try:
        plt, previous_backend = _pyplot(show)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'enrg-ai[plot]' "
            "(or pip install matplotlib)"
        ) from exc

    history = life.history()
    economy = life.economy
    kinds = [kind for kind, _ in economy.market.mix]
    colours = _economy_colours(kinds)
    figure, axes = plt.subplots(5, 1, figsize=(11.0, 15.0))
    if not len(history["iteration"]):
        axes[0].text(
            0.5, 0.5, "no life lived", ha="center", va="center",
            transform=axes[0].transAxes,
        )
        for axis in axes[1:]:
            axis.axis("off")
        return _finish(figure, plt, path, dpi, show, previous_backend)

    iteration = np.asarray(history["iteration"], dtype=int)
    balance = np.asarray(history["balance"], dtype=float)
    credits = np.asarray(history["credits"], dtype=float)
    taken = np.asarray(history["take"], dtype=bool)
    correct = np.asarray(history["correct"], dtype=bool)
    probability = np.asarray(history["probability"], dtype=float)
    lane = np.asarray(history["lane"], dtype=object)
    saved = np.asarray(history["saved"], dtype=float)
    reward = float(economy.reward)
    fine = float(economy.penalty)
    rent_rate = float(economy.rent) / float(economy.rent_period)
    chance = 1.0 / float(economy.market.streams)
    start = float(balance[0]) - float(credits[0])
    won = taken & correct
    lost = taken & ~correct
    refused = ~taken




    # 1. the wallet, against the two policies that do not think
    steps = np.arange(1, int(iteration.max()) + 1)
    guess_edge = chance * reward - (1.0 - chance) * fine
    school = lane == "schooling"
    if bool(school.any()):
        last_lesson = int(iteration[school].max())
        axes[0].axvspan(
            float(iteration.min()) - 0.5, float(last_lesson), color="#fdf2e9", zorder=0,
            label="learning: information is bought",
        )
    axes[0].axhline(0.0, color="#922b21", linewidth=1.2)
    axes[0].plot(
        steps, start - rent_rate * steps, linestyle=":", color="#7f8c8d", linewidth=1.4,
        label="refuse every job: rent only",
    )
    axes[0].plot(
        steps, start + (guess_edge - rent_rate) * steps, linestyle="--",
        color="#b9770e", linewidth=1.4, label="guess every job: expectation",
    )
    axes[0].plot(iteration, balance, color="#1b4f72", linewidth=2.0, label="this agent")
    axes[0].scatter(iteration[won], balance[won], s=12, color="#1e8449", zorder=3,
                    label=f"won ({int(won.sum())})")
    axes[0].scatter(iteration[lost], balance[lost], s=22, color="#b03a2e", marker="x",
                    zorder=3, label=f"fined ({int(lost.sum())})")
    axes[0].scatter(iteration[refused], balance[refused], s=10, color="#95a5a6",
                    marker="|", zorder=2, label=f"refused ({int(refused.sum())})")
    school = lane == "schooling"
    if bool(school.any()):
        last_lesson = int(iteration[school].max())
        axes[0].axvline(last_lesson, color="#b9770e", linestyle="-.", linewidth=1.1)
        axes[0].annotate(
            " last paid lesson", xy=(last_lesson, 0.95),
            xycoords=("data", "axes fraction"), fontsize=8, color="#7e5109",
        )
    axes[0].set_ylabel("credits")
    axes[0].set_title(title, loc="left")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)

    # 2. the refusals, as fines that were not paid
    refused_at = iteration[refused]
    saved_at = saved[refused]
    if refused_at.size:
        axes[1].vlines(refused_at, 0.0, saved_at, color="#1e8449", linewidth=1.4)
        axes[1].scatter(refused_at, saved_at, s=18, color="#1e8449",
                        label=f"saved {float(saved_at.sum()):.0f} credits")
        axes[1].scatter(refused_at, np.zeros_like(saved_at), s=16, facecolors="none",
                        edgecolors="#7f8c8d", label="refused, would have been right")
    axes[1].axhline(fine, color="#b03a2e", linestyle=":", linewidth=1.1)
    axes[1].annotate(
        f" one fine = {fine:.0f}", xy=(0.005, fine), xycoords=("axes fraction", "data"),
        fontsize=8, color="#922b21", ha="left", va="bottom",
    )
    axes[1].set_ylabel("credits not lost")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylim(-1.0, max(fine * 1.35, 1.0))
    axes[1].grid(alpha=0.3)
    twin = axes[1].twinx()
    twin.plot(iteration, np.cumsum(saved), color="#145a32", linewidth=1.6, alpha=0.8)
    twin.set_ylabel("not lost, cumulative")
    axes[1].legend(loc="upper left", fontsize=8)


    # 3. the confidence, against the line the market drew
    for name, colour in (
        ("evidence", "#1b4f72"),
        ("schooling", "#b9770e"),
        ("refusal", "#b03a2e"),
    ):
        mask = lane == name
        if bool(mask.any()):
            axes[2].scatter(iteration[mask], probability[mask], s=16, color=colour,
                            alpha=0.75, label=f"{name} ({int(mask.sum())})")
    axes[2].axhline(float(economy.breakeven), color="#922b21", linewidth=1.4,
                    label=f"break-even {float(economy.breakeven):.3f}")
    axes[2].axhline(chance, color="#7f8c8d", linestyle=":", linewidth=1.1,
                    label=f"chance 1/{economy.market.streams} = {chance:.3f}")
    axes[2].set_ylim(-0.03, 1.09)
    axes[2].set_ylabel("p (told) / break-even")
    axes[2].set_xlabel("iteration")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="lower right", fontsize=8, ncol=2)

    # 4. the table, replayed with the same arithmetic it was played with
    engine = getattr(life, "table", None)
    table = UCB(c=float(getattr(engine, "c", 1.0)))
    series = {kind: [] for kind in kinds}
    for item in life.experiences:
        if item.take:
            table.observe(item.kind, float(item.credits) / max(reward, 1e-9))
        for kind in kinds:
            series[kind].append(table.score(kind))
    for kind in kinds:
        axes[3].plot(iteration[: len(series[kind])], series[kind],
                     color=colours[kind], linewidth=1.6, label=kind)
    axes[3].axhline(float(getattr(life, "explore_bar", 0.5)), color="#7f8c8d",
                    linestyle=":", linewidth=1.1)
    axes[3].annotate(
        "below this line a kind is not worth a lesson",
        xy=(0.5, float(getattr(life, "explore_bar", 0.5))),
        xycoords=("axes fraction", "data"), fontsize=8, color="#566573",
        ha="center", va="top",
    )
    low = 0.0
    if any(series.values()):
        low = min(min(values) for values in series.values() if values)
    axes[3].set_ylim(min(low, float(getattr(life, "explore_bar", 0.5))) - 0.15, None)
    axes[3].set_ylabel("UCB score")
    axes[3].set_xlabel("iteration")
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="upper right", fontsize=8, ncol=2)

    # 5. the share of takes, by kind: where the attention actually went
    span = max(10, int(iteration.size) // 8)
    total = _rolling(taken.astype(float), span)
    for kind in kinds:
        flags = np.array(
            [1.0 if (item.take and item.kind == kind) else 0.0
             for item in life.experiences]
        )
        share = _rolling(flags, span) / np.maximum(total, 1e-9)
        axes[4].plot(iteration, share, color=colours[kind], linewidth=1.6, label=kind)
    axes[4].set_ylim(-0.03, 1.03)
    axes[4].set_ylabel(f"share of takes ({span}-point window)")
    axes[4].set_xlabel("iteration")
    axes[4].grid(alpha=0.3)
    axes[4].legend(loc="upper right", fontsize=8, ncol=2)

    figure.tight_layout()
    return _finish(figure, plt, path, dpi, show, previous_backend)


def _finish(figure, plt, path, dpi, show, previous_backend):
    """Write the figure, close it, and put the matplotlib backend back as it was."""
    target = Path(path)
    if target.parent != Path("") and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")
    if show:  # pragma: no cover - interactive path
        plt.show()
    plt.close(figure)
    if not show and previous_backend:  # leave the process as it was found
        try:
            import matplotlib

            matplotlib.use(previous_backend)
        except Exception:  # pragma: no cover - a backend that cannot be restored
            pass
    return target


def plot_funnel(
    stages,
    path: os.PathLike | str = "discovery_funnel.png",
    *,
    reasons=None,
    title: str = "Discovery funnel: what a market offers, and what survives",
    show: bool = False,
    dpi: int = 130,
):
    """Draw a search: the widths that survive each gate, and what each gate cost.

    ``stages`` is a sequence of ``(label, count)`` in the order the gates are passed —
    ``DiscoveryReport.stages()`` returns exactly that — and the figure draws the counts as
    bars that shrink, with the *difference* between two consecutive bars labelled as a
    loss. That is the whole design decision: a funnel is a picture of what was removed,
    not of what was measured, and a reader who subtracts two numbers should not have to
    do it in their head.

    ``reasons`` is optional and, when given, is the scanner's own count of *why* it
    dropped what it dropped (:meth:`agent.physics.discovery.Watchlist.reasons`). Without
    it the figure is one panel; with it, the reasons are a second panel, because "120
    dropped" and "120 dropped for three different reasons" are different statements about
    a market. The reasons are necessarily a subset of level 1's loss: level 2's refusals
    are verdicts, and a verdict is not a one-line reason.

    Everything is duck-typed: nothing here imports ``discovery``, ``portfolio`` or
    ``risk_assessment``, so a figure can be drawn from a plain list of tuples in a test
    and cannot make the plotting layer a dependency of the decision layer.

    Returns the path written. Raises ``ImportError`` with the install hint when
    ``matplotlib`` is missing.
    """
    try:
        plt, previous_backend = _pyplot(show)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "plotting needs matplotlib: pip install 'enrg-ai[plot]' "
            "(or pip install matplotlib)"
        ) from exc

    rows = [(str(label), int(count)) for label, count in stages] or [("market", 0)]
    counts = np.array([count for _, count in rows], dtype=float)
    top = max(float(counts.max()), 1.0)
    palette = ("#1b4f72", "#2874a6", "#5499c7", "#a9cce3", "#d4e6f1")
    height = 0.95 * len(rows) + 2.4
    if reasons:
        figure, (left, right) = plt.subplots(
            1,
            2,
            figsize=(12.4, height),
            gridspec_kw={"width_ratios": [2.6, 1.0]},
        )
    else:  # pragma: no cover - the one-panel figure is drawn by the same code below
        figure, left = plt.subplots(1, 1, figsize=(9.6, height))
        right = None

    positions = np.arange(len(rows))[::-1].astype(float)
    left.barh(
        positions,
        counts,
        height=0.62,
        color=[palette[index % len(palette)] for index in range(len(rows))],
        edgecolor="#1b2631",
        linewidth=0.8,
    )
    for position, (label, count), width in zip(positions, rows, counts):
        left.text(
            width + top * 0.012,
            position,
            f"{count}",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#1b2631",
        )
        left.text(
            top * 0.015,
            position,
            label,
            va="center",
            fontsize=9.5,
            color="white" if width > top * 0.2 else "#1b2631",
        )
    for index in range(len(rows) - 1):
        dropped = int(rows[index][1]) - int(rows[index + 1][1])
        if dropped <= 0:
            continue
        share = dropped / max(int(rows[index][1]), 1)
        left.annotate(
            f"−{dropped} ({share:.0%})",
            xy=(max(float(counts[index + 1]), 1.0), positions[index]),
            xytext=(
                top * 1.04,
                (positions[index] + positions[index + 1]) / 2.0,
            ),
            fontsize=9,
            color="#b03a2e",
            va="center",
            arrowprops={"arrowstyle": "->", "color": "#b03a2e", "linewidth": 0.9},
        )
    left.set_xlim(0.0, top * 1.24)
    left.set_yticks([])
    left.set_xlabel("protocols")
    left.set_title(title, fontsize=11, loc="left")
    left.grid(axis="x", alpha=0.25)

    if right is not None:
        names = list(reasons)[::-1]
        values = [int(reasons[name]) for name in names]
        right.barh(
            np.arange(len(names)),
            values,
            height=0.6,
            color="#b03a2e",
            alpha=0.85,
            edgecolor="#1b2631",
            linewidth=0.6,
        )
        for index, value in enumerate(values):
            right.text(value + max(values) * 0.02, index, f"{value}", va="center",
                       fontsize=9, color="#1b2631")
        right.set_yticks(np.arange(len(names)), names, fontsize=8.5)
        right.set_xlim(0.0, max(values) * 1.22)
        right.set_xlabel("protocols")
        right.set_title("why level 1 dropped them", fontsize=10, loc="left")
        right.grid(axis="x", alpha=0.25)

    figure.tight_layout()
    return _finish(figure, plt, path, dpi, show, previous_backend)

