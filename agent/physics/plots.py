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


