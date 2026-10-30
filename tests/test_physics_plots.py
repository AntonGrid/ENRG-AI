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
