"""The park reading: a day's phase, its silence, its frequency, its order.

Each test names one claim of the reading, so a regression says which claim
broke. The two that matter most for the design are here: the reading does not
fire on a producer that was always offset (the park's own shape is not a
change), and the reading is blind to level by construction (the book's state is
frequency, not size).
"""

import numpy as np
import pytest

from agent.physics.park import ParkConfig, demodulate, observe_park

DAY = 24.0
BUCKETS = 168


def _bump(hours, peak_hour: float, width_hours: float = 3.2, amplitude: float = 1000.0):
    distance = np.abs((np.asarray(hours, dtype=float) - peak_hour + 12.0) % 24.0 - 12.0)
    return amplitude * np.exp(-0.5 * (distance / width_hours) ** 2)


def _series(peak_hour: float = 12.0, buckets: int = BUCKETS):
    return _bump(np.arange(buckets, dtype=float), peak_hour)


def _demodulate_tapered(values, taper):
    """The same reading with a taper — used only to measure why we do not."""
    x = np.asarray(values, dtype=float)
    window = len(taper)
    omega0 = 2.0 * np.pi / DAY
    phi = np.zeros(x.size)
    for t in range(x.size):
        k = min(window, t + 1)
        seg = x[t - k + 1 : t + 1][::-1]
        tap = taper[window - k :]
        total = float(np.sum(tap))
        if total == 0.0:
            continue
        z = np.sum(tap * seg * np.exp(-1j * omega0 * (t - np.arange(k)))) / total
        phi[t] = np.angle(z)
    return phi


# ── the demodulator ─────────────────────────────────────────────────────────


def test_demodulate_is_flat_for_a_clean_day():
    phi, envelope = demodulate(_series(11.2), DAY)
    assert np.ptp(phi[30:]) < 1e-9
    assert np.ptp(envelope[30:]) < 1e-6


def test_demodulate_ignores_a_constant_level():
    plain, _ = demodulate(_series(12.0), DAY)
    lifted, _ = demodulate(_series(12.0) + 500.0, DAY)
    assert np.allclose(plain[30:], lifted[30:], atol=1e-9)


def test_the_shape_of_the_day_does_not_move_the_phase():
    # A Gaussian bump has a strong second harmonic; it must not leak into the
    # phase — the window is exactly one revolution, so integer harmonics of the
    # day sum to zero over it. What is left (1.5e-6 rad, a tenth of a second) is
    # the wrap kink of the sampled bump, not the harmonic.
    hours = np.arange(BUCKETS, dtype=float)
    bump = _bump(hours, 11.2)
    fundamental = 0.5 * np.cos(2.0 * np.pi * (hours - 11.2) / DAY)
    phi_bump, _ = demodulate(bump, DAY)
    phi_fundamental, _ = demodulate(fundamental, DAY)
    error = np.abs(np.angle(np.exp(1j * (phi_bump[30:] - phi_fundamental[30:]))))
    assert float(error.max()) < 1e-5


def test_a_taper_of_one_revolution_is_not_orthogonal():
    """The reason ``demodulate`` is rectangular, measured on the same series.

    A Hann taper of 24 points has its own period at 23, so its spectrum reaches
    the bin next to the day's fundamental and the second harmonic lands back in
    the reading: the phase moves by up to ~1 rad (median ~0.5 rad — two hours
    invented by the window). The rectangular one-revolution window cannot do
    that, which is exactly why it was chosen.
    """
    hours = np.arange(BUCKETS, dtype=float)
    bump = _bump(hours, 11.2)
    fundamental = 0.5 * np.cos(2.0 * np.pi * (hours - 11.2) / DAY)
    taper = np.hanning(int(DAY))
    error = np.abs(
        np.angle(
            np.exp(
                1j
                * (
                    _demodulate_tapered(bump, taper)
                    - _demodulate_tapered(fundamental, taper)
                )
            )
        )
    )[30:]
    assert float(np.median(error)) > 0.2
    # ... while the rectangular window reads the same phase as the fundamental.
    rectangular = np.abs(
        np.angle(
            np.exp(
                1j
                * (
                    demodulate(bump, DAY)[0][30:]
                    - demodulate(fundamental, DAY)[0][30:]
                )
            )
        )
    )
    assert float(rectangular.max()) < 1e-5



def test_demodulate_sees_a_shift_as_a_phase_step():
    base, _ = demodulate(_series(12.0), DAY)
    shifted, _ = demodulate(_series(16.0), DAY)
    step = np.abs(np.angle(np.exp(1j * (shifted[40:] - base[40:]))))
    assert float(np.median(step)) == pytest.approx(4.0 * 2.0 * np.pi / DAY, rel=0.02)


def test_demodulate_is_blind_to_level():
    phase, envelope = demodulate(_series(12.0), DAY)
    scaled_phase, scaled_envelope = demodulate(_series(12.0) * 1.6, DAY)
    assert np.allclose(phase[30:], scaled_phase[30:], atol=1e-9)
    assert np.allclose(scaled_envelope[30:], 1.6 * envelope[30:], rtol=1e-9)


def test_demodulate_needs_one_revolution():
    with pytest.raises(ValueError):
        demodulate(np.ones(10), DAY)


# ── the park ────────────────────────────────────────────────────────────────


def test_a_park_needs_two_producers():
    with pytest.raises(ValueError):
        observe_park(_series()[None, :])


def test_a_park_needs_a_revolution():
    with pytest.raises(ValueError):
        observe_park(np.vstack([_series()[:20], _series()[:20]]))


def test_gaps_are_refused_rather_than_guessed():
    energy = np.vstack([_series(), _series()])
    energy[1, 10] = np.nan
    with pytest.raises(ValueError):
        observe_park(energy)


def test_order_is_one_for_identical_producers_and_low_for_opposites():
    together = observe_park(np.vstack([_series(12.0)] * 3))
    assert float(np.mean(together.order[30:])) == pytest.approx(1.0, abs=1e-6)
    opposed = observe_park(np.vstack([_series(12.0), _series(0.0), _series(12.0)]))
    assert float(np.mean(opposed.order[30:])) < 0.5


def test_outage_reads_as_silence():
    energy = np.vstack([_series()] * 4)
    energy[1, 120:144] = 0.0
    observation = observe_park(energy)
    assert observation.readings["silent_fraction"][1] > 0.2
    assert observation.readings["silent_fraction"][0] == 0.0
    assert observation.scores[1] >= observation.config.threshold
    assert "silence" in observation.reasons(1)


def test_a_shorter_day_reads_as_a_different_frequency():
    hours = np.arange(336, dtype=float)
    slow = _bump(hours, 12.0)
    fast = 500.0 * (1.0 + np.cos(2.0 * np.pi * hours / 20.0))
    observation = observe_park(
        np.vstack([slow, slow, slow, fast]), config=ParkConfig(baseline_hours=192.0)
    )
    assert observation.readings["period_hours"][3] == pytest.approx(20.0, abs=0.1)
    assert observation.readings["resonance_s"][3] < 0.6
    assert observation.readings["resonance_s"][0] == pytest.approx(1.0, abs=1e-6)
    assert "resonance" in observation.reasons(3)
    assert observation.scores[0] < 0.1


def test_a_producer_that_was_always_offset_is_not_an_anomaly():
    # The park's own shape is not a change: a west-facing producer that never
    # moved is read as part of the chord, not as an anomaly.
    observation = observe_park(
        np.vstack([_series(12.0), _series(12.0), _series(12.0), _series(16.0)])
    )
    assert observation.scores[3] < observation.config.threshold
    assert observation.readings["phase_drift_rad"][3] == pytest.approx(0.0, abs=0.05)
    assert observation.reasons(3) == []


def test_a_step_in_the_middle_is_located_within_a_day():
    energy = np.vstack([_series(12.0)] * 4)
    energy[2, 96:] = _series(16.0)[96:]
    observation = observe_park(energy)
    assert "phase" in observation.reasons(2)
    since = observation.anomaly_since[2]
    assert since is not None
    assert 96 <= since <= 120  # the reading lags by at most one revolution


@pytest.mark.parametrize(
    "field,value",
    [
        ("bucket_minutes", 0),
        ("day_hours", 0.0),
        ("band_rad_per_hour", 0.0),
        ("memory_hours", 0.0),
        ("silent_ratio", 1.0),
        ("threshold", 0.0),
        ("memory_scale", 0.0),
        ("degree", 0),
    ],
)
def test_config_rejects_nonsense(field, value):
    with pytest.raises(ValueError):
        ParkConfig(**{field: value})


def test_config_derives_buckets_from_units():
    hourly = ParkConfig(bucket_minutes=60)
    quarter = ParkConfig(bucket_minutes=15)
    assert hourly.day_buckets == pytest.approx(24.0)
    assert quarter.day_buckets == pytest.approx(96.0)
    assert hourly.baseline_buckets == 96
    assert quarter.baseline_buckets == 384
    assert hourly.epsilon * hourly.bucket_hours == pytest.approx(0.08)
