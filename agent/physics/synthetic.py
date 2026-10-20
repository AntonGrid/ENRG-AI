"""Deterministic synthetic park: a clean week and declared injections.

No telemetry is used here and none is claimed: this is a model, with the same
honesty as the book's own scenes. What it is for is the *criterion* — the
injections are declared before the reading, and ``benchmark.py`` says which of
them this physics catches and which it cannot. A detector that is only ever
shown what it catches proves nothing.

The park: ``devices`` producers on one site, each with a daily bump, small
differences of longitude/orientation (so the park has a chord of its own),
slow cloud modulation and 3 % noise. The injections are applied to the same
draws, so a shifted device is the *same* device with a different timing, not a
different random series.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

#: The four declared kinds of change. Names are stable: a report may print them.
OUTAGE = "outage"
PEAK_SHIFT = "peak_shift"
CLOCK_OFFSET = "clock_offset"
LEVEL_SCALE = "level_scale"
CLEAN = "clean"


@dataclass(frozen=True)
class Injection:
    """One declared change: what, where, when and how much."""

    device_id: str
    kind: str
    start: int  # bucket index
    magnitude: float


@dataclass
class Scenario:
    """A park's energy matrix plus the truth about what was done to it."""

    name: str
    device_ids: List[str]
    starts: List[dt.datetime]
    energy: np.ndarray
    truth: Dict[str, str]
    injections: List[Injection] = field(default_factory=list)
    seed: int = 7

    def injection(self, device_id: str) -> Optional[Injection]:
        for injection in self.injections:
            if injection.device_id == device_id:
                return injection
        return None

    def truth_of(self, device_id: str) -> str:
        return self.truth.get(device_id, CLEAN)


@dataclass
class _Clean:
    """The draws a park is made of, kept so injections stay the same device."""

    energy: np.ndarray
    amplitude: np.ndarray
    peak: np.ndarray
    cloud: np.ndarray
    noise: np.ndarray
    hours: np.ndarray


def _bump(hours, peak_hour: float, width_hours: float = 3.2):
    """One day's bump: a smooth daylight lobe centred on ``peak_hour``."""
    hours = np.asarray(hours, dtype=float)
    distance = np.abs((hours - peak_hour + 12.0) % 24.0 - 12.0)
    return np.exp(-0.5 * (distance / width_hours) ** 2)


def _render(amplitude, cloud, noise, index, peak_hour, hours, sl) -> np.ndarray:
    """Render one producer's series with a given peak, same clouds and noise."""
    values = (
        amplitude[index]
        * _bump(hours[sl], peak_hour)
        * (cloud[index, sl] + noise[index, sl])
    )
    return np.maximum(values, 0.0)


def _timeline(
    buckets: int, bucket_minutes: int, ends_at: Optional[dt.datetime]
) -> List[dt.datetime]:
    """Bucket starts, aligned to the grid and ending at ``ends_at`` (now)."""
    end = ends_at or dt.datetime.now(dt.timezone.utc)
    step = dt.timedelta(minutes=bucket_minutes)
    end = end.replace(second=0, microsecond=0)
    end = end.replace(minute=(end.minute // bucket_minutes) * bucket_minutes)
    return [end - (buckets - 1 - i) * step for i in range(buckets)]


def clean_park(
    seed: int = 7,
    devices: int = 8,
    buckets: int = 168,
    bucket_minutes: int = 60,
    peak_hour: float = 12.0,
    amplitude_wh: float = 5000.0,
    spread_hours: float = 0.8,
    cloud_depth: float = 0.25,
    noise_level: float = 0.03,
) -> _Clean:
    """A week of a park that is doing nothing wrong.

    The producers differ by ``spread_hours`` in either direction (longitude and
    orientation — the park's own chord), by up to ±15 % in size, and are
    modulated by a slow cloud pattern. Deterministic per ``seed``.
    """
    if devices < 2:
        raise ValueError("a park is at least two producers")
    if buckets < 27:
        raise ValueError("a park is at least a revolution plus a reading")
    rng = np.random.default_rng(seed)
    hours = np.arange(buckets, dtype=float) * (bucket_minutes / 60.0)
    offsets = np.linspace(-spread_hours, spread_hours, devices)
    amplitude = amplitude_wh * rng.uniform(0.85, 1.15, devices)
    peak = peak_hour + offsets
    cloud = np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(3) / 3.0, mode="same"),
        1,
        1.0 - cloud_depth * rng.random((devices, buckets)),
    )
    noise = rng.normal(0.0, noise_level, (devices, buckets))
    energy = np.stack(
        [
            _render(amplitude, cloud, noise, i, peak[i], hours, slice(None))
            for i in range(devices)
        ]
    )
    return _Clean(energy, amplitude, peak, cloud, noise, hours)


def build_scenario(
    seed: int = 7,
    devices: int = 8,
    buckets: int = 168,
    bucket_minutes: int = 60,
    ends_at: Optional[dt.datetime] = None,
    scenario: str = "mixed",
) -> Scenario:
    """A week with four declared changes — one of each kind (see the benchmark).

    ``scenario="clean"`` returns the same park with nothing done to it, so a
    report can show what a park that is fine looks like.
    """
    if scenario not in ("clean", "mixed"):
        raise ValueError("scenario must be 'clean' or 'mixed'")
    if scenario == "mixed" and devices < 8:
        raise ValueError("the mixed scenario needs eight producers")
    clean = clean_park(
        seed=seed, devices=devices, buckets=buckets, bucket_minutes=bucket_minutes
    )
    device_ids = [f"dev-{i:02d}" for i in range(devices)]
    energy = np.array(clean.energy, dtype=float, copy=True)
    hours = clean.hours
    truth = {device_id: CLEAN for device_id in device_ids}
    injections: List[Injection] = []

    if scenario == "mixed":
        # 1. A producer stops: not a spike, not a gap in its nonces — its day
        #    simply stops sounding for a day.
        energy[2, 120:144] = 0.0
        injections.append(Injection(device_ids[2], OUTAGE, 120, 1.0))
        truth[device_ids[2]] = OUTAGE

        # 2. A physical change: the day moves four hours later (re-aimed or
        #    shaded panels) — the level is unchanged, the timing is not.
        energy[5, 96:] = _render(
            clean.amplitude, clean.cloud, clean.noise, 5, clean.peak[5] + 4.0,
            hours, slice(96, None),
        )
        injections.append(Injection(device_ids[5], PEAK_SHIFT, 96, 4.0))
        truth[device_ids[5]] = PEAK_SHIFT

        # 3. A timestamp lie: the gateway rewrites ts by +2 h. The reports are
        #    signed and the nonces are in order — only the phase can see it.
        energy[6, 96:] = _render(
            clean.amplitude, clean.cloud, clean.noise, 6, clean.peak[6],
            hours + 2.0, slice(96, None),
        )
        injections.append(Injection(device_ids[6], CLOCK_OFFSET, 96, 2.0))
        truth[device_ids[6]] = CLOCK_OFFSET

        # 4. A level change: the same day, multiplied by 1.6. Phase-invisible
        #    by construction — the book's state is frequency, not level.
        energy[7, 120:] = clean.energy[7, 120:] * 1.6
        injections.append(Injection(device_ids[7], LEVEL_SCALE, 120, 1.6))
        truth[device_ids[7]] = LEVEL_SCALE

    return Scenario(
        name=scenario,
        device_ids=device_ids,
        starts=_timeline(buckets, bucket_minutes, ends_at),
        energy=energy,
        truth=truth,
        injections=injections,
        seed=seed,
    )
