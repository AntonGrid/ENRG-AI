"""Reading a park of producers as one medium (Entrain's physics over proofs).

A park is a medium in the book's sense: every producer is a vibration whose
state is a frequency, and what one producer has in common with another is
written into the link between them. Nothing here decides anything — it reads
four things and hands them on:

* ``silence`` — the size of a producer's daily oscillation fell away: the
  vibration stopped sounding;
* ``resonance`` — does it still sound at the park's frequency? ``S(dw, Pi)``;
* ``phase`` — has it moved from where it was, relative to the park (a device
  that was always west-facing is not an anomaly: the park's own shape is not
  a change);
* ``memory`` — did its *relation* to the other producers change? This is the
  written link ``W``: what passed through it, not what was announced.

The reading is a *day's* reading: a producer's phase is demodulated over
exactly one revolution of the park's clock (24 h by default), so a value at
time ``t`` is what the last day sounded like. That is honest about latency —
to hear a frequency you must listen for a period. The window is rectangular
and exactly one revolution long, so integer harmonics of the day and the
constant daylight offset sum to zero over it and cannot leak into the phase.
A taper of the same length would not have that property: a Hann window of N
points has its own period at N-1 and mixes the second harmonic back into the
phase (measured, not assumed — see ``tests/test_physics_park.py``).

Limitations, stated where they are made:

* the phase is answered from the trailing day, so a step at ``t0`` is smeared
  over one revolution (it reads half-way at ``t0 + 12 h``);
* the reading is blind to level by construction: a producer that multiplies
  its reports by 1.6 has the same phase. That is not a defect of the port —
  the book's state is frequency. Level stays with the MAD detectors, and the
  benchmark says so with numbers instead of adjectives.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from agent.physics.oscillator import MEMORY_REFERENCE, coherence, memory_goal, overlap

#: Names of the four readings, in the order they are reported.
COMPONENTS = ("silence", "resonance", "phase", "memory")


@dataclass(frozen=True)
class ParkConfig:
    """How a park is read. Units are explicit; buckets are derived.

    ``bucket_minutes`` is the width of one bucket; ``day_hours`` is the park's
    clock; ``band_rad_per_hour`` is Pi (0.05 rad/h: two days that differ by
    about four hours read as different frequencies); ``memory_hours`` is a
    written link's lifetime (12.5 h, so a link follows a week's weather and not
    one cloudy noon); ``link`` and ``reference`` are the book's numbers
    (``basic.jl:143``, ``basic.jl:66``); ``degree`` is K, default N.

    ``baseline_hours`` teaches the park what "normal" is and the reading window
    is everything after it; ``warmup_hours`` is the revolution the demodulator
    needs before its envelope means anything. ``threshold`` is the score above
    which a producer is flagged, and the four ``*_scale`` values are the levels
    that read as unambiguous: a quarter of the window silent, pi/4 of movement
    (three hours), 0.4 of the full link, and a loss of three quarters of S.
    """

    bucket_minutes: int = 60
    day_hours: float = 24.0
    band_rad_per_hour: float = 0.05
    memory_hours: float = 12.5
    link: float = 0.6
    reference: float = MEMORY_REFERENCE
    degree: Optional[int] = None
    silent_ratio: float = 0.35
    baseline_hours: float = 96.0
    warmup_hours: float = 24.0
    threshold: float = 0.5
    silence_scale: float = 0.25
    phase_scale_rad: float = math.pi / 4.0
    memory_scale: float = 0.4
    resonance_scale: float = 0.75

    def __post_init__(self) -> None:
        if int(self.bucket_minutes) <= 0:
            raise ValueError("bucket_minutes must be positive")
        if self.day_hours <= 0:
            raise ValueError("day_hours must be positive")
        if self.band_rad_per_hour <= 0:
            raise ValueError("band_rad_per_hour must be positive")
        if self.memory_hours <= 0:
            raise ValueError("memory_hours must be positive")
        if not 0.0 < self.silent_ratio < 1.0:
            raise ValueError("silent_ratio must be in (0, 1)")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        for name in ("silence_scale", "phase_scale_rad", "memory_scale", "resonance_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.degree is not None and int(self.degree) < 1:
            raise ValueError("degree must be at least 1")

    @property
    def bucket_hours(self) -> float:
        return float(self.bucket_minutes) / 60.0

    @property
    def day_buckets(self) -> float:
        return self.day_hours / self.bucket_hours

    @property
    def warmup_buckets(self) -> int:
        return max(1, int(round(self.warmup_hours / self.bucket_hours)))

    @property
    def baseline_buckets(self) -> int:
        return max(
            self.warmup_buckets + 1,
            int(round(self.baseline_hours / self.bucket_hours)),
        )

    @property
    def epsilon(self) -> float:
        """Memory rate per hour: a link forgets with lifetime ``memory_hours``."""
        return 1.0 / float(self.memory_hours)


def demodulate(values, day_buckets: float):
    """Phase offset and size of the daily oscillation of one series.

    ``phi[t]`` is the phase of the day-long component at bucket ``t``, read
    from the trailing revolution: ``x(t') ~ A cos(omega0 * t' + phi)``. The
    window is exactly one revolution and rectangular, so the constant level of
    the series and every integer harmonic of the day sum to zero over it.

    Returns:
        ``(phi, envelope)`` — radians (relative to the park's clock) and the
        size of the daily mode, in the units of the series.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("demodulate reads one series at a time")
    window = int(round(float(day_buckets)))
    if window < 2:
        raise ValueError("a revolution must be at least two buckets")
    if x.size < window + 1:
        raise ValueError(
            f"need at least one revolution plus a bucket ({window + 1}), got {x.size}"
        )
    omega0 = 2.0 * math.pi / float(day_buckets)
    phi = np.zeros(x.size, dtype=float)
    envelope = np.zeros(x.size, dtype=float)
    for t in range(x.size):
        k = min(window, t + 1)
        # seg[j] = x(t - j), so the demodulation factor is exp(-i*omega0*(t-j)).
        seg = x[t - k + 1 : t + 1][::-1]
        z = np.sum(seg * np.exp(-1j * omega0 * (t - np.arange(k)))) / k
        phi[t] = np.angle(z)
        envelope[t] = 2.0 * abs(z)
    return phi, envelope


def _circular_mean(values) -> float:
    """Mean direction of a set of angles (radians); 0.0 for an empty set."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.angle(np.mean(np.exp(1j * arr))))


def _circular_distance(a, b):
    """Distance between two angles on the circle, in ``[0, pi]``."""
    return np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))


@dataclass
class ParkObservation:
    """What a park sounded like: one row per producer, one value per reading."""

    device_ids: List[str]
    starts: List[dt.datetime]
    config: ParkConfig
    theta: np.ndarray  # (N, T) relative phase — what cos() is taken on
    omega: np.ndarray  # (N,)   rad per hour
    envelope: np.ndarray  # (N, T) size of the daily mode
    silent: np.ndarray  # (N, T) bool
    order: np.ndarray  # (T,)   R = |mean e^{i theta}|
    park_phase: np.ndarray  # (T,)   mean direction of the park
    coupling: np.ndarray  # (N, N) written links at the end
    baseline_coupling: np.ndarray  # (N, N) written links of "normal"
    window: slice
    readings: Dict[str, np.ndarray]
    components: Dict[str, np.ndarray]
    scores: np.ndarray
    anomaly_since: List[Optional[int]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def flagged(self) -> List[int]:
        """Indices of producers whose score cleared the threshold."""
        return [i for i, s in enumerate(self.scores) if float(s) >= self.config.threshold]

    def reasons(self, index: int) -> List[str]:
        """Component names that cleared the threshold for one producer."""
        return [
            name
            for name in COMPONENTS
            if float(self.components[name][index]) >= self.config.threshold
        ]

    def rows(self) -> List[Dict[str, Any]]:
        """Per-device reading, JSON-ready (numpy floats turned into floats)."""
        out: List[Dict[str, Any]] = []
        for i, device_id in enumerate(self.device_ids):
            since = self.anomaly_since[i] if i < len(self.anomaly_since) else None
            out.append(
                {
                    "device_id": device_id,
                    "score": round(float(self.scores[i]), 4),
                    "is_anomaly": bool(float(self.scores[i]) >= self.config.threshold),
                    "reasons": self.reasons(i),
                    "components": {
                        name: round(float(self.components[name][i]), 4)
                        for name in COMPONENTS
                    },
                    "readings": {
                        key: round(float(values[i]), 6)
                        for key, values in self.readings.items()
                    },
                    "anomaly_since": (
                        self.starts[since].isoformat(timespec="minutes")
                        if since is not None and since < len(self.starts)
                        else None
                    ),
                }
            )
        return out

    def park_reading(self) -> Dict[str, float]:
        """The park as one thing: its order over the reading window."""
        window = self.order[self.window]
        return {
            "coherence_mean": round(float(np.mean(window)) if window.size else 0.0, 4),
            "coherence_min": round(float(np.min(window)) if window.size else 0.0, 4),
            "silent_fraction": round(
                float(np.mean(self.silent[:, self.window])) if window.size else 0.0, 4
            ),
            "producers": len(self.device_ids),
            "window_buckets": int(self.window.stop - self.window.start),
        }


def _advance(theta, silent, config, start, stop, record_from=0, coupling=None, reference=None):
    """Step the written links over ``[start, stop)`` and accumulate a reading.

    The links are symmetric — two producers that both see the same sun write
    each other — and the target is the book's ``(link + N0 * cos) / K``
    (``basic.jl:312``), written with rate ``eps * dt`` (``basic.jl:358``).
    When either side is silent nothing is heard on that link, so it keeps what
    it had: a memory does not invent a coincidence it did not hear.

    ``reference`` is what the reading is measured against (the mean link of the
    baseline); without it the accumulator sums the links themselves.
    """
    n = theta.shape[0]
    k = int(config.degree or n)
    if coupling is None:
        written = np.full((n, n), config.link / k)
    else:
        written = np.array(coupling, dtype=float, copy=True)
    total = np.zeros((n, n))
    count = 0
    eps_dt = config.epsilon * config.bucket_hours
    for t in range(start, stop):
        if t > 0:
            for i in range(n):
                for j in range(i + 1, n):
                    if silent[i, t] or silent[j, t]:
                        continue
                    target = memory_goal(
                        config.link,
                        k,
                        math.cos(theta[i, t] - theta[j, t]),
                        1.0,
                        written[i, j],
                        config.reference,
                    )
                    written[i, j] = written[j, i] = written[i, j] + eps_dt * (
                        target - written[i, j]
                    )
        if t >= record_from:
            total += written if reference is None else np.abs(written - reference)
            count += 1
    return written, total, count


def observe_park(energy, device_ids=None, starts=None, config=None) -> ParkObservation:
    """Read a park of producers from a bucket-by-bucket energy matrix.

    Args:
        energy: ``(producers, buckets)`` energy per bucket; two or more
            producers are required (coherence is a relation, not a property of
            one device).
        device_ids: names, in row order; default ``dev-00, dev-01, ...``.
        starts: bucket timestamps; default a fixed epoch grid, so the same
            numbers always read the same reading.
        config: :class:`ParkConfig`.

    Returns:
        :class:`ParkObservation` — phases, written links, four readings per
        producer and the park's order over the window.

    Raises:
        ValueError: fewer than two producers, a shorter series than one
        revolution plus a reading window, or a series with gaps in it (fill
        gaps with zero first — a gap is "no energy reported", not "nothing").
    """
    cfg = config or ParkConfig()
    energy = np.asarray(energy, dtype=float)
    if energy.ndim == 1:
        energy = energy[None, :]
    if energy.ndim != 2:
        raise ValueError("energy must be (producers, buckets)")
    if np.isnan(energy).any():
        raise ValueError("energy has gaps; fill them before reading a park")
    producers, buckets = energy.shape
    if producers < 2:
        raise ValueError("a park is at least two producers: coherence is a relation")
    need = int(math.ceil(cfg.day_buckets)) + 3
    if buckets < need:
        raise ValueError(
            f"need at least {need} buckets (one revolution plus a reading), got {buckets}"
        )
    if device_ids is None:
        device_ids = [f"dev-{i:02d}" for i in range(producers)]
    device_ids = list(device_ids)
    if len(device_ids) != producers:
        raise ValueError("device_ids must name every row")
    if starts is None:
        step = dt.timedelta(minutes=cfg.bucket_minutes)
        starts = [
            dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + i * step
            for i in range(buckets)
        ]
    starts = list(starts)
    if len(starts) != buckets:
        raise ValueError("starts must label every bucket")

    warnings: List[str] = []
    baseline_end = min(cfg.baseline_buckets, buckets - 1)
    warmup = min(cfg.warmup_buckets, max(1, baseline_end // 2))
    if baseline_end < cfg.baseline_buckets:
        warnings.append(
            f"baseline clamped to {baseline_end} buckets: the series is shorter than "
            f"{cfg.baseline_hours:.0f} h"
        )
    baseline_slice = slice(warmup, baseline_end)

    # 1. Phase and size of every producer's day.
    phi = np.empty((producers, buckets))
    envelope = np.empty((producers, buckets))
    for i in range(producers):
        phi[i], envelope[i] = demodulate(energy[i], cfg.day_buckets)

    # 2. Silence: the size fell away against the producer's own baseline.
    reference_size = np.median(envelope[:, baseline_slice], axis=1)
    silent = np.zeros((producers, buckets), dtype=bool)
    for i in range(producers):
        if reference_size[i] <= 0.0:
            silent[i, :] = True
            warnings.append(
                f"{device_ids[i]}: no daily size in the baseline — silent throughout"
            )
        else:
            silent[i, :] = envelope[i] < cfg.silent_ratio * reference_size[i]

    # 3. Frequency: the park's day, corrected by the phase slope of each producer.
    hours = np.arange(buckets) * cfg.bucket_hours
    day_omega = 2.0 * math.pi / cfg.day_hours
    omega = np.full(producers, day_omega)
    for i in range(producers):
        alive = ~silent[i, baseline_slice]
        if int(alive.sum()) >= 3:
            slope = np.polyfit(
                hours[baseline_slice][alive],
                np.unwrap(phi[i, baseline_slice][alive]),
                1,
            )[0]
            omega[i] = day_omega + float(slope)

    # 4. Phase with the frequency taken out, then relative to the park.
    phit = phi - (omega - day_omega)[:, None] * hours[None, :]
    park_phase = np.zeros(buckets)
    for t in range(buckets):
        alive = ~silent[:, t]
        if alive.any():
            park_phase[t] = _circular_mean(phit[alive, t])
    theta = np.angle(np.exp(1j * (phit - park_phase[None, :])))

    # 5. Does it still sound at the park's frequency?
    detune = np.abs(omega - float(np.median(omega)))
    resonance_s = np.asarray(overlap(detune, cfg.band_rad_per_hour), dtype=float)

    # 6. Did the relation change? Written links of "normal", then of the window.
    written_at_baseline, base_total, base_count = _advance(
        theta, silent, cfg, 0, baseline_end, record_from=warmup
    )
    baseline_coupling = base_total / max(1, base_count)
    written, div_total, div_count = _advance(
        theta,
        silent,
        cfg,
        baseline_end,
        buckets,
        coupling=written_at_baseline,
        reference=baseline_coupling,
    )
    divergence = div_total / max(1, div_count)
    w_scale = (cfg.link + cfg.reference) / float(cfg.degree or producers)
    memory_divergence = np.array(
        [
            float(np.mean([divergence[i, j] for j in range(producers) if j != i]))
            / w_scale
            for i in range(producers)
        ]
    )

    # 7. Movement from its own baseline position, relative to the park's own.
    base_off = np.array(
        [
            _circular_mean(theta[i, baseline_slice][~silent[i, baseline_slice]])
            for i in range(producers)
        ]
    )
    moved = np.angle(np.exp(1j * (theta - base_off[:, None])))
    common = np.array(
        [_circular_mean(moved[~silent[:, t], t]) for t in range(buckets)]
    )
    drift = _circular_distance(moved, common[None, :])

    # 8. The park's order — one number per bucket, over those still sounding.
    order = np.zeros(buckets)
    for t in range(buckets):
        alive = ~silent[:, t]
        if int(alive.sum()) >= 2:
            order[t] = coherence(theta[alive, t])

    # 9. Read the window.
    window = slice(baseline_end, buckets)
    silent_fraction = silent[:, window].mean(axis=1)
    phase_drift = np.zeros(producers)
    for i in range(producers):
        alive = ~silent[i, window]
        values = drift[i, window][alive]
        phase_drift[i] = float(values.mean()) if values.size else 0.0

    readings: Dict[str, np.ndarray] = {
        "silent_fraction": np.asarray(silent_fraction, dtype=float),
        "phase_drift_rad": phase_drift,
        "memory_divergence": memory_divergence,
        "resonance_s": resonance_s,
        "detune_rad_per_hour": detune,
        "period_hours": 2.0 * math.pi / omega,
        "coherence": np.full(producers, float(np.mean(order[window])) if order[window].size else 0.0),
    }

    components: Dict[str, np.ndarray] = {
        "silence": np.clip(readings["silent_fraction"] / cfg.silence_scale, 0.0, 1.0),
        "resonance": np.clip(
            (1.0 - readings["resonance_s"]) / cfg.resonance_scale, 0.0, 1.0
        ),
        "phase": np.clip(readings["phase_drift_rad"] / cfg.phase_scale_rad, 0.0, 1.0),
        "memory": np.clip(readings["memory_divergence"] / cfg.memory_scale, 0.0, 1.0),
    }
    scores = np.max(np.vstack([components[name] for name in COMPONENTS]), axis=0)

    # 10. When did it first read as a change? (Where the reading crossed half
    # the declared scale, or went silent — the memory follows, it does not lead.)
    half_phase = cfg.phase_scale_rad / 2.0
    anomaly_since: List[Optional[int]] = []
    for i in range(producers):
        found: Optional[int] = None
        for t in range(window.start, window.stop):
            if bool(silent[i, t]) or float(drift[i, t]) > half_phase:
                found = t
                break
        anomaly_since.append(found)

    return ParkObservation(
        device_ids=device_ids,
        starts=starts,
        config=cfg,
        theta=theta,
        omega=omega,
        envelope=envelope,
        silent=silent,
        order=order,
        park_phase=park_phase,
        coupling=written,
        baseline_coupling=baseline_coupling,
        window=window,
        readings=readings,
        components=components,
        scores=scores,
        anomaly_since=anomaly_since,
        warnings=warnings,
    )

