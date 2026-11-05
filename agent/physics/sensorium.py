"""Organs of sense: the world heard through many feeds, the self through dreams.

The organism has exactly two kinds of input and one duty about them: **keep
their provenance**. A stream that came from outside is :data:`WORLD`; a stream
the organism produced is :data:`SELF`. Every perception carries the SHA-256 of
its own values, so nothing downstream has to trust a label — the ledger, the
comparator and the diagnosis all key on the bytes, not on a promise.

World feeds, in four domains, because a system fed on one noise type learns
that noise and calls it the world:

=========================  ===================================================
``listen_finance``         daily/hourly closes (BTC-USD, ...) — a market
``listen_space_weather``   NOAA planetary K index — the Earth's own field
``listen_solar``           the sunspot record, monthly since 1749 — astronomy
``listen_attention``       Wikimedia pageviews — collective human attention
``listen_commits``         GitHub commit cadence — collective human work
``listen_dataset``         sklearn's bundled measurements — offline, always there
``listen_file``            any CSV/TXT/NPY on disk — the operator's own data
=========================  ===================================================

Dreams, in four kinds, and the last one is the dangerous one:

=========================  ===================================================
:data:`DREAM_RED`          AR(1): the organism's own Markov voice, one slope
:data:`DREAM_RHYTHM`       its own clock: a few sinusoids, quantized, jittered
:data:`DREAM_REPLAY`       memory of the world plus its own noise, anchored by λ
:data:`DREAM_SURROGATE`    the world's exact spectrum *and* exact marginal
                           distribution with its phases randomised
=========================  ===================================================

A phase-randomised surrogate is a counterfeit no linear statistic can refuse:
it has the same periodogram and the same histogram as the stream it mimics.
An organism that claims to tell it from the world is measuring its own prior.
The self-model reports that number instead of hiding it, and the notebook says
so out loud.

Organs — the reading itself, and both hand the core its own object:

``Cochlea``   eight log-spaced bands; band ``j`` reports how far its energy
              centroid sits from the band's centre, in units of the organ's
              whole band width. A medium at rest reports ~0 in every band; a
              voice *between* the bands pulls one of them.
``Segments``  the window cut into eight parts; part ``j`` reports its level
              against the window's own scale. A step in the stream moves one
              part and leaves the other seven where they were.

``sigma`` is always measured on the stream itself — the medium's own noise,
taken as MAD x 1.4826 over all of that stream's fixations. It is never guessed:
a sensor with too small a sigma would call every ordinary reading a liar.

Declared limits:

* a stream shorter than :data:`MIN_SAMPLES` cannot be fixated (the FFT of 64
  points has 32 bins, and eight log-spaced bands then have nothing to average);
* the feeds reach the network unless a cache is warm or a parser is called
  directly; :class:`Sensorium` never invents values to fill a gap — an absent
  feed is a warning, not a simulation, and the pool then says which domains it
  actually has;
* ``listen_dataset`` gives a *recorded sequence*, not a time series. Its rows
  are samples of one instrument, flattened in order, and the docstring says so.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from agent.physics.genome import core as _core
from agent.physics.genome import observation as _core_observation

from functools import cached_property

#: Outside the organism. Everything it did not produce itself.
WORLD = "world"
#: The organism's own output. Everything it produced.
SELF = "self"

DOMAIN_FINANCE = "finance"
DOMAIN_PHYSICS = "physics"
DOMAIN_BEHAVIOR = "behavior"
DOMAIN_RECORDED = "recorded"
DOMAIN_SELF = "self"

#: Domains a world pool must cover to be worth calibrating on. Fewer than three
#: and the organism is learning one noise type; :meth:`Sensorium.warnings` says
#: which domain is missing instead of quietly proceeding.
WORLD_DOMAINS = (DOMAIN_FINANCE, DOMAIN_PHYSICS, DOMAIN_BEHAVIOR)

DREAM_RED = "red"
DREAM_RHYTHM = "rhythm"
DREAM_REPLAY = "replay"
DREAM_SURROGATE = "surrogate"
DREAMS = (DREAM_RED, DREAM_RHYTHM, DREAM_REPLAY, DREAM_SURROGATE)

#: Shortest stream the organs can fixate: eight bands need resolvable bins.
MIN_SAMPLES = 64
#: Samples per fixation and the hop between fixations.
WINDOW = 128
STRIDE = 64
#: Samples either side of an anchor a probe asks for by default: "a hundred
#: steps before the anomaly and a hundred after", in one number.
PROBE_CONTEXT = 96
#: What a probe can ask for. The names are the same words
#: :data:`agent.physics.curiosity.KINDS` uses, so a question and the answer to it
#: are spelled alike — but they are repeated here as literals on purpose: the
#: sensorium's job is to cut a slice, and it must be able to do that without
#: importing the thing that decides the slice is worth cutting.
PROBE_AROUND = "context"
PROBE_AFTER = "next"
PROBE_TAIL = "compare"
PROBE_KINDS = (PROBE_AROUND, PROBE_AFTER, PROBE_TAIL)
#: Channels per organ. Fixed, because the core's null table depends on it.
BANDS = 8
#: Band edges of the cochlea, in cycles per sample.
F_MIN = 1.0 / 32.0
F_MAX = 0.5

#: The four shape numbers a window reports, in the declared order the
#: comparator consumes them. ``chord`` and ``peak`` are added by
#: :func:`core_shape`, because they are the core's statistics of the reading
#: itself and nothing about them may be re-derived here.
SIGNATURE_FIELDS = ("volatility", "tail", "slope", "reversibility", "stillness")
#: Everything the posture's two models see, in one frozen order. The first two
#: are the core's; the last four are the window's shape.
FEATURES = ("chord", "peak") + SIGNATURE_FIELDS


class SensorError(RuntimeError):
    """A feed could not be heard: no payload, or a payload that is not numbers."""


def _digest(values: np.ndarray) -> str:
    """Content address of a stream: the ledger keys on this, nothing else."""
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class Perception:
    """One stream, as heard, with the provenance glued to the values.

    ``values`` is a 1-D float64 array; ``origin`` is :data:`WORLD` or
    :data:`SELF`; ``fingerprint`` is the SHA-256 of the values and is computed
    if not given. ``degraded`` marks a stream that reached the organism by a
    fallback route (a cache, another symbol, a bundled dataset) — the number may
    be old, so it travels flagged rather than silently trusted.
    """

    values: np.ndarray
    origin: str
    source: str
    domain: str
    label: str = ""
    fingerprint: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    degraded: bool = False

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float64).ravel())
        if self.origin not in (WORLD, SELF):
            raise SensorError(
                f"origin must be {WORLD!r} or {SELF!r}, got {self.origin!r}"
            )
        if values.size < MIN_SAMPLES:
            raise SensorError(
                f"{self.source}: {values.size} samples, fewer than the "
                f"{MIN_SAMPLES} an organ needs to fixate"
            )
        if not np.all(np.isfinite(values)):
            raise SensorError(f"{self.source}: stream contains NaN or infinity")
        object.__setattr__(self, "values", values)
        if not self.fingerprint:
            object.__setattr__(self, "fingerprint", _digest(values))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def is_world(self) -> bool:
        return self.origin == WORLD

    def __len__(self) -> int:
        return int(self.values.size)

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "origin": self.origin,
            "domain": self.domain,
            "label": self.label,
            "samples": int(self.values.size),
            "mean": round(float(self.values.mean()), 6),
            "std": round(float(self.values.std(ddof=1)), 6),
            "fingerprint": self.fingerprint[:16],
            "degraded": self.degraded,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class Signature:
    """The shape of a window, in four numbers that survive a change of units.

    All four are dimensionless on purpose, so a market in dollars, a magnetic
    index in thirds of a unit and a pageview count in thousands are comparable
    without a normalisation table:

    ``volatility``     mean autocorrelation of ``|Δv|`` at lags 1..3. The world's
                       volatility has memory (a storm follows a storm); white
                       noise and phase-randomised surrogates have none.
    ``tail``           ``log1p`` of the excess kurtosis of the increments. Heavy
                       tails are a property of the world, not of a Gaussian dream.
    ``slope``          OLS slope of log power against log frequency: 0 is white,
                       -2 is a random walk, a line means one strong voice.
    ``reversibility``  ``mean(Δv³) / mean(|Δv|³)`` — ``+1`` the stream leans
                       forward, ``-1`` backwards, ``0`` symmetric. A
                       phase-randomised surrogate is symmetric by construction.
    ``stillness``      the share of increments that are *exactly* zero. A digital
                       medium repeats itself — two ends of a sensor reading the same
                       count, a magnetic index in thirds of a unit, a scan of a
                       handwritten digit — and a continuous one never does. Measured
                       to be the feature that separates a handwriting scan from a
                       quantised sinusoid, which four moments could not.
    """

    volatility: float = 0.0
    tail: float = 0.0
    slope: float = 0.0
    reversibility: float = 0.0
    stillness: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in SIGNATURE_FIELDS], dtype=float)

    def as_dict(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in SIGNATURE_FIELDS}


@dataclass(frozen=True)
class Reading:
    """One fixation: the core's observation of a window, plus its shape.

    ``start`` is where the window begins in the stream, in samples. It is carried
    because a body that decides to *look again* has to say where, and "the
    fourteenth fixation" is not an address — the stream may be re-read with a
    different geometry, and a probe has to name a place in the samples.
    """

    observation: Any  # entrain_infer.model.Observation; Any keeps the import lazy
    signature: Signature
    index: int = 0
    start: int = 0

    @property
    def x(self) -> np.ndarray:
        return np.asarray(self.observation.x, dtype=float)

    @property
    def sigma(self) -> float:
        return float(self.observation.sigma)

    def z(self) -> np.ndarray:
        return self.x / self.sigma

    @cached_property
    def features(self) -> np.ndarray:
        """The six judged numbers, computed once and remembered.

        A calibration evaluates one fixation hundreds of times — every grid cell,
        every mutation, every Powell step — and the features do not depend on λ, ε
        or α. Computing them once turns a thirty-second calibration into a
        two-second one, and ``cached_property`` writes straight into ``__dict__``,
        so it works on a frozen dataclass without weakening the frozen-ness.
        """
        return feature_vector(self)


@dataclass(frozen=True)
class Episode:
    """One stream heard through one organ: a trajectory, not a bag of samples.

    The posture is carried *through* an episode and onward across episodes, so
    order matters. A bag of fixations would let the organism predict a window
    from its successor, which is not prediction, it is hindsight.
    """

    readings: Tuple[Reading, ...]
    origin: str
    source: str
    domain: str
    fingerprint: str
    label: str = ""

    def __len__(self) -> int:
        return len(self.readings)

    @property
    def observations(self) -> List[Any]:
        return [reading.observation for reading in self.readings]

    @property
    def signatures(self) -> List[Signature]:
        return [reading.signature for reading in self.readings]

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "origin": self.origin,
            "domain": self.domain,
            "fixations": len(self.readings),
            "channels": int(self.readings[0].x.size) if self.readings else 0,
            "fingerprint": self.fingerprint[:16],
        }


def _autocorr(a: np.ndarray, lag: int) -> float:
    """Pearson correlation of ``a`` with itself at ``lag``; 0 if undefined."""
    if a.size <= lag + 2:
        return 0.0
    left, right = a[:-lag], a[lag:]
    ls, rs = left.std(), right.std()
    if ls <= 0.0 or rs <= 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _excess_kurtosis(d: np.ndarray) -> float:
    """Fourth standardised moment minus 3. Gaussian noise gives 0."""
    scale = d.std()
    if scale <= 0.0:
        return 0.0
    return float(np.mean(((d - d.mean()) / scale) ** 4) - 3.0)


def _spectral_slope(values: np.ndarray) -> float:
    """OLS slope of log power against log frequency; 0 white, -2 a random walk."""
    v = np.asarray(values, dtype=float)
    v = v - v.mean()
    if v.size < 16 or not np.any(v):
        return 0.0
    taper = np.hanning(v.size)
    power = np.abs(np.fft.rfft(v * taper)) ** 2
    freqs = np.fft.rfftfreq(v.size, d=1.0)
    mask = (freqs > 0.0) & (power > 0.0)
    if mask.sum() < 4:
        return 0.0
    return float(np.polyfit(np.log10(freqs[mask]), np.log10(power[mask]), 1)[0])


def shape(values: Sequence[float]) -> Signature:
    """Measure a window's four shape numbers.

    Computed on the *increments* for three of the four, because a stream's
    increments are what a stationary medium does not have in common with a
    dream: a dream reproduces a level history and loses the way it steps.

    Args:
        values: one window, at least :data:`MIN_SAMPLES` samples.

    Returns:
        :class:`Signature`; a degenerate (constant) window gives all zeros
        rather than an exception, since a stuck sensor is a reading too.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    d = np.diff(v)
    if d.size < 8 or d.std() <= 0.0:
        return Signature()
    scaled = d / d.std()
    absd = np.abs(scaled)
    volatility = float(
        np.mean([_autocorr(absd, lag) for lag in (1, 2, 3)])
    )
    tail = float(np.log1p(max(_excess_kurtosis(scaled), 0.0)))
    slope = _spectral_slope(v)
    third = float(np.mean(scaled ** 3))
    magnitude = float(np.mean(absd ** 3))
    reversibility = float(third / magnitude) if magnitude > 0.0 else 0.0
    stillness = float(np.mean(d == 0.0))
    return Signature(
        volatility=volatility,
        tail=tail,
        slope=slope,
        reversibility=max(-1.0, min(1.0, reversibility)),
        stillness=stillness,
    )


def core_shape(reading: "Reading") -> Tuple[float, float]:
    """The core's two numbers about one fixation: the chord and the GLRT.

    * ``chord`` — ``‖P_H x‖ / (σ√n)``: the norm of the core's own contrast of the
      reading, in units of one channel's measured noise. It is the only quantity
      the theorem leaves invariant, so it is the reading's honest "how split is
      this profile" number.
    * ``peak`` — ``max_k |t_k|``, the core's GLRT: has any single channel been
      pulled further than the honest spread allows?

    Both are computed *by the core*: ``theorem.contrast`` and
    ``detector.statistic``. Nothing here re-derives them, and both are
    profile-invariant, which is what makes them usable across domains whose
    organs carry different biases.
    """
    namespace = _core()
    x = np.asarray(reading.x, dtype=float)
    sigma = float(reading.sigma)
    if sigma <= 0.0:
        sigma = 1e-12
    contrast = np.asarray(namespace.theorem.contrast(x), dtype=float)
    chord = float(np.linalg.norm(contrast) / (sigma * math.sqrt(max(x.size, 1))))
    peak = float(namespace.detector.statistic(x / sigma))
    return chord, peak


def feature_vector(reading: "Reading") -> np.ndarray:
    """The six numbers a fixation is judged by, in the order of :data:`FEATURES`."""
    chord, peak = core_shape(reading)
    return np.concatenate(
        [np.array([chord, peak], dtype=float), reading.signature.as_array()]
    )


def _robust_sigma(rows: Sequence[np.ndarray]) -> float:
    """The medium's own noise: MAD x 1.4826 over every channel of every fixation.

    Robust on purpose. A stream with one big regime change would inflate a plain
    standard deviation until the change looked ordinary, and the organ would then
    report silence about the only thing that happened.
    """
    stacked = np.concatenate([np.asarray(row, dtype=float).ravel() for row in rows])
    if stacked.size == 0:
        return 1e-9
    centre = float(np.median(stacked))
    mad = float(np.median(np.abs(stacked - centre)))
    scale = 1.4826 * mad
    if scale <= 0.0:
        scale = float(stacked.std(ddof=1)) if stacked.size > 1 else 0.0
    return scale if scale > 0.0 else 1e-9


@dataclass
class Cochlea:
    """Eight log-spaced bands; each band reports where its energy actually sits.

    The reading of band ``j`` is

    .. math:: x_j = (f^{centroid}_j - f^{centre}_j) / (f_{max} - f_{min}),

    the displacement of the band's spectral centroid from the band's own centre,
    in units of the organ's whole band width. A flat medium scatters around zero
    in every band; a voice that falls *between* two bands pulls one of them and
    leaves its neighbours low. The core's GLRT then asks the only question that
    matters — is one channel pulled more than the honest spread allows — which
    is why the organ needs no detector of its own.

    The systematic bias of the organ (low bands hold few bins, so their centroid
    is quantised) is *not* corrected: it is common to every window and the core
    profiles out a common offset by construction.
    """

    bands: int = BANDS
    window: int = WINDOW
    stride: int = STRIDE
    f_min: float = F_MIN
    f_max: float = F_MAX
    pad: int = 4
    name: str = "cochlea"

    def edges(self) -> np.ndarray:
        return np.geomspace(self.f_min, self.f_max, self.bands + 1)

    def read(self, values: Sequence[float]) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size < 16:
            raise SensorError(f"cochlea needs 16+ samples, got {v.size}")
        v = v - v.mean()
        size = int(v.size * max(1, self.pad))
        power = np.abs(np.fft.rfft(v, n=size)) ** 2
        freqs = np.fft.rfftfreq(size, d=1.0)
        edges = self.edges()
        span = self.f_max - self.f_min
        out = np.zeros(self.bands, dtype=float)
        for j in range(self.bands):
            mask = (freqs >= edges[j]) & (freqs < edges[j + 1])
            if not mask.any():
                continue
            band_power = power[mask]
            total = float(band_power.sum())
            if total <= 0.0:
                continue
            centroid = float((freqs[mask] * band_power).sum() / total)
            centre = math.sqrt(edges[j] * edges[j + 1])
            out[j] = (centroid - centre) / span
        return out

    def sigma(self, rows: Sequence[np.ndarray]) -> float:
        return _robust_sigma(rows)


@dataclass
class Segments:
    """The window cut into eight parts; each part reports its level.

    The reading of part ``j`` is ``x_j = (mean_j - mean_all) / std_all``: how far
    that slice of the window sits from the window's own level, in the window's
    own units. A stationary medium scatters the eight parts by about
    ``1/sqrt(window/bands)`` and no more; a step, a spike or a change of regime
    moves one part and leaves the rest where they were.

    This is the organ that sees what a spectrum cannot: a level offset is
    invisible to the periodogram and obvious to a slice. It is also the organ
    whose bias is easiest to read — the core's ``contrast`` removes the common
    mean, and that common mean is exactly the part of a slice reading that says
    nothing about a liar.
    """

    bands: int = BANDS
    name: str = "segments"

    def read(self, values: Sequence[float]) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size < self.bands * 2:
            raise SensorError(f"segments needs {self.bands * 2}+ samples, got {v.size}")
        parts = np.array_split(v, self.bands)
        means = np.array([float(p.mean()) for p in parts])
        scale = float(v.std(ddof=1))
        if scale <= 0.0:
            return np.zeros(self.bands, dtype=float)
        return (means - float(v.mean())) / scale

    def sigma(self, rows: Sequence[np.ndarray]) -> float:
        return _robust_sigma(rows)


#: The organs a :class:`Sensorium` can be born with.
ORGANS = ("cochlea", "segments")


def build_organ(name: str = "cochlea", bands: int = BANDS):
    """Return the organ requested by name, or raise for an unknown one."""
    if name == "cochlea":
        return Cochlea(bands=bands)
    if name == "segments":
        return Segments(bands=bands)
    raise SensorError(f"unknown organ {name!r}; known: {ORGANS}")


def _sigma_of(organ, rows: Sequence[np.ndarray]) -> float:
    method = getattr(organ, "sigma", None)
    if method is not None:
        return float(method(rows))
    return _robust_sigma(rows)


# --------------------------------------------------------------------------- #
# Cache and network. A feed that is unreachable is a warning, never an invention.
# --------------------------------------------------------------------------- #

#: Where fetched payloads live between runs. Overridable for tests and for an
#: operator who wants a read-only sensorium on a machine with no home directory.
CACHE_ENV = "ENRG_SENSORIUM_CACHE"
_USER_AGENT = "enrg-ai/0.1 (autopoietic organism; +https://github.com/enrg-ai)"


def default_cache_dir() -> Path:
    import os

    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "enrg-ai" / "sensorium"


def _cache_path(key: str, cache_dir: Optional[Path]) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:120]
    return Path(cache_dir or default_cache_dir()) / f"{safe}.json"


def _cached(key: str, producer, *, force: bool = False, cache_dir: Optional[Path] = None):
    """Return ``(payload, from_cache)``; store what was fetched for next time.

    A warm cache is not a degraded reading — it is the same bytes, fetched
    earlier. It *is* flagged in ``Perception.degraded`` when the producer failed
    and only the cache answered, because then the number may be old.
    """
    path = _cache_path(key, cache_dir)
    if not force and path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8")), True
        except (OSError, ValueError):
            pass
    payload = producer()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass
    return payload, False


def _fetch_json(
    url: str, timeout: float = 12.0, headers: Optional[Dict[str, str]] = None
):
    """GET ``url`` and parse JSON. Every failure becomes a :class:`SensorError`."""
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, **(headers or {})}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SensorError(f"{url} answered HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SensorError(f"{url} unreachable: {exc.reason}") from exc
    except (TimeoutError, ValueError) as exc:
        raise SensorError(f"{url} gave no usable JSON: {exc}") from exc


def _perception(
    values: Sequence[float],
    *,
    origin: str,
    source: str,
    domain: str,
    label: str = "",
    meta: Optional[Dict[str, Any]] = None,
    warnings: Sequence[str] = (),
    degraded: bool = False,
) -> Perception:
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    return Perception(
        values=array,
        origin=origin,
        source=source,
        domain=domain,
        label=label or source,
        meta=dict(meta or {}),
        warnings=tuple(warnings),
        degraded=degraded,
    )


# --------------------------------------------------------------------------- #
# Parsers: pure functions of a payload. Every feed is tested through these, so
# no test needs the network and no operator has to trust an untested fetch.
# --------------------------------------------------------------------------- #


def parse_finance(payload: Mapping[str, Any]) -> np.ndarray:
    """Closes out of ``{"close": [...]}``; raises if too little is usable."""
    closes = np.asarray(payload.get("close", []), dtype=np.float64).ravel()
    closes = closes[np.isfinite(closes)]
    if closes.size < MIN_SAMPLES:
        raise SensorError(f"finance payload held {closes.size} usable closes")
    return closes


def parse_space_weather(
    payload: Sequence[Mapping[str, Any]], limit: int = 1440
) -> np.ndarray:
    """Planetary K index, in thirds of a unit, oldest first.

    Three field names are tried because NOAA has renamed the column before; a
    payload that matches none of them raises instead of yielding an empty array.
    """
    out: List[float] = []
    for row in payload:
        for key in ("kp_index", "estimated_kp", "kp"):
            if key in row:
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    break
                if math.isfinite(value):
                    out.append(value)
                break
    array = np.asarray(out, dtype=np.float64)
    if array.size < MIN_SAMPLES:
        raise SensorError(f"space weather payload held {array.size} usable entries")
    return array[-limit:]


def parse_solar(payload: Sequence[Mapping[str, Any]], limit: int = 500) -> np.ndarray:
    """Monthly sunspot number; ``-1`` marks a missing month and is dropped."""
    out: List[float] = []
    for row in payload:
        try:
            value = float(row.get("ssn"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0.0:
            out.append(value)
    array = np.asarray(out, dtype=np.float64)
    if array.size < MIN_SAMPLES:
        raise SensorError(f"solar payload held {array.size} usable months")
    return array[-limit:]


def parse_attention(payload: Mapping[str, Any], limit: int = 512) -> np.ndarray:
    """Daily views of one article: collective attention, not a market."""
    out: List[float] = []
    for item in payload.get("items", []):
        try:
            out.append(float(item["views"]))
        except (KeyError, TypeError, ValueError):
            continue
    array = np.asarray(out, dtype=np.float64)
    if array.size < MIN_SAMPLES:
        raise SensorError(f"pageview payload held {array.size} usable days")
    return array[-limit:]


def parse_commits(
    payload: Sequence[Mapping[str, Any]], mode: str = "gaps", limit: int = 512
) -> np.ndarray:
    """Commit cadence as a stream.

    ``mode="gaps"`` is hours between consecutive commits: a waiting-time stream,
    heavy tailed and asymmetric, which is what human work looks like.
    ``mode="counts"`` is commits per day: shorter and blunter.
    """
    from datetime import datetime

    stamps: List[float] = []
    for row in payload:
        raw = (row.get("commit") or {}).get("author", {}).get("date")
        if not raw:
            continue
        try:
            moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        stamps.append(moment.timestamp())
    if len(stamps) < 8:
        raise SensorError(f"commit payload held {len(stamps)} usable timestamps")
    ordered = np.sort(np.asarray(stamps, dtype=np.float64))
    if mode == "counts":
        days = np.floor(ordered / 86400.0)
        array = np.unique(days, return_counts=True)[1].astype(np.float64)
    else:
        gaps = np.diff(ordered) / 3600.0
        array = gaps[gaps > 0.0]
    if array.size < MIN_SAMPLES:
        raise SensorError(f"commit stream held {array.size} usable points")
    return array[-limit:]


def parse_dataset(name: str = "digits", limit: int = 4096, offset: int = 0) -> np.ndarray:
    """A recorded sequence out of sklearn's bundled measurements.

    This is a *recording*, not a time series: one instrument's samples flattened
    in order. ``meta["kind"]`` says so, and it is what keeps the sensorium alive
    with no network at all — three domains are reachable offline (handwriting,
    physiology, wine chemistry).
    """
    try:
        from sklearn import datasets
    except ImportError as exc:  # pragma: no cover - sklearn is a declared dep
        raise SensorError(f"sklearn is not installed: {exc}") from exc
    loaders = {
        "digits": datasets.load_digits,
        "diabetes": datasets.load_diabetes,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "iris": datasets.load_iris,
        "linnerud": datasets.load_linnerud,
    }
    if name not in loaders:
        raise SensorError(f"unknown dataset {name!r}; known: {sorted(loaders)}")
    flat = np.asarray(loaders[name]().data, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size < MIN_SAMPLES:
        raise SensorError(f"dataset {name} held {flat.size} usable values")
    flat = flat[offset:] if offset > 0 else flat
    return flat[:limit] if limit else flat


def parse_file(path: "str | Path", column: int = -1, limit: int = 8192) -> np.ndarray:
    """Any numeric file on disk: ``.npy``, or text with one number per row."""
    target = Path(path).expanduser()
    if not target.is_file():
        raise SensorError(f"no such file: {target}")
    if target.suffix == ".npy":
        flat = np.asarray(np.load(target, allow_pickle=False), dtype=np.float64).ravel()
    else:
        try:
            data = np.genfromtxt(target, delimiter=",", usecols=column, dtype=np.float64)
        except (ValueError, IndexError):
            data = np.genfromtxt(target, dtype=np.float64)
        flat = np.asarray(data, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size < MIN_SAMPLES:
        raise SensorError(f"{target} held {flat.size} usable numbers")
    return flat[:limit] if limit else flat


# --------------------------------------------------------------------------- #
# Listeners: the world. Each one either returns a Perception or raises
# SensorError — none of them fills a gap with a simulation.
# --------------------------------------------------------------------------- #

NOAA_KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
NOAA_SOLAR_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
)
WIKIMEDIA_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/daily/{start}/{end}"
)
GITHUB_COMMITS_URL = "https://api.github.com/repos/{repo}/commits?per_page=100&page={page}"


def _yfinance_payload(
    symbol: str, period: str, interval: str, timeout: float
) -> Dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise SensorError(f"yfinance is not installed: {exc}") from exc
    try:
        frame = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001 - yfinance raises anything at all
        raise SensorError(f"yfinance refused {symbol}: {exc}") from exc
    if frame is None or len(frame) == 0:
        raise SensorError(f"yfinance returned no rows for {symbol}")
    column = frame["Close"] if "Close" in frame.columns else frame.iloc[:, -1]
    closes = np.asarray(column, dtype=np.float64).ravel()
    stamps = [str(index) for index in frame.index]
    return {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "timestamps": stamps,
        "close": [float(c) for c in closes],
    }


def listen_finance(
    symbol: str = "BTC-USD",
    period: str = "30d",
    interval: str = "1h",
    *,
    force: bool = False,
    cache_dir: Optional[Path] = None,
    timeout: float = 20.0,
) -> Perception:
    """A market: hourly closes of ``symbol``. The one feed that is only money."""
    payload, cached = _cached(
        f"finance_{symbol}_{period}_{interval}",
        lambda: _yfinance_payload(symbol, period, interval, timeout),
        force=force,
        cache_dir=cache_dir,
    )
    values = parse_finance(payload)
    return _perception(
        values,
        origin=WORLD,
        source=f"finance:{symbol}",
        domain=DOMAIN_FINANCE,
        label=f"{symbol} {interval} closes",
        meta={
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "from_cache": cached,
            "kind": "time series",
        },
        degraded=cached,
    )


def listen_space_weather(
    *,
    limit: int = 1440,
    force: bool = False,
    cache_dir: Optional[Path] = None,
    timeout: float = 12.0,
    payload: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Perception:
    """The Earth's own field: NOAA planetary K index, one value per minute.

    Not a human artefact and not a market: geomagnetic storms are driven by the
    sun, and their cadence has nothing in common with either of the other two
    domains. That is the point of listening to them together.
    """
    if payload is None:
        payload, cached = _cached(
            "space_weather_kp",
            lambda: _fetch_json(NOAA_KP_URL, timeout=timeout),
            force=force,
            cache_dir=cache_dir,
        )
    else:
        cached = False
    values = parse_space_weather(payload, limit=limit)
    return _perception(
        values,
        origin=WORLD,
        source="physics:noaa-kp",
        domain=DOMAIN_PHYSICS,
        label="planetary K index (1 min)",
        meta={
            "url": NOAA_KP_URL,
            "from_cache": cached,
            "kind": "time series",
            "units": "thirds of a K unit",
        },
        degraded=cached,
    )


def listen_solar(
    *,
    limit: int = 500,
    force: bool = False,
    cache_dir: Optional[Path] = None,
    timeout: float = 12.0,
    payload: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Perception:
    """Astronomy: the monthly sunspot number, a record running back to 1749."""
    if payload is None:
        payload, cached = _cached(
            "solar_cycle_ssn",
            lambda: _fetch_json(NOAA_SOLAR_URL, timeout=timeout),
            force=force,
            cache_dir=cache_dir,
        )
    else:
        cached = False
    values = parse_solar(payload, limit=limit)
    return _perception(
        values,
        origin=WORLD,
        source="physics:sunspots",
        domain=DOMAIN_PHYSICS,
        label="monthly sunspot number",
        meta={
            "url": NOAA_SOLAR_URL,
            "from_cache": cached,
            "kind": "time series",
            "units": "sunspot number",
        },
        degraded=cached,
    )


def listen_attention(
    article: str = "Bitcoin",
    days: int = 180,
    *,
    force: bool = False,
    cache_dir: Optional[Path] = None,
    timeout: float = 12.0,
    payload: Optional[Mapping[str, Any]] = None,
) -> Perception:
    """Behaviour: how many humans opened one page, per day.

    Collective attention is behaviour in the strict sense — no price, no physics,
    just people. It has a weekly rhythm a dream has no reason to invent.
    """
    from datetime import date, timedelta

    if payload is None:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=int(days))
        url = WIKIMEDIA_URL.format(
            article=article, start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d")
        )
        payload, cached = _cached(
            f"attention_{article}_{start}_{end}",
            lambda: _fetch_json(url, timeout=timeout),
            force=force,
            cache_dir=cache_dir,
        )
    else:
        cached = False
    values = parse_attention(payload)
    return _perception(
        values,
        origin=WORLD,
        source=f"behavior:pageviews:{article}",
        domain=DOMAIN_BEHAVIOR,
        label=f"daily views of {article}",
        meta={"from_cache": cached, "kind": "time series", "units": "views/day"},
        degraded=cached,
    )


def listen_commits(
    repo: str = "bitcoin/bitcoin",
    pages: int = 2,
    mode: str = "gaps",
    *,
    force: bool = False,
    cache_dir: Optional[Path] = None,
    timeout: float = 15.0,
    payload: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Perception:
    """Behaviour: the cadence of human work, from a repository's commit log."""
    cached = False
    if payload is None:
        rows: List[Mapping[str, Any]] = []
        for page in range(1, max(1, pages) + 1):
            url = GITHUB_COMMITS_URL.format(repo=repo, page=page)
            batch, cached = _cached(
                f"commits_{repo.replace('/', '_')}_{page}",
                lambda url=url: _fetch_json(
                    url,
                    timeout=timeout,
                    headers={"Accept": "application/vnd.github+json"},
                ),
                force=force,
                cache_dir=cache_dir,
            )
            rows.extend(batch)
        payload = rows
    values = parse_commits(payload, mode=mode)
    return _perception(
        values,
        origin=WORLD,
        source=f"behavior:commits:{repo}",
        domain=DOMAIN_BEHAVIOR,
        label=f"{repo} commit cadence",
        meta={
            "repo": repo,
            "mode": mode,
            "kind": "point process",
            "from_cache": cached,
            "units": "hours between commits",
        },
        degraded=cached,
    )


def listen_dataset(name: str = "digits", *, limit: int = 4096, offset: int = 0) -> Perception:
    """A recording from disk: the sensorium's floor when the network is absent."""
    values = parse_dataset(name, limit=limit, offset=offset)
    domain = DOMAIN_BEHAVIOR if name == "digits" else DOMAIN_RECORDED
    return _perception(
        values,
        origin=WORLD,
        source=f"recorded:{name}",
        domain=domain,
        label=f"sklearn {name} recording",
        meta={"dataset": name, "kind": "recording, not a time series"},
    )


def listen_file(path: "str | Path", *, column: int = -1, limit: int = 8192) -> Perception:
    """Whatever the operator points at: the sensorium accepts any numbers."""
    values = parse_file(path, column=column, limit=limit)
    return _perception(
        values,
        origin=WORLD,
        source=f"file:{Path(path).name}",
        domain=DOMAIN_RECORDED,
        label=str(path),
        meta={"path": str(path), "kind": "file"},
    )


#: Spec strings ``kind:argument`` accepted by :func:`hear`.
FEEDS = {
    "finance": listen_finance,
    "space-weather": listen_space_weather,
    "solar": listen_solar,
    "attention": listen_attention,
    "commits": listen_commits,
    "dataset": listen_dataset,
    "file": listen_file,
}

#: Kinds whose first positional argument is the spec's tail.
_ARGUMENT_FEEDS = ("finance", "attention", "commits", "dataset", "file")

#: Kinds that reach the network and therefore accept a cache directory.
NETWORK_FEEDS = ("finance", "space-weather", "solar", "attention", "commits")


def hear(spec: str, **kwargs: Any) -> Perception:
    """Hear one feed by spec, e.g. ``"finance:ETH-USD"`` or ``"dataset:digits"``.

    Args:
        spec: ``kind`` or ``kind:argument``; kinds are :data:`FEEDS`.
        **kwargs: forwarded to the feed (``force``, ``limit``, ``days``, ...).
            ``cache_dir`` is dropped for feeds that read nothing over the network,
            so a caller may pass one uniform set of options to every kind.

    Raises:
        SensorError: unknown kind, or the feed itself failed.
    """
    kind, _, argument = spec.partition(":")
    if kind not in FEEDS:
        raise SensorError(f"unknown feed {kind!r}; known: {sorted(FEEDS)}")
    options = dict(kwargs)
    if kind not in NETWORK_FEEDS:
        options.pop("cache_dir", None)
        options.pop("timeout", None)
    feed = FEEDS[kind]
    if argument and kind in _ARGUMENT_FEEDS:
        return feed(argument, **options)
    return feed(**options)


# --------------------------------------------------------------------------- #
# Dreams: the self. Four kinds of endogenous noise, none of them the world.
# --------------------------------------------------------------------------- #

#: AR(1) coefficient and the smoothing of a replay when a caller asks for a
#: dream without saying which. Declared, not fitted: a dream is the organism's
#: own voice, and its voice is a parameter of its body, not of the world.
DREAM_PHI = 0.9
DREAM_JITTER = 0.02


def dream_red(length: int = 512, phi: float = DREAM_PHI, seed: int = 0) -> np.ndarray:
    """AR(1) noise: one spectral slope, no regime, no memory of its own size.

    The organism's cheapest voice. It is what a homeostasis loop converges to
    when it has nothing to hold on to, and it is the dream a single-slope
    detector can never catch, because it *is* a single slope.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=int(length)) * math.sqrt(max(1.0 - phi * phi, 1e-12))
    out = np.empty(int(length), dtype=np.float64)
    state = 0.0
    for i in range(int(length)):
        state = phi * state + shocks[i]
        out[i] = state
    return out


def dream_rhythm(
    length: int = 512, seed: int = 0, voices: int = 3, jitter: float = DREAM_JITTER
) -> np.ndarray:
    """The organism's own clock: a few sinusoids at frequencies it chose itself.

    Quantised to three decimals and jittered by ``jitter``, because a biological
    clock is a digital counter with a bad crystal. Its periodogram is a handful
    of lines — trivially separable from the world, and therefore the control
    case in the demonstration.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(length), dtype=np.float64)
    freqs = rng.uniform(0.02, 0.25, size=int(voices))
    phases = rng.uniform(0.0, 2.0 * math.pi, size=int(voices))
    amps = rng.uniform(0.4, 1.0, size=int(voices))
    signal = np.zeros(int(length), dtype=np.float64)
    for f, p, a in zip(freqs, phases, amps):
        signal += a * np.sin(2.0 * math.pi * f * t + p)
    signal += jitter * rng.normal(size=int(length))
    return np.round(signal, 3)


def dream_replay(
    anchor: Sequence[float],
    lam: float = 0.35,
    epsilon: float = 0.05,
    seed: int = 0,
    length: Optional[int] = None,
) -> np.ndarray:
    """Memory of the world, smoothed at rate ``λ`` and buried in its own noise.

    This is the dream that matters for the calibration. ``λ`` is the *same* λ the
    posture uses to write the world into itself, so the dream is anchored to
    memory by exactly the parameter homeostasis is searching over:

    .. math:: s_t = s_{t-1} + \\lambda (a_t - s_{t-1}), \\quad
              \\text{dream}_t = s_t + (1-\\lambda)\\,\\sigma_a\\,\\xi_t

    ``λ = 1`` replays the memory exactly and is caught instantly (its fingerprint
    is the anchor's). Small ``λ`` keeps the rhythm of the world and loses its
    detail; ``ε`` additionally lets the memory decay while the dream runs, which
    is the phantom the self-model hunts for.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(anchor, dtype=np.float64).ravel()
    if values.size < 8:
        raise SensorError("a replay needs 8+ samples of memory to replay")
    size = int(length) if length else int(values.size)
    if size != values.size:
        grid = np.linspace(0.0, values.size - 1.0, size)
        memory = np.interp(grid, np.arange(values.size, dtype=np.float64), values)
    else:
        memory = values.copy()
    lam = float(min(max(lam, 1e-3), 1.0))
    scale = float(memory.std(ddof=1)) or 1.0
    out = np.empty(size, dtype=np.float64)
    state = float(memory[0])
    decay = math.exp(-max(float(epsilon), 0.0))
    for i in range(size):
        state = state + lam * (memory[i] - state)
        state *= decay
        out[i] = state
    out = out + (1.0 - lam) * scale * rng.normal(size=size)
    return out


def phase_randomise(values: Sequence[float], seed: int = 0) -> np.ndarray:
    """The perfect counterfeit: same spectrum, same histogram, foreign phases.

    Magnitudes of the DFT are kept to the last bit, phases are drawn uniform, and
    the result is rank-mapped back onto the original sorted values — so the
    surrogate has *identical* marginal distribution and, to within the rank map,
    an identical periodogram. Any linear statistic (spectrum, autocorrelation,
    variance, kurtosis) reads it as the same stream.

    Only the *order* is destroyed, and order is exactly what a phase-blind
    detector never looked at. This is the honest limit of the whole exercise and
    the reason the ledger exists: no statistic can decide "I made this" — only
    remembering that one did.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 16:
        raise SensorError("phase randomisation needs 16+ samples")
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(v)
    magnitude = np.abs(spectrum)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=magnitude.size)
    phases[0] = 0.0
    if v.size % 2 == 0:
        phases[-1] = 0.0
    surrogate = np.fft.irfft(magnitude * np.exp(1j * phases), n=v.size)
    order = np.argsort(surrogate, kind="stable")
    out = np.empty(v.size, dtype=np.float64)
    out[order] = np.sort(v)
    return out


def dream(
    kind: str = DREAM_RED,
    *,
    length: Optional[int] = None,
    anchor: Optional[Perception] = None,
    lam: float = 0.35,
    epsilon: float = 0.05,
    seed: int = 0,
) -> Perception:
    """Produce one dream, with its provenance set to :data:`SELF`.

    Args:
        kind: one of :data:`DREAMS`.
        length: samples; defaults to the anchor's length, else 512.
        anchor: the world stream being dreamed about. Required for
            :data:`DREAM_REPLAY` and :data:`DREAM_SURROGATE`, because both are
            made *of* something the organism heard.
        lam: memory anchor of a replay, in ``(0, 1]`` — the same λ homeostasis
            calibrates, which is why a badly tuned λ produces recognisable
            dreams.
        epsilon: decay of the replayed memory while the dream runs.
        seed: reproducibility; the seed travels in ``meta``.

    Raises:
        SensorError: an anchored dream without an anchor, or too few samples.
    """
    if kind not in DREAMS:
        raise SensorError(f"unknown dream {kind!r}; known: {DREAMS}")
    size = int(length) if length else (len(anchor) if anchor is not None else 512)
    if kind == DREAM_RED:
        values = dream_red(size, seed=seed)
        label = "AR(1) voice"
    elif kind == DREAM_RHYTHM:
        values = dream_rhythm(size, seed=seed)
        label = "own clock"
    elif kind == DREAM_REPLAY:
        if anchor is None:
            raise SensorError("a replay needs an anchor to replay")
        values = dream_replay(
            anchor.values, lam=lam, epsilon=epsilon, seed=seed, length=size
        )
        label = f"replay of {anchor.source}"
    else:  # DREAM_SURROGATE
        if anchor is None:
            raise SensorError("a surrogate needs an anchor to counterfeit")
        values = phase_randomise(anchor.values, seed=seed)
        if size != values.size:
            grid = np.linspace(0.0, values.size - 1.0, size)
            values = np.interp(grid, np.arange(values.size, dtype=float), values)
        label = f"surrogate of {anchor.source}"
    return _perception(
        values,
        origin=SELF,
        source=f"self:{kind}",
        domain=DOMAIN_SELF,
        label=label,
        meta={
            "kind": kind,
            "seed": int(seed),
            "lam": float(lam),
            "epsilon": float(epsilon),
            "anchor": anchor.fingerprint[:16] if anchor is not None else "",
            "anchor_source": anchor.source if anchor is not None else "",
        },
    )


def counterfeit(
    perception: Perception,
    *,
    block: int = 256,
    source: str = "",
) -> Perception:
    """A stream that looks like the world and is not: a block of it, repeated.

    This function lies on purpose. The values are real — a genuine stretch of a
    genuine feed, repeated — so every provenance-free statistic sees a world
    stream: the shape is a world shape, the reality score is a world score, and
    the ledger does not fire, because the organism has never emitted these bytes.
    The stream is a *forgery*, not a dream, and the difference matters: a dream
    can be caught by the self model, a forgery can only be caught by noticing
    that the feed says the same thing again.

    It exists for one reason: to make the organism's suspicion measurable. A
    detector of forgeries that is only ever shown honest feeds is not a detector,
    it is a decoration. Nothing in :meth:`Sensorium.hear` ever produces one, and
    :attr:`Perception.label` says ``"counterfeit"`` so the report cannot confuse
    the two.

    Args:
        perception: the stream to copy from. Its provenance, source, domain and
            warnings are inherited, and its values are not modified.
        block: samples per copy. Defaults to ``4 * MIN_SAMPLES``, which is a
            whole number of the survey's 64-sample windows — a forger who caches
            a payload of 256 samples is both realistic and catchable, and the
            alignment is what makes the repeat exact rather than approximate.
        source: name for the forged stream; defaults to the parent's with
            ``"#counterfeit"`` appended.

    Raises:
        SensorError: a block shorter than an organ can read.
    """
    if int(block) < MIN_SAMPLES:
        raise SensorError(f"a counterfeit block needs {MIN_SAMPLES}+ samples")
    values = np.asarray(perception.values, dtype=np.float64)
    span = min(int(block), values.size)
    copies = int(math.ceil(values.size / span))
    forged = np.tile(values[:span], copies)[: values.size]
    return _perception(
        forged,
        origin=perception.origin,
        source=source or f"{perception.source}#counterfeit",
        domain=perception.domain,
        label=f"counterfeit of {perception.source}",
        meta={
            "counterfeit": True,
            "block": span,
            "copies": copies,
            "parent": perception.source,
            "parent_fingerprint": perception.fingerprint[:16],
        },
        warnings=perception.warnings,
        degraded=perception.degraded,
    )


# --------------------------------------------------------------------------- #
# The sensorium: feeds plus organs, and the one thing it must never do —
# present a dream as a perception of the world.
# --------------------------------------------------------------------------- #

@dataclass
class Sensorium:
    """A body's sensory surface: what it can hear, and how it reads what it hears.

    One organ at a time (``organ`` names it), one window geometry, and a log of
    every stream refused and why. :meth:`coverage` is the honest answer to "is
    this organism blind on one side": with fewer than three world domains it says
    so, because a system calibrated on two domains learns two noises and calls
    the pair of them the world.

    Args:
        organ: ``"cochlea"`` (spectral) or ``"segments"`` (level slices).
        window: samples per fixation; clamped down for short streams.
        stride: hop between fixations; defaults to half a window.
        bands: channels per organ, fixed at :data:`BANDS` unless a caller knows
            that the core's null table is built per channel count.
        cache_dir: where fetched payloads are stored; ``None`` uses
            :func:`default_cache_dir`.
    """

    organ: str = "cochlea"
    window: int = WINDOW
    stride: int = STRIDE
    bands: int = BANDS
    cache_dir: Optional[Path] = None
    timeout: float = 12.0
    warnings: List[str] = field(default_factory=list, init=False)
    heard: List[Dict[str, Any]] = field(default_factory=list, init=False)
    seen: Dict[str, Perception] = field(default_factory=dict, init=False)
    probes: List[Dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        build_organ(self.organ, self.bands)  # fail early on an unknown organ

    # -- the world ---------------------------------------------------------- #

    def remember(self, perception: Perception) -> Perception:
        """Keep a stream by name, so it can be looked at again.

        A body that cannot re-address what it has already taken in cannot be
        curious: curiosity without memory is a twitch. The sensorium holds the
        payload of everything it heard (a few thousand floats per feed — cheaper
        than the alternative, which is hearing it twice) and this is the seam
        through which a caller that built a perception by hand — a
        :func:`counterfeit`, a test fixture — registers it.
        """
        self.seen[perception.source] = perception
        return perception

    def recall(self, stream: str) -> Perception:
        """The stream of that name, or a :class:`SensorError` saying what is known."""
        found = self.seen.get(stream)
        if found is None:
            known = ", ".join(sorted(self.seen)) or "nothing"
            raise SensorError(f"no stream {stream!r} to look at again; heard: {known}")
        return found

    def hear(self, spec: str, **kwargs: Any) -> Perception:
        """Hear one feed; record it in the manifest and keep its warnings."""
        perception = hear(spec, cache_dir=self.cache_dir, **kwargs)
        self.heard.append(perception.summary())
        self.remember(perception)
        for warning in perception.warnings:
            self.warnings.append(f"{perception.source}: {warning}")
        return perception

    def hear_all(self, specs: Sequence[str], **kwargs: Any) -> List[Perception]:
        """Hear every spec that answers; failures become warnings, not silence.

        A missing feed is never replaced by a simulation. The pool is then
        smaller than was asked for, and :meth:`coverage` reports exactly how.
        """
        out: List[Perception] = []
        for spec in specs:
            try:
                out.append(self.hear(spec, **kwargs))
            except SensorError as exc:
                self.warnings.append(f"{spec}: {exc}")
        return out

    def coverage(self, perceptions: Sequence[Perception]) -> Dict[str, Any]:
        """Which world domains this pool actually covers."""
        present = sorted({p.domain for p in perceptions if p.is_world})
        missing = [d for d in WORLD_DOMAINS if d not in present]
        return {
            "domains": present,
            "missing": missing,
            "ok": len([d for d in present if d in WORLD_DOMAINS]) >= 3,
            "sources": [p.source for p in perceptions],
        }

    # -- the self ----------------------------------------------------------- #

    def dream(self, kind: str = DREAM_RED, **kwargs: Any) -> Perception:
        """Dream one stream; a dream's provenance is always :data:`SELF`."""
        perception = dream(kind, **kwargs)
        self.heard.append(perception.summary())
        self.remember(perception)
        return perception

    # -- reading ------------------------------------------------------------ #

    def geometry(self, samples: int) -> Tuple[int, int]:
        """Window and stride that give at least two fixations for ``samples``."""
        window = int(self.window)
        if samples < window:
            window = max(MIN_SAMPLES, samples // 2)
        window = min(window, max(MIN_SAMPLES, samples))
        stride = max(1, min(int(self.stride), window // 2))
        if samples - window < stride:  # guarantee a second fixation
            stride = max(1, samples - window)
        return window, stride

    def fixate(
        self,
        perception: Perception,
        *,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Episode:
        """Read one perception through the organ, fixation by fixation.

        The noise ``sigma`` is measured on *this* stream over *all* its
        fixations and handed to every fixation unchanged: the core requires a
        background to standardise against, and the medium itself supplies it.

        ``limit`` caps the number of fixations, which is how a *survey* is taken:
        ``fixate(p, window=64, stride=64, limit=24)`` reads the first 1536 samples
        in consecutive, non-overlapping windows, and consecutive is the point —
        see :data:`~agent.physics.curiosity.SURVEY_WINDOW`.
        """
        organ = build_organ(self.organ, self.bands)
        values = perception.values
        default_window, default_stride = self.geometry(values.size)
        win = int(window or default_window)
        hop = int(stride or default_stride)
        if values.size < win:
            raise SensorError(
                f"{perception.source}: {values.size} samples cannot hold a "
                f"window of {win}"
            )
        starts = list(range(0, values.size - win + 1, hop)) or [0]
        if limit is not None and int(limit) > 0:
            starts = starts[: int(limit)]
        rows = [organ.read(values[start : start + win]) for start in starts]
        sigma = _sigma_of(organ, rows)
        readings = tuple(
            Reading(
                observation=_core_observation(
                    row, sigma, hours=float(win), source=perception.source
                ),
                signature=shape(values[start : start + win]),
                index=index,
                start=int(start),
            )
            for index, (start, row) in enumerate(zip(starts, rows))
        )
        return Episode(
            readings=readings,
            origin=perception.origin,
            source=perception.source,
            domain=perception.domain,
            fingerprint=perception.fingerprint,
            label=perception.label,
        )

    def episodes(self, perceptions: Iterable[Perception]) -> List[Episode]:
        return [self.fixate(perception) for perception in perceptions]

    # -- looking again ------------------------------------------------------ #

    def request_probe(
        self,
        stream: str,
        *,
        context_window: int = PROBE_CONTEXT,
        anchor: Optional[int] = None,
        kind: str = PROBE_AROUND,
    ) -> Perception:
        """Go and look again, where the organism decided to look.

        This is the *answer* half of a probe: the organism has said "there is
        something here", and this cuts the slice of samples that question is
        about. Three cuts, because a body that can only look has three moves:

        ``context``  ``±context_window`` around ``anchor`` — what happened *there*.
        ``next``     the stretch immediately after the anchor — what happened
                     *next*, which is the only cut that can tell a real cycle from
                     a copied block: a copy repeats inside the recording, a cycle
                     continues outside it.
        ``compare``  the tail of the stream, the part a survey of the first
                     ``~1500`` samples never covered — the stream compared with
                     what was already said about it.

        The slice is at least :data:`MIN_SAMPLES` long, is clipped into the
        stream rather than padded with anything invented, and keeps the parent's
        source, provenance, warnings and degradation: the answer *is* the stream,
        only at another address. ``Perception.meta["probe"]`` records the address
        so a journal can be read back.

        Args:
            stream: the name the sensorium knows the stream by.
            context_window: samples either side of the anchor for ``context``.
            anchor: address in samples; defaults to the middle of the stream.
            kind: one of :data:`PROBE_KINDS`.

        Raises:
            SensorError: an unknown stream, or an unknown kind of question.
        """
        if kind not in PROBE_KINDS:
            raise SensorError(f"unknown probe {kind!r}; known: {PROBE_KINDS}")
        parent = self.recall(stream)
        values = parent.values
        size = int(values.size)
        span = max(2 * int(context_window), MIN_SAMPLES)
        if size <= span:
            start, span = 0, size
            where = "whole stream"
        elif kind == PROBE_AROUND:
            centre = size // 2 if anchor is None else int(anchor)
            start = min(max(centre - int(context_window), 0), size - span)
            where = f"{context_window} either side of {centre}"
        elif kind == PROBE_AFTER:
            start = min(max(int(anchor or 0), 0), size - MIN_SAMPLES)
            span = min(span, size - start)
            where = f"after {start}"
        else:  # PROBE_TAIL
            start = max(size - span, 0)
            where = f"the tail from {start}"
        answer = _perception(
            values[start : start + span],
            origin=parent.origin,
            source=parent.source,
            domain=parent.domain,
            label=f"{parent.label} @ probe {kind}",
            meta={
                "probe": kind,
                "of": parent.source,
                "start": int(start),
                "samples": int(span),
                "parent_fingerprint": parent.fingerprint[:16],
            },
            warnings=parent.warnings,
            degraded=parent.degraded,
        )
        self.probes.append(
            {
                "stream": parent.source,
                "kind": kind,
                "start": int(start),
                "samples": int(span),
                "where": where,
                "fingerprint": answer.fingerprint[:16],
            }
        )
        return answer

    def manifest(self, perceptions: Sequence[Perception]) -> Dict[str, Any]:
        """Everything this sensorium heard, with provenance, for the log."""
        world = [p for p in perceptions if p.is_world]
        return {
            "organ": self.organ,
            "window": self.window,
            "stride": self.stride,
            "bands": self.bands,
            "coverage": self.coverage(world),
            "streams": [p.summary() for p in perceptions],
            "probes": list(self.probes),
            "warnings": list(self.warnings),
        }


__all__ = [
    "BANDS",
    "CACHE_ENV",
    "Cochlea",
    "DREAMS",
    "DREAM_PHI",
    "DREAM_RED",
    "DREAM_REPLAY",
    "DREAM_RHYTHM",
    "DREAM_SURROGATE",
    "DOMAIN_BEHAVIOR",
    "DOMAIN_FINANCE",
    "DOMAIN_PHYSICS",
    "DOMAIN_RECORDED",
    "DOMAIN_SELF",
    "Episode",
    "FEATURES",
    "FEEDS",
    "MIN_SAMPLES",
    "ORGANS",
    "Perception",
    "PROBE_AFTER",
    "PROBE_AROUND",
    "PROBE_CONTEXT",
    "PROBE_KINDS",
    "PROBE_TAIL",
    "Reading",
    "SELF",
    "SIGNATURE_FIELDS",
    "STRIDE",
    "Segments",
    "SensorError",
    "Sensorium",
    "Signature",
    "WINDOW",
    "WORLD",
    "WORLD_DOMAINS",
    "build_organ",
    "core_shape",
    "counterfeit",
    "default_cache_dir",
    "dream",
    "dream_red",
    "dream_replay",
    "dream_rhythm",
    "feature_vector",
    "hear",
    "listen_attention",
    "listen_commits",
    "listen_dataset",
    "listen_finance",
    "listen_file",
    "listen_solar",
    "listen_space_weather",
    "parse_attention",
    "parse_commits",
    "parse_dataset",
    "parse_finance",
    "parse_file",
    "parse_solar",
    "parse_space_weather",
    "phase_randomise",
    "shape",
]
