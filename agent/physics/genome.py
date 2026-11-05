"""The frozen genome: entrain's proved core, reached through one door.

An organism may grow organs. It does not get to edit its DNA. This module is
the only place in ``agent.physics`` that imports ``entrain_infer``, and it does
two things besides the import:

* it **resolves** the core: an installed package, ``ENTRAIN_INFER_PATH``, or the
  sibling checkout ``<workspace>/entrain/entrain-infer`` — in that order, so the
  organism works both from a wheel and from the repository it was born in;
* it **fingerprints** the two files every claim rests on (``theorem.py``,
  ``detector.py``) with SHA-256, so "the core is untouched" is a check at run
  time rather than a promise in a README. :func:`verify` is the check; the
  self-model stores the fingerprint of the genome it was calibrated against.

Nothing here is imported eagerly: :func:`core` imports on first use. That keeps
``import agent.physics`` working in an environment where the core is absent —
the organs then raise :class:`GenomeError` when they are asked to *do*
something, not when they are merely imported.

Which quantity comes from the core, and why nothing is re-derived here:

=================================  ========================================
``core().detector.statistic``      ``max_k |t_k|`` — the GLRT of one liar
``core().detector.studentize``     ``t_k = (n x_k - sum x) / sqrt(n (n-1))``
``core().detector.null_threshold`` ``tau_alpha`` from the declared risk
``core().detector.detect``         the whole verdict, at the operator's ``alpha``
``core().theorem.contrast``        ``P_H v = v - mean(v)`` — what a park sees
``core().theorem.chord``           ``sqrt(k (n-k) / n)`` — the only invariant
``core().model.Observation``       the object a park reading is returned in
=================================  ========================================
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

#: The files whose bytes the claim rests on. A third file here would mean the
#: claim changed, so the list is short on purpose.
CORE_FILES = ("theorem.py", "detector.py")

#: The package name the core installs under.
CORE_PACKAGE = "entrain_infer"

#: Environment variable that pins the core wherever the operator keeps it.
ENV_PATH = "ENTRAIN_INFER_PATH"

#: Submodules reached through :func:`core`. ``bench``/``report`` are not here:
#: the organs observe, they do not measure a park's false-alarm risk.
CORE_MODULES = ("model", "estimator", "detector", "theorem")


class GenomeError(RuntimeError):
    """The core could not be reached: the organism has no mathematics to stand on."""


class GenomeFrozenError(GenomeError):
    """The core's bytes changed under a fingerprint already taken."""


def _sibling() -> Optional[Path]:
    """The repository's own checkout of the core, if this is a source tree.

    ``agent/physics/genome.py`` -> ``<workspace>/entrain/entrain-infer``. The
    walk is by named ancestors rather than by index, so moving the file one
    directory deeper does not silently point at the wrong place.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "entrain" / "entrain-infer" / CORE_PACKAGE
        if candidate.is_dir():
            return candidate
        if parent.name == "entrain-core-workspace":
            break
    return None


def _installed() -> Optional[Path]:
    """Where an *importable* ``entrain_infer`` lives, without importing it."""
    try:
        spec = importlib.util.find_spec(CORE_PACKAGE)
    except (ImportError, ValueError):  # pragma: no cover - broken meta path
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def core_root() -> Path:
    """Directory that holds the core package.

    Resolution order, declared so a reading can be traced to a checkout:

    1. ``$ENTRAIN_INFER_PATH`` (a directory, either the package or its parent);
    2. the installed ``entrain_infer``;
    3. the sibling checkout of the workspace this file lives in.

    Raises:
        GenomeError: none of the three resolved.
    """
    env = os.environ.get(ENV_PATH)
    if env:
        path = Path(env).expanduser().resolve()
        candidates = (path / CORE_PACKAGE, path) if path.name != CORE_PACKAGE else (path,)
        for candidate in candidates:
            if (candidate / "detector.py").is_file():
                return candidate
        raise GenomeError(
            f"{ENV_PATH}={env} does not contain a core (looked for "
            f"{CORE_PACKAGE}/detector.py)"
        )
    for found in (_installed(), _sibling()):
        if found is not None and (found / "detector.py").is_file():
            return found
    raise GenomeError(
        "entrain_infer not found: install it (pip install -e entrain/entrain-infer) "
        f"or set {ENV_PATH} to the directory holding it"
    )


def _importable() -> bool:
    try:
        return importlib.util.find_spec(CORE_PACKAGE) is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


@lru_cache(maxsize=1)
def core() -> SimpleNamespace:
    """The core's modules as one namespace: ``core().detector``, ``core().theorem``.

    On first use the core's parent directory is put on ``sys.path`` if the
    package is not importable yet, and the four modules are imported explicitly
    rather than through ``entrain_infer.__init__`` — so which module is loaded
    is visible in one place.

    Raises:
        GenomeError: the core is absent or unreadable; the message says how to
            point at it.
    """
    root = core_root()
    parent = str(root.parent)
    if not _importable() and parent not in sys.path:
        sys.path.insert(0, parent)
    modules: Dict[str, Any] = {}
    for name in CORE_MODULES:
        try:
            modules[name] = importlib.import_module(f"{CORE_PACKAGE}.{name}")
        except Exception as exc:  # noqa: BLE001 - any failure is a missing organ
            raise GenomeError(
                f"could not import {CORE_PACKAGE}.{name} from {root}: {exc}"
            ) from exc
    return SimpleNamespace(**modules)


@lru_cache(maxsize=1)
def fingerprint() -> Dict[str, str]:
    """SHA-256 of every file in :data:`CORE_FILES`, keyed by file name.

    Taken once and cached: the genome of one process does not change, and a
    changing genome is exactly the thing this function exists to notice.
    """
    root = core_root()
    out: Dict[str, str] = {}
    for name in CORE_FILES:
        path = root / name
        if not path.is_file():
            raise GenomeError(f"core file missing: {path}")
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def verify(expected: Mapping[str, str]) -> None:
    """Raise unless the core still has the bytes described by ``expected``.

    Args:
        expected: ``{file_name: sha256}`` as returned by :func:`fingerprint`.

    Raises:
        GenomeFrozenError: a file is missing, extra, or changed. The message
            names the file and both digests, because "the genome moved" is
            useless without the two hashes.
    """
    actual = fingerprint()
    for name, digest in expected.items():
        if name not in actual:
            raise GenomeFrozenError(
                f"{name} is not part of the genome (known: {sorted(actual)})"
            )
        if actual[name] != digest:
            raise GenomeFrozenError(
                f"{name} changed: expected {digest[:16]}..., found {actual[name][:16]}..."
            )


def describe() -> Dict[str, Any]:
    """A JSON-able identity of the mathematics this organism stands on."""
    namespace = core()
    root = core_root()
    return {
        "package": CORE_PACKAGE,
        "root": str(root),
        "source": "sibling" if root == _sibling() else "resolved",
        "version": getattr(
            importlib.import_module(CORE_PACKAGE), "__version__", "unknown"
        ),
        "fingerprint": dict(fingerprint()),
        "modules": list(CORE_MODULES),
        "statistic": (namespace.detector.statistic.__doc__ or "").splitlines()[0].strip(),
    }


def observation(x, sigma: float, hours: float = 1.0, source: str = ""):
    """Build the core's ``Observation`` from an organ's reading.

    A one-line adapter on purpose: the organs hand the core its own object, so
    the GLRT reads a market exactly the way it reads a park. ``sigma`` is the
    medium's own measured noise and is never guessed here.
    """
    return core().model.Observation(
        x=x, sigma=float(sigma), hours=float(hours), source=source
    )


__all__ = [
    "CORE_FILES",
    "CORE_MODULES",
    "CORE_PACKAGE",
    "ENV_PATH",
    "GenomeError",
    "GenomeFrozenError",
    "core",
    "core_root",
    "describe",
    "fingerprint",
    "observation",
    "verify",
]
