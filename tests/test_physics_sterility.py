"""Statement #87 (sterility index) and #88 (Banach contraction) at the port.

The fixture is the output of ``julia 1.10.12`` reading
``entrain/julia/kuramoto/consistency.jl``: one lived relief trace (seed 1,
ε = 1e-3) with the ``A²`` and ``CV`` the book's own statement computes from it.
The port must read the same two numbers off the same array — the statistics are
a port, not an interpretation, exactly like ``oscillator.py`` against
``basic.jl``.

``A²`` agrees to ~1e-06 rather than to the last bit: Julia has no ``erf`` in
``Base`` and the book uses the Abramowitz–Stegun approximation, while numpy has
``math.erf`` exactly. The tolerance is stated here so nobody reads the
difference as disagreement about the property.

The invariants below are the statement restated, so a regression names the
property it broke: a too-perfect relief is caught, a lived one is not, and the
smoothed λ* operator contracts with the constant the book proves.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from agent.physics import sterility as st

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "sterility_reference.json").read_text(
        encoding="utf-8"
    )
)


# ── the port against Julia ──────────────────────────────────────────────────


def test_anderson_darling_matches_julia():
    readings = FIXTURE["readings"]
    verdict = st.sterility_index(
        FIXTURE["times"], FIXTURE["relief"], FIXTURE["memory"]
    )
    # Not the last bit but very close: numpy sums pairwise, the book's Julia
    # loop sums in order, and the two orders differ in the last digits.
    assert verdict.a2 == pytest.approx(readings["a2"], abs=1e-9)
    assert verdict.cv == pytest.approx(readings["cv"], abs=1e-9)
    assert verdict.r0 == pytest.approx(readings["r0"], abs=1e-15)
    assert verdict.verdict == readings["verdict"] == "passed"


def test_smooth_max_matches_julia():
    readings = FIXTURE["smooth_max"]
    values = readings["values"]
    assert st.smooth_max(values, 16.0) == pytest.approx(readings["beta16"], abs=1e-12)
    assert st.smooth_max(values, 1.0) == pytest.approx(readings["beta1"], abs=1e-12)


def test_banach_limit_matches_julia():
    readings = FIXTURE["banach"]
    lift, slope = readings["lift"], readings["slope"]

    def rows_at(lam):
        return [lift[k] + slope[k] * lam for k in range(len(lift))]

    result = st.banach_iteration(lambda lam: max(rows_at(lam)) - 0.8, 1.0)
    assert result.converged is True
    assert result.lam == pytest.approx(readings["lam"], abs=1e-9)


# ── A² itself ───────────────────────────────────────────────────────────────


def test_anderson_darling_is_zero_for_constant():
    assert st.anderson_darling([3.0] * 40) == 0.0


def test_anderson_darling_is_small_for_normal_sample():
    sample = np.random.default_rng(7).normal(size=101)
    assert st.anderson_darling(sample) < 1.0


def test_anderson_darling_is_large_for_skewed_sample():
    sample = np.random.default_rng(7).exponential(size=101)
    assert st.anderson_darling(sample) > 1.0


def test_anderson_darling_needs_a_sample():
    assert math.isinf(st.anderson_darling([1.0, 2.0, 3.0]))


def test_normal_cdf_is_close_to_the_exact_erf():
    # The port keeps the book's approximation so both sides read the same
    # number; that is only acceptable because the approximation is good.
    for z in (-6.0, -2.5, -0.3, 0.0, 0.7, 1.9, 4.4):
        exact = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        assert abs(st.normal_cdf(z) - exact) <= 2e-7



# ── the statement itself ────────────────────────────────────────────────────


def _ideal_trace(memory=1e-3, level=0.01, seconds=600.0, step=6.0):
    times = np.arange(0.0, seconds + step, step)
    return times, level * np.exp(-memory * times)


def test_ideal_decay_is_caught():
    times, relief = _ideal_trace()
    verdict = st.sterility_index(times, relief, 1e-3)
    assert verdict.verdict == "failed"
    assert verdict.reason == "synthetic_collusion_detected"
    assert verdict.a2 < st.STERILE_A2
    assert verdict.synthetic_collusion is True


def test_lived_decay_passes():
    # 1 % noise is what the book writes into the medium (``written.jl``); the
    # detector must not confuse it with a formula.
    times, relief = _ideal_trace()
    rng = np.random.default_rng(7)
    lived = relief * (1.0 + 0.01 * rng.normal(size=times.size))
    verdict = st.sterility_index(times, lived, 1e-3)
    assert verdict.verdict == "passed"
    assert verdict.reason == "ok"
    assert verdict.a2 >= st.STERILE_A2
    assert verdict.cv >= st.STERILE_CV


def test_flat_decay_is_caught_by_the_variation_floor():
    # A relief with no spread at all is below the 0.5 % floor the written 1 %
    # miss puts under it: the second half of the statement, on its own.
    times = np.arange(0.0, 601.0, 6.0)
    verdict = st.sterility_index(times, np.full(times.size, 0.01), 0.0)
    assert verdict.cv < st.STERILE_CV
    assert verdict.verdict == "failed"


def test_short_trace_decides_nothing():
    times = np.arange(0.0, 66.0, 6.0)
    verdict = st.sterility_index(times, np.exp(-1e-3 * times), 1e-3)
    assert verdict.verdict == "short"
    assert verdict.reason == "insufficient_residuals"


def test_mismatched_series_is_an_error():
    with pytest.raises(ValueError):
        st.sterility_index([0.0, 6.0, 12.0], [1.0, 0.9], 1e-3)


# ── the operator of Statement #88 ───────────────────────────────────────────


def test_smooth_max_is_between_max_and_max_plus_log_k():
    values = [1.0, 0.9, 0.8]
    for beta in (1.0, 4.0, 16.0, 64.0):
        smoothed = st.smooth_max(values, beta)
        assert max(values) <= smoothed <= max(values) + math.log(len(values)) / beta


def test_smooth_max_rejects_bad_arguments():
    with pytest.raises(ValueError):
        st.smooth_max([1.0], 0.0)
    with pytest.raises(ValueError):
        st.smooth_max([], 1.0)


def test_banach_iteration_has_one_limit_from_two_starts():
    lift, slope = (0.0, 1.0, 2.0), (0.5, 0.2, -0.1)

    def rows_at(lam):
        return [lift[k] + slope[k] * lam for k in range(3)]

    operator = lambda lam: st.smooth_max(rows_at(lam), 16.0) - 0.8
    near = st.banach_iteration(operator, 1.0)
    far = st.banach_iteration(operator, 40.0)
    assert near.converged and far.converged
    assert abs(near.lam - far.lam) <= 1e-9
    assert near.rate < 1.0


def test_banach_iteration_reports_non_convergence():
    result = st.banach_iteration(lambda lam: lam + 1.0, 0.0, steps=5)
    assert result.converged is False


# ── the signal the Policy Engine reads ─────────────────────────────────────


def test_signal_carries_the_readings_the_engine_duck_types():
    times, relief = _ideal_trace()
    verdict = st.sterility_index(times, relief, 1e-3)
    signal = st.SterilitySignal.from_verdict("dev-1", verdict)
    assert signal.device_id == "dev-1"
    assert signal.synthetic_collusion is True
    assert signal.a2 == pytest.approx(verdict.a2)
    assert signal.cv == pytest.approx(verdict.cv)
    assert signal.details["reason"] == "synthetic_collusion_detected"

