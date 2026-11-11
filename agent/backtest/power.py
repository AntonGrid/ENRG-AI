"""How small an edge this many trades can resolve — the resolution of a trading claim.

:func:`agent.backtest.walkforward.summary` answers "did the book beat the gate". This module
answers the question underneath that one: *at this many trades, could an edge the size of
the gate have been seen at all*. They are different questions, and a report that prints only
the first invites a reader to take ``edge +12 bps < 30 bps`` as "the strategy is bad" when
the honest reading is "nineteen trades cannot see thirty basis points". A backtest that
cannot resolve its own gate has not measured a strategy; it has measured a coin.

The number here is the trading twin of ``b_min`` in ``entrain_infer`` — the smallest shift a
park of nodes can name at a declared risk and a declared power (``bench/REPORT.md`` §3: for
N = 10, T = 168 h, b_min is 11 min/day at alpha = 0.001). There the observable is
``max_k |t_k|`` over a park; here it is the mean net return of one trade, and the test is the
same one-sample t the gate already uses. Nothing is assumed here that the gate does not
already assume.

What is declared, not derived
=============================

=================================  =======================================================
``DECLARED_ALPHA`` = 0.05          two-sided risk: a book that makes nothing is called a
                                   winner one time in twenty — the usual line, declared
                                   here rather than left to a reader's taste
``DECLARED_POWER`` = 0.9           four times in five, an edge sitting exactly on the line
                                   is seen rather than missed
returns are normal                 the assumption the gate's own ``t`` already makes. A
                                   stopped strategy's returns are not normal — a heavy left
                                   tail makes the true resolution *worse* than the number
                                   printed, never better
=================================  =======================================================

The formula, in one line::

    b_min = nc * sigma / sqrt(n),  where nc solves
    P( T'(n-1, nc) > t_{1-alpha/2}(n-1) ) = power

``T'`` is the noncentral t: the distribution of a t statistic whose true mean is not zero.
The normal approximation ``(z_{1-alpha/2} + z_power) * sigma / sqrt(n)`` that most
sample-size tables print is optimistic exactly where backtests live — small ``n``, measured
at 161.6 bps against 171.0 bps exact for the sweep's own 19 trades. So it is not used here;
``tests/test_backtest_power.py`` checks the module against the closed-form limit
``z_{0.975} + z_{0.9} = 3.2415`` at large ``n`` instead.

One numerical fact worth writing down, because it cost a debugging session: ``scipy``'s
``nct`` returns ``nan`` in the far left tail (``nct.cdf(-c, 1, 8)``), and a root search over
basis points therefore walks into ``nan`` and dies. Two consequences, both in the code: the
tail terms are read as zero when they underflow (they are below ``1e-300`` there, so the
reading is exact to any precision that matters), and the root is taken over the
**noncentrality**, whose bracket is bounded and safe, rather than over bps.

What this module does not do
============================

It does not say whether an edge is real — that is :mod:`agent.backtest.null` (are these bets
better than the same number of coin-flipped ones) and the gate's own ``t`` in
:mod:`agent.backtest.walkforward`. It says only how finely the book's record can be read,
which is a property of ``n`` and ``sigma`` and of nothing else. And it inherits one
dependence it cannot fix: trades that overlap in time are not independent bets, so the
effective ``n`` is smaller than the count — the modules that produce these numbers keep
trades non-overlapping for that reason, and a caller who pools overlapping books gets a
resolution that is too fine.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Sequence

from scipy import optimize, stats

#: The two-sided risk a claim is read at: one false winner in twenty.
DECLARED_ALPHA = 0.05

#: The chance of seeing an edge that sits exactly on the line.
DECLARED_POWER = 0.9

#: The widest noncentrality the root search will look at. At ``100`` the power is 1.0 to
#: machine precision for every degree of freedom, and the far-left tail is still computable,
#: so the bracket cannot miss a root and cannot wander into ``nan`` (measured: the widest
#: root is ``n = 2``, which needs 20.96).
NC_MAX = 100.0

#: The largest book :func:`trades_needed` will search before it says "unreachable".
CEILING = 10_000_000


class ResolutionError(ValueError):
    """A resolution was asked of numbers that cannot have one."""


def _check(trades: int, sigma_bps: float, alpha: float, power: float) -> None:
    """Refuse the arguments that would make the arithmetic meaningless, with a sentence."""
    if int(trades) < 2:
        raise ResolutionError(
            f"{int(trades)} trade(s) have no dispersion to read: a resolution needs at "
            "least two returns, and an honest one needs hundreds"
        )
    if not math.isfinite(float(sigma_bps)) or float(sigma_bps) < 0.0:
        raise ResolutionError(
            f"sigma {sigma_bps!r} is not a spread: it must be a finite, non-negative number "
            "of basis points"
        )
    if not 0.0 < float(alpha) < 1.0:
        raise ResolutionError(f"alpha {alpha!r} is not a risk: it must lie in (0, 1)")
    if not 0.0 < float(power) < 1.0:
        raise ResolutionError(f"power {power!r} is not a chance: it must lie in (0, 1)")


def _clamped(value: float) -> float:
    """A probability from scipy, with the underflow ``nan`` read as the zero it stands for."""
    number = float(value)
    if math.isnan(number):
        return 0.0
    return min(max(number, 0.0), 1.0)


def spread_bps(returns: Sequence[float]) -> float:
    """The per-trade spread of a book, in basis points — the ``sigma`` of every formula here.

    Sample standard deviation (``n - 1``), because the book is a sample of a strategy and not
    the strategy itself. One trade has no spread: ``0.0`` is returned and the caller's
    resolution will say so rather than dividing by nothing.
    """
    values = [float(value) for value in returns]
    if len(values) < 2:
        return 0.0
    return 1e4 * float(statistics.stdev(values))


def _power_at_noncentrality(noncentrality: float, nu: int, alpha: float) -> float:
    """The two-sided power of a t test whose true mean is ``noncentrality`` standard errors.

    Both tails are counted: a book with a real *loss* is meant to be called a loss, and a
    power that ignored the left tail would overstate what the test can do near the edge.
    """
    critical = float(stats.t.ppf(1.0 - float(alpha) / 2.0, int(nu)))
    upper = _clamped(stats.nct.sf(critical, int(nu), float(noncentrality)))
    lower = _clamped(stats.nct.cdf(-critical, int(nu), float(noncentrality)))
    return min(upper + lower, 1.0)


def _root_noncentrality(nu: int, alpha: float, power: float) -> float:
    """The noncentrality at which the test reaches ``power`` — the root of the power curve."""

    def gap(noncentrality: float) -> float:
        return _power_at_noncentrality(noncentrality, int(nu), float(alpha)) - float(power)

    if gap(NC_MAX) <= 0.0:
        raise ResolutionError(
            f"power {power!r} is not reachable at {int(nu) + 1} trades with alpha {alpha!r}: "
            f"even a {NC_MAX:.0f}-sigma edge stays below it, so no edge is measurable here"
        )
    return float(optimize.brentq(gap, 0.0, NC_MAX, xtol=1e-10, rtol=1e-12, maxiter=400))


def power_at(
    edge_bps: float,
    trades: int,
    sigma_bps: float,
    *,
    alpha: float = DECLARED_ALPHA,
) -> float:
    """The chance this many trades flags an edge of ``edge_bps`` when ``alpha`` is declared.

    ``edge_bps = 0`` gives back ``alpha`` itself — the test's own false-positive rate, the
    one value of this function a reader can check without a table.
    """
    _check(trades, sigma_bps, alpha, 0.5)
    if float(sigma_bps) <= 0.0:
        return 1.0 if float(edge_bps) else float(alpha)
    noncentrality = float(edge_bps) * math.sqrt(int(trades)) / float(sigma_bps)
    return _power_at_noncentrality(noncentrality, int(trades) - 1, float(alpha))


def minimum_edge_bps(
    trades: int,
    sigma_bps: float,
    *,
    alpha: float = DECLARED_ALPHA,
    power: float = DECLARED_POWER,
) -> float:
    """The smallest mean edge ``n`` trades of this spread can name at ``alpha`` / ``power``.

    This is ``b_min`` for a trading claim: below it the book's own noise is wider than the
    thing being looked for, so the gate's verdict would be decided by which side of zero a
    rounding error fell on. A perfectly repeatable book (``sigma = 0``) resolves any edge and
    gets ``0.0`` — reported rather than refused, because a caller may be holding several
    identical trades.
    """
    _check(trades, sigma_bps, alpha, power)
    if float(sigma_bps) <= 0.0:
        return 0.0
    root = _root_noncentrality(int(trades) - 1, float(alpha), float(power))
    return root * float(sigma_bps) / math.sqrt(int(trades))


def trades_needed(
    edge_bps: float,
    sigma_bps: float,
    *,
    alpha: float = DECLARED_ALPHA,
    power: float = DECLARED_POWER,
) -> float:
    """How many trades it takes to resolve ``edge_bps`` — the answer to "run longer?".

    Returned as a float so that an unreachable edge is ``inf`` rather than an exception: the
    callers are reports, and a report that crashed on the honest answer would be a report
    that never printed it. The search starts at the normal approximation, which is always
    smaller than the truth, and walks up one trade at a time — so the result is the exact
    smallest ``n`` for the declared risk and power, not an estimate of it.
    """
    if float(edge_bps) <= 0.0:
        raise ResolutionError(
            f"edge {edge_bps!r} bps is not an edge: nothing is resolved by resolving zero"
        )
    if not 0.0 < float(alpha) < 1.0:
        raise ResolutionError(f"alpha {alpha!r} is not a risk: it must lie in (0, 1)")
    if not 0.0 < float(power) < 1.0:
        raise ResolutionError(f"power {power!r} is not a chance: it must lie in (0, 1)")
    if not math.isfinite(float(sigma_bps)) or float(sigma_bps) < 0.0:
        raise ResolutionError(f"sigma {sigma_bps!r} is not a spread")
    if float(sigma_bps) <= 0.0:
        return 2.0
    z = float(stats.norm.ppf(1.0 - float(alpha) / 2.0) + stats.norm.ppf(float(power)))
    trades = max(2, int(math.ceil((z * float(sigma_bps) / float(edge_bps)) ** 2)))
    while trades > 2 and minimum_edge_bps(
        trades - 1, sigma_bps, alpha=alpha, power=power
    ) <= float(edge_bps):
        trades -= 1
    while trades <= CEILING and minimum_edge_bps(
        trades, sigma_bps, alpha=alpha, power=power
    ) > float(edge_bps):
        trades += 1
    return float(trades) if trades <= CEILING else math.inf


@dataclass(frozen=True)
class Resolution:
    """A book's own reading resolution, set beside the gate it is asked to clear.

    ``min_edge_bps`` is ``inf`` when there are fewer than two trades: not a large number,
    but the honest absence of one — the book has no spread to divide by, so no edge is
    measurable in either direction, and "unmeasured" is not "disproved".
    """

    trades: int
    sigma_bps: float
    gate_bps: float
    min_edge_bps: float
    alpha: float = DECLARED_ALPHA
    power: float = DECLARED_POWER

    @property
    def measurable(self) -> bool:
        """Can this book settle the gate's question at all?"""
        return math.isfinite(self.min_edge_bps) and self.min_edge_bps <= float(self.gate_bps)

    @property
    def needed_trades(self) -> float:
        """How many trades the gate's own edge would need — ``inf`` when it cannot be had."""
        if float(self.gate_bps) <= 0.0 or math.isinf(self.min_edge_bps):
            return math.inf
        if self.measurable:
            return float(self.trades)
        return trades_needed(
            self.gate_bps, self.sigma_bps, alpha=self.alpha, power=self.power
        )

    def describe(self) -> str:
        """One sentence for an operator, with the limit attached where it belongs."""
        if not math.isfinite(self.min_edge_bps):
            return (
                f"{int(self.trades)} trade(s): no dispersion, so no resolution — the book "
                f"is unmeasured rather than disproved, and the gate's "
                f"{float(self.gate_bps):.0f} bps cannot be read from it"
            )
        heard = (
            f"{int(self.trades)} trades at {float(self.sigma_bps):.0f} bps of spread "
            f"resolve {float(self.min_edge_bps):.0f} bps at alpha {float(self.alpha):.2f} "
            f"/ power {float(self.power):.0%}"
        )
        if self.measurable:
            return f"{heard}: the gate's {float(self.gate_bps):.0f} bps is inside that"
        needed = self.needed_trades
        wanted = "unreachable" if math.isinf(needed) else f"{needed:.0f}"
        return (
            f"{heard}, which is outside the gate's {float(self.gate_bps):.0f} bps: "
            f"{wanted} trades would be needed, so this many cannot say yes and cannot say no"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trades": int(self.trades),
            "sigma_bps": round(float(self.sigma_bps), 4),
            "gate_bps": round(float(self.gate_bps), 4),
            "min_edge_bps": (
                None if math.isinf(self.min_edge_bps) else round(float(self.min_edge_bps), 4)
            ),
            "alpha": float(self.alpha),
            "power": float(self.power),
            "measurable": bool(self.measurable),
            "needed_trades": (
                None if math.isinf(self.needed_trades) else int(self.needed_trades)
            ),
        }


def resolution(
    trades: int,
    sigma_bps: float,
    gate_bps: float,
    *,
    alpha: float = DECLARED_ALPHA,
    power: float = DECLARED_POWER,
) -> Resolution:
    """The resolution of a book, tolerating the counts too small to have one.

    The strict functions above raise on a single trade; a report cannot, because a single
    trade is a thing that happens. Here it becomes ``min_edge_bps = inf`` with the reason in
    :meth:`Resolution.describe`, so the sentence reaches the operator instead of a traceback.
    """
    if int(trades) < 2 or not math.isfinite(float(sigma_bps)):
        return Resolution(
            trades=int(trades),
            sigma_bps=float(sigma_bps) if math.isfinite(float(sigma_bps)) else 0.0,
            gate_bps=float(gate_bps),
            min_edge_bps=math.inf,
            alpha=float(alpha),
            power=float(power),
        )
    return Resolution(
        trades=int(trades),
        sigma_bps=float(sigma_bps),
        gate_bps=float(gate_bps),
        min_edge_bps=minimum_edge_bps(
            int(trades), float(sigma_bps), alpha=float(alpha), power=float(power)
        ),
        alpha=float(alpha),
        power=float(power),
    )


__all__ = [
    "CEILING",
    "DECLARED_ALPHA",
    "DECLARED_POWER",
    "NC_MAX",
    "Resolution",
    "ResolutionError",
    "minimum_edge_bps",
    "power_at",
    "resolution",
    "spread_bps",
    "trades_needed",
]
