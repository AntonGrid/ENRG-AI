"""The exit mechanics an operator may move, in one place, with one sentence each.

The resolution layer (:mod:`agent.backtest.power`) says the honest thing about this
strategy's record: with ``n`` trades of spread ``sigma``, the smallest edge that can be
read is ``c(n) * sigma / sqrt(n)``. The book on 2026-11-11 had 19 trades at 217 bps and
could resolve 171 bps — five times the gate. The count is expensive (554 trades for a
30 bps gate), and no input filter changed ``sigma`` at all. ``sigma`` is the cheap lever:
it is set by how a position ends, and how a position ends is a decision.

So the knobs below are not cosmetic. Each one trades a share of the upside for a narrower
distribution of outcomes, and the only way to know the price is to run the same
walk-forward with one knob moved — which is what the flags exist for: they reach the same
:class:`agent.backtest.engine.Settings` the ablations reach, from every entry point, so a
grid is a loop over a command line and not a patch.

The six knobs
=============

``--stop-mode`` / ``--atr-multiple`` / ``--atr-window``
    How far below the fill the stop sits: a fixed fraction, or a multiple of the bar's own
    average true range. The fixed 2 % stop fired on 64.6 % of entries **on their own entry
    bar** — on a pair whose ATR is 5 %, that stop is inside the noise and the trade is a
    coin flip that pays a fee either way. The ATR stop asks the pair how wide a bar is
    before deciding what a loss means.

``--risk-per-trade``
    The stake, as a fraction of the book risked at the stop, instead of "all the cash".
    It does not change a trade's return in basis points — the return is the return — but it
    changes whether one bad trade ends the run, and so whether the mean is the mean of the
    book an operator would actually have held.

``--take-profit-r``
    A target at a multiple of ``R``, the stop distance. It clips the right tail, which is
    exactly the tail a panic-reversal strategy is paid by; it narrows ``sigma`` and it is
    the knob most likely to narrow the *edge* faster.

``--trail-r``
    A stop that follows the best price at a multiple of ``R``. It keeps some of the tail and
    gives back some of it, and it is the honest middle between the two above.

``--time-stop-bars``
    A clock. A panic that has not resolved in ``n`` bars is a thesis with no evidence left;
    holding it longer adds ``sigma`` and no information.

What this file does not do
==========================

It does not choose. Every value here is refused if it cannot mean anything (a negative
``R``, a ``risk_per_trade`` above one, a window of zero bars) by
:meth:`agent.backtest.engine.Settings.__post_init__`, and none of them is fitted: a knob
turned until the backtest agrees is a knob that has learned the past. The grid is run, the
rows are read against the same gate and the same resolution, and a row that looks better
is a hypothesis for the next fold — never a result.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from agent.backtest.engine import STOP_MODES

#: Flag name (with the leading dashes) to the ``Settings`` field it sets. One table, so a
#: test can assert the CLI reaches every field and no entry point forgets one.
EXIT_FLAGS: Mapping[str, str] = {
    "--stop-mode": "stop_mode",
    "--atr-window": "atr_window",
    "--atr-multiple": "atr_multiple",
    "--risk-per-trade": "risk_per_trade",
    "--take-profit-r": "take_profit_r",
    "--trail-r": "trail_r",
    "--time-stop-bars": "time_stop_bars",
}

#: The same mapping with argparse's attribute spelling (``stop_mode``, ``atr_window`` …).
EXIT_ATTRS: Mapping[str, str] = {
    flag: field.replace("-", "_") for flag, field in EXIT_FLAGS.items()
}


def add_exit_arguments(parser: Any) -> None:
    """Register the exit-mechanics flags on a parser, identically in all four entry points.

    Every default is ``None`` on purpose. ``None`` means "this run did not move the knob",
    so :func:`exits_from_args` can return only the knobs a caller actually named — which is
    what keeps a run without these flags byte-identical to the run before they existed, and
    what lets an explicit flag win over an ablation's own override instead of being quietly
    replaced by it.
    """
    group = parser.add_argument_group(
        "exit mechanics",
        "how a position ends: the lever on per-trade spread, not on the physics",
    )
    group.add_argument(
        "--stop-mode",
        default=None,
        choices=STOP_MODES,
        help="how the stop distance is measured: 'fixed' (a fraction) or 'atr' (bars' own range)",
    )
    group.add_argument(
        "--atr-window",
        type=int,
        default=None,
        help="bars behind the average true range --stop-mode atr multiplies",
    )
    group.add_argument(
        "--atr-multiple",
        type=float,
        default=None,
        help="the stop distance in average true ranges when --stop-mode atr",
    )
    group.add_argument(
        "--risk-per-trade",
        type=float,
        default=None,
        help="fraction of the book risked at the stop (e.g. 0.02); 0 sizes by --share",
    )
    group.add_argument(
        "--take-profit-r",
        type=float,
        default=None,
        help="take-profit as a multiple of the stop distance (R); 0 is off",
    )
    group.add_argument(
        "--trail-r",
        type=float,
        default=None,
        help="trailing stop as a multiple of R from the best price; 0 is off",
    )
    group.add_argument(
        "--time-stop-bars",
        type=int,
        default=None,
        help="bars after which the position is closed at the next open; 0 is off",
    )


def exits_from_args(args: Any) -> Dict[str, Any]:
    """The ``Settings`` overrides a command line asked for — empty when it asked for none.

    Only ``None``-valued flags are skipped, so ``--take-profit-r 0`` reaches the engine as
    an explicit ``0.0`` (the same thing as "off", said out loud) rather than vanishing into
    the same silence as a flag nobody passed.
    """
    overrides: Dict[str, Any] = {}
    for flag, field in EXIT_FLAGS.items():
        attr = EXIT_ATTRS[flag]
        value = getattr(args, attr, None)
        if value is None:
            continue
        overrides[field] = int(value) if isinstance(value, int) else value
    return overrides


def describe_exits(overrides: Mapping[str, Any]) -> str:
    """One line for a report header, so a table is never read without its exit rules."""
    if not overrides:
        return "exits: the engine's defaults (fixed stop, no target, no clock)"
    parts = []
    if "stop_mode" in overrides:
        parts.append(f"stop {overrides['stop_mode']}")
    if "atr_window" in overrides:
        parts.append(f"atr {int(overrides['atr_window'])} bars")
    if "atr_multiple" in overrides:
        parts.append(f"{float(overrides['atr_multiple']):g}x ATR")
    if "risk_per_trade" in overrides:
        parts.append(f"risk {float(overrides['risk_per_trade']):.2%}")
    if "take_profit_r" in overrides:
        target = float(overrides["take_profit_r"])
        parts.append("no target" if target <= 0 else f"{target:g}R target")
    if "trail_r" in overrides:
        trail = float(overrides["trail_r"])
        parts.append("no trail" if trail <= 0 else f"{trail:g}R trail")
    if "time_stop_bars" in overrides:
        clock = int(overrides["time_stop_bars"])
        parts.append("no clock" if clock <= 0 else f"{clock}-bar clock")
    return "exits: " + " · ".join(parts)


__all__ = [
    "EXIT_ATTRS",
    "EXIT_FLAGS",
    "add_exit_arguments",
    "describe_exits",
    "exits_from_args",
]
