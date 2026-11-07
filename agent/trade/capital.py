"""How much a trade may be: the ceiling, the wallet, and the arithmetic between them.

The venue reads a balance (:meth:`agent.physics.real_testnet_provider.RealTestnetProvider.balance`),
the hunter produces a verdict, and this file is the one step nobody else can do for the
loop: turning those two facts into an **amount** with a reason attached. It holds no key,
imports no client and touches no network — every input is a number somebody else measured,
so the whole policy is testable in microseconds and reviewable in one screen.

Three ceilings, and the smallest wins
=====================================

``MAX_TRADE_SIZE_USD``   what the operator will risk on a single transaction. It is a
                         *ceiling*, not a target: the other two can only make the trade
                         smaller.
``MAX_TRADE_FRACTION``   a share of what the wallet actually holds (default 0.20, a fifth).
                         This is the ceiling that makes an $18 wallet unable to place a
                         $50 trade, and it is the one that matters on a box whose balance
                         changes without anyone editing a file.
``token balance``        the credits the vault can actually pull. A loop that sizes past
                         this one reverts, and a revert costs gas for nothing.

The gas reserve is a *refusal*, not a ceiling
=============================================

A stake spends the vault's ERC20 and costs a little of the native coin. If the wallet
cannot pay the fee, the trade is not small — it is impossible, and this file says so with
a sentence instead of letting the node say it with a revert. ``GAS_RESERVE_ETH`` defaults
to a figure that covers an ``approve`` **and** a ``stake`` on an L2 with room to spare.

What the confidence does
========================

A ``PANIC`` signal may raise the *operator's* dollar ceiling — by the same half the
portfolio caps its alpha boost at (``portfolio.ALPHA_BOOST_CAP``), scaled by the hunter's
own ``confidence``. It can never raise the **fraction**: "a fifth of the wallet" is the
answer to *what is the worst one mistake can do*, and a measurement is not allowed to
enlarge the size of the mistake it is measuring. So ``MAX_TRADE_SIZE_USD=50`` and a $68
wallet produce the same $13.60 whether the pool is calm or panicking — a panic just makes
sure the operator's ceiling is not the thing that stops the trade.

Why dollars, and what that assumes
==================================

The token a vault pulls is read as a dollar, because that is the assumption the venue
already makes (``DEFAULT_DECIMALS`` is a USDC-like six). A venue whose token is *not* a
stablecoin must not be sized by this file: ``eth_price_usd`` is how the native balance is
valued, and there is deliberately no price oracle here — a price the loop guessed is a
size the operator cannot audit. With no price the fraction applies to the stable balance
alone, and the cycle prints that it did.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "ALPHA_BOOST",
    "DEFAULT_GAS_RESERVE_ETH",
    "DEFAULT_MAX_FRACTION",
    "ENV_DRY_RUN",
    "ENV_ETH_PRICE",
    "ENV_GAS_RESERVE",
    "ENV_MAX_FRACTION",
    "ENV_MAX_TRADE_USD",
    "Sizing",
    "SizingError",
    "TradeSize",
    "size_trade",
]

ENV_DRY_RUN = "DRY_RUN"
ENV_MAX_TRADE_USD = "MAX_TRADE_SIZE_USD"
ENV_MAX_FRACTION = "MAX_TRADE_FRACTION"
ENV_ETH_PRICE = "ETH_USD_PRICE"
ENV_GAS_RESERVE = "GAS_RESERVE_ETH"

#: A fifth of the wallet. Chosen so that a wallet that is wrong about its own size is
#: still wrong five times before it is empty.
DEFAULT_MAX_FRACTION = 0.20

#: Enough for ``approve`` + ``stake`` on an L2 at a bad gas price, with room to spare.
#: On Arbitrum a stake costs ~0.0001–0.0003 ETH; on Ethereum L1 nothing is enough.
DEFAULT_GAS_RESERVE_ETH = 0.0005

#: The alpha boost a ``PANIC`` may earn, as a share of the capped size — the same half
#: ``agent.physics.portfolio.ALPHA_BOOST_CAP`` allows a bucket to grow by.
ALPHA_BOOST = 0.5


class SizingError(ValueError):
    """A configured number cannot be used — said before a trade, not after one."""


def _clean(value: object) -> str:
    """A value from an environment: stripped, and without the quotes a ``.env`` leaves."""
    text = str(value if value is not None else "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    return text


def _flag(value: object, default: bool = False) -> bool:
    """``1/true/yes/on`` and ``0/false/no/off``; anything else is the default.

    A misspelt ``DRY_RUN=ture`` therefore leaves the loop in whatever mode it already
    was — and the default there is ``True``, because the expensive mistake is the other
    one.
    """
    text = _clean(value).lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _number(value: object, default: float, name: str) -> float:
    """A finite float, or a named error — never a silent zero."""
    text = _clean(value)
    if not text:
        return float(default)
    try:
        number = float(text)
    except ValueError as exc:
        raise SizingError(f"{name} must be a number, got {text!r}") from exc
    if not math.isfinite(number):
        raise SizingError(f"{name} must be finite, got {text!r}")
    return number


@dataclass(frozen=True)
class Sizing:
    """The policy, as one value: three ceilings, a reserve, and whether to sign.

    ``dry_run`` lives here rather than in the loop because it is part of the *decision*
    the operator made, and a report that prints an amount without printing the mode it
    was computed in is a report someone will misread.
    """

    max_trade_usd: float = 0.0
    max_fraction: float = DEFAULT_MAX_FRACTION
    eth_price_usd: float = 0.0
    gas_reserve_eth: float = DEFAULT_GAS_RESERVE_ETH
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.max_trade_usd)) or float(self.max_trade_usd) < 0:
            raise SizingError(f"{ENV_MAX_TRADE_USD} must be a number >= 0")
        if not 0.0 < float(self.max_fraction) <= 1.0:
            raise SizingError(f"{ENV_MAX_FRACTION} must be within (0, 1]")
        if not math.isfinite(float(self.eth_price_usd)) or float(self.eth_price_usd) < 0:
            raise SizingError(f"{ENV_ETH_PRICE} must be a number >= 0")
        if not math.isfinite(float(self.gas_reserve_eth)) or float(self.gas_reserve_eth) < 0:
            raise SizingError(f"{ENV_GAS_RESERVE} must be a number >= 0")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Sizing":
        """Read every name in the module docstring from ``os.environ`` (or a mapping).

        ``DRY_RUN`` defaults to **true**: an unset variable is not an instruction to
        spend money, and a loop that has to be told twice before it signs is the only
        kind worth deploying unattended.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            max_trade_usd=_number(source.get(ENV_MAX_TRADE_USD), 0.0, ENV_MAX_TRADE_USD),
            max_fraction=_number(
                source.get(ENV_MAX_FRACTION), DEFAULT_MAX_FRACTION, ENV_MAX_FRACTION
            ),
            eth_price_usd=_number(source.get(ENV_ETH_PRICE), 0.0, ENV_ETH_PRICE),
            gas_reserve_eth=_number(
                source.get(ENV_GAS_RESERVE), DEFAULT_GAS_RESERVE_ETH, ENV_GAS_RESERVE
            ),
            dry_run=_flag(source.get(ENV_DRY_RUN), default=True),
        )

    def replace(self, **changes: Any) -> "Sizing":
        """A copy with some fields changed — the CLI's overrides, without a rewrite."""
        fields = {
            "max_trade_usd": self.max_trade_usd,
            "max_fraction": self.max_fraction,
            "eth_price_usd": self.eth_price_usd,
            "gas_reserve_eth": self.gas_reserve_eth,
            "dry_run": self.dry_run,
        }
        fields.update(changes)
        return Sizing(**fields)

    def describe(self) -> Dict[str, Any]:
        return {
            "max_trade_usd": float(self.max_trade_usd),
            "max_fraction": float(self.max_fraction),
            "eth_price_usd": float(self.eth_price_usd),
            "gas_reserve_eth": float(self.gas_reserve_eth),
            "dry_run": bool(self.dry_run),
        }

    def __str__(self) -> str:
        usd = (
            "no USD ceiling"
            if self.max_trade_usd <= 0.0
            else f"${self.max_trade_usd:,.2f} per trade"
        )
        price = (
            "ETH unpriced"
            if self.eth_price_usd <= 0.0
            else f"ETH ${self.eth_price_usd:,.2f}"
        )
        mode = "dry run" if self.dry_run else "LIVE"
        return (
            f"{mode}: min({usd}, {self.max_fraction:.0%} of the wallet) — {price}, "
            f"gas reserve {self.gas_reserve_eth:g} ETH"
        )


@dataclass(frozen=True)
class TradeSize:
    """The amount a cycle is allowed to move, and the sentence that explains it.

    ``bound`` names the ceiling that decided the number (``""`` when nothing traded),
    which is the difference between a report a reviewer can audit and one they have to
    re-derive: "20 % of the wallet" and "the operator's $50 ceiling" can produce the same
    number on different days and mean very different things.
    """

    amount: float = 0.0
    usd: float = 0.0
    equity_usd: float = 0.0
    bound: str = ""
    reason: str = ""
    refusal: str = ""

    @property
    def ok(self) -> bool:
        return not self.refusal and self.amount > 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "amount": round(float(self.amount), 6),
            "usd": round(float(self.usd), 4),
            "equity_usd": round(float(self.equity_usd), 4),
            "bound": self.bound,
            "reason": self.reason,
            "refusal": self.refusal,
        }

    def __str__(self) -> str:
        if self.refusal:
            return f"no trade: {self.refusal}"
        return f"{self.amount:.6f} (${self.usd:.2f}) — {self.reason}"


def size_trade(
    *,
    token_balance: float,
    native_eth: float,
    sizing: Sizing,
    confidence: float = 0.0,
) -> TradeSize:
    """The size of one trade: the smallest of three ceilings, or a named refusal.

    ``token_balance`` is the credit the vault pulls (what ``execute_stake`` spends) and
    ``native_eth`` is the gas money. They are separate arguments on purpose: a wallet with
    ether and no credits cannot stake, and the most common way a first trading run fails
    quietly is by confusing the two.
    """
    token = max(float(token_balance), 0.0)
    native = max(float(native_eth), 0.0)
    price = float(sizing.eth_price_usd)
    equity = token + (native * price if price > 0.0 else 0.0)

    # 1. can the wallet pay for the transaction at all?
    if native < float(sizing.gas_reserve_eth):
        return TradeSize(
            equity_usd=equity,
            bound="gas_reserve",
            refusal=(
                f"{native:.6f} ETH is below the {float(sizing.gas_reserve_eth):g} ETH gas "
                f"reserve: the stake could be approved and then never land"
            ),
        )

    # 2. how much is the ceiling, in dollars? A panic may lift the operator's dollar
    #    ceiling by at most half — and never the fraction, which is the survival cap.
    boost = 1.0 + ALPHA_BOOST * min(max(float(confidence), 0.0), 1.0)
    ceilings = []
    if sizing.max_trade_usd > 0.0:
        ceilings.append(
            (float(sizing.max_trade_usd) * boost, ENV_MAX_TRADE_USD + ("+alpha" if boost > 1.0 else ""))
        )
    if sizing.max_fraction > 0.0:
        ceilings.append((equity * float(sizing.max_fraction), ENV_MAX_FRACTION))
    if not ceilings:
        return TradeSize(
            equity_usd=equity,
            bound="no_ceiling",
            refusal=(
                f"no ceiling configured: set {ENV_MAX_TRADE_USD} or {ENV_MAX_FRACTION}"
            ),
        )
    cap, bound = min(ceilings, key=lambda pair: pair[0])
    wanted = cap

    # 3. what the wallet can actually move.
    if token <= 0.0:
        return TradeSize(
            equity_usd=equity,
            bound="token_balance",
            refusal=(
                f"the wallet holds {native:.6f} ETH of gas money and no credits: a stake "
                f"spends the vault's ERC20, and the native coin only pays the fee"
            ),
        )
    if wanted > token:
        wanted, bound = token, "token_balance"
    if wanted <= 0.0:
        return TradeSize(
            equity_usd=equity,
            bound=bound,
            refusal=f"the {bound or 'ceiling'} works out to {wanted:.6f}: nothing to stake",
        )

    share = f"{wanted / equity:.1%} of the wallet" if equity > 0.0 else "the whole balance"
    if price > 0.0:
        reason = f"{bound} decides it: {share} at ${wanted:,.2f} of ${equity:,.2f}"
    else:
        reason = (
            f"{bound} decides it: {share}, stable credits only — no {ENV_ETH_PRICE}, so "
            f"the {native:.6f} ETH is not valued"
        )
    return TradeSize(
        amount=round(float(wanted), 6),
        usd=float(wanted),
        equity_usd=equity,
        bound=bound,
        reason=reason,
    )
