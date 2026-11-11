"""A venue made of candles: the same operations as a chain, and none of the keys.

:class:`agent.physics.interfaces.IWeb3Provider` is five operations and three facts, and
everything in this layer that moves capital is a call to one of them. The real end of that
seam (:class:`agent.physics.real_testnet_provider.RealTestnetProvider`) answers by signing
transactions. This file answers by reading the next row of a price series, which is the
whole trick of a backtest: the *decisions* are unchanged, and only the venue is a memory.

What it copies, and what it deliberately does not
=================================================

It copies the shape — ``name``, ``step``, ``whitelist``, ``get_protocol_list``,
``fetch_protocol_history``, ``submit_validation_proof``, ``execute_stake``, ``withdraw``,
``advance`` — and the last one is the only one with a different meaning: on a chain
``advance`` lets a block happen, here it lets a *bar* happen. A loop that reads the clock
and settles its orders against it runs unchanged against this venue, which is exactly the
claim worth testing.

It does not copy the key, the ABI, the RPC, the gas, the nonce or the revert. There is
nothing in this file to sign with, nothing to send, and no address that could receive
anything: :attr:`SimulatedProvider.address` is a label (``sim:btcusdt``) and not a wallet,
and a test asserts that the package never imports ``web3`` at all.

The three assumptions, each one declared
========================================

**Slippage is 0.1 % on spot.** A market order does not fill at the price that was on the
screen, and a backtest that fills at the close is a backtest of a market that does not
exist. 0.1 % is the number the brief names; it is a *field*, so an operator trading a
thinner pair can raise it and see what the strategy's edge does when the edge is expensive.

**The fee is 0.1 % a side.** Same argument, same field, and the report prints the total, so
a strategy that is profitable only before fees is visible instead of impressive.

**A decision settles one bar later, at the next open.** This is the network delay, and it is
expressed in bars because a bar is the only clock here. It is not a detail: the decision at
bar *t* is made on the close of *t*, and a venue that filled it at that same close would be
a venue that let the order arrive before the price existed. ``latency_bars=0`` is available
— and is exactly the optimistic case a reader should compare against — while the default
stays at one bar, because the default is the one an operator will quote.

The stop-loss is the one order that does **not** wait for the next bar, and that asymmetry
is deliberate: a resting stop on a real venue is executed inside the bar that crosses it.
It fills at the stop price when the bar traded through it, and at the bar's *open* when the
bar gapped below it — a backtest that filled a gap at the stop price would be a backtest
that invented seven per cent of return on every crash.

Failure is a value here too
===========================

:meth:`SimulatedProvider.execute_stake` answers ``False`` and records a :class:`Refusal`
when the account cannot honour the order — no cash, an open position, an amount of zero —
instead of raising. The live provider does the same thing for the same reason: a loop's
answer to "the venue said no" is to trade nothing this cycle, and a refusal with a sentence
is what the report prints. :attr:`SimulatedProvider.fills` is the ledger of what actually
happened; nothing in this file computes a profit, and the engine reads the fills rather
than the intent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.backtest.data import Candle, CandleSeries, declines
from agent.physics.portfolio import WhiteList

__all__ = [
    "FEE_RATE",
    "FILL_SLIPPAGE",
    "LATENCY_BARS",
    "Fill",
    "Refusal",
    "SimBalance",
    "SimulatedProvider",
]

#: What a market order costs in price: 0.1 % against the trader, in both directions.
FILL_SLIPPAGE = 0.001

#: What the venue charges per side: 0.1 %, the spot taker rate.
FEE_RATE = 0.001

#: Bars between a decision and its fill. One, because the close a decision was made on has
#: already happened by the time the order can arrive.
LATENCY_BARS = 1


@dataclass(frozen=True)
class Fill:
    """One executed order: what the venue actually did, at what price, and why.

    ``reason`` is the *trigger* (``open``, ``verdict``, ``stop``) and not a verdict: the
    engine pairs a fill with the decision that caused it, and keeping the two apart is how
    a report can say "stopped out" without the venue having an opinion about the strategy.

    ``usd`` is signed by the side without being negative: on a buy it is what the account
    *paid* (fee included), on a sell it is what it *received* (fee excluded). Profit is
    therefore one subtraction of those two numbers and nothing else — which is the reason
    the field is not called ``value``.
    """

    bar: int
    at: float
    side: str
    price: float
    units: float
    usd: float
    fee: float
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bar": int(self.bar),
            "at": float(self.at),
            "side": self.side,
            "price": round(float(self.price), 8),
            "units": round(float(self.units), 10),
            "usd": round(float(self.usd), 4),
            "fee": round(float(self.fee), 6),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Refusal:
    """An order the venue would not take, and the sentence explaining it."""

    bar: int
    at: float
    side: str
    usd: float
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "bar": int(self.bar),
            "at": float(self.at),
            "side": self.side,
            "usd": round(float(self.usd), 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SimBalance:
    """What the account holds at the current bar — the backtest's ``eth_getBalance``.

    ``equity`` is the number a report compounds and a drawdown is measured on: cash plus
    the position marked at the bar's close. ``error`` exists so that the shape matches the
    real venue's ``Balance``, where a failed read is a value and never an exception; here a
    read can only fail before the first bar.
    """

    cash: float = 0.0
    units: float = 0.0
    price: float = 0.0
    equity: float = 0.0
    at: float = 0.0
    bar: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def staked(self) -> bool:
        return self.units > 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cash": round(float(self.cash), 6),
            "units": round(float(self.units), 10),
            "price": round(float(self.price), 8),
            "equity": round(float(self.equity), 6),
            "at": float(self.at),
            "bar": int(self.bar),
            "error": self.error,
        }


@dataclass
class _Order:
    """An order waiting for its bar: the venue's own memory, not the engine's."""

    side: str
    due: int
    usd: float = 0.0
    units: float = 0.0
    stop_pct: float = 0.0
    reason: str = ""


@dataclass
class SimulatedProvider:
    """The venue: a series, a clock, and a ledger — and no key anywhere in it.

    The clock is :attr:`index` and it only moves forward through :meth:`advance`, which is
    the same discipline the chain imposes on the live loop: a bar happens, and only then can
    anything be read from it. Orders queued on one bar settle on the next one, at that bar's
    open, so a decision can never be filled at a price it was allowed to see.
    """

    series: CandleSeries
    cash: float = 1000.0
    slippage: float = FILL_SLIPPAGE
    fee: float = FEE_RATE
    latency_bars: int = LATENCY_BARS
    name: str = "simulated"
    address: str = ""
    index: int = 0
    units: float = 0.0
    entry_price: float = 0.0
    entry_bar: int = -1
    stop_price: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    refusals: List[Refusal] = field(default_factory=list)
    orders: List[_Order] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.address:
            self.address = f"sim:{self.series.symbol or 'symbol'}"
        self.index = max(int(self.index), 0)
        if self.series.rows and self.index >= len(self.series.rows):
            self.index = len(self.series.rows) - 1
        self.cash = float(self.cash)
        if not math.isfinite(self.cash) or self.cash < 0.0:
            raise ValueError(f"cash must be a non-negative number, not {self.cash!r}")

    # -- the facts --------------------------------------------------------- #

    @property
    def step(self) -> int:
        """The venue's clock: the index of the last bar that has happened."""
        return int(self.index)

    @property
    def whitelist(self) -> WhiteList:
        """Empty, and that is the point: nothing here can be approved to receive capital.

        The live provider's whitelist is the operator's book and the reason capital cannot
        reach an address nobody named. This venue has no addresses at all, so the list is
        empty and the engine never consults it.
        """
        return WhiteList()

    @property
    def bar(self) -> Optional[Candle]:
        """The bar the clock is standing on, or ``None`` on an empty series."""
        if not self.series.rows:
            return None
        return self.series.rows[min(max(self.index, 0), len(self.series.rows) - 1)]

    @property
    def price(self) -> float:
        """The close of the current bar — the price a decision is allowed to see."""
        bar = self.bar
        return float(bar.close) if bar is not None else 0.0

    @property
    def at(self) -> float:
        """The open time of the current bar."""
        bar = self.bar
        return float(bar.at) if bar is not None else 0.0

    @property
    def holding(self) -> bool:
        return self.units > 0.0

    @property
    def busy(self) -> bool:
        """True while an order is still waiting for its bar."""
        return bool(self.orders)

    @property
    def equity(self) -> float:
        return float(self.cash) + float(self.units) * self.price

    # -- the clock ---------------------------------------------------------- #

    def to_bar(self, index: int) -> int:
        """Move the clock to ``index``, settling every bar in between; returns the clock."""
        target = int(index)
        while self.index < target:
            self.advance()
        return self.index

    def advance(self, steps: int = 1) -> int:
        """Let ``steps`` bars happen; each one settles due orders and runs the stop.

        The stop runs *after* the settlement, so a position opened at this bar's open can
        be stopped out inside the same bar — which is what happens on a real venue and what
        a backtest must not smooth away.
        """
        for _ in range(max(int(steps), 1)):
            if self.index + 1 >= len(self.series.rows):
                return self.index
            self.index += 1
            self._settle()
            self._stop()
        return self.index

    # -- the five the layer is built on ------------------------------------- #

    def execute_stake(
        self,
        protocol_address: str,
        amount: float,
        *,
        stop_pct: float = 0.0,
        reason: str = "open",
    ) -> bool:
        """Buy ``amount`` USDT worth of the instrument; ``True`` when the order was taken.

        ``protocol_address`` is ignored on purpose and the argument is kept because the
        interface declares it: on a chain it names the vault, and here there is exactly one
        instrument — the series this venue was built from. ``stop_pct`` rides on the ticket
        rather than arriving later, because a stop is a property of the position being
        opened and a venue that could open a position without one would be a venue that
        could be held overnight with no exit by a bug in the caller.
        """
        if self.holding:
            self._refuse("buy", float(amount), "this venue holds one position at a time")
            return False
        if self.busy:
            self._refuse("buy", float(amount), "an order is already waiting for its bar")
            return False
        budget = self._budget(float(amount))
        if budget <= 0.0:
            self._refuse(
                "buy",
                float(amount),
                f"${float(amount):,.2f} against ${self.cash:,.2f} of cash is nothing to buy",
            )
            return False
        order = _Order(
            side="buy",
            due=self.index + max(int(self.latency_bars), 0),
            usd=budget,
            stop_pct=max(float(stop_pct), 0.0),
            reason=str(reason),
        )
        if int(self.latency_bars) <= 0:
            self._execute(order)
        else:
            self.orders.append(order)
        return True

    def withdraw(
        self,
        protocol_address: str,
        amount: float,
        *,
        reason: str = "verdict",
    ) -> bool:
        """Sell ``amount`` **units** of the instrument; ``True`` when the order was taken.

        Units and not dollars, because that is what the live provider's ``withdraw`` takes
        out of a vault: the engine sells *the position it holds*, and a sell sized in
        dollars would leave a residue that is an artefact of the price rather than a
        decision.
        """
        if not self.holding:
            self._refuse("sell", float(amount), "there is no position to close")
            return False
        if self.busy:
            self._refuse("sell", float(amount), "an order is already waiting for its bar")
            return False
        units = min(float(amount), self.units)
        if units <= 0.0:
            self._refuse("sell", float(amount), "nothing to sell")
            return False
        order = _Order(
            side="sell",
            due=self.index + max(int(self.latency_bars), 0),
            units=units,
            reason=str(reason),
        )
        if int(self.latency_bars) <= 0:
            self._execute(order)
        else:
            self.orders.append(order)
        return True

    def balance(
        self,
        protocol_address: Optional[str] = None,
        *,
        address: Optional[str] = None,
    ) -> SimBalance:
        """Cash, position and equity at the current bar — never an exception.

        ``protocol_address`` is accepted for the same reason the interface asks for it on
        the real venue: a caller that sizes a trade against a wallet should not have to know
        which venue answered.
        """
        bar = self.bar
        if bar is None:
            return SimBalance(error="this venue was built from an empty series")
        price = float(bar.close)
        return SimBalance(
            cash=float(self.cash),
            units=float(self.units),
            price=price,
            equity=float(self.cash) + float(self.units) * price,
            at=float(bar.at),
            bar=int(self.index),
        )

    # -- the rest of the interface, answered honestly ------------------------ #

    def get_protocol_list(self) -> List[Dict[str, Any]]:
        """The one instrument on offer — the same shape a vault's listing has."""
        bar = self.bar
        return [
            {
                "address": self.address,
                "name": self.series.symbol or "instrument",
                "chain": "simulated",
                "kind": "spot",
                "price": None if bar is None else float(bar.close),
                "step": int(self.index),
            }
        ]

    def fetch_protocol_history(
        self, address: str, steps: Optional[int] = None
    ) -> List[float]:
        """The record at ``address``, up to the clock and never past it.

        This is the same reading the hunter makes of a pool: the per-step rates the venue
        has published so far. It stops at :attr:`index` because a backtest that could hand
        the assessment a record including the future would be measuring its own answer.
        """
        if not self.series.rows:
            return []
        upto = min(int(self.index) + 1, len(self.series.rows))
        values = declines(self.series.rows[:upto])
        published = [float(value) for value in values if math.isfinite(value)]
        if steps is not None and int(steps) > 0:
            published = published[-int(steps) :]
        return published

    def submit_validation_proof(self, protocol_address: str, proof_data: Any) -> bool:
        """Always ``False``: this venue keeps no evidence but the fills.

        The real provider hands a verdict's evidence to a contract. Nothing here is
        transmitted anywhere, and answering ``True`` would be a claim that something was
        stored — so the honest answer is the refusal, with the reason in the docstring.
        """
        return False

    def as_dict(self) -> Dict[str, Any]:
        """The venue, described for the report: what it is, and what it assumed."""
        return {
            "name": self.name,
            "address": self.address,
            "symbol": self.series.symbol,
            "period": self.series.period,
            "bars": len(self.series.rows),
            "slippage": float(self.slippage),
            "fee": float(self.fee),
            "latency_bars": int(self.latency_bars),
            "filled": len(self.fills),
            "refused": len(self.refusals),
        }

    # -- the book, which is the only thing that moves ----------------------- #

    def _budget(self, amount: float) -> float:
        """What of ``amount`` the account can actually pay, in USDT."""
        if not math.isfinite(amount) or amount <= 0.0:
            return 0.0
        return float(min(amount, self.cash))

    def _refuse(self, side: str, usd: float, reason: str) -> None:
        self.refusals.append(
            Refusal(bar=int(self.index), at=self.at, side=side, usd=float(usd), reason=reason)
        )

    def _settle(self) -> None:
        """Fill every order whose bar has arrived, at that bar's open."""
        if not self.orders:
            return
        due = [order for order in self.orders if order.due <= self.index]
        for order in due:
            self.orders.remove(order)
            self._execute(order)

    def _execute(self, order: _Order) -> None:
        """The one path both sides take, so neither can skip the price it was given."""
        bar = self.bar
        if bar is None:
            return
        opening = float(bar.open) if bar.open > 0.0 else float(bar.close)
        if order.side == "buy":
            self._buy(order.usd, opening * (1.0 + float(self.slippage)), order)
        else:
            self._sell(order.units, opening * (1.0 - float(self.slippage)), order)

    def _buy(self, usd: float, price: float, order: _Order) -> None:
        budget = self._budget(usd)
        if budget <= 0.0 or price <= 0.0:
            self._refuse("buy", usd, "the account had nothing left to spend when the bar opened")
            return
        charged = budget * float(self.fee)
        units = (budget - charged) / price
        if units <= 0.0:
            self._refuse("buy", usd, "the fee consumed the whole order at this price")
            return
        self.cash = float(self.cash) - budget
        self.units = float(units)
        self.entry_price = float(price)
        self.entry_bar = int(self.index)
        self.stop_price = (
            float(price) * (1.0 - float(order.stop_pct)) if float(order.stop_pct) > 0.0 else 0.0
        )
        self.fills.append(
            Fill(
                bar=int(self.index),
                at=self.at,
                side="buy",
                price=float(price),
                units=float(units),
                usd=float(budget),
                fee=float(charged),
                reason=order.reason,
            )
        )

    def _sell(self, units: float, price: float, order: _Order) -> None:
        held = min(float(units), float(self.units))
        if held <= 0.0 or price <= 0.0:
            self._refuse(
                "sell", float(units), "there was nothing left to sell when the bar opened"
            )
            return
        gross = held * price
        charged = gross * float(self.fee)
        self.cash = float(self.cash) + gross - charged
        self.units = float(self.units) - held
        if self.units <= 1.0e-12:
            self.units = 0.0
            self.entry_price = 0.0
            self.entry_bar = -1
            self.stop_price = 0.0
        self.fills.append(
            Fill(
                bar=int(self.index),
                at=self.at,
                side="sell",
                price=float(price),
                units=float(held),
                usd=float(gross - charged),
                fee=float(charged),
                reason=order.reason,
            )
        )

    def _stop(self) -> None:
        """The resting stop, executed inside the bar that crosses it.

        Two prices matter and they are not the same: the stop price the position named, and
        the price the bar actually opened at. When the bar opened *below* the stop the order
        fills at the open — the gap is real, and a backtest that filled it at the stop would
        be inventing the difference — and otherwise it fills at the stop itself.
        """
        if not self.holding or self.stop_price <= 0.0:
            return
        bar = self.bar
        if bar is None or float(bar.low) > float(self.stop_price):
            return
        crossing = float(self.stop_price)
        opening = float(bar.open) if bar.open > 0.0 else crossing
        price = opening if opening < crossing else crossing
        self._sell(
            float(self.units),
            price * (1.0 - float(self.slippage)),
            _Order(side="sell", due=int(self.index), reason="stop"),
        )
