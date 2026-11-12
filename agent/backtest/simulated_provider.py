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
    take_profit_pct: float = 0.0
    trail_pct: float = 0.0
    max_bars: int = 0
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
    take_profit_price: float = 0.0
    trail_pct: float = 0.0
    peak_price: float = 0.0
    max_bars: int = 0
    mfe: float = 0.0
    mae: float = 0.0
    last_mfe: float = 0.0
    last_mae: float = 0.0
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
        """True on either side: a short is a position, and a venue that answered otherwise
        would let its caller open a second one on top of it."""
        return float(self.units) != 0.0

    @property
    def long(self) -> bool:
        """True while the ledger holds units it bought — the side every exit rule knew first."""
        return float(self.units) > 0.0

    @property
    def short(self) -> bool:
        """True while the ledger owes units it sold — the same position with the sign turned.

        The units are negative and the proceeds of the sale sit in :attr:`cash` as collateral,
        so :attr:`equity` — ``cash + units × price`` — stays the one arithmetic that is right
        on both sides. There is no borrow fee, no funding rate and no liquidation price in this
        ledger: they are declared absent in the report rather than modelled as zero, because a
        zero that looks like a measurement is worse than a sentence that says what is missing.
        """
        return float(self.units) < 0.0

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
        """Let ``steps`` bars happen; each one settles due orders and runs the exits.

        The stop runs *after* the settlement, so a position opened at this bar's open can
        be stopped out inside the same bar — which is what happens on a real venue and what
        a backtest must not smooth away.

        The order of the four exits is a judgement, and it is the pessimistic one: the stop
        is checked before the take-profit, so a bar that touched both is a loss. Nothing in
        a daily candle says which came first, and a backtest that resolves that ambiguity in
        its own favour is a backtest reporting an exit it invented.
        """
        for _ in range(max(int(steps), 1)):
            if self.index + 1 >= len(self.series.rows):
                return self.index
            self.index += 1
            self._settle()
            self._stop()
            self._take_profit()
            self._trail()
            self._time_stop()
            self._mark()
        return self.index

    # -- the five the layer is built on ------------------------------------- #

    def execute_stake(
        self,
        protocol_address: str,
        amount: float,
        *,
        stop_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        trail_pct: float = 0.0,
        max_bars: int = 0,
        reason: str = "open",
        direction: str = "long",
    ) -> bool:
        """Open ``amount`` USDT of exposure; ``True`` when the order was taken.

        ``protocol_address`` is ignored on purpose and the argument is kept because the
        interface declares it: on a chain it names the vault, and here there is exactly one
        instrument — the series this venue was built from. ``stop_pct`` rides on the ticket
        rather than arriving later, because a stop is a property of the position being
        opened and a venue that could open a position without one would be a venue that
        could be held overnight with no exit by a bug in the caller.

        ``direction="short"`` opens the same size on the other side: the venue sells units it
        does not own and keeps the proceeds as collateral, so the ledger's position is
        negative and every rule below — the stop, the target, the trail, the clock — reads the
        same fields with the sign of the move it is watching. The size is capped by cash and
        not by leverage: this is a 1× book, and a 1× book cannot be liquidated by a wick.
        """
        wanted = str(direction).strip().lower()
        if wanted not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', not {direction!r}")
        side = "buy" if wanted == "long" else "short"
        if self.holding:
            self._refuse(side, float(amount), "this venue holds one position at a time")
            return False
        if self.busy:
            self._refuse(side, float(amount), "an order is already waiting for its bar")
            return False
        budget = self._budget(float(amount))
        if budget <= 0.0:
            self._refuse(
                side,
                float(amount),
                f"${float(amount):,.2f} against ${self.cash:,.2f} of cash is nothing to "
                f"{'buy' if wanted == 'long' else 'short'}",
            )
            return False
        order = _Order(
            side=side,
            due=self.index + max(int(self.latency_bars), 0),
            usd=budget,
            stop_pct=max(float(stop_pct), 0.0),
            take_profit_pct=max(float(take_profit_pct), 0.0),
            trail_pct=max(float(trail_pct), 0.0),
            max_bars=max(int(max_bars), 0),
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
        """Close ``amount`` **units** of the position; ``True`` when the order was taken.

        Units and not dollars, because that is what the live provider's ``withdraw`` takes
        out of a vault: the engine sells *the position it holds*, and a sell sized in
        dollars would leave a residue that is an artefact of the price rather than a
        decision. ``amount`` is a magnitude, so the same call closes a long (a sell) and a
        short (a cover) — the caller says how much of the position to be done with, and the
        venue is the one that knows which side it is holding.
        """
        side = "sell" if self.long else "cover"
        if not self.holding:
            self._refuse("sell", float(amount), "there is no position to close")
            return False
        if self.busy:
            self._refuse(side, float(amount), "an order is already waiting for its bar")
            return False
        units = min(abs(float(amount)), abs(float(self.units)))
        if units <= 0.0:
            self._refuse(side, float(amount), "nothing to sell")
            return False
        order = _Order(
            side=side,
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
        """Fill every order whose bar has arrived, at that bar's open.

        A sell whose position is already gone — the stop fired inside this bar before the
        queued exit settled — is dropped without a refusal: it is not a mistake by the
        caller, it is a race the venue won, and a ledger of phantom refusals would be noise
        pretending to be a reading.
        """
        if not self.orders:
            return
        due = [order for order in self.orders if order.due <= self.index]
        for order in due:
            self.orders.remove(order)
            if order.side == "sell" and not self.long:
                continue
            if order.side == "cover" and not self.short:
                continue
            self._execute(order)

    def _execute(self, order: _Order) -> None:
        """The one path both sides take, so neither can skip the price it was given."""
        bar = self.bar
        if bar is None:
            return
        opening = float(bar.open) if bar.open > 0.0 else float(bar.close)
        if order.side == "buy":
            self._buy(order.usd, opening * (1.0 + float(self.slippage)), order)
        elif order.side == "short":
            # Borrowed units are sold at the bid, *below* the mid: the entry pays the same
            # slippage a long pays, in the direction the position is pointing.
            self._open_short(order.usd, opening * (1.0 - float(self.slippage)), order)
        elif order.side == "cover":
            # And they are bought back at the ask, *above* the mid, again against the position.
            self._cover(order.units, opening * (1.0 + float(self.slippage)), order)
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
        self.take_profit_price = (
            float(price) * (1.0 + float(order.take_profit_pct))
            if float(order.take_profit_pct) > 0.0
            else 0.0
        )
        self.trail_pct = max(float(order.trail_pct), 0.0)
        self.peak_price = float(price)
        self.max_bars = max(int(order.max_bars), 0)
        self.mfe = 0.0
        self.mae = 0.0
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
            self._flat()
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

    def _open_short(self, usd: float, price: float, order: _Order) -> None:
        """Sell borrowed units: the position goes negative and the proceeds sit in cash.

        The arithmetic is the long's, turned around. A long spends ``budget`` and receives
        ``(budget - fee) / price`` units; a short borrows ``budget / price`` units, sells them
        for ``budget``, pays the fee out of the proceeds and owes the units back — so equity at
        the fill is ``cash - fee``, exactly what a long pays to open. The stop is placed
        *above* the fill and the target *below* it: the two prices have swapped sides, and the
        rest of this file reads them by position rather than by direction.
        """
        budget = self._budget(usd)
        if budget <= 0.0 or price <= 0.0:
            self._refuse("short", usd, "the account had nothing left to short when it opened")
            return
        charged = budget * float(self.fee)
        units = budget / float(price)
        if units <= 0.0:
            self._refuse("short", usd, "the fee consumed the whole order at this price")
            return
        self.cash = float(self.cash) + budget - charged
        self.units = -float(units)
        self.entry_price = float(price)
        self.entry_bar = int(self.index)
        self.stop_price = (
            float(price) * (1.0 + float(order.stop_pct)) if float(order.stop_pct) > 0.0 else 0.0
        )
        self.take_profit_price = (
            float(price) * (1.0 - float(order.take_profit_pct))
            if float(order.take_profit_pct) > 0.0
            else 0.0
        )
        self.trail_pct = max(float(order.trail_pct), 0.0)
        # The extreme the trail follows: the highest high for a long, the lowest low for a
        # short. One field, because a ratchet only ever needs the worst price already seen.
        self.peak_price = float(price)
        self.max_bars = max(int(order.max_bars), 0)
        self.mfe = 0.0
        self.mae = 0.0
        self.fills.append(
            Fill(
                bar=int(self.index),
                at=self.at,
                side="short",
                price=float(price),
                units=float(units),
                # The proceeds, net of the fee — what the entry put in the account, and the
                # number a trade's profit is measured against.
                usd=float(budget - charged),
                fee=float(charged),
                reason=order.reason,
            )
        )

    def _cover(self, units: float, price: float, order: _Order) -> None:
        """Buy the borrowed units back, paying the ask and the fee out of the collateral."""
        held = min(abs(float(units)), abs(float(self.units)))
        if held <= 0.0 or price <= 0.0:
            self._refuse(
                "cover", float(units), "there was no short left to cover when the bar opened"
            )
            return
        gross = held * float(price)
        charged = gross * float(self.fee)
        self.cash = float(self.cash) - gross - charged
        self.units = float(self.units) + held
        if abs(float(self.units)) <= 1.0e-12:
            self._flat()
        self.fills.append(
            Fill(
                bar=int(self.index),
                at=self.at,
                side="cover",
                price=float(price),
                units=float(held),
                # What the cover cost, fee included: a short's profit is the entry's proceeds
                # minus this number, with no sign to remember.
                usd=float(gross + charged),
                fee=float(charged),
                reason=order.reason,
            )
        )

    def _flat(self) -> None:
        """The ledger with no position in it: every field a position owns forgets it.

        One place, called by both sides, so that a long and a short cannot leave a different
        residue behind — a stop price outliving the position is the kind of ghost that would
        fire on the next one.
        """
        self.units = 0.0
        self.entry_price = 0.0
        self.entry_bar = -1
        self.stop_price = 0.0
        self.take_profit_price = 0.0
        self.trail_pct = 0.0
        self.peak_price = 0.0
        self.max_bars = 0
        self.last_mfe = float(self.mfe)
        self.last_mae = float(self.mae)

    def _stop(self) -> None:
        """The resting stop, executed inside the bar that crosses it.

        Two prices matter and they are not the same: the stop price the position named, and
        the price the bar actually opened at. When the bar opened *below* the stop the order
        fills at the open — the gap is real, and a backtest that filled it at the stop would
        be inventing the difference — and otherwise it fills at the stop itself.

        A short's stop sits *above* the fill, so the same rule runs with the comparison turned
        around: a bar that gapped up through it fills at its open, which is worse than the
        stop, and the money the gap took is in the ledger rather than in a footnote.
        """
        if not self.holding or self.stop_price <= 0.0:
            return
        bar = self.bar
        if bar is None:
            return
        crossing = float(self.stop_price)
        opening = float(bar.open) if bar.open > 0.0 else crossing
        if self.short:
            if float(bar.high) < crossing:
                return
            price = opening if opening > crossing else crossing
            self._cover(
                float(self.units),
                price * (1.0 + float(self.slippage)),
                _Order(side="cover", due=int(self.index), reason="stop"),
            )
            return
        if float(bar.low) > crossing:
            return
        price = opening if opening < crossing else crossing
        self._sell(
            float(self.units),
            price * (1.0 - float(self.slippage)),
            _Order(side="sell", due=int(self.index), reason="stop"),
        )

    def _take_profit(self) -> None:
        """The resting take-profit, executed inside the bar that crosses it.

        A bar that gapped *above* the target fills at the open, not at the target: the gap is
        real money and a backtest that kept it would be inventing profit. A bar that merely
        traded through the target fills at the target. For a short the target is a floor, and
        the same sentence holds with the sign of the move turned around.
        """
        if not self.holding or self.take_profit_price <= 0.0:
            return
        bar = self.bar
        if bar is None:
            return
        target = float(self.take_profit_price)
        opening = float(bar.open) if bar.open > 0.0 else target
        if self.short:
            if float(bar.low) > target:
                return
            price = opening if opening < target else target
            self._cover(
                float(self.units),
                price * (1.0 + float(self.slippage)),
                _Order(side="cover", due=int(self.index), reason="target"),
            )
            return
        if float(bar.high) < target:
            return
        price = opening if opening > target else target
        self._sell(
            float(self.units),
            price * (1.0 - float(self.slippage)),
            _Order(side="sell", due=int(self.index), reason="target"),
        )

    def _trail(self) -> None:
        """Ratchet the stop behind the move, and never back.

        The trail is a *fraction of the fill* rather than a fraction of the ATR, because the
        venue does not know what an ATR is: the engine that wanted an ATR trail multiplies
        before it hands the number over, which keeps this file free of market statistics.

        The extreme it follows is the highest high for a long and the lowest low for a short —
        :attr:`peak_price` is whichever of the two the position is watching — and the stop only
        ever moves in the direction that protects what is already on the table.
        """
        if not self.holding or self.trail_pct <= 0.0:
            return
        bar = self.bar
        if bar is None:
            return
        if self.short:
            low = float(bar.low)
            if low < float(self.peak_price):
                self.peak_price = low
            candidate = float(self.peak_price) * (1.0 + float(self.trail_pct))
            current = float(self.stop_price)
            if (current <= 0.0 or candidate < current) and candidate > float(bar.close):
                self.stop_price = candidate
            return
        high = float(bar.high)
        if high > float(self.peak_price):
            self.peak_price = high
        candidate = float(self.peak_price) * (1.0 - float(self.trail_pct))
        if candidate > float(self.stop_price) and candidate < float(bar.close) * 1.0:
            self.stop_price = candidate

    def _time_stop(self) -> None:
        """Queue the exit once the position has lived ``max_bars`` bars.

        The order rides to the *next* open like every other decision in this venue: a rule
        that sold at a close the caller was allowed to see would be a rule that filled at a
        price nobody could have traded. Firing the order here and filling it there is the
        same delay an operator's own automation would impose, and the report counts the
        holding time on the fills, so the extra bar is visible.
        """
        if not self.holding or self.max_bars <= 0:
            return
        if int(self.index) - int(self.entry_bar) < int(self.max_bars):
            return
        if self.busy:
            return
        bar = self.bar
        if bar is None:
            return
        self.orders.append(
            _Order(
                side="cover" if self.short else "sell",
                due=int(self.index) + max(int(self.latency_bars), 1),
                units=float(self.units),
                reason="time",
            )
        )

    def _mark(self) -> None:
        """Record the excursions: the best and worst the position ever saw.

        ``mfe`` and ``mae`` are in fractions of the fill and are what a *stop placed after
        the fact* would have captured. They are the difference between "the trade lost" and
        "the trade was a winner that the stop cut at the wrong height", which is the whole
        argument for an ATR stop.
        """
        if not self.holding or self.entry_price <= 0.0:
            return
        bar = self.bar
        if bar is None:
            return
        entry = float(self.entry_price)
        if self.short:
            # The best a short saw is the low, the worst is the high, and both are measured
            # against the fill — the same denominator a long's excursions are measured in.
            self.mfe = max(float(self.mfe), 1.0 - float(bar.low) / entry)
            self.mae = min(float(self.mae), 1.0 - float(bar.high) / entry)
            return
        self.mfe = max(float(self.mfe), float(bar.high) / entry - 1.0)
        self.mae = min(float(self.mae), float(bar.low) / entry - 1.0)
