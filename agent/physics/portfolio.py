"""The portfolio: three buckets, a cap on each, and the arithmetic of survival.

The worker earns one contract at a time. This file is the next question a body with a
balance has to answer: **what do I do with the money between jobs?** The answer is not
"put it where the yield is highest", because the highest yield in this market is offered
by the protocols that are lying. It is a *cap*: a fifth of the empire may look at a
farm, and everything else must be somewhere a curve can be believed.

The three buckets
=================

``safe``          60 % — a staking vault whose rate moves with utilisation. Dull, and
                  measured: the record carries market noise,``cv ≈ 3``, ``A² ≈ 2.5``, so
                  the assessment below can tell it from a formula.
``active``        30 % — bounty boards and the MEV router: bursty payouts, fat tails,
                  ``cv ≈ 2.8``, ``A² ≈ 220``. This is where the anomaly work from the
                  previous phase meets a wallet.
``experimental``  10 % — farms that claim triple-digit APY. Some of them are real and
                  pay; the ones that are lying are why the bucket is one tenth.

The split is not decoration and it is not diversification theatre. It is the answer to
"what is the worst a single mistake can do": at these targets, a protocol that goes to
zero costs a *tenth* of the empire, and one that is caught by the assessment costs
nothing at all. That is a cruder rule than ``worker.py``'s decision, and it is
deliberately cruder — a cap has to work on the days the measurement is wrong.

Why the whitelist is necessary and not sufficient
=================================================

:class:`WhiteList` is the operator's list of approved addresses, and no capital
leaves the portfolio for an address that is not on it (:class:`NotWhitelisted` is an
error, not a refusal — a refusal is a decision and this is a bug). But an address being
approved says only that *someone checked who runs it*, not that its curve is a record of
anything. Every scam in this simulation is whitelisted: that is the point. The whitelist
stops the portfolio from being *drained*; it does not stop it from being *lied to*, and
the lying is what ``risk_assessment.py`` is for.

What is simulated, and what a real adapter would need
=====================================================

:meth:`ProtocolMarket.mock` builds the market: it *generates* the return of every
protocol for every step, including the two rug pulls and the crash, and hands the
portfolio only the part of each curve that has already happened
(:meth:`Protocol.record`). The agent cannot see the future because the object it is
given does not contain it — the same rule the contract market plays by.

Nothing here has a key, a signature, an RPC endpoint or an address that exists. A
protocol address is ``0x`` plus the first forty hex digits of a SHA-256 of its name
(:func:`sim_address`), which makes it stable, unique, checkable, and provably not a
wallet. The seam for a real adapter is :class:`ProtocolFeed`: ``history(address)``
returns the record, ``quote(address)`` the claim and ``balance(address)`` the position,
and an implementation over Web3 would need an RPC endpoint, an ABI, and a signer — three
things this file must never have, because a simulation that can move real money is not a
simulation. Swapping in such an adapter changes
:class:`ProtocolMarket` and nothing else: the portfolio, the assessment and the kill
switch are written against the record and the claim, and neither of those cares where
the numbers came from.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

#: The three buckets, in the order a report prints them: safest first.
BUCKET_SAFE = "safe"
BUCKET_ACTIVE = "active"
BUCKET_EXPERIMENTAL = "experimental"
BUCKETS = (BUCKET_SAFE, BUCKET_ACTIVE, BUCKET_EXPERIMENTAL)

#: What share of the empire each bucket is *supposed* to hold. Declared, not
#: derived, and the whole risk budget of the file is in the third number: at ten
#: percent, a protocol that goes to zero costs a tenth of the empire.
TARGETS = {BUCKET_SAFE: 0.60, BUCKET_ACTIVE: 0.30, BUCKET_EXPERIMENTAL: 0.10}

#: Credits the empire starts with. Small on purpose: the first opportunity has to be
#: found, not merely afforded.
START_CAPITAL = 100.0

#: Steps in a "cycle": the period over which the kill switch measures a drawdown,
#: and the period at which the empire re-reads its market. Twenty-four, because the
#: brief's fifteen percent is *per cycle* and a cycle has to be long enough to hold a
#: decision and short enough to be a risk — a day, in this simulation's calendar.
CYCLE = 24

#: Steps a year. The only place a per-step rate becomes an APY, and the conversion
#: is declared here so a claim of "1000 % APY" can be turned into a per-step number
#: by arithmetic rather than by trust.
STEPS_PER_YEAR = 365

#: Samples a record needs before anybody is allowed to have an opinion about it.
#: The same floor the sensorium uses (:data:`agent.physics.sensorium.MIN_SAMPLES`):
#: below it there is nothing to judge, and the honest answer is "not yet".
MIN_RECORD = 64

#: Excess over a bucket's target, as a share of equity, worth selling for. Below it
#: the trim is not worth the trade and is skipped; without a band, a bucket that
#: drifts by a credit is trimmed every cycle forever.
TRIM_BAND = 0.02

#: The columns of a life, in the order they are recorded: what the empire was worth,
#: what was in cash, and what was standing in each bucket.
TRACE_FIELDS = ("step", "equity", "cash", "safe", "active", "experimental")


class PortfolioError(RuntimeError):
    """The portfolio was asked to do something an accounting cannot do."""


class NotWhitelisted(PortfolioError):
    """Capital was pointed at an address the operator has not approved."""


class DeadProtocol(PortfolioError):
    """A protocol that has stopped paying is not an investment."""


# ── addresses, rates and the conversion between them ─────────────────────────


def sim_address(name: str) -> str:
    """A stable, unique, *provably simulated* address: ``0x`` + SHA-256 of the name.

    Not a checksummed address and not meant to be one: forty hex digits of a hash of
    ``"enrg-sim:<name>"``. Two runs of the demo fund the same address, a test can assert
    on it, and nobody can mistake it for a wallet — which is the only property a
    simulated address has to have.
    """
    digest = hashlib.sha256(f"enrg-sim:{name}".encode("utf-8")).hexdigest()
    return "0x" + digest[:40]


def per_step_rate(apy: float, steps: int = STEPS_PER_YEAR) -> float:
    """Turn a quoted APY into the per-step rate that compounds to it."""
    growth = 1.0 + float(apy)
    if growth <= 0.0:
        raise PortfolioError(f"an APY of {apy!r} is not a growth rate")
    return float(growth ** (1.0 / float(steps)) - 1.0)


def annualise(rate: float, steps: int = STEPS_PER_YEAR) -> float:
    """Turn a per-step rate into the APY it compounds to — the other direction."""
    return float((1.0 + float(rate)) ** float(steps) - 1.0)


def implied_apy(returns: Sequence[float], steps: int = STEPS_PER_YEAR) -> float:
    """The APY the *record* implies, compounded at the record's own length.

    ``(Π(1 + r))^(steps/n) − 1``: the same conversion as :func:`per_step_rate`, run
    backwards over what was actually paid. A record that lost everything has a growth
    factor of zero and this returns ``-1.0`` — "total loss", not a small number.
    """
    values = np.asarray(returns, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    growth = float(np.prod(np.clip(1.0 + values, 0.0, None)))
    if growth <= 0.0:
        return -1.0
    return float(growth ** (float(steps) / float(values.size)) - 1.0)


# ── the curves: what the market pays, generated and declared ─────────────────
#
# Each generator is one kind of protocol, and the docstring states the signature the
# assessment measures, so the market and the detector can be checked against each
# other: a curve is honest if it carries the noise a market carries.
#
# The per-step rates are named here because a *claim* is built out of them
# (:func:`annualise`): the honest protocols claim exactly the rate they pay, so the
# only lie in the market is the one in the traps.

#: The declared per-step rate of each honest kind, and of the formula trap.
RATE_STAKING = 1.8e-4
RATE_BOND = 5.0e-4
RATE_BOUNTY = 2.8e-3
RATE_MEV = 1.83e-3
RATE_FARM = 4.13e-3
RATE_TRAP = 6.6e-3


def staking_returns(
    rng: np.random.Generator, steps: int, *, rate: float = RATE_STAKING, cv: float = 3.0
) -> np.ndarray:
    """A variable staking rate: the record carries the market's own noise.

    A student-t of six degrees of freedom at ``cv`` times the mean, which is a fat tail
    and not a Gaussian: real yield has bad days. Measured through the body:
    ``cv = 3.0``, ``A² = 2.5`` — two orders of magnitude above the sterility floor,
    which is exactly what a lived record has to be.
    """
    noise = rng.standard_t(6, size=int(steps)) / math.sqrt(6.0 / 4.0)
    return float(rate) + float(rate) * float(cv) * noise


def bond_returns(
    rng: np.random.Generator, steps: int, *, rate: float = RATE_BOND
) -> np.ndarray:
    """A contractual fixed rate — sterile by construction, and honest about it.

    ``cv = 1e-4`` and ``A²`` below the floor: this record *is* a formula, because a bond
    promise is one. It is here to be refused: the assessment cannot tell a bond from a
    Ponzi (both are sterile), so the rule refuses both, and the notebook prices what
    that caution costs in forgone yield.
    """
    return float(rate) * (1.0 + 1e-4 * rng.normal(size=int(steps)))


def bounty_returns(
    rng: np.random.Generator, steps: int, *, rate: float = RATE_BOUNTY
) -> np.ndarray:
    """Bounty work: a quiet retainer, jackpots when a bug is found, dry spells."""
    n = int(steps)
    paid = rng.normal(float(rate) * 0.15, float(rate) * 0.15, size=n)
    jackpot = rng.random(n) < 0.10
    paid[jackpot] += rng.lognormal(np.log(float(rate) * 7.0), 0.6, int(jackpot.sum()))
    dry = rng.random(n) < 0.06
    paid[dry] -= float(rate) * 2.5
    return paid


def mev_returns(
    rng: np.random.Generator, steps: int, *, rate: float = RATE_MEV
) -> np.ndarray:
    """The router: a fee stream with a fee tail. Between safe and farm."""
    return float(rate) * (1.0 + 1.6 * rng.normal(size=int(steps))) * np.exp(
        0.35 * rng.normal(size=int(steps))
    )


def farm_returns(
    rng: np.random.Generator, steps: int, *, rate: float = RATE_FARM
) -> np.ndarray:
    """An honest degen farm: emissions paid in a token that moves, both ways.

    Honest and *not* recommendable — the record is loud (``cv ≈ 2``), which is the only
    thing keeping it out of the trap column: a lived farm is a lived record, however
    badly it pays.
    """
    n = int(steps)
    paid = rng.normal(float(rate) * 0.4, float(rate) * 1.8, size=n)
    harvest = rng.random(n) < 0.12
    paid[harvest] += rng.lognormal(np.log(float(rate) * 9.0), 0.8, int(harvest.sum()))
    dumped = rng.random(n) < 0.10
    paid[dumped] -= float(rate) * 3.0
    return paid


def formula_returns(
    rng: np.random.Generator, steps: int, *, rate: float = RATE_TRAP
) -> np.ndarray:
    """The Ponzi that pays exactly what it promised: a formula wearing a record.

    ``cv = 1e-4``, ``A² ≈ 0``, no market noise at all — and the claim is a thousand
    percent a year. Sterility alone does not convict it (a bond is sterile too); what
    convicts it is that the sterile curve is *also* an impossible promise.
    """
    return float(rate) * (1.0 + 1e-4 * rng.normal(size=int(steps)))


def replayed_returns(
    rng: np.random.Generator,
    steps: int,
    *,
    block: int = 8,
    rate: float = 5.9e-3,
    lap: float = 2.0e-3,
) -> np.ndarray:
    """A "record" re-serving the same eight numbers, growing a hair each lap.

    The second kind of trap, and the one the *structural* arm catches: measured through
    the body it returns ``R² = 1.00`` at lag 1, ``14.5 nats`` of structure, a mirror
    verdict of ``0.00`` and an ``internal_resonance`` of exactly ``0.0000`` — because
    what it re-serves is a formula, and the mirror has heard one before.
    """
    n = int(steps)
    numbers = float(rate) * (0.25 + 1.5 * rng.random(int(block)))
    base = np.resize(numbers, n)
    lap_index = np.repeat(np.arange(n // int(block) + 1), int(block))[:n]
    return base * (1.0 + float(lap) * lap_index)


# ── the protocol, as the portfolio and the assessment both see it ────────────


@dataclass
class Protocol:
    """One protocol: an address, a claim, and the record it has published so far.

    The three fields that matter to a decision are separated on purpose. ``claim_apy``
    is what the protocol *says*; ``returns`` is what it *paid*; and nothing here says
    which of the two is true. :meth:`record` is the only way to read the second, and it
    cannot be asked for a step that has not happened — the future is not in the object
    the agent holds, which is the same honesty ``economy_sim.Contract`` keeps about the
    answer.
    """

    name: str
    bucket: str
    claim_apy: float
    returns: np.ndarray
    address: str = ""
    alive: bool = True
    rug_step: Optional[int] = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.bucket not in BUCKETS:
            raise PortfolioError(
                f"{self.name}: bucket must be one of {BUCKETS}, got {self.bucket!r}"
            )
        values = np.ascontiguousarray(np.asarray(self.returns, dtype=float).ravel())
        if values.size == 0:
            raise PortfolioError(f"{self.name}: a protocol with no record is not one")
        if not np.all(np.isfinite(values)):
            raise PortfolioError(f"{self.name}: the record contains NaN or infinity")
        object.__setattr__(self, "returns", values)
        address = str(self.address or sim_address(self.name)).lower()
        if len(address) != 42 or not address.startswith("0x"):
            raise PortfolioError(f"{self.name}: {address!r} is not a 20-byte address")
        object.__setattr__(self, "address", address)
        object.__setattr__(self, "claim_apy", float(self.claim_apy))

    # -- what it published ----------------------------------------------- #

    def record(self, step: int) -> np.ndarray:
        """The part of the record that has already happened — never the rest."""
        return self.returns[: max(int(step), 0)]

    def step_return(self, step: int) -> float:
        """What one credit earned at ``step``; zero once the protocol is dead."""
        index = int(step)
        if index < 0 or index >= self.returns.size:
            return 0.0
        return float(self.returns[index])

    @property
    def implied_apy(self) -> float:
        """The APY the whole record implies (``portfolio.implied_apy``)."""
        return implied_apy(self.returns)

    def as_opportunity(self, step: int) -> Dict[str, Any]:
        """The published opportunity, as a mapping — what an adapter would hand over."""
        alive = bool(self.alive) and not (
            self.rug_step is not None and int(step) > int(self.rug_step)
        )
        return {
            "address": self.address,
            "name": self.name,
            "bucket": self.bucket,
            "claim_apy": float(self.claim_apy),
            "returns": [float(value) for value in self.record(step)],
            "alive": alive,
        }

    def summary(self, step: Optional[int] = None) -> Dict[str, Any]:
        """The protocol as a table row. With ``step``, only what had happened."""
        published = self.returns if step is None else self.record(int(step))
        return {
            "name": self.name,
            "address": self.address,
            "bucket": self.bucket,
            "claim_apy": round(float(self.claim_apy), 6),
            "implied_apy": round(float(implied_apy(published)), 6),
            "steps": int(published.size),
            "recorded": int(self.returns.size),
            "alive": bool(self.alive),
            "rug_step": None if self.rug_step is None else int(self.rug_step),
            "note": self.note,
        }


@dataclass(frozen=True)
class WhiteList:
    """The operator's approved addresses, and the only ones capital may reach.

    Fail-closed: a whitelist with nothing on it approves nothing, so a portfolio built
    without one is a portfolio that can hold cash and do nothing else. That is the right
    default — the alternative, an empty list meaning "everything", would make the safety
    property true only when someone remembers it.
    """

    addresses: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cleaned = tuple(sorted({str(item).lower() for item in self.addresses}))
        for item in cleaned:
            if len(item) != 42 or not item.startswith("0x"):
                raise PortfolioError(f"{item!r} is not a 20-byte address")
        object.__setattr__(self, "addresses", cleaned)

    @classmethod
    def of(cls, *protocols: Protocol) -> "WhiteList":
        """The list that approves exactly these protocols."""
        return cls(tuple(protocol.address for protocol in protocols))

    def allows(self, address: str) -> bool:
        return str(address).lower() in self.addresses

    def require(self, address: str) -> str:
        """The address, or :class:`NotWhitelisted`. A bug, never a decision."""
        key = str(address).lower()
        if key not in self.addresses:
            raise NotWhitelisted(f"{key} is not on the whitelist")
        return key

    def __len__(self) -> int:
        return len(self.addresses)

    def __contains__(self, address: object) -> bool:
        return self.allows(str(address))

    def as_list(self) -> List[str]:
        return list(self.addresses)


@dataclass
class Holding:
    """One position: a protocol and the credits standing in it."""

    protocol: Protocol
    credits: float = 0.0

    @property
    def address(self) -> str:
        return self.protocol.address


# ── the portfolio: three buckets, a cap, and an accounting ───────────────────


@dataclass
class Portfolio:
    """Capital, three buckets, and the arithmetic that keeps them in proportion.

    Every number in here is checkable by hand: ``equity = cash + Σ holdings``, a deposit
    moves credits between two of those terms, and a return multiplies one holding. There
    is no fee, no slippage and no gas — deliberately, because a cost that is modelled
    has to be modelled *right*, and a wrong cost would hide a wrong decision. What this
    portfolio does model is the only cost that matters to the argument: a protocol that
    goes to zero, and how much of the empire it was allowed to hold.
    """

    whitelist: WhiteList
    cash: float = START_CAPITAL
    targets: Dict[str, float] = field(default_factory=lambda: dict(TARGETS))
    holdings: Dict[str, Holding] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = float(self.cash)
        missing = [bucket for bucket in BUCKETS if bucket not in self.targets]
        if missing:
            raise PortfolioError(f"targets must name every bucket, missing {missing}")
        total = sum(float(value) for value in self.targets.values())
        if abs(total - 1.0) > 1e-9:
            raise PortfolioError(f"targets must sum to one, they sum to {total!r}")

    # -- what it is worth ------------------------------------------------- #

    @property
    def equity(self) -> float:
        """Cash plus every position: the number the drawdown is measured on."""
        return float(self.cash + sum(item.credits for item in self.holdings.values()))

    def value(self, address: str) -> float:
        """Credits standing at one address; zero when there is no position."""
        holding = self.holdings.get(str(address).lower())
        return float(holding.credits) if holding else 0.0

    def bucket_value(self, bucket: str) -> float:
        return float(
            sum(
                item.credits
                for item in self.holdings.values()
                if item.protocol.bucket == bucket
            )
        )

    def weights(self) -> Dict[str, float]:
        """Each bucket's share of equity, plus the cash that is in none of them."""
        total = max(self.equity, 1e-12)
        shares = {bucket: self.bucket_value(bucket) / total for bucket in BUCKETS}
        shares["cash"] = float(self.cash / total)
        return shares

    def addresses(self) -> List[str]:
        return sorted(self.holdings)

    # -- moving credits --------------------------------------------------- #

    def deposit(self, protocol: Protocol, amount: float, *, step: int = 0) -> float:
        """Stand ``amount`` credits in ``protocol``; returns what was placed.

        Three refusals, and all three are errors rather than decisions: an address the
        operator has not approved (:class:`NotWhitelisted`), a protocol that has stopped
        paying (:class:`DeadProtocol`), and a non-positive amount. A short balance is
        not an error — it clamps, because a top-up that cannot be filled should fill
        what it can instead of failing a cycle.
        """
        self.whitelist.require(protocol.address)
        if not protocol.alive:
            raise DeadProtocol(f"{protocol.name} has stopped paying")
        want = float(amount)
        if want <= 0.0:
            return 0.0
        placed = min(want, max(self.cash, 0.0))
        if placed <= 0.0:
            return 0.0
        holding = self.holdings.get(protocol.address)
        if holding is None:
            holding = Holding(protocol=protocol)
            self.holdings[protocol.address] = holding
        holding.credits = float(holding.credits) + placed
        self.cash -= placed
        self._log(step, "deposit", protocol, placed)
        return float(placed)

    def withdraw(
        self, address: str, amount: float, *, step: int = 0, reason: str = ""
    ) -> float:
        """Take credits out of one position, back to cash; returns what came out."""
        key = str(address).lower()
        holding = self.holdings.get(key)
        if holding is None or float(amount) <= 0.0:
            return 0.0
        taken = min(float(amount), float(holding.credits))
        holding.credits = float(holding.credits) - taken
        self.cash += taken
        self._log(step, "withdraw", holding.protocol, taken, reason=reason)
        if holding.credits <= 1e-9:
            self._close(key, step, "closed", taken=0.0)
        return float(taken)

    def liquidate(
        self,
        buckets: Sequence[str] = BUCKETS,
        *,
        step: int = 0,
        reason: str = "liquidated",
    ) -> float:
        """Sell everything in these buckets back to cash; returns the credits raised."""
        wanted = set(buckets)
        raised = 0.0
        for key in list(self.holdings):
            holding = self.holdings[key]
            if holding.protocol.bucket not in wanted:
                continue
            raised += float(holding.credits)
            self.cash += float(holding.credits)
            self._close(key, step, reason, taken=float(holding.credits))
        return float(raised)

    def rebalance(
        self, plan: Mapping[str, Optional[Protocol]], *, step: int = 0
    ) -> Dict[str, float]:
        """Bring each bucket to its target, with ``plan`` naming where it may stand.

        ``plan[bucket] is None`` means *nothing in this bucket passed the assessment*,
        and the bucket goes to cash — that is the whole answer to a market full of
        traps: an empty bucket is a position. Buckets are filled safest first, so a
        portfolio short of cash buys safety before it buys yield.

        The trim band is why a bucket that is a little over its target is left alone: at
        ten percent of equity, a trim of less than :data:`TRIM_BAND` of the whole empire
        costs a trade to move nothing.
        """
        placed: Dict[str, float] = {}
        for bucket in BUCKETS:
            chosen = plan.get(bucket)
            target = float(self.targets[bucket]) * self.equity
            for key in list(self.holdings):
                holding = self.holdings[key]
                if holding.protocol.bucket != bucket:
                    continue
                if chosen is None or holding.protocol.address != chosen.address:
                    self.liquidate([bucket], step=step, reason="rebalanced out")
                    break
            if chosen is None:
                placed[bucket] = 0.0
                continue
            current = self.value(chosen.address)
            gap = target - current
            if gap > 0.0:
                placed[bucket] = self.deposit(chosen, gap, step=step)
            elif gap < -float(TRIM_BAND) * max(self.equity, 1e-12):
                placed[bucket] = -self.withdraw(chosen.address, -gap, step=step)
            else:
                placed[bucket] = 0.0
        return placed

    def accrue(self, step: int) -> float:
        """Apply one step of every protocol's own return; returns the new equity.

        A return of ``-1`` is not a bad day, it is the end of the position: the capital
        is written off and the holding is closed. That case exists because the market
        contains a rug pull, and a simulation that cannot lose *everything* in one step
        would flatter every rule that stays in.
        """
        for key in list(self.holdings):
            holding = self.holdings[key]
            rate = holding.protocol.step_return(step)
            if rate <= -1.0 + 1e-12:
                self._log(step, "loss", holding.protocol, float(holding.credits))
                self._close(key, step, "wiped out", taken=-float(holding.credits))
                continue
            holding.credits = float(holding.credits) * (1.0 + float(rate))
            if not holding.protocol.alive:
                # A dead protocol is not a position even when its last number was
                # positive: the market marks it dead and the capital leaves with it.
                self._close(key, step, "dead protocol", taken=0.0)
        return self.equity

    # -- the trace -------------------------------------------------------- #

    def record(self, step: int) -> Dict[str, Any]:
        """One row of the life: what the empire was worth at ``step``, by bucket."""
        row = {
            "step": float(int(step)),
            "equity": float(self.equity),
            "cash": float(self.cash),
            "safe": self.bucket_value(BUCKET_SAFE),
            "active": self.bucket_value(BUCKET_ACTIVE),
            "experimental": self.bucket_value(BUCKET_EXPERIMENTAL),
        }
        self.trace.append(row)
        return row

    def history(self) -> Dict[str, np.ndarray]:
        """The trace as arrays, one row per recorded step."""
        names = TRACE_FIELDS
        if not self.trace:
            return {name: np.zeros(0) for name in names}
        return {
            name: np.array([row[name] for row in self.trace], dtype=float)
            for name in names
        }

    def _close(self, key: str, step: int, reason: str, *, taken: float) -> None:
        holding = self.holdings.pop(str(key).lower(), None)
        if holding is not None:
            self._log(step, "close", holding.protocol, taken, reason=reason)

    def _log(
        self,
        step: int,
        action: str,
        protocol: Protocol,
        amount: float,
        *,
        reason: str = "",
    ) -> None:
        self.events.append(
            {
                "step": int(step),
                "action": action,
                "protocol": protocol.name,
                "address": protocol.address,
                "amount": round(float(amount), 6),
                "reason": reason,
            }
        )

    def report(self) -> Dict[str, Any]:
        """The wallet, bucket by bucket, with the weights it ended the run at."""
        return {
            "cash": round(float(self.cash), 6),
            "equity": round(float(self.equity), 6),
            "targets": {key: float(value) for key, value in self.targets.items()},
            "buckets": {
                bucket: round(self.bucket_value(bucket), 6) for bucket in BUCKETS
            },
            "weights": {key: round(value, 6) for key, value in self.weights().items()},
            "positions": [
                {
                    "protocol": item.protocol.name,
                    "address": item.protocol.address,
                    "bucket": item.protocol.bucket,
                    "credits": round(float(item.credits), 6),
                }
                for item in sorted(
                    self.holdings.values(), key=lambda row: row.protocol.address
                )
            ],
            "events": len(self.events),
        }


# ── the venue: a market of protocols, and a calendar with a bad year in it ───

#: The scenario, declared: ``(start, stop, mean, sigma)`` of a regime that pays a
#: different rate than the one before it. The first window is a grind — a slow bleed
#: that a cycle drawdown can catch; the second is a spike, which is what a crash looks
#: like when it arrives — and the kill switch has an arm for each.
CRASH_BLEED = (384, 428, -2.4e-2, 6.0e-3)
CRASH_SPIKE = (470, 492, -7.5e-2, 3.0e-2)

#: The steps at which the two traps stop paying. Both are after the empire has funded
#: its experimental bucket, so both are a real loss for whoever was standing there.
TRAP_RUGS = {"YieldMax": 300, "AurumDAO": 350}


def _apply_regime(
    returns: np.ndarray, window: Tuple[int, int, float, float]
) -> np.ndarray:
    """Rewrite one window of a record: the market changed its mind about the rate."""
    start, stop, mean, sigma = window
    rng = np.random.default_rng(abs(int(start)) * 1000 + abs(int(stop)))
    out = np.array(returns, dtype=float, copy=True)
    stop = min(int(stop), out.size)
    if int(start) < stop:
        span = stop - int(start)
        out[int(start) : stop] = rng.normal(float(mean), float(sigma), span)
    return out


@dataclass
class ProtocolMarket:
    """A market of protocols, a whitelist, and a clock that only goes forward.

    :meth:`advance` moves the clock; nothing else in the class hands out a number from
    the future. The two rug pulls and the crash are *baked into the records at
    generation time* and only become visible when their step arrives — which is what
    makes the demo a measurement rather than a script: the agent's assessment on step
    ``t`` sees exactly ``returns[:t]``, and the reader can check that it saw nothing
    else.
    """

    protocols: List[Protocol]
    whitelist: WhiteList
    step: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    # -- the declared market ---------------------------------------------- #

    @classmethod
    def mock(
        cls,
        *,
        steps: int = 720,
        seed: int = 7,
        crash_bleed: Tuple[int, int, float, float] = CRASH_BLEED,
        crash_spike: Tuple[int, int, float, float] = CRASH_SPIKE,
        rugs: Mapping[str, int] = TRAP_RUGS,
    ) -> "ProtocolMarket":
        """Build the market the demo trades: two of each honest kind and two traps.

        Every number a protocol claims is the rate its own generator was given, so the
        claim is honest for the honest protocols and the *only* lie in the market is the
        two traps' — one of which promises a thousand percent and pays it exactly until
        it stops, and one of which re-serves eight numbers and claims forty-two hundred.
        """
        n = int(steps)
        safe = np.random.default_rng(seed + 1)
        active = np.random.default_rng(seed + 2)
        wild = np.random.default_rng(seed + 3)

        candidates = [
            Protocol(
                name="StakingVault",
                bucket=BUCKET_SAFE,
                claim_apy=annualise(RATE_STAKING),
                returns=staking_returns(safe, n),
                note="utilisation-based staking: the record carries market noise",
            ),
            Protocol(
                name="FixedBond",
                bucket=BUCKET_SAFE,
                claim_apy=annualise(RATE_BOND),
                returns=bond_returns(safe, n),
                note="a contractual fixed rate: sterile by construction",
            ),
            Protocol(
                name="BountyBoard",
                bucket=BUCKET_ACTIVE,
                claim_apy=annualise(RATE_BOUNTY),
                returns=bounty_returns(active, n),
                note="quiet retainer, jackpots when a bug is found, dry spells",
            ),
            Protocol(
                name="MevRouter",
                bucket=BUCKET_ACTIVE,
                claim_apy=annualise(RATE_MEV),
                returns=mev_returns(active, n),
                note="a fee stream with a fee tail",
            ),
            Protocol(
                name="NicheFarm",
                bucket=BUCKET_EXPERIMENTAL,
                claim_apy=annualise(RATE_FARM),
                returns=farm_returns(wild, n),
                note="honest degen farm: emissions in a token that moves",
            ),
            Protocol(
                name="YieldMax",
                bucket=BUCKET_EXPERIMENTAL,
                claim_apy=10.0,
                returns=formula_returns(wild, n),
                note="a thousand percent, paid exactly as promised, until it stops",
            ),
            Protocol(
                name="AurumDAO",
                bucket=BUCKET_EXPERIMENTAL,
                claim_apy=25.0,
                returns=replayed_returns(wild, n),
                note="the same eight numbers every lap, and a claim three times "
                "what they paid",
            ),
        ]
        for protocol in candidates:
            rug = rugs.get(protocol.name)
            if rug is not None and int(rug) < n:
                # A scenario shorter than a rug's date has no rug: the record ends
                # before
                # the drama, which is what a short simulation is.
                protocol.rug_step = int(rug)
                protocol.returns[int(rug)] = -1.0
            # A protocol that has already stopped paying does not trade in the crash: it
            # is not in the market any more, and writing the regime over its record
            # would
            # only blur what the structural arm measured about it.
            dead_before_crash = protocol.rug_step is not None and (
                protocol.rug_step < int(crash_bleed[0])
            )
            if protocol.bucket in (BUCKET_ACTIVE, BUCKET_EXPERIMENTAL):
                if dead_before_crash:
                    continue
                protocol.returns = _apply_regime(protocol.returns, crash_bleed)
                protocol.returns = _apply_regime(protocol.returns, crash_spike)
        return cls(protocols=candidates, whitelist=WhiteList.of(*candidates))

    def __post_init__(self) -> None:
        names = [protocol.name for protocol in self.protocols]
        if len(set(names)) != len(names):
            raise PortfolioError(f"two protocols share a name: {names}")
        addresses = [protocol.address for protocol in self.protocols]
        if len(set(addresses)) != len(addresses):
            raise PortfolioError("two protocols share an address")
        self.step = int(self.step)

    # -- the clock -------------------------------------------------------- #

    def advance(self, steps: int = 1) -> int:
        """Move the clock forward; a protocol whose rug step has passed is dead."""
        for _ in range(max(int(steps), 0)):
            self.step += 1
            for protocol in self.protocols:
                if not protocol.alive:
                    continue
                if protocol.rug_step is not None and self.step > int(protocol.rug_step):
                    protocol.alive = False
                    self.events.append(
                        {
                            "step": int(self.step),
                            "action": "rug",
                            "protocol": protocol.name,
                            "address": protocol.address,
                        }
                    )
        return self.step

    def alive(self, bucket: Optional[str] = None) -> List[Protocol]:
        """Every protocol still paying, in declaration order, optionally one bucket."""
        return [
            protocol
            for protocol in self.protocols
            if protocol.alive and (bucket is None or protocol.bucket == bucket)
        ]

    def protocol(self, name: str) -> Protocol:
        """One protocol by name; raises rather than returning a stranger."""
        for protocol in self.protocols:
            if protocol.name == str(name):
                return protocol
        raise PortfolioError(f"the market has no protocol called {name!r}")

    # -- what the agent may see ------------------------------------------- #

    def opportunities(self) -> List[Dict[str, Any]]:
        """Every protocol's opportunity *at the current step* — and nothing more."""
        return [protocol.as_opportunity(self.step) for protocol in self.protocols]

    def records(self) -> Dict[str, np.ndarray]:
        """Name to published record, for a report or a plot."""
        return {
            protocol.name: protocol.record(self.step) for protocol in self.protocols
        }

    def summary(self) -> Dict[str, Any]:
        """The market as a table: what each protocol claims, and what it has paid."""
        return {
            "step": int(self.step),
            "whitelisted": [protocol.address for protocol in self.protocols],
            "protocols": [protocol.summary(self.step) for protocol in self.protocols],
            "events": list(self.events),
        }


# ── the empire: capital, a market, an assessment and a switch ───────────────

#: Names the two policies that do *not* measure anything, so a figure can label them.
NAIVE_APE = "ape"
NAIVE_SPLIT = "split"


@dataclass
class Empire:
    """The manager: it reads the market, decides where capital may stand, and survives.

    One method decides (:meth:`review`), one method lives (:meth:`advance`), one method
    runs (:meth:`run`). The decision is a *value*: :meth:`review` returns the
    assessments and logs the plan — so the notebook can ask "why is this bucket empty"
    without spending a step, exactly as ``worker.py`` does with
    :class:`~agent.physics.worker.Decision`.

    The division of labour is the argument of the file. The **assessment** says what an
    opportunity *is*; the **cap** says how much of the empire the answer may be wrong
    about; the **switch** says when no answer should be trusted at all. None of the
    three is allowed to do another's job: the switch never chooses a protocol, the
    assessment never sizes a position, and the cap never decides what is honest.
    """

    market: ProtocolMarket
    portfolio: Portfolio
    kill_switch: Any
    oracle: Any = None
    review_every: int = CYCLE
    min_record: int = MIN_RECORD
    step: int = 0
    assessments: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    log: List[Dict[str, Any]] = field(default_factory=list)
    volatility: Optional[float] = None
    resonance: Optional[float] = None

    # -- birth -------------------------------------------------------------- #

    @classmethod
    def launch(
        cls,
        *,
        body: Any = None,
        market: Optional[ProtocolMarket] = None,
        capital: float = START_CAPITAL,
        steps: int = 720,
        seed: int = 7,
        review_every: int = CYCLE,
        kill_switch: Any = None,
    ) -> "Empire":
        """Open for business: a market, a wallet, and (if given) a body's own oracle.

        ``body`` is an :class:`~agent.physics.organism.Organism`. Without one the empire
        still trades — the sterility and claim arms need no body — but the structural
        and mirror arms are blind, and every assessment says so in
        ``details["measured"]``. That is the honest way to run a cheaper simulation: the
        absence of a measurement is recorded, not filled in.
        """
        from agent.physics.risk_assessment import KillSwitch

        if market is None:
            market = ProtocolMarket.mock(steps=int(steps), seed=int(seed))
        oracle = None
        if body is not None:
            from agent.physics.curiosity import Curiosity
            from agent.physics.worker import RealityOracle

            engine = Curiosity(body.mirror.posture, prior=body.mirror.prior)
            oracle = RealityOracle(
                sensorium=body.sensorium,
                mirror=body.mirror,
                engine=engine,
                seed=int(seed),
            )
        portfolio = Portfolio(whitelist=market.whitelist, cash=float(capital))
        return cls(
            market=market,
            portfolio=portfolio,
            kill_switch=kill_switch if kill_switch is not None else KillSwitch(),
            oracle=oracle,
            review_every=int(review_every),
        )

    # -- the decision ------------------------------------------------------- #

    def review(self) -> Dict[str, Any]:
        """Assess the published market, then let capital stand only where it may.

        The plan is one protocol per bucket — the admissible one whose recent record
        paid the most — or ``None``, and ``None`` is a decision, not an omission: a
        bucket with nothing admissible goes to cash. While the switch is engaged the two
        risky buckets are forced to ``None`` whatever the assessment says, because the
        switch's whole purpose is to be the rule that is obeyed on the day the
        measurement is wrong.
        """
        from agent.physics.risk_assessment import review as assess_all

        found = assess_all(self.market, oracle=self.oracle, min_record=self.min_record)
        plan: Dict[str, Optional[Protocol]] = {}
        for bucket in BUCKETS:
            if bucket != BUCKET_SAFE and bool(getattr(self.kill_switch, "engaged")):
                plan[bucket] = None
                continue
            admissible = [
                item
                for item in found.values()
                if item.bucket == bucket and item.admissible
            ]
            if not admissible:
                plan[bucket] = None
                continue
            best = max(admissible, key=lambda item: (item.implied_apy, item.name))
            plan[bucket] = self.market.protocol(best.name)
        placed = self.portfolio.rebalance(plan, step=self.step)
        measured = {name: item.as_dict() for name, item in found.items()}
        self.resonance = self._resonance_of(plan, measured)
        self.assessments[self.step] = measured
        decision = {
            "step": int(self.step),
            "equity": round(float(self.portfolio.equity), 6),
            "engaged": bool(getattr(self.kill_switch, "engaged")),
            "plan": {
                bucket: (plan[bucket].name if plan[bucket] is not None else None)
                for bucket in BUCKETS
            },
            "placed": {
                bucket: round(float(value), 6) for bucket, value in placed.items()
            },
            "verdicts": {name: item.verdict for name, item in found.items()},
            "refused": [
                {"name": name, "verdict": item.verdict, "reason": item.reason}
                for name, item in sorted(found.items())
                if not item.admissible
            ],
            "resonance": None if self.resonance is None else round(self.resonance, 4),
        }
        self.decisions.append(decision)
        self.log.append({"step": int(self.step), "action": "review", **decision})
        return decision

    def _resonance_of(
        self, plan: Mapping[str, Optional[Protocol]], measured: Mapping[str, Any]
    ) -> Optional[float]:
        """The worst ``internal_resonance`` among the records about to be held.

        ``None`` when nothing risky is held or when the body's mirror did not get to
        speak: an unmeasured resonance must not be read as a healthy one, and the switch
        treats ``None`` as "this arm cannot fire" rather than as zero.
        """
        values = [
            float(measured[protocol.name]["resonance"])
            for bucket, protocol in plan.items()
            if bucket != BUCKET_SAFE and protocol is not None
            if measured.get(protocol.name, {}).get("resonance") is not None
        ]
        return float(min(values)) if values else None

    # -- the life ------------------------------------------------------------ #

    def market_volatility(self) -> Optional[float]:
        """Realised volatility of the risky market: the *median* protocol's ratio.

        Measured on the *market*, not on the book: once the switch has sold everything,
        a book-based number would be zero and the release would be granted by arithmetic
        to a market that is still on fire.

        The median rather than the maximum, and that is the whole difference between a
        regime change and a jackpot: a crash moves every protocol at once, so it moves
        the middle one too, while one bounty board landing a payout moves only itself.
        Measured on this market, a single jackpot lifts one protocol's ratio to about
        2.3 and the median stays at 1.0.
        """
        from agent.physics.risk_assessment import volatility_ratio

        ratios = [
            float(volatility_ratio(protocol.record(self.step)))
            for protocol in self.market.protocols
            if protocol.alive
            and protocol.bucket in (BUCKET_ACTIVE, BUCKET_EXPERIMENTAL)
            and protocol.record(self.step).size >= 4
        ]
        return float(np.median(ratios)) if ratios else None

    def equity_ago(self, steps: int = CYCLE) -> float:
        """Equity ``steps`` rows back in the trace; the start when there is no past."""
        index = len(self.trace) - 1 - int(steps)
        if index < 0:
            return float(self.trace[0]["equity"]) if self.trace else self.portfolio.cash
        return float(self.trace[index]["equity"])

    def advance(self, steps: int = 1) -> None:
        """Live ``steps``: accrue, measure, obey the switch, review on the cycle."""
        for _ in range(max(int(steps), 0)):
            step = self.step
            self.portfolio.accrue(step)
            self.volatility = self.market_volatility()
            self._watch(step)
            row = self.portfolio.record(step)
            row["volatility"] = -1.0 if self.volatility is None else self.volatility
            row["engaged"] = float(bool(getattr(self.kill_switch, "engaged")))
            self.trace.append(dict(row))
            self.market.advance()
            self.step = int(self.market.step)
            if (
                self.step >= int(self.min_record)
                and int(self.review_every) > 0
                and self.step % int(self.review_every) == 0
            ):
                self.review()

    def run(self, steps: int = 720) -> List[Dict[str, Any]]:
        """Live ``steps`` steps and return the trace."""
        self.advance(steps)
        return self.trace

    def _watch(self, step: int) -> None:
        """Ask the switch, and act on the answer within the same step."""
        reference = self.equity_ago(CYCLE)
        if bool(getattr(self.kill_switch, "engaged")):
            released = self.kill_switch.release(
                step=step,
                equity=self.portfolio.equity,
                reference=reference,
                volatility=self.volatility,
                resonance=self.resonance,
            )
            if released is not None:
                payload = released.as_dict()
                payload.pop("action", None)
                self.log.append({"step": int(step), "action": "release", **payload})
            return
        trigger = self.kill_switch.check(
            step=step,
            equity=self.portfolio.equity,
            reference=reference,
            volatility=self.volatility,
            resonance=self.resonance,
        )
        if trigger is None:
            return
        raised = self.portfolio.liquidate(
            [BUCKET_ACTIVE, BUCKET_EXPERIMENTAL],
            step=step,
            reason=f"kill switch: {trigger.arm}",
        )
        payload = trigger.as_dict()
        payload.pop("action", None)
        self.log.append(
            {
                "step": int(step),
                "action": "kill",
                "raised": round(float(raised), 6),
                **payload,
            }
        )

    # -- what it can say ------------------------------------------------------ #

    def history(self) -> Dict[str, np.ndarray]:
        """The life as arrays, one row per step — what the figure draws."""
        names = TRACE_FIELDS + ("volatility", "engaged")
        if not self.trace:
            return {name: np.zeros(0) for name in names}
        return {
            name: np.array([row[name] for row in self.trace], dtype=float)
            for name in names
        }

    def report(self) -> Dict[str, Any]:
        """Everything a report prints: market, wallet, switch, verdicts, log."""
        return {
            "step": int(self.step),
            "market": self.market.summary(),
            "portfolio": self.portfolio.report(),
            "kill_switch": self.kill_switch.as_dict(),
            "equity": {
                "start": (
                    round(float(self.trace[0]["equity"]), 6)
                    if self.trace
                    else 0.0
                ),
                "end": round(float(self.portfolio.equity), 6),
            },
            "decisions": list(self.decisions),
            "log": list(self.log),
            "assessments": {
                str(step): verdicts for step, verdicts in self.assessments.items()
            },
        }

    def plot(self, path: str, *, foils=(), dpi: int = 130) -> Any:
        """Draw the four panels the brief asked for, and return the path."""
        from agent.physics.plots import plot_empire

        return plot_empire(self, path, foils=foils, dpi=dpi)

    def summary_text(self) -> str:
        """A short readable account, in the order a person would ask the questions."""
        start = float(self.trace[0]["equity"]) if self.trace else START_CAPITAL
        end = float(self.portfolio.equity)
        triggers = list(getattr(self.kill_switch, "triggers", []))
        counts = _count_verdicts(self.assessments)
        lines = [
            f"life: {self.step} steps, equity {start:.0f} -> {end:.0f} "
            f"({(end / max(start, 1e-9) - 1.0) * 100:+.1f}%)",
            f"wallet: {len(self.portfolio.holdings)} positions, cash "
            f"{self.portfolio.cash:.0f}, targets "
            + ", ".join(f"{bucket} {share:.0%}" for bucket, share in TARGETS.items()),
            f"reviews: {len(self.decisions)}, verdicts "
            + ", ".join(f"{verdict} {count}" for verdict, count in counts.items()),
            "kill switch: "
            f"{'engaged' if getattr(self.kill_switch, 'engaged') else 'idle'}, "
            f"{len(triggers)} events",
        ]
        for event in triggers:
            lines.append(f"  step {event.step:>4}  {event.arm:<10} {event.reason}")
        for bucket in BUCKETS:
            lines.append(
                f"  {bucket:<13} {self.portfolio.bucket_value(bucket):>8.1f} credits "
                f"({self.portfolio.weights()[bucket]:.0%})"
            )
        return "\n".join(lines)


def _count_verdicts(assessments: Mapping[Any, Mapping[str, Any]]) -> Dict[str, int]:
    """How many times each verdict was reached over the whole life."""
    counts: Dict[str, int] = {}
    for verdicts in assessments.values():
        for row in verdicts.values():
            verdict = str(row.get("verdict", ""))
            counts[verdict] = counts.get(verdict, 0) + 1
    return dict(sorted(counts.items()))


# ── the control group: two investors who do not measure anything ─────────────


@dataclass
class NaiveInvestor:
    """The same market, the same caps, one difference: it believes the claim.

    Two policies, both of them real behaviour rather than strawmen:

    ``split``  the same 60/30/10 split, each bucket in whichever protocol *claims*
               the most. This is the honest foil for the cap — it survives, because the
               cap is a crude risk control that works, and it bleeds, because a cap
               cannot tell a farm from a fraud.
    ``ape``    everything into the highest claim in the market, always. This is what
               a thousand percent a year is designed to recruit, and the figure shows
               what it is worth at the moment the yield stops.

    No assessment, no switch, no cash beyond the instant before a trade. Nothing here is
    called "stupid": both are the optimal policies of a person who is only reading the
    advertisement, which is the majority of the market this simulation is about.
    """

    market: ProtocolMarket
    portfolio: Portfolio
    policy: str = NAIVE_SPLIT
    review_every: int = CYCLE
    step: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.policy not in (NAIVE_APE, NAIVE_SPLIT):
            raise PortfolioError(f"unknown policy {self.policy!r}")

    @classmethod
    def launch(
        cls,
        market: ProtocolMarket,
        *,
        policy: str = NAIVE_SPLIT,
        capital: float = START_CAPITAL,
        review_every: int = CYCLE,
    ) -> "NaiveInvestor":
        """The same wallet and the same whitelist; only the book differs."""
        return cls(
            market=market,
            portfolio=Portfolio(whitelist=market.whitelist, cash=float(capital)),
            policy=str(policy),
            review_every=int(review_every),
        )

    def choose(self, bucket: Optional[str] = None) -> Optional[Protocol]:
        """The highest claim still paying — the whole of the policy, in one line."""
        pool = self.market.alive(bucket)
        if not pool:
            return None
        return max(pool, key=lambda protocol: (protocol.claim_apy, protocol.name))

    def review(self) -> None:
        """Put the money where the advertisement is."""
        if self.policy == NAIVE_APE:
            best = self.choose(None)
            self.portfolio.liquidate(BUCKETS, step=self.step, reason="rebalanced out")
            if best is not None:
                self.portfolio.deposit(best, self.portfolio.cash, step=self.step)
            return
        plan = {bucket: self.choose(bucket) for bucket in BUCKETS}
        self.portfolio.rebalance(plan, step=self.step)

    def advance(self, steps: int = 1) -> None:
        """Live ``steps`` steps with no measurement and no switch."""
        for _ in range(max(int(steps), 0)):
            step = self.step
            self.portfolio.accrue(step)
            row = self.portfolio.record(step)
            self.trace.append(dict(row))
            self.market.advance()
            self.step = int(self.market.step)
            if int(self.review_every) > 0 and self.step % int(self.review_every) == 0:
                self.review()

    def run(self, steps: int = 720) -> List[Dict[str, Any]]:
        self.advance(steps)
        return self.trace

    def history(self) -> Dict[str, np.ndarray]:
        if not self.trace:
            return {name: np.zeros(0) for name in TRACE_FIELDS}
        return {
            name: np.array([row[name] for row in self.trace], dtype=float)
            for name in TRACE_FIELDS
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m agent.physics.portfolio --steps 720 --plot empire.png``."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--capital", type=float, default=START_CAPITAL)
    parser.add_argument("--body", action="store_true", help="measure with a real body")
    parser.add_argument(
        "--body-seed",
        type=int,
        default=11,
        help="the body's own seed; 11 is the one the notebook uses",
    )
    parser.add_argument("--output", default="", help="write the report as JSON here")
    parser.add_argument("--plot", default="", help="write the figure here")
    parser.add_argument("--against", action="store_true", help="draw the two foils too")
    args = parser.parse_args(argv)

    body = None
    if args.body:
        from agent.physics.organism import Organism

        body = Organism.hatch(offline=True, steps=120, seed=args.body_seed)
    empire = Empire.launch(
        body=body, steps=args.steps, seed=args.seed, capital=args.capital
    )
    empire.run(args.steps)
    print(empire.summary_text())

    foils = []
    if args.against:
        for policy, label in ((NAIVE_SPLIT, "split"), (NAIVE_APE, "ape")):
            investor = NaiveInvestor.launch(
                ProtocolMarket.mock(steps=args.steps, seed=args.seed), policy=policy
            )
            investor.run(args.steps)
            path = investor.history()["equity"]
            print(f"{label:<6} ends at {path[-1]:8.1f} (peak {path.max():8.1f})")
            foils.append((label, investor))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(empire.report(), handle, indent=2, ensure_ascii=False)
        print(f"report written to {args.output}")
    if args.plot:
        empire.plot(args.plot, foils=foils)
        print(f"figure written to {args.plot}")
    return 0 if empire.portfolio.equity > 0.0 else 1


__all__ = [
    "BUCKETS",
    "BUCKET_ACTIVE",
    "BUCKET_EXPERIMENTAL",
    "BUCKET_SAFE",
    "CRASH_BLEED",
    "CRASH_SPIKE",
    "CYCLE",
    "DeadProtocol",
    "Empire",
    "Holding",
    "MIN_RECORD",
    "NAIVE_APE",
    "NAIVE_SPLIT",
    "NaiveInvestor",
    "NotWhitelisted",
    "Portfolio",
    "PortfolioError",
    "Protocol",
    "ProtocolMarket",
    "START_CAPITAL",
    "STEPS_PER_YEAR",
    "TARGETS",
    "TRACE_FIELDS",
    "TRAP_RUGS",
    "TRIM_BAND",
    "WhiteList",
    "annualise",
    "bond_returns",
    "main",
    "bounty_returns",
    "farm_returns",
    "formula_returns",
    "implied_apy",
    "mev_returns",
    "per_step_rate",
    "replayed_returns",
    "sim_address",
    "staking_returns",
]


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
