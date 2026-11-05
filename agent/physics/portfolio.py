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

The three buckets are the *policy*, and a measurement may re-weight them but not rewrite
them: :meth:`Portfolio.alpha_hunting_weight` turns a fired panic signal
(:mod:`agent.physics.volatility_hunter`) into a multiplier on **its own** bucket, capped at
:data:`ALPHA_BOOST_CAP` (half) and funded only out of cash above :data:`ALPHA_CASH_FLOOR`
(five percent, never spent). :meth:`Portfolio.boosted_targets` is the resulting map and
:meth:`Portfolio.rebalance` runs the *same* arithmetic against it, so a panic gets a bigger
number and not a second code path. A book that has not measured a panic is exactly the book
it was before any of this existed.

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
wallet. The seam for a real adapter is ``interfaces.IWeb3Provider``:
``get_protocol_list()`` says what is on offer, ``fetch_protocol_history(address)``
gives the record, ``submit_validation_proof()`` takes the evidence behind a verdict,
and ``execute_stake()`` / ``withdraw()`` move capital. An implementation over Web3
would need an RPC endpoint, an ABI and a signer — three things this package must never
have, because a simulation that can move real money is not a simulation — and swapping
one in changes *which provider is constructed*, not the portfolio, the assessment or
the kill switch.

That is why :class:`Portfolio` is handed a provider rather than a market: every credit
it moves is a call through the interface
(:class:`~agent.physics.interfaces.MockWeb3Provider` for the declared market,
:class:`~agent.physics.interfaces.NoVenueProvider` for the bare book the tests build),
and the answer that comes back decides whether anything moved at all. This file keeps
no second path for "no venue": the absence of one is itself a provider that says so.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from agent.physics.discovery import EXECUTION_LIMIT, DiscoveryReport, Watchlist

if TYPE_CHECKING:  # pragma: no cover - for the type checker, never imported at runtime
    from agent.physics.interfaces import IWeb3Provider
    from agent.physics.transaction_cost_analyzer import CostReport, TransactionCostAnalyzer

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

# ── the alpha budget: what a measured panic may move ────────────────────────

#: The most a panic signal may widen *its own* bucket, as a share of that bucket's
#: target. Half, and the reason it is a share rather than a number of credits: the
#: question the cap answers is "how much of my policy am I willing to let one measurement
#: rewrite", and a policy that can be rewritten by half by a single reading is still a
#: policy. At the declared targets the widest possible move is ``0.5 × 0.60 = 30 %`` of
#: equity — and it is funded out of cash, so a book with no spare cash moves nothing at all
#: no matter how loud the signal is (:data:`ALPHA_CASH_FLOOR`).
ALPHA_BOOST_CAP = 0.5

#: The share of equity the book never spends, however loud a signal is. A panic is an
#: argument for *using cash*, never for having none: the whole point of the bucket split
#: is that capital which has not been committed is the only capital that can answer a
#: surprise, and the largest surprise in this market is a protocol that stops paying.
ALPHA_CASH_FLOOR = 0.05


class PortfolioError(RuntimeError):
    """The portfolio was asked to do something an accounting cannot do."""


class NotWhitelisted(PortfolioError):
    """Capital was pointed at an address the operator has not approved."""


class DeadProtocol(PortfolioError):
    """A protocol that has stopped paying is not an investment."""


class StakeRefused(PortfolioError):
    """The venue said no to a stake the book had already decided to place.

    An error and never a decision, in the same sense as :class:`NotWhitelisted`: the
    cap and the assessment decide *where* capital goes, and a venue that then declines
    the transfer has broken an invariant rather than made a choice. It is a separate
    class because the fix is different — the whitelist is the operator's list being
    wrong, this is the venue and the book disagreeing about what is allowed, which on a
    real chain means the ABI, the balance or the allowance.
    """


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

    ``depth_usd`` is the one number here that is about *exiting* rather than about
    earning: the pool's published depth in dollars, or ``None`` when the venue publishes
    none. It is what
    :class:`~agent.physics.transaction_cost_analyzer.TransactionCostAnalyzer` prices
    slippage against, and ``None`` is not zero — an unmeasured pool is not a bottomless
    one, and the cost model reports the difference rather than filling it in.
    """

    name: str
    bucket: str
    claim_apy: float
    returns: np.ndarray
    address: str = ""
    alive: bool = True
    rug_step: Optional[int] = None
    turnover: Optional[float] = None
    depth_usd: Optional[float] = None
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
        """The published opportunity, as a mapping — what an adapter would hand over.

        ``turnover`` is ``None`` when the venue does not publish one, and ``None`` is not
        zero: an unmeasured liquidity is unmeasured, and the scanner's rule about volume
        must fire on a *published* zero rather than on an absent field, or the first
        protocol a testnet lists would be refused for the crime of having no history.
        """
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
            "turnover": None if self.turnover is None else float(self.turnover),
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
            "turnover": None if self.turnover is None else float(self.turnover),
            "depth_usd": None if self.depth_usd is None else float(self.depth_usd),
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


# ── reading a signal the book did not compute ───────────────────────────────


def _signal_fired(signal: Any) -> bool:
    """True when a signal says a panic is *measured now*.

    The book does not import the hunter: ``volatility_hunter`` imports this file, and a
    cycle would mean neither could be read alone. So the one fact the book needs is read
    off the value itself — ``fired`` when the hunter's signal provides it, and otherwise
    the verdict by name — and the two are held together by a test rather than by an
    import, exactly as :func:`agent.physics.discovery.measure` is held to
    :func:`agent.physics.sterility.variation`.
    """
    if signal is None:
        return False
    fired = getattr(signal, "fired", None)
    if fired is not None:
        return bool(fired)
    verdict = getattr(signal, "verdict", None)
    if verdict is None and isinstance(signal, Mapping):
        verdict = signal.get("verdict")
    return str(verdict or "") == "PANIC"


def _signal_confidence(signal: Any) -> float:
    """The signal's own 0 … 1 confidence, clamped — never a size the book invents."""
    value = getattr(signal, "confidence", None)
    if value is None and isinstance(signal, Mapping):
        value = signal.get("confidence")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(max(number, 0.0), 1.0)


def _signal_name(signal: Any) -> str:
    """The name a log line prints for a signal, whatever shape it arrived in."""
    for attribute in ("name", "protocol", "address"):
        value = getattr(signal, attribute, None)
        if value:
            return str(value)
    if isinstance(signal, Mapping):
        for key in ("name", "protocol", "address"):
            if signal.get(key):
                return str(signal[key])
    return "unnamed"


def _signal_bucket(signal: Any) -> str:
    """The bucket a signal asks to widen — a name, or nothing the book will honour.

    Read off the value rather than imported for the reason :func:`_signal_fired` gives, and
    an unknown bucket is *ignored* rather than raised: a measurement may name a bucket this
    book does not keep, and a strategy that crashed on one would be a strategy that cannot
    be pointed at a second venue.
    """
    value = getattr(signal, "bucket", None)
    if not value and isinstance(signal, Mapping):
        value = signal.get("bucket")
    return str(value or "").strip().lower()


# ── the portfolio: three buckets, a cap, and an accounting ───────────────────


@dataclass
class Portfolio:
    """Capital, three buckets, and the arithmetic that keeps them in proportion.

    Every number in here is checkable by hand: ``equity = cash + Σ holdings``, a deposit
    moves credits between two of those terms, and a return multiplies one holding. The
    book models one cost of its own — a protocol that goes to zero, and how much of the
    empire it was allowed to hold — and it charges the cost of *trading* only when it is
    handed a model for it: ``friction`` is an optional
    :class:`~agent.physics.transaction_cost_analyzer.TransactionCostAnalyzer`, and with
    one the book asks it before any credit moves. Left as ``None`` the arithmetic is
    exactly what it always was, because a cost that is modelled has to be modelled
    *right*, and a wrong cost would hide a wrong decision.

    ``provider`` is the venue the credits actually move through: every stake is a call
    to :meth:`IWeb3Provider.execute_stake` and every sale one to ``withdraw``. Built
    without one, the book gets a
    :class:`~agent.physics.interfaces.NoVenueProvider` — the same code path, a venue
    that is only the ledger — so there is no configuration in which capital moves by
    some other route.

    Two lists, and the difference between them
    ------------------------------------------

    ``watchlist`` is what the *scanner* proposed (:meth:`discover`, and
    :class:`agent.physics.discovery.Watchlist`): names worth the expensive reading, and
    nothing more. It is a record of a proposal and it grants nothing — a book whose only
    list were this one would be a book that buys whatever looked interesting, which is
    the failure mode of every screener ever written.

    ``execution_whitelist`` is what this book has **approved**: the addresses that passed
    the deep assessment and for which there was room under ``execution_limit``. It is the
    only list :meth:`deposit` consults, and it is written in exactly one place
    (:meth:`approve`), which is reached only from :meth:`discover` — from a measurement,
    never from a proposal. Left as ``None`` the book behaves as it always did and the
    operator's ``whitelist`` is the whole of the permission, which is what every existing
    caller and test gets.

    What friction does, when there is one
    -------------------------------------

    ``friction`` prices the **entry**, because that is where a cost is paid and where a
    trade can still be declined: :meth:`assess_entry` asks the model for a
    :class:`~agent.physics.transaction_cost_analyzer.CostReport` on the amount about to be
    placed, against the depth the protocol publishes (:attr:`Protocol.depth_usd`; ``None``
    is *unmeasured*, and the report says so rather than inventing a slippage). A refused
    entry places nothing, keeps the credits in cash and is logged as ``skip`` with the
    model's own reason
    (:data:`~agent.physics.transaction_cost_analyzer.UNPROFITABLE_AFTER_FEES`) instead of
    raising: an entry that cannot pay for itself is a *decision*, not a bug, and a
    rebalance that raised on every thin pool would spend the cycle apologising. An
    accepted entry is charged — the position is credited ``amount − slippage − gas``, so
    ``equity`` reports what survived the trade rather than what the trade was advertised
    at. Every report is kept in :attr:`Portfolio.cost_log`, refusals included, which is
    what lets the notebook put the naive balance next to the realistic one.

    The **exit** is priced differently on purpose: it is not refused, and it is not
    charged. It cannot be refused because leaving a position is the risk-reducing move —
    the kill switch's own liquidation would then be held hostage by a gas estimate — and
    it is not charged because a cost that cannot change a decision belongs in a report and
    not in the accounting. :meth:`assess_entry` still prices one on request, so the cost of
    an exit is a number the caller may read before taking it.
    """

    whitelist: WhiteList
    cash: float = START_CAPITAL
    targets: Dict[str, float] = field(default_factory=lambda: dict(TARGETS))
    holdings: Dict[str, Holding] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    provider: Optional["IWeb3Provider"] = None
    watchlist: Optional["Watchlist"] = None
    execution_whitelist: Optional[WhiteList] = None
    execution_limit: int = EXECUTION_LIMIT
    alpha_log: List[Dict[str, Any]] = field(default_factory=list)
    friction: Optional["TransactionCostAnalyzer"] = None
    cost_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.provider is None:
            from agent.physics.interfaces import NoVenueProvider

            self.provider = NoVenueProvider(self.whitelist)
        self.cash = float(self.cash)
        if self.friction is not None and not hasattr(self.friction, "assess"):
            raise PortfolioError(
                f"friction {self.friction!r} cannot price a trade: it has no assess()"
            )
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


    # -- what an entry costs ----------------------------------------------- #

    def liquidity_of(self, protocol: Protocol) -> Optional[float]:
        """The depth a trade into ``protocol`` is priced against, or ``None``.

        Read from :attr:`Protocol.depth_usd` and from nothing else. ``turnover`` is *not*
        a substitute: it is what the pool moves in a step, and a book that priced slippage
        against flow would call every honest pool a desert. A venue that publishes no depth
        leaves this ``None``, which the cost model reports as unmeasured.
        """
        depth = getattr(protocol, "depth_usd", None)
        return None if depth is None else float(depth)

    def assess_entry(
        self, protocol: Protocol, amount: float
    ) -> Optional["CostReport"]:
        """Price one entry, or ``None`` when the book has no model to price it with.

        A *question*, and therefore pure: nothing is logged, nothing is charged, and no
        credit moves, which is what lets a review ask what an entry would cost without
        making it. :meth:`deposit` is the caller that acts on the answer, and it is the
        one that writes the :attr:`cost_log` row — a decision is recorded, a question is
        not.

        The rate handed to the model is the protocol's **measured** APY
        (:attr:`Protocol.implied_apy`), not its ``claim_apy``: a cost is subtracted from
        what the record paid, and subtracting it from what a protocol promised would
        flatter every liar in the market.
        """
        if self.friction is None:
            return None
        return self.friction.assess(
            float(protocol.implied_apy),
            float(amount),
            tvl_usd=self.liquidity_of(protocol),
            stable=protocol.bucket == BUCKET_SAFE,
        )

    def _log_cost(
        self, step: int, action: str, protocol: Protocol, report: "CostReport"
    ) -> Dict[str, Any]:
        """One row of friction history: the report, and which protocol it was about."""
        row = {
            "step": int(step),
            "action": action,
            "protocol": protocol.name,
            "address": protocol.address,
            **report.as_dict(),
        }
        self.cost_log.append(row)
        return row


    # -- what may be funded, and who decides ------------------------------- #

    def is_executable(self, address: str) -> bool:
        """True when capital may reach ``address`` at all.

        Two gates, and the answer is the *narrower* of them: the operator's list is the
        outer boundary (a venue will not process what it has not approved) and the
        execution whitelist is the inner one (the book will not buy what it has not
        measured).

        A book that has not looked at anything yet has no inner gate, and that is exactly
        the behaviour every caller had before discovery existed — left as it was so that
        the declared market, the CLI and seven hundred tests keep their semantics. But a
        book that *has* proposed a watchlist and has not yet approved anything approves
        **nothing**: an empty execution whitelist is a book that is still deciding, and
        the one thing it must not do while deciding is buy.
        """
        key = str(address).lower()
        if not self.whitelist.allows(key):
            return False
        if self.execution_whitelist is None:
            return self.watchlist is None
        return self.execution_whitelist.allows(key)

    def propose(self, watchlist: "Watchlist") -> "Watchlist":
        """Record what the scanner proposed, and refuse to record a stranger.

        A proposal is not a permission: this writes ``watchlist`` and touches nothing
        else, so a scanner that proposed the whole market would leave the book with a
        long list and the same empty execution whitelist. The check that every candidate
        is on the operator's list is defence in depth — the venue would refuse the stake
        anyway — and it is here because a proposal that could widen the outer boundary
        would make the outer boundary decorative.
        """
        for candidate in watchlist.kept:
            self.whitelist.require(candidate.address)
        self.watchlist = watchlist
        return watchlist

    def approve(self, protocols: Sequence[Protocol]) -> WhiteList:
        """Write the execution whitelist: the only method that grants permission.

        Two invariants, both of them errors rather than decisions. An address that is not
        on the operator's list cannot be approved (:class:`NotWhitelisted`), and more
        approvals than ``execution_limit`` is a bug in whoever counted
        (:class:`PortfolioError`) — the cap is enforced here so that no caller can widen
        it by asking nicely, which is the same argument the whitelist itself is built on.
        """
        chosen = list(protocols)
        if len(chosen) > int(self.execution_limit):
            raise PortfolioError(
                f"{len(chosen)} protocols approved against a cap of "
                f"{int(self.execution_limit)}"
            )
        for protocol in chosen:
            self.whitelist.require(protocol.address)
        self.execution_whitelist = WhiteList(
            tuple(protocol.address for protocol in chosen)
        )
        return self.execution_whitelist

    def discover(
        self,
        provider: Optional["IWeb3Provider"] = None,
        *,
        scanner: Any = None,
        oracle: Any = None,
        min_record: int = MIN_RECORD,
    ) -> DiscoveryReport:
        """Level 1 then level 2: ask the scanner what is worth a look, then look.

        This is the method the whole two-tier design exists for, and it is deliberately
        short: the scanner produces a :class:`~agent.physics.discovery.Watchlist`, the
        book records it (:meth:`propose`), the loop below runs the *expensive* assessment
        over **exactly those candidates and nothing else**, and the survivors — the
        admissible, in queue order, up to ``execution_limit`` — are written into the
        execution whitelist (:meth:`approve`).

        The three outcomes after level 2 are kept apart because they are three different
        facts: ``refused`` is a verdict about a record, ``capped`` is a decision about
        room, ``approved`` is permission. A funnel that merged the first two would let a
        policy look like a measurement, and :class:`DiscoveryReport` refuses to be built
        when the counts do not add up.

        ``oracle`` is optional in the same sense it is everywhere else: without a body the
        structural and mirror arms are blind, every assessment says so in
        ``details["measured"]``, and the verdict rests on Statement #87 and the claim.
        That is what makes the two-hundred-protocol run in the notebook affordable — and
        the report of that run prints how long the deep pass took, so the trade is visible
        rather than hidden.
        """
        from agent.physics.discovery import Scanner
        from agent.physics.risk_assessment import evaluate_opportunity

        venue = self.provider if provider is None else provider
        scan = Scanner() if scanner is None else scanner
        watch = self.propose(scan.scan(venue))
        step = int(getattr(venue, "step", 0))
        cap = int(self.execution_limit)
        approved: List[Any] = []
        capped: List[Any] = []
        refused: List[Dict[str, Any]] = []
        judged: Dict[str, Any] = {}
        chosen: Dict[str, Protocol] = {}
        for candidate in watch.kept:
            protocol = venue.protocol(candidate.name)
            item = evaluate_opportunity(
                protocol.as_opportunity(step),
                oracle=oracle,
                min_record=int(min_record),
            )
            judged[candidate.name] = item
            if not item.admissible:
                refused.append(
                    {
                        "name": item.name,
                        "address": item.address,
                        "verdict": item.verdict,
                        "reason": item.reason,
                    }
                )
                continue
            if len(approved) < cap:
                approved.append(candidate)
                chosen[candidate.name] = protocol
            else:
                capped.append(candidate)
        self.approve([chosen[candidate.name] for candidate in approved])
        return DiscoveryReport(
            step=step,
            watchlist=watch,
            approved=tuple(approved),
            capped=tuple(capped),
            refused=tuple(refused),
            assessments=judged,
            limit=cap,
        )


    # -- moving credits --------------------------------------------------- #

    def deposit(self, protocol: Protocol, amount: float, *, step: int = 0) -> float:
        """Stand ``amount`` credits in ``protocol``; returns what was placed.

        Three refusals, and all three are errors rather than decisions: an address the
        operator has not approved (:class:`NotWhitelisted`), a protocol that has stopped
        paying (:class:`DeadProtocol`), and a non-positive amount. A short balance is
        not an error — it clamps, because a top-up that cannot be filled should fill
        what it can instead of failing a cycle.

        The fourth gate is the execution whitelist, and it is the same error on purpose:
        an address on the operator's list that no verdict has approved is *not* a place
        the book may put capital, and the difference between "not allowed" and "not
        approved yet" belongs in the message rather than in the exception type — both are
        the book refusing, not the venue.

        The transfer itself goes to the venue (:meth:`IWeb3Provider.execute_stake`) and
        the book only moves what the venue accepted: if the answer is ``False`` the
        holding is untouched and :class:`StakeRefused` is raised, so a stake the book
        believes in and the venue does not can never be recorded as a position.

        The fifth gate is *friction*, and it is the only one that is a decision rather than
        an error: with a ``friction`` model (:meth:`assess_entry`) an entry that cannot pay
        for its own gas and slippage places nothing, logs ``skip`` with the model's reason,
        and returns ``0.0`` — the credits stay in cash and the bucket stays unfilled. An
        entry that clears the model is **charged**: of the ``placed`` credits taken out of
        cash, the slippage and the gas never reach the position, so the holding is credited
        ``placed − total_cost_usd``. That is the whole point of the gate — ``equity`` after
        a trade is what survived it, and a book that only refuses the worst trades but
        banks the costs of the rest would still be measuring a market that does not exist.
        """
        self.whitelist.require(protocol.address)
        if not self.is_executable(protocol.address):
            raise NotWhitelisted(
                f"{protocol.name} ({protocol.address}) is on the operator's list but not "
                "on the execution whitelist: no verdict approved it, or there was no room"
            )
        if not protocol.alive:
            raise DeadProtocol(f"{protocol.name} has stopped paying")
        want = float(amount)
        if want <= 0.0:
            return 0.0
        placed = min(want, max(self.cash, 0.0))
        if placed <= 0.0:
            return 0.0
        report = self.assess_entry(protocol, placed)
        if report is not None and report.refused:
            self._log_cost(step, "skip", protocol, report)
            self._log(step, "skip", protocol, placed, reason=report.reason)
            return 0.0
        charged = placed if report is None else max(placed - float(report.total_cost_usd), 0.0)
        if charged <= 0.0:
            # A trade whose friction eats the whole stake is not a stake: nothing is sent,
            # because a position of zero credits would be a line in the book that paid for
            # the privilege of being empty.
            self._log_cost(step, "skip", protocol, report)
            self._log(
                step,
                "skip",
                protocol,
                placed,
                reason=report.reason or "friction consumed the trade",
            )
            return 0.0
        if not self.provider.execute_stake(protocol.address, placed):
            raise StakeRefused(
                f"{self.provider.name} refused a stake of {placed!r} credits in "
                f"{protocol.name} ({protocol.address})"
            )
        if report is not None:
            self._log_cost(step, "deposit", protocol, report)
        holding = self.holdings.get(protocol.address)
        if holding is None:
            holding = Holding(protocol=protocol)
            self.holdings[protocol.address] = holding
        holding.credits = float(holding.credits) + charged
        self.cash -= placed
        self._log(step, "deposit", protocol, charged)
        return float(placed)

    def withdraw(
        self, address: str, amount: float, *, step: int = 0, reason: str = ""
    ) -> float:
        """Take credits out of one position, back to cash; returns what came out.

        Routed like every other movement: the venue is asked first
        (:meth:`IWeb3Provider.withdraw`) and the book follows the answer.
        """
        key = str(address).lower()
        holding = self.holdings.get(key)
        if holding is None or float(amount) <= 0.0:
            return 0.0
        taken = min(float(amount), float(holding.credits))
        if taken <= 0.0:
            return 0.0
        if not self.provider.withdraw(key, taken):
            raise StakeRefused(
                f"{self.provider.name} refused a withdrawal of {taken!r} credits from "
                f"{holding.protocol.name} ({key})"
            )
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
        """Sell everything in these buckets back to cash; returns the credits raised.

        A sale is a withdrawal from the venue, position by position, so the kill
        switch's exit is the same call a rebalance makes — there is no "sell everything"
        shortcut that could skip the interface on the one path where it matters most.
        """
        wanted = set(buckets)
        raised = 0.0
        for key in list(self.holdings):
            holding = self.holdings[key]
            if holding.protocol.bucket not in wanted:
                continue
            credits = float(holding.credits)
            if credits > 0.0 and not self.provider.withdraw(key, credits):
                raise StakeRefused(
                    f"{self.provider.name} refused to release {credits!r} credits from "
                    f"{holding.protocol.name} ({key})"
                )
            raised += credits
            self.cash += credits
            self._close(key, step, reason, taken=credits)
        return float(raised)

    # -- the alpha budget: what a measured panic may move ------------------- #

    def alpha_capacity(self) -> float:
        """The share of equity this book may still commit: cash above the floor.

        Not "cash": a book whose cash is already at the floor has nothing to answer a
        panic with, and a book that spends the floor to buy a panic is a book that has
        converted a measured opportunity into an unhedged position. The number is
        computed from the *current* book, so it shrinks as the boost is spent.
        """
        equity = max(float(self.equity), 1e-12)
        return max(float(self.cash) / equity - float(ALPHA_CASH_FLOOR), 0.0)

    def alpha_hunting_weight(self, signal: Any, *, step: int = 0) -> float:
        """The multiplier a panic signal puts on its bucket's target — the strategy.

        ``1.0`` is the answer for everything that is not a fired signal, and that is the
        whole of the safety argument: a book with no measurement behaves exactly as it
        did before this method existed. A fired signal buys at most
        :data:`ALPHA_BOOST_CAP` of its bucket, scaled by the signal's own ``confidence``,
        so the two numbers that decide the size are *the cap* (policy, declared here) and
        *the measurement* (the ``z`` and the size of the dislocation, computed in
        :mod:`agent.physics.volatility_hunter`).

        The boost is temporary by construction rather than by a timer: nothing here is
        stored, permissioned or remembered, and the weight is a function of a signal that
        is itself a function of the last
        :data:`~agent.physics.volatility_hunter.PANIC_RECENT` days. When the dislocation
        closes the signal stops firing and the weight returns to one on the next review,
        without anyone having to remember to undo it.
        """
        if not _signal_fired(signal):
            return 1.0
        return 1.0 + float(ALPHA_BOOST_CAP) * _signal_confidence(signal)

    def boosted_targets(
        self, signals: Optional[Sequence[Any]] = None, *, step: int = 0
    ) -> Dict[str, float]:
        """The bucket targets with every fired panic priced in — what ``rebalance`` takes.

        Three limits, and each one is a different fear:

        * the **cap** (:data:`ALPHA_BOOST_CAP`) — one measurement may rewrite at most
          half of its own bucket, and never another bucket's;
        * the **cash floor** (:data:`ALPHA_CASH_FLOOR`) — the boost is funded out of cash
          the book is holding, so the sum of the targets may exceed one while the book
          can never run out of the capital that answers a surprise;
        * the **order** — when the floor binds, the strongest signal is served first, so
          the room goes to the loudest *measurement* rather than to the first name in a
          list.

        Everything granted is appended to :attr:`alpha_log` with the numbers that granted
        it, because a book that moved a tenth of its equity on a measurement should be
        able to say which measurement.
        """
        base = {bucket: float(value) for bucket, value in self.targets.items()}
        targets = dict(base)
        capacity = self.alpha_capacity()
        if capacity <= 0.0:
            return targets
        ranked = sorted(
            (signal for signal in (signals or ()) if _signal_fired(signal)),
            key=lambda signal: (-_signal_confidence(signal), str(_signal_name(signal))),
        )
        granted_by: Dict[str, float] = {}
        for signal in ranked:
            bucket = _signal_bucket(signal)
            if bucket not in targets:
                continue
            weight = self.alpha_hunting_weight(signal, step=step)
            want = (weight - 1.0) * base[bucket]
            ceiling = float(ALPHA_BOOST_CAP) * base[bucket] - granted_by.get(bucket, 0.0)
            grant = min(want, max(ceiling, 0.0), capacity)
            if grant <= 0.0:
                continue
            granted_by[bucket] = granted_by.get(bucket, 0.0) + grant
            targets[bucket] = targets[bucket] + grant
            capacity -= grant
            self.alpha_log.append(
                {
                    "step": int(step),
                    "bucket": bucket,
                    "protocol": _signal_name(signal),
                    "confidence": round(_signal_confidence(signal), 4),
                    "weight": round(weight, 4),
                    "wanted": round(want, 6),
                    "granted": round(grant, 6),
                    "capacity_left": round(capacity, 6),
                }
            )
        return targets

    def rebalance(
        self,
        plan: Mapping[str, Optional[Protocol]],
        *,
        step: int = 0,
        targets: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, float]:
        """Bring each bucket to its target, with ``plan`` naming where it may stand.

        ``plan[bucket] is None`` means *nothing in this bucket passed the assessment*,
        and the bucket goes to cash — that is the whole answer to a market full of
        traps: an empty bucket is a position. Buckets are filled safest first, so a
        portfolio short of cash buys safety before it buys yield.

        The trim band is why a bucket that is a little over its target is left alone: at
        ten percent of equity, a trim of less than :data:`TRIM_BAND` of the whole empire
        costs a trade to move nothing.

        ``targets`` is how a *temporary* policy gets in: with ``None`` the book uses its
        own numbers, exactly as it always has, and with a mapping —
        :meth:`boosted_targets` is the one this file produces — the same arithmetic runs
        against those instead. That is deliberate: a panic does not get its own code path
        through the accounting, it gets a bigger number in the same one.
        """
        scale = dict(self.targets) if targets is None else dict(targets)
        missing = [bucket for bucket in BUCKETS if bucket not in scale]
        if missing:
            raise PortfolioError(f"targets must name every bucket, missing {missing}")
        placed: Dict[str, float] = {}
        for bucket in BUCKETS:
            chosen = plan.get(bucket)
            target = float(scale[bucket]) * self.equity
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

    def friction_summary(self) -> Optional[Dict[str, Any]]:
        """What the cost model charged and what it refused, as one row.

        ``None`` when the book has no model, because the absence of a measurement is
        reported as absent rather than as a row of zeros that would read like a very cheap
        venue — the same discipline a ``CostReport`` keeps about an unmeasured depth.

        ``paid_usd`` is what the entries actually cost and ``refused_usd`` is the capital
        the model kept in cash. Those two numbers are the whole distance between the naive
        balance and the realistic one, so a reader does not have to sweep
        :attr:`cost_log` to find them.
        """
        if self.friction is None:
            return None
        taken = [row for row in self.cost_log if row["action"] == "deposit"]
        skipped = [row for row in self.cost_log if row["action"] == "skip"]
        return {
            "entries": len(taken),
            "refused": len(skipped),
            "paid_usd": round(sum(float(row["total_cost_usd"]) for row in taken), 6),
            "refused_usd": round(
                sum(float(row["trade_size_usd"]) for row in skipped), 6
            ),
            "horizon_days": float(getattr(self.friction, "horizon_days", 0.0)),
            "eth_price_usd": float(getattr(self.friction, "eth_price_usd", 0.0)),
        }

    def report(self) -> Dict[str, Any]:
        """The wallet, bucket by bucket, with the weights it ended the run at."""
        return {
            "cash": round(float(self.cash), 6),
            "equity": round(float(self.equity), 6),
            "provider": str(self.provider.name),
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
            "watchlist": (
                None
                if self.watchlist is None
                else {
                    "watched": len(self.watchlist),
                    "dropped": len(self.watchlist.dropped),
                    "step": int(self.watchlist.step),
                }
            ),
            "execution_whitelist": (
                None
                if self.execution_whitelist is None
                else self.execution_whitelist.as_list()
            ),
            "execution_limit": int(self.execution_limit),
            "friction": self.friction_summary(),
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

    The fourth party is the **venue**
    (:class:`~agent.physics.interfaces.IWeb3Provider`): it publishes what is on offer,
    keeps the clock, and is the only thing that moves a credit. The empire never holds a
    market's numbers in its own hands — it asks the provider, and the portfolio moves
    capital through the same one — so replacing the simulation with a testnet is a
    change of constructor argument in
    :meth:`launch`, not a change in this class.

    The fifth is the **scanner** (:mod:`agent.physics.discovery`), and it is the only
    thing here that exists for scale. With no scanner the empire assesses every protocol
    a venue publishes, which is the right reading for a market of seven and an
    impossibility for a market of five hundred: a full review cycle would run the deep
    assessment — Statement #87, the structural survey, the mirror — hundreds of times to
    find the same handful of answers. With a scanner, :meth:`review` hands the *portfolio*
    the job of looking, and the portfolio looks only at what survived level 1
    (:meth:`Portfolio.discover`). Nothing else changes: the verdicts are the same verdicts
    on the same records, and a protocol the scanner never watched was never judged — the
    report says so as ``truncated`` rather than pretending it was refused.

    The sixth is optional and inert by default: ``alpha`` is a list of *signals* the caller
    measured — the ``PanicSignal`` values of :mod:`agent.physics.volatility_hunter`, or
    anything shaped like them — and :meth:`_targets_with_alpha` is the only place they touch
    capital. They can re-weight a bucket the plan already chose and nothing else, so the
    hunter cannot name a protocol, cannot widen a whitelist, and cannot size a position
    beyond the book's own cap.
    """

    market: Optional[ProtocolMarket]
    portfolio: Portfolio
    kill_switch: Any
    provider: Optional["IWeb3Provider"] = None
    oracle: Any = None
    review_every: int = CYCLE
    min_record: int = MIN_RECORD
    step: int = 0
    assessments: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    log: List[Dict[str, Any]] = field(default_factory=list)
    proofs: List[Dict[str, Any]] = field(default_factory=list)
    volatility: Optional[float] = None
    resonance: Optional[float] = None
    scanner: Any = None
    funnel: Optional[Dict[str, Any]] = None
    alpha: Optional[Sequence[Any]] = None

    def __post_init__(self) -> None:
        """Hand the one venue to the book, or refuse two of them.

        A run that assessed one market and moved capital through another would be a run
        whose numbers mean nothing, so the disagreement is an error rather than a
        preference: a book with no venue attached adopts the empire's, and a book with a
        *different* one is a bug.

        ``market`` may be ``None`` when the venue is not a declared market at all — a
        testnet answers ``IWeb3Provider`` and publishes no ``ProtocolMarket``. Nothing in
        the decision reads this attribute; it is the *simulation* the empire was born in,
        kept for the report, and its absence is reported as absent rather than filled
        with a mock that was never tradeable.
        """
        from agent.physics.interfaces import MockWeb3Provider, NoVenueProvider

        if self.provider is None:
            if self.market is None:
                raise PortfolioError("an empire needs a venue: pass a provider")
            self.provider = MockWeb3Provider(self.market)
        book = self.portfolio.provider
        if isinstance(book, NoVenueProvider) or book is None:
            self.portfolio.provider = self.provider
        elif book is not self.provider:
            raise PortfolioError(
                "the portfolio and the empire must trade through the same provider"
            )
        self.step = int(self.step)

    # -- birth -------------------------------------------------------------- #

    @classmethod
    def launch(
        cls,
        *,
        body: Any = None,
        market: Optional[ProtocolMarket] = None,
        provider: Optional["IWeb3Provider"] = None,
        capital: float = START_CAPITAL,
        steps: int = 720,
        seed: int = 7,
        review_every: int = CYCLE,
        kill_switch: Any = None,
        scanner: Any = None,
        execution_limit: int = EXECUTION_LIMIT,
        friction: Optional["TransactionCostAnalyzer"] = None,
    ) -> "Empire":
        """Open for business: a venue, a wallet, and (if given) a body's own oracle.

        ``provider`` is the whole of the dependency injection: give it a
        :class:`~agent.physics.interfaces.MockWeb3Provider` and the empire trades the
        simulated chain, give it anything else that answers
        :class:`~agent.physics.interfaces.IWeb3Provider` and the same assessment, cap
        and switch run against that instead. Omitted, one is built over ``market`` — or
        over the declared market of ``steps`` and ``seed`` — so every existing entry
        point keeps its behaviour and still moves capital through the interface.

        ``body`` is an :class:`~agent.physics.organism.Organism`. Without one the empire
        still trades — the sterility and claim arms need no body — but the structural
        and mirror arms are blind, and every assessment says so in
        ``details["measured"]``. That is the honest way to run a cheaper simulation: the
        absence of a measurement is recorded, not filled in.

        A provider that is not a declared market — a testnet, say — leaves ``market`` as
        ``None``: there is no ``ProtocolMarket`` to stand in for it, and inventing one
        would mean the report printed a simulation that nothing was ever traded on. The
        wallet's whitelist comes from the venue in every case, because the venue is the
        only thing that knows what it will actually accept.

        ``scanner`` is the only argument that changes *what the empire looks at* rather
        than what it looks through: given one, every review goes through
        :meth:`Portfolio.discover` and the deep assessment runs on the watchlist instead
        of on the whole market. ``execution_limit`` is the cap on the resulting
        whitelist, and it belongs to the wallet — the thing that spends — so it is written
        there and enforced there (:meth:`Portfolio.approve`), not merely passed along.

        ``friction`` is the one argument that changes *what a trade costs*: given a
        :class:`~agent.physics.transaction_cost_analyzer.TransactionCostAnalyzer`, the
        wallet refuses entries that cannot pay their own gas and slippage and charges the
        ones it takes. Omitted, the book is exactly as frictionless as it has always
        been, which is the honest default for a simulation whose protocols publish no
        depth.
        """
        from agent.physics.interfaces import MockWeb3Provider
        from agent.physics.risk_assessment import KillSwitch

        venue = getattr(provider, "market", None)
        if market is None:
            if venue is not None:
                market = venue
            elif provider is None:
                market = ProtocolMarket.mock(steps=int(steps), seed=int(seed))
        elif venue is not None and venue is not market:
            # The provider publishes a clock and a record list, and both are the
            # market's: two different markets in one run would mean assessing one and
            # trading the other, which is the failure mode this file is written against.
            raise PortfolioError(
                "the provider publishes a different market than the one given"
            )
        if provider is None:
            provider = MockWeb3Provider(market)
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
        portfolio = Portfolio(
            whitelist=provider.whitelist,
            cash=float(capital),
            provider=provider,
            execution_limit=int(execution_limit),
            friction=friction,
        )
        return cls(
            market=market,
            portfolio=portfolio,
            kill_switch=kill_switch if kill_switch is not None else KillSwitch(),
            provider=provider,
            oracle=oracle,
            review_every=int(review_every),
            scanner=scanner,
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

        Everything read here comes from the provider and everything moved here goes
        through it: the assessment is handed the venue, the protocols in the plan are
        fetched *by name from the venue*, and each stake is one ``execute_stake`` call.
        A different provider therefore changes what the empire can see and touch without
        changing a line of the decision.

        With a ``scanner`` attached the reading is the same reading, taken over a smaller
        set: :meth:`Portfolio.discover` runs level 1 over everything the venue publishes
        and level 2 over the watchlist alone, writes the surviving addresses into the
        wallet's execution whitelist, and the plan below may only name one of them. That
        last clause is the whole safety argument of the two-tier design — the scanner can
        put a protocol in front of the assessment, never in front of the money.
        """
        from agent.physics.risk_assessment import CONVICTIONS
        from agent.physics.risk_assessment import review as assess_all

        approved: Optional[set] = None
        if self.scanner is not None:
            report = self.portfolio.discover(
                self.provider,
                scanner=self.scanner,
                oracle=self.oracle,
                min_record=self.min_record,
            )
            found: Dict[str, Any] = dict(report.assessments)
            self.funnel = report.as_dict()
            whitelist = self.portfolio.execution_whitelist
            approved = set() if whitelist is None else set(whitelist.as_list())
        else:
            found = assess_all(
                self.provider, oracle=self.oracle, min_record=self.min_record
            )
        plan: Dict[str, Optional[Protocol]] = {}
        for bucket in BUCKETS:
            if bucket != BUCKET_SAFE and bool(getattr(self.kill_switch, "engaged")):
                plan[bucket] = None
                continue
            admissible = [
                item
                for item in found.values()
                if item.bucket == bucket
                and item.admissible
                and (approved is None or str(item.address).lower() in approved)
            ]
            if not admissible:
                plan[bucket] = None
                continue
            best = max(admissible, key=lambda item: (item.implied_apy, item.name))
            plan[bucket] = self.provider.protocol(best.name)
        placed = self.portfolio.rebalance(
            plan, step=self.step, targets=self._targets_with_alpha()
        )
        measured = {name: item.as_dict() for name, item in found.items()}
        self.resonance = self._resonance_of(plan, measured)
        self.assessments[self.step] = measured
        submitted = self._submit_proofs(found, CONVICTIONS)
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
            "proofs": int(submitted),
            "resonance": None if self.resonance is None else round(self.resonance, 4),
            "funnel": self.funnel,
            "approved": (
                None
                if self.portfolio.execution_whitelist is None
                else len(self.portfolio.execution_whitelist)
            ),
        }
        self.decisions.append(decision)
        self.log.append({"step": int(self.step), "action": "review", **decision})
        return decision

    def _targets_with_alpha(self) -> Optional[Dict[str, float]]:
        """The book's targets with today's measured panics priced in — or ``None``.

        ``alpha`` is how the hunter (:mod:`agent.physics.volatility_hunter`) reaches the
        manager: the caller hands over the signals it measured — ``PanicSignal`` values,
        mappings, or nothing at all — and this is the only place they are allowed to touch
        capital. A signal cannot name a protocol, cannot add an address to a whitelist and
        cannot size a position: it can only *re-weight* buckets the plan already chose, via
        :meth:`Portfolio.boosted_targets`, which is bounded by the cap and the cash floor.

        ``None`` is the answer when there is nothing to price in, and ``None`` means
        :meth:`Portfolio.rebalance` uses the book's own targets — so an empire with no
        hunter attached behaves exactly as it did before this method existed.
        """
        signals = [signal for signal in (self.alpha or ()) if signal is not None]
        if not signals:
            return None
        return self.portfolio.boosted_targets(signals, step=self.step)

    def _submit_proofs(
        self, found: Mapping[str, Any], convictions: Sequence[str]
    ) -> int:
        """Hand the venue the evidence behind a conviction; returns how many it took.

        A conviction is the one thing this layer knows that the venue does not: the
        record says one thing and the advertisement says another, and *why* is a
        measurement, not an opinion. Publishing it is a call to
        :meth:`IWeb3Provider.submit_validation_proof` — the one operation that
        publishes instead of moving — so the step a real deployment would charge for is
        never quietly skipped in the simulation. ``False`` is recorded as a refusal and
        never as a success: a proof the venue did not take did not travel.
        """
        taken = 0
        for _, item in sorted(found.items()):
            if item.verdict not in convictions:
                continue
            proof = {
                "protocol": item.name,
                "address": item.address,
                "verdict": item.verdict,
                "reason": item.reason,
                "claim_apy": round(float(item.claim_apy), 6),
                "implied_apy": round(float(item.implied_apy), 6),
                "points": int(item.points),
            }
            if self.provider.submit_validation_proof(item.address, proof):
                taken += 1
                self.proofs.append({"step": int(self.step), **proof})
        return taken

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

        The record is fetched through the venue
        (:meth:`IWeb3Provider.fetch_protocol_history`), not read off a local array: a
        provider that publishes a shorter or later record therefore moves this number,
        which is the point — the arm has to be able to fire on what the venue says, not
        on what the simulation knows.

        With a scanner attached the median is taken over the *watchlist* rather than over
        every published protocol, and that is the same argument one level down: the arm
        reads the world the empire is actually looking at. A market of five hundred
        protocols measured on all five hundred, every step, is a volatility estimate that
        costs more than the capital it protects.
        """
        from agent.physics.risk_assessment import volatility_ratio

        protocols = list(self.provider.get_protocol_list())
        if self.portfolio.watchlist is not None:
            watched = set(self.portfolio.watchlist.addresses())
            protocols = [
                protocol for protocol in protocols if protocol.address in watched
            ]
        ratios: List[float] = []
        for protocol in protocols:
            if not protocol.alive:
                continue
            if protocol.bucket not in (BUCKET_ACTIVE, BUCKET_EXPERIMENTAL):
                continue
            record = np.asarray(
                self.provider.fetch_protocol_history(protocol.address), dtype=float
            )
            if record.size < 4:
                continue
            ratios.append(float(volatility_ratio(record)))
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
            self.provider.advance()
            self.step = int(self.provider.step)
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
        """Everything a report prints: venue, wallet, switch, verdicts, proofs, log."""
        return {
            "step": int(self.step),
            "provider": self.provider.describe(),
            "market": None if self.market is None else self.market.summary(),
            "portfolio": self.portfolio.report(),
            "kill_switch": self.kill_switch.as_dict(),
            "proofs": list(self.proofs),
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
            "funnel": self.funnel,
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
            f"venue: {self.provider.name}, {len(self.proofs)} proofs accepted, "
            f"{int(self.provider.describe().get('stakes', 0))} stakes executed",
            f"wallet: {len(self.portfolio.holdings)} positions, cash "
            f"{self.portfolio.cash:.0f}, targets "
            + ", ".join(f"{bucket} {share:.0%}" for bucket, share in TARGETS.items()),
            f"reviews: {len(self.decisions)}, verdicts "
            + ", ".join(f"{verdict} {count}" for verdict, count in counts.items()),
            "kill switch: "
            f"{'engaged' if getattr(self.kill_switch, 'engaged') else 'idle'}, "
            f"{len(triggers)} events",
        ]
        if self.funnel is not None:
            lines.append(
                f"funnel: {self.funnel['market']} published -> "
                f"{self.funnel['watchlist']} watched -> "
                f"{self.funnel['approved']} approved "
                f"(refused by scanner {self.funnel['refused_by_scanner']}, "
                f"by portfolio {self.funnel['refused_by_portfolio']}, "
                f"no room {self.funnel['capped']})"
            )
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
        provider: Optional["IWeb3Provider"] = None,
    ) -> "NaiveInvestor":
        """The same wallet and the same whitelist; only the book differs.

        The control group moves its capital through the same venue as the empire — a
        foil that traded by some other route would be comparing two things, not one,
        and the figure exists to compare exactly one: whether the record was measured.
        """
        from agent.physics.interfaces import MockWeb3Provider

        return cls(
            market=market,
            portfolio=Portfolio(
                whitelist=market.whitelist,
                cash=float(capital),
                provider=provider if provider is not None else MockWeb3Provider(market),
            ),
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
    "StakeRefused",
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
