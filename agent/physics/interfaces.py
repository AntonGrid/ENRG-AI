"""The seam between the decision and whoever publishes the numbers.

Everything in this layer that *decides* — ``risk_assessment``, ``Portfolio``, the kill
switch — needs a small, fixed set of things from the outside world: a list of what is on
offer, the record each offering has published, a way to hand over the evidence behind a
verdict, and a way to move capital in and out. Nothing else. Those five operations are
:class:`IWeb3Provider`, and writing them down is the whole point: with them the rest of
the layer can be run against *anything* that answers them, and the declared market of
``portfolio.ProtocolMarket`` is simply the first implementation, not the only one.

Why an interface, and not a flag
================================

A ``live=True`` switch inside ``ProtocolMarket`` would have been shorter and much worse.
It would keep a branch of code that only runs in production — the one branch no test
ever executes — and it would put a network call two lines away from a loop that runs
seven hundred times. :class:`MockWeb3Provider` is not a stub: it is *the adapter for the
simulated chain*, and it is the adapter the demo, the CLI and the notebook all use. A
statement measured through it is a statement about the code that would face a real
venue; the only thing that changes is where the bytes come from.

What a real provider answers, and what this file must never contain
===================================================================

::

    class RealTestnetProvider(IWeb3Provider):
        def __init__(self, rpc_url: str, key: str) -> None:
            self._web3 = Web3(HTTPProvider(rpc_url))          # not in this package
            self._signer = self._web3.eth.account.from_key(key)  # not in this package

Those two lines are the entire difference between this file and a testnet, and they are
deliberately *absent*: there is no RPC endpoint, no ABI, no signer and no private key
anywhere in ``agent/physics``, so no line of it can move real value even when it is run
by mistake. A provider that can sign is a provider that can lose money; that decision
belongs to an operator holding a key, in a file that is not this one. What lives here is
the shape such a file has to satisfy — and a mock that satisfies it exactly.

The contract
============

=========================  ==============================================
get_protocol_list()        what is on offer, as records the agent may read
fetch_protocol_history()   the published record of one address, up to now
submit_validation_proof()  the evidence behind a verdict, handed over
execute_stake()            move capital *in*
withdraw()                 move capital *out*
=========================  ==============================================

plus three facts every caller needs: ``step`` (the venue's clock, which is the frontier
of what has happened), ``whitelist`` (the operator's approved addresses) and ``name``
(which venue this is, for the log). :meth:`IWeb3Provider.opportunities` and the other
read helpers are *defaults on the interface*, written once in terms of the list, so two
providers cannot disagree about how a published record looks to the assessment.

``Protocol`` is the value object both sides speak. It carries the address, the claim and
the record, and :meth:`Protocol.record` cannot be asked for a step that has not happened
— so a provider that hands out ``Protocol`` objects cannot leak the future into a
decision, however it is implemented. That is a property of the type, not of the mock.

Where the interface is enforced
===============================

``IWeb3Provider`` is a :class:`typing.Protocol`: an implementation is checked for
*shape* (``isinstance`` works, ``issubclass`` does not, because the interface has
attributes). Shape is not semantics, and pretending otherwise would be the kind of claim
this layer refuses to make — the semantics are held by
``tests/test_physics_interfaces.py``, which runs every statement twice: once through the
market, once through the adapter.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - for the type checker, never imported at runtime
    from agent.physics.portfolio import Protocol as PublishedProtocol
    from agent.physics.portfolio import ProtocolMarket, WhiteList

__all__ = ["IWeb3Provider", "MockWeb3Provider", "NoVenueProvider"]


@runtime_checkable
class IWeb3Provider(Protocol):
    """What a venue must answer before this layer can decide anything about it.

    Five operations and three facts. Everything the agent does with capital is a call to
    one of those five, which is why swapping the implementation is a one-line change in
    an entry point and not a refactor: the assessment, the cap and the switch are
    written against this, never against a market.
    """

    # -- identity: which venue, and where its clock stands ------------------ #

    @property
    def name(self) -> str:
        """A short label for the log: ``mock``, ``none``, or a network's own name."""
        ...

    @property
    def step(self) -> int:
        """The venue's clock — the frontier of what has happened and may be read."""
        ...

    @property
    def whitelist(self) -> "WhiteList":
        """The operator's approved addresses, as the venue publishes them.

        Read, not written: an adapter that could add to this list would be an adapter
        that can widen its own permissions, and the whitelist's only value is that it
        cannot be widened by the thing it constrains.
        """
        ...

    # -- the five the layer is built on ------------------------------------- #

    def get_protocol_list(self) -> List["PublishedProtocol"]:
        """Everything on offer, each with the record it has published so far."""
        ...

    def fetch_protocol_history(
        self, address: str, steps: Optional[int] = None
    ) -> List[float]:
        """The record at ``address``: all of it, or the last ``steps`` points.

        Returns an empty list for an address the venue does not publish, and never
        raises: a feed that crashes on a bad address cannot be run against a real one,
        and an unreadable protocol is a protocol to refuse, not an exception.
        """
        ...

    def submit_validation_proof(
        self, protocol_address: str, proof_data: Mapping[str, Any]
    ) -> bool:
        """Hand the venue the evidence behind a verdict; ``True`` when it was taken.

        ``False`` means *not accepted* — an address the venue does not publish, or
        nothing to send. It never means "sent and rejected by a network", because
        nothing here can reach a network.
        """
        ...

    def execute_stake(self, protocol_address: str, amount: float) -> bool:
        """Move ``amount`` credits in; ``True`` when the venue accepted the stake."""
        ...

    def withdraw(self, protocol_address: str, amount: float) -> bool:
        """Move ``amount`` credits out; ``True`` when the venue accepted it."""
        ...

    # -- the clock ---------------------------------------------------------- #

    def advance(self, steps: int = 1) -> int:
        """Let ``steps`` steps happen; returns the clock afterwards."""
        ...

    # -- defaults: one reading of a record, whatever the provider is --------- #

    def opportunities(self) -> List[Dict[str, Any]]:
        """Every published opportunity at the current step — and nothing beyond it."""
        step = int(self.step)
        return [
            protocol.as_opportunity(step) for protocol in self.get_protocol_list()
        ]

    def alive(self, bucket: Optional[str] = None) -> List["PublishedProtocol"]:
        """Every protocol still paying, in publication order, optionally one bucket."""
        return [
            protocol
            for protocol in self.get_protocol_list()
            if protocol.alive and (bucket is None or protocol.bucket == bucket)
        ]

    def protocol(self, name: str) -> "PublishedProtocol":
        """One protocol by name; raises rather than returning a stranger."""
        for protocol in self.get_protocol_list():
            if protocol.name == str(name):
                return protocol
        from agent.physics.portfolio import PortfolioError

        raise PortfolioError(f"the venue has no protocol called {name!r}")

    def records(self) -> Dict[str, Any]:
        """Name to published record — what a table or a figure draws."""
        step = int(self.step)
        return {
            protocol.name: protocol.record(step)
            for protocol in self.get_protocol_list()
        }

    def describe(self) -> Dict[str, Any]:
        """The venue as a report line: who it is, its clock, what it publishes."""
        return {
            "name": str(self.name),
            "step": int(self.step),
            "protocols": len(self.get_protocol_list()),
        }


class MockWeb3Provider(IWeb3Provider):
    """The declared market, behind the interface a real chain would stand behind.

    Constructed with no arguments it *is* the demo: :meth:`ProtocolMarket.mock` with the
    seven protocols, the two traps (``YieldMax`` pays a thousand percent exactly until
    step 300, ``AurumDAO`` re-serves eight numbers and claims four thousand two hundred)
    and the two crashes. Passing an existing market in adapts that one instead, which is
    what the tests and the notebook do so that a run cannot accidentally trade a second,
    differently seeded market.

    The adapter keeps what a venue keeps: the stakes it accepted, the withdrawals it
    processed and the proofs it was handed. Those ledgers are not decoration — they are
    how a test can say "the capital went through the interface" rather than "the numbers
    happen to match" — and none of them is a signature, an endpoint or a key.

    What this class deliberately does *not* have is a way to make the market pay more
    than its records say. ``step`` is the frontier of what may be read, ``advance`` is
    the only thing that moves it, and :meth:`fetch_protocol_history` slices at that
    frontier. The future is in the market's arrays and behind the clock, exactly as in a
    real chain where it is behind the next block.
    """

    def __init__(
        self,
        market: Optional["ProtocolMarket"] = None,
        *,
        steps: int = 720,
        seed: int = 7,
    ) -> None:
        if market is None:
            from agent.physics.portfolio import ProtocolMarket as DeclaredMarket

            market = DeclaredMarket.mock(steps=int(steps), seed=int(seed))
        self._market = market
        self.stakes: List[Dict[str, Any]] = []
        self.withdrawals: List[Dict[str, Any]] = []
        self.proofs: List[Dict[str, Any]] = []
        self.refusals: int = 0

    # -- identity ----------------------------------------------------------- #

    @property
    def name(self) -> str:
        return "mock"

    @property
    def step(self) -> int:
        return int(self._market.step)

    @property
    def whitelist(self) -> "WhiteList":
        return self._market.whitelist

    @property
    def market(self) -> "ProtocolMarket":
        """The simulated venue underneath, for a report, a figure or a table."""
        return self._market

    # -- the five the layer is built on -------------------------------------- #

    def get_protocol_list(self) -> List["PublishedProtocol"]:
        return list(self._market.protocols)

    def fetch_protocol_history(
        self, address: str, steps: Optional[int] = None
    ) -> List[float]:
        for protocol in self._market.protocols:
            if protocol.address != str(address).lower():
                continue
            record = protocol.record(self.step)
            if steps is None:
                return [float(value) for value in record]
            window = max(int(steps), 0)
            if window <= 0:
                return []
            return [float(value) for value in record[max(record.size - window, 0) :]]
        return []

    def submit_validation_proof(
        self, protocol_address: str, proof_data: Mapping[str, Any]
    ) -> bool:
        key = str(protocol_address).lower()
        if not proof_data or not self._market.whitelist.allows(key):
            self.refusals += 1
            return False
        self.proofs.append(
            {"step": self.step, "address": key, "proof": dict(proof_data)}
        )
        return True

    def execute_stake(self, protocol_address: str, amount: float) -> bool:
        key = str(protocol_address).lower()
        value = float(amount)
        if value <= 0.0 or not self._market.whitelist.allows(key):
            self.refusals += 1
            return False
        self.stakes.append({"step": self.step, "address": key, "amount": value})
        return True

    def withdraw(self, protocol_address: str, amount: float) -> bool:
        key = str(protocol_address).lower()
        value = float(amount)
        if value <= 0.0 or not self._market.whitelist.allows(key):
            self.refusals += 1
            return False
        self.withdrawals.append({"step": self.step, "address": key, "amount": value})
        return True

    # -- the clock ----------------------------------------------------------- #

    def advance(self, steps: int = 1) -> int:
        return int(self._market.advance(steps))

    # -- what it can say ------------------------------------------------------ #

    def describe(self) -> Dict[str, Any]:
        line = super().describe()
        line.update(
            {
                "stakes": len(self.stakes),
                "withdrawals": len(self.withdrawals),
                "proofs": len(self.proofs),
                "refusals": int(self.refusals),
            }
        )
        return line


class NoVenueProvider(IWeb3Provider):
    """No venue at all: the book itself is the ledger.

    This is what a :class:`~agent.physics.portfolio.Portfolio` is given when it is
    constructed without one, and it exists so that *"there is no venue"* is an
    implementation of the interface rather than a branch that skips it. The capital path
    is then the same code in every case: a stake is a call to ``execute_stake``, and the
    only thing that differs between this provider and a real one is who records it.

    What it says, it says honestly:

    * it publishes no protocols — there is nothing to read, so ``get_protocol_list`` is
      empty and every read helper on the interface degenerates with it;
    * it accepts every stake and every withdrawal, because the portfolio's own
      accounting is the only record there is and this provider has nothing to check it
      against;
    * it **refuses every proof**. There is nowhere to send one, and returning ``True``
      would be a claim to have transmitted something — the one thing a simulation must
      never pretend.
    """

    def __init__(self, whitelist: Optional["WhiteList"] = None) -> None:
        if whitelist is None:
            from agent.physics.portfolio import WhiteList as OperatorList

            whitelist = OperatorList()
        self._whitelist = whitelist
        self.stakes: List[Dict[str, Any]] = []
        self.withdrawals: List[Dict[str, Any]] = []

    # -- identity ----------------------------------------------------------- #

    @property
    def name(self) -> str:
        return "none"

    @property
    def step(self) -> int:
        """A clock that stands still: without a venue, nothing happens to anybody."""
        return 0

    @property
    def whitelist(self) -> "WhiteList":
        return self._whitelist

    # -- the five, answered as honestly as an absent venue can ---------------- #

    def get_protocol_list(self) -> List["PublishedProtocol"]:
        return []

    def fetch_protocol_history(
        self, address: str, steps: Optional[int] = None
    ) -> List[float]:
        return []

    def submit_validation_proof(
        self, protocol_address: str, proof_data: Mapping[str, Any]
    ) -> bool:
        return False

    def execute_stake(self, protocol_address: str, amount: float) -> bool:
        self.stakes.append(
            {
                "step": int(self.step),
                "address": str(protocol_address).lower(),
                "amount": float(amount),
            }
        )
        return True

    def withdraw(self, protocol_address: str, amount: float) -> bool:
        self.withdrawals.append(
            {
                "step": int(self.step),
                "address": str(protocol_address).lower(),
                "amount": float(amount),
            }
        )
        return True

    # -- the clock ----------------------------------------------------------- #

    def advance(self, steps: int = 1) -> int:
        return 0
