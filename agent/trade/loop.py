"""The trading cycle: what the wallet holds → what the market says → a size → a tick.

This is the live loop. It is the file the container runs, and it answers four questions
in this order, each of which can stop the cycle:

1. **What may be traded?** The operator's address book
   (``TESTNET_PROTOCOLS``) and the record each vault has published on chain
   (``RealTestnetProvider.get_protocol_list``). This file never chooses an address: a
   venue with an empty book trades nothing, and says why in as many lines as it has
   entries — an empty book and a book of addresses that publish no ``YieldDistributed``
   are different faults, and the message names the one that happened. A record can also be
   the *feed's* (``TESTNET_HISTORY=auto``/``feed``), and then it is a fact about a name
   rather than about the address: this loop prints it and holds, because a signature is
   the one thing that needs the second kind.
2. **What does the market say about it?** ``DiscoveryEngine`` reads DeFiLlama, the two
   gates screen it, and ``VolatilityHunter`` measures each approved pool — ``PANIC``,
   ``CALM``, ``THIN``, ``FORMULA``, ``DISHONEST``, ``UNMEASURED``. Those are the verdicts
   of :mod:`agent.physics.volatility_hunter`, unchanged: this loop **consumes** them and
   does not re-implement a single threshold. How wide the hunt is, is the operator's:
   ``SCAN_LIMIT`` pools read (200), ``MEASURE_LIMIT`` measured (50), ``--limit``/``--hunt``
   over them, and ``--book-report`` to say per entry why nothing was priced.
3. **How much?** The balance is read from the chain *on the same cycle*
   (``provider.balance``) and :func:`agent.trade.capital.size_trade` cuts the amount to the
   smaller of ``MAX_TRADE_SIZE_USD`` and ``MAX_TRADE_FRACTION`` of the wallet — so an $18
   wallet cannot place a $50 trade because a file said it could.
4. **Sign, or say what it would have done?** ``DRY_RUN`` decides. In a dry run the cycle
   computes the amount, prints the sentence it would have printed, records it in the state
   file and **never calls** ``execute_stake``. The decision and the signature are two
   different events, and the second one needs the operator to have written ``DRY_RUN=0``.

What this loop cannot do, and says out loud
===========================================

A DeFiLlama pool is not a vault. The feed publishes a *pool* — a project, a chain, a
UUID — and the operator's book publishes a *contract address*; this file has no way to
prove the second implements the first, and inventing that mapping is the sort of
plausibility that costs money. So a signal is acted on only when the pool and a book
entry share a name (the operator's own declaration, matched on normalised words). A
``PANIC`` with no matching vault is recorded as a **signal**, not an action:

    signal   aave-v3-usdc  [PANIC 0.42] — a measured dislocation, but no vault in the
             operator's book implements this pool: nothing was sized and nothing signed

That line is the honest answer, and on a first run it is usually the answer.

The state file, and why a corrupt one stops the loop
====================================================

Positions are written to ``TRADER_STATE`` (``/data/trader_state.json`` in the container)
after every cycle. A file that cannot be read is **not** treated as an empty book: an
empty book means "no position, so the stake is new", which is how a loop stakes the same
money twice. A corrupt state file stops the process, and the message says which file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from agent.physics.real_testnet_provider import (
    ENV_HISTORY,
    RECORD_FROM_CHAIN,
    YIELD_EVENT_SIGNATURE,
)
from agent.physics.volatility_hunter import PanicSignal
from agent.trade.capital import Sizing, TradeSize, _flag, size_trade

__all__ = [
    "Action",
    "Cycle",
    "DEFAULT_MEASURE_LIMIT",
    "DEFAULT_SCAN_LIMIT",
    "ENV_INTERVAL",
    "ENV_MEASURE",
    "ENV_MARKET",
    "ENV_SCAN",
    "ENV_STATE",
    "EXIT_VERDICTS",
    "ENTRY_VERDICTS",
    "Position",
    "TradeError",
    "Trader",
    "main",
]

ENV_STATE = "TRADER_STATE"
ENV_INTERVAL = "TRADE_INTERVAL"
ENV_MARKET = "TRADE_MARKET"
ENV_SCAN = "SCAN_LIMIT"
ENV_MEASURE = "MEASURE_LIMIT"

#: How wide the hunt is: how many pools the funnel reads, and how many of the approved
#: ones the hunter measures. The first is the feed's own ``POOLS_LIMIT`` (a test holds
#: the two equal, so this number cannot drift away from the funnel it describes); the
#: second is the part a cycle can afford, because every measured pool is another read of
#: its history. Both are the loop's to widen — an operator on a fast box with a feed that
#: answers in a second may hunt two hundred pools a cycle, and one on a slow route may
#: not — which is why they are environment variables and not constants.
DEFAULT_SCAN_LIMIT = 200
DEFAULT_MEASURE_LIMIT = 50

#: The state the loop writes, and the default the container mounts. A relative path
#: because a host run should not need ``/data`` to exist.
DEFAULT_STATE = "data/trader_state.json"
DEFAULT_INTERVAL = 300

#: Verdicts a position may be opened on. ``PANIC`` is the hunt; ``CALM`` and ``GROWTH``
#: are an ordinary day in a pool whose record the two gates approved — carry, not alpha.
#: ``THIN``/``FORMULA``/``DISHONEST``/``UNMEASURED`` are the refusals the hunter already
#: wrote, and this file does not soften any of them.
ENTRY_VERDICTS = frozenset({"PANIC", "CALM", "GROWTH"})

#: Verdicts that make a held position leave: the pool stopped being what it was bought
#: as. ``UNMEASURED`` is deliberately *not* here — "I cannot measure it today" is not
#: "get out", and a feed outage must not be able to liquidate a good position.
EXIT_VERDICTS = frozenset({"FORMULA", "DISHONEST", "THIN"})


class TradeError(RuntimeError):
    """The loop cannot run: a state file it cannot read, or a policy it cannot apply."""


def _normalise(name: object) -> str:
    """A pool or vault name as comparable words: lower case, letters and digits only."""
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _count(value: object, source: str) -> int:
    """A positive count, and the refusal names where it came from.

    ``SCAN_LIMIT=0`` and ``--limit 0`` fail the same way and for the same reason: a scan
    of nothing hunts nothing, and a loop that reads that as "unlimited" would hammer a
    feed until it stopped answering. ``200.0`` is accepted because a ``.env`` file is
    written by hand and writes decimals; a word is not.
    """
    text = str(value if value is not None else "").strip()
    try:
        number = int(float(text))
    except (TypeError, ValueError):
        raise TradeError(
            f"{source}={text!r} is not a count: give a positive integer"
        ) from None
    if number <= 0:
        raise TradeError(f"{source}={number} is not a count: nothing would be scanned")
    return number


def _limit(name: str, default: int, env: Optional[Mapping[str, str]] = None) -> int:
    """``SCAN_LIMIT``/``MEASURE_LIMIT`` if written down, else the loop's own default."""
    source: Mapping[str, str] = os.environ if env is None else env
    raw = str(source.get(name, "")).strip()
    return int(default) if not raw else _count(raw, name)


def _signal_index(report: Any) -> Dict[str, PanicSignal]:
    """The report's signals by normalised pool name — the entry point both the cycle and
    the book report read, so a diagnostic can never disagree with the decision it is
    supposed to explain."""
    index: Dict[str, PanicSignal] = {}
    for signal in getattr(report, "signals", ()) or ():
        key = _normalise(getattr(signal, "name", ""))
        if not key:
            continue
        known = index.get(key)
        if known is None or (
            getattr(signal, "fired", False) and not getattr(known, "fired", False)
        ):
            index[key] = signal
    return index


@dataclass
class Position:
    """One stake this loop put somewhere, as the state file remembers it."""

    address: str
    name: str = ""
    amount: float = 0.0
    at: float = 0.0
    verdict: str = ""
    confidence: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "name": self.name,
            "amount": round(float(self.amount), 6),
            "at": float(self.at),
            "verdict": self.verdict,
            "confidence": round(float(self.confidence), 4),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Position":
        return cls(
            address=str(payload.get("address", "")).lower(),
            name=str(payload.get("name", "")),
            amount=float(payload.get("amount", 0.0) or 0.0),
            at=float(payload.get("at", 0.0) or 0.0),
            verdict=str(payload.get("verdict", "")),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
        )


@dataclass
class Action:
    """One decision of one cycle: what it was, how much, why, and whether it was sent.

    ``kind`` is the *decision*, not the transaction: ``stake`` and ``withdraw`` are
    transactions, ``signal`` is a verdict with nothing behind it in the book, ``hold`` is
    a refusal with a reason, and ``error`` is a thing that broke. ``sent`` is the only
    field that says money moved, and in a dry run it is ``False`` on every row.
    """

    kind: str
    venue: str = ""
    address: str = ""
    amount: float = 0.0
    usd: float = 0.0
    verdict: str = ""
    confidence: float = 0.0
    bound: str = ""
    reason: str = ""
    sent: bool = False
    tx: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "vault": self.venue,
            "address": self.address,
            "amount": round(float(self.amount), 6),
            "usd": round(float(self.usd), 4),
            "verdict": self.verdict,
            "confidence": round(float(self.confidence), 4),
            "bound": self.bound,
            "reason": self.reason,
            "sent": bool(self.sent),
            "tx": self.tx,
        }

    def line(self, dry_run: bool = True) -> str:
        """The operator's one line: the verb, the number, and the sentence behind it."""
        head = f"{self.kind:<8} {self.venue or self.address}"
        if self.kind in {"stake", "withdraw"} and self.amount > 0.0:
            verb = "would " + self.kind if dry_run else self.kind
            head = f"{verb:<12} {self.venue or self.address}  {self.amount:.6f} (${self.usd:.2f})"
        if self.verdict:
            head += f"  [{self.verdict}"
            head += f" {self.confidence:.2f}]" if self.confidence > 0.0 else "]"
        if self.reason:
            head += f" — {self.reason}"
        if self.kind in {"stake", "withdraw"} and not dry_run:
            head += " · signed" if self.sent else " · NOT sent"
        return head


@dataclass
class Cycle:
    """One iteration, as a report: the book, the market, the wallet, and the decisions."""

    number: int = 0
    at: float = 0.0
    dry_run: bool = True
    block: int = 0
    chain_id: int = 0
    book: List[Dict[str, Any]] = field(default_factory=list)
    balances: List[Dict[str, Any]] = field(default_factory=list)
    market: Dict[str, Any] = field(default_factory=dict)
    actions: List[Action] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def sent(self) -> int:
        return sum(1 for action in self.actions if action.sent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": int(self.number),
            "at": float(self.at),
            "dry_run": bool(self.dry_run),
            "chain_id": int(self.chain_id),
            "block": int(self.block),
            "book": list(self.book),
            "balances": list(self.balances),
            "market": dict(self.market),
            "actions": [action.as_dict() for action in self.actions],
            "warnings": list(self.warnings),
            "sent": self.sent,
        }

    def summary_text(self) -> str:
        """The whole cycle as lines: what was read, what was decided, what was sent."""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.at or time.time()))
        mode = "DRY RUN (nothing will be signed)" if self.dry_run else "LIVE"
        lines = [
            f"── cycle {self.number} · {stamp} · {mode} · chain {self.chain_id or '?'} "
            f"· block {self.block or '?'}"
        ]
        if self.book:
            for row in self.book:
                apy = row.get("implied_apy")
                shown = "unmeasured" if apy is None else f"{float(apy):+.2%} implied"
                line = f"   book  {row.get('name', '?'):<24} {shown}"
                source = str(row.get("source") or "")
                if source and source != RECORD_FROM_CHAIN:
                    line += f" · {source}"
                lines.append(line)
        else:
            lines.append("   book  empty — no vault published a record, so nothing is priced")
        if self.market:
            if self.market.get("error"):
                lines.append(f"   market no scan — {self.market['error']}")
            else:
                lines.append(
                    f"   market {self.market.get('scanned', 0)} pools read · "
                    f"panics {self.market.get('fired', 0)} · "
                    f"verdicts {self.market.get('verdicts', {})}"
                )
        for action in self.actions:
            lines.append("   " + action.line(dry_run=self.dry_run))
        for warning in self.warnings:
            lines.append(f"   warn  {warning}")
        return "\n".join(lines)


@dataclass
class Trader:
    """The cycle itself: a venue, a policy, a place to remember, and a market to read.

    Every collaborator is injected, which is what makes this file testable without a
    chain: ``venue`` is anything answering the venue calls a loop needs
    (``get_protocol_list``, ``balance``, ``execute_stake``, ``withdraw``), ``market`` is a
    callable returning a ``PanicReport``, ``now`` is the clock, and ``printer`` is where
    the cycle's lines go. The default of every one of them is the real thing.
    """

    venue: Any
    sizing: Sizing = field(default_factory=Sizing.from_env)
    state_path: Optional[Path] = None
    market: Optional[Callable[[], Any]] = None
    now: Callable[[], float] = time.time
    printer: Callable[[str], None] = print
    cycles: int = 0
    positions: Dict[str, Position] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.state_path, str):
            self.state_path = Path(self.state_path)
        if self.state_path is not None:
            self._load()

    # -- what it remembers --------------------------------------------------- #

    def _load(self) -> None:
        """Read the positions this loop already holds; refuse to guess if it cannot.

        A missing file is a first run. An *unreadable* one is not an empty portfolio: an
        empty portfolio means the next stake is a new one, and that is how the same money
        gets committed twice. So a corrupt file stops the process instead.
        """
        path = self.state_path
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            stored = dict(payload.get("positions", {}) or {})
            self.positions = {
                str(address).lower(): Position.from_dict(
                    {**row, "address": row.get("address", address)}
                )
                for address, row in stored.items()
            }
            self.cycles = int(payload.get("cycles", 0) or 0)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise TradeError(
                f"the state file {path} cannot be read ({type(exc).__name__}: {exc}). "
                "Remove it to start with no positions — this loop will not guess which "
                "stakes it already holds, because guessing means staking them again."
            ) from exc

    def _save(self, cycle: Cycle) -> None:
        """Write the positions and the last cycle. The file is the loop's heartbeat."""
        path = self.state_path
        if path is None:
            return
        payload = {
            "version": 1,
            "cycles": int(self.cycles),
            "updated": float(self.now()),
            "dry_run": bool(self.sizing.dry_run),
            "positions": {
                address: position.as_dict() for address, position in self.positions.items()
            },
            "last": cycle.to_dict(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            # A live loop that cannot write its positions would stake the same money
            # again on the next cycle, so this is fatal in live mode and a warning in a
            # dry run — where the file is a report and nothing is at risk.
            message = f"the state file {path} could not be written: {exc}"
            if not self.sizing.dry_run:
                raise TradeError(
                    message + " — a live loop that cannot remember a position would "
                    "stake it again on the next cycle, so it stops instead"
                ) from exc
            cycle.warnings.append(message + " (dry run: nothing is at risk)")

    def describe(self) -> Dict[str, Any]:
        return {
            "venue": str(getattr(self.venue, "name", "venue")),
            "sizing": self.sizing.describe(),
            "state": None if self.state_path is None else str(self.state_path),
            "cycles": int(self.cycles),
            "positions": [position.as_dict() for position in self.positions.values()],
        }

    # -- what it reads ------------------------------------------------------- #

    def _book(self, cycle: Cycle) -> List[Any]:
        """The vaults that have a record — the only things that can be priced."""
        read = getattr(self.venue, "get_protocol_list", None)
        if read is None:  # pragma: no cover - every venue has it; a stub might not
            cycle.warnings.append("the venue answers no get_protocol_list: nothing to price")
            return []
        try:
            protocols = list(read())
        except Exception as exc:  # noqa: BLE001 - a cycle reports, it does not crash
            cycle.warnings.append(
                f"the book could not be read ({type(exc).__name__}: {exc})"
            )
            return []
        if not protocols:
            self._empty_book(cycle)
        for protocol in protocols:
            apy = getattr(protocol, "implied_apy", None)
            record = getattr(protocol, "returns", None)
            try:
                points = int(len(record))
            except TypeError:  # a venue that publishes no record at all
                points = 0
            address = str(getattr(protocol, "address", ""))
            cycle.book.append(
                {
                    "name": str(getattr(protocol, "name", "")),
                    "address": address,
                    "points": points,
                    "claim_apy": float(getattr(protocol, "claim_apy", 0.0) or 0.0),
                    "implied_apy": None if apy is None else round(float(apy), 6),
                    "source": self._record_source(address),
                }
            )
        return protocols

    def _record_source(self, address: str) -> str:
        """Where this address's record came from, as the venue says it — ``""`` if it can't.

        ``chain`` is a record read from the address itself; anything else is somebody's
        statement about a *name* (the feed's label), and the two are never the same thing.
        A venue that does not answer the question yields ``""``, which is honest: this file
        will not guess provenance, it will only refuse to sign without it.
        """
        return _source_of(self.venue, address)

    def _empty_book(self, cycle: Cycle) -> None:
        """Why nothing was priced: no entry at all, or entries that published nothing.

        "book empty" on its own sends an operator looking for a network fault. The two
        causes are different and only one of them is a fault: an empty book is a
        configuration that says "trade nothing", and a book whose vaults published no
        ``YieldDistributed`` log is usually a book of the wrong kind of address. Naming
        which one this is costs one log line and saves an afternoon.
        """
        config = getattr(self.venue, "config", None)
        book = getattr(config, "book", None)
        if book is None:  # a stub, or a venue without a book: run_once's line is enough
            return
        entries = list(book)
        if not entries:
            cycle.warnings.append(
                "the operator's book is empty — no address is listed, so nothing can be "
                "priced and nothing staked (see book.example.json)"
            )
            return
        lookback = int(getattr(config, "lookback", 0) or 0)
        for entry in entries:
            cycle.warnings.append(
                f"{getattr(entry, 'name', '?')} ({getattr(entry, 'address', '?')}): no "
                f"{YIELD_EVENT_SIGNATURE} log in the last {lookback} blocks — an entry is "
                "a vault of this interface, not a lending pool's share token"
            )
        # The line that turns a diagnosis into an instruction. A venue with a policy of
        # `chain` has not asked the feed at all, and that is a decision the operator can
        # reverse with one variable — so the warning says which variable.
        policy = str(getattr(config, "history", "") or "")
        if policy in {"", RECORD_FROM_CHAIN}:
            cycle.warnings.append(
                f"the feed was not asked: {ENV_HISTORY}=chain prices only a vault's "
                f"own log — set {ENV_HISTORY}=auto to price these entries from "
                "DeFiLlama's history of their names, or =feed to ask only the feed"
            )
        else:
            cycle.warnings.append(
                f"{ENV_HISTORY}={policy} was asked and named none of these entries "
                "either: a name has to be spelled like the pool you mean (see --book-report)"
            )

    def _market(self, cycle: Cycle) -> Any:
        """Read the wild market once; a dead feed is a warning, never an exception."""
        if self.market is None:
            cycle.warnings.append(
                "the market arm is off: only the operator's book was read, so no pool "
                "was judged and no signal can fire"
            )
            return None
        try:
            report = self.market()
        except Exception as exc:  # noqa: BLE001 - the loop outlives the feed
            cycle.warnings.append(
                f"the market arm failed ({type(exc).__name__}: {exc})"
            )
            return None
        if report is None:
            return None
        payload = report.as_dict()
        cycle.market = {
            "venue": payload.get("venue", ""),
            "scanned": payload.get("scanned", 0),
            "measured": payload.get("measured", 0),
            "fired": payload.get("fired", 0),
            "verdicts": payload.get("verdicts", {}),
            "error": payload.get("error", ""),
        }
        for warning in list(payload.get("warnings", []))[:3]:
            cycle.warnings.append(str(warning))
        return report

    @staticmethod
    def _index(report: Any) -> Dict[str, PanicSignal]:
        """The report's signals by normalised pool name — a ``PANIC`` wins a name clash.

        The hunter returns one signal per approved pool, so a name is normally unique.
        When two pools normalise to the same words, the one that fired is the one that
        matters to a book: it is the only verdict that sizes a trade.
        """
        return _signal_index(report)

    def _balance(self, address: str, cycle: Cycle) -> Any:
        """The wallet, read now. ``None`` means the loop cannot see it and will not size."""
        read = getattr(self.venue, "balance", None)
        if read is None:  # pragma: no cover - the provider has it; a stub might not
            cycle.warnings.append("the venue cannot read a balance: nothing can be sized")
            return None
        try:
            balance = read(address)
        except Exception as exc:  # noqa: BLE001 - a refused read is a refused trade
            cycle.warnings.append(
                f"the wallet could not be read ({type(exc).__name__}: {exc}): "
                "no size is computed without one"
            )
            return None
        cycle.balances.append(balance.as_dict())
        return balance

    # -- what it decides ----------------------------------------------------- #

    def _consider(
        self, protocol: Any, index: Dict[str, PanicSignal], cycle: Cycle
    ) -> Action:
        """One vault: does the market say buy it, and can the wallet afford a fifth?"""
        name = str(getattr(protocol, "name", ""))
        address = str(getattr(protocol, "address", "")).lower()
        signal = index.get(_normalise(name))
        if signal is None:
            return Action(
                kind="hold",
                venue=name,
                address=address,
                reason=(
                    "the market has no pool named like this vault — a signal about a pool "
                    "is not a signal about this address, and this loop does not guess which "
                    "contract implements which pool"
                ),
            )

        verdict = str(getattr(signal, "verdict", ""))
        confidence = float(getattr(signal, "confidence", 0.0) or 0.0)
        reason = str(getattr(signal, "reason", ""))
        held = self.positions.get(address)

        if verdict not in ENTRY_VERDICTS:
            return Action(
                kind="hold",
                venue=name,
                address=address,
                verdict=verdict,
                confidence=confidence,
                reason=reason or "the hunter refused this pool",
            )
        # A record the feed vouched for is a fact about a *name*. The verdict on it is real
        # and worth printing — and it is not evidence that this address implements
        # `stake(uint256)`, which is the one thing a signature would need to be true.
        source = self._record_source(address)
        if source and source != RECORD_FROM_CHAIN:
            return Action(
                kind="hold",
                venue=name,
                address=address,
                verdict=verdict,
                confidence=confidence,
                reason=(
                    f"priced from {source}, not from this address's own log: the verdict is "
                    "about the pool's name, and this loop signs only into a vault whose "
                    "record it has read itself"
                ),
            )
        if held is not None:
            return Action(
                kind="hold",
                venue=name,
                address=address,
                verdict=verdict,
                confidence=confidence,
                reason=(
                    f"already holding {held.amount:.6f} opened at "
                    f"{held.verdict or 'unknown'}: one position per vault"
                ),
            )

        balance = self._balance(address, cycle)
        if balance is None:
            return Action(
                kind="hold",
                venue=name,
                address=address,
                verdict=verdict,
                confidence=confidence,
                reason="the wallet could not be read on this cycle",
            )
        if not balance.ok:
            return Action(
                kind="hold",
                venue=name,
                address=address,
                verdict=verdict,
                confidence=confidence,
                reason=f"the wallet could not be read — {balance.error}",
            )

        size = size_trade(
            token_balance=balance.token_balance,
            native_eth=balance.native_eth,
            sizing=self.sizing,
            confidence=confidence if verdict == "PANIC" else 0.0,
        )
        if not size.ok:
            return Action(
                kind="hold",
                venue=name,
                address=address,
                verdict=verdict,
                confidence=confidence,
                bound=size.bound,
                reason=size.refusal,
            )
        return self._stake(name, address, size, verdict, confidence)

    def _stake(
        self,
        name: str,
        address: str,
        size: TradeSize,
        verdict: str,
        confidence: float,
    ) -> Action:
        """Send the stake — or, in a dry run, produce the row and touch nothing.

        The dry-run branch returns *before* the venue is called at all, which is the only
        version of this guarantee a reader can check: there is no flag inside
        ``execute_stake`` that could be wrong, because in a dry run the call never happens.
        """
        action = Action(
            kind="stake",
            venue=name,
            address=address,
            amount=float(size.amount),
            usd=float(size.usd),
            verdict=verdict,
            confidence=confidence,
            bound=size.bound,
            reason=size.reason,
            sent=False,
        )
        if self.sizing.dry_run:
            return action
        try:
            ok = bool(self.venue.execute_stake(address, size.amount))
        except Exception as exc:  # noqa: BLE001 - a failed send is a row, not a crash
            action.kind = "error"
            action.reason = f"the stake raised {type(exc).__name__}: {exc}"
            return action
        action.sent = ok
        if ok:
            self.positions[address] = Position(
                address=address,
                name=name,
                amount=float(size.amount),
                at=float(self.now()),
                verdict=verdict,
                confidence=confidence,
            )
        else:
            action.reason = self._refusal_line() or "the venue refused the stake"
        return action

    def _refusal_line(self) -> str:
        """The venue's own reason for the last refusal, when it keeps one."""
        entries = getattr(self.venue, "refusals", None) or []
        if not entries:
            return ""
        last = entries[-1]
        if isinstance(last, Mapping):
            return str(last.get("reason", "") or "")
        return str(last)

    def _revisit(
        self, protocols: List[Any], index: Dict[str, PanicSignal], cycle: Cycle
    ) -> None:
        """Positions the market has since disowned: leave, with the verdict as the reason.

        Only the verdicts that *convict* a pool close a position. ``UNMEASURED`` does not:
        a feed that stopped publishing is not evidence against a record, and a loop that
        liquidates on its own blindness is a loop that sells at the bottom of an outage.
        """
        published = {
            str(getattr(protocol, "address", "")).lower() for protocol in protocols
        }
        for address, position in list(self.positions.items()):
            signal = index.get(_normalise(position.name))
            if signal is not None:
                verdict = str(getattr(signal, "verdict", ""))
                if verdict in EXIT_VERDICTS:
                    cycle.actions.append(
                        self._exit(position, verdict, str(getattr(signal, "reason", "")))
                    )
                    continue
            if address not in published:
                cycle.warnings.append(
                    f"{position.name}: still holding {position.amount:.6f}, but the vault "
                    "published no record this cycle — the position is kept, not guessed at"
                )

    def _exit(self, position: Position, verdict: str, reason: str) -> Action:
        """Close a position — the same dry-run rule as a stake, in the other direction."""
        action = Action(
            kind="withdraw",
            venue=position.name,
            address=position.address,
            amount=float(position.amount),
            usd=float(position.amount),
            verdict=verdict,
            confidence=float(position.confidence),
            reason=reason or "the pool is no longer what it was bought as",
            sent=False,
        )
        if self.sizing.dry_run:
            return action
        try:
            ok = bool(self.venue.withdraw(position.address, position.amount))
        except Exception as exc:  # noqa: BLE001 - same rule as the stake
            action.kind = "error"
            action.reason = f"the withdrawal raised {type(exc).__name__}: {exc}"
            return action
        action.sent = ok
        if ok:
            self.positions.pop(position.address, None)
        else:
            action.reason = self._refusal_line() or "the venue refused the withdrawal"
        return action

    def _unclaimed(
        self, protocols: List[Any], index: Dict[str, PanicSignal], cycle: Cycle
    ) -> None:
        """Panics the book cannot express: reported as **signals**, never sized.

        A dislocation the operator has no vault for is the most interesting thing this
        loop can see and the one thing it must not trade. No amount is computed, because
        without a vault there is no token to stake and a dollar figure would be a
        number about nothing — the row carries the verdict, the confidence and the
        sentence that says why nothing happened.
        """
        claimed = {_normalise(str(getattr(protocol, "name", ""))) for protocol in protocols}
        for key, signal in index.items():
            if key in claimed or not getattr(signal, "fired", False):
                continue
            cycle.actions.append(
                Action(
                    kind="signal",
                    venue=str(getattr(signal, "name", "")),
                    address=str(getattr(signal, "address", "")),
                    verdict=str(getattr(signal, "verdict", "")),
                    confidence=float(getattr(signal, "confidence", 0.0) or 0.0),
                    reason=(
                        "a measured dislocation, but no vault in the operator's book "
                        "implements this pool: nothing was sized and nothing was signed"
                    ),
                )
            )

    # -- one cycle, and then another ----------------------------------------- #

    def run_once(self) -> Cycle:
        """Read, judge, size, act — once. Never raises for a venue, a feed or a vault."""
        cycle = Cycle(
            number=int(self.cycles) + 1,
            at=float(self.now()),
            dry_run=bool(self.sizing.dry_run),
            chain_id=int(getattr(self.venue, "chain_id", 0) or 0),
            block=int(getattr(self.venue, "block_number", 0) or 0),
        )
        report = self._market(cycle)
        index = self._index(report)
        protocols = self._book(cycle)
        if not protocols and not cycle.warnings:
            cycle.warnings.append(
                "no vault in the operator's book has published a record: there is nothing "
                "to price and nothing to stake"
            )
        for protocol in protocols:
            cycle.actions.append(self._consider(protocol, index, cycle))
        self._unclaimed(protocols, index, cycle)
        self._revisit(protocols, index, cycle)
        self.cycles = cycle.number
        self._save(cycle)
        return cycle

    def run(
        self,
        *,
        cycles: Optional[int] = None,
        interval: float = DEFAULT_INTERVAL,
    ) -> int:
        """Cycle forever (``cycles=None``) or a bounded number of times; returns the count."""
        done = 0
        while cycles is None or done < int(cycles):
            cycle = self.run_once()
            self.printer(cycle.summary_text())
            done += 1
            if cycles is not None and done >= int(cycles):
                break
            try:
                time.sleep(max(float(interval), 0.0))
            except KeyboardInterrupt:  # pragma: no cover - a terminal signal
                self.printer("interrupted: the state file holds this loop's positions")
                break
        return done


# ── the collaborators, built from the environment ───────────────────────────


def _market_arm(mode: str, limit: int, hunt: int) -> Optional[Callable[[], Any]]:
    """The market arm as a callable: DeFiLlama → two gates → :class:`VolatilityHunter`.

    The composition is not written here — it is the one ``agent.physics.volatility_hunter``
    already ships in its own CLI (``DiscoveryEngine.run`` then ``VolatilityHunter.scan``),
    so a verdict this loop acts on is the same verdict that CLI prints. ``synthetic``
    reads the drawn market instead of the feed, which is how a box with no egress can
    still exercise the whole path.
    """
    name = str(mode or "").strip().lower()
    if name in {"off", "none", "no", "0", ""}:
        return None
    from agent.physics.defillama_adapter import (
        MODE_REAL,
        MODE_SYNTHETIC,
        DiscoveryEngine,
    )
    from agent.physics.volatility_hunter import VolatilityHunter

    engine = DiscoveryEngine(mode=MODE_SYNTHETIC if name == "synthetic" else MODE_REAL)
    hunter = VolatilityHunter(scanner=engine.scanner)

    def once() -> Any:
        funnel = engine.run(int(limit))
        return hunter.scan(funnel, limit=int(hunt))

    return once


def _book_report(
    venue: Any,
    market: Optional[Callable[[], Any]],
    printer: Callable[[str], None] = print,
) -> int:
    """``--book-report``: for every address in the book, the three things that must be
    true before a cycle can act — does it publish a record, does the market name it, and
    is it a vault at all.

    "book empty" sends an operator looking for a broken node. The cause is usually one
    address: a token that is not a vault, or a vault whose record predates the lookback
    window. This mode says which, per entry, and exits ``4`` when nothing can be priced —
    the same code the hunter uses for "there is nothing to hunt in", so a shell script
    can tell an empty market from a broken one.
    """
    config = getattr(venue, "config", None)
    book = getattr(config, "book", None)
    entries = list(book) if book is not None else []
    lookback = int(getattr(config, "lookback", 0) or 0)
    policy = str(getattr(config, "history", "") or "(unset → chain)")
    chain = getattr(venue, "chain_id", 0) or "?"
    printer(
        f"book report — {len(entries)} entry/entries · lookback {lookback} blocks · "
        f"chain {chain} · {ENV_HISTORY}={policy}"
    )
    if not entries:
        printer("  empty — no address is listed, so nothing can be priced or staked")
        printer(
            '  an entry is {"address": "0x…", "name": "USDC", "token": "0x…", '
            '"decimals": 6} — see book.example.json'
        )
        return 4

    index: Dict[str, PanicSignal] = {}
    if market is not None:
        try:
            index = _signal_index(market())
        except Exception as exc:  # noqa: BLE001 - a report is a report, not a crash
            printer(
                f"  market unreadable ({type(exc).__name__}: {exc}): only the chain is "
                "reported"
            )

    published = 0
    priced_from_feed = 0
    for entry in entries:
        name = str(getattr(entry, "name", ""))
        address = str(getattr(entry, "address", ""))
        records = 0
        note = ""
        read = getattr(venue, "fetch_protocol_history", None)
        if read is None:  # pragma: no cover - a stub might not have it
            note = "the venue cannot read a record"
        else:
            try:
                records = len(list(read(address)))
            except Exception as exc:  # noqa: BLE001 - the line is the report
                note = f"the record could not be read ({type(exc).__name__}: {exc})"
        published += 1 if records else 0
        source = _source_of(venue, address)
        foreign = bool(source) and source != RECORD_FROM_CHAIN
        priced_from_feed += 1 if foreign else 0
        signal = index.get(_normalise(name))
        if market is None:
            verdict = "market off"
        elif signal is not None:
            verdict = (
                f"{getattr(signal, 'verdict', '?')} "
                f"{float(getattr(signal, 'confidence', 0.0) or 0.0):.2f}"
            )
        else:
            verdict = "no pool in the report is named like this vault"
        if records:
            where = f" · {source}" if source else ""
            record = f"{records} record(s){where}"
        else:
            why = note or _note_of(venue, address)
            record = "no record" + (f" — {why}" if why else "")
        printer(f"  {name:<24} {address}  {record}")
        if foreign:
            printer(
                f"  {'':<24} not stakable as-is: the record above is the feed's about this "
                "name, and this loop signs only into a vault whose own log it has read"
            )
        printer(f"  {'':<24} market: {verdict}")
    if not published:
        if policy in {"", "(unset → chain)", RECORD_FROM_CHAIN}:
            printer(
                f"  nothing to price: an entry has to emit {YIELD_EVENT_SIGNATURE} "
                "(and a cycle has to see it inside the lookback window) before the hunter "
                "can measure it"
            )
            printer(
                f"  the feed was not asked — {ENV_HISTORY}=auto prices an entry from "
                "DeFiLlama's daily history of its name when the vault itself published "
                "nothing, and =feed asks only the feed"
            )
        else:
            printer(
                "  nothing to price: no entry has a record — the feed named no pool for "
                "these names (or published no series for the pool they pin), and no vault "
                f"emitted {YIELD_EVENT_SIGNATURE} inside the lookback window"
            )
    elif priced_from_feed:
        printer(
            f"  {priced_from_feed} of {published} record(s) came from the feed, not from an "
            "address's own log: measured, and deliberately not staked"
        )
    return 0 if published else 4


def _source_of(venue: Any, address: str) -> str:
    """Where the venue says this address's record came from, or ``""`` if it cannot say."""
    read = getattr(venue, "history_source", None)
    if read is None:
        return ""
    try:
        return str(read(address) or "")
    except Exception:  # noqa: BLE001 - a report line is not worth an exception
        return ""


def _note_of(venue: Any, address: str) -> str:
    """The venue's own sentence about why an entry has no record."""
    read = getattr(venue, "history_note", None)
    if read is None:
        return ""
    try:
        return str(read(address) or "")
    except Exception:  # noqa: BLE001 - same rule as above
        return ""


def _venue_from_env() -> Any:
    """The real venue. An L2 is a mainnet here, so the two-key lock still applies."""
    from agent.physics.real_testnet_provider import RealTestnetProvider

    return RealTestnetProvider.from_env(
        allow_mainnet=_flag(os.environ.get("TESTNET_ALLOW_MAINNET"))
    )


def _parser() -> "argparse.ArgumentParser":
    parser = argparse.ArgumentParser(
        prog="python -m agent.trade.loop",
        description=(
            "The ENRG-AI trading cycle: the operator's vault book, the DeFiLlama market, "
            "the hunter's verdict, a size cut to a fifth of the wallet, and a tick — or, "
            "with DRY_RUN on, the tick that would have been sent."
        ),
    )
    parser.add_argument(
        "--once", action="store_true", help="one cycle and exit (the container default)"
    )
    parser.add_argument(
        "--loop", action="store_true", help="cycle until stopped (with --interval)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="a bounded number of cycles, for a rehearsal",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get(ENV_INTERVAL) or DEFAULT_INTERVAL),
        help=f"seconds between cycles (default {DEFAULT_INTERVAL}, env {ENV_INTERVAL})",
    )
    parser.add_argument(
        "--market",
        choices=["real", "synthetic", "off"],
        default=str(os.environ.get(ENV_MARKET) or "real").lower(),
        help=f"which market to read (default real, env {ENV_MARKET})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            f"pools the funnel scans (default {DEFAULT_SCAN_LIMIT}, env {ENV_SCAN})"
        ),
    )
    parser.add_argument(
        "--hunt",
        type=int,
        default=None,
        help=(
            "approved pools to measure "
            f"(default {DEFAULT_MEASURE_LIMIT}, env {ENV_MEASURE})"
        ),
    )
    parser.add_argument(
        "--book-report",
        dest="book_report",
        action="store_true",
        help="say per book entry why it is or is not priced, and exit (4 = nothing to price)",
    )
    parser.add_argument(
        "--state",
        default=os.environ.get(ENV_STATE) or DEFAULT_STATE,
        help=f"where positions are remembered (default {DEFAULT_STATE}, env {ENV_STATE})",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=None,
        help="compute and log, sign nothing (the default unless DRY_RUN=0)",
    )
    parser.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
        default=None,
        help="sign and send — the same as DRY_RUN=0, said on the command line",
    )
    parser.add_argument("--max-usd", type=float, default=None, help="MAX_TRADE_SIZE_USD")
    parser.add_argument(
        "--max-fraction", type=float, default=None, help="MAX_TRADE_FRACTION"
    )
    parser.add_argument("--eth-price", type=float, default=None, help="ETH_USD_PRICE")
    parser.add_argument("--json", default="", help="write this cycle's report here")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m agent.trade.loop --loop --interval 300`` — the container's entry point."""
    args = _parser().parse_args(argv)

    try:
        # The two widths of the hunt: the operator's `.env` first, the flag over it.
        scan_limit = (
            _limit(ENV_SCAN, DEFAULT_SCAN_LIMIT)
            if args.limit is None
            else _count(args.limit, "--limit")
        )
        measure_limit = (
            _limit(ENV_MEASURE, DEFAULT_MEASURE_LIMIT)
            if args.hunt is None
            else _count(args.hunt, "--hunt")
        )
        sizing = Sizing.from_env()
        overrides: Dict[str, Any] = {}
        if args.max_usd is not None:
            overrides["max_trade_usd"] = float(args.max_usd)
        if args.max_fraction is not None:
            overrides["max_fraction"] = float(args.max_fraction)
        if args.eth_price is not None:
            overrides["eth_price_usd"] = float(args.eth_price)
        if args.dry_run is not None:
            overrides["dry_run"] = bool(args.dry_run)
        if overrides:
            sizing = sizing.replace(**overrides)
    except Exception as exc:  # noqa: BLE001 - a configuration error is one sentence
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        venue = _venue_from_env()
    except Exception as exc:  # noqa: BLE001 - a venue that cannot be configured
        print(f"error: the venue cannot be configured — {exc}", file=sys.stderr)
        return 2

    if args.book_report:
        # Before the trader is built: a report about the book must not depend on a state
        # file, a policy or a cycle.
        return _book_report(venue, _market_arm(args.market, scan_limit, measure_limit))

    try:
        trader = Trader(
            venue=venue,
            sizing=sizing,
            state_path=Path(args.state) if args.state else None,
            market=_market_arm(args.market, scan_limit, measure_limit),
        )
    except TradeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"ENRG-AI trader — {sizing}")
    print(f"  venue  {getattr(venue, 'name', 'venue')}")
    print(f"  market {args.market} (scan {scan_limit}, measure {measure_limit})")
    print(f"  state  {trader.state_path}")
    if trader.positions:
        print(f"  held   {len(trader.positions)} position(s) from earlier cycles")

    bounded = args.cycles if args.cycles is not None else (1 if args.once else None)
    try:
        trader.run(cycles=bounded, interval=args.interval)
    except TradeError as exc:  # pragma: no cover - raised before the first cycle
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json and trader.state_path is not None:
        payload: Dict[str, Any] = {}
        try:
            payload = json.loads(trader.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - the file was just written
            payload = {}
        try:
            Path(args.json).write_text(
                json.dumps(payload.get("last", {}), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"warning: {args.json} could not be written: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - the module is normally imported
    raise SystemExit(main())
