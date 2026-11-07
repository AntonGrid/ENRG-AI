"""First contact: a venue that is a real chain, behind the same five operations.

Everything above this file was written against
:class:`agent.physics.interfaces.IWeb3Provider` and has never seen a network. This
module is the other end of that seam: it answers the same five operations by talking to
an EVM node, so ``Portfolio.deposit``, ``Empire.review`` and the kill switch run
unchanged while the credits they move are real token transfers on Sepolia — or on a
local Anvil, which is how this file is tested.

The claim worth testing is not "it compiles against web3.py". It is that the *shape* of
the interface was right: nothing in ``portfolio.py`` or ``risk_assessment.py`` needed a
line changed for this to work, and every failure a real venue can produce — a revert, a
missing allowance, a gas estimate that refuses, a node that stops answering — arrives at
the book as either ``False`` or :class:`~agent.physics.portfolio.StakeRefused`, which
is what those files already knew how to handle.

What this file is not
=====================

It is not a bot. There is no loop here, no scheduler, no retry storm: one call to
``execute_stake`` is one transaction, signed locally, awaited to a receipt, and
recorded. It holds a key for the duration of a call and never writes it anywhere —
``repr`` of the configuration is masked, ``describe()`` prints the host of the RPC
endpoint and never the endpoint, and a test parses this file to prove that no literal
key, URL or address is baked into it.

It also refuses to sign on a chain where credits are money. :data:`MAINNET_CHAIN_IDS`
is a declared blocklist, and connecting to one raises
:class:`TestnetNotConfigured` unless an operator says ``TESTNET_ALLOW_MAINNET=1`` *and*
passes ``allow_mainnet=True``: two independent decisions, because the mistake this
guards against is expensive and one keystroke away.

One read the interface does not declare
=======================================

:meth:`RealTestnetProvider.balance` is the only method here that
:class:`~agent.physics.interfaces.IWeb3Provider` does not ask for, and it is here for one
caller: a trading loop that has to size a trade against a wallet it can see. It reads
``eth_getBalance`` always and ``balanceOf`` of one vault's token when it is given a vault,
and it answers a failed read with ``Balance(error=...)`` rather than an exception —
because a loop's response to "the node will not say" is to trade nothing this cycle, and a
caller that gets ``0.0`` instead cannot tell that apart from an empty wallet. Two decimals
counts live in one place for the same reason: :data:`ETH_DECIMALS` is 18 and
:data:`DEFAULT_DECIMALS` is 6, and confusing them is how a loop believes it is rich.

Configuration, all of it from the environment
=============================================

Nothing below has a default that could move value. Missing configuration is an error
with a name, never a silent fallback.

=============================  ==================================================
``TESTNET_RPC_URL``            the node. Required. Host is printed, path is not.
``TESTNET_PRIVATE_KEY``        the signer. Optional: without it the venue is
                               **read-only** and every capital call raises.
``TESTNET_UNLOCKED_ACCOUNT``   an address the *node* signs for (Anvil, Hardhat).
                               Alternative to a key; the key never leaves the box.
``TESTNET_CHAIN_ID``           asserted against the node's own chain id.
``TESTNET_PROFILE``            ``arbitrum``, ``base``, ``polygon`` or ``sepolia``: the
                               chain's *name*, so the names a node provider hands out
                               (``ARBITRUM_RPC_URL``, ``ARBITRUM_PRIVATE_KEY``,
                               ``ARBITRUM_CHAIN_ID``) can answer for the canonical
                               ones and the id need not be written down. A spelling,
                               never a permission — an L2 is still a mainnet here.
``TESTNET_PROTOCOLS_FILE``     path to the operator's address book (JSON).
``TESTNET_PROTOCOLS``          the same book inline, as JSON.
``TESTNET_HISTORY``            where a record comes from: ``chain`` (the vault's own
                               ``YieldDistributed`` logs — the default), ``feed`` (DeFiLlama's
                               daily APY history for the entry's name or pinned ``pool``
                               uuid, no node read), or ``auto`` (the log first, the feed when
                               the log is silent).
``TESTNET_DECIMALS``           credits per unit (default ``6``, a USDC-like token).
``TESTNET_LOOKBACK``           blocks scanned for a record (default ``5000``).
``TESTNET_RECEIPT_TIMEOUT``    seconds to wait for a receipt (default ``120``).
``TESTNET_BLOCK_TIMEOUT``      seconds to wait for a block (default ``30``).
``TESTNET_ALLOW_MAINNET``      ``1`` to let a real-money chain through.
``TESTNET_ABI_FILE``           override the built-in vault ABI (JSON).
=============================  ==================================================

A ``.env`` file is read when ``python-dotenv`` is installed; it is never written, and
``os.environ`` always wins, because a file that can override an exported variable is a
file that can silently redirect a signer.

The address book
================

The whitelist is the operator's file, and a venue that could extend it would be a venue
that can widen its own permissions — so this one only reads it::

    {
      "protocols": [
        {
          "address": "0x...",        # the vault this venue publishes
          "name": "TestVaultA",
          "bucket": "active",        # safe | active | experimental
          "claim_apy": 0.05,         # what it says, as a fraction
          "decimals": 6,             # credits per token unit
          "token": "0x...",          # the ERC20 the vault pulls from
          "pool": "",                # optional: the DeFiLlama pool uuid this vault is
          "note": "sepolia demo"
        }
      ]
    }

An address not in this file is not published, is not on the whitelist and cannot be
staked into: :meth:`RealTestnetProvider.execute_stake` answers ``False`` for it, which
``Portfolio`` turns into a refusal, which is exactly what a closed list is for.

Reading a record: the vault's own logs, and — when it publishes none — the feed
=================================================================================

The record is what everything above measures, and there are exactly two places it can come
from. ``chain`` (the default) reads ``YieldDistributed(uint256,uint256)`` logs of one address
inside ``TESTNET_LOOKBACK`` blocks and decodes the second word of each log's data as a
per-epoch rate scaled by ``1e18``: one static word, decoded by hand, so a record needs no ABI.

That is the only honest source for a vault of this interface, and it is silent for almost
every real address: Aave's ``aUSDC``, a Uniswap V3 pool, a Curve gauge all publish yield in
their own events, and a book of them therefore prices nothing. ``TESTNET_HISTORY=feed`` is
the second source, and it is the market's: DeFiLlama's daily APY history for the entry's
**name** (or for the ``pool`` uuid the operator pinned), converted to per-step rates by the
same ``(1+apy)^(1/365) − 1`` the funnel uses. ``TESTNET_HISTORY=auto`` reads the log first
and asks the feed only when the log says nothing, so an address that publishes its own record
is never replaced by a name match.

Two facts about the fallback, both of them load-bearing:

* **A feed-priced record is about a name, not about an address.**
  :meth:`RealTestnetProvider.history_source` answers ``chain`` for a record read from the
  address itself and the feed's label (``the feed aave-v3 USDC (Arbitrum) [d9fa8e]``)
  otherwise, the protocol's note carries the same sentence, and the trading loop **signs only
  into the first kind** — a vault whose own log it has not read is measured, refused, and
  named, never funded.
* **A node that will not answer is not the same silence as a vault that published nothing.**
  ``auto`` falls back on an empty log, not on an outage: a dead node still surfaces as
  :class:`TestnetUnavailable`, because "DeFiLlama can price this name" is not an answer to
  "the chain you point at is unreachable". An operator who wants the feed regardless sets
  ``TESTNET_HISTORY=feed`` and the node is not asked for a record at all.

The chain's clock, and what a "step" means here
===============================================

The interface's ``step`` is "the frontier of what may be read"; in the simulation it is
an index into an array. A chain has blocks, so ``step`` here is **blocks lived since
this provider first connected** (``block_number - origin``), and ``advance`` waits for
the chain to produce them — bounded, and never by manufacturing a block, because time on
a real venue is not ours to move. A venue that owes no block returns the clock it has.
That keeps ``Empire``'s arithmetic meaningful (one step is one block) without pretending
blocks arrive on demand.

Reading a record
================

``fetch_protocol_history`` asks the node for ``YieldDistributed(uint256,uint256)`` logs
of one address and decodes the second word of each log's data as a per-epoch rate
scaled by ``1e18``. One static word, decoded by hand: a record does not need an ABI, and
a feed that fails to decode a log drops that log rather than the record. Arriving out of
order is normal on a chain, so logs are sorted by ``(blockNumber, logIndex)`` before
anything reads them as a sequence.
"""

from __future__ import annotations

import json
import os
import time
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
from urllib.parse import urlsplit

import numpy as np

from agent.physics import aave_v3
from agent.physics.interfaces import IWeb3Provider
from agent.physics.portfolio import (
    BUCKETS,
    PortfolioError,
    Protocol,
    StakeRefused,
    WhiteList,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, and web3 is optional at runtime
    from web3 import Web3

__all__ = [
    "AAVE_KIND",
    "VAULT_KIND",
    "KINDS",
    "Balance",
    "DryRun",
    "HISTORY_AUTO",
    "HISTORY_CHAIN",
    "HISTORY_FEED",
    "HISTORY_SOURCES",
    "ProtocolBook",
    "ProtocolEntry",
    "ProtocolIdentity",
    "RECORD_FROM_CHAIN",
    "RealTestnetProvider",
    "TestnetConfig",
    "TestnetError",
    "TestnetNotConfigured",
    "TestnetUnavailable",
    "MAINNET_CHAIN_IDS",
    "CHAIN_PROFILES",
    "ETH_DECIMALS",
    "YIELD_EVENT_SIGNATURE",
]

# ── the names of the environment, in one place ───────────────────────────────

ENV_RPC = "TESTNET_RPC_URL"
ENV_KEY = "TESTNET_PRIVATE_KEY"
ENV_UNLOCKED = "TESTNET_UNLOCKED_ACCOUNT"
ENV_CHAIN_ID = "TESTNET_CHAIN_ID"
ENV_BOOK_FILE = "TESTNET_PROTOCOLS_FILE"
ENV_BOOK = "TESTNET_PROTOCOLS"
ENV_DECIMALS = "TESTNET_DECIMALS"
ENV_LOOKBACK = "TESTNET_LOOKBACK"
ENV_RECEIPT_TIMEOUT = "TESTNET_RECEIPT_TIMEOUT"
ENV_BLOCK_TIMEOUT = "TESTNET_BLOCK_TIMEOUT"
ENV_ALLOW_MAINNET = "TESTNET_ALLOW_MAINNET"
ENV_ABI_FILE = "TESTNET_ABI_FILE"
ENV_PROFILE = "TESTNET_PROFILE"
ENV_HISTORY = "TESTNET_HISTORY"

#: Where a record comes from, and the three answers this venue gives.
#:
#: ``chain`` is the vault's own ``YieldDistributed`` logs and nothing else — the default,
#: because a record that *this address* published is the only one this file can read without
#: borrowing somebody's word for it. ``feed`` prices an entry from DeFiLlama's daily APY
#: history for its name and never touches the node to do it: the answer to a book of real
#: dApp addresses (an aToken, a Uniswap pool) that emit our event never.
#: ``auto`` reads the log first and asks the feed when the log says nothing — the address's
#: own record always wins, and the feed is what keeps a silent vault from being unmeasurable.
HISTORY_CHAIN = "chain"
HISTORY_FEED = "feed"
HISTORY_AUTO = "auto"
HISTORY_SOURCES = (HISTORY_CHAIN, HISTORY_FEED, HISTORY_AUTO)
DEFAULT_HISTORY = HISTORY_CHAIN

#: What :meth:`RealTestnetProvider.history_source` calls a record read from the address
#: itself. Every other value it can return names a foreign source, and a caller that signs
#: capital may compare against this one string and nothing else.
RECORD_FROM_CHAIN = HISTORY_CHAIN

#: The chains this venue knows by name, and nothing else about them. A profile is a
#: **spelling plus a chain id**: ``TESTNET_PROFILE=arbitrum`` lets the operator keep the
#: names their node provider gave them (``ARBITRUM_RPC_URL``, ``ARBITRUM_PRIVATE_KEY``,
#: ``ARBITRUM_CHAIN_ID``) and supplies ``42161`` when the id is not written down. It is
#: not a permission: every L2 here is in :data:`MAINNET_CHAIN_IDS`, so signing on it
#: still needs *both* yeses, and a profile never changes a timeout, a decimal count or
#: the whitelist. A bare ``RPC_URL`` / ``PRIVATE_KEY`` is deliberately **not** an alias
#: — a stray variable from another project must not be able to become a signer.
#: ``sepolia`` rides along because the rehearsal space is a chain too: this map is
#: about which network is meant, not about which layer it settles on.
CHAIN_PROFILES: Dict[str, int] = {
    "arbitrum": 42161,  # Arbitrum One: the cheap seat, ~200x cheaper gas than L1
    "base": 8453,  # Base
    "polygon": 137,  # Polygon PoS
    "sepolia": 11155111,  # the testnet this file was written against
}

#: Credits per token unit when nobody says otherwise: a USDC-like token, because a
#: credit is a cent-shaped thing and six decimals is what a test stablecoin has.
DEFAULT_DECIMALS = 6

#: Wei per ether. The native balance is read in wei and reported in ether, and the one
#: thing a trading loop must never do is confuse the two: ``0.008`` read as wei is a
#: wallet that looks empty, and ``8e15`` read as ether is a wallet that looks infinite.
ETH_DECIMALS = 18

#: Blocks scanned when a record is asked for. A chain cannot be asked for "the last N
#: events" without an indexer, so the window is declared and printed rather than
#: guessed: an event older than this is invisible, and the provider says so in
#: :meth:`RealTestnetProvider.describe`.
DEFAULT_LOOKBACK = 5_000

#: Seconds to wait for a receipt before calling the venue unavailable. Long on purpose:
#: a transaction that was accepted and then timed out is a position of unknown size, and
#: the honest response is to fail loudly rather than to send another one.
DEFAULT_RECEIPT_TIMEOUT = 120.0

#: Seconds to wait for the chain to produce a block in :meth:`advance`.
DEFAULT_BLOCK_TIMEOUT = 30.0

#: Seconds between two looks at the chain's height.
BLOCK_POLL = 0.25

#: Chains where the thing being staked is money. Connecting to one is refused unless the
#: operator opts in **twice** (an environment variable and an argument), because this
#: file exists to prove a seam, not to trade.
MAINNET_CHAIN_IDS = frozenset(
    {
        1,  # Ethereum
        10,  # OP Mainnet
        56,  # BNB Smart Chain
        100,  # Gnosis
        137,  # Polygon
        250,  # Fantom
        324,  # zkSync Era
        1101,  # Polygon zkEVM
        8453,  # Base
        42161,  # Arbitrum One
        42220,  # Celo
        43114,  # Avalanche C-Chain
        59144,  # Linea
        534352,  # Scroll
    }
)

#: The one event a record is made of: ``YieldDistributed(epoch, rateE18)``. The rate is
#: the *second* word of the data — a per-epoch fraction scaled by :data:`RATE_SCALE`.
YIELD_EVENT_SIGNATURE = "YieldDistributed(uint256,uint256)"

#: What ``1e18``-scaled means: a rate of ``0.0002`` per epoch arrives as ``2e14``.
RATE_SCALE = 1e18

#: The two kinds of venue a book entry may be. ``vault`` is the interface *this file*
#: wrote: ``stake(uint256)``, ``withdraw(uint256)``, and a rate the address published
#: itself. ``aave-v3`` is somebody else's interface — ``supply(asset, amount, onBehalfOf,
#: referralCode)`` into a Pool that a book cannot choose, because
#: :mod:`agent.physics.aave_v3` holds the only pools this file will sign into and the
#: venue checks the entry against that table by chain id.
#:
#: The distinction is not cosmetic. For a ``vault`` the address *is* the claim: a record
#: read from it proves it implements the interface, and a record from the feed proves
#: nothing. For ``aave-v3`` the address is a line in a table this file maintains, so the
#: proof arrives at load time, and the feed is then only asked what the position *earns* —
#: a question the protocol, not the operator, answers.
VAULT_KIND = "vault"
AAVE_KIND = aave_v3.AAVE_KIND
KINDS = (VAULT_KIND, AAVE_KIND)

#: The vault this provider speaks to. Four calls, and every one of them has an obvious
#: meaning on a chain: money in, money out, evidence, and the claim.
VAULT_ABI: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "stake",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "amount", "type": "uint256"}],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "withdraw",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "amount", "type": "uint256"}],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "submitValidationProof",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "digest", "type": "bytes32"}],
        "outputs": [],
    },
    {
        "type": "function",
        "name": "claimApyE18",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "type": "function",
        "name": "token",
        "stateMutability": "view",
        "inputs": [],
        "outputs": [{"name": "", "type": "address"}],
    },
]

#: The token the vault pulls from. Three calls: what is allowed, allow it, and read a
#: balance — the last one only so a dry run can say what it thinks you have.
ERC20_ABI: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "allowance",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "type": "function",
        "name": "approve",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "type": "function",
        "name": "balanceOf",
        "stateMutability": "view",
        "inputs": [{"name": "owner", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


# ── the errors a venue can produce, named ────────────────────────────────────


class TestnetError(PortfolioError):
    """A testnet venue could not do what it was asked, for a reason worth printing."""


class TestnetNotConfigured(TestnetError):
    """The venue's configuration cannot support the call: no node, no signer, no token.

    Raised instead of guessing. A provider that fell back to a default endpoint or a
    default account would be a provider whose mistakes are transactions.
    """


class TestnetUnavailable(TestnetError):
    """The node did not answer, or answered something that is not a chain."""


# ── the operator's address book ─────────────────────────────────────────────


def _is_address(value: object) -> bool:
    """``0x`` and forty hex digits. No checksum test: the operator's case is theirs."""
    text = str(value or "")
    if len(text) != 42 or not text.startswith("0x"):
        return False
    return all(char in "0123456789abcdefABCDEF" for char in text[2:])


@dataclass(frozen=True)
class ProtocolEntry:
    """One line of the operator's book: an address, a name, and what it is allowed."""

    address: str
    name: str
    bucket: str = "active"
    claim_apy: float = 0.0
    decimals: int = DEFAULT_DECIMALS
    token: str = ""
    note: str = ""
    #: The DeFiLlama ``pool`` uuid this vault implements, when the operator knows it. It
    #: removes the name match from the feed fallback entirely — the entry says which pool
    #: prices it instead of hoping a label resolves — and it is optional because most
    #: operators meet a pool through its label first.
    pool: str = ""
    #: How money actually moves into this address. ``vault`` (the default) means the
    #: interface this file wrote — ``stake(uint256)``. ``aave-v3`` means an Aave Pool, whose
    #: calls are ``supply``/``withdraw`` and whose whitelist is the deployment table in
    #: :mod:`agent.physics.aave_v3` rather than this file. An entry cannot become an Aave
    #: pool by saying so: the venue asks that table, with the chain id the node reported,
    #: and refuses a pair it does not know.
    kind: str = VAULT_KIND
    #: How the entry is recognised in the feed's own words, when the operator knows them.
    #: ``name`` is what a *report* prints; these three are what a *match* needs, and they
    #: exist because a label is not a key: DeFiLlama calls the row ``USDC`` under project
    #: ``aave-v3`` on chain ``Arbitrum``, and a book that spelled the label
    #: ``aave-v3 USDC (Arbitrum)`` would otherwise be a stranger to its own pool. A book
    #: entry may therefore declare the identity instead of hoping a string resolves — and
    #: for ``kind: aave-v3`` the deployment table is asked, so a declaration that
    #: contradicts it (a different asset, another chain) is refused before anything is
    #: signed. All three are optional: empty means "match nothing here".
    project: str = ""
    symbol: str = ""
    chain: str = ""

    def __post_init__(self) -> None:
        if not _is_address(self.address):
            raise TestnetNotConfigured(f"{self.address!r} is not a 20-byte address")
        if not str(self.name).strip():
            raise TestnetNotConfigured(f"{self.address}: an entry with no name")
        if self.bucket not in BUCKETS:
            raise TestnetNotConfigured(
                f"{self.name}: bucket must be one of {BUCKETS}, got {self.bucket!r}"
            )
        if not (0 <= int(self.decimals) <= 18):
            raise TestnetNotConfigured(
                f"{self.name}: {self.decimals!r} decimals is not a token"
            )
        if self.token and not _is_address(self.token):
            raise TestnetNotConfigured(
                f"{self.name}: token {self.token!r} is not a 20-byte address"
            )
        if not np.isfinite(float(self.claim_apy)):
            raise TestnetNotConfigured(f"{self.name}: the claim is not a number")
        pool = str(self.pool or "").strip()
        if len(pool) > 64 or any(character.isspace() for character in pool):
            raise TestnetNotConfigured(f"{self.name}: {pool!r} is not a pool uuid")
        kind = str(self.kind or VAULT_KIND).strip().lower()
        if kind not in KINDS:
            raise TestnetNotConfigured(
                f"{self.name}: kind must be one of {KINDS}, got {self.kind!r}"
            )
        if kind == AAVE_KIND and not _is_address(self.token):
            raise TestnetNotConfigured(
                f"{self.name}: an {AAVE_KIND} entry must name the asset it supplies in "
                "'token' — an Aave supply takes the asset as its first argument, and "
                "guessing it would supply the wrong coin"
            )
        for label in ("project", "symbol", "chain"):
            text = str(getattr(self, label) or "").strip()
            if len(text) > 32 or any(character in "\n\r\t" for character in text):
                raise TestnetNotConfigured(
                    f"{self.name}: {label} {text!r} is not a label (an empty string is "
                    "the way to say 'do not match on this')"
                )
            object.__setattr__(self, label, text)
        object.__setattr__(self, "address", str(self.address).lower())
        object.__setattr__(self, "token", str(self.token).lower())
        object.__setattr__(self, "claim_apy", float(self.claim_apy))
        object.__setattr__(self, "decimals", int(self.decimals))
        object.__setattr__(self, "pool", pool)
        object.__setattr__(self, "kind", kind)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "name": self.name,
            "bucket": self.bucket,
            "claim_apy": self.claim_apy,
            "decimals": self.decimals,
            "token": self.token,
            "note": self.note,
            "pool": self.pool,
            "kind": self.kind,
            "project": self.project,
            "symbol": self.symbol,
            "chain": self.chain,
        }


@dataclass(frozen=True)
class ProtocolIdentity:
    """How a book entry is recognised in the feed's report — and what a sentence calls it.

    The trading loop matches a hunter verdict to a book entry, and a *name* is the weakest
    possible key for that: DeFiLlama names the row ``USDC`` under project ``aave-v3`` on
    chain ``Arbitrum``, while the book needs a label a human can read, so the two spellings
    drift apart and a real pool becomes "no pool in the report is named like this vault".

    This value is the answer that does not depend on spelling: the project, the symbol, the
    chain, and the pinned pool uuid, read from the operator's *declaration* in the book and,
    for an ``aave-v3`` entry, completed from the deployment table. It is deliberately not a
    permission — it says how to *find* the verdict, never that the entry may be signed into;
    that is :meth:`RealTestnetProvider.protocol_kind` plus the deployment table's job.
    """

    kind: str = ""
    project: str = ""
    symbol: str = ""
    chain: str = ""
    pool: str = ""
    #: What the sentence calls the protocol (``Aave``), when the venue knows a name for it.
    protocol: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "project": self.project,
            "symbol": self.symbol,
            "chain": self.chain,
            "pool": self.pool,
            "protocol": self.protocol,
        }


@dataclass(frozen=True)
class ProtocolBook:
    """The addresses this venue is allowed to publish, and nothing else.

    Fail-closed in the same sense as :class:`~agent.physics.portfolio.WhiteList`: an
    empty book publishes nothing, and a book is only ever *read* — there is no method
    here that adds an address, because a venue that can widen its own list is a venue
    whose list means nothing.
    """

    entries: Tuple[ProtocolEntry, ...] = ()

    def __post_init__(self) -> None:
        seen: Dict[str, ProtocolEntry] = {}
        for entry in self.entries:
            if entry.address in seen:
                raise TestnetNotConfigured(
                    f"two entries for {entry.address}: "
                    f"{seen[entry.address].name!r} and {entry.name!r}"
                )
            seen[entry.address] = entry
        object.__setattr__(
            self, "entries", tuple(sorted(seen.values(), key=lambda item: item.name))
        )

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_payload(cls, payload: Any) -> "ProtocolBook":
        """Build from the JSON shape the book has on disk (or inline in an env var)."""
        if isinstance(payload, Mapping):
            rows: Sequence[Any] = payload.get("protocols", ())
        elif isinstance(payload, Sequence):
            rows = payload
        else:
            raise TestnetNotConfigured(
                f"an address book is an object or a list, not {type(payload).__name__}"
            )
        allowed = set(ProtocolEntry.__dataclass_fields__)
        entries: List[ProtocolEntry] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TestnetNotConfigured(f"{row!r} is not a protocol entry")
            known = {key: row[key] for key in row if key in allowed}
            entries.append(ProtocolEntry(**known))
        return cls(tuple(entries))

    @classmethod
    def load(cls, path: str) -> "ProtocolBook":
        """Read the book from a JSON file; a missing file is an error, not an empty
        one."""
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except OSError as exc:
            raise TestnetNotConfigured(f"no address book at {path}: {exc}") from exc
        except ValueError as exc:
            raise TestnetNotConfigured(f"{path} is not JSON: {exc}") from exc
        return cls.from_payload(payload)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ProtocolBook":
        """``TESTNET_PROTOCOLS_FILE`` if set, else ``TESTNET_PROTOCOLS``, else empty.

        The file wins because a file is the operator's document; the inline variable is
        for a job that has nowhere to put one. Neither is required: a venue with no book
        is a venue with nothing on offer, which is the safe direction to fail.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        path = str(source.get(ENV_BOOK_FILE, "")).strip()
        if path:
            return cls.load(path)
        inline = str(source.get(ENV_BOOK, "")).strip()
        if inline:
            try:
                return cls.from_payload(json.loads(inline))
            except ValueError as exc:
                raise TestnetNotConfigured(f"{ENV_BOOK} is not JSON: {exc}") from exc
        return cls()

    # -- reading ------------------------------------------------------------ #

    def whitelist(self) -> WhiteList:
        """The operator's approved addresses, as the list that produced the book."""
        return WhiteList(tuple(entry.address for entry in self.entries))

    def get(self, address: str) -> Optional[ProtocolEntry]:
        key = str(address).lower()
        for entry in self.entries:
            if entry.address == key:
                return entry
        return None

    def has(self, address: str) -> bool:
        return self.get(address) is not None

    def addresses(self) -> List[str]:
        return [entry.address for entry in self.entries]

    def describe(self) -> List[Dict[str, Any]]:
        return [entry.as_dict() for entry in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Any:
        return iter(self.entries)


# ── what a dry run answers ──────────────────────────────────────────────────


@dataclass(frozen=True)
class DryRun:
    """The result of asking a chain *what would happen* — and doing nothing.

    ``ok`` is the venue's own verdict, not this file's: it is ``True`` when the call
    neither reverted nor ran out of gas, which is the most a simulation can know. A
    transaction hash is absent by construction — nothing was signed, so nothing has one.
    """

    action: str
    address: str
    ok: bool
    gas: Optional[int] = None
    revert: Optional[str] = None
    amount: float = 0.0
    units: int = 0
    sender: str = ""
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "address": self.address,
            "ok": bool(self.ok),
            "gas": None if self.gas is None else int(self.gas),
            "revert": self.revert,
            "amount": float(self.amount),
            "units": int(self.units),
            "sender": self.sender,
            "note": self.note,
        }

    def __str__(self) -> str:
        verdict = "would pass" if self.ok else f"would fail: {self.revert}"
        gas = "unknown" if self.gas is None else str(int(self.gas))
        return f"dry run {self.action} {self.amount} -> {verdict} (gas {gas})"


# ── what the wallet actually holds ──────────────────────────────────────────


@dataclass(frozen=True)
class Balance:
    """What an account holds, read off the chain — never assumed, never cached.

    Two numbers, because on an L2 they are two different budgets: ``native`` is the gas
    money, and ``token`` is the credit the vault pulls. A stake spends the second and
    costs a little of the first, so a loop that only knows one of them is a loop that
    either reverts or stalls.

    ``error`` is the honest shape of a failed read: a node that will not answer produces
    *no balance*, not a balance of zero, and a caller that cannot tell those apart sizes
    a trade on a fact it does not have. Every caller here treats a non-empty ``error``
    as a refusal to trade.
    """

    address: str = ""
    native_eth: float = 0.0
    native_units: int = 0
    token: str = ""
    token_balance: float = 0.0
    token_units: int = 0
    decimals: int = DEFAULT_DECIMALS
    chain_id: int = 0
    block: int = 0
    error: str = ""
    #: What is *already* in the venue, when the venue can say: for an Aave position, the
    #: aToken balance, which is the supplied amount plus every interest payment since. Zero
    #: for a vault — a ``vault`` reports its own accounting through its record, not through
    #: an ERC20 the wallet holds — and the two numbers are separate because "the wallet has
    #: 13.82 USDC" and "1.00 of it is lent out" are different facts about the same cent.
    position_token: str = ""
    position_balance: float = 0.0
    position_units: int = 0

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def staked(self) -> bool:
        """Whether the venue already holds something of this entry."""
        return self.position_units > 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "native_eth": round(float(self.native_eth), 9),
            "native_units": int(self.native_units),
            "token": self.token,
            "token_balance": round(float(self.token_balance), 6),
            "token_units": int(self.token_units),
            "decimals": int(self.decimals),
            "chain_id": int(self.chain_id),
            "block": int(self.block),
            "error": self.error,
            "position_token": self.position_token,
            "position_balance": round(float(self.position_balance), 6),
            "position_units": int(self.position_units),
        }

    def __str__(self) -> str:
        if self.error:
            return f"balance of {self.address or 'the signer'}: unread — {self.error}"
        line = f"{self.native_eth:.6f} ETH"
        if self.token:
            line += f" + {self.token_balance:.6f} of {self.token}"
        if self.position_token:
            line += f" ({self.position_balance:.6f} of {self.position_token} in the venue)"
        return f"balance of {self.address or 'the signer'}: {line}"


# ── the configuration, and where it is allowed to come from ──────────────────


def _clean(value: object) -> str:
    """A value from an environment: stripped, and without the quotes a .env leaves."""
    text = str(value if value is not None else "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    return text


def _flag(value: object) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "on"}


def _int_or(value: object, default: int, name: str) -> int:
    text = _clean(value)
    if not text:
        return int(default)
    try:
        return int(text)
    except ValueError as exc:
        raise TestnetNotConfigured(f"{name}={text!r} is not a number") from exc


def _float_or(value: object, default: float, name: str) -> float:
    text = _clean(value)
    if not text:
        return float(default)
    try:
        return float(text)
    except ValueError as exc:
        raise TestnetNotConfigured(f"{name}={text!r} is not a number") from exc


_DOTENV_READ = False


def _load_dotenv_once() -> None:
    """Read a ``.env`` if that optional package is there. Never writes, never overrides.

    Best-effort on purpose: a missing ``python-dotenv`` is no reason to refuse a venue
    that was configured through the environment, and an operator who exported a variable
    must not be overruled by a file they forgot about.
    """
    global _DOTENV_READ
    if _DOTENV_READ:
        return
    _DOTENV_READ = True
    try:  # pragma: no cover - depends on an optional package
        from dotenv import load_dotenv
    except ImportError:
        return
    try:  # pragma: no cover - depends on the operator's file
        load_dotenv(override=False)
    except Exception:  # pragma: no cover - a broken .env must not break a run
        return


def _abi_from(path: object) -> Optional[List[Dict[str, Any]]]:
    """A vault ABI from a JSON file, or ``None`` to use :data:`VAULT_ABI`."""
    target = _clean(path)
    if not target:
        return None
    try:
        with open(target, encoding="utf-8") as handle:
            abi = json.load(handle)
    except (OSError, ValueError) as exc:
        raise TestnetNotConfigured(f"cannot read the ABI at {target}: {exc}") from exc
    if not isinstance(abi, list):
        raise TestnetNotConfigured(f"{target} is not an ABI (expected a JSON list)")
    return abi


def _is_private_key(value: str) -> bool:
    """Thirty-two bytes of hex, the only shape a signer key has."""
    text = value[2:] if value.startswith(("0x", "0X")) else value
    if len(text) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in text)


@dataclass(frozen=True)
class TestnetConfig:
    """Everything the venue needs, and nothing it could invent.

    Built by :meth:`from_env` in every path that matters. The key is ``repr=False`` so a
    stray ``print(config)``, a log line or a traceback cannot publish it — the only way
    to read it is to ask for the attribute by name, which is what the signing call does.
    """

    rpc_url: str = ""
    private_key: str = field(default="", repr=False)
    unlocked_account: str = ""
    chain_id: Optional[int] = None
    #: ``TESTNET_PROFILE``: the chain's name, if the operator named one. Supplies the
    #: chain id and the spelling of the node/signer variables — never a permission.
    profile: str = ""
    #: Which variable actually answered, when it was not the canonical ``TESTNET_*``
    #: name (``{"TESTNET_RPC_URL": "ARBITRUM_RPC_URL"}``). Provenance, not secrets:
    #: names only, so a report can say where a signer came from.
    sources: Dict[str, str] = field(default_factory=dict)
    book: ProtocolBook = field(default_factory=ProtocolBook)
    decimals: int = DEFAULT_DECIMALS
    lookback: int = DEFAULT_LOOKBACK
    #: ``TESTNET_HISTORY``: ``chain`` (the vault's own logs), ``feed`` (DeFiLlama's daily
    #: APY history for the entry's name) or ``auto`` (log first, feed when the log is
    #: silent). A *policy*, not a fallback: the venue is told which records it may publish,
    #: because a record from the feed is a fact about a name and never about an address.
    history: str = DEFAULT_HISTORY
    receipt_timeout: float = DEFAULT_RECEIPT_TIMEOUT
    block_timeout: float = DEFAULT_BLOCK_TIMEOUT
    allow_mainnet: bool = False
    abi: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        # A profile is resolved before anything is validated, because it is allowed to
        # supply exactly one value: which chain this is. A misspelt profile is a named
        # error rather than a silent testnet, so a typo cannot quietly point the venue
        # at the wrong network.
        profile = _clean(self.profile).lower()
        if profile:
            if profile not in CHAIN_PROFILES:
                raise TestnetNotConfigured(
                    f"{ENV_PROFILE}={self.profile!r} is not one of: "
                    + ", ".join(sorted(CHAIN_PROFILES))
                )
            if self.chain_id is None:
                object.__setattr__(self, "chain_id", int(CHAIN_PROFILES[profile]))
        object.__setattr__(self, "profile", profile)
        object.__setattr__(
            self, "sources", {str(k): str(v) for k, v in dict(self.sources).items()}
        )

        # Which records the venue may publish. A policy with three values and no fourth:
        # a misspelt one is a named error rather than a silent "chain", because the
        # operator asked a question about provenance and a typo must not answer it.
        history = _clean(self.history).lower() or DEFAULT_HISTORY
        if history not in HISTORY_SOURCES:
            raise TestnetNotConfigured(
                f"{ENV_HISTORY}={self.history!r} is not one of: "
                + ", ".join(HISTORY_SOURCES)
            )
        object.__setattr__(self, "history", history)

        # The name that answered, so an error about a value from an alias sends the
        # operator to the variable they actually set.
        named_rpc = self.sources.get(ENV_RPC, ENV_RPC)
        named_key = self.sources.get(ENV_KEY, ENV_KEY)

        url = _clean(self.rpc_url)
        if not url:
            raise TestnetNotConfigured(
                f"no node: set {named_rpc} (this venue has no default endpoint)"
            )
        if "://" not in url:
            raise TestnetNotConfigured(
                f"{named_rpc} must carry a scheme (http, https or ws), got {url!r}"
            )
        object.__setattr__(self, "rpc_url", url)

        key = _clean(self.private_key)
        if key and not _is_private_key(key):
            # Never echo the value: a wrong key is still a key.
            raise TestnetNotConfigured(
                f"{named_key} is not 32 bytes of hex (its length is {len(key)})"
            )
        object.__setattr__(self, "private_key", key)

        unlocked = _clean(self.unlocked_account)
        if unlocked and not _is_address(unlocked):
            raise TestnetNotConfigured(f"{ENV_UNLOCKED} is not an address")
        if unlocked and key:
            raise TestnetNotConfigured(
                f"both {ENV_KEY} and {ENV_UNLOCKED} are set: one signer, not two"
            )
        object.__setattr__(self, "unlocked_account", unlocked.lower())

        if self.chain_id is not None and int(self.chain_id) <= 0:
            raise TestnetNotConfigured(f"{ENV_CHAIN_ID} must be a positive number")
        if not (0 <= int(self.decimals) <= 18):
            raise TestnetNotConfigured(f"{ENV_DECIMALS} must be within 0..18")
        if int(self.lookback) <= 0:
            raise TestnetNotConfigured(f"{ENV_LOOKBACK} must be positive")
        if float(self.receipt_timeout) <= 0 or float(self.block_timeout) <= 0:
            raise TestnetNotConfigured("a timeout of zero is not a timeout")
        if not isinstance(self.book, ProtocolBook):
            raise TestnetNotConfigured("the address book must be a ProtocolBook")

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None, **overrides: Any
    ) -> "TestnetConfig":
        """Read every name in the table above; raise on anything malformed.

        ``env`` is a parameter so a test can hand over a dictionary instead of editing
        the process's environment — which is also the demonstration that the file reads
        configuration from nowhere else.

        ``allow_mainnet`` is the one value that needs **two** yeses: the environment
        must permit a real-money chain (``TESTNET_ALLOW_MAINNET=1``) *and* this call
        must ask for it. An operator who exported the variable months ago should not be
        one forgotten argument away from signing on Ethereum, and a run that asks for
        it should not be one exported variable away from the same.

        ``TESTNET_PROFILE`` is a *spelling*, not a permission. Naming a chain
        (``arbitrum``) lets the names an operator already has exported answer for the
        ``TESTNET_*`` ones — ``ARBITRUM_RPC_URL`` for ``TESTNET_RPC_URL``, and the same
        for the signer, the unlocked account and the chain id — and supplies the chain
        id when it is not written down. The canonical name always wins, and whichever
        name answered is recorded in :attr:`sources`, so nothing about a signer's
        provenance is left to be guessed. An L2 named here is still a chain where
        credits are money, so it is still refused without both yeses.
        """
        if env is None:
            _load_dotenv_once()
        source: Mapping[str, str] = os.environ if env is None else env
        requested = bool(overrides.pop("allow_mainnet", False))

        profile_name = _clean(source.get(ENV_PROFILE)).lower()
        if profile_name and profile_name not in CHAIN_PROFILES:
            raise TestnetNotConfigured(
                f"{ENV_PROFILE}={profile_name!r} is not one of: "
                + ", ".join(sorted(CHAIN_PROFILES))
            )
        prefix = f"{profile_name.upper()}_" if profile_name else ""
        sources: Dict[str, str] = {}

        def ask(canonical: str) -> str:
            """The canonical variable, or the profile's spelling of it — never both."""
            value = _clean(source.get(canonical))
            if value or not prefix:
                return value
            alias = prefix + canonical[len("TESTNET_") :]
            answer = _clean(source.get(alias))
            if answer:
                sources[canonical] = alias
            return answer

        chain_text = ask(ENV_CHAIN_ID)
        values: Dict[str, Any] = {
            "rpc_url": ask(ENV_RPC),
            "private_key": ask(ENV_KEY),
            "unlocked_account": ask(ENV_UNLOCKED),
            "chain_id": (
                _int_or(chain_text, 0, sources.get(ENV_CHAIN_ID, ENV_CHAIN_ID))
                if chain_text
                else (int(CHAIN_PROFILES[profile_name]) if profile_name else None)
            ),
            "profile": profile_name,
            "sources": sources,
            "book": ProtocolBook.from_env(source),
            "decimals": _int_or(
                source.get(ENV_DECIMALS), DEFAULT_DECIMALS, ENV_DECIMALS
            ),
            "lookback": _int_or(
                source.get(ENV_LOOKBACK), DEFAULT_LOOKBACK, ENV_LOOKBACK
            ),
            "history": _clean(source.get(ENV_HISTORY)) or DEFAULT_HISTORY,
            "receipt_timeout": _float_or(
                source.get(ENV_RECEIPT_TIMEOUT),
                DEFAULT_RECEIPT_TIMEOUT,
                ENV_RECEIPT_TIMEOUT,
            ),
            "block_timeout": _float_or(
                source.get(ENV_BLOCK_TIMEOUT), DEFAULT_BLOCK_TIMEOUT, ENV_BLOCK_TIMEOUT
            ),
            "allow_mainnet": _flag(source.get(ENV_ALLOW_MAINNET)) and requested,
            "abi": _abi_from(source.get(ENV_ABI_FILE)),
        }
        values.update(overrides)
        return cls(**values)

    # -- what it can say about itself --------------------------------------- #

    @property
    def can_sign(self) -> bool:
        """Whether this venue can move capital at all, or only read."""
        return bool(self.private_key or self.unlocked_account)

    @property
    def rpc_host(self) -> str:
        """Host only. An RPC path is often an API key and never belongs in a report."""
        parts = urlsplit(self.rpc_url)
        return parts.hostname or "(unnamed host)"

    @property
    def masked_key(self) -> str:
        if not self.private_key:
            return "(no key)"
        return f"0x...{self.private_key[-4:]}"

    @property
    def vault_abi(self) -> List[Dict[str, Any]]:
        return list(self.abi) if self.abi is not None else list(VAULT_ABI)

    def describe(self) -> Dict[str, Any]:
        return {
            "rpc_host": self.rpc_host,
            "signer": (
                "unlocked account"
                if self.unlocked_account
                else ("key " + self.masked_key if self.private_key else "read-only")
            ),
            "chain_id": self.chain_id,
            "profile": self.profile or None,
            "sources": dict(self.sources),
            "decimals": int(self.decimals),
            "lookback": int(self.lookback),
            "history": str(self.history),
            "protocols": len(self.book),
            "allow_mainnet": bool(self.allow_mainnet),
        }


# ── the venue itself ────────────────────────────────────────────────────────


class RealTestnetProvider(IWeb3Provider):
    """A testnet, answering the five operations the layer above already knows.

    Construct it three ways, in decreasing order of how much it can do:

    >>> provider = RealTestnetProvider.from_env()          # doctest: +SKIP
    >>> provider = RealTestnetProvider(                    # doctest: +SKIP
    ...     rpc_url="http://127.0.0.1:8545", book=ProtocolBook.load("book.json")
    ... )
    >>> provider = RealTestnetProvider(config=TestnetConfig.from_env())  # +SKIP

    With a key (or an unlocked node account) it can stake, withdraw and hand over
    evidence. Without one it is **read-only**: it publishes what the book says and what
    the chain's logs paid, and every capital call raises
    :class:`TestnetNotConfigured`. That is the mode the notebook runs in, because a demo
    that needs a funded key is a demo nobody runs.

    The two answers a caller can get, and why they are different
    -----------------------------------------------------------

    ``False`` is a *policy* answer: the address is not in the operator's book, the
    amount is not positive, the venue has no signer. Nothing was signed, nothing moved,
    and the refusal is appended to :attr:`refusals` with its reason.

    :class:`~agent.physics.portfolio.StakeRefused` is a *venue* answer: the node
    reverted the call, the gas estimate refused, the receipt never came. A transaction
    may or may not exist, so the failure is raised with the reason attached rather than
    reported as ``False`` — the caller must not record a position it cannot see.

    :meth:`submit_validation_proof` is the exception to that exception: it never raises,
    because a 720-step run must not die on a flaky node's opinion of evidence. It
    returns ``False`` and keeps the reason in :attr:`failures`.
    """

    def __init__(
        self,
        *,
        config: Optional[TestnetConfig] = None,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        book: Optional[ProtocolBook] = None,
        web3: Any = None,
        env: Optional[Mapping[str, str]] = None,
        allow_mainnet: bool = False,
        feed: Any = None,
        **overrides: Any,
    ) -> None:
        """Build the venue; do not touch the network yet.

        The construction is pure on purpose: no connection, no ``chain_id``, no block —
        so importing, printing and describing a provider costs nothing and works on a
        machine that has never seen a chain. :meth:`_connect` is the first touch, and it
        happens on the first call that needs an answer.

        ``web3`` is a seam for tests (and for a node library other than web3.py): any
        object with an ``eth`` attribute answering the calls in this file will do. ``feed``
        is the same seam for the other wire: anything with
        ``history(name, steps=None, uuid="")`` answering a
        :class:`~agent.physics.defillama_adapter.FeedRecord`. Left out, it is built on first
        use from :class:`~agent.physics.defillama_adapter.FeedHistory` — one catalogue per
        process, because a catalogue is 1.4 MB.
        """
        if config is None:
            given: Dict[str, Any] = dict(overrides)
            if rpc_url is not None:
                given["rpc_url"] = rpc_url
            if private_key is not None:
                given["private_key"] = private_key
            if book is not None:
                given["book"] = book
            if allow_mainnet:
                given["allow_mainnet"] = True
            config = TestnetConfig.from_env(env, **given)
        elif overrides:
            raise TestnetNotConfigured("pass a config, or its parts, but not both")

        self.config: TestnetConfig = config
        self._web3: Any = web3
        self._feed: Any = feed
        self._signer: str = config.unlocked_account
        self._account: Any = None
        self._chain_id: Optional[int] = config.chain_id
        self._origin: Optional[int] = None
        self._block: int = 0
        self._nonce: Optional[int] = None
        self._topic: Optional[str] = None
        self._claims: Dict[str, float] = {}
        self._tokens: Dict[str, str] = {}
        # Where each address's last record came from: ``chain``, or the feed's label for the
        # pool. Kept per address and never inferred later — a report that guessed would be a
        # report that can say "the vault paid this" about a number the vault never saw.
        self._history: Dict[str, str] = {}
        self._records: Dict[str, Any] = {}
        self._whitelist: WhiteList = config.book.whitelist()

        # the ledger a report prints; every row is a thing that happened
        self.stakes: List[Dict[str, Any]] = []
        self.withdrawals: List[Dict[str, Any]] = []
        self.approvals: List[Dict[str, Any]] = []
        self.proofs: List[Dict[str, Any]] = []
        self.refusals: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []
        self.dry_runs: List[DryRun] = []
        self.errors: List[str] = []

    # -- construction from the environment ---------------------------------- #

    @classmethod
    def from_env(
        cls, env: Optional[Mapping[str, str]] = None, **overrides: Any
    ) -> "RealTestnetProvider":
        """The configured venue: everything from ``os.environ`` (or a given mapping)."""
        return cls(config=TestnetConfig.from_env(env, **overrides))

    # -- identity ----------------------------------------------------------- #

    @property
    def name(self) -> str:
        """``testnet`` — the chain's own id is a fact, not a name (see ``describe``)."""
        return "testnet"

    @property
    def step(self) -> int:
        """Blocks this venue has lived since it first connected; 0 before that.

        Refreshed by :meth:`advance` and by every read of the book, and never by a
        property access: a clock that costs a request would make a 720-step loop 720
        requests slower for no new information.
        """
        if self._origin is None:
            return 0
        return max(int(self._block) - int(self._origin), 0)

    @property
    def whitelist(self) -> WhiteList:
        """The operator's addresses, as read from the book at construction."""
        return self._whitelist

    @property
    def chain_id(self) -> Optional[int]:
        """The chain this venue is bound to, once it has been asked."""
        return self._chain_id

    @property
    def block_number(self) -> int:
        """The height this venue last saw."""
        return int(self._block)

    @property
    def can_sign(self) -> bool:
        """Whether this venue can move capital, or only read."""
        return bool(self.config.can_sign)

    # -- the node ----------------------------------------------------------- #

    def _w3(self) -> Any:
        """The web3 handle, built on first use and never at import time.

        Importing web3 here, in a method, is what keeps ``agent.physics`` free of a
        network library on the import path: the package that decides things must not
        carry the thing that signs. It is also what lets this module be imported on a
        machine where web3 is not installed at all.
        """
        if self._web3 is not None:
            return self._web3
        try:
            from web3 import Web3
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise TestnetNotConfigured(
                "web3 is not installed: pip install web3 (or entrain-ai[testnet])"
            ) from exc
        self._web3 = Web3(
            Web3.HTTPProvider(self.config.rpc_url, request_kwargs={"timeout": 30})
        )
        return self._web3

    def _checksum(self, address: str) -> str:
        """The form web3 insists on for an address it is asked to call."""
        try:
            return str(self._w3().to_checksum_address(address))
        except (AttributeError, ValueError):  # pragma: no cover - a permissive client
            return str(address)

    def _refresh_block(self) -> int:
        """Read the height, keeping the last good one when the node will not answer."""
        try:
            height = int(self._w3().eth.block_number)
        except Exception as exc:
            self.errors.append(f"block_number: {type(exc).__name__}: {exc}")
            return int(self._block)
        if self._origin is None:
            self._origin = height
        if height > self._block:
            self._block = height
        return int(self._block)

    def _connect(self) -> None:
        """Bind to a chain once: its id, its height, and whether we are allowed here."""
        if self._origin is not None:
            return
        w3 = self._w3()
        try:
            height = int(w3.eth.block_number)
            found = int(w3.eth.chain_id)
        except Exception as exc:
            raise TestnetUnavailable(
                f"no chain at {self.config.rpc_host}: {type(exc).__name__}: {exc}"
            ) from exc
        if self.config.chain_id is not None and found != int(self.config.chain_id):
            raise TestnetNotConfigured(
                f"the node is chain {found}, but the configuration says "
                f"{int(self.config.chain_id)}"
            )
        if found in MAINNET_CHAIN_IDS and not self.config.allow_mainnet:
            raise TestnetNotConfigured(
                f"chain {found} is a mainnet: this venue is configured to move credits "
                f"on a testnet, and letting it sign here would need "
                f"{ENV_ALLOW_MAINNET}=1 and allow_mainnet=True"
            )
        self._chain_id = found
        self._origin = height
        self._block = max(int(self._block), height)

    # -- the clock ---------------------------------------------------------- #

    def advance(self, steps: int = 1) -> int:
        """Wait for the chain to produce ``steps`` blocks; return the new clock.

        Waiting, not mining: a testnet's block time is the operator's, and a provider
        that manufactured blocks on demand would be a provider that spends someone
        else's gas to keep its own loop moving. So ``advance`` polls the height, gives
        up after ``TESTNET_BLOCK_TIMEOUT``, and reports the clock it actually has — a
        step that did not happen is a step the empire must see as not having happened.
        """
        self._connect()
        wanted = int(self._block) + max(int(steps), 1)
        deadline = time.monotonic() + float(self.config.block_timeout)
        while True:
            height = self._refresh_block()
            if height >= wanted:
                break
            if time.monotonic() >= deadline:
                self.errors.append(
                    f"advance: no block for {self.config.block_timeout:g}s "
                    f"(height {height}, wanted {wanted})"
                )
                break
            time.sleep(BLOCK_POLL)
        return self.step

    # -- what is on offer --------------------------------------------------- #

    def protocol_kind(self, address: str) -> str:
        """How money reaches this address, as the operator's book declares it.

        ``vault`` for the interface this file wrote, ``aave-v3`` for a Pool the deployment
        table names, and ``""`` for an address the book does not contain — the same
        three-valued answer :meth:`history_source` gives about a record, because a caller
        that is about to sign deserves to know both *what I read* and *what I would call*.

        Reading the book is all this does. Whether the declared kind is *true* is
        :meth:`_deployment`'s question, and it is asked on the chain's chain id, so an
        entry cannot become an Aave pool by spelling its kind that way.
        """
        entry = self.config.book.get(str(address).lower())
        return str(entry.kind) if entry is not None else ""

    def protocol_identity(self, address: str) -> ProtocolIdentity:
        """How the feed's report names this entry — by project, symbol, chain and uuid.

        The companion of :meth:`protocol_kind`, and the answer to a problem a label cannot
        solve: a book entry is written for a human (``aave-v3 USDC (Arbitrum)``) and the
        feed writes its rows for a database (``USDC``, project ``aave-v3``, chain
        ``Arbitrum``). Matching the two by spelling is what turns a real, verified pool into
        "the market has no pool named like this vault".

        So the entry's own declaration is what is returned, and for an ``aave-v3`` entry the
        deployment table completes it: the asset's symbol (``USDC``) and the chain's name
        come from ``aave_v3.DEPLOYMENTS``, and the feed's project slug for such a pool is
        this kind's own name. Empty fields mean "the operator said nothing and the table
        cannot fill it in", and a caller that matches on them must then fall back to the
        name — which is the strict path, and still the right one for a vault.

        An entry the book does not contain answers nothing — the same three-valued honesty
        as :meth:`history_source`, because a caller about to sign deserves to know what it
        does *not* know.
        """
        entry = self.config.book.get(str(address).lower())
        if entry is None:
            return ProtocolIdentity()
        kind = str(entry.kind)
        project = str(entry.project)
        symbol = str(entry.symbol)
        chain = str(entry.chain)
        protocol = ""
        if kind == AAVE_KIND:
            deployment = self._deployment_or_none(entry)
            project = project or AAVE_KIND
            protocol = aave_v3.PROTOCOL_NAME
            if deployment is not None:
                symbol = symbol or deployment.symbol
                chain = chain or deployment.chain
        return ProtocolIdentity(
            kind=kind,
            project=project,
            symbol=symbol,
            chain=chain,
            pool=str(entry.pool),
            protocol=protocol,
        )

    def _deployment(self, entry: ProtocolEntry) -> "aave_v3.AaveDeployment":
        """The Aave deployment this entry *is*, or a refusal that names the difference.

        The check is a table lookup on ``(chain_id, pool, asset)`` and nothing else: no
        heuristic, no "looks like Aave", no probe of the contract's bytecode. A pool this
        file has not written down is a pool whose ``supply`` is an unknown function on an
        unknown contract, and the honest answer to that is to refuse before the approval
        that would have paid for it.
        """
        self._connect()
        deployment = aave_v3.deployment_for(self._chain_id, entry.address, entry.token)
        if deployment is None:
            raise TestnetNotConfigured(
                f"{entry.name}: {entry.address} over {entry.token or 'no asset'} is not an "
                f"Aave v3 deployment this file knows on chain {self._chain_id}. The pool "
                f"and the asset have to be the pair in agent.physics.aave_v3.DEPLOYMENTS "
                f"(known here: {', '.join(str(key) for key in sorted(aave_v3.DEPLOYMENTS))})"
                " — a book cannot declare a venue into existence"
            )
        if int(entry.decimals) != int(deployment.decimals):
            raise TestnetNotConfigured(
                f"{entry.name}: the book says {int(entry.decimals)} decimals and "
                f"{deployment.symbol} on {deployment.chain} has {int(deployment.decimals)}. "
                "A supply in the wrong units is a revert at best and a thousandfold approval "
                "at worst, so the mismatch is refused before anything is signed"
            )
        self._check_declaration(entry, deployment)
        return deployment

    @staticmethod
    def _check_declaration(
        entry: ProtocolEntry, deployment: "aave_v3.AaveDeployment"
    ) -> None:
        """The entry's ``project``/``symbol``/``chain`` against the pool it says it is.

        These three keys exist to *match* a verdict, and a match is only worth having when
        it agrees with the thing being matched: an entry that pins ``symbol: USDT`` on the
        USDC pool would be looking for the wrong row's verdict, and pinning another chain
        would be looking for another pool's. Both are silent failures — a hold, or worse a
        signature with a sentence that names the wrong token — so they are refused here,
        next to the decimals, where a contradiction already cannot get past.

        A key the operator left empty is not a claim and is never refused: an empty string
        is the way to say "match nothing here", and the table fills the answer in.
        """
        wrong: List[str] = []
        if entry.project and entry.project.strip().lower() != AAVE_KIND:
            wrong.append(
                f"project {entry.project!r} is not {AAVE_KIND!r} (the feed's slug for an "
                "Aave v3 pool)"
            )
        if entry.symbol and entry.symbol.strip().upper() != deployment.symbol:
            wrong.append(
                f"symbol {entry.symbol!r} is not {deployment.symbol!r} (the asset this "
                "pool lends)"
            )
        if entry.chain:
            spelled = entry.chain.strip()
            digits = spelled.isdigit()
            matches = (
                int(spelled) == int(deployment.chain_id)
                if digits
                else spelled.lower() == deployment.chain.lower()
            )
            if not matches:
                wrong.append(
                    f"chain {entry.chain!r} is not {deployment.chain!r} "
                    f"(chain id {deployment.chain_id})"
                )
        if wrong:
            raise TestnetNotConfigured(
                f"{entry.name}: the book's declaration contradicts the deployment "
                f"({deployment.label}): " + "; ".join(wrong) + ". A declaration that names "
                "another pool would match another pool's verdict, so it is refused before "
                "anything is signed"
            )

    def _deployment_or_none(self, entry: ProtocolEntry) -> Optional[Any]:
        """The deployment, or ``None`` where a refusal would be a report's problem.

        Used by :meth:`balance` and :meth:`_note`, which have to answer even about a book
        this venue would refuse to *sign* into: "I hold nothing here" and "I will not stake
        here" are two different sentences, and a balance report that conflated them would
        hide the second. The lookup is the same table; the difference is that a missing
        answer is reported instead of raised.
        """
        try:
            self._connect()
        except TestnetError:
            return None
        return aave_v3.deployment_for(self._chain_id, entry.address, entry.token)

    def get_protocol_list(self) -> List[Protocol]:
        """Every book entry that has a record — its own, or one the feed vouches for.

        A book entry with no ``YieldDistributed`` log is still published when the operator
        allowed the feed to speak for it (``TESTNET_HISTORY=auto`` or ``feed``): the record
        is then the feed's sentence about the entry's **name**, and the protocol's note says
        so. What this method never does is invent one — a ``Protocol`` refuses an empty
        record (correctly), and a record nobody published is not a measurement.

        The distinction travels with the object: :meth:`history_source` answers ``chain`` for
        a record read from the address itself and the feed's label otherwise, and
        :meth:`protocol_kind` says what a stake into that address would call. The trading
        loop signs into a ``vault`` only on its own record, and into an ``aave-v3`` pool on
        the deployment table's word — never the other way round.
        """
        self._connect()
        self._refresh_block()
        published: List[Protocol] = []
        for entry in self.config.book:
            if str(entry.kind) == AAVE_KIND and str(self.config.history) == HISTORY_CHAIN:
                self.errors.append(
                    f"{entry.name}: an Aave position emits no {YIELD_EVENT_SIGNATURE} — set "
                    f"{ENV_HISTORY}=auto (or feed) so the feed prices the pool it supplies"
                )
                continue
            record = self.fetch_protocol_history(entry.address)
            if not record:
                continue
            try:
                published.append(
                    Protocol(
                        name=entry.name,
                        bucket=entry.bucket,
                        claim_apy=self._claim_apy(entry),
                        returns=np.asarray(record, dtype=float),
                        address=entry.address,
                        alive=True,
                        note=self._note(entry),
                    )
                )
            except PortfolioError as exc:  # a bad record is a protocol to skip
                self.errors.append(f"{entry.name}: {exc}")
        return published

    def history_source(self, address: str) -> str:
        """Where this address's record came from, in a form a report can print.

        ``chain`` when the address published its own ``YieldDistributed`` logs (that is
        :data:`RECORD_FROM_CHAIN`); the feed's label — ``the feed aave-v3 USDC (Arbitrum)
        [d9fa8e]`` — when the record is the feed's statement about the entry's *name*; and
        ``""`` when there is no record at all, which is not a source but an absence.
        """
        return self._history.get(str(address).lower(), "")

    def history_note(self, address: str) -> str:
        """Why an entry has no record, in the feed's own words when the feed said so."""
        record = self._records.get(str(address).lower())
        return str(getattr(record, "reason", "") or "")

    def _note(self, entry: ProtocolEntry) -> str:
        """The book's note, plus the record's provenance when it is not the vault's own.

        A feed-priced protocol is a real record about a *name*; a note that stayed silent
        about it would let an operator believe the address had published it.
        """
        source = self._history.get(entry.address, "")
        if str(entry.kind) == AAVE_KIND:
            deployment = self._deployment_or_none(entry)
            where = str(deployment) if deployment is not None else (
                f"{entry.address} is not an Aave deployment this file knows on chain "
                f"{self._chain_id}"
            )
            extra = f"{where}; calls {aave_v3.SUPPLY_SIGNATURE} / {aave_v3.WITHDRAW_SIGNATURE}"
            note = f"{entry.note} · {extra}" if entry.note else extra
        elif not source or source == RECORD_FROM_CHAIN:
            return entry.note
        else:
            extra = f"record from {source}: this address published no log"
            note = f"{entry.note} · {extra}" if entry.note else extra
        return note

    def _yield_topic(self) -> str:
        """``keccak("YieldDistributed(uint256,uint256)")``, computed once."""
        if self._topic is None:
            try:
                digest = self._w3().keccak(text=YIELD_EVENT_SIGNATURE)
                self._topic = "0x" + bytes(digest).hex()
            except Exception as exc:  # pragma: no cover - a client without keccak
                self.errors.append(f"keccak: {type(exc).__name__}: {exc}")
                return "0x" + "00" * 32
        return self._topic

    def fetch_protocol_history(
        self, address: str, steps: Optional[int] = None
    ) -> List[float]:
        """The rates this entry is priced at, oldest first — its own logs, or the feed's.

        Which of the two answers is the operator's policy (:data:`ENV_HISTORY`), and the
        choice is recorded per address (see :meth:`history_source`) so no caller has to guess
        afterwards:

        ``chain``  the vault's own ``YieldDistributed`` logs, and nothing else (the default),
                   because a record that *this address* published is the only one this file
                   can read without borrowing somebody's word for it;
        ``feed``   the feed's daily APY history for the entry's name or pinned pool — the
                   node is not asked at all, which is how a book of real dApp addresses (an
                   aToken, a Uniswap pool) becomes measurable;
        ``auto``   the log first, the feed when the log is silent. The address's own record
                   always wins: it is the only statement about *this* address.

        Never raises on a venue problem: an address the book does not contain, a node that
        will not answer and a log that will not decode all come back as "nothing published" —
        which the assessment reads as a protocol to refuse. A *configuration* error is
        different and does surface, because the difference between "this venue said nothing"
        and "this venue points at a mainnet" must not be flattened into an empty array.
        """
        key = str(address).lower()
        entry = self.config.book.get(key)
        if entry is None:
            return []
        if str(self.config.history) == HISTORY_FEED:
            return self._feed_record(entry, steps)
        record = self._chain_record(key, steps)
        if record or str(self.config.history) == HISTORY_CHAIN:
            self._history[key] = RECORD_FROM_CHAIN if record else ""
            return record
        # ``auto``: the vault published nothing, so the feed is asked about its name. The
        # silence is the *fallback's* reason to exist, not a fault — which is why the trade
        # loop's warning about a quiet book names this switch instead of blaming the node.
        return self._feed_record(entry, steps)

    def _chain_record(self, key: str, steps: Optional[int] = None) -> List[float]:
        """The rates the address itself published, decoded from its logs."""
        try:
            self._connect()
            height = self._refresh_block()
            window = max(int(self.config.lookback), int(steps or 0))
            logs = self._w3().eth.get_logs(
                {
                    "address": self._checksum(key),
                    "fromBlock": max(int(height) - int(window), 0),
                    "toBlock": "latest",
                    "topics": [self._yield_topic()],
                }
            )
        except (TestnetNotConfigured, TestnetUnavailable):
            raise
        except Exception as exc:
            self.errors.append(f"{key}: get_logs {type(exc).__name__}: {exc}")
            return []
        record = self._decode(key, logs)
        if steps is not None and int(steps) > 0:
            record = record[-int(steps) :]
        return record

    def _feed_source(self) -> Any:
        """The feed, built once and filtered to this venue's chain.

        One instance per provider, because the object holds the scanner's cache: a cycle that
        prices four book entries must read the catalogue once, not four times. The chain
        filter is the feed's own spelling of the chain id (``Arbitrum``), and an id this file
        does not know filters nothing — the record's label prints the chain either way.
        """
        if self._feed is None:
            from agent.physics.defillama_adapter import FeedHistory, chain_name

            self._feed = FeedHistory(
                chain=chain_name(self._chain_id or self.config.chain_id or 0)
            )
        return self._feed

    def _feed_record(
        self, entry: ProtocolEntry, steps: Optional[int] = None
    ) -> List[float]:
        """The feed's daily history for this entry, as per-step rates.

        Never raises: a feed that cannot be read is a record that is absent and a sentence in
        :attr:`errors`, because a cycle that died whenever DeFiLlama had a bad morning would
        be a loop that cannot report the book it *can* read. The pinned ``pool`` uuid wins
        over the name match, and a uuid the feed cannot chart is not quietly replaced by a
        name match — the operator stated which pool this vault is.
        """
        key = entry.address
        try:
            record = self._feed_source().history(
                entry.name, steps=steps, uuid=entry.pool
            )
        except Exception as exc:  # noqa: BLE001 - one entry, one sentence, never a cycle
            self.errors.append(
                f"{entry.name}: the feed could not be read ({type(exc).__name__}: {exc})"
            )
            self._history[key] = ""
            return []
        self._records[key] = record
        rates = record.rates().tolist() if record else []
        if not rates:
            self._history[key] = ""
            reason = str(getattr(record, "reason", "") or "")
            self.errors.append(
                f"{entry.name}: no record from the feed"
                + (f" — {reason}" if reason else "")
            )
            return []
        self._history[key] = record.label
        return rates

    def _decode(self, key: str, logs: Any) -> List[float]:
        """Turn ``YieldDistributed`` logs into per-epoch rates, in chain order.

        The event is ``YieldDistributed(uint256 indexed epoch, uint256 rateE18)``, so
        the data is one static word and the epoch is a topic: a record needs no ABI, and
        a log this file cannot read is dropped with its reason rather than corrupting
        the series around it.

        The word arrives as ``HexBytes`` from web3 and as a hex string from anything
        else, so both are accepted: a decoder that only knew one of them would work in a
        test and fail on the first real node.
        """
        rows: List[Tuple[int, int, float]] = []
        for log in logs or ():
            try:
                raw = log.get("data")
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    word = bytes(raw).hex()
                else:
                    text = str(raw)
                    word = text[2:] if text.startswith("0x") else text
                if len(word) < 64:
                    raise ValueError("data is shorter than one word")
                rate = int(word[:64], 16) / RATE_SCALE
                block = int(log.get("blockNumber") or 0)
                index = int(log.get("logIndex") or 0)
            except (AttributeError, TypeError, ValueError) as exc:
                self.errors.append(
                    f"{key}: unreadable log ({type(exc).__name__}: {exc})"
                )
                continue
            if not np.isfinite(rate):
                self.errors.append(f"{key}: unreadable rate in block {block}")
                continue
            rows.append((block, index, float(rate)))
        rows.sort(key=lambda row: (row[0], row[1]))
        return [rate for _, _, rate in rows]

    def _claim_apy(self, entry: ProtocolEntry) -> float:
        """What the vault *says* it pays: its own answer if it has one, else the book's.

        A claim is a fact about the operator's file until the chain can be asked, and it
        is cached once asked: the claim is a declaration, not a measurement, and reading
        it again on every review would be a request for a number that never changes.
        """
        if entry.address in self._claims:
            return self._claims[entry.address]
        if str(entry.kind) == AAVE_KIND:
            # Aave's Pool has no ``claimApyE18``: the rate is published by the protocol,
            # not declared by the contract, so the book's number is the only claim there is
            # — and the *record* (the feed's history) is what the assessment measures.
            self._claims[entry.address] = float(entry.claim_apy)
            return self._claims[entry.address]
        claim = float(entry.claim_apy)
        try:
            function = self._vault(entry.address).functions.claimApyE18()
            raw = int(function.call({"from": self._sender_or_zero()}))
            claim = float(raw) / RATE_SCALE
        except Exception as exc:
            self.errors.append(
                f"{entry.name}: claim unreadable ({type(exc).__name__}), using the book"
            )
        self._claims[entry.address] = claim
        return claim

    # -- who signs, and how it is sent -------------------------------------- #

    ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

    def _signer_address(self) -> str:
        """The account capital moves from: a key's address, or the node's account."""
        if self._signer:
            return self._signer
        if not self.config.private_key:
            raise TestnetNotConfigured(
                f"this venue has no signer: set {ENV_KEY} or {ENV_UNLOCKED}"
            )
        try:
            account = self._w3().eth.account.from_key(self.config.private_key)
        except Exception as exc:
            raise TestnetNotConfigured(
                f"{ENV_KEY} is not a usable key: {type(exc).__name__}"
            ) from exc
        self._account = account
        self._signer = str(account.address).lower()
        return self._signer

    def _sender_or_zero(self) -> str:
        """The ``from`` for a simulation: the signer when there is one, else nobody."""
        if self.config.can_sign:
            return self._checksum(self._signer_address())
        return self._checksum(self.ZERO_ADDRESS)

    def _next_nonce(self) -> int:
        """The next nonce, read from the node once and counted locally in between.

        The local cache is dropped after every failure (:meth:`_send`), because a nonce
        that survives a broken send is how a bot builds a queue of transactions that can
        never land: the node is the only authority on what is still pending.
        """
        if self._nonce is None:
            self._nonce = int(
                self._w3().eth.get_transaction_count(self._sender_or_zero(), "pending")
            )
        nonce = int(self._nonce)
        self._nonce = nonce + 1
        return nonce

    def _fee(self) -> Dict[str, int]:
        """A legacy gas price: accepted by Sepolia, Anvil, and every testnet alive."""
        try:
            return {"gasPrice": int(self._w3().eth.gas_price)}
        except Exception as exc:  # pragma: no cover - a node without eth_gasPrice
            self.errors.append(f"gas_price: {type(exc).__name__}: {exc}")
            return {}

    def _dispatch(self, tx: Dict[str, Any]) -> Any:
        """Hand one transaction to the node — signed here, or signed by the node.

        With a key the signing happens in this process and only the raw bytes travel;
        with an unlocked account the node signs, which is the local-Anvil path and the
        one place ``eth_sendTransaction`` is the right call.
        """
        w3 = self._w3()
        if self.config.unlocked_account:
            return w3.eth.send_transaction(tx)
        signed = w3.eth.account.sign_transaction(tx, self.config.private_key)
        raw = getattr(signed, "raw_transaction", None)
        if raw is None:  # web3 < 6.9 spelled it rawTransaction
            raw = getattr(signed, "rawTransaction", None)
        return w3.eth.send_raw_transaction(raw)

    @staticmethod
    def _reason(exc: BaseException) -> str:
        """One line for a revert: the node's own words when it has them."""
        text = str(exc).strip().replace("\n", " ")
        return text or type(exc).__name__

    @staticmethod
    def _hex(digest: Any) -> str:
        """A transaction hash as ``0x...``, whatever the client handed back."""
        try:
            return "0x" + bytes(digest).hex()
        except (TypeError, ValueError):  # pragma: no cover - an exotic client
            return str(digest)

    # -- one transaction, from estimate to receipt -------------------------- #

    def _send(
        self,
        function: Any,
        *,
        action: str,
        address: str,
        amount: float = 0.0,
        units: int = 0,
    ) -> Dict[str, Any]:
        """Estimate, sign, send, and **wait for the receipt** — in that order, once.

        The receipt is not an optimisation, it is the nonce discipline: a loop that
        sends without waiting keeps a stale local idea of what is pending and produces a
        train of transactions that replace each other or never land. So the call returns
        only when the chain has said what happened, and any failure drops the local
        nonce so the next call asks the node the truth.

        A mined transaction with ``status == 0`` is a refusal too: it cost gas, moved
        nothing, and reporting it as success would put a position in the book that the
        venue never took.

        The record carries ``gas`` (the limit the node accepted) *and* the two facts the
        chain reported — ``gas_used`` and ``effective_gas_price`` — because those are what
        :meth:`agent.physics.transaction_cost_analyzer.TransactionCostAnalyzer.from_receipt`
        prices the next trade with: a limit is an intention, and a receipt is a receipt.
        """
        self._connect()
        sender = self._checksum(self._signer_address())
        try:
            gas = int(function.estimate_gas({"from": sender}))
            tx = function.build_transaction(
                {
                    "from": sender,
                    "nonce": self._next_nonce(),
                    "chainId": int(self._chain_id or 0),
                    "gas": gas,
                    **self._fee(),
                }
            )
            digest = self._dispatch(tx)
            receipt = self._w3().eth.wait_for_transaction_receipt(
                digest, timeout=float(self.config.receipt_timeout)
            )
            row = dict(receipt) if receipt is not None else {}
        except TestnetNotConfigured:
            self._nonce = None
            raise
        except Exception as exc:
            self._nonce = None
            raise StakeRefused(
                f"{action} {address} ({float(amount)!r} credits) failed: "
                f"{_failure_kind(exc)} ({self._reason(exc)})"
            ) from exc
        record = {
            "action": action,
            "address": address,
            "amount": round(float(amount), 6),
            "units": int(units),
            "nonce": int(tx["nonce"]),
            "gas": int(gas),
            "gas_used": int(row.get("gasUsed") or 0),
            "effective_gas_price": int(row.get("effectiveGasPrice") or 0),
            "hash": self._hex(digest),
            "block": int(row.get("blockNumber") or 0),
            "status": int(row.get("status") or 0),
        }
        if record["status"] != 1:
            raise StakeRefused(
                f"{action} {address} was mined in block {record['block']} and reverted "
                f"(tx {record['hash']})"
            )
        self._block = max(int(self._block), int(record["block"]))
        return record

    # -- the pieces a call is made of --------------------------------------- #

    def _contract(self, address: str, abi: Any) -> Any:
        """A contract handle at ``address`` with that ABI — building one sends no tx."""
        return self._w3().eth.contract(address=self._checksum(address), abi=list(abi))

    def _vault(self, address: str) -> Any:
        """The staking vault the operator's book points at."""
        return self._contract(address, self.config.vault_abi)

    def _token_contract(self, address: str) -> Any:
        """The ERC20 the vault pulls from."""
        return self._contract(address, ERC20_ABI)

    def _token_of(self, entry: ProtocolEntry) -> str:
        """The ERC20 this vault pulls from: the book's answer, or the vault's own."""
        if entry.token:
            return entry.token
        if entry.address in self._tokens:
            return self._tokens[entry.address]
        try:
            function = self._vault(entry.address).functions.token()
            token = str(function.call({"from": self._sender_or_zero()}))
        except Exception as exc:
            raise TestnetNotConfigured(
                f"{entry.name}: no token in the book and none readable from the vault "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        if not _is_address(token):
            raise TestnetNotConfigured(f"{entry.name}: the vault points at {token!r}")
        self._tokens[entry.address] = token.lower()
        return self._tokens[entry.address]

    def _units(self, amount: float, decimals: int) -> int:
        """Credits to token units, refusing anything that would round to nothing."""
        value = float(amount)
        if not np.isfinite(value):
            raise TestnetNotConfigured(f"{amount!r} credits is not a number")
        units = int(round(value * (10 ** int(decimals))))
        if units <= 0:
            raise TestnetNotConfigured(
                f"{value!r} credits is below one unit at {int(decimals)} decimals"
            )
        return units

    def _refuse(self, action: str, address: str, amount: float, reason: str) -> None:
        """Record a policy refusal: nothing was signed, and the reason is kept."""
        self.refusals.append(
            {
                "action": action,
                "address": str(address).lower(),
                "amount": round(float(amount), 6),
                "reason": reason,
                "step": int(self.step),
            }
        )

    # -- what the wallet holds ---------------------------------------------- #

    def balance(
        self,
        protocol_address: Optional[str] = None,
        *,
        address: Optional[str] = None,
    ) -> Balance:
        """Read the signer's balance off the node — gas money, and one vault's credit.

        Asked on every cycle rather than remembered, because the number a loop sizes a
        trade against is the number the chain will check: a cached balance is a balance
        that was true once, and ``MAX_TRADE_SIZE_USD`` against a stale wallet is how a
        bot spends money it already spent.

        Two reads, and the second one is optional:

        * ``eth_getBalance`` — always, in wei, reported in ether (:data:`ETH_DECIMALS`);
        * ``balanceOf`` — only when a ``protocol_address`` from the operator's book is
          given, because the credit a stake moves is the token *that vault* pulls.

        A failure is returned as ``Balance(error=...)`` rather than raised: the caller is
        a loop, it has to keep cycling, and "I could not read the wallet" is a reason to
        refuse a trade, not a reason to stop existing. A missing signer *is* raised
        (``TestnetNotConfigured``), because there is no address to ask about at all.
        """
        owner = str(address).lower() if address else self._signer_address()
        self._connect()
        block = self._refresh_block()
        base: Dict[str, Any] = {
            "address": owner,
            "chain_id": int(self._chain_id or 0),
            "block": int(block),
        }
        try:
            units = int(self._w3().eth.get_balance(self._checksum(owner)))
        except Exception as exc:  # noqa: BLE001 - a read that failed is a fact
            return Balance(
                **base,
                error=f"native balance: {_failure_kind(exc)} ({self._reason(exc)})",
            )
        native = units / (10 ** ETH_DECIMALS)
        if not protocol_address:
            return Balance(**base, native_eth=native, native_units=units)

        key = str(protocol_address).lower()
        entry = self.config.book.get(key)
        if entry is None:
            return Balance(
                **base,
                native_eth=native,
                native_units=units,
                error="the vault is not in the operator's address book",
            )
        decimals = int(entry.decimals)
        try:
            token = self._token_of(entry)
            raw = int(
                self._token_contract(token)
                .functions.balanceOf(self._checksum(owner))
                .call({"from": self._checksum(owner)})
            )
        except Exception as exc:  # noqa: BLE001 - the same rule as above
            return Balance(
                **base,
                native_eth=native,
                native_units=units,
                decimals=decimals,
                error=f"token balance: {_failure_kind(exc)} ({self._reason(exc)})",
            )
        return Balance(
            **base,
            native_eth=native,
            native_units=units,
            token=str(token),
            token_units=raw,
            token_balance=raw / (10 ** decimals),
            decimals=decimals,
            **self._position_of(entry, owner),
        )

    def _position_of(self, entry: ProtocolEntry, owner: str) -> Dict[str, Any]:
        """What the venue already holds of this entry — the aToken balance, when there is one.

        Only an ``aave-v3`` entry has a receipt a wallet can be asked about: the supplied
        asset leaves as USDC and comes back as aToken, which is why the amount grew. A vault
        keeps its own accounting, and a balance report that invented a receipt for it would be
        guessing at a contract's internals.

        A read that fails is a *position of unknown size*, and that is said with an error on
        the balance rather than a zero — the loop treats a non-empty error as a refusal to
        trade, which is the right response to "I do not know what I already have in there".
        """
        if str(entry.kind) != AAVE_KIND:
            return {}
        deployment = self._deployment_or_none(entry)
        if deployment is None or not deployment.a_token:
            return {}
        try:
            raw = int(
                self._token_contract(deployment.a_token)
                .functions.balanceOf(self._checksum(owner))
                .call({"from": self._checksum(owner)})
            )
        except Exception as exc:  # noqa: BLE001 - the same rule as a native read
            return {
                "position_token": deployment.a_token,
                "error": f"position balance: {_failure_kind(exc)} ({self._reason(exc)})",
            }
        return {
            "position_token": deployment.a_token,
            "position_units": raw,
            "position_balance": raw / (10 ** int(deployment.decimals)),
        }

    # -- capital in, capital out -------------------------------------------- #

    def execute_stake(self, protocol_address: str, amount: float) -> bool:
        """Move credits into a vault: approve if the vault cannot pull, then stake."""
        return self._move("stake", protocol_address, amount)

    def withdraw(self, protocol_address: str, amount: float) -> bool:
        """Move credits out of a vault, in the vault's own units."""
        return self._move("withdraw", protocol_address, amount)

    def _move(self, action: str, protocol_address: str, amount: float) -> bool:
        """The one path both directions take, so neither can skip a guard.

        Order matters and is the same on both sides: the operator's book, then the
        amount, then the signer, then the chain. Nothing reaches the node until the
        first three have said yes, which is why a refusal costs no gas.
        """
        key = str(protocol_address).lower()
        entry = self.config.book.get(key)
        try:
            value = float(amount)
        except (TypeError, ValueError):
            self._refuse(action, key, 0.0, f"{amount!r} is not an amount")
            return False
        if entry is None:
            self._refuse(action, key, value, "not in the operator's address book")
            return False
        if not np.isfinite(value) or value <= 0.0:
            self._refuse(action, key, value, "a non-positive amount")
            return False
        if not self.config.can_sign:
            raise TestnetNotConfigured(
                f"{action} needs a signer: set {ENV_KEY} or {ENV_UNLOCKED}"
            )
        self._connect()
        units = self._units(value, entry.decimals)
        if str(entry.kind) == AAVE_KIND:
            return self._move_aave(action, entry, value, units)
        if action == "stake":
            self._ensure_allowance(entry, units)
            function = self._vault(entry.address).functions.stake(units)
            self.stakes.append(
                self._send(
                    function,
                    action="stake",
                    address=entry.address,
                    amount=value,
                    units=units,
                )
            )
        else:
            function = self._vault(entry.address).functions.withdraw(units)
            self.withdrawals.append(
                self._send(
                    function,
                    action="withdraw",
                    address=entry.address,
                    amount=value,
                    units=units,
                )
            )
        return True

    def _move_aave(
        self, action: str, entry: ProtocolEntry, value: float, units: int
    ) -> bool:
        """Aave's two calls, with the same guards the vault path goes through.

        ``supply`` needs the Pool to be allowed to pull the asset, so the approval this file
        already knows how to make is made first — to ``entry.address``, which *is* the Pool,
        which is why an Aave entry's address cannot be the aToken (a claim on the pool cannot
        pull the asset) and cannot be invented (the table in :mod:`agent.physics.aave_v3`
        holds the pairs this venue will sign into).

        The record of a supply says ``supply`` and not ``stake``: the two are different
        functions on different contracts, and a log that used one word for both would be the
        first place a report started lying.
        """
        deployment = self._deployment(entry)
        owner = self._signer_address()
        pool = self._contract(deployment.pool, aave_v3.pool_abi()).functions
        if action == "stake":
            self._ensure_allowance(entry, units)
            self.stakes.append(
                self._send(
                    pool.supply(*aave_v3.supply_args(deployment, units, owner)),
                    action="supply",
                    address=entry.address,
                    amount=value,
                    units=units,
                )
            )
        else:
            self.withdrawals.append(
                self._send(
                    pool.withdraw(*aave_v3.withdraw_args(deployment, units, owner)),
                    action="withdraw",
                    address=entry.address,
                    amount=value,
                    units=units,
                )
            )
        return True

    def _ensure_allowance(self, entry: ProtocolEntry, units: int) -> None:
        """Let the venue pull ``units`` — once, and only if it cannot already.

        An ``approve`` is a transaction like any other and is awaited like one; a stake
        that assumed its own approval had landed is the classic way a stake reverts on a
        chain that is one block behind. The spender is ``entry.address``, which is the vault
        for a ``vault`` entry and the Aave Pool for an ``aave-v3`` one — both are the contract
        that pulls the asset, and neither can be anything else.
        """
        token = self._token_of(entry)
        owner = self._checksum(self._signer_address())
        spender = self._checksum(entry.address)
        contract = self._token_contract(token)
        try:
            allowed = int(
                contract.functions.allowance(owner, spender).call({"from": owner})
            )
        except Exception as exc:
            raise StakeRefused(
                f"{entry.name}: the allowance cannot be read from {token} "
                f"({_failure_kind(exc)}: {self._reason(exc)})"
            ) from exc
        if allowed >= units:
            return
        approve = self._token_contract(token).functions.approve(spender, units)
        self.approvals.append(
            self._send(
                approve,
                action="approve",
                address=token,
                amount=units / (10 ** int(entry.decimals)),
                units=units,
            )
        )

    # -- evidence ----------------------------------------------------------- #

    def submit_validation_proof(
        self, protocol_address: str, proof_data: Mapping[str, Any]
    ) -> bool:
        """Hand the venue the *fingerprint* of a verdict, signed.

        The evidence itself never goes on a chain — it is a paragraph, and a paragraph
        costs gas by the byte. What is written is ``keccak`` of the canonical JSON,
        which is enough for anyone holding the paragraph to check that this venue was
        given it, at that block, by that account.

        Never raises: ``False`` with a reason in :attr:`failures` is what a long run can
        survive, and the caller's contract is only "did the venue take it".
        """
        key = str(protocol_address).lower()
        entry = self.config.book.get(key)
        if entry is None:
            self._refuse("proof", key, 0.0, "not in the operator's address book")
            return False
        if not proof_data:
            self._refuse("proof", key, 0.0, "no evidence to hand over")
            return False
        if not self.config.can_sign:
            self._refuse("proof", key, 0.0, "this venue has no signer")
            return False
        try:
            digest = self._digest(proof_data)
            function = self._vault(entry.address).functions.submitValidationProof(
                digest
            )
            record = self._send(function, action="proof", address=entry.address)
        except (StakeRefused, TestnetError) as exc:
            self.failures.append(
                {"action": "proof", "address": key, "reason": str(exc)}
            )
            return False
        except Exception as exc:  # a client that failed before reaching the chain
            self.failures.append(
                {
                    "action": "proof",
                    "address": key,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            return False
        self.proofs.append({**record, "digest": "0x" + digest.hex()})
        return True

    def _digest(self, proof_data: Mapping[str, Any]) -> bytes:
        """``keccak`` of the canonical JSON: same evidence, same 32 bytes, always."""
        payload = json.dumps(
            dict(proof_data), sort_keys=True, separators=(",", ":"), default=str
        )
        return bytes(self._w3().keccak(text=payload))

    # -- what would happen -------------------------------------------------- #

    def dry_run(
        self,
        action: str,
        protocol_address: str,
        amount: float = 0.0,
        proof_data: Optional[Mapping[str, Any]] = None,
    ) -> DryRun:
        """Simulate one call against the chain as it stands. Nothing is signed.

        ``eth_call`` answers "would it revert", ``estimate_gas`` answers "and at what
        cost" — both without a transaction, a nonce or a fee. The two are reported
        separately because they fail separately: a call can be perfectly valid and still
        be unaffordable, and the difference is the whole reason to look before signing.

        An approval that has not happened yet shows up here as a revert, which is the
        honest answer to *"could I stake right now"* — a dry run describes the chain,
        not the chain with this file's optimism applied to it.
        """
        kind = str(action).lower()
        key = str(protocol_address).lower()
        entry = self.config.book.get(key)
        if entry is None:
            return DryRun(
                action=kind,
                address=key,
                ok=False,
                amount=float(amount),
                revert="not in the operator's address book",
            )
        if kind not in {"stake", "withdraw", "proof"}:
            raise TestnetNotConfigured(
                f"no dry run for {action!r}: stake, withdraw, proof"
            )
        try:
            self._connect()
            units = self._units(float(amount), entry.decimals) if kind != "proof" else 0
            function = self._function_for(kind, entry, units, proof_data or {})
            sender = self._sender_or_zero()
            try:
                function.call({"from": sender})
            except Exception as exc:
                run = DryRun(
                    action=kind,
                    address=entry.address,
                    ok=False,
                    amount=float(amount),
                    units=units,
                    sender=sender,
                    revert=f"{_failure_kind(exc)}: {self._reason(exc)}",
                )
                self.dry_runs.append(run)
                return run
            gas = int(function.estimate_gas({"from": sender}))
            run = DryRun(
                action=kind,
                address=entry.address,
                ok=True,
                amount=float(amount),
                units=units,
                gas=gas,
                sender=sender,
            )
        except TestnetError as exc:
            run = DryRun(
                action=kind,
                address=key,
                ok=False,
                amount=float(amount),
                revert=str(exc),
            )
        self.dry_runs.append(run)
        return run

    def _function_for(
        self, kind: str, entry: ProtocolEntry, units: int, proof: Mapping[str, Any]
    ) -> Any:
        """The exact contract function a real call of this kind would send."""
        if str(entry.kind) == AAVE_KIND:
            deployment = self._deployment(entry)
            owner = self._sender_or_zero()
            pool = self._contract(deployment.pool, aave_v3.pool_abi()).functions
            if kind == "stake":
                return pool.supply(*aave_v3.supply_args(deployment, units, owner))
            if kind == "withdraw":
                return pool.withdraw(*aave_v3.withdraw_args(deployment, units, owner))
            raise TestnetNotConfigured(
                f"{entry.name}: an Aave position has no evidence call — there is no "
                "submitValidationProof on a Pool, and inventing one would be a dry run of "
                "a function nobody will ever answer"
            )
        vault = self._vault(entry.address).functions
        if kind == "stake":
            return vault.stake(units)
        if kind == "withdraw":
            return vault.withdraw(units)
        return vault.submitValidationProof(self._digest(proof))

    # -- what it can say ----------------------------------------------------- #

    def _record_labels(self) -> Dict[str, str]:
        """``{entry name: where its last record came from}`` — the report's provenance table."""
        labels: Dict[str, str] = {}
        for entry in self.config.book:
            source = self._history.get(entry.address, "")
            if source:
                labels[entry.name] = source
        return labels

    def describe(self) -> Dict[str, Any]:
        """The venue as a report line: who, where, what clock, what happened.

        Deliberately free of the endpoint and of the key: an RPC path is usually an API
        key and a report usually ends up in a notebook someone shares. The host is
        enough to know which chain you were talking to.
        """
        line: Dict[str, Any] = {
            "name": str(self.name),
            "step": int(self.step),
            "protocols": len(self.config.book),
            "stakes": len(self.stakes),
            "withdrawals": len(self.withdrawals),
            "approvals": len(self.approvals),
            "proofs": len(self.proofs),
            "refusals": len(self.refusals),
            "failures": len(self.failures),
            "dry_runs": len(self.dry_runs),
            "book": [entry.name for entry in self.config.book],
            "kinds": {entry.name: str(entry.kind) for entry in self.config.book},
            "aave": list(aave_v3.describe(self._chain_id)),
            "history": str(self.config.history),
            "records": self._record_labels(),
            "config": self.config.describe(),
            "errors": list(self.errors[-5:]),
        }
        try:
            self._connect()
            line["chain_id"] = int(self._chain_id or 0)
            line["block_number"] = int(self._block)
            line["origin_block"] = int(self._origin or 0)
        except TestnetError as exc:
            line["unavailable"] = str(exc)
        return line

    def __repr__(self) -> str:
        return (
            f"RealTestnetProvider(rpc_host={self.config.rpc_host!r}, "
            f"chain_id={self._chain_id!r}, protocols={len(self.config.book)}, "
            f"signer={self.config.describe()['signer']})"
        )


#: How a web3 exception reads in a refusal. Matched by class *name*, walking the MRO,
#: rather than by importing the classes at module level: this file must stay importable
#: without web3, and every client that borrows web3's names gets the same wording.
FAILURE_KINDS: Dict[str, str] = {
    "ContractLogicError": "the call reverted",
    "InsufficientFunds": "not enough to pay for gas",
    "TimeExhausted": "no receipt before the timeout",
    "TransactionNotFound": "the node does not know that transaction",
    "BadFunctionCallOutput": "the address does not answer like a vault",
    "MismatchedABI": "the vault has no such function",
    "Web3RPCError": "the node refused the request",
    "Web3ValidationError": "the node refused the values",
    "ProviderConnectionError": "the node could not be reached",
}


def _failure_kind(exc: BaseException) -> str:
    """The most specific honest description of a failed call that this file has."""
    for cls in type(exc).__mro__:
        label = FAILURE_KINDS.get(cls.__name__)
        if label:
            return label
    if isinstance(exc, OSError):
        return "the node could not be reached"
    return "the venue refused"
