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
``TESTNET_PROTOCOLS_FILE``     path to the operator's address book (JSON).
``TESTNET_PROTOCOLS``          the same book inline, as JSON.
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
          "note": "sepolia demo"
        }
      ]
    }

An address not in this file is not published, is not on the whitelist and cannot be
staked into: :meth:`RealTestnetProvider.execute_stake` answers ``False`` for it, which
``Portfolio`` turns into a refusal, which is exactly what a closed list is for.

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
    "DryRun",
    "ProtocolBook",
    "ProtocolEntry",
    "RealTestnetProvider",
    "TestnetConfig",
    "TestnetError",
    "TestnetNotConfigured",
    "TestnetUnavailable",
    "MAINNET_CHAIN_IDS",
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

#: Credits per token unit when nobody says otherwise: a USDC-like token, because a
#: credit is a cent-shaped thing and six decimals is what a test stablecoin has.
DEFAULT_DECIMALS = 6

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
        object.__setattr__(self, "address", str(self.address).lower())
        object.__setattr__(self, "token", str(self.token).lower())
        object.__setattr__(self, "claim_apy", float(self.claim_apy))
        object.__setattr__(self, "decimals", int(self.decimals))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "name": self.name,
            "bucket": self.bucket,
            "claim_apy": self.claim_apy,
            "decimals": self.decimals,
            "token": self.token,
            "note": self.note,
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
    book: ProtocolBook = field(default_factory=ProtocolBook)
    decimals: int = DEFAULT_DECIMALS
    lookback: int = DEFAULT_LOOKBACK
    receipt_timeout: float = DEFAULT_RECEIPT_TIMEOUT
    block_timeout: float = DEFAULT_BLOCK_TIMEOUT
    allow_mainnet: bool = False
    abi: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        url = _clean(self.rpc_url)
        if not url:
            raise TestnetNotConfigured(
                f"no node: set {ENV_RPC} (this venue has no default endpoint)"
            )
        if "://" not in url:
            raise TestnetNotConfigured(
                f"{ENV_RPC} must carry a scheme (http, https or ws), got {url!r}"
            )
        object.__setattr__(self, "rpc_url", url)

        key = _clean(self.private_key)
        if key and not _is_private_key(key):
            # Never echo the value: a wrong key is still a key.
            raise TestnetNotConfigured(
                f"{ENV_KEY} is not 32 bytes of hex (its length is {len(key)})"
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
        """
        if env is None:
            _load_dotenv_once()
        source: Mapping[str, str] = os.environ if env is None else env
        requested = bool(overrides.pop("allow_mainnet", False))
        values: Dict[str, Any] = {
            "rpc_url": source.get(ENV_RPC, ""),
            "private_key": source.get(ENV_KEY, ""),
            "unlocked_account": source.get(ENV_UNLOCKED, ""),
            "chain_id": (
                _int_or(source.get(ENV_CHAIN_ID), 0, ENV_CHAIN_ID)
                if _clean(source.get(ENV_CHAIN_ID))
                else None
            ),
            "book": ProtocolBook.from_env(source),
            "decimals": _int_or(
                source.get(ENV_DECIMALS), DEFAULT_DECIMALS, ENV_DECIMALS
            ),
            "lookback": _int_or(
                source.get(ENV_LOOKBACK), DEFAULT_LOOKBACK, ENV_LOOKBACK
            ),
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
            "decimals": int(self.decimals),
            "lookback": int(self.lookback),
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
        **overrides: Any,
    ) -> None:
        """Build the venue; do not touch the network yet.

        The construction is pure on purpose: no connection, no ``chain_id``, no block —
        so importing, printing and describing a provider costs nothing and works on a
        machine that has never seen a chain. :meth:`_connect` is the first touch, and it
        happens on the first call that needs an answer.

        ``web3`` is a seam for tests (and for a node library other than web3.py): any
        object with an ``eth`` attribute answering the calls in this file will do.
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
        self._signer: str = config.unlocked_account
        self._account: Any = None
        self._chain_id: Optional[int] = config.chain_id
        self._origin: Optional[int] = None
        self._block: int = 0
        self._nonce: Optional[int] = None
        self._topic: Optional[str] = None
        self._claims: Dict[str, float] = {}
        self._tokens: Dict[str, str] = {}
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

    def get_protocol_list(self) -> List[Protocol]:
        """Every book entry that has published something, and nothing else.

        A book entry with no ``YieldDistributed`` log is *not* published: it has said
        nothing, so there is no record, and a protocol with no record cannot be one
        (``Protocol`` refuses it, correctly). Filtering here rather than inventing a
        placeholder keeps the absence of a measurement visible to the assessment, which
        will call it ``UNMEASURED`` and refuse to size a position on it.
        """
        self._connect()
        self._refresh_block()
        published: List[Protocol] = []
        for entry in self.config.book:
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
                        note=entry.note,
                    )
                )
            except PortfolioError as exc:  # a bad record is a protocol to skip
                self.errors.append(f"{entry.name}: {exc}")
        return published

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
        """The rates this address published, oldest first, decoded from its logs.

        Never raises on a venue problem: an address the book does not contain, a node
        that will not answer and a log that will not decode all come back as "nothing
        published" — which the assessment reads as a protocol to refuse. A
        *configuration* error is different and does surface, because the difference
        between "this venue said nothing" and "this venue points at a mainnet" must not
        be flattened into an empty array.
        """
        key = str(address).lower()
        if not self.config.book.has(key):
            return []
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

    def _ensure_allowance(self, entry: ProtocolEntry, units: int) -> None:
        """Let the vault pull ``units`` — once, and only if it cannot already.

        An ``approve`` is a transaction like any other and is awaited like one; a stake
        that assumed its own approval had landed is the classic way a stake reverts on a
        chain that is one block behind.
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
        vault = self._vault(entry.address).functions
        if kind == "stake":
            return vault.stake(units)
        if kind == "withdraw":
            return vault.withdraw(units)
        return vault.submitValidationProof(self._digest(proof))

    # -- what it can say ----------------------------------------------------- #

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
