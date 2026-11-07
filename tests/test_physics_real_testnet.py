"""First contact, without a network: a chain that is a recording double.

The provider that signs is the one file in this package that could lose money, so it is
tested the way a dangerous thing is tested — by watching exactly what it does and
proving that nothing else happens:

* **the translation.** Every interface call is checked against the *call* it produces on
  web3.py: which contract, which function, which units, which nonce, and what the answer
  was turned back into. A mapping nobody checks is a mapping that drifts;
* **the discipline.** Every send is followed by ``wait_for_transaction_receipt`` and
  preceded by an estimate: the two things that keep a loop from queueing transactions it
  can never land. Nothing is signed by a dry run — the test watches ``sent`` stay empty;
* **the failures.** A revert, a refused gas estimate, a reverted receipt and an address
  outside the operator's book arrive at the book as ``False`` or ``StakeRefused``, never
  as a position. ``web3.exceptions`` are matched by name, so the test does not need web3
  installed to prove it;
* **the secrets.** The file is parsed: no literal 64-hex key, no URL, no address. The
  configuration is read from a mapping and nowhere else, the endpoint is printed as a
  host, and the key is masked in every report.

``web3`` itself is not needed: :class:`FakeWeb3` remembers what it was asked. When web3
*is* installed and ``anvil`` is on the path, :func:`test_a_local_chain_answers_the_same
_seam` runs the real thing against a real EVM and skips itself politely otherwise.
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from agent.physics import portfolio as P
from agent.physics import real_testnet_provider as T
from agent.physics.defillama_adapter import apy_to_rate as feed_rate
from agent.physics.interfaces import IWeb3Provider

#: Aliased away from ``Test*`` so pytest does not try to collect them as test classes.
RealTestnetProvider = T.RealTestnetProvider
ProtocolBook = T.ProtocolBook
ProtocolEntry = T.ProtocolEntry
DryRun = T.DryRun
NetConfig = T.TestnetConfig
NetError = T.TestnetError
NetNotConfigured = T.TestnetNotConfigured
NetUnavailable = T.TestnetUnavailable
MAINNET_CHAIN_IDS = T.MAINNET_CHAIN_IDS

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "agent" / "physics" / "real_testnet_provider.py"

VAULT = "0x" + "11" * 20
TOKEN = "0x" + "22" * 20
SIGNER = "0x" + "33" * 20
OTHER = "0x" + "44" * 20

#: A key composed at runtime, so no 64-hex literal exists in this file either: a test
#: that carried one would be the thing the module is checked for not carrying.
KEY = "0x" + "ab" * 32

SEPOLIA = 11155111


class ContractLogicError(Exception):
    """Stand-in for ``ContractLogicError``: matched by name, not by import."""


class TimeExhausted(Exception):
    """Stand-in for ``web3.exceptions.TimeExhausted``."""


class InsufficientFunds(Exception):
    """Stand-in for ``web3.exceptions.InsufficientFunds``."""


def yield_log(rate: float, block: int, index: int = 0) -> Dict[str, Any]:
    """One ``YieldDistributed(epoch, rateE18)`` log, as a node would hand it over."""
    word = int(round(float(rate) * 10 ** 18))
    return {
        "address": VAULT,
        "blockNumber": block,
        "logIndex": index,
        "data": "0x" + word.to_bytes(32, "big").hex(),
        "topics": ["0x" + "99" * 32, "0x" + (1).to_bytes(32, "big").hex()],
    }


class World:
    """What the fake chain remembers: its state, and everything it was asked.

    One object, so a test can assert on the whole conversation at once: the contract
    calls, the estimates, the transactions actually sent, the receipts waited for and
    the log filters scanned. Nothing here talks to anything.
    """

    def __init__(self) -> None:
        self.height = 100
        self.chain = SEPOLIA
        self.gas_price = 7
        self.nonce = 0
        self.gas = 51_000
        self.logs: List[Dict[str, Any]] = []
        self.receipt: Optional[Dict[str, Any]] = {"status": 1, "blockNumber": 101}
        self.allowance = 0
        self.claim = int(0.07 * 10 ** 18)
        self.token = TOKEN
        #: What the wallet holds: 0.008 ETH of gas money and 50 credits. Read by
        #: ``eth_getBalance`` and ``balanceOf`` — the two calls a sizing loop makes.
        self.native = int(0.008 * 10 ** 18)
        self.token_balance = int(50 * 10 ** 6)
        self.reverts: Dict[str, Exception] = {}
        self.gas_refusals: Dict[str, Exception] = {}
        self.receipt_error: Optional[Exception] = None
        self.stalled = False
        self.calls: List[Tuple[str, str, Tuple[Any, ...], Dict[str, Any]]] = []
        self.filters: List[Dict[str, Any]] = []
        self.counts: List[Tuple[str, str]] = []
        self.sent: List[Any] = []
        self.waits: List[Any] = []
        #: The ``to`` of the last transaction a call was built for — so ``send_raw`` can say
        #: where the money went without the signed bytes having to be decoded.
        self.built_to = ""

    # -- assertion helpers --------------------------------------------------- #

    def names(self, kind: str) -> List[str]:
        """The function names called, estimated or built, in the order they happened."""
        return [name for action, name, _, _ in self.calls if action == kind]

    def sent_to(self) -> List[str]:
        """Every ``to`` a transaction was actually submitted with."""
        return [str(tx["to"]) for tx in self.sent]


class FakeCall:
    """One contract function call: ``stake(units)``, and what would happen to it."""

    def __init__(
        self, world: World, contract: str, name: str, args: Tuple[Any, ...]
    ) -> None:
        self.world = world
        self.contract = str(contract)
        self.name = name
        self.args = args

    def call(self, tx: Optional[Dict[str, Any]] = None) -> Any:
        self.world.calls.append(("call", self.name, self.args, dict(tx or {})))
        if self.name in self.world.reverts:
            raise self.world.reverts[self.name]
        if self.name == "allowance":
            return self.world.allowance
        if self.name == "balanceOf":
            return self.world.token_balance
        if self.name == "claimApyE18":
            return self.world.claim
        if self.name == "token":
            return self.world.token
        return None

    def estimate_gas(self, tx: Optional[Dict[str, Any]] = None) -> int:
        self.world.calls.append(("estimate_gas", self.name, self.args, dict(tx or {})))
        if self.name in self.world.gas_refusals:
            raise self.world.gas_refusals[self.name]
        return int(self.world.gas)

    def build_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        self.world.calls.append(("build", self.name, self.args, dict(tx)))
        self.world.built_to = self.contract
        return {
            **tx,
            "to": self.contract,
            "data": "0x" + self.name.encode().hex(),
            "value": 0,
        }


class FakeFunctions:
    """``contract.functions``: any attribute is a function of that name."""

    def __init__(self, world: World, address: str = "") -> None:
        self._world = world
        self._address = str(address)

    def __getattr__(self, name: str) -> Any:
        def factory(*args: Any) -> FakeCall:
            return FakeCall(self._world, self._address, name, args)

        return factory


class FakeContract:
    """A contract handle: an address and a namespace of functions."""

    def __init__(self, world: World, address: str, abi: Any) -> None:
        self.address = address
        self.abi = abi
        self.functions = FakeFunctions(world, address)


class FakeAccount:
    """``eth.account``: the two calls a locally signed transaction makes."""

    def from_key(self, key: str) -> Any:
        class _Account:
            address = SIGNER

        return _Account()

    def sign_transaction(self, tx: Dict[str, Any], key: str) -> Any:
        class _Signed:
            raw_transaction = b"\x01" * 32
            rawTransaction = raw_transaction
            transaction = tx

        return _Signed()


class FakeEth:
    """``web3.eth``: the namespace every call in the provider goes through."""

    def __init__(self, world: World) -> None:
        self._world = world
        self.account = FakeAccount()

    @property
    def block_number(self) -> int:
        return int(self._world.height)

    @property
    def chain_id(self) -> int:
        return int(self._world.chain)

    @property
    def gas_price(self) -> int:
        return int(self._world.gas_price)

    def get_transaction_count(self, address: str, block: str = "latest") -> int:
        self._world.counts.append((str(address), block))
        return int(self._world.nonce)

    def get_balance(self, address: str, block: str = "latest") -> int:
        self._world.counts.append((str(address), f"balance:{block}"))
        if "get_balance" in self._world.reverts:
            raise self._world.reverts["get_balance"]
        return int(self._world.native)

    def get_logs(self, filter_: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._world.filters.append(dict(filter_))
        return list(self._world.logs)

    def contract(self, address: str = "", abi: Any = None) -> FakeContract:
        return FakeContract(self._world, address, abi)

    def send_raw_transaction(self, raw: Any) -> bytes:
        self._world.sent.append(
            {"raw": raw, "to": self._world.built_to or VAULT}
        )
        self._world.height += 1  # instant mining: a sent transaction is a block
        return b"\x02" * 32

    def send_transaction(self, tx: Dict[str, Any]) -> bytes:
        self._world.sent.append(dict(tx))
        self._world.height += 1
        return b"\x03" * 32

    def wait_for_transaction_receipt(
        self, digest: Any, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        self._world.waits.append((digest, timeout))
        if self._world.receipt_error is not None:
            raise self._world.receipt_error
        return self._world.receipt


class FakeWeb3:
    """A node-shaped object: ``eth``, ``to_checksum_address`` and ``keccak``."""

    def __init__(self, world: World) -> None:
        self._world = world
        self.eth = FakeEth(world)

    def to_checksum_address(self, address: str) -> str:
        return str(address)

    def keccak(self, text: str = "", hexstr: str = "") -> bytes:
        return b"\x99" * 32


# ── fixtures ────────────────────────────────────────────────────────────────


def book_payload(**overrides: Any) -> Dict[str, Any]:
    entry = {
        "address": VAULT,
        "name": "VaultA",
        "bucket": "active",
        "claim_apy": 0.05,
        "decimals": 6,
        "token": TOKEN,
        "note": "a test vault",
    }
    entry.update(overrides)
    return {"protocols": [entry]}


def venue_env(**overrides: str) -> Dict[str, str]:
    """The environment a configured operator would have — and nothing else."""
    env = {
        "TESTNET_RPC_URL": "http://127.0.0.1:8545/v2/secret-key-in-the-path",
        "TESTNET_PRIVATE_KEY": KEY,
        "TESTNET_PROTOCOLS": json.dumps(book_payload()),
        "TESTNET_BLOCK_TIMEOUT": "0.05",
        "TESTNET_RECEIPT_TIMEOUT": "5",
    }
    env.update(overrides)
    return env


@pytest.fixture()
def world() -> World:
    """A chain that pays 0.0004 a step, eight times, in blocks 90..97."""
    made = World()
    made.logs = [yield_log(0.0004, 90 + index) for index in range(8)]
    return made


@pytest.fixture()
def provider(world: World) -> RealTestnetProvider:
    return RealTestnetProvider(env=venue_env(), web3=FakeWeb3(world))


@pytest.fixture()
def read_only(world: World) -> RealTestnetProvider:
    env = venue_env()
    env.pop("TESTNET_PRIVATE_KEY")
    return RealTestnetProvider(env=env, web3=FakeWeb3(world))


# ── the interface, answered ─────────────────────────────────────────────────


def test_the_provider_answers_the_whole_contract(provider):
    """Five operations and three facts — the same list the mock is held to."""
    for name in (
        "get_protocol_list",
        "fetch_protocol_history",
        "submit_validation_proof",
        "execute_stake",
        "withdraw",
        "advance",
        "describe",
    ):
        assert callable(getattr(provider, name))
    assert isinstance(provider, IWeb3Provider)  # the seam, checked at runtime
    assert provider.name == "testnet"
    assert isinstance(provider.step, int)
    assert provider.whitelist.allows(VAULT)
    assert not provider.whitelist.allows(OTHER)


def test_the_book_is_the_whitelist_and_it_cannot_be_widened(provider):
    """The approved addresses come from the operator's file, and only from it."""
    assert provider.whitelist.as_list() == [VAULT]
    assert not hasattr(provider.whitelist, "add")
    assert not hasattr(provider, "add_to_whitelist")


# ── configuration: the environment, and nothing else ────────────────────────


def test_configuration_is_read_from_the_environment(provider):
    config = provider.config
    assert config.rpc_url.endswith("secret-key-in-the-path")
    assert config.private_key == KEY
    assert config.receipt_timeout == pytest.approx(5.0)
    assert config.block_timeout == pytest.approx(0.05)
    assert len(config.book) == 1


def test_no_node_is_a_named_error_and_not_a_default():
    with pytest.raises(NetNotConfigured) as raised:
        RealTestnetProvider(env={})
    assert "TESTNET_RPC_URL" in str(raised.value)


def test_a_bare_host_is_not_an_endpoint():
    with pytest.raises(NetNotConfigured):
        RealTestnetProvider(env={"TESTNET_RPC_URL": "127.0.0.1:8545"})


def test_a_key_of_the_wrong_length_is_refused_without_being_echoed():
    shout = "0x" + "cd" * 31
    with pytest.raises(NetNotConfigured) as raised:
        RealTestnetProvider(env=venue_env(TESTNET_PRIVATE_KEY=shout))
    assert shout not in str(raised.value)
    assert "not 32 bytes" in str(raised.value)


def test_one_signer_and_not_two():
    with pytest.raises(NetNotConfigured):
        RealTestnetProvider(
            env=venue_env(TESTNET_UNLOCKED_ACCOUNT=SIGNER)
        )


def test_a_broken_book_is_an_error_not_an_empty_list():
    with pytest.raises(NetNotConfigured):
        RealTestnetProvider(env=venue_env(TESTNET_PROTOCOLS="{not json"))
    with pytest.raises(NetNotConfigured):
        ProtocolBook.from_payload({"protocols": [{"address": "0x1234", "name": "x"}]})
    with pytest.raises(NetNotConfigured):
        ProtocolBook.from_payload(
            {"protocols": [{"address": VAULT, "name": "x", "bucket": "moonshot"}]}
        )


def test_two_entries_for_one_address_is_refused():
    payload = book_payload()
    payload["protocols"].append(dict(payload["protocols"][0], name="VaultB"))
    with pytest.raises(NetNotConfigured):
        ProtocolBook.from_payload(payload)


def test_the_env_inline_book_and_a_file_book_both_load(tmp_path):
    path = tmp_path / "book.json"
    path.write_text(json.dumps(book_payload()), encoding="utf-8")
    from_file = ProtocolBook.from_env({"TESTNET_PROTOCOLS_FILE": str(path)})
    from_env = ProtocolBook.from_env({"TESTNET_PROTOCOLS": json.dumps(book_payload())})
    assert from_file.addresses() == from_env.addresses() == [VAULT]
    assert ProtocolBook.from_env({}).addresses() == []


def test_a_mainnet_needs_two_decisions(world):
    """A chain where credits are money is refused unless both say yes — twice over."""
    world.chain = 1
    mainnet = FakeWeb3(world)
    with pytest.raises(NetNotConfigured) as raised:
        RealTestnetProvider(env=venue_env(), web3=mainnet).get_protocol_list()
    assert "mainnet" in str(raised.value)

    # the environment alone is not enough
    permitted = venue_env(TESTNET_ALLOW_MAINNET="1")
    with pytest.raises(NetNotConfigured):
        RealTestnetProvider(env=permitted, web3=mainnet).get_protocol_list()

    # the argument alone is not enough either
    with pytest.raises(NetNotConfigured):
        RealTestnetProvider(
            env=venue_env(), web3=mainnet, allow_mainnet=True
        ).get_protocol_list()

    # both together are, and then the venue answers
    allowed = RealTestnetProvider(env=permitted, web3=mainnet, allow_mainnet=True)
    assert allowed.describe()["chain_id"] == 1
    assert 1 in MAINNET_CHAIN_IDS


def test_a_chain_id_that_disagrees_with_the_configuration_is_refused(world):
    venue = RealTestnetProvider(
        env=venue_env(TESTNET_CHAIN_ID="5"), web3=FakeWeb3(world)
    )
    with pytest.raises(NetNotConfigured) as raised:
        venue.get_protocol_list()
    assert "chain" in str(raised.value)


def test_a_node_that_does_not_answer_is_unavailable():
    class DeadEth(FakeEth):
        @property
        def block_number(self) -> int:
            raise ConnectionError("connection refused")

    class DeadWeb3(FakeWeb3):
        def __init__(self) -> None:
            self._world = World()
            self.eth = DeadEth(self._world)

    with pytest.raises(NetUnavailable) as raised:
        RealTestnetProvider(env=venue_env(), web3=DeadWeb3()).get_protocol_list()
    assert "127.0.0.1" in str(raised.value)


# ── the L2 profile: a spelling, not a permission ────────────────────────────
#
# Real gas prices make a small trade uneconomic on Ethereum and cheap on an L2, so the
# operator must be able to switch networks by editing variables and nothing else. What
# the profile is *not* allowed to do is make a real-money chain easier to sign on: the
# tests below hold both halves of that sentence.

ARBITRUM = 42161
ARBITRUM_SEPOLIA = 421614


def test_a_profile_supplies_the_chain_id_and_accepts_its_own_spellings():
    """``TESTNET_PROFILE=arbitrum`` and the names the node provider handed out."""
    config = NetConfig.from_env(
        {
            "TESTNET_PROFILE": "arbitrum",
            "ARBITRUM_RPC_URL": "http://arb.example/v3/secret-key-in-the-path",
            "ARBITRUM_PRIVATE_KEY": KEY,
        }
    )
    assert config.profile == "arbitrum"
    assert config.chain_id == ARBITRUM == T.CHAIN_PROFILES["arbitrum"]
    assert config.rpc_host == "arb.example"
    assert config.can_sign and config.private_key == KEY

    # Provenance, in names: the report can say where each value came from, and the
    # report still carries neither the endpoint nor the key.
    assert config.sources == {
        "TESTNET_RPC_URL": "ARBITRUM_RPC_URL",
        "TESTNET_PRIVATE_KEY": "ARBITRUM_PRIVATE_KEY",
    }
    line = config.describe()
    assert line["profile"] == "arbitrum" and line["chain_id"] == ARBITRUM
    printed = json.dumps(line)
    assert "secret-key-in-the-path" not in printed and KEY not in printed
    assert "abab" in printed  # the mask, and no more


def test_a_chain_id_in_the_profile_spelling_wins_over_the_preset():
    """A preset is a default, not a verdict: Arbitrum Sepolia is still reachable."""
    config = NetConfig.from_env(
        {
            "TESTNET_PROFILE": "arbitrum",
            "ARBITRUM_RPC_URL": "http://arb.example/v3/k",
            "ARBITRUM_CHAIN_ID": str(ARBITRUM_SEPOLIA),
        }
    )
    assert config.chain_id == ARBITRUM_SEPOLIA
    assert config.sources["TESTNET_CHAIN_ID"] == "ARBITRUM_CHAIN_ID"
    assert ARBITRUM_SEPOLIA not in MAINNET_CHAIN_IDS


def test_a_canonical_name_wins_over_the_profile_spelling():
    """One value, from one variable — the canonical one, when both are set."""
    config = NetConfig.from_env(
        {
            "TESTNET_PROFILE": "arbitrum",
            "TESTNET_RPC_URL": "http://canon.example/v3/k",
            "TESTNET_CHAIN_ID": "5",
            "TESTNET_PRIVATE_KEY": KEY,
            "ARBITRUM_RPC_URL": "http://arb.example/v3/k",
            "ARBITRUM_CHAIN_ID": str(ARBITRUM),
        }
    )
    assert config.rpc_host == "canon.example"
    assert config.chain_id == 5
    assert config.sources == {}


def test_a_profile_does_not_unlock_a_chain_where_credits_are_money(world):
    """An L2 is cheap, not free: 42161 is a mainnet here and still needs both yeses."""
    world.chain = ARBITRUM
    env = venue_env(TESTNET_PROFILE="arbitrum")
    env.pop("TESTNET_RPC_URL")
    env["ARBITRUM_RPC_URL"] = "http://arb.example/v3/k"

    with pytest.raises(NetNotConfigured) as raised:
        RealTestnetProvider(env=env, web3=FakeWeb3(world)).get_protocol_list()
    assert "mainnet" in str(raised.value)
    assert "TESTNET_ALLOW_MAINNET" in str(raised.value)

    # and with both yeses the venue answers on the chain the profile named
    allowed = RealTestnetProvider(
        env=dict(env, TESTNET_ALLOW_MAINNET="1"),
        web3=FakeWeb3(world),
        allow_mainnet=True,
    )
    assert allowed.describe()["chain_id"] == ARBITRUM
    assert allowed.config.profile == "arbitrum"


def test_an_unknown_profile_is_a_named_error():
    """A typo must not quietly leave the venue on some other chain."""
    with pytest.raises(NetNotConfigured) as raised:
        NetConfig.from_env(
            {
                "TESTNET_PROFILE": "optimism",
                "OPTIMISM_RPC_URL": "http://op.example/v3/k",
            }
        )
    message = str(raised.value)
    assert "TESTNET_PROFILE" in message and "arbitrum" in message


def test_a_bare_rpc_url_or_private_key_is_not_an_alias():
    """A stray variable from another project must not become a signer."""
    stray = {
        "RPC_URL": "http://other.example/v3/k",
        "PRIVATE_KEY": KEY,
        "CHAIN_ID": str(ARBITRUM),
    }
    with pytest.raises(NetNotConfigured) as raised:
        NetConfig.from_env(stray)
    assert "TESTNET_RPC_URL" in str(raised.value)
    with pytest.raises(NetNotConfigured):
        NetConfig.from_env(dict(stray, TESTNET_PROFILE="arbitrum"))


def test_no_chain_id_is_written_into_the_venue_by_default():
    """Chain-agnostic by construction: an id that is not given is not invented."""
    config = NetConfig.from_env({"TESTNET_RPC_URL": "http://any.example/v3/k"})
    assert config.chain_id is None and config.profile == ""
    assert config.sources == {}
    assert "DEFAULT_CHAIN_ID" not in MODULE.read_text(encoding="utf-8")
    # the chains this project recommends are all in the lock, so none of them is silent
    for chain in (ARBITRUM, 137, 8453):
        assert chain in MAINNET_CHAIN_IDS


# ── the clock is the chain's, and it is waited for ──────────────────────────


def test_the_step_is_blocks_lived_since_the_venue_connected(provider, world):
    assert provider.step == 0  # nothing read yet
    provider.get_protocol_list()
    assert provider.step == 0
    world.height += 3
    provider.advance(1)
    assert provider.step == 3


def test_advance_waits_and_reports_what_the_chain_gave(provider, world):
    """A step is a block the chain produced; the venue never manufactures one."""
    provider.get_protocol_list()
    assert provider.advance(1) == 0  # nothing was mined inside the (tiny) timeout
    assert provider.errors and "no block" in provider.errors[-1]
    world.height += 2
    assert provider.advance(1) == 2


# ── what is on offer ────────────────────────────────────────────────────────


def test_the_record_comes_from_the_logs_in_chain_order(provider, world):
    world.logs = [
        yield_log(0.0003, 95),
        yield_log(0.0001, 91),
        yield_log(0.0002, 93),
    ]
    assert provider.fetch_protocol_history(VAULT) == pytest.approx(
        [0.0001, 0.0002, 0.0003]
    )


def test_a_tail_of_the_record_can_be_asked_for(provider, world):
    world.logs = [yield_log(0.0001 * (index + 1), 90 + index) for index in range(6)]
    assert provider.fetch_protocol_history(VAULT, steps=2) == pytest.approx(
        [0.0005, 0.0006]
    )


def test_the_log_filter_names_the_vault_the_topic_and_a_window(provider, world):
    provider.fetch_protocol_history(VAULT)
    filter_ = world.filters[0]
    assert filter_["address"] == VAULT
    assert filter_["topics"] == ["0x" + "99" * 32]
    assert filter_["toBlock"] == "latest"
    assert filter_["fromBlock"] == 0  # a young chain: scan from its first block

    world.height = 60_000
    provider.fetch_protocol_history(VAULT)
    assert world.filters[1]["fromBlock"] == 60_000 - 5_000


def test_an_address_the_book_does_not_contain_publishes_nothing(provider, world):
    assert provider.fetch_protocol_history(OTHER) == []
    assert world.filters == []  # and costs no request


def test_an_unreadable_log_is_dropped_and_the_rest_survive(provider, world):
    broken = dict(yield_log(0.0004, 96), data="0x00")
    world.logs = [yield_log(0.0001, 90), broken, yield_log(0.0002, 97)]
    assert provider.fetch_protocol_history(VAULT) == pytest.approx([0.0001, 0.0002])
    assert any("unreadable log" in line for line in provider.errors)


def test_a_log_of_raw_bytes_reads_the_same_as_one_of_hex(provider, world):
    """web3 hands over ``HexBytes``; anything else hands over a string. Both decode."""
    raw = dict(yield_log(0.0004, 96))
    raw["data"] = bytes.fromhex(raw["data"][2:])
    world.logs = [raw]
    assert provider.fetch_protocol_history(VAULT) == pytest.approx([0.0004])


def test_a_protocol_with_no_logs_is_not_published(provider, world):
    """No record is not a zero-return protocol: it is one that has said nothing."""
    world.logs = []
    assert provider.get_protocol_list() == []


def test_the_published_protocol_carries_the_book_and_the_claim(provider, world):
    published = provider.get_protocol_list()
    assert len(published) == 1
    protocol = published[0]
    assert isinstance(protocol, P.Protocol)
    assert (protocol.name, protocol.bucket, protocol.address) == (
        "VaultA",
        "active",
        VAULT,
    )
    # the vault answers 0.07e18, so the chain's claim wins over the book's 0.05
    assert protocol.claim_apy == pytest.approx(0.07)
    assert protocol.record(8) == pytest.approx([0.0004] * 8)


def test_a_vault_that_cannot_answer_its_claim_falls_back_to_the_book(world):
    world.reverts["claimApyE18"] = ContractLogicError("execution reverted")
    venue = RealTestnetProvider(env=venue_env(), web3=FakeWeb3(world))
    assert venue.get_protocol_list()[0].claim_apy == pytest.approx(0.05)
    assert any("claim" in line for line in venue.errors)


# ── capital: approve, stake, wait, record ───────────────────────────────────


def test_a_stake_approves_then_stakes_and_waits_for_both_receipts(provider, world):
    assert provider.execute_stake(VAULT, 10.0) is True

    # credits are token units: 10.0 at six decimals is 10_000_000
    assert ("call", "allowance", (SIGNER, VAULT), {"from": SIGNER}) in world.calls
    assert ("estimate_gas", "stake", (10_000_000,), {"from": SIGNER}) in world.calls

    names = world.names("build")
    assert names == ["approve", "stake"]
    approve, stake = [call for call in world.calls if call[0] == "build"]
    assert approve[2] == (VAULT, 10_000_000)
    assert stake[2] == (10_000_000,)

    # every send is followed by a receipt, and the nonce counts up once per send
    assert len(world.sent) == 2
    assert len(world.waits) == 2
    assert [tx["nonce"] for tx in (approve[3], stake[3])] == [0, 1]
    assert {tx.get("chainId") for tx in (approve[3], stake[3])} == {SEPOLIA}
    assert all(tx.get("gas") == world.gas for tx in (approve[3], stake[3]))

    assert len(provider.approvals) == 1
    assert len(provider.stakes) == 1
    assert provider.stakes[0]["units"] == 10_000_000
    assert provider.stakes[0]["status"] == 1
    assert provider.stakes[0]["hash"] == "0x" + "02" * 32


def test_an_existing_allowance_is_not_approved_again(provider, world):
    world.allowance = 50_000_000
    assert provider.execute_stake(VAULT, 10.0) is True
    assert world.names("build") == ["stake"]
    assert provider.approvals == []


def test_a_withdrawal_is_one_signed_call(provider, world):
    world.allowance = 10 ** 12
    assert provider.withdraw(VAULT, 2.5) is True
    assert world.names("build") == ["withdraw"]
    assert ("build", "withdraw", (2_500_000,), {}) in [
        (kind, name, args, {}) for kind, name, args, _ in world.calls
    ]
    assert provider.withdrawals[0]["units"] == 2_500_000
    assert provider.stakes == []


def test_an_address_outside_the_book_is_refused_free_of_charge(provider, world):
    assert provider.execute_stake(OTHER, 10.0) is False
    assert provider.refusals[0]["reason"] == "not in the operator's address book"
    assert world.sent == []


def test_a_non_positive_amount_is_refused(provider, world):
    assert provider.execute_stake(VAULT, 0.0) is False
    assert provider.withdraw(VAULT, -1.0) is False
    assert {row["reason"] for row in provider.refusals} == {"a non-positive amount"}
    assert world.sent == []


def test_an_amount_below_one_unit_is_an_error_not_a_rounding(provider, world):
    with pytest.raises(NetNotConfigured):
        provider.execute_stake(VAULT, 1e-9)
    assert world.sent == []


# ── the failures a real venue produces ──────────────────────────────────────


def test_a_refused_gas_estimate_is_a_stake_refused(provider, world):
    world.gas_refusals["stake"] = InsufficientFunds("insufficient funds for gas")
    with pytest.raises(P.StakeRefused) as raised:
        provider.execute_stake(VAULT, 10.0)
    assert "not enough to pay for gas" in str(raised.value)
    assert "insufficient funds" in str(raised.value)
    assert provider.stakes == []
    built = [call for call in world.calls if call[0] == "build"]
    assert all(call[1] != "stake" for call in built)


def test_a_revert_in_the_estimate_is_a_stake_refused(provider, world):
    world.gas_refusals["stake"] = ContractLogicError("execution reverted: paused")
    with pytest.raises(P.StakeRefused) as raised:
        provider.execute_stake(VAULT, 10.0)
    assert "the call reverted" in str(raised.value)
    assert "paused" in str(raised.value)


def test_a_reverted_receipt_is_a_refusal_and_not_a_position(provider, world):
    world.receipt = {"status": 0, "blockNumber": 120}
    with pytest.raises(P.StakeRefused) as raised:
        provider.execute_stake(VAULT, 10.0)
    assert "reverted" in str(raised.value)
    assert provider.stakes == []


def test_a_missing_receipt_is_a_refusal(provider, world):
    world.receipt_error = TimeExhausted("timeout")
    with pytest.raises(P.StakeRefused) as raised:
        provider.execute_stake(VAULT, 10.0)
    assert "no receipt before the timeout" in str(raised.value)


def test_the_nonce_is_re_read_after_a_failure(provider, world):
    world.gas_refusals["stake"] = InsufficientFunds("no funds")
    with pytest.raises(P.StakeRefused):
        provider.execute_stake(VAULT, 10.0)
    world.gas_refusals.clear()
    world.nonce = 7
    assert provider.execute_stake(VAULT, 10.0) is True
    assert world.counts[-1] == (SIGNER, "pending")
    assert ("build", "stake", (10_000_000,), {}) in [
        (kind, name, args, {}) for kind, name, args, _ in world.calls
    ]


# ── evidence: a fingerprint on the chain, never the paragraph ────────────────


def test_a_proof_is_the_digest_of_the_evidence(provider, world):
    evidence = {"verdict": "HONEST", "step": 24, "score": 0.62}
    assert provider.submit_validation_proof(VAULT, evidence) is True

    (call,) = [
        item
        for item in world.calls
        if item[1] == "submitValidationProof" and item[0] == "estimate_gas"
    ]
    digest = call[2][0]
    assert len(digest) == 32 and digest == b"\x99" * 32
    assert provider.proofs[0]["digest"] == "0x" + "99" * 32
    assert provider.proofs[0]["action"] == "proof"
    assert len(world.sent) == 1 and len(world.waits) == 1


def test_the_same_evidence_hashes_the_same_way(provider):
    """Canonical JSON: the fingerprint must not depend on a dict's insertion order."""
    left = provider._digest({"a": 1, "b": 2})
    right = provider._digest({"b": 2, "a": 1})
    assert left == right


def test_a_proof_for_a_stranger_is_not_accepted(provider, world):
    assert provider.submit_validation_proof(OTHER, {"verdict": "HONEST"}) is False
    assert world.sent == []
    assert provider.refusals[-1]["reason"] == "not in the operator's address book"


def test_empty_evidence_is_not_accepted(provider, world):
    assert provider.submit_validation_proof(VAULT, {}) is False
    assert provider.refusals[-1]["reason"] == "no evidence to hand over"


def test_a_proof_that_reverts_is_false_with_a_reason_and_never_raises(provider, world):
    world.gas_refusals["submitValidationProof"] = ContractLogicError("reverted")
    assert provider.submit_validation_proof(VAULT, {"verdict": "TRAP"}) is False
    assert provider.proofs == []
    assert "reverted" in provider.failures[0]["reason"]


# ── a dry run signs nothing ─────────────────────────────────────────────────


def test_a_dry_run_answers_without_sending(provider, world):
    run = provider.dry_run("stake", VAULT, 10.0)
    assert isinstance(run, DryRun)
    assert run.ok is True
    assert run.gas == world.gas
    assert run.units == 10_000_000
    assert run.sender == SIGNER
    assert world.sent == [] and world.waits == []
    assert world.names("build") == []  # not even built, let alone signed
    assert len(provider.dry_runs) == 1


def test_a_dry_run_reports_a_revert_the_way_the_chain_states_it(provider, world):
    world.reverts["stake"] = ContractLogicError("execution reverted: StakeTooBig")
    run = provider.dry_run("stake", VAULT, 10.0)
    assert run.ok is False
    assert "StakeTooBig" in str(run.revert)
    assert "reverted" in str(run.revert)
    assert world.sent == []
    assert "would fail" in str(run)


def test_a_dry_run_for_a_stranger_is_a_refusal_not_an_exception(provider, world):
    run = provider.dry_run("stake", OTHER, 10.0)
    assert run.ok is False and "address book" in str(run.revert)
    assert world.sent == []


def test_a_dry_run_of_a_withdrawal_and_of_a_proof(provider, world):
    assert provider.dry_run("withdraw", VAULT, 1.0).units == 1_000_000
    proof_run = provider.dry_run("proof", VAULT, proof_data={"verdict": "HONEST"})
    assert proof_run.ok is True and proof_run.units == 0
    assert world.sent == []
    with pytest.raises(NetNotConfigured):
        provider.dry_run("liquidate", VAULT, 1.0)


# ── a venue with no key is a venue that can read ────────────────────────────


def test_a_read_only_venue_publishes_but_will_not_sign(read_only, world):
    assert read_only.can_sign is False
    assert len(read_only.get_protocol_list()) == 1  # reading is allowed
    with pytest.raises(NetNotConfigured) as raised:
        read_only.execute_stake(VAULT, 10.0)
    assert "signer" in str(raised.value)
    assert world.sent == []


def test_a_read_only_venue_declines_evidence_without_raising(read_only):
    assert read_only.submit_validation_proof(VAULT, {"verdict": "HONEST"}) is False
    assert read_only.refusals[-1]["reason"] == "this venue has no signer"


def test_an_unlocked_node_account_is_a_signer_without_a_key(world):
    env = venue_env()
    env.pop("TESTNET_PRIVATE_KEY")
    env["TESTNET_UNLOCKED_ACCOUNT"] = SIGNER
    venue = RealTestnetProvider(env=env, web3=FakeWeb3(world))
    world.allowance = 10 ** 12
    assert venue.execute_stake(VAULT, 1.0) is True
    assert world.names("build") == ["stake"]  # no approve needed
    assert "raw" not in world.sent[0]  # the node signed, nothing was signed here


# ── the seam holds from the book to a chain ─────────────────────────────────


def test_the_wallet_moves_capital_through_the_real_venue(provider):
    """The whole point of the seam: unchanged ``Portfolio`` code, a venue that signs."""
    book = P.Portfolio(whitelist=provider.whitelist, cash=1_000.0, provider=provider)
    protocol = provider.get_protocol_list()[0]

    assert book.deposit(protocol, 10.0, step=1) == pytest.approx(10.0)
    assert book.holdings[protocol.address].credits == pytest.approx(10.0)
    assert book.cash == pytest.approx(990.0)
    assert [row["units"] for row in provider.stakes] == [10_000_000]

    assert book.withdraw(protocol.address, 4.0, step=2) == pytest.approx(4.0)
    assert provider.withdrawals[0]["units"] == 4_000_000
    assert book.cash == pytest.approx(994.0)


def test_a_refused_stake_never_becomes_a_position(provider, world):
    world.gas_refusals["stake"] = InsufficientFunds("insufficient funds for gas")
    book = P.Portfolio(whitelist=provider.whitelist, cash=100.0, provider=provider)
    protocol = provider.get_protocol_list()[0]
    with pytest.raises(P.StakeRefused):
        book.deposit(protocol, 10.0, step=1)
    assert book.holdings == {}
    assert book.cash == pytest.approx(100.0)


def test_an_empire_can_trade_a_testnet_with_no_market_behind_it(provider):
    """``market`` is the simulation; a testnet is not one, and the report says so."""
    empire = P.Empire.launch(provider=provider, capital=1_000.0)
    assert empire.market is None
    assert empire.portfolio.provider is provider
    assert empire.portfolio.whitelist.as_list() == [VAULT]

    empire.advance(1)
    empire.review()
    report = empire.report()
    assert report["provider"]["name"] == "testnet"
    assert report["market"] is None
    assert empire.decisions and empire.proofs == []


# ── no endpoint, no key, no address in the file ─────────────────────────────


def test_the_file_carries_no_endpoint_no_key_and_no_address():
    """Prose may say where a node lives; code may not. Only code is checked twice."""
    source = MODULE.read_text(encoding="utf-8")
    assert not re.search(r"0x[0-9a-fA-F]{64}", source), "a private key in the source"
    addresses = set(re.findall(r"0x[0-9a-fA-F]{40}", source))
    assert addresses <= {"0x" + "0" * 40}, addresses  # only the null address

    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            continue
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            assert not re.search(r"[a-zA-Z]+://", node.value), node.value
            assert not re.search(r"0x[0-9a-fA-F]{64}", node.value), node.value


def test_web3_is_not_imported_at_import_time():
    """Parsed, not promised: the network library is only named inside a call."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    top: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.append(node.module.split(".")[0])
    assert "web3" not in top
    assert set(top) <= {
        "__future__",
        "dataclasses",
        "json",
        "os",
        "time",
        "typing",
        "urllib",
        "numpy",
        "agent",
    }, top

    def named(node: ast.AST) -> List[str]:
        names = [alias.name.split(".")[0] for alias in getattr(node, "names", ())]
        module = getattr(node, "module", None) or ""
        return names + [module.split(".")[0]]

    nested = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and "web3" in named(node)
    ]
    assert nested, "the lazy import is what makes a real call possible"
    for node in nested:
        assert node.col_offset > 0, "a module-level web3 import would tie the package"


def test_importing_the_package_leaves_web3_out_of_the_interpreter():
    """A subprocess, because a module already imported cannot prove anything."""
    program = "import agent.physics, sys; print('web3' in sys.modules)"
    done = subprocess.run(
        [sys.executable, "-c", program],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False", done.stdout


def test_the_report_prints_the_host_and_never_the_endpoint_or_the_key(provider):
    line = provider.describe()
    printed = json.dumps(line)
    assert "127.0.0.1" in printed  # the host, which is the useful half
    assert "secret-key-in-the-path" not in printed
    assert KEY not in printed and KEY[2:] not in printed
    assert "abab" in printed  # the mask shows the last four characters, and no more
    assert KEY not in repr(provider) and "secret-key" not in repr(provider)
    dry = NetConfig.from_env(venue_env(TESTNET_PRIVATE_KEY=""))
    assert dry.describe()["signer"] == "read-only"


# ── the real thing, when a chain is installed ───────────────────────────────
#
# The tests above prove the translation against a recording double. This one proves it
# against an EVM: a local Anvil, a token and a vault compiled by Foundry, real signing,
# real receipts, real balances. It is skipped — politely, and by name — on a machine
# without foundry, because a fixture pretending to be a chain is not evidence.

ANVIL = shutil.which("anvil")
FORGE = shutil.which("forge")

needs_a_local_chain = pytest.mark.skipif(
    not (ANVIL and FORGE),
    reason="needs anvil and forge on PATH (a local EVM); skipping rather than faking",
)

#: Anvil's documented development mnemonic. It funds nothing and secures nothing, and it
#: lives here rather than a key: the account is derived, never written down as hex.
ANVIL_MNEMONIC = " ".join(["test"] * 11 + ["junk"])

SOLIDITY = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract TestToken {
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    function mint(address to, uint256 amount) external {
        balanceOf[to] += amount;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        return true;
    }

    function transferFrom(address from, address to, uint256 amount)
        external
        returns (bool)
    {
        require(balanceOf[from] >= amount, "balance");
        require(allowance[from][msg.sender] >= amount, "allowance");
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        require(balanceOf[msg.sender] >= amount, "balance");
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        return true;
    }
}

contract TestVault {
    TestToken public token;
    uint256 public claim;
    uint256 public epochs;

    event YieldDistributed(uint256 indexed epoch, uint256 rateE18);
    event Staked(address indexed from, uint256 amount);
    event Withdrawn(address indexed from, uint256 amount);
    event ProofSubmitted(address indexed from, bytes32 indexed digest);

    constructor(TestToken token_, uint256 claimE18) {
        token = token_;
        claim = claimE18;
    }

    function publish(uint256 rateE18) external {
        emit YieldDistributed(epochs++, rateE18);
    }

    function claimApyE18() external view returns (uint256) {
        return claim;
    }

    function stake(uint256 amount) external {
        require(token.transferFrom(msg.sender, address(this), amount), "pull");
        emit Staked(msg.sender, amount);
    }

    function withdraw(uint256 amount) external {
        require(token.transfer(msg.sender, amount), "push");
        emit Withdrawn(msg.sender, amount);
    }

    function submitValidationProof(bytes32 digest) external {
        emit ProofSubmitted(msg.sender, digest);
    }
}
"""


def _artifacts(tmp_path: Path) -> Dict[str, Any]:
    """Compile the two contracts with the Foundry that is already on PATH."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "TestVault.sol").write_text(SOLIDITY, encoding="utf-8")
    (tmp_path / "foundry.toml").write_text(
        '[profile.default]\nsrc = "src"\nout = "out"\nlibs = []\n', encoding="utf-8"
    )
    built = subprocess.run(
        [str(FORGE), "build"], cwd=str(tmp_path), capture_output=True, text=True
    )
    assert built.returncode == 0, built.stderr
    artifacts: Dict[str, Any] = {}
    for name in ("TestToken", "TestVault"):
        artifact = json.loads(
            (tmp_path / "out" / "TestVault.sol" / f"{name}.json").read_text("utf-8")
        )
        artifacts[name] = {
            "abi": artifact["abi"],
            "bin": artifact["bytecode"]["object"],
        }
    return artifacts


@contextlib.contextmanager
def _anvil() -> Any:
    """A local node for the length of one test, on a port nobody else is using."""
    import socket
    import time

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    node = subprocess.Popen(
        [str(ANVIL), "--port", str(port), "--silent"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:  # pragma: no cover - an anvil that will not start
            raise RuntimeError("anvil did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        node.terminate()
        node.wait(timeout=15)


@needs_a_local_chain
def test_a_local_chain_answers_the_same_seam(tmp_path):
    """The first breath: a book, a signed transaction, a receipt, a moved balance."""
    web3 = pytest.importorskip("web3")
    eth_account = pytest.importorskip("eth_account")
    eth_account.Account.enable_unaudited_hdwallet_features()
    account = eth_account.Account.from_mnemonic(ANVIL_MNEMONIC)
    artifacts = _artifacts(tmp_path)

    with _anvil() as rpc:
        node = web3.Web3(web3.Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
        assert node.is_connected()
        owner = account.address

        def deploy(factory: Any, *args: Any) -> Any:
            """Deploy, and bind to the address the receipt reports."""
            digest = factory.constructor(*args).transact({"from": owner})
            receipt = node.eth.wait_for_transaction_receipt(digest, timeout=30)
            assert receipt["status"] == 1
            return node.eth.contract(
                address=receipt["contractAddress"], abi=factory.abi
            )

        token = deploy(
            node.eth.contract(
                abi=artifacts["TestToken"]["abi"],
                bytecode=artifacts["TestToken"]["bin"],
            )
        )
        vault = deploy(
            node.eth.contract(
                abi=artifacts["TestVault"]["abi"],
                bytecode=artifacts["TestVault"]["bin"],
            ),
            token.address,
            int(0.08 * 10 ** 18),
        )
        for call in (
            token.functions.mint(owner, 1_000 * 10 ** 6),
            vault.functions.publish(int(0.0001 * 10 ** 18)),
            vault.functions.publish(int(0.0002 * 10 ** 18)),
            vault.functions.publish(int(0.0003 * 10 ** 18)),
        ):
            node.eth.wait_for_transaction_receipt(
                call.transact({"from": owner}), timeout=30
            )

        provider = RealTestnetProvider(
            env={
                "TESTNET_RPC_URL": rpc,
                "TESTNET_PRIVATE_KEY": "0x" + bytes(account.key).hex(),
                "TESTNET_PROTOCOLS": json.dumps(
                    {
                        "protocols": [
                            {
                                "address": vault.address.lower(),
                                "name": "LocalVault",
                                "bucket": "active",
                                "claim_apy": 0.0,
                                "decimals": 6,
                                "token": token.address.lower(),
                            }
                        ]
                    }
                ),
                "TESTNET_BLOCK_TIMEOUT": "1",
            }
        )

        # 1. the venue publishes what the chain published, and what the chain claims
        published = provider.get_protocol_list()
        assert [item.name for item in published] == ["LocalVault"]
        assert published[0].claim_apy == pytest.approx(0.08)
        assert published[0].record(3) == pytest.approx([0.0001, 0.0002, 0.0003])
        assert provider.describe()["chain_id"] == 31337

        # 2. a dry run answers without spending anything — and it answers the truth:
        #    with no allowance yet, the vault would refuse to pull
        before = provider.dry_run("stake", vault.address, 10.0)
        assert before.ok is False
        assert "allowance" in str(before.revert).lower()
        assert token.functions.balanceOf(vault.address).call() == 0

        # 3. a stake is two transactions (approve, stake), awaited, and it moved money
        assert provider.execute_stake(vault.address, 10.0) is True
        assert token.functions.balanceOf(vault.address).call() == 10_000_000
        assert provider.stakes[-1]["status"] == 1
        assert provider.stakes[-1]["hash"].startswith("0x")

        # 4. and now that the money is there, the same simulation says yes
        after = provider.dry_run("withdraw", vault.address, 1.0)
        assert after.ok is True and after.gas and after.gas > 21_000
        too_much = provider.dry_run("stake", vault.address, 999.0)
        assert too_much.ok is False  # the balance, the allowance, or both

        # 4. and back out again
        assert provider.withdraw(vault.address, 4.0) is True
        assert token.functions.balanceOf(vault.address).call() == 6_000_000

        # 5. the evidence went on chain as a fingerprint
        assert provider.submit_validation_proof(
            vault.address, {"verdict": "HONEST", "step": 3}
        )
        assert provider.proofs[-1]["status"] == 1

        # 6. the clock moved, because our own transactions mined blocks
        assert provider.step > 0

        # 7. and the unchanged Portfolio moved capital through it
        book = P.Portfolio(
            whitelist=provider.whitelist, cash=1_000.0, provider=provider
        )
        assert book.deposit(published[0], 5.0, step=1) == pytest.approx(5.0)
        assert token.functions.balanceOf(vault.address).call() == 11_000_000
        assert book.withdraw(vault.address, 1.0, step=2) == pytest.approx(1.0)
        assert token.functions.balanceOf(vault.address).call() == 10_000_000
        assert book.cash == pytest.approx(996.0)


# ── the wallet, read rather than assumed ────────────────────────────────────


def test_the_balance_is_two_budgets_in_two_decimals(provider, world):
    """Gas money and credits are separate reads, and neither is a small integer."""
    balance = provider.balance(VAULT)

    assert isinstance(balance, T.Balance)
    assert balance.ok and balance.error == ""
    assert balance.native_units == int(0.008 * 10 ** 18)
    assert balance.native_eth == pytest.approx(0.008)
    assert balance.token == TOKEN
    assert balance.token_units == 50_000_000
    assert balance.token_balance == pytest.approx(50.0)
    assert balance.decimals == 6
    assert balance.chain_id == SEPOLIA
    assert "balanceOf" in world.names("call")
    assert balance.as_dict()["native_eth"] == pytest.approx(0.008)


def test_a_balance_without_a_vault_reads_only_the_native_coin(provider):
    """No vault means no token to ask about: one read, and no invented credit."""
    balance = provider.balance()
    assert balance.ok
    assert balance.token == "" and balance.token_balance == 0.0
    assert balance.native_eth == pytest.approx(0.008)


def test_a_read_only_venue_cannot_be_asked_whose_wallet_it_is(read_only):
    """No signer is no address — an error with a name, not the zero address."""
    with pytest.raises(NetNotConfigured):
        read_only.balance()


def test_a_balance_that_cannot_be_read_is_an_error_not_a_zero(provider, world):
    """The number a loop sizes against must be absent when it is unknown, not 0.0."""
    world.reverts["get_balance"] = InsufficientFunds("the node says no")
    balance = provider.balance(VAULT)

    assert balance.ok is False
    assert "native balance" in balance.error
    assert balance.native_eth == 0.0
    assert "unread" in str(balance)


def test_a_vault_outside_the_book_has_no_credit_to_report(provider):
    balance = provider.balance(OTHER)
    assert balance.ok is False
    assert "address book" in balance.error


def test_a_token_read_that_fails_keeps_the_gas_number(provider, world):
    """Half an answer is still an answer: the native read succeeded and is reported."""
    world.reverts["balanceOf"] = ContractLogicError("reverted")
    balance = provider.balance(VAULT)

    assert balance.native_eth == pytest.approx(0.008)
    assert balance.token_balance == 0.0
    assert "token balance" in balance.error


# ── the second source of a record: the feed, and only when it is allowed ────
#
# A real market's addresses publish no ``YieldDistributed``: an aToken, a Uniswap pool and
# a Curve gauge all report yield in their own events. The venue's answer is a *policy*
# (``TESTNET_HISTORY``) with three settings, and what these tests pin down is that the
# policy is obeyed exactly — the log wins in ``auto``, the node is not asked in ``feed``,
# and a feed-priced record is never presented as the address's own.

#: Sixty days at 0.02 % — a plausible daily APY series (about 7.5 % a year).
FEED_APYS = [0.02] * 60


class FakeFeed:
    """A feed that answers from a dict and remembers exactly what it was asked.

    The real collaborator is :class:`agent.physics.defillama_adapter.FeedHistory`; this
    double keeps the same three-argument call and returns the same
    ``FeedRecord``, so the provider's handling of a label, a series and a reason is the
    code under test rather than the double's.
    """

    def __init__(
        self,
        apys: Optional[List[float]] = None,
        *,
        pool: str = "aave-v3 USDC (Arbitrum)",
        uuid: str = "d9fa8e14-0447-4207-9ae8-7810199dfa1f",
        reason: str = "",
    ) -> None:
        self.apys = list(FEED_APYS if apys is None else apys)
        self.pool = pool
        self.uuid = uuid
        self.reason = reason
        self.calls: List[Dict[str, Any]] = []

    def history(
        self, name: str, *, steps: Optional[int] = None, uuid: str = ""
    ) -> Any:
        from agent.physics.defillama_adapter import FeedRecord

        self.calls.append({"name": str(name), "steps": steps, "uuid": str(uuid)})
        apys = self.apys[-int(steps) :] if steps and int(steps) > 0 else self.apys
        return FeedRecord(
            name=str(name),
            uuid=self.uuid if apys else "",
            pool=self.pool if apys else "",
            apy=tuple(apys),
            reason="" if apys else (self.reason or "the feed published no history"),
        )


class DeadNode:
    """A node that raises on every read: what ``TESTNET_HISTORY=feed`` must not need."""

    class eth:  # noqa: N801 - a namespace, not a class anyone instantiates
        @property
        def block_number(self) -> int:
            raise RuntimeError("the node is down")

        @property
        def chain_id(self) -> int:
            raise RuntimeError("the node is down")


def test_the_default_policy_is_the_vault_s_own_log(world):
    """Nothing changes for a vault that publishes: the feed is not asked at all."""
    feed = FakeFeed()
    quiet = RealTestnetProvider(env=venue_env(), web3=FakeWeb3(world), feed=feed)

    assert quiet.config.history == T.DEFAULT_HISTORY == T.HISTORY_CHAIN
    assert quiet.fetch_protocol_history(VAULT) == pytest.approx([0.0004] * 8)
    assert feed.calls == []
    assert quiet.history_source(VAULT) == T.RECORD_FROM_CHAIN


def test_auto_keeps_the_address_s_own_record_when_there_is_one(world):
    """A native record is about *this* address; a name match is not. The log wins."""
    feed = FakeFeed()
    venue = RealTestnetProvider(
        env=venue_env(**{T.ENV_HISTORY: "auto"}), web3=FakeWeb3(world), feed=feed
    )

    assert venue.fetch_protocol_history(VAULT) == pytest.approx([0.0004] * 8)
    assert feed.calls == []


def test_feed_policy_does_not_need_a_node_at_all():
    """The feed is the other wire: a dead RPC must not stop the record being read."""
    feed = FakeFeed()
    venue = RealTestnetProvider(
        env=venue_env(**{T.ENV_HISTORY: "feed"}), web3=DeadNode(), feed=feed
    )

    record = venue.fetch_protocol_history(VAULT)

    assert len(record) == len(FEED_APYS)
    assert venue.errors == []  # the node was not even asked
    assert feed.calls


def test_a_feed_priced_record_is_published_with_its_provenance(world):
    """The protocol exists, is measured, and its note says who vouched for it."""
    world.logs = []
    feed = FakeFeed()
    venue = RealTestnetProvider(
        env=venue_env(**{T.ENV_HISTORY: "auto"}), web3=FakeWeb3(world), feed=feed
    )

    published = venue.get_protocol_list()

    assert len(published) == 1
    protocol = published[0]
    assert protocol.returns.size == len(FEED_APYS)
    assert "a test vault" in protocol.note  # the operator's own note survives
    assert "record from the feed aave-v3 USDC (Arbitrum)" in protocol.note
    assert "this address published no log" in protocol.note
    assert venue.describe()["records"]["VaultA"].startswith("the feed")
    assert venue.describe()["history"] == "auto"


def test_an_entry_can_pin_which_pool_prices_it(world):
    """``pool`` in the book is the operator's declaration, and it reaches the feed."""
    feed = FakeFeed()
    entry = {**book_payload()["protocols"][0], "pool": "pinned-uuid"}
    book = ProtocolBook.from_payload({"protocols": [entry]})
    venue = RealTestnetProvider(
        env=venue_env(**{T.ENV_HISTORY: "feed"}),
        web3=FakeWeb3(world),
        book=book,
        feed=feed,
    )

    venue.fetch_protocol_history(VAULT)

    assert feed.calls[-1]["uuid"] == "pinned-uuid"
    assert feed.calls[-1]["name"] == "VaultA"


def test_a_pool_that_is_not_a_uuid_is_refused_by_name():
    entry = {**book_payload()["protocols"][0], "pool": "two words"}
    with pytest.raises(NetNotConfigured) as raised:
        ProtocolBook.from_payload({"protocols": [entry]})
    assert "not a pool uuid" in str(raised.value)


def test_a_name_the_feed_does_not_know_is_no_record_and_a_reason(world):
    """A feed with nothing to say is an absence with a sentence, not a zero."""
    world.logs = []
    feed = FakeFeed(apys=[], reason="no pool in the catalogue carries this name")
    venue = RealTestnetProvider(
        env=venue_env(**{T.ENV_HISTORY: "auto"}), web3=FakeWeb3(world), feed=feed
    )

    assert venue.fetch_protocol_history(VAULT) == []
    assert venue.get_protocol_list() == []
    assert venue.history_source(VAULT) == ""
    assert any(
        "no record from the feed" in line and "catalogue" in line for line in venue.errors
    )


def test_a_feed_that_raises_is_a_sentence_and_not_a_cycle(world):
    """A bad morning at DeFiLlama must not be able to kill a trading cycle."""

    class Broken:
        def history(self, name: str, **kwargs: Any) -> Any:
            raise RuntimeError("the feed answered HTTP 503")

    venue = RealTestnetProvider(
        env=venue_env(**{T.ENV_HISTORY: "feed"}), web3=FakeWeb3(world), feed=Broken()
    )

    assert venue.fetch_protocol_history(VAULT) == []
    assert any("the feed could not be read" in line for line in venue.errors)


def test_steps_takes_the_tail_of_the_feed_series(world):
    world.logs = []
    feed = FakeFeed()
    venue = RealTestnetProvider(
        env=venue_env(**{T.ENV_HISTORY: "auto"}), web3=FakeWeb3(world), feed=feed
    )

    assert len(venue.fetch_protocol_history(VAULT, steps=5)) == 5
    assert feed.calls[-1]["steps"] == 5


def test_an_unknown_history_policy_is_a_named_error():
    with pytest.raises(NetNotConfigured) as raised:
        RealTestnetProvider(env=venue_env(**{T.ENV_HISTORY: "oracle"}))
    message = str(raised.value)
    assert T.ENV_HISTORY in message and "oracle" in message
    for source in (T.HISTORY_CHAIN, T.HISTORY_FEED, T.HISTORY_AUTO):
        assert source in message


def test_the_policy_is_printed_by_the_venue(provider):
    line = provider.describe()
    assert line["history"] == T.DEFAULT_HISTORY
    assert line["config"]["history"] == T.DEFAULT_HISTORY
    assert T.ENV_HISTORY == "TESTNET_HISTORY"


def test_auto_prices_a_silent_vault_from_the_feed(world):
    """The whole point: no log, and the book still gets a record with real history."""
    world.logs = []
    feed = FakeFeed()
    quiet = RealTestnetProvider(
        env=venue_env(**{T.ENV_HISTORY: "auto"}), web3=FakeWeb3(world), feed=feed
    )

    record = quiet.fetch_protocol_history(VAULT)

    assert len(record) == len(FEED_APYS)
    # the APYs arrive as the per-step rates that compound back to them
    assert record[0] == pytest.approx(feed_rate(0.02))
    assert feed.calls == [{"name": "VaultA", "steps": None, "uuid": ""}]
    assert quiet.history_source(VAULT).startswith("the feed aave-v3 USDC (Arbitrum)")
    assert quiet.history_source(VAULT) != T.RECORD_FROM_CHAIN


# ── Aave v3: somebody else's interface, and the table that vouches for it ────
#
# A vault is trusted because it published its own record; a lending pool cannot be —
# it has no ``YieldDistributed`` to emit. So the trust moves to a different place, and
# these tests pin down where: the deployment table in ``agent.physics.aave_v3``, keyed by
# the chain id the *node* reported. What follows is that a supply is Aave's call on
# Aave's pool, that a pair outside the table is refused before the approval that would
# have paid for it, and that the position is read back off the aToken rather than
# remembered.

#: The pool and the asset, exactly as Arbitrum One spells them.
AAVE_POOL = "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
AAVE_USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
AAVE_ATOKEN = "0x724dc807b04555b71ed48a6896b6f41593b8c637"
STRANGER = "0x" + "77" * 20


def aave_entry(**overrides: Any) -> Dict[str, Any]:
    """One book entry that is an Aave pool rather than a vault of this interface."""
    entry = {
        "address": AAVE_POOL,
        "name": "aave-v3 USDC (Arbitrum)",
        "bucket": "active",
        "claim_apy": 0.05,
        "decimals": 6,
        "token": AAVE_USDC,
        "pool": "d9fa8e14-0447-4207-9ae8-7810199dfa1f",
        "kind": T.AAVE_KIND,
        "note": "the real Aave v3 pool on Arbitrum One",
    }
    entry.update(overrides)
    return {"protocols": [entry]}


def aave_env(**overrides: str) -> Dict[str, str]:
    """Mainnet, with both yeses: the flag below and ``allow_mainnet=True`` at the venue."""
    env = {
        "TESTNET_RPC_URL": "http://127.0.0.1:8545/v2/secret-key-in-the-path",
        "TESTNET_PRIVATE_KEY": KEY,
        "TESTNET_PROTOCOLS": json.dumps(aave_entry()),
        "TESTNET_ALLOW_MAINNET": "1",
        "TESTNET_BLOCK_TIMEOUT": "0.05",
        "TESTNET_RECEIPT_TIMEOUT": "5",
    }
    env.update(overrides)
    return env


@pytest.fixture()
def aave_world(world: World) -> World:
    """A world sitting on Arbitrum One, which is where the table has an answer."""
    world.chain = ARBITRUM
    return world


@pytest.fixture()
def aave_venue(aave_world: World) -> RealTestnetProvider:
    return RealTestnetProvider(
        env=aave_env(), web3=FakeWeb3(aave_world), allow_mainnet=True
    )




def test_an_aave_entry_has_to_name_the_asset_it_supplies():
    """``supply`` takes the asset as its first argument; guessing one supplies the wrong coin."""
    with pytest.raises(NetNotConfigured) as raised:
        ProtocolEntry(
            address=AAVE_POOL, name="aave", kind=T.AAVE_KIND, token="", decimals=6
        )
    assert "token" in str(raised.value)


def test_a_kind_outside_the_book_s_vocabulary_is_refused():
    with pytest.raises(NetNotConfigured) as raised:
        ProtocolEntry(address=VAULT, name="VaultA", kind="curve-v2")
    message = str(raised.value)
    assert "curve-v2" in message and T.AAVE_KIND in message and T.VAULT_KIND in message


def test_the_kind_survives_the_json_the_operator_writes():
    book = ProtocolBook.from_payload(aave_entry())

    assert book.get(AAVE_POOL).kind == T.AAVE_KIND
    assert book.get(AAVE_POOL).as_dict()["kind"] == T.AAVE_KIND
    # an entry that says nothing is a vault, which is what every entry used to be
    plain = ProtocolBook.from_payload(book_payload()).get(VAULT)
    assert plain.kind == T.VAULT_KIND


def test_protocol_kind_answers_for_the_book_and_nothing_else(aave_venue):
    assert aave_venue.protocol_kind(AAVE_POOL) == T.AAVE_KIND
    assert aave_venue.protocol_kind(AAVE_POOL.lower()) == T.AAVE_KIND
    assert aave_venue.protocol_kind(STRANGER) == ""


def test_an_aave_supply_approves_the_pool_and_calls_supply(aave_venue, aave_world):
    assert aave_venue.execute_stake(AAVE_POOL, 1.0) is True

    pool = AAVE_POOL.lower()
    usdc = AAVE_USDC.lower()
    # the spender is the pool, because the pool is what pulls the asset
    assert ("call", "allowance", (SIGNER, pool), {"from": SIGNER}) in aave_world.calls
    assert aave_world.names("build") == ["approve", "supply"]
    approve, supply = [call for call in aave_world.calls if call[0] == "build"]
    assert approve[2] == (pool, 1_000_000)
    assert supply[2] == (usdc, 1_000_000, SIGNER, 0)  # no referral, the signer's position

    assert aave_world.sent_to() == [usdc, pool]
    assert len(aave_world.waits) == 2
    assert {tx.get("chainId") for tx in (approve[3], supply[3])} == {ARBITRUM}

    assert len(aave_venue.stakes) == 1
    assert aave_venue.stakes[0]["action"] == "supply"
    assert aave_venue.stakes[0]["units"] == 1_000_000
    assert aave_venue.stakes[0]["status"] == 1
    assert aave_venue.refusals == []


def test_an_aave_pair_the_table_does_not_know_is_refused_before_any_approval(world):
    """The whole safety story: a pool this file cannot name is a pool it cannot sign into."""
    world.chain = ARBITRUM
    venue = RealTestnetProvider(
        env=aave_env(**{"TESTNET_PROTOCOLS": json.dumps(aave_entry(address=STRANGER))}),
        web3=FakeWeb3(world),
        allow_mainnet=True,
    )

    with pytest.raises(NetNotConfigured) as raised:
        venue.execute_stake(STRANGER, 1.0)

    message = str(raised.value)
    assert STRANGER in message and str(ARBITRUM) in message
    assert world.names("build") == [] and venue.approvals == [] and venue.stakes == []


def test_a_known_pool_over_an_unknown_asset_is_refused_too(world):
    """The pair is checked, not the pool alone: the wrong asset is a supply of the wrong coin."""
    world.chain = ARBITRUM
    venue = RealTestnetProvider(
        env=aave_env(**{"TESTNET_PROTOCOLS": json.dumps(aave_entry(token=TOKEN))}),
        web3=FakeWeb3(world),
        allow_mainnet=True,
    )

    with pytest.raises(NetNotConfigured) as raised:
        venue.execute_stake(AAVE_POOL, 1.0)
    assert TOKEN in str(raised.value) and world.names("build") == []


def test_an_aave_withdraw_returns_the_asset_to_the_signer(aave_venue, aave_world):
    assert aave_venue.withdraw(AAVE_POOL, 2.5) is True

    assert aave_world.names("build") == ["withdraw"]
    withdraw = [call for call in aave_world.calls if call[0] == "build"][0]
    assert withdraw[2] == (AAVE_USDC.lower(), 2_500_000, SIGNER)
    assert aave_world.sent_to() == [AAVE_POOL.lower()]
    assert aave_venue.withdrawals[0]["action"] == "withdraw"
    assert aave_venue.approvals == []  # nothing to approve on the way out


def test_the_balance_reports_the_a_token_position(aave_venue, aave_world):
    """A position is a receipt in the wallet, so the wallet can be asked about it."""
    aave_world.token_balance = int(1.5 * 10 ** 6)

    read = aave_venue.balance(AAVE_POOL)

    assert read.ok and read.staked
    assert read.token == AAVE_USDC.lower() and read.token_balance == pytest.approx(1.5)
    assert read.position_token == AAVE_ATOKEN.lower()
    assert read.position_balance == pytest.approx(1.5)
    assert "in the venue" in str(read)
    assert read.as_dict()["position_units"] == 1_500_000


def test_a_vault_reports_no_a_token_position(provider):
    """A vault keeps its own accounting; inventing a receipt for it would be a guess."""
    read = provider.balance(VAULT)
    assert read.position_token == "" and read.position_units == 0 and not read.staked


def test_a_dry_run_of_a_supply_simulates_the_pool(aave_venue, aave_world):
    run = aave_venue.dry_run("stake", AAVE_POOL, 1.0)

    assert run.ok and run.units == 1_000_000 and run.sender == SIGNER
    assert aave_world.names("call") == ["supply"]
    assert "stake" not in aave_world.names("estimate_gas")
    assert aave_world.sent == []  # a simulation signs nothing


def test_a_proof_has_no_aave_equivalent(aave_venue):
    run = aave_venue.dry_run("proof", AAVE_POOL, 0.0, {"verdict": "CALM"})

    assert run.ok is False
    assert "evidence call" in (run.revert or "")


def test_the_chain_policy_tells_the_operator_to_ask_the_feed(world):
    """An Aave position emits nothing, so ``=chain`` cannot price it — and the log says so."""
    world.chain = ARBITRUM
    venue = RealTestnetProvider(
        env=aave_env(**{T.ENV_HISTORY: T.HISTORY_CHAIN}),
        web3=FakeWeb3(world),
        allow_mainnet=True,
    )

    assert venue.get_protocol_list() == []
    assert any(T.ENV_HISTORY in line and "Aave" in line for line in venue.errors)


def test_a_book_that_counts_decimals_wrong_is_refused(world):
    """``decimals`` is what turns $1 into units: 18 on a 6-decimal asset is a ×10¹² approval."""
    world.chain = ARBITRUM
    venue = RealTestnetProvider(
        env=aave_env(**{"TESTNET_PROTOCOLS": json.dumps(aave_entry(decimals=18))}),
        web3=FakeWeb3(world),
        allow_mainnet=True,
    )

    with pytest.raises(NetNotConfigured) as raised:
        venue.execute_stake(AAVE_POOL, 1.0)
    message = str(raised.value)
    assert "18" in message and "6" in message and world.names("build") == []


def test_a_feed_priced_aave_entry_is_published_with_the_pool_in_its_note(world):
    world.chain = ARBITRUM
    world.logs = []
    venue = RealTestnetProvider(
        env=aave_env(**{T.ENV_HISTORY: T.HISTORY_AUTO}),
        web3=FakeWeb3(world),
        feed=FakeFeed(),
        allow_mainnet=True,
    )

    published = venue.get_protocol_list()

    assert len(published) == 1
    protocol = published[0]
    assert protocol.address == AAVE_POOL.lower()
    assert protocol.claim_apy == pytest.approx(0.05)  # the book's claim, not a chain read
    assert "claimApyE18" not in world.names("call")
    assert "supply(" in protocol.note and "Aave v3 USDC (Arbitrum)" in protocol.note
    assert venue.protocol_kind(AAVE_POOL) == T.AAVE_KIND
    assert venue.history_source(AAVE_POOL) != T.RECORD_FROM_CHAIN

