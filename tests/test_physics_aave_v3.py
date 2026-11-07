"""The Aave table: which contract, which call, and the refusals that keep both honest.

A lending pool is the first venue this project did not write, so the tests here are the
same shape as the provider's — *what exactly is signed* — with one addition the vault
never needed: the address itself is now the thing that has to be right. A vault's address
is checked by its own log; a pool's address is checked by being **in this table**, and a
table is only worth something if it refuses.

Three claims, and every test below is one of them:

* **the deployment is a fact.** ``0x794a61358D…814aD`` over ``0xaf88d0…5831`` on chain
  ``42161`` is the pair the trading loop was pointed at, and the aToken written beside it
  is the receipt a position grows into. Pin them here, so a careless edit is a red test
  rather than a transaction;
* **the call is Aave's, not ours.** ``supply(asset, amount, onBehalfOf, referralCode)``
  with ``referralCode`` as a ``uint16`` and the signer as its own beneficiary; a
  ``withdraw`` that pays the signer. No ``stake``, no third party, no affiliate;
* **nothing is inferred.** A triple that is not in the table is ``None``, and the venue
  turns that into a refusal. The module is a table plus four functions: it holds no key,
  opens no socket, and does not import the file that signs.

The last test reads this module's own source: no ``web3``, no URL, no 64-hex literal. It
is the same sterility check the provider carries, applied to the file that says where the
money may go.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict

import pytest

from agent.physics import aave_v3 as A

MODULE = Path(__file__).resolve().parents[1] / "agent" / "physics" / "aave_v3.py"

#: Arbitrum One, exactly as the operator's book and the feed's catalogue spell it.
POOL = "0x794a61358D6845594F94dc1DB02A252b5b4814aD"
USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
A_USDC = "0x724dc807b04555b71ed48a6896b6f41593b8c637"

ARBITRUM = 42161
ARBITRUM_SEPOLIA = 421614
SIGNER = "0x" + "33" * 20
OTHER = "0x" + "44" * 20


def deployment() -> A.AaveDeployment:
    return A.DEPLOYMENTS[ARBITRUM]


# ── the addresses ───────────────────────────────────────────────────────────


def test_the_arbitrum_deployment_is_the_pair_the_loop_was_pointed_at():
    """Pinned, because a pool address is the one field a typo cannot be forgiven."""
    found = deployment()

    assert found.pool == POOL.lower()
    assert found.asset == USDC.lower()
    assert found.a_token == A_USDC.lower()
    assert found.symbol == "USDC" and found.decimals == 6
    assert found.chain_id == ARBITRUM
    assert found.label == "Aave v3 USDC (Arbitrum)"


def test_every_deployment_is_filed_under_its_own_chain_id():
    for chain_id, row in A.DEPLOYMENTS.items():
        assert row.chain_id == chain_id
        assert re.fullmatch(r"0x[0-9a-f]{40}", row.pool)
        assert re.fullmatch(r"0x[0-9a-f]{40}", row.asset)
        assert row.a_token == "" or re.fullmatch(r"0x[0-9a-f]{40}", row.a_token)
        assert 0 <= row.decimals <= 18
        assert row.symbol == row.symbol.upper() and row.symbol
        assert row.chain.strip() == row.chain and row.chain


def test_the_rehearsal_chain_is_a_pool_of_its_own():
    """Sepolia is not "the same thing on another id": a test pool is another contract."""
    rehearsal = A.DEPLOYMENTS[ARBITRUM_SEPOLIA]

    assert rehearsal.pool != deployment().pool
    assert rehearsal.asset != deployment().asset
    assert rehearsal.decimals == deployment().decimals


def test_a_triple_that_is_not_in_the_table_is_not_a_deployment():
    assert A.deployment_for(ARBITRUM, "0x" + "77" * 20) is None
    assert A.deployment_for(ARBITRUM, POOL, "0x" + "77" * 20) is None
    assert A.deployment_for(1, POOL, USDC) is None
    assert A.deployment_for(None, POOL, USDC) is None
    assert A.deployment_for("not a chain", POOL, USDC) is None


def test_the_asset_is_optional_but_never_wrong():
    """A book may leave the asset to the table; it may not name a different one."""
    assert A.deployment_for(ARBITRUM, POOL) is deployment()
    assert A.deployment_for(ARBITRUM, POOL, "") is deployment()
    assert A.deployment_for(ARBITRUM, POOL.lower(), USDC.upper().lower()) is deployment()
    assert A.deployment_for(ARBITRUM, POOL, "0x" + "00" * 20) is None


def test_the_chain_id_may_arrive_as_the_string_an_environment_holds():
    """``TESTNET_CHAIN_ID`` is text; the table is keyed by numbers, and both must work."""
    assert A.deployment_for("42161", POOL, USDC) is deployment()
    assert A.deployment_for(float(ARBITRUM), POOL, USDC) is deployment()


# ── the calls ───────────────────────────────────────────────────────────────


def test_supply_args_are_in_aaves_order_and_pay_nobody():
    """``(asset, amount, onBehalfOf, referralCode)`` — the third is the signer, the last 0."""
    args = A.supply_args(deployment(), 1_000_000, SIGNER)

    assert args == (USDC.lower(), 1_000_000, SIGNER, 0)
    assert args[3] == A.REFERRAL_NONE
    assert isinstance(args[1], int)
    assert args[2] == SIGNER  # the position belongs to the account that can withdraw it


def test_supply_refuses_a_zero_a_max_and_a_bad_owner():
    with pytest.raises(ValueError):
        A.supply_args(deployment(), 0, SIGNER)
    with pytest.raises(ValueError):
        A.supply_args(deployment(), -1, SIGNER)
    with pytest.raises(ValueError):  # Aave's "everything" is a withdrawal, never a supply
        A.supply_args(deployment(), A.MAX_UINT256, SIGNER)
    with pytest.raises(ValueError):
        A.supply_args(deployment(), 1_000_000, "0xnotanaddress")
    with pytest.raises(ValueError):
        A.supply_args(deployment(), 1_000_000, "")


def test_withdraw_args_pay_the_signer_and_refuse_a_stranger():
    assert A.withdraw_args(deployment(), 2_000_000, SIGNER) == (
        USDC.lower(),
        2_000_000,
        SIGNER,
    )
    with pytest.raises(ValueError):
        A.withdraw_args(deployment(), 0, SIGNER)
    with pytest.raises(ValueError):
        A.withdraw_args(deployment(), 1_000_000, OTHER[:-1])


def test_the_abi_declares_referral_as_uint16_and_both_calls_as_nonpayable():
    """The type list is the encoding: a ``uint256`` referral is a different selector."""
    abi = A.pool_abi()
    by_name: Dict[str, Dict[str, Any]] = {entry["name"]: entry for entry in abi}

    assert set(by_name) == {"supply", "withdraw"}
    supply = by_name["supply"]
    assert [item["type"] for item in supply["inputs"]] == [
        "address",
        "uint256",
        "address",
        "uint16",
    ]
    assert [item["name"] for item in supply["inputs"]] == [
        "asset",
        "amount",
        "onBehalfOf",
        "referralCode",
    ]
    assert supply["stateMutability"] == "nonpayable" and supply["outputs"] == []
    withdraw = by_name["withdraw"]
    assert [item["type"] for item in withdraw["inputs"]] == ["address", "uint256", "address"]
    assert [item["name"] for item in withdraw["inputs"]] == ["asset", "amount", "to"]
    assert withdraw["stateMutability"] == "nonpayable"


def test_the_signatures_agree_with_the_abi():
    """A signature is what a report prints; it must name the same types the ABI encodes."""
    abi = {entry["name"]: entry for entry in A.pool_abi()}
    for name, signature in (
        ("supply", A.SUPPLY_SIGNATURE),
        ("withdraw", A.WITHDRAW_SIGNATURE),
    ):
        types = ",".join(item["type"] for item in abi[name]["inputs"])
        assert signature == f"{name}({types})"


def test_describe_says_which_pool_for_which_chain():
    every = A.describe()
    assert len(every) == len(A.DEPLOYMENTS)
    one = A.describe(ARBITRUM)
    assert len(one) == 1 and POOL.lower() in one[0] and "Arbitrum" in one[0]
    assert A.describe(1) == []


def test_the_label_is_the_pools_own_name():
    """It is also a DeFiLlama row: the feed is matched on this spelling."""
    assert deployment().label == "Aave v3 USDC (Arbitrum)"
    assert A.DEPLOYMENTS[ARBITRUM_SEPOLIA].label == "Aave v3 USDC (Arbitrum Sepolia)"


def test_a_deployment_with_a_broken_field_is_refused_at_construction():
    with pytest.raises(ValueError):
        A.AaveDeployment(
            chain_id=1, chain="Nowhere", pool="0x123", asset=USDC, symbol="USDC", decimals=6
        )
    with pytest.raises(ValueError):
        A.AaveDeployment(
            chain_id=1, chain="Nowhere", pool=POOL, asset=USDC, symbol="USDC", decimals=19
        )
    with pytest.raises(ValueError):
        A.AaveDeployment(
            chain_id=1, chain="Nowhere", pool="", asset=USDC, symbol="USDC", decimals=6
        )


def test_the_row_serialises_without_an_opinion_about_the_rate():
    row = deployment().as_dict()

    assert row["pool"] == POOL.lower() and row["asset"] == USDC.lower()
    assert row["a_token"] == A_USDC.lower() and row["decimals"] == 6
    assert row["label"] == "Aave v3 USDC (Arbitrum)"
    assert row["supply"] == A.SUPPLY_SIGNATURE and row["withdraw"] == A.WITHDRAW_SIGNATURE
    assert "apy" not in " ".join(row)  # a rate here would be a measurement with no source




# ── the file itself ─────────────────────────────────────────────────────────


def test_the_module_is_a_table_and_not_a_client():
    """No key, no socket, no signer: the file that says *where* must not be able to sign."""
    source = MODULE.read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__", "dataclasses", "typing"}
    assert "http://" not in source and "https://" not in source
    assert not re.search(r"\b0x[0-9a-fA-F]{64}\b", source)
    assert "private_key" not in source


def test_the_kind_is_the_string_the_loop_and_the_book_share():
    assert A.AAVE_KIND == "aave-v3"
    assert A.MAX_UINT256 == 2 ** 256 - 1
    assert A.REFERRAL_NONE == 0
