"""Aave v3 as a *table of deployments*, not as a paragraph of configuration.

The venue in ``real_testnet_provider`` speaks one interface it wrote itself
(``stake(uint256)`` / ``withdraw(uint256)`` / ``YieldDistributed``). A lending pool is
not that interface: money goes in with ``supply(address,uint256,address,uint16)``, comes
out with ``withdraw(address,uint256,address)``, and the receipt is an aToken whose
balance grows. Nothing here tries to make one look like the other. What this module does
is the two things an operator cannot be trusted to retype and a book must not be able to
invent:

**where the pool is.** A pool address is a fact about a chain id. It is written here,
once, with the asset it lends, the decimals of that asset and the aToken that proves a
position — and :func:`deployment_for` answers *only* for triples that are in this table.
A book entry marked ``kind: aave-v3`` is therefore not "a contract the operator would
like to call": it is the Aave pool of this chain, or it is refused before a single byte
is signed.

**what the call is.** :func:`supply_args` and :func:`withdraw_args` build the argument
tuple in the order the ABI demands, and :func:`pool_abi` is the ABI that encodes it —
``referralCode`` is a ``uint16``, and a pool told otherwise reverts on a chain where the
operator pays for the privilege. The ``onBehalfOf`` and the recipient are always the
signer: a position opened for somebody else is one this loop cannot withdraw, and a
withdrawal sent to somebody else is a gift.

**What this module deliberately does not have**: an APY. Aave's rate is a market fact
published by Aave (and by DeFiLlama, which is where the trading loop reads it), not a
number this file could compute from a balance. A deployment here is an *address book*,
and a rate that appeared in it would be a measurement with no source.

Adding a chain is a code change with a test — not an environment variable. That is the
point: the only way to widen what this file will sign into is to write down which
contract it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "AAVE_KIND",
    "DEPLOYMENTS",
    "MAX_UINT256",
    "REFERRAL_NONE",
    "SUPPLY_SIGNATURE",
    "WITHDRAW_SIGNATURE",
    "AaveDeployment",
    "deployment_for",
    "describe",
    "pool_abi",
    "supply_args",
    "withdraw_args",
]

#: What the trading loop calls a book entry of this kind, and what the venue answers in
#: ``protocol_kind()``. A string, because it travels through a JSON file and a report.
AAVE_KIND = "aave-v3"

#: The signatures exactly as the Pool declares them — printed in reports and used to
#: label a transaction, never to encode one (the ABI below does the encoding).
SUPPLY_SIGNATURE = "supply(address,uint256,address,uint16)"
WITHDRAW_SIGNATURE = "withdraw(address,uint256,address)"

#: Aave's "everything" for a withdrawal. Not used by the loop — it withdraws the amount
#: it recorded — but a value worth naming, because ``type(uint256).max`` sent as a
#: *supply* is the largest number this file could ever let anyone approve.
def _is_address(value: object) -> bool:
    """``0x`` and forty hex characters — the same test the venue applies to a vault."""
    text = str(value or "")
    if len(text) != 42 or not text.startswith("0x"):
        return False
    return all(character in "0123456789abcdefABCDEF" for character in text[2:])


@dataclass(frozen=True)
class AaveDeployment:
    """One pool, on one chain, for one asset — and the aToken that proves a position.

    Every field is read from the protocol's own documentation and written here by hand;
    :meth:`__post_init__` enforces only what a typo would break (a 20-byte address, a
    plausible decimals count), and the venue adds the check that matters: the chain id the
    node reports has to be the key this deployment was filed under.
    """

    chain_id: int
    chain: str
    pool: str
    asset: str
    symbol: str
    decimals: int
    a_token: str = ""

    def __post_init__(self) -> None:
        for name in ("pool", "asset", "a_token"):
            value = str(getattr(self, name) or "").strip().lower()
            if value and not _is_address(value):
                raise ValueError(
                    f"{type(self).__name__}: {name} {value!r} is not an address"
                )
            object.__setattr__(self, name, value)
        if not _is_address(self.pool):
            raise ValueError(f"{type(self).__name__}: a pool is required")
        if not _is_address(self.asset):
            raise ValueError(f"{type(self).__name__}: an asset is required")
        if not 0 <= int(self.decimals) <= 18:
            raise ValueError(
                f"{type(self).__name__}: {self.decimals!r} decimals is not a token"
            )
        object.__setattr__(self, "chain_id", int(self.chain_id))
        object.__setattr__(self, "decimals", int(self.decimals))
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "chain", str(self.chain).strip())

    @property
    def label(self) -> str:
        """``Aave v3 USDC (Arbitrum)`` — the pool's own name, which is also a feed row."""
        return f"Aave v3 {self.symbol} ({self.chain})"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": int(self.chain_id),
            "chain": self.chain,
            "pool": self.pool,
            "asset": self.asset,
            "symbol": self.symbol,
            "decimals": int(self.decimals),
            "a_token": self.a_token,
            "label": self.label,
            "supply": SUPPLY_SIGNATURE,
            "withdraw": WITHDRAW_SIGNATURE,
        }

    def __str__(self) -> str:
        a_token = f", aToken {self.a_token}" if self.a_token else ""
        return (
            f"{self.label}: pool {self.pool} over {self.asset} "
            f"({self.decimals} decimals{a_token})"
        )

MAX_UINT256 = (1 << 256) - 1

#: No referral. Aave pays a cut to whoever is named here, and this loop is nobody's
#: affiliate: the operator's own address earns the whole rate.
REFERRAL_NONE = 0

#: Every deployment this file will sign into, by chain id. Written out rather than read
#: from the environment: an endpoint is a secret and belongs in ``.env``, but *which
#: contract* is a fact, and a fact a stray variable could change is not one.
#:
#: Arbitrum One is the chain the operator runs. The pool is the one above aArbUSDCn
#: (``0x724dc807…8c637``) — the address the book used to carry. That aToken is a *claim
#: on* the pool, not a venue a wallet can ``supply`` into, and moving the entry from the
#: receipt to the pool is precisely what turns a measurement into a position.
DEPLOYMENTS: Dict[int, AaveDeployment] = {
    42161: AaveDeployment(
        chain_id=42161,
        chain="Arbitrum",
        pool="0x794a61358D6845594F94dc1DB02A252b5b4814aD",
        asset="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        symbol="USDC",
        decimals=6,
        a_token="0x724dc807b04555b71ed48a6896b6f41593b8c637",
    ),
    # The rehearsal space: Aave v3 on Arbitrum Sepolia lends a *test* USDC, which is why
    # the pair is here at all — a venue that only knew mainnet could not be rehearsed, and
    # one that guessed a testnet address would be signing into nothing.
    421614: AaveDeployment(
        chain_id=421614,
        chain="Arbitrum Sepolia",
        pool="0x6Ae43d3271ff6888e7Fc43Fd7321a503ff738951",
        asset="0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d",
        symbol="USDC",
        decimals=6,
    ),
}


def deployment_for(
    chain_id: object, pool: object, asset: object = ""
) -> Optional[AaveDeployment]:
    """The deployment a ``(chain, pool, asset)`` triple *is*, or ``None``.

    A triple that is not in the table is not "probably fine": it is an address this file
    has no reason to believe is Aave, and the venue turns ``None`` into a refusal that
    names the entry. ``asset`` is optional — a book may leave the asset to the table — but
    when it is given it has to match, and that is the case that matters: the wrong asset is
    a supply of the wrong token, which Aave rejects *after* the approval has been paid for.
    """
    try:
        key = int(chain_id or 0)
    except (TypeError, ValueError):
        return None
    deployment = DEPLOYMENTS.get(key)
    if deployment is None:
        return None
    if deployment.pool != str(pool or "").strip().lower():
        return None
    wanted = str(asset or "").strip().lower()
    if wanted and deployment.asset != wanted:
        return None
    return deployment


def pool_abi() -> List[Dict[str, Any]]:
    """The two calls this file makes, with Aave's exact types and no extras.

    ``referralCode`` is a ``uint16``: web3 encodes what the ABI says, so a Python ``int``
    that fits is fine here, while a pool told the code was a ``uint256`` would revert.
    ``outputs`` are declared only where the return value is used — ``withdraw`` returns the
    amount burned, which this loop does not spend.
    """
    return [
        {
            "name": "supply",
            "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [
                {"name": "asset", "type": "address"},
                {"name": "amount", "type": "uint256"},
                {"name": "onBehalfOf", "type": "address"},
                {"name": "referralCode", "type": "uint16"},
            ],
            "outputs": [],
        },
        {
            "name": "withdraw",
            "type": "function",
            "stateMutability": "nonpayable",
            "inputs": [
                {"name": "asset", "type": "address"},
                {"name": "amount", "type": "uint256"},
                {"name": "to", "type": "address"},
            ],
            "outputs": [{"name": "", "type": "uint256"}],
        },
    ]


def supply_args(
    deployment: AaveDeployment, units: int, on_behalf_of: str
) -> Tuple[str, int, str, int]:
    """``supply(asset, amount, onBehalfOf, referralCode)`` — the position is the signer's."""
    amount = int(units)
    if amount <= 0:
        raise ValueError(
            f"supply: {units!r} is not a positive amount of {deployment.symbol}"
        )
    if amount >= MAX_UINT256:
        raise ValueError("supply: that amount is Aave's 'everything' constant, not a supply")
    owner = str(on_behalf_of or "").strip()
    if not _is_address(owner):
        raise ValueError(f"supply: {owner!r} is not an address a position can belong to")
    return (deployment.asset, amount, owner, REFERRAL_NONE)


def withdraw_args(
    deployment: AaveDeployment, units: int, recipient: str
) -> Tuple[str, int, str]:
    """``withdraw(asset, amount, to)`` — and ``to`` is the signer, never a third party."""
    amount = int(units)
    if amount <= 0:
        raise ValueError(
            f"withdraw: {units!r} is not a positive amount of {deployment.symbol}"
        )
    target = str(recipient or "").strip()
    if not _is_address(target):
        raise ValueError(f"withdraw: {target!r} is not an address that can be paid")
    return (deployment.asset, amount, target)


def describe(chain_id: object = None) -> List[str]:
    """The table as sentences — for ``--book-report`` and for a reader of the log."""
    wanted = None if chain_id is None else int(chain_id or 0)
    rows: List[str] = []
    for key in sorted(DEPLOYMENTS):
        if wanted is not None and key != wanted:
            continue
        rows.append(str(DEPLOYMENTS[key]))
    return rows

