"""The first breath: one real transfer, signed on this machine, awaited to a receipt.

Why this file exists
====================

``agent/physics/real_testnet_provider.py`` proved that the five operations of a venue
survive contact with a chain. This script proves something smaller and prior to that:
that *this* machine, with *this* ``.env``, can reach a node, sign with the configured
key, and get a transaction mined. One transfer, no loop, no scheduler, no retry — an
organism that cannot take a single breath should not be asked to run a marathon.

It is deliberately outside the package: nothing in ``agent/`` imports it, so a node
that is down can never make an import fail or a test flaky.

The amount is 0.001 **test** ether by default. A mainnet chain id is refused unless the
same two yeses the venue demands are present (``TESTNET_ALLOW_MAINNET=1`` *and*
``--allow-mainnet``), because the first transaction anyone sends by accident should not
be the expensive kind.

Usage
=====

    python scripts/first_breath.py                      # .env, the node's own accounts
    python scripts/first_breath.py --to 0x... --amount 0.5
    python scripts/first_breath.py --dry-run            # sign, print, send nothing

Configuration is not read here: it is read by :meth:`TestnetConfig.from_env`, the one
place that already knows the names, the ``.env`` fallback and the masked ``repr``. The
key is printed only as ``0x...last4``, and the endpoint only as ``scheme://host:port``,
because an RPC path is often an API key.

Exit codes
==========

======  ======================================================================
0       mined, and the receipt says status 1
2       configuration: no web3/eth-account, no node, no signer, bad address/amount
3       the node did not answer: anvil is not running, or the port is wrong
4       the node refused it, it reverted, or no receipt arrived in time
5       a mainnet chain id: this script moves test ether only
======  ======================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_UNREACHABLE = 3
EXIT_REFUSED = 4
EXIT_MAINNET = 5

#: Test ether. Small enough that a wrong recipient is embarrassing, not expensive.
DEFAULT_AMOUNT = "0.001"

#: A plain transfer is 21000 gas. Used only if the node will not estimate.
FALLBACK_GAS = 21_000

#: Seconds to wait for any single RPC answer. The receipt has its own, longer clock.
CONNECT_TIMEOUT = 10.0

#: How many steps the script narrates. Fixed, so "[3/5]" is always true.
STEPS = 5

EXAMPLES = """\
examples:
  python scripts/first_breath.py
  python scripts/first_breath.py --to 0x70997970C51812dc3A010C7d01b50e0d17dc79C8
  python scripts/first_breath.py --dry-run --no-color
"""


class Voice:
    """What the script says while it works: a step first, the values under it."""

    def __init__(self, color: bool) -> None:
        self.color = color

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def head(self, title: str) -> None:
        print(self._paint("-" * 68, "2"))
        print(self._paint(title, "1"))

    def foot(self, title: str) -> None:
        print(self._paint(title, "1"))
        print(self._paint("-" * 68, "2"))

    def step(self, index: int, text: str) -> None:
        print(f"{self._paint(f'[{index}/{STEPS}]', '36')} {text}")

    def row(self, label: str, value: str) -> None:
        print(f"       {self._paint(f'{label:<10}', '2')} {value}")

    def bad(self, message: str) -> None:
        print(self._paint("error: ", "31") + message, file=sys.stderr)


def _wants_color(disabled: bool) -> bool:
    """Colour only for a terminal that asked for it: logs stay greppable."""
    return not disabled and sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _endpoint(url: str) -> str:
    """``scheme://host:port`` — enough to debug anvil, never a path that holds a key."""
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return f"{parts.scheme or 'http'}://{parts.hostname or '(unnamed)'}:{port}"


def _short(exc: BaseException, limit: int = 180) -> str:
    """One line of an exception: nodes explain themselves in paragraphs."""
    text = " ".join(str(exc).split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _ether(wei: int, places: int = 6) -> str:
    return f"{Decimal(wei) / Decimal(10**18):.{places}f}"


def _fail(voice: Voice, code: int, message: str) -> int:
    """Every unhappy path is a sentence and a code, never a traceback."""
    voice.bad(message)
    return code


def _breath(args: argparse.Namespace, voice: Voice) -> int:
    voice.head("ENRG-AI - first breath: one transfer, awaited to a receipt")

    # 1. the library, then the configuration that already exists in the package
    voice.step(1, "loading web3 and the environment")
    try:
        from eth_account import Account
        from requests.exceptions import RequestException
        from web3 import Web3
        from web3.exceptions import TimeExhausted, Web3Exception
    except ImportError as exc:
        return _fail(
            voice,
            EXIT_CONFIG,
            f"web3 / eth-account are not installed ({_short(exc)}); "
            "install the optional extra: pip install -e .[testnet]",
        )

    from agent.physics.real_testnet_provider import (
        ENV_ALLOW_MAINNET,
        ENV_KEY,
        ENV_RPC,
        ENV_UNLOCKED,
        MAINNET_CHAIN_IDS,
        TestnetConfig,
        TestnetNotConfigured,
    )

    try:
        # The one reader of .env, the names and the masked repr. Not reinvented here.
        config = TestnetConfig.from_env(allow_mainnet=args.allow_mainnet)
    except TestnetNotConfigured as exc:
        return _fail(
            voice,
            EXIT_CONFIG,
            f"{exc}. A local anvil answers on {ENV_RPC}=http://127.0.0.1:8545",
        )

    if not config.can_sign:
        return _fail(voice, EXIT_CONFIG, f"no signer: set {ENV_KEY} or {ENV_UNLOCKED}")

    voice.row("node", _endpoint(config.rpc_url))
    voice.row(
        "signer",
        f"key {config.masked_key}" if config.private_key else f"unlocked {config.unlocked_account}",
    )

    # 2. is there a chain on the other end, and which one
    voice.step(2, "connecting")
    w3 = Web3(Web3.HTTPProvider(config.rpc_url, request_kwargs={"timeout": CONNECT_TIMEOUT}))
    network_errors = (RequestException, Web3Exception, OSError)
    try:
        client = str(w3.client_version)
        chain_id = int(w3.eth.chain_id)
        height = int(w3.eth.block_number)
    except network_errors as exc:
        return _fail(
            voice,
            EXIT_UNREACHABLE,
            f"{_endpoint(config.rpc_url)} did not answer "
            f"({type(exc).__name__}: {_short(exc)}). Is anvil up?  anvil --port 8545",
        )

    if config.chain_id is not None and int(config.chain_id) != chain_id:
        return _fail(
            voice,
            EXIT_CONFIG,
            f"the node is chain {chain_id}, but the configuration says {config.chain_id}",
        )
    if chain_id in MAINNET_CHAIN_IDS and not config.allow_mainnet:
        return _fail(
            voice,
            EXIT_MAINNET,
            f"chain {chain_id} is a mainnet and this script moves test ether; "
            f"it would need both {ENV_ALLOW_MAINNET}=1 and --allow-mainnet",
        )
    voice.row("chain", f"{chain_id} - {client} - height {height}")

    # 3. who signs, who receives, how much
    voice.step(3, "reading the signer, the recipient and the amount")
    if config.private_key:
        sender = Account.from_key(config.private_key).address
    else:
        sender = Web3.to_checksum_address(config.unlocked_account)

    if args.to:
        if not Web3.is_address(args.to):
            return _fail(voice, EXIT_CONFIG, f"--to {args.to!r} is not a 20-byte address")
        recipient = Web3.to_checksum_address(args.to)
    else:
        try:
            exposed = [Web3.to_checksum_address(a) for a in w3.eth.accounts]
        except network_errors as exc:
            return _fail(
                voice,
                EXIT_UNREACHABLE,
                f"the node would not list its accounts ({_short(exc)})",
            )
        others = [a for a in exposed if a.lower() != sender.lower()]
        if not others:
            return _fail(
                voice,
                EXIT_CONFIG,
                "no recipient: pass --to (the node exposed no second account to send to)",
            )
        recipient = others[0]

    try:
        amount = Decimal(str(args.amount))
    except InvalidOperation:
        return _fail(voice, EXIT_CONFIG, f"--amount {args.amount!r} is not a number of ether")
    if amount <= 0:
        return _fail(
            voice, EXIT_CONFIG, "--amount must be positive: a zero transfer proves nothing"
        )
    value = int(Web3.to_wei(amount, "ether"))

    try:
        sender_before = int(w3.eth.get_balance(sender))
        recipient_before = int(w3.eth.get_balance(recipient))
        gas_price = int(w3.eth.gas_price)
        nonce = int(w3.eth.get_transaction_count(sender))
    except network_errors as exc:
        return _fail(
            voice,
            EXIT_UNREACHABLE,
            f"the node stopped answering while reading state ({_short(exc)})",
        )

    if sender_before <= value:
        return _fail(
            voice,
            EXIT_REFUSED,
            f"{sender} holds {_ether(sender_before)} ETH, which cannot cover "
            f"{_ether(value)} ETH plus gas",
        )

    # 4. the transaction, and the signature that never leaves this process
    voice.step(4, "preparing the transfer and signing it locally")
    tx: dict = {
        "from": sender,
        "to": recipient,
        "value": value,
        "nonce": nonce,
        "chainId": chain_id,
        "gasPrice": gas_price,
    }
    try:
        tx["gas"] = int(
            w3.eth.estimate_gas({key: val for key, val in tx.items() if key != "chainId"})
        )
    except (Web3Exception, ValueError):
        tx["gas"] = FALLBACK_GAS  # a plain transfer is 21000; the node may know better

    raw: Optional[bytes] = None
    digest = "(the node will sign)"
    if config.private_key:
        try:
            signed = Account.sign_transaction(tx, config.private_key)
        except (ValueError, TypeError) as exc:
            return _fail(
                voice, EXIT_REFUSED, f"the key cannot sign this transaction ({_short(exc)})"
            )
        raw = getattr(signed, "raw_transaction", None)
        if raw is None:  # eth-account < 0.13 spelled the same field rawTransaction
            raw = getattr(signed, "rawTransaction")
        digest = signed.hash.hex()

    voice.row("transfer", f"{_ether(value)} ETH -> {recipient}")
    voice.row("gas", f"{tx['gas']} @ {Decimal(gas_price) / Decimal(10**9):.3f} gwei, nonce {nonce}")

    # 5. one send, one receipt, no retry
    if args.dry_run:
        voice.step(5, "dry run: signing only")
        voice.row("signed", f"{len(raw)} bytes, hash {digest}" if raw is not None else digest)
        voice.foot("dry run: the chain saw nothing")
        return EXIT_OK

    voice.step(5, "sending")
    try:
        if raw is not None:
            tx_hash = w3.eth.send_raw_transaction(raw)
        else:
            tx_hash = w3.eth.send_transaction(tx)
    except (Web3Exception, ValueError, RequestException) as exc:
        return _fail(voice, EXIT_REFUSED, f"the node refused it: {_short(exc)}")
    voice.row("tx", tx_hash.hex())

    timeout = float(args.timeout) if float(args.timeout) > 0 else float(config.receipt_timeout)
    voice.row("waiting", f"up to {timeout:.0f}s for a receipt")
    try:
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout, poll_latency=0.1)
    except TimeExhausted:
        return _fail(
            voice,
            EXIT_REFUSED,
            f"{tx_hash.hex()} was accepted but not mined within {timeout:.0f}s. It may still "
            "land: check the node before sending another one",
        )
    except network_errors as exc:
        return _fail(
            voice,
            EXIT_UNREACHABLE,
            f"lost the node while waiting ({_short(exc)}); {tx_hash.hex()} may still land",
        )

    status = int(receipt.get("status", 0))
    block = receipt.get("blockNumber")
    gas_used = int(receipt.get("gasUsed", 0))
    paid = int(receipt.get("effectiveGasPrice", gas_price))
    fee = gas_used * paid

    try:
        sender_after = int(w3.eth.get_balance(sender))
        recipient_after = int(w3.eth.get_balance(recipient))
    except network_errors:
        sender_after = recipient_after = None

    if status != 1:
        voice.row("block", f"{block} - status {status}")
        voice.foot("the breath was refused")
        return _fail(
            voice,
            EXIT_REFUSED,
            f"mined in block {block} and reverted, spending {_ether(fee)} ETH of gas",
        )

    voice.foot("the breath was taken")
    voice.row("tx", tx_hash.hex())
    voice.row(
        "block",
        f"{block} - status 1 - gas {gas_used} @ {Decimal(paid) / Decimal(10**9):.3f} gwei "
        f"= {_ether(fee)} ETH of fees",
    )
    if sender_after is not None and recipient_after is not None:
        arrived = recipient_after - recipient_before
        left = sender_before - sender_after
        voice.row("moved", f"{_ether(arrived)} ETH arrived, {_ether(left)} ETH left the sender")
        voice.row("sender", f"{sender}  {_ether(sender_before)} -> {_ether(sender_after)} ETH")
        voice.row(
            "recipient",
            f"{recipient}  {_ether(recipient_before)} -> {_ether(recipient_after)} ETH",
        )
    return EXIT_OK


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="first_breath.py",
        description="Send one real transfer on a testnet and wait for its receipt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES,
    )
    parser.add_argument(
        "--to",
        default="",
        metavar="ADDRESS",
        help="recipient; default: the node's first account that is not the signer",
    )
    parser.add_argument(
        "--amount",
        default=DEFAULT_AMOUNT,
        metavar="ETH",
        help=f"test ether to send (default {DEFAULT_AMOUNT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="sign and print the raw transaction, send nothing",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="receipt timeout (default: TESTNET_RECEIPT_TIMEOUT, else 120)",
    )
    parser.add_argument(
        "--allow-mainnet",
        action="store_true",
        help="permit a mainnet chain id; also needs TESTNET_ALLOW_MAINNET=1",
    )
    parser.add_argument("--no-color", action="store_true", help="plain output")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = _parser().parse_args(argv)
    voice = Voice(color=_wants_color(args.no_color))
    try:
        return _breath(args, voice)
    except KeyboardInterrupt:
        print("\ninterrupted: nothing here retries, so nothing was sent twice", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
