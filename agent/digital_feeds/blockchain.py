"""Blockchain feed — Solana public RPC (no API key).

On-chain activity: block height, slot, epoch. A stable, keyless RPC is used;
DeFi TVL / gas metrics would attach as additional pluggable feeds.
"""
from __future__ import annotations

import datetime as dt
import random
from typing import Dict, Optional

SOLANA_RPC = "https://api.mainnet-beta.solana.com"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _rpc(client, method: str) -> dict:
    resp = client.post(
        SOLANA_RPC,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": []},
    )
    resp.raise_for_status()
    return resp.json()["result"]


def fetch(client) -> Dict:
    block_height = float(_rpc(client, "getBlockHeight"))
    epoch_info = _rpc(client, "getEpochInfo")
    return {
        "domain": "blockchain",
        "source": "solana-rpc",
        "ts": _now_iso(),
        "metrics": {
            "solana_block_height": block_height,
            "solana_slot": float(epoch_info.get("absoluteSlot", 0)),
            "solana_epoch": float(epoch_info.get("epoch", 0)),
        },
    }


def fetch_offline(step: int = 0, ts: Optional[str] = None) -> Dict:
    """Deterministic synthetic chain activity (monotonic blocks + noise)."""
    rng = random.Random(5_000 + step)
    block = 240_000_000 + step * 400 + rng.uniform(-50, 50)
    slot = block * 1.0 + rng.uniform(-10, 10)
    epoch = (block // 432_000) * 1.0
    return {
        "domain": "blockchain",
        "source": "solana-rpc-offline",
        "ts": ts or _now_iso(),
        "metrics": {
            "solana_block_height": round(block, 0),
            "solana_slot": round(slot, 0),
            "solana_epoch": epoch,
        },
    }
