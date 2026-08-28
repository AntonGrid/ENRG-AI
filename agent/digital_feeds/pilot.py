"""DePIN pilot feed — real device proofs from the live ENRG oracle.

Sources (in order):
1. **On-chain** (default): the producer PDA on Solana devnet is written by
   every successful ``mint_energy`` transaction. We scan recent signatures of
   the producer, decode each mint_energy report (borsh) and reconstruct the
   proof stream (device_id, timestamp, energy_wh, nonce, oracle) — works even
   when the ESP32 counter is currently powered off, because the history is
   already committed to the blockchain.
2. Oracle REST (``/api/v1/proofs``) — recent proofs persisted by the oracle
   (ADR-0010 data bridge).
3. Offline deterministic generator (graceful degradation for tests).

Feed contract: ``fetch(client) -> dict`` in the same shape as every other
``digital_feeds`` module (``FeedResult(domain, source, ts, metrics)``).
Additionally exposes ``fetch_history()`` for the anomaly detector / digital
twin, and ``assess()`` for a lightweight plausibility score.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import random
from typing import Any, Dict, List, Optional

# ── Pilot constants (devnet v7.1 deployment) ────────────────────────────────
ENRG_MVP_PROGRAM_ID = "HkuC3FTGAf9ryPqH7fi3RbUHwP4TKFMg5WgHNWm6Vaxb"
RPC_URL = "https://api.devnet.solana.com"
ORACLE_URL = "https://enrg-oracle.onrender.com"
ORACLE_PROOFS_URL = ORACLE_URL + "/api/v1/proofs"
DEVICE_HEX = "cbec5afc382549012faf845ab25f593fe8f119d2ceb93f34ed308c283521584a"
DEVICE_PUBKEY = None  # resolved lazily (base58 of DEVICE_HEX)
# PDA(["producer", device_pubkey], enrg-mvp) — verified on devnet
# (4FCwMe82XRbbnNRBmK3sqH4uXN8dPwmys1PLaAjBq9M4).
PRODUCER_PDA = "4FCwMe82XRbbnNRBmK3sqH4uXN8dPwmys1PLaAjBq9M4"

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58_ALPHABET[r] + out
    pad = 0
    for b in data:
        if b == 0:
            pad += 1
        else:
            break
    return "1" * pad + out


def _b58decode(s: str) -> bytes:
    n = 0
    for c in s:
        n = n * 58 + _B58_ALPHABET.index(c)
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = 0
    for c in s:
        if c == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + body


def _resolve_addresses() -> None:
    global DEVICE_PUBKEY
    if DEVICE_PUBKEY is not None:
        return
    device_bytes = bytes.fromhex(DEVICE_HEX)
    DEVICE_PUBKEY = _b58encode(device_bytes)

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _rpc(client, method: str, params: List[Any]) -> Dict[str, Any]:
    resp = client.post(
        RPC_URL,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("error"):
        raise RuntimeError(f"RPC {method}: {body['error']}")
    return body["result"]


def decode_mint_report(data_b58: str) -> Optional[Dict[str, Any]]:
    """Decode a mint_energy instruction payload (anchor borsh OracleReport).

    OracleReport layout (after the 8-byte anchor discriminator):
    oracle(32) device_id(32) nonce(u64) device_timestamp(i64) verified_at(i64)
    energy_wh(u64) device_signature(64) oracle_signature(64) — 232 bytes total.
    """
    try:
        raw = _b58decode(data_b58)
    except Exception:
        return None
    if len(raw) != 232:
        return None
    oracle = _b58encode(raw[8:40])
    device_id = raw[40:72].hex()
    nonce = int.from_bytes(raw[72:80], "little")
    device_timestamp = int.from_bytes(raw[80:88], "little", signed=True)
    verified_at = int.from_bytes(raw[88:96], "little", signed=True)
    energy_wh = int.from_bytes(raw[96:104], "little")
    return {
        "device_id": device_id,
        "timestamp": device_timestamp,
        "verified_at": verified_at,
        "energy_wh": energy_wh,
        "nonce": nonce,
        "oracle": oracle,
    }


def fetch_history(
    client=None,
    limit: int = 40,
    max_scan: int = 60,
) -> List[Dict[str, Any]]:
    """Reconstruct recent proofs from on-chain mint_energy transactions.

    Returns a chronological list of ``{device_id, timestamp, energy_wh, nonce,
    oracle, slot, signature}`` (oldest first).
    """
    _resolve_addresses()
    owns = client is None
    if owns:
        import httpx

        client = httpx.Client(timeout=20.0)
    proofs: List[Dict[str, Any]] = []
    try:
        sigs = _rpc(
            client,
            "getSignaturesForAddress",
            [PRODUCER_PDA, {"limit": max_scan}],
        )
        for entry in sigs:
            sig = entry["signature"]
            if entry.get("err"):
                continue
            try:
                tx = _rpc(
                    client,
                    "getTransaction",
                    [sig, {"encoding": "json", "maxSupportedTransactionVersion": 0}],
                )
            except Exception:
                continue
            if not tx:
                continue
            msg = (tx.get("transaction") or {}).get("message") or {}
            keys = [
                k["pubkey"] if isinstance(k, dict) else k
                for k in msg.get("accountKeys", [])
            ]
            for ix in msg.get("instructions", []):
                pid_idx = ix.get("programIdIndex", -1)
                if pid_idx < 0 or pid_idx >= len(keys):
                    continue
                if keys[pid_idx] != ENRG_MVP_PROGRAM_ID:
                    continue
                rep = decode_mint_report(ix.get("data", ""))
                if not rep:
                    continue
                rep["slot"] = tx.get("slot", 0)
                rep["signature"] = sig
                proofs.append(rep)
                break
    finally:
        if owns:
            client.close()
    proofs.sort(key=lambda p: p["timestamp"])
    return proofs[-limit:]


def fetch(
    client,
    device_id: Optional[str] = None,
    limit: int = 30,
) -> Dict[str, Any]:
    """One FeedResult-shaped observation built from the latest proofs.

    Primary source: on-chain mint history (works with the counter powered
    off). Fallback: the oracle REST history; final fallback: offline.
    """
    history: List[Dict[str, Any]] = []
    source = "enrg-pilot-onchain"
    try:
        history = fetch_history(client, limit=limit, max_scan=max(limit * 2, 40))
    except Exception:
        source = "enrg-pilot-oracle"
        try:
            resp = client.get(
                ORACLE_PROOFS_URL,
                params={"device_id": device_id or DEVICE_HEX, "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            rows = data.get("proofs") or []
            history = [
                {
                    "device_id": r.get("device_id"),
                    "timestamp": r.get("ts"),
                    "energy_wh": r.get("energy_wh"),
                    "nonce": r.get("nonce"),
                    "signature": r.get("mint_tx"),
                }
                for r in rows
            ]
        except Exception:
            return fetch_offline(step=0)

    if not history:
        return fetch_offline(step=0)

    last = history[-1]
    total = sum(float(p["energy_wh"] or 0) for p in history)
    minted = sum(1 for p in history if p.get("signature"))
    intervals = []
    for a, b in zip(history, history[1:]):
        d = int(b["timestamp"]) - int(a["timestamp"])
        if d > 0:
            intervals.append(d)
    avg_interval = round(sum(intervals) / len(intervals)) if intervals else 0
    last_dt = dt.datetime.fromtimestamp(int(last["timestamp"]), dt.timezone.utc)
    return {
        "domain": "pilot",
        "source": source,
        "ts": last_dt.isoformat(timespec="seconds"),
        "metrics": {
            "energy_wh": float(last["energy_wh"] or 0),
            "nonce": float(last["nonce"] or 0),
            "proof_count": float(len(history)),
            "total_energy_wh": float(total),
            "minted_count": float(minted),
            "avg_interval_sec": float(avg_interval),
            "hour_of_day": float(last_dt.hour + last_dt.minute / 60.0),
            "day_of_week": float(last_dt.weekday()),
        },
    }


def fetch_offline(step: int = 0, ts: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic synthetic pilot point for ``step`` (0, 1, 2, …)."""
    rng = random.Random(2_000 + step)
    energy = max(0.0, 1.0 + rng.uniform(-0.3, 0.3))
    t = (step % 48) / 48.0  # daily cycle
    solar = max(0.0, 1.0 * (1.0 - abs(2 * t - 1)))  # daylight window
    return {
        "domain": "pilot",
        "source": "enrg-pilot-offline",
        "ts": ts or _now_iso(),
        "metrics": {
            "energy_wh": round(energy, 3),
            "nonce": float(100 + step),
            "proof_count": float(1),
            "total_energy_wh": round(energy * (step + 1), 3),
            "minted_count": float(1),
            "avg_interval_sec": 60.0,
            "hour_of_day": float(12.0 + 0.5 * math.sin(2 * math.pi * t)),
            "solar_plausibility": round(solar, 3),
        },
    }


def assess(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Lightweight plausibility assessment of a proof stream (ADR-0010 L2).

    Returns ``{score: 0..1, flags: [str], metrics: {...}}``. Advisory only —
    never gates proof acceptance (that stays in the Policy Engine).
    """
    if len(history) < 2:
        return {"score": 0.5, "flags": ["insufficient_data"], "metrics": {}}
    flags: List[str] = []
    intervals = []
    for a, b in zip(history, history[1:]):
        d = int(b["timestamp"]) - int(a["timestamp"])
        if d > 0:
            intervals.append(d)
    avg = sum(intervals) / len(intervals) if intervals else 0
    # 1) interval sanity: ±50 % around 60 s nominal
    if avg < 30 or avg > 300:
        flags.append("abnormal_interval")
    # 2) daylight plausibility: solar panel should not produce at night
    night_hits = 0
    for p in history:
        h = dt.datetime.fromtimestamp(int(p["timestamp"]), dt.timezone.utc).hour
        if h < 5 or h >= 21:
            if float(p["energy_wh"]) > 0.1:
                night_hits += 1
    if night_hits > max(1, len(history) // 4):
        flags.append("night_production")
    # 3) constant energy (no variance) is suspicious for a physical sensor
    vals = [float(p["energy_wh"]) for p in history]
    if max(vals) == min(vals) and len(set(vals)) == 1:
        flags.append("constant_energy")
    score = max(0.0, 1.0 - 0.33 * len(flags))
    return {
        "score": round(score, 3),
        "flags": flags,
        "metrics": {
            "n_proofs": len(history),
            "avg_interval_sec": round(avg, 1),
            "night_production_count": night_hits,
        },
    }

