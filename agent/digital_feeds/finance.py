"""Finance feed — currencies (open.er-api.com) and crypto (CoinGecko).

Both are free, keyless public feeds. Energy-commodity spot prices usually
require a key (EIA / ICE); the pluggable interface below is where such a
feed would attach.
"""
from __future__ import annotations

import datetime as dt
import math
import random
from typing import Dict, Optional

ER_API_URL = "https://open.er-api.com/v6/latest/USD"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def fetch(client) -> Dict:
    er = client.get(ER_API_URL)
    er.raise_for_status()
    rates = er.json()["rates"]

    cg = client.get(
        COINGECKO_URL,
        params={"ids": "bitcoin", "vs_currencies": "usd"},
    )
    cg.raise_for_status()
    btc_usd = float(cg.json()["bitcoin"]["usd"])

    return {
        "domain": "finance",
        "source": "er-api+coingecko",
        "ts": _now_iso(),
        "metrics": {
            "usd_eur": float(rates.get("EUR", 0.0)),
            "usd_rub": float(rates.get("RUB", 0.0)),
            "btc_usd": btc_usd,
        },
    }


def fetch_offline(step: int = 0, ts: Optional[str] = None) -> Dict:
    """Deterministic synthetic market point (trend + weekly cycle + noise)."""
    rng = random.Random(2_000 + step)
    t = (step % 168) / 168.0  # weekly cycle
    usd_eur = 0.92 + 0.03 * math.sin(2 * math.pi * t) + 0.0002 * step + rng.uniform(-0.005, 0.005)
    usd_rub = 88.0 + 0.05 * step + 2.0 * math.sin(2 * math.pi * t / 2) + rng.uniform(-0.5, 0.5)
    btc_usd = 50_000 + 80.0 * step + 3_000 * math.sin(2 * math.pi * t) + rng.uniform(-200, 200)
    return {
        "domain": "finance",
        "source": "er-api+coingecko-offline",
        "ts": ts or _now_iso(),
        "metrics": {
            "usd_eur": round(usd_eur, 4),
            "usd_rub": round(usd_rub, 2),
            "btc_usd": round(btc_usd, 2),
        },
    }
