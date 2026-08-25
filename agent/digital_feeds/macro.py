"""Macroeconomic feed — World Bank open indicators (no API key).

GDP, inflation (CPI) and unemployment for a country, latest annual values.
"""
from __future__ import annotations

import datetime as dt
import random
from typing import Dict, List, Optional

WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

# indicator → short metric name
INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_usd",
    "FP.CPI.TOTL.ZG": "inflation_pct",
    "SL.UEM.TOTL.ZS": "unemployment_pct",
}
DEFAULT_COUNTRY = "US"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _latest_value(data) -> Optional[float]:
    """World Bank returns [{header}, {entry}, ...]; pick the newest non-null."""
    if not isinstance(data, list) or len(data) < 2:
        return None
    for entry in data[1:]:
        value = entry.get("value")
        if value is not None:
            return float(value)
    return None


def fetch(client, country: str = DEFAULT_COUNTRY) -> Dict:
    metrics: Dict[str, float] = {}
    for indicator, name in INDICATORS.items():
        resp = client.get(WB_URL.format(country=country, indicator=indicator), params={"format": "json", "per_page": 5})
        resp.raise_for_status()
        value = _latest_value(resp.json())
        if value is not None:
            metrics[name] = value
    return {
        "domain": "macro",
        "source": "worldbank",
        "ts": _now_iso(),
        "metrics": metrics,
    }


def fetch_offline(step: int = 0, ts: Optional[str] = None) -> Dict:
    """Deterministic synthetic macro point (slow growth + noise)."""
    rng = random.Random(3_000 + step)
    gdp = 27_000_000_000_000 + 60_000_000_000 * step + rng.uniform(-1e10, 1e10)
    inflation = 3.0 + 0.01 * step + rng.uniform(-0.3, 0.3)
    unemployment = 4.0 + 0.002 * step + rng.uniform(-0.2, 0.2)
    return {
        "domain": "macro",
        "source": "worldbank-offline",
        "ts": ts or _now_iso(),
        "metrics": {
            "gdp_usd": round(gdp, 0),
            "inflation_pct": round(inflation, 2),
            "unemployment_pct": round(unemployment, 2),
        },
    }
