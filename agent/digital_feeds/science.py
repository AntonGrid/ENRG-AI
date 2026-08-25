"""Science / demography feed — World Bank open datasets (no API key).

Population and urbanization — the human-world context any physical network
operates in. Long-horizon, slow-moving signals.
"""
from __future__ import annotations

import datetime as dt
import random
from typing import Dict, Optional

WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"

INDICATORS = {
    "SP.POP.TOTL": "population",
    "SP.URB.TOTL.IN.ZS": "urbanization_pct",
}
DEFAULT_COUNTRY = "US"


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _latest_value(data) -> Optional[float]:
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
        resp = client.get(
            WB_URL.format(country=country, indicator=indicator),
            params={"format": "json", "per_page": 5},
        )
        resp.raise_for_status()
        value = _latest_value(resp.json())
        if value is not None:
            metrics[name] = value
    return {
        "domain": "science",
        "source": "worldbank",
        "ts": _now_iso(),
        "metrics": metrics,
    }


def fetch_offline(step: int = 0, ts: Optional[str] = None) -> Dict:
    """Deterministic synthetic demography point (slow growth)."""
    rng = random.Random(6_000 + step)
    population = 331_000_000 + 1_500 * step + rng.uniform(-2_000, 2_000)
    urbanization = 83.0 + 0.001 * step + rng.uniform(-0.05, 0.05)
    return {
        "domain": "science",
        "source": "worldbank-offline",
        "ts": ts or _now_iso(),
        "metrics": {
            "population": round(population, 0),
            "urbanization_pct": round(urbanization, 3),
        },
    }
