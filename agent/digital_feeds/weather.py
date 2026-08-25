"""Weather feed — Open-Meteo (free, no API key).

Online: real forecast for a location. Offline: a deterministic synthetic
series (daily temperature sinusoid + slow trend + noise) so training and
tests run without network access.
"""
from __future__ import annotations

import datetime as dt
import math
import random
from typing import Dict, Optional

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_LAT = 52.52  # Berlin
DEFAULT_LON = 13.41


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def fetch(client, lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> Dict:
    resp = client.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,cloud_cover,wind_speed_10m,precipitation",
        },
    )
    resp.raise_for_status()
    current = resp.json()["current"]
    return {
        "domain": "weather",
        "source": "open-meteo",
        "ts": current.get("time") or _now_iso(),
        "metrics": {
            "temperature_c": float(current["temperature_2m"]),
            "cloud_cover_pct": float(current["cloud_cover"]),
            "wind_speed_kmh": float(current["wind_speed_10m"]),
            "precipitation_mm": float(current["precipitation"]),
        },
    }


def fetch_offline(step: int = 0, ts: Optional[str] = None) -> Dict:
    """Deterministic synthetic weather point for ``step`` (0, 1, 2, …)."""
    rng = random.Random(1_000 + step)
    t = (step % 48) / 48.0  # daily cycle
    temperature = 15.0 + 8.0 * math.sin(2 * math.pi * t) + 0.02 * step + rng.uniform(-0.4, 0.4)
    cloud = 40.0 + 20.0 * math.sin(2 * math.pi * t / 7) + rng.uniform(-4.0, 4.0)
    wind = 12.0 + 3.0 * math.sin(2 * math.pi * t / 5) + rng.uniform(-1.0, 1.0)
    precipitation = max(0.0, rng.uniform(0.0, 2.5))
    return {
        "domain": "weather",
        "source": "open-meteo-offline",
        "ts": ts or _now_iso(),
        "metrics": {
            "temperature_c": round(temperature, 2),
            "cloud_cover_pct": round(cloud, 1),
            "wind_speed_kmh": round(wind, 1),
            "precipitation_mm": round(precipitation, 2),
        },
    }
