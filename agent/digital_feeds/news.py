"""News feed — RSS parsing without external dependencies.

Tracks "global event activity": how many items each feed publishes. A spike
in a feed's item count (or headline length) is a real-world event signal.
"""
from __future__ import annotations

import datetime as dt
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from urllib.parse import urlparse

DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://techcrunch.com/feed/",
]


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _host(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "").split(".")[0]


def _count_items(client, url: str) -> Optional[int]:
    resp = client.get(url)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    items = root.findall(".//item")
    return len(items)


def fetch(client, feeds: List[str] = DEFAULT_FEEDS) -> Dict:
    metrics: Dict[str, float] = {}
    for url in feeds:
        try:
            count = _count_items(client, url)
        except Exception:
            continue
        if count is not None:
            metrics[f"news_{_host(url)}_items"] = float(count)
    return {
        "domain": "news",
        "source": "rss",
        "ts": _now_iso(),
        "metrics": metrics,
    }


def fetch_offline(step: int = 0, ts: Optional[str] = None) -> Dict:
    """Deterministic synthetic news activity (occasional event spikes)."""
    rng = random.Random(4_000 + step)
    spike = 25 if (step % 17 == 0) else 0  # a "global event" every 17 steps
    return {
        "domain": "news",
        "source": "rss-offline",
        "ts": ts or _now_iso(),
        "metrics": {
            "news_bbc_items": float(10 + spike + rng.uniform(0, 4)),
            "news_techcrunch_items": float(8 + (5 if step % 23 == 0 else 0) + rng.uniform(0, 3)),
        },
    }
