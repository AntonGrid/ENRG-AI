"""Universal digital data feeds (domain-agnostic, Phase: digital layer).

Collects open-world signals — weather, finance, macro, news, blockchain,
science — in one normalized shape so the ENRG-AI model can learn trends,
anomalies and cross-domain links of *any* physical process, not just energy.

- ``registry`` — unified ``collect`` / ``collect_series`` + ``FeedResult``;
- ``weather``, ``finance``, ``macro``, ``news``, ``blockchain``, ``science``
  — pluggable sources (each with a deterministic offline generator).
"""

from agent.digital_feeds.registry import (
    DEFAULT_FEEDS,
    FEED_SOURCES,
    OFFLINE_SOURCES,
    FeedResult,
    collect,
    collect_series,
)

__all__ = [
    "DEFAULT_FEEDS",
    "FEED_SOURCES",
    "OFFLINE_SOURCES",
    "FeedResult",
    "collect",
    "collect_series",
]
