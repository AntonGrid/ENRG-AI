"""Market layer — normalized price feeds and price×time model.

Per the constitution (ADR-0003) this module produces **signals**, never
trades: prices feed the ``Recommender`` (``axis_core.ai``), and execution
belongs to the Policy Engine with limits and DAO gates.

- ``feeds`` — normalized USD/kWh feeds (dayahead, p2p, spot, macro) with a
  TTL cache and deterministic offline generators;
- ``model`` — price×time structures (matrix, naive forecast hook).
"""

from agent.market.feeds import PROVIDERS, UNITS, MarketCache, PriceFeed, fetch_prices
from agent.market.model import (
    PricePoint,
    forecast_price_series,
    forecast_price_with_intervals,
    price_matrix,
)

__all__ = [
    "PROVIDERS",
    "UNITS",
    "MarketCache",
    "PriceFeed",
    "PricePoint",
    "fetch_prices",
    "forecast_price_series",
    "forecast_price_with_intervals",
    "price_matrix",
]
