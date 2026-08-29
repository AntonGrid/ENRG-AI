"""Market feeds & model tests (agent.market) — offline/deterministic."""

import numpy as np

from agent.market import (
    MarketCache,
    fetch_prices,
    forecast_price_series,
    forecast_price_with_intervals,
    price_matrix,
)


def test_fetch_prices_offline_has_all_providers():
    feeds = fetch_prices(offline=True)
    sources = {f.source for f in feeds}
    assert sources == {"dayahead", "p2p", "spot", "macro"}
    for feed in feeds:
        assert feed.ts
        assert feed.prices


def test_energy_prices_are_sane_usd_per_kwh():
    feeds = fetch_prices(offline=True)
    for feed in feeds:
        if feed.source == "macro":
            continue
        price = feed.prices["usd_per_kwh"]
        assert 0.001 < price < 1.0  # a sane electricity price band (USD/kWh)


def test_price_series_is_not_constant():
    prices = [f.prices["usd_per_kwh"] for f in fetch_prices(offline=True) if f.source == "dayahead"]
    prices += [
        fetch_prices(offline=True, step=i)[0].prices["usd_per_kwh"]
        for i in range(1, 12)
    ]
    assert len(set(prices)) > 3  # intraday sinusoid, not a constant


def test_market_cache_returns_cached_snapshot():
    cache = MarketCache(ttl_sec=300, offline=True)
    first = cache.prices()
    second = cache.prices()
    assert len(first) == len(second)
    assert first[0].ts == second[0].ts  # served from cache, not refetched


def test_price_matrix_builds_chronological_rows():
    points = [
        {"ts": "2026-01-01T00:00:00Z", "prices": {"dayahead": 0.10}},
        {"ts": "2026-01-01T01:00:00Z", "prices": {"dayahead": 0.14, "spot": 0.12}},
    ]
    matrix, columns = price_matrix(points)
    assert columns == ["dayahead", "spot"]
    assert matrix.shape == (2, 2)
    assert matrix[0].tolist() == [0.10, 0.0]  # spot forward-filled as zero


def test_forecast_price_series_keeps_shape_contract():
    series = np.array([[0.10, 0.12], [0.14, 0.11]])
    forecast = forecast_price_series(series, steps=3)
    assert forecast.shape == (3, 2)


def test_forecast_price_series_continues_linear_trend():
    series = np.array([[0.10], [0.14], [0.18]])  # slope +0.04
    forecast = forecast_price_series(series, steps=3)
    assert forecast.shape == (3, 1)
    # Holt should keep the learned slope, not repeat the last value.
    assert np.allclose(forecast[:, 0], [0.22, 0.26, 0.30], atol=0.02)


def test_forecast_price_with_intervals_orders_band():
    series = np.linspace(0.10, 0.16, 7).reshape(-1, 1)
    spec = forecast_price_with_intervals(series, steps=4)
    assert len(spec) == 1
    band = spec[0]
    assert len(band["point"]) == 4
    assert len(band["low"]) == 4
    assert len(band["high"]) == 4
    for lo, pt, hi in zip(band["low"], band["point"], band["high"]):
        assert lo <= pt <= hi
    assert band["rmse"] >= 0
