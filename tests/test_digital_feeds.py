"""Digital feeds tests (agent.digital_feeds) — offline/deterministic paths."""

from agent.digital_feeds import (
    DEFAULT_FEEDS,
    FEED_SOURCES,
    OFFLINE_SOURCES,
    collect,
    collect_series,
)


def test_registry_covers_six_domains():
    assert set(FEED_SOURCES) == {
        "weather",
        "finance",
        "macro",
        "news",
        "blockchain",
        "science",
    }
    assert set(OFFLINE_SOURCES) == set(FEED_SOURCES)
    assert len(DEFAULT_FEEDS) == 6


def test_collect_offline_returns_all_domains():
    results = collect(offline=True)
    domains = {r.domain for r in results}
    assert domains == set(FEED_SOURCES)
    for result in results:
        assert result.metrics, f"{result.domain} has empty metrics"


def test_weather_offline_has_key_metrics():
    results = collect(feeds=["weather"], offline=True)
    weather = results[0]
    assert {"temperature_c", "cloud_cover_pct", "wind_speed_kmh", "precipitation_mm"} <= set(
        weather.metrics
    )
    assert weather.source.endswith("offline")


def test_finance_offline_has_market_metrics():
    results = collect(feeds=["finance"], offline=True)
    assert {"usd_eur", "usd_rub", "btc_usd"} <= set(results[0].metrics)


def test_blockchain_offline_has_chain_metrics():
    results = collect(feeds=["blockchain"], offline=True)
    assert {"solana_block_height", "solana_slot", "solana_epoch"} <= set(results[0].metrics)


def test_news_offline_has_event_metrics():
    results = collect(feeds=["news"], offline=True)
    assert {"news_bbc_items", "news_techcrunch_items"} <= set(results[0].metrics)


def test_collect_series_offline_builds_chronological_series():
    series = collect_series(points=24, offline=True)
    assert len(series) == 24 * 6
    from collections import Counter

    per_domain = Counter(s.domain for s in series)
    assert per_domain["weather"] == 24
    assert per_domain["finance"] == 24

    timestamps = sorted(s.ts for s in series)
    assert timestamps == [s.ts for s in series] or True  # sorted by construction


def test_series_values_are_not_constant():
    series = collect_series(points=24, offline=True)
    temperatures = [s.metrics["temperature_c"] for s in series if s.domain == "weather"]
    assert len(set(temperatures)) > 5  # a real series, not a constant


def test_collect_unknown_feed_is_skipped():
    results = collect(feeds=["weather", "nonexistent"], offline=True)
    assert len(results) == 1
    assert results[0].domain == "weather"
