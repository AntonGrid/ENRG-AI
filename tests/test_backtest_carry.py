"""The funding collector: every page read once, every hole left where the venue left it.

These tests never open a socket. Each answer from HTX is a dictionary built here and handed to
the fetchers through their ``fetch=`` parameter, which is why that parameter exists: a cache
that talks to a live venue on every test run is a test that fails on a train.

What is proven, in the order the file proves it:

* **the walk.** ``page_size=200`` is clamped to 100 and ``total_page`` is computed against
  100, so the walk is driven by the venue's own total rather than by the width requested:
  250 records over three pages arrive as 250, once each, oldest first.
* **the hole.** A missing funding period stays missing. It is counted, located and measured
  in periods — never interpolated, never forward-filled, never quietly dropped.
* **the zero.** ``null`` and ``0E-18`` mean *the venue published no rate* and are carried as
  ``None``; a literal ``"0.000000000000000000"`` is a published zero and stays ``0.0``. Every
  average is computed over the reported records only.
* **the cache.** A second read makes zero requests, the manifest's ``sha256`` is the CSV's
  own, and ``data_sha256`` is unchanged by a refetch — same records, different fetch time.
  A warm manifest is also enough for the CLI to skip the venue, cadence included.
* **the ranking.** ``batch_merged`` answers under ``ticks`` and quotes dollars in
  ``trade_turnover`` — not in ``data``, which it sends empty, and not in ``vol``, which is a
  count a thousand times smaller than the dollars.
* **the report.** Two runs over the same numbers produce the same file, byte for byte.
* **the bars.** The perp kline endpoint has one page and no ``from``/``to``: the fetcher asks
  once, reads ``trade_turnover`` and never ``vol``, and says so when the window is short.
* **the join.** Perp and spot bars are matched on their opens, and the offsets are counted.
* **the caveat.** ``DOES_NOT_SHOW`` is inside the hashed criteria block, so a report cannot
  drop the sentence that says what it did not prove.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from agent.backtest import carry as C
from agent.backtest.criteria import verify
from agent.backtest.data import Candle, CandleSeries, HtxError

STEP = 8 * 3600
CODE = "BTC-USDT"


def _funding_row(at: float, rate: object, *, premium: object = "0.0001") -> dict:
    """One row in the shape ``swap_historical_funding_rate`` sends it, milliseconds and all."""
    return {
        "contract_code": CODE,
        "funding_time": int(at * 1000),
        "funding_rate": rate,
        "avg_premium_index": premium,
        "realized_rate": None,
    }


def _page(rows: list, *, total_size: int, total_page: int, page: int = 1) -> dict:
    """One answer in the shape the endpoint sends it: totals outside, rows inside ``data``."""
    return {
        "status": "ok",
        "ts": 1760000000000,
        "data": {
            "total_size": str(total_size),
            "total_page": total_page,
            "current_page": page,
            "data": rows,
        },
    }


def _history(count: int, *, start: float = 1_600_000_000.0, step: int = STEP) -> list:
    """``count`` rows ``step`` seconds apart, oldest first, all with a published rate."""
    return [_funding_row(start + index * step, "0.0001") for index in range(count)]


class _Venue:
    """A fake venue: answers a URL from a dictionary and counts every request it served.

    ``strict`` is the switch the cache test needs: with it on, any request at all is a failure,
    which is how "the second read made no call" is asserted rather than assumed.
    """

    def __init__(self, answers: dict, *, strict: bool = False) -> None:
        self.answers = answers
        self.strict = strict
        self.urls: list = []

    def __call__(self, url: str, timeout: float = 0.0) -> object:
        self.urls.append(url)
        if self.strict:
            raise AssertionError(f"the venue was called with {url} and it should not have been")
        for fragment, answer in sorted(self.answers.items(), key=lambda item: -len(item[0])):
            if fragment in url:
                if callable(answer):
                    return answer(url)
                return answer
        raise AssertionError(f"the fake venue has no answer for {url}")


def _one_page(rows: list) -> _Venue:
    """A fake venue that serves these rows as its only page, saying it has exactly that many."""
    return _Venue({"page_index=1": _page(rows, total_size=len(rows), total_page=1)})


def test_the_walk_reads_every_page_once_and_against_the_venues_own_total():
    """250 records, three pages of 100, and a request that asked for 200 rows a page.

    The clamp is the whole point: a walk that trusted ``page_size=200`` would stop at page 2
    of 3 and call 200 of 250 records a complete history. The walk is driven by ``total_page``,
    which the venue computed against the 100 it actually serves.
    """
    venue = _Venue(_paged_history(250))
    series = C.fetch_funding(CODE, period_seconds=float(STEP), fetch=venue, delay=0.0)

    assert len(series.records) == 250
    times = [record.at for record in series.records]
    assert times == sorted(times)
    assert len(set(times)) == 250
    assert [C._page_index(url) for url in venue.urls] == [1, 2, 3]
    assert all("page_size=200" in url for url in venue.urls)
    assert any("clamped" in note for note in series.warnings)
    assert any("total_page" in note for note in series.warnings)
    assert not any("shortfall" in note for note in series.warnings)


def test_a_missing_period_is_a_hole_and_is_never_filled():
    """Eleven payments where twelve were due: the series is twelve periods long, not eleven."""
    rows = _history(12)
    del rows[5]
    series = C.fetch_funding(
        CODE, period_seconds=float(STEP), fetch=_one_page(rows), delay=0.0
    )

    assert len(series.records) == 11
    assert len(series.rates()) == 11
    holes = series.gaps()
    assert len(holes) == 1
    before, after, missing = holes[0]
    assert after - before == 2 * STEP
    assert missing == 1
    assert series.max_gap_periods == 1
    # The neighbours are the venue's own timestamps: nothing was invented between them.
    assert before == 1_600_000_000.0 + 4 * STEP
    assert after == 1_600_000_000.0 + 6 * STEP


def test_an_unreported_rate_is_not_a_zero():
    """``null`` and ``0E-18`` are absences; a written-out zero is a published zero."""
    rows = [
        _funding_row(1_600_000_000.0, None),
        _funding_row(1_600_000_000.0 + STEP, "0E-18"),
        _funding_row(1_600_000_000.0 + 2 * STEP, "0.000000000000000000"),
        _funding_row(1_600_000_000.0 + 3 * STEP, "-0.0001"),
    ]
    series = C.fetch_funding(
        CODE, period_seconds=float(STEP), fetch=_one_page(rows), delay=0.0
    )

    assert len(series.records) == 4
    assert series.unreported == 2
    assert series.reported == 2
    assert [record.reported for record in series.records] == [False, False, True, True]
    assert series.records[0].funding_rate is None
    assert series.records[1].funding_rate is None
    assert series.records[2].funding_rate == 0.0
    assert series.records[3].funding_rate == pytest.approx(-0.0001)
    # The average is over what the venue published: two records, not four.
    assert series.rates() == [0.0, pytest.approx(-0.0001)]
    assert series.mean_bps == pytest.approx((-0.0001 / 2.0) * 1e4)
    assert series.reported_share == pytest.approx(0.5)


def _paged_history(count: int, *, page_size: int = 100) -> dict:
    """A venue answer keyed by ``page_index``, newest first, ``page_size`` rows per page."""
    rows = _history(count)
    total_page = (count + page_size - 1) // page_size
    answers = {}
    for index in range(total_page):
        chunk = rows[index * page_size : (index + 1) * page_size]
        answers[f"page_index={index + 1}"] = _page(
            list(reversed(chunk)), total_size=count, total_page=total_page, page=index + 1
        )
    return answers


def _series_of(
    contract: str,
    count: int,
    *,
    start: float = 1_600_000_000.0,
    step: int = STEP,
    rate: float = 0.0001,
) -> C.FundingSeries:
    """A series built in memory, for the tests that are about the report and not the walk."""
    return C.FundingSeries(
        contract=contract,
        period_seconds=float(step),
        records=[
            C.FundingRecord(at=start + index * step, funding_rate=float(rate), premium_index=0.0001)
            for index in range(count)
        ],
    )


def test_the_second_read_is_a_cache_hit_and_makes_no_request(tmp_path):
    """A valid manifest means zero calls — proven with a transport that fails if called."""
    rows = _history(5)
    venue = _one_page(list(reversed(rows)))
    first = C.fetch_funding_cached(CODE, cache=tmp_path, fetch=venue, delay=0.0)
    assert len(venue.urls) == 1
    assert len(first.records) == 5

    quiet = _Venue({}, strict=True)
    second = C.fetch_funding_cached(CODE, cache=tmp_path, fetch=quiet, delay=0.0)
    assert quiet.urls == []
    assert [record.at for record in second.records] == [record.at for record in first.records]
    assert [record.funding_rate for record in second.records] == [
        record.funding_rate for record in first.records
    ]
    assert second.period_seconds == float(STEP)

    folder = tmp_path / CODE
    csv_bytes = (folder / C.FUNDING_FILE).read_bytes()
    manifest = C.load_manifest(folder / C.FUNDING_MANIFEST)
    assert manifest["schema"] == C.MANIFEST_SCHEMA
    assert manifest["contract"] == CODE
    assert manifest["records"] == 5
    assert manifest["sha256"] == hashlib.sha256(csv_bytes).hexdigest()
    assert manifest["data_sha256"] == C.manifest_hash(manifest)


def test_a_warm_cache_needs_no_socket_even_for_the_cadence(tmp_path):
    """``--fetch`` on a valid manifest is answered from disk, so the CLI asks the venue nothing.

    The cadence is the reason this needs proving: it used to be read from the live universe on
    every run, which made a repeat fetch fail on a machine without a route to HTX even though
    every byte of the answer was already on disk. Since the manifest records the declared period,
    the question "is this cache complete" is the whole decision — and it is answered here.
    """
    folder = tmp_path / "funding"
    assert C._cache_is_fresh(folder, CODE) is False

    C.write_funding(folder / CODE, _series_of(CODE, 6))
    assert C._cache_is_fresh(folder, CODE) is True
    assert C._cache_is_fresh(folder, CODE.lower()) is True
    assert C._cache_is_fresh(folder, "") is False

    csv_path = folder / CODE / C.FUNDING_FILE
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "1,0.0002,,\n", encoding="utf-8")
    assert C._cache_is_fresh(folder, CODE) is False

    (folder / CODE / C.FUNDING_MANIFEST).write_text("{not json", encoding="utf-8")
    assert C._cache_is_fresh(folder, CODE) is False


def test_a_refetch_of_the_same_records_keeps_data_sha256(tmp_path):
    """Same records are the same data, however long after the first fetch they arrive."""
    series = _series_of(CODE, 6)
    first = C.write_funding(tmp_path / "first", series, fetched_at="2026-01-01T00:00:00Z")
    second = C.write_funding(tmp_path / "second", series, fetched_at="2026-06-01T00:00:00Z")

    assert first["fetched_at"] == "2026-01-01T00:00:00Z"
    assert second["fetched_at"] == "2026-06-01T00:00:00Z"
    assert first["sha256"] == second["sha256"]
    assert first["data_sha256"] == second["data_sha256"]

    tampered = dict(second, records=second["records"] + 1)
    assert C.manifest_hash(tampered) != second["data_sha256"]


def test_the_universe_is_ranked_by_the_field_the_feed_actually_sends():
    """``batch_merged`` answers under ``ticks`` and quotes dollars in ``trade_turnover``.

    Both halves are easy to get wrong and neither is visible in a table: reading ``data`` (which
    the answer also carries, empty) leaves every turnover at zero and the ranking becomes the
    alphabet; reading ``vol`` ranks a thousand-fold lie. The table below would print in the same
    order either way, so the numbers are asserted and not the order alone.
    """
    payload = {
        "status": "ok",
        "ts": 1_760_000_000_000,
        "data": [],
        "ticks": [
            {
                "contract_code": "ETH-USDT",
                "amount": "900.0",
                "vol": "900000",
                "trade_turnover": "9000000.0",
            },
            {
                "contract_code": CODE,
                "amount": "2001.488",
                "vol": "2001488",
                "trade_turnover": "138370818.63",
            },
        ],
    }
    turnover = C.parse_turnover(payload)
    assert turnover[CODE] == pytest.approx(138_370_818.63)
    assert turnover["ETH-USDT"] == pytest.approx(9_000_000.0)

    contracts = C.parse_contracts(
        {
            "status": "ok",
            "data": [
                {"contract_code": "ETH-USDT", "settlement_period": "8", "create_date": "20201120"},
                {"contract_code": CODE, "settlement_period": "4", "create_date": "20201120"},
            ],
        },
        turnover=turnover,
    )
    ranked = C.rank_contracts(contracts)
    assert [item.contract_code for item in ranked] == [CODE, "ETH-USDT"]
    assert ranked[0].turnover == pytest.approx(138_370_818.63)
    assert ranked[0].period_seconds == 4 * 3600.0
    assert "138,370,819" in C.format_universe(contracts)

    # An empty list of tickers is an empty ranking, not a failure dressed up as data.
    assert C.parse_turnover({"status": "ok", "data": [], "ticks": []}) == {}
    with pytest.raises(HtxError, match="nope"):
        C.parse_turnover({"status": "error", "err-msg": "nope"})


def test_the_perp_kline_endpoint_has_one_page_and_no_window():
    """One request, no ``from``/``to``, ``trade_turnover`` as volume — and a warning on the cut."""
    bars = [
        {
            "id": 1_600_014_400,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "amount": 2.5,
            "vol": 2500.0,
            "count": 7,
            "trade_turnover": 999.0,
        },
        {
            "id": 1_600_000_000,
            "open": 99.0,
            "high": 100.5,
            "low": 98.0,
            "close": 100.0,
            "amount": 2.0,
            "vol": 2000.0,
            "count": 3,
        },
    ]
    venue = _Venue({"contract_code=BTC-USDT": {"status": "ok", "data": bars}})
    series = C.fetch_swap_klines(CODE, size=5, fetch=venue)

    assert len(venue.urls) == 1
    url = venue.urls[0]
    assert "from=" not in url and "to=" not in url
    assert "size=5" in url and "period=4hour" in url and "contract_code=BTC-USDT" in url
    assert [bar.at for bar in series.rows] == [1_600_000_000.0, 1_600_014_400.0]
    # 999.0 is what the venue called turnover; ``vol`` is a base count and ``amount * close`` is
    # only the fallback. Reading either of those as dollars would be off by a factor of a thousand.
    assert series.rows[0].volume == pytest.approx(2.0 * 100.0)
    assert series.rows[1].volume == pytest.approx(999.0)
    assert any("amount * close" in note for note in series.warnings)
    assert any("no second page" in note for note in series.warnings)
    assert series.interval == 14400.0
    assert series.period == "4hour"


def test_the_spot_perp_join_counts_what_it_could_not_pair():
    """Two feeds on one clock, one bar on each side with no partner: both are counted."""
    perp = CandleSeries(
        symbol=CODE,
        period="4hour",
        rows=[
            Candle(at=1_600_000_000.0, open=100.0, high=102.0, low=99.0, close=101.0, volume=1.0),
            Candle(
                at=1_600_000_000.0 + STEP,
                open=101.0,
                high=103.0,
                low=100.0,
                close=102.0,
                volume=1.0,
            ),
            Candle(
                at=1_600_000_000.0 + 2 * STEP,
                open=102.0,
                high=104.0,
                low=101.0,
                close=103.0,
                volume=1.0,
            ),
            Candle(
                at=1_600_000_000.0 + 3 * STEP,
                open=103.0,
                high=105.0,
                low=102.0,
                close=104.0,
                volume=1.0,
            ),
        ],
    )
    spot = CandleSeries(
        symbol="btcusdt",
        period="4hour",
        rows=[
            Candle(at=1_600_000_000.0, open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0),
            Candle(
                at=1_600_000_000.0 + STEP,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1.0,
            ),
            Candle(
                at=1_600_000_000.0 + 2 * STEP,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1.0,
            ),
            Candle(
                at=1_600_000_000.0 + 4 * STEP,
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=1.0,
            ),
        ],
    )
    alignment = C.check_alignment(perp, spot, contract=CODE)

    assert alignment.contract == CODE
    assert alignment.perp_bars == 4
    assert alignment.spot_bars == 4
    assert alignment.joined == 3
    assert alignment.perp_only == 1
    assert alignment.spot_only == 1
    assert alignment.worst_offset_seconds == 0.0
    assert alignment.basis_mean_bps == pytest.approx(200.0)
    assert alignment.basis_median_bps == pytest.approx(200.0)
    assert alignment.basis_p95_bps == pytest.approx(290.0)
    assert "sampled" in alignment.note
    assert C.check_alignment(CandleSeries(), spot).joined == 0


def test_the_report_carries_what_it_does_not_show_inside_the_hash(tmp_path):
    """The honest half is hashed with the gate, so a passing run cannot quietly drop it."""
    report = C.report_of([_series_of(CODE, 40)])
    written = C.write_coverage(tmp_path / "coverage.json", report)
    document = json.loads(written["json"].read_text(encoding="utf-8"))

    assert list(document["criteria"]["does_not_show"]) == list(C.DOES_NOT_SHOW)
    assert list(document["run"]["coverage"]["does_not_show"]) == list(C.DOES_NOT_SHOW)
    assert list(document["run"]["coverage"]["acceptance"]) == list(C.COVERAGE_ACCEPTANCE)
    assert document["criteria"]["gate"] == dict(sorted(C.COVERAGE_GATE.items()))
    assert document["criteria_sha256"] == C.coverage_criteria().fingerprint()
    assert document["run"]["coverage"]["criteria_sha256"] == document["criteria_sha256"]
    assert verify(document) == ""

    document["criteria"]["does_not_show"] = document["criteria"]["does_not_show"][:-1]
    assert "changed after the run" in verify(document)


def test_two_reports_over_the_same_numbers_are_the_same_bytes(tmp_path):
    """The file is a function of the data: same cache, same bytes, twice."""
    report = C.report_of(
        [_series_of(CODE, 200), _series_of("ETH-USDT", 120)],
        hashes={CODE: "a" * 64, "ETH-USDT": "b" * 64},
        universe={"contracts": 2, "read_utc": "2026-01-01T00:00:00Z"},
    )
    first = C.write_coverage(tmp_path / "one" / "coverage.json", report)
    second = C.write_coverage(tmp_path / "two" / "coverage.json", report)

    assert first["json"].read_bytes() == second["json"].read_bytes()
    assert first["criteria"].read_bytes() == second["criteria"].read_bytes()
    document = json.loads(first["json"].read_text(encoding="utf-8"))
    assert verify(document) == ""
    assert document["run"]["coverage"]["totals"]["contracts"] == 2
    assert document["run"]["coverage"]["totals"]["rates_averaged_across_contracts"] is False
    coverages = document["run"]["coverage"]["coverages"]
    assert [row["contract"] for row in coverages] == [CODE, "ETH-USDT"]
    assert coverages[0]["data_sha256"] == "a" * 64
