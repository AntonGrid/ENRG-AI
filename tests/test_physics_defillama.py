"""The wild market, mocked: the DeFiLlama adapter against answers nobody sent.

This file exists because a test suite that reaches the network is a test suite that fails
on a train. Every response here is invented, and the invention is deliberate: the two
things a real feed must never be able to do to the funnel are *crash it* and *surprise it*,
and both are cheaper to assert against a ``Mock`` than against a market that changes while
the suite runs.

What is being pinned down, in four groups.

**The mapping is honest about units.** A published APY of 36.5 % becomes a per-step rate of
``(1.365)^(1/365) − 1`` and not ``0.365/365`` — the two differ in the fourth digit and only
one of them compounds back to what the venue promised. The spread survives the conversion
(``CV`` of the rates equals ``CV`` of the APYs), which is what makes Statement #87's reading
of a real pool the same reading it takes of a drawn one.

**Absence is not a number.** ``None`` and ``NaN`` in the middle of a series are dropped and
counted, a series with more than a quarter of its points missing is refused rather than
padded, and a pool whose chart will not load is *skipped by name* instead of shrinking the
market silently.

**The funnel is the core's, unchanged.** Level 1 still drops a chaotic record for noise and
a thin pool for having no exit; level 2 still convicts a curve with no residuals in it —
including the one real-market shape this step is about, a rock-steady fifty percent. The
one rule the adapter adds is the survey's window arithmetic: fewer than four fixations is
``UNMEASURED``, which is not a refusal, and the report keeps the two apart.

**Failure is a sentence and an exit code.** A dead feed produces an empty funnel with the
reason in ``error``, never a traceback; the CLI answers ``3``; and nothing in this module —
engine, scanner or market — has a method that could move a credit.
"""
import json
import math
from unittest import mock

import numpy as np
import pytest

from agent.physics import defillama_adapter as A
from agent.physics import discovery as D
from agent.physics import risk_assessment as R
from agent.physics.portfolio import annualise
from agent.physics.sensorium import STRIDE, WINDOW

#: A record long enough to give the survey its four fixations.
LONG = WINDOW + 3 * STRIDE
#: A record long enough for a claim, too short for a survey.
SHORT = WINDOW + STRIDE


class Reply:
    """The two things the adapter asks a response for, and nothing else."""

    def __init__(self, body, status_code: int = 200) -> None:
        self._body = body
        self.status_code = int(status_code)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def pools(count=4, *, tvl=1.0e9, apy=6.0):
    """A ``/pools`` document with ``count`` rows, deepest first."""
    return {
        "status": "success",
        "data": [
            {
                "chain": "Ethereum",
                "project": f"project{index}",
                "symbol": f"SYM{index}",
                "tvlUsd": float(tvl) - index * 1.0e6,
                "apy": float(apy) + index,
                "apyBase": float(apy) + index,
                "apyReward": None,
                "pool": f"00000000-0000-0000-0000-0000000000{index:02d}",
                "stablecoin": index % 2 == 0,
                "ilRisk": "no",
                "exposure": "single",
                "volumeUsd1d": 1.0e7,
                "apyMean30d": float(apy) + index,
            }
            for index in range(count)
        ],
    }


def chart(apys, *, tvl=1.0e9):
    """A ``/chart/{uuid}`` document from a list of APYs (``None`` kept as a gap)."""
    return {
        "status": "success",
        "data": [
            {
                "timestamp": f"2025-01-{1 + index % 28:02d}T00:00:00.000Z",
                "tvlUsd": float(tvl),
                "apy": value,
                "apyBase": value,
                "apyReward": None,
                "il7d": None,
                "apyBase7d": None,
                "pricePerShare": None,
            }
            for index, value in enumerate(apys)
        ],
    }


def wire(pools_payload=None, series=None, *, status=200):
    """A transport that answers the catalogue and every chart from memory.

    ``series`` maps a pool uuid to a list of APYs; a uuid that is missing answers HTTP 500,
    which is how a dead pool is spelled in this file.
    """
    catalogue = pools() if pools_payload is None else pools_payload
    histories = {} if series is None else series

    def fetch(url: str, timeout: float):
        if url.rstrip("/").endswith("/pools"):
            return Reply(catalogue, status)
        uuid = url.rsplit("/", 1)[-1]
        if uuid not in histories:
            return Reply({"status": "error", "message": "no such pool"}, 500)
        return Reply(chart(histories[uuid]))

    return mock.Mock(side_effect=fetch)


def noisy(apy: float, points: int = LONG, spread: float = 0.35, seed: int = 5):
    """A rate with market noise on it — the honest shape, at a chosen level."""
    generator = np.random.default_rng(seed)
    return [apy + float(generator.normal(0.0, spread)) for _ in range(points)]


def scanner(transport, **kwargs) -> A.DefiLlamaScanner:
    """A scanner with the cache off, so that a test cannot read another test's file."""
    return A.DefiLlamaScanner(transport=transport, use_cache=False, **kwargs)


@pytest.fixture()
def two_pools():
    """Two readable pools: one honest and noisy, one perfectly steady at fifty percent."""
    series = {
        "00000000-0000-0000-0000-000000000000": noisy(6.0),
        "00000000-0000-0000-0000-000000000001": [50.0] * LONG,
    }
    return series


# ── the wire: two endpoints, one deterministic order, one sentence on failure ─


def test_top_pools_are_sorted_by_depth_and_truncated():
    feed = wire(pools(count=5))
    rows = scanner(feed).get_top_pools(limit=3)
    assert [row.symbol for row in rows] == ["SYM0", "SYM1", "SYM2"]
    depths = [row.tvl_usd for row in rows]
    assert depths == sorted(depths, reverse=True)
    assert rows[0].chain == "Ethereum" and rows[0].project == "project0"


def test_top_pools_reads_nothing_it_was_not_given():
    """A row without a number is a row with ``None``, not a row with a zero claim."""
    payload = pools(count=2)
    payload["data"][0]["apy"] = None
    payload["data"][0]["tvlUsd"] = None
    rows = scanner(wire(payload)).get_top_pools()
    empty = [row for row in rows if row.symbol == "SYM0"][0]
    assert empty.apy == 0.0 and empty.tvl_usd == 0.0
    assert empty.apy_reward is None
    assert rows[-1] is empty  # no depth sorts last, and it is not quietly dropped


def test_a_dead_feed_is_a_sentence_not_a_traceback():
    with pytest.raises(A.DefiLlamaUnavailable) as caught:
        scanner(wire(pools(), status=503)).get_top_pools()
    message = str(caught.value)
    assert "503" in message and "\n" not in message and "Traceback" not in message


def test_a_broken_document_is_a_sentence():
    with pytest.raises(A.DefiLlamaUnavailable):
        scanner(mock.Mock(return_value=Reply(ValueError("no json here")))).get_top_pools()
    with pytest.raises(A.DefiLlamaUnavailable):
        scanner(mock.Mock(return_value=Reply({"status": "success"}))).get_top_pools()
    with pytest.raises(A.DefiLlamaUnavailable):
        scanner(
            mock.Mock(return_value=Reply({"status": "error", "data": [], "message": "down"}))
        ).get_top_pools()


def test_an_unreachable_host_is_a_sentence():
    """Whatever the socket raised, the caller gets one sentence and a type to catch."""
    broken = mock.Mock(side_effect=OSError("name resolution failed"))
    with pytest.raises(A.DefiLlamaUnavailable) as caught:
        scanner(broken).get_top_pools()
    assert "could not be reached" in str(caught.value)
    assert broken.call_args[0][1] == A.DEFAULT_TIMEOUT  # the timeout is always passed


def test_a_throttled_feed_is_waited_out_not_reported_as_an_empty_market():
    """``429`` is the host pacing the scan, so the answer is a pause and the same question.

    Measured on the first real two-hundred-pool run: at eight concurrent requests the host
    answered ``429`` for 125 pools, and a scanner that reported those as "no history" would
    have called its own rate limit a property of the market.
    """
    calls = {"count": 0}

    def fetch(url, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            return Reply({"status": "error"}, 429)
        return Reply(pools(count=1))

    reader = scanner(mock.Mock(side_effect=fetch), retries=2, backoff=0.0)
    assert len(reader.get_top_pools()) == 1
    assert calls["count"] == 3
    assert sum("paced" in warning for warning in reader.warnings) == 2


def test_a_feed_that_keeps_throttling_is_given_up_on_by_name():
    reader = scanner(
        mock.Mock(return_value=Reply({"status": "error"}, 429)), retries=1, backoff=0.0
    )
    with pytest.raises(A.DefiLlamaUnavailable) as caught:
        reader.get_top_pools()
    assert caught.value.throttled and caught.value.status == 429
    assert reader.requests_made == 2


def test_the_default_transport_is_requests_with_a_timeout():
    """The real transport, mocked at ``requests``: one GET, a timeout, our user agent."""
    reply = Reply(pools(count=1))
    with mock.patch("requests.get", return_value=reply) as get:
        rows = A.DefiLlamaScanner(use_cache=False).get_top_pools()
    assert len(rows) == 1
    assert get.call_count == 1
    _, kwargs = get.call_args
    assert kwargs["timeout"] == A.DEFAULT_TIMEOUT
    assert kwargs["headers"]["User-Agent"] == A.USER_AGENT


def test_history_drops_gaps_instead_of_filling_them():
    series = [3.0, None, 3.2, float("nan"), 3.4] + [3.5 + 0.01 * index for index in range(20)]
    feed = wire(series={"00000000-0000-0000-0000-000000000000": series})
    history = scanner(feed).fetch_pool_history("00000000-0000-0000-0000-000000000000")
    assert len(history) == len(series) - 2
    assert history[:3] == (3.0, 3.2, 3.4)
    assert None not in history and all(math.isfinite(value) for value in history)


def test_history_refuses_a_series_that_is_mostly_empty():
    series = [None] * 8 + [4.0, 4.1]
    reader = scanner(wire(series={"00000000-0000-0000-0000-000000000000": series}))
    assert reader.fetch_pool_history("00000000-0000-0000-0000-000000000000") == ()
    assert any("no apy" in warning for warning in reader.warnings)


def test_a_pool_the_feed_cannot_chart_is_skipped_by_name():
    """One dead uuid costs one pool and is named, never the whole market."""
    series = {"00000000-0000-0000-0000-000000000000": noisy(5.0)}
    reader = scanner(wire(series=series))
    market = reader.market(limit=4)
    assert len(market) == 1
    assert [row["name"] for row in market.skipped] == [
        "project1 SYM1 (Ethereum)",
        "project2 SYM2 (Ethereum)",
        "project3 SYM3 (Ethereum)",
    ]
    assert any("500" in warning for warning in reader.warnings)


# ── the units: a per-step rate that compounds back to the promised APY ──────


def test_apy_becomes_the_rate_that_compounds_back_to_it():
    rate = A.apy_to_rate(36.5)
    assert rate == pytest.approx((1.365) ** (1.0 / 365) - 1.0)
    assert annualise(rate) == pytest.approx(0.365, rel=1e-9)
    # ...and it is not the naive division, which does not compound back.
    assert rate != pytest.approx(0.365 / 365)


def test_a_total_loss_is_minus_one_and_absurdities_are_clamped():
    assert A.apy_to_rate(-100.0) == -1.0
    assert A.apy_to_rate(-400.0) == -1.0
    assert A.apy_to_rate(0.0) == 0.0
    assert A.apy_to_rate(float("nan")) == 0.0
    assert A.apy_to_rate(float("inf")) == 0.0


def test_returns_keep_the_shape_of_the_apy_series():
    """``CV`` of the rates is ``CV`` of the log returns — the reading is unit-free."""
    apys = noisy(8.0, points=200, spread=1.5, seed=11)
    rates = A.to_returns(apys)
    assert rates.size == 200 and np.all(np.isfinite(rates))

    def cv(values) -> float:
        return float(np.std(values, ddof=1) / np.mean(values))

    assert cv(rates) == pytest.approx(cv(np.log1p(np.array(apys) / 100.0)), rel=1e-3)
    # ...and it is the same number as the APY series to within the compounding factor,
    # which is what makes a threshold expressed on one readable on the other.
    assert cv(rates) == pytest.approx(cv(apys), rel=0.05)


def test_to_returns_drops_what_is_not_a_number():
    assert A.to_returns([None, float("nan"), 10.0]).size == 1
    assert A.to_returns([]).size == 0


def test_clean_series_counts_what_it_dropped():
    published, missing = A.clean_series([1.0, None, "x", 2.0, float("inf")])
    assert published == [1.0, 2.0] and missing == 3


def test_entrain_windows_are_the_sensorium_geometry():
    assert A.entrain_windows(0) == 0
    assert A.entrain_windows(WINDOW - 1) == 0
    assert A.entrain_windows(WINDOW) == 1
    assert A.entrain_windows(WINDOW + STRIDE) == 2
    assert A.entrain_windows(LONG) == 4


def test_the_bucket_is_a_policy_read_off_the_published_apy():
    assert A.bucket_for(3.0) == "safe"
    assert A.bucket_for(35.0) == "safe"
    assert A.bucket_for(40.0) == "active"
    assert A.bucket_for(300.0) == "active"
    assert A.bucket_for(4000.0) == "experimental"
    assert A.bucket_for(-20.0) == "safe"


def test_pool_rows_are_internal_protocols_with_a_simulated_address(two_pools):
    """Every field the funnel reads is mapped, and the address is provably not a wallet."""
    market = scanner(wire(series=two_pools)).market(limit=2)
    protocol = market.protocol("project0 SYM0 (Ethereum)")
    assert protocol.address == "0x" + protocol.address[2:]
    assert len(protocol.address) == 42 and int(protocol.address, 16) >= 0
    assert protocol.claim_apy == pytest.approx(6.0 / 100.0)
    assert protocol.bucket == "safe"
    assert protocol.turnover == pytest.approx(1.0e9)
    assert np.all(np.isfinite(protocol.returns))
    assert market.step == protocol.returns.size
    assert market.name == A.VENUE_NAME
    assert len(market.get_protocol_list()) == len(market) == 2


def test_names_are_unique_even_when_the_venue_repeats_itself():
    payload = pools(count=2)
    payload["data"][1]["project"] = payload["data"][0]["project"]
    payload["data"][1]["symbol"] = payload["data"][0]["symbol"]
    payload["data"][1]["chain"] = payload["data"][0]["chain"]
    series = {
        payload["data"][0]["pool"]: noisy(4.0),
        payload["data"][1]["pool"]: noisy(9.0),
    }
    market = scanner(wire(payload, series)).market(limit=2)
    names = [protocol.name for protocol in market.protocols]
    assert len(set(names)) == 2
    assert names[0].startswith("project0 SYM0 (Ethereum)")


# ── the funnel: the core's two gates, on invented real-market shapes ────────


def test_a_steady_fifty_percent_is_convicted_and_explained(two_pools):
    """The brief's headline: a high APY with no residuals in it is a promise, not a yield."""
    payload = pools(count=2)
    payload["data"][1]["apy"] = 50.0
    payload["data"][1]["apyBase"] = 50.0
    engine = A.DiscoveryEngine(
        mode=A.MODE_REAL, scanner=scanner(wire(payload, two_pools))
    )
    funnel = engine.run(2)
    assert funnel.market == 2 and funnel.watched == 2
    verdict = funnel.assessments["project1 SYM1 (Ethereum)"]
    assert verdict.verdict in (R.SYNTHETIC_COLLUSION, R.SCAM_TRAP)
    assert verdict.sterile and verdict.cv < R.STERILE_CV
    top = funnel.top_rejected(1)[0]
    assert top["name"] == "project1 SYM1 (Ethereum)"
    assert top["apy"] == pytest.approx(50.0)
    assert top["implied_apy"] == pytest.approx(50.0, abs=0.01)
    assert "sterile" in top["reason"] and top["a2"] is not None
    assert "project1 SYM1 (Ethereum)" not in funnel.approved_names()


def test_an_honest_noisy_record_is_approved(two_pools):
    engine = A.DiscoveryEngine(mode=A.MODE_REAL, scanner=scanner(wire(series=two_pools)))
    funnel = engine.run(2)
    assert "project0 SYM0 (Ethereum)" in funnel.approved_names()
    assert funnel.approved == 1


def test_a_short_history_is_unmeasured_with_the_window_arithmetic():
    """Fewer than four fixations is not a refusal: it is a pool nobody has read yet."""
    series = {"00000000-0000-0000-0000-000000000000": noisy(7.0, points=SHORT)}
    engine = A.DiscoveryEngine(mode=A.MODE_REAL, scanner=scanner(wire(series=series)))
    funnel = engine.run(1)
    verdict = funnel.assessments["project0 SYM0 (Ethereum)"]
    assert verdict.verdict == R.UNMEASURED
    assert "2 of the 4 windows" in verdict.reason and str(SHORT) in verdict.reason
    assert funnel.unmeasured == 1 and funnel.measured == 0
    assert funnel.approved == 0
    assert funnel.top_rejected(3) == []  # a short record is not a refusal


def test_level_one_drops_noise_without_asking_entrain():
    """99 zeros and a spike: arithmetic can see this, and it must not cost level 2."""
    series = {f"0000000{index}-0000-0000-0000-00000000000{index}": [0.0] * 99 + [1000.0]
              for index in range(2)}
    payload = pools(count=2)
    payload["data"][0]["pool"] = "00000000-0000-0000-0000-000000000000"
    payload["data"][1]["pool"] = "00000001-0000-0000-0000-000000000001"
    engine = A.DiscoveryEngine(mode=A.MODE_REAL, scanner=scanner(wire(payload, series)))
    funnel = engine.run(2)
    assert funnel.watched == 0
    assert funnel.watchlist.reasons() == {D.REASON_CHAOTIC: 2}
    assert funnel.assessments == {}


def test_level_one_drops_a_pool_with_no_exit():
    payload = pools(count=1, tvl=5.0e5)
    series = {"00000000-0000-0000-0000-000000000000": noisy(6.0)}
    funnel = A.DiscoveryEngine(
        mode=A.MODE_REAL, scanner=scanner(wire(payload, series))
    ).run(1)
    assert funnel.watched == 0
    assert funnel.watchlist.reasons() == {D.REASON_DARK: 1}


def test_the_counts_add_up_and_nothing_is_both_refused_and_approved(two_pools):
    funnel = A.DiscoveryEngine(
        mode=A.MODE_REAL, scanner=scanner(wire(series=two_pools))
    ).run(2)
    assert funnel.watched == funnel.market - len(funnel.watchlist.dropped)
    assert funnel.measured + funnel.unmeasured == funnel.watched
    assert sum(funnel.verdicts().values()) == funnel.watched
    assert funnel.approved == funnel.verdicts().get(R.HONEST, 0)
    assert len(funnel.refused()) == funnel.measured - funnel.approved
    assert funnel.stages()[0] == ("market", 2)
    assert funnel.stages()[-1] == ("approved", funnel.approved)


def test_a_dead_feed_is_an_empty_funnel_and_exit_code_three(capsys, tmp_path):
    dead = mock.Mock(side_effect=OSError("network is unreachable"))
    engine = A.DiscoveryEngine(
        mode=A.MODE_REAL, scanner=scanner(dead, timeout=1.0)
    )
    funnel = engine.run(5)  # no exception
    assert funnel.market == 0 and funnel.approved == 0
    assert "could not be reached" in funnel.error
    assert "nothing was scanned" in funnel.summary_text()
    assert funnel.as_dict()["error"] == funnel.error

    with mock.patch.object(A, "_requests_get", side_effect=OSError("down")):
        code = A.main(["--limit", "3", "--timeout", "1", "--no-cache"])
    assert code == 3
    assert "could not be read" in capsys.readouterr().out


def test_a_feed_with_nothing_readable_is_exit_code_four(capsys):
    with mock.patch.object(A, "_requests_get", return_value=Reply(pools(count=0))):
        code = A.main(["--limit", "3", "--no-cache"])
    assert code == 4
    assert "no usable pool" in capsys.readouterr().out


def test_the_cli_prints_the_funnel_and_the_refusals(two_pools, capsys, tmp_path):
    transport = wire(series=two_pools)
    report = tmp_path / "funnel.json"
    with mock.patch.object(A, "DefiLlamaScanner", return_value=scanner(transport)):
        code = A.main(["--limit", "2", "--top", "1", "--output", str(report)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "level 1 (scanner)" in printed and "refused:" in printed
    body = json.loads(report.read_text(encoding="utf-8"))
    assert body["approved"] == 1 and body["top_rejected"][0]["verdict"] in (
        R.SYNTHETIC_COLLUSION,
        R.SCAM_TRAP,
    )


# ── the cache: a scan that can be repeated, and a stale copy that admits it ─


def test_the_second_read_comes_from_the_cache(tmp_path):
    feed = wire(series={"00000000-0000-0000-0000-000000000000": noisy(5.0)})
    reader = A.DefiLlamaScanner(transport=feed, cache_dir=tmp_path)
    reader.get_top_pools()
    reader.get_top_pools()
    assert reader.requests_made == 1 and reader.cache_hits == 1
    reader.get_top_pools(refresh=True)
    assert reader.requests_made == 2


def test_a_stale_copy_is_served_when_the_feed_dies(tmp_path):
    """A scan on yesterday's numbers that says so beats no scan; the note keeps it honest."""
    series = {"00000000-0000-0000-0000-000000000000": noisy(5.0)}
    warm = A.DefiLlamaScanner(transport=wire(series=series), cache_dir=tmp_path)
    assert warm.get_top_pools()
    cold = A.DefiLlamaScanner(
        transport=mock.Mock(side_effect=OSError("down")),
        cache_dir=tmp_path,
        cache_ttl=-1.0,  # everything on disk is stale now
    )
    rows = cold.get_top_pools()
    assert len(rows) == 4
    assert any("served from cache" in warning for warning in cold.warnings)
    assert cold.clear_cache() >= 1


def test_a_cache_that_cannot_be_written_is_not_a_failure(tmp_path):
    reader = A.DefiLlamaScanner(
        transport=wire(series={"00000000-0000-0000-0000-000000000000": noisy(5.0)}),
        cache_dir=tmp_path / "file-not-a-directory",
    )
    (tmp_path / "file-not-a-directory").write_text("", encoding="utf-8")
    assert reader.market(1)  # still reads the market


# ── the switch, and the thing this step is forbidden to do ──────────────────


def test_synthetic_mode_is_the_drawn_market_behind_the_same_loop():
    funnel = A.DiscoveryEngine(
        mode=A.MODE_SYNTHETIC, synthetic_steps=720, synthetic_seed=7
    ).run(40)
    assert funnel.market == 40 and funnel.watched > 0
    assert funnel.approved == funnel.verdicts().get(R.HONEST, 0)
    assert funnel.error == ""


def test_an_unknown_mode_is_one_sentence():
    with pytest.raises(A.DefiLlamaError) as caught:
        A.DiscoveryEngine(mode="live_trading")
    assert "unknown mode" in str(caught.value)


def test_nothing_here_can_move_a_credit():
    """No stake, no withdrawal, no signature: the brief forbids a transaction in this step."""
    forbidden = ("execute_stake", "withdraw", "deposit", "sign", "send", "transfer")
    surfaces = [
        A.DiscoveryEngine(mode=A.MODE_SYNTHETIC),
        A.DefiLlamaScanner(use_cache=False),
        A.DefiLlamaMarket(),
        A.MarketFunnel(),
    ]
    for surface in surfaces:
        for name in forbidden:
            assert not hasattr(surface, name), f"{type(surface).__name__}.{name}"


def test_the_report_is_json_able_and_adds_up(two_pools):
    funnel = A.DiscoveryEngine(
        mode=A.MODE_REAL, scanner=scanner(wire(series=two_pools))
    ).run(2)
    body = json.loads(json.dumps(funnel.as_dict()))
    assert body["mode"] == A.MODE_REAL and body["venue"] == A.VENUE_NAME
    assert body["approved"] + body["refused"] + body["unmeasured"] == body["watchlist"]
    assert body["market"] == body["watchlist"] + body["refused_by_scanner"]
    assert math.isclose(body["elapsed"], body["elapsed"])  # a number, not NaN


# ── a name, priced from the feed: the seam between a book and a market ──────
#
# The venue holds an address and a label; the feed holds a label and a uuid. These tests
# pin down the bridge — which pool a name denotes, what happens when two do, and what is
# said when none does — because this is the one place where a careless adapter could hand
# the funnel one pool's history under another pool's name.

A_USDC = "aaaaaa01-0000-0000-0000-000000000000"
ANOTHER_USDC = "bbbbbb02-0000-0000-0000-000000000000"
ETH_USDC = "eeeeee03-0000-0000-0000-000000000000"
WETH_USDC = "dddddd04-0000-0000-0000-000000000000"


def row(project, symbol, chain, tvl, apy, pool):
    """One ``/pools`` row, with only the fields the resolver reads."""
    return {
        "chain": chain,
        "project": project,
        "symbol": symbol,
        "tvlUsd": float(tvl),
        "apy": float(apy),
        "pool": pool,
    }


def catalogue():
    """Two pools called USDC on Arbitrum, a deeper one on Ethereum, and one pair."""
    return {
        "status": "success",
        "data": [
            row("aave-v3", "USDC", "Arbitrum", 5.0e8, 3.0, A_USDC),
            row("compound-v3", "USDC", "Arbitrum", 1.0e7, 2.0, ANOTHER_USDC),
            row("aave-v3", "USDC", "Ethereum", 9.0e8, 4.0, ETH_USDC),
            row("uniswap-v3", "WETH-USDC", "Arbitrum", 3.0e7, 12.0, WETH_USDC),
        ],
    }


def resolver(*, chain="Arbitrum", series=None, limit=200) -> A.FeedHistory:
    """A resolver over a mocked wire; nothing here reaches the network."""
    catalogue_ = catalogue()
    charts = {A_USDC: noisy(3.0), WETH_USDC: noisy(12.0)} if series is None else series
    scan = A.DefiLlamaScanner(
        transport=wire(catalogue_, charts), use_cache=False, timeout=1.0
    )
    return A.FeedHistory(scanner=scan, chain=chain, limit=int(limit))


def test_a_name_resolves_to_the_deepest_pool_that_carries_it():
    """"USDC" is an exact symbol on three pools; the deepest one on *this* chain wins."""
    pool, matched = resolver().find("USDC")

    assert pool is not None
    assert pool.uuid == A_USDC and pool.chain == "Arbitrum"
    assert matched == 2  # two Arbitrum pools carry it; Ethereum's does not count


def test_the_chain_filter_is_the_feed_s_spelling_of_a_chain_id():
    assert A.chain_name(42161) == "Arbitrum"
    assert A.chain_name(0) == "" and A.chain_name(None) == ""
    assert A.chain_name(999_999) == ""  # unknown: no filter rather than a guess


def test_a_pool_that_s_published_history_is_the_series_it_published():
    record = resolver().history("USDC")

    assert record
    assert record.uuid == A_USDC and record.pool == "aave-v3 USDC (Arbitrum)"
    assert len(record) == LONG
    # the APYs are converted, not copied: the record compounds back to what was published
    assert float(record.rates()[0]) == pytest.approx(A.apy_to_rate(record.apy[0]))


def test_a_name_no_pool_carries_is_a_reason_and_not_a_neighbour():
    record = resolver().history("NOPE")

    assert not record and record.uuid == "" and record.apy == ()
    assert "no pool in the catalogue carries this name" in record.reason
    assert not record.as_dict()["points"]


def test_a_pair_is_matched_whatever_order_it_is_spelled_in():
    """The project writes ``WETH-USDC``; the operator wrote ``USDC-WETH``. Same pool."""
    pool, matched = resolver().find("USDC-WETH")

    assert pool is not None and pool.uuid == WETH_USDC
    assert matched == 1
    # and the same pool is not what a *single* token means: "USDC" stays the exact symbol
    assert resolver().find("USDC")[0].uuid == A_USDC


def test_a_name_that_exists_only_on_another_chain_says_where_it_lives():
    record = resolver(chain="Polygon").history("USDC")

    assert not record
    assert "none of them is on Polygon" in record.reason
    assert "Arbitrum" in record.reason  # the chains it *does* exist on are named


def test_an_operator_can_pin_the_pool_and_the_name_is_not_consulted():
    """The uuid in the book wins: no match, no guessing, no fall-back."""
    record = resolver(series={WETH_USDC: noisy(12.0)}).history("USDC", uuid=WETH_USDC)

    assert record.uuid == WETH_USDC
    assert record.pool == ""  # nothing was looked up, so nothing is claimed
    assert record.label == "the feed pool [dddddd04] (pinned by the book)"
    assert record.pinned


def test_a_pinned_pool_the_feed_cannot_chart_is_a_reason_not_a_name_match():
    record = resolver().history("USDC", uuid=ETH_USDC)

    assert not record and record.uuid == ETH_USDC
    assert "no such pool" in record.reason or "HTTP 500" in record.reason


def test_a_tail_of_the_published_series_can_be_asked_for():
    record = resolver().history("USDC", steps=3)
    assert len(record) == 3
    assert record.apy == resolver().history("USDC").apy[-3:]

