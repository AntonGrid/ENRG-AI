"""The sweep, on a universe that exists only in a dictionary.

A sweep is where a backtest stops being an anecdote, and it is also where a backtest is
easiest to lie with: rank the pairs by their returns and the table becomes a highlight reel;
drop the pairs that failed and the mean becomes a wish. This file pins the three habits that
make the difference:

* **the universe is the venue's, not the operator's.** :func:`agent.backtest.data.universe`
  ranks by 24-hour quote turnover, filters by quote, and breaks ties by name so that the
  same market always gives the same twenty in the same order;
* **a failure is a row.** A dead feed and a pair too young to judge keep their place in the
  table with the sentence that explains them — a twenty-pair run with one apology answers
  nineteen questions, and a sweep that silently drops what it could not measure is a sweep
  that reports its own successes as the market;
* **the numbers that matter are three.** ROI, buy-and-hold, and their difference. Sorting by
  ROI alone cannot tell "the thesis worked" from "the coin fell less than the strategy".

Everything here runs offline: the transport of both the ticker list and the candles is a
parameter, so no test in this file can open a socket even by accident.
"""
from __future__ import annotations

import ast
import csv
from pathlib import Path

import pytest

from agent.backtest import sweep as S
from agent.backtest.data import (
    HTX_TICKERS,
    Candle,
    CandleSeries,
    HtxError,
    Pair,
    universe,
)
from agent.backtest.engine import Settings

START = 1_600_000_000.0
PERIOD = 86400.0


def chart(bars: int = 20, *, change: float = 0.01) -> CandleSeries:
    """A quiet series: alternating small bars on constant turnover, so no panic is called."""
    rows = []
    price = 100.0
    for index in range(bars):
        step = change if index % 2 else -change
        opening = price
        price = price * (1.0 + step)
        rows.append(
            Candle(
                at=START + index * PERIOD,
                open=opening,
                high=max(opening, price) * 1.001,
                low=min(opening, price) * 0.999,
                close=price,
                volume=1.0e6,
            )
        )
    return CandleSeries(symbol="probe", period="1day", rows=rows, source="fixture")


def loader_for(charts) -> object:
    """A candle loader that answers from a dictionary — or raises what the dictionary says."""

    def load(symbol, *, period, days, timeout, endpoint):
        answer = charts[str(symbol)]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return load


def settings(**overrides) -> Settings:
    """Settings that will judge a small fixture: the core floor is too high for it."""
    values = {"capital": 1000.0, "min_points": 8}
    values.update(overrides)
    return Settings(**values)


def tickers(*rows) -> dict:
    """The shape of the venue's answer, so that each test writes only the interesting part."""
    return {"status": "ok", "data": list(rows)}


def ticker(symbol: str, turnover: float, price: float = 1.0) -> dict:
    return {"symbol": symbol, "vol": turnover, "close": price}


# ── which pairs are "the top twenty" ────────────────────────────────────────


def test_the_universe_is_ranked_by_turnover():
    payload = tickers(
        ticker("btcusdt", 900.0), ticker("ethusdt", 1200.0), ticker("solusdt", 300.0)
    )

    pairs = universe(fetch=lambda url, timeout: payload)

    assert [pair.symbol for pair in pairs] == ["ethusdt", "btcusdt", "solusdt"]
    assert pairs[0].turnover == 1200.0


def test_the_universe_keeps_only_the_quote_it_was_asked_for():
    payload = tickers(
        ticker("btcusdt", 900.0), ticker("ethbtc", 2000.0), ticker("btcusdc", 1500.0)
    )

    pairs = universe("usdt", fetch=lambda url, timeout: payload)

    assert [pair.symbol for pair in pairs] == ["btcusdt"]


def test_the_universe_honours_its_limit():
    payload = tickers(*[ticker(f"a{index}usdt", 100.0 * index) for index in range(1, 11)])

    pairs = universe("usdt", 3, fetch=lambda url, timeout: payload)

    assert len(pairs) == 3
    assert [pair.turnover for pair in pairs] == [1000.0, 900.0, 800.0]


def test_a_pair_that_publishes_only_base_volume_is_still_ranked_in_dollars():
    payload = tickers(
        {"symbol": "ethusdt", "amount": 2.0, "close": 100.0},
        {"symbol": "btcusdt", "amount": 1.0, "close": 150.0},
    )

    pairs = universe(fetch=lambda url, timeout: payload)

    assert [pair.symbol for pair in pairs] == ["ethusdt", "btcusdt"]
    assert pairs[0].turnover == pytest.approx(200.0)


def test_ties_are_broken_by_name_so_that_a_sweep_can_be_compared_with_itself():
    payload = tickers(ticker("zebusdt", 500.0), ticker("abcusdt", 500.0))

    pairs = universe(fetch=lambda url, timeout: payload)

    assert [pair.symbol for pair in pairs] == ["abcusdt", "zebusdt"]


def test_the_ticker_transport_is_the_tickers_endpoint_and_not_the_candle_one():
    seen = []

    def transport(url, timeout):
        seen.append(url)
        return tickers(ticker("btcusdt", 1.0))

    universe(fetch=transport)

    assert seen == [HTX_TICKERS]


def test_a_universe_of_zero_pairs_is_refused_before_the_venue_is_asked():
    calls = []

    def transport(url, timeout):
        calls.append(url)
        return tickers(ticker("btcusdt", 1.0))

    with pytest.raises(HtxError, match="not a universe"):
        universe(limit=0, fetch=transport)
    assert calls == []


def test_an_answer_with_no_pairs_is_a_sentence_and_not_an_empty_table():
    with pytest.raises(HtxError, match="no pairs"):
        universe(fetch=lambda url, timeout: tickers())


def test_a_venue_that_refuses_says_why():
    payload = {"status": "error", "err-msg": "invalid symbol"}

    with pytest.raises(HtxError, match="invalid symbol"):
        universe(fetch=lambda url, timeout: payload)


def test_an_answer_that_is_not_a_mapping_is_a_sentence():
    with pytest.raises(HtxError, match="where a list of pairs was expected"):
        universe(fetch=lambda url, timeout: ["btcusdt"])


def test_a_pair_payload_is_json_ready():
    pair = Pair(symbol="btcusdt", turnover=12.5, price=60000.0)

    assert pair.as_dict() == {"symbol": "btcusdt", "turnover": 12.5, "price": 60000.0}


# ── the sweep itself ────────────────────────────────────────────────────────


def test_every_pair_gets_a_run_and_a_directory(tmp_path):
    loader = loader_for({"btcusdt": chart(), "ethusdt": chart()})

    rows = S.sweep(
        ["btcusdt", "ethusdt"], settings(), out_root=tmp_path, loader=loader, period="1day"
    )

    assert [row.symbol for row in rows] == ["btcusdt", "ethusdt"]
    assert all(row.measured for row in rows)
    assert (tmp_path / "btcusdt" / "backtest_report.json").is_file()
    assert (tmp_path / "ethusdt" / "backtest_trades.csv").is_file()
    assert rows[0].bars == 20
    assert rows[0].first_stamp < rows[0].last_stamp


def test_a_dead_pair_is_a_row_and_not_an_exception(tmp_path):
    loader = loader_for(
        {
            "btcusdt": chart(),
            "ghostusdt": HtxError("the HTX endpoint did not answer (timeout)"),
            "ethusdt": chart(),
        }
    )

    rows = S.sweep(
        ["btcusdt", "ghostusdt", "ethusdt"],
        settings(),
        out_root=tmp_path,
        loader=loader,
        period="1day",
    )

    assert [row.measured for row in rows] == [True, False, True]
    assert "did not answer" in rows[1].error
    assert rows[1].roi == 0.0
    assert not (tmp_path / "ghostusdt").exists()


def test_a_pair_too_young_for_the_detector_says_so_in_its_row(tmp_path):
    loader = loader_for({"newusdt": chart(bars=5)})

    rows = S.sweep(
        ["newusdt"], settings(min_points=64), out_root=tmp_path, loader=loader, period="1day"
    )

    assert not rows[0].measured
    assert "not enough for a verdict" in rows[0].error
    assert "listed too recently" in rows[0].error


def test_the_edge_is_what_the_strategy_did_that_holding_did_not(tmp_path):
    rows = S.sweep(
        ["btcusdt"],
        settings(),
        out_root=tmp_path,
        loader=loader_for({"btcusdt": chart()}),
        period="1day",
    )

    row = rows[0]
    assert row.edge == pytest.approx(row.roi - row.buy_hold_roi)
    assert row.as_dict()["edge"] == pytest.approx(row.edge)


def test_each_row_is_announced_as_it_lands(tmp_path):
    seen = []

    rows = S.sweep(
        ["btcusdt", "ghostusdt", "ethusdt"],
        settings(),
        out_root=tmp_path,
        loader=loader_for(
            {
                "btcusdt": chart(),
                "ghostusdt": HtxError("no answer"),
                "ethusdt": chart(),
            }
        ),
        period="1day",
        on_row=lambda row: seen.append(row.symbol),
    )

    assert seen == [row.symbol for row in rows] == ["btcusdt", "ghostusdt", "ethusdt"]


def test_a_symbol_written_twice_in_the_list_is_measured_twice(tmp_path):
    rows = S.sweep(
        ["btcusdt", "BTCUSDT", " btcusdt "],
        settings(),
        out_root=tmp_path,
        loader=loader_for({"btcusdt": chart()}),
        period="1day",
    )

    assert [row.symbol for row in rows] == ["btcusdt"] * 3


def test_an_empty_request_measures_nothing_and_says_so():
    assert S.sweep([], settings()) == []
    assert S.format_table([]) == "no pairs were measured"


# ── the table ───────────────────────────────────────────────────────────────


def row(symbol: str, **values) -> S.SweepRow:
    """A finished row, so that the table can be tested without running a backtest."""
    defaults = dict(
        bars=2000,
        first_stamp="2021-05-22",
        last_stamp="2026-11-11",
        trades=4,
        roi=0.12,
        buy_hold_roi=-0.30,
        max_drawdown=0.05,
        win_rate=0.5,
        fees=9.0,
        out="/tmp/x",
    )
    defaults.update(values)
    return S.SweepRow(symbol=symbol, **defaults)


def test_the_table_has_a_header_and_one_line_per_pair():
    table = S.format_table([row("btcusdt"), row("ethusdt")])
    lines = table.splitlines()

    assert len(lines) == 4
    header = lines[0]
    for name in S.HEADERS:
        assert name in header
    assert set("─") == set(lines[1].replace(" ", ""))
    assert lines[2].startswith("btcusdt")
    assert "+12.00%" in lines[2]
    assert "-30.00%" in lines[2]
    assert "+42.00%" in lines[2]


def test_the_columns_are_as_wide_as_their_widest_cell():
    table = S.format_table([row("btcusdt"), row("averylongsymbolusdt")])
    body = table.splitlines()[2:]

    assert body[1].startswith("averylongsymbolusdt")
    assert body[0].index("2000") == body[1].index("2000")


def test_a_failed_pair_keeps_its_place_with_its_sentence():
    failed = S.SweepRow(symbol="ghostusdt", error="the feed did not answer")
    table = S.format_table([row("btcusdt"), failed])

    assert "ghostusdt" in table
    assert "the feed did not answer" in table
    assert "—" in table


def test_the_totals_count_who_beat_holding_and_where_the_middle_is():
    rows = [
        row("a", roi=0.10, buy_hold_roi=-0.10),
        row("b", roi=-0.05, buy_hold_roi=-0.10),
        row("c", roi=0.20, buy_hold_roi=0.50),
        S.SweepRow(symbol="d", error="no answer"),
    ]

    reading = S.totals(rows)

    assert reading["measured"] == 3
    assert reading["failed"] == 1
    assert reading["median_roi"] == pytest.approx(0.10)
    assert reading["mean_roi"] == pytest.approx(0.25 / 3)
    assert reading["beat_buy_hold"] == 2
    assert reading["trades"] == 12


def test_the_middle_of_an_even_table_is_the_mean_of_the_two_middles():
    rows = [row("a", roi=0.0), row("b", roi=0.10), row("c", roi=0.30), row("d", roi=0.40)]

    assert S.totals(rows)["median_roi"] == pytest.approx(0.20)


def test_totals_of_nothing_are_a_sentence_and_not_a_division_by_zero():
    reading = S.totals([S.SweepRow(symbol="x", error="no answer")])

    assert reading == {"measured": 0, "failed": 1}
    assert "no pair could be measured" in S.totals_text([S.SweepRow(symbol="x", error="y")])


def test_the_reading_line_names_the_edge_and_the_count():
    rows = [row("a", roi=0.10, buy_hold_roi=-0.10), row("b", roi=0.0, buy_hold_roi=0.50)]

    text = S.totals_text(rows)

    assert "2 pairs measured" in text
    assert "beat buy & hold 1/2" in text
    assert "mean edge" in text


# ── the command line ────────────────────────────────────────────────────────


def test_the_cli_runs_a_named_list_offline(monkeypatch, tmp_path, capsys):
    loader = loader_for({"btcusdt": chart(), "ethusdt": chart()})
    monkeypatch.setattr(S, "fetch_klines", loader)

    code = S.main(
        [
            "--symbols",
            "btcusdt,ethusdt",
            "--period",
            "1day",
            "--days",
            "20",
            "--min-points",
            "8",
            "--out",
            str(tmp_path),
        ]
    )

    printed = capsys.readouterr().out
    assert code == 0
    assert "2 pairs" in printed
    assert "btcusdt" in printed and "ethusdt" in printed
    assert "beat buy & hold" in printed
    assert (tmp_path / "sweep.csv").is_file()


def test_the_cli_reads_the_universe_from_the_venue(monkeypatch, tmp_path, capsys):
    asked = []

    def fake_universe(quote, limit, **kwargs):
        asked.append((quote, limit))
        return [Pair(symbol="btcusdt", turnover=1.0), Pair(symbol="ethusdt", turnover=2.0)]

    monkeypatch.setattr(S, "universe", fake_universe)
    monkeypatch.setattr(S, "fetch_klines", loader_for({"btcusdt": chart(), "ethusdt": chart()}))

    code = S.main(
        [
            "--top",
            "2",
            "--period",
            "1day",
            "--days",
            "20",
            "--min-points",
            "8",
            "--out",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert asked == [("usdt", 2)]
    assert "btcusdt" in capsys.readouterr().out
    assert (tmp_path / "btcusdt" / "backtest_report.json").is_file()


def test_the_cli_reports_a_dead_universe_with_exit_three(monkeypatch, capsys):
    def dead(quote, limit, **kwargs):
        raise HtxError("the HTX ticker endpoint answered nothing")

    monkeypatch.setattr(S, "universe", dead)

    code = S.main(["--top", "5"])

    assert code == 3
    assert "answered nothing" in capsys.readouterr().err


def test_the_cli_exits_four_when_every_pair_is_too_young(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(S, "fetch_klines", loader_for({"btcusdt": chart(bars=5)}))

    code = S.main(
        ["--symbols", "btcusdt", "--period", "1day", "--days", "5", "--out", str(tmp_path)]
    )

    assert code == 4
    assert "not one pair could be measured" in capsys.readouterr().err


def test_the_cli_exits_two_on_a_setting_that_cannot_mean_anything(tmp_path, capsys):
    code = S.main(["--symbols", "btcusdt", "--stop", "2.0", "--out", str(tmp_path)])

    assert code == 2
    assert "stop must be" in capsys.readouterr().err


def test_the_csv_keeps_fractions_where_the_table_keeps_percentages(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "fetch_klines", loader_for({"btcusdt": chart()}))

    S.main(
        [
            "--symbols",
            "btcusdt",
            "--period",
            "1day",
            "--days",
            "20",
            "--min-points",
            "8",
            "--out",
            str(tmp_path),
            "--quiet",
        ]
    )

    with (tmp_path / "sweep.csv").open(newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))

    assert len(written) == 1
    assert written[0]["symbol"] == "btcusdt"
    assert float(written[0]["roi"]) == pytest.approx(0.0, abs=1e-9)
    assert written[0]["error"] == ""
    assert written[0]["first"] == written[0]["last"] or written[0]["first"] < written[0]["last"]


def test_the_sweep_has_no_path_to_a_chain_a_key_or_the_live_state():
    """Read the imports rather than the prose: a docstring may say "socket" and mean nothing."""
    tree = ast.parse(Path(S.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            imported.update(alias.name for alias in node.names)

    for forbidden in ("urllib", "socket", "http", "web3", "subprocess", "eth_account"):
        assert forbidden not in imported
    assert "agent.trade" not in imported


def test_a_position_still_open_is_shown_beside_the_closed_ones():
    table = S.format_table([row("btcusdt", trades=3, open_trades=1)])

    assert "3+1" in table


def test_a_failed_sentence_never_widens_the_table():
    measured = row("btcusdt")
    failed = S.SweepRow(symbol="ghostusdt", error="x" * 200)
    table = S.format_table([measured, failed])
    body = [line for line in table.splitlines() if line.startswith("btcusdt")][0]

    assert len(body) < 100
    assert "! ghostusdt" in table


def test_the_median_edge_is_reported_beside_the_mean():
    rows = [row("a", roi=0.0, buy_hold_roi=10.0), row("b", roi=0.0, buy_hold_roi=0.0)]

    reading = S.totals(rows)

    assert reading["mean_edge"] == pytest.approx(-5.0)
    assert reading["median_edge"] == pytest.approx(-5.0)
