"""The engine is only worth its output if the arithmetic can be redone with a pencil.

Every test here builds its funding history in memory, so nothing opens a socket and every
expected number is written out as a formula rather than pasted from a previous run of the code
under test. The four that matter most:

* a flat rate pays ``notional * rate * periods`` and the APY divides by *capital*, not notional;
* an unreported rate pays nothing and is counted — it is not a zero;
* a hole in the venue's history is neither bridged nor charged;
* the basis block never touches ``net_apy``, because a six-year clock and a one-year clock do
  not add up to a number.
"""
import json

import pytest

from agent.backtest import carry_engine as E
from agent.backtest.carry import FundingRecord, FundingSeries, write_funding
from agent.backtest.data import Candle, CandleSeries

STEP = 8 * 3600.0
START = 1_600_000_000.0


def series_of(rates, *, contract="TEST-USDT", start=START, step=STEP):
    """A funding series with one record per rate, on the cadence the rates were paid at."""
    records = []
    at = float(start)
    for rate in rates:
        records.append(FundingRecord(at=at, funding_rate=rate))
        at += float(step)
    return FundingSeries(contract=contract, period_seconds=float(step), records=records)


def flat_year(rate=0.0001, days=365):
    """One year of an 8-hour cadence: ``days * 3`` payments plus the record the book opens on."""
    return series_of([rate] * (int(days * 3) + 1))


def test_a_flat_rate_pays_the_notional_and_the_apy_divides_by_capital():
    run = E.simulate(flat_year(0.0001), settings=E.Settings(), turnover=1e9)
    book = run.settings
    assert run.years == pytest.approx(1.0)
    assert run.periods == 1095
    assert run.gross_usdt == pytest.approx(book.notional * 0.0001 * 1095)
    assert run.net_usdt == pytest.approx(run.gross_usdt - book.round_trip_usdt)
    assert run.net_apy == pytest.approx(run.net_usdt / book.capital / run.years)
    # On notional the same year reads 1.33x better, which is the error this test exists for:
    # $50 000 at 3x carries $37 500, and the funding is earned on that.
    assert run.net_usdt / book.notional > run.net_apy


def test_the_round_trip_is_charged_on_both_legs_twice():
    book = E.Settings()
    run = E.simulate(flat_year(0.0001), settings=book)
    leg = book.notional * (book.spot_fee + book.perp_fee + 2.0 * book.slippage)
    assert run.entry_cost_usdt == pytest.approx(leg)
    assert run.exit_cost_usdt == pytest.approx(leg)
    assert run.net_usdt == pytest.approx(run.gross_usdt - 2.0 * leg)
    free = E.Settings(spot_fee=0.0, perp_fee=0.0, slippage=0.0)
    unpaid = E.simulate(flat_year(0.0001), settings=free)
    assert unpaid.net_usdt == pytest.approx(unpaid.gross_usdt)


def test_the_record_the_book_opened_on_pays_it_nothing():
    """A pair that appears at the settlement instant cannot claim what that settlement paid."""
    run = E.simulate(series_of([0.5, 0.0001, 0.0001]))
    assert run.periods == 2
    assert run.paid_periods == 2
    assert run.gross_usdt == pytest.approx(run.settings.notional * 0.0002)


def test_an_unreported_rate_pays_nothing_and_is_counted_separately():
    run = E.simulate(series_of([0.0001, None, 0.0, None, 0.0001]))
    book = run.settings
    assert run.periods == 4
    # The literal zero is a payment of nothing and counts as reported; the two ``None`` do not.
    assert run.paid_periods == 2
    assert run.unreported_periods == 2
    assert run.reported_share == pytest.approx(0.5)
    assert run.gross_usdt == pytest.approx(book.notional * 0.0001)
    # The share is over the series' reported rates, the same number the coverage report prints.
    assert run.positive_share == pytest.approx(2.0 / 3.0)


def test_a_negative_stretch_is_the_drawdown_and_it_is_in_capital():
    rates = [0.0001] * 400 + [-0.005] * 3 + [0.0001] * 10
    run = E.simulate(series_of(rates))
    book = run.settings
    assert run.drawdown == pytest.approx(3 * 0.005 * book.notional / book.capital)
    assert run.drawdown > 0.0


def test_a_hole_in_the_history_is_neither_bridged_nor_charged():
    records = []
    at = START
    for index in range(1200):
        if index == 600:
            at += STEP  # one settlement the venue never published
        records.append(FundingRecord(at=at, funding_rate=0.0001))
        at += STEP
    series = FundingSeries(contract="TEST-USDT", period_seconds=STEP, records=records)
    run = E.simulate(series)
    assert run.gap_periods == 1
    assert run.gross_usdt == pytest.approx(run.settings.notional * 0.0001 * 1199)
    assert any("never published" in note for note in run.warnings)



def test_a_short_history_is_unmeasured_rather_than_failed():
    run = E.simulate(series_of([0.0001] * 50))
    assert run.net_apy is not None, "the number is computed, it is just not judged"
    assert run.verdict.overall == "unmeasured"
    states = {item.name: item.verdict for item in run.verdict.criteria}
    assert states["net_apy"] == E.UNMEASURED
    assert states["max_drawdown"] == E.UNMEASURED
    assert "held period(s)" in run.readability()


def test_a_rate_below_the_gate_fails_and_the_line_says_which():
    run = E.simulate(flat_year(rate=0.0000001), turnover=1e9)
    assert run.verdict.overall == "fail"
    assert run.verdict.missed == ["net_apy"]
    line = next(item for item in run.verdict.criteria if item.name == "net_apy")
    assert "12.00 %" in line.reason


def test_a_venue_that_published_nothing_half_the_time_is_unmeasured():
    rates = [None if index % 2 else 0.0001 for index in range(1096)]
    run = E.simulate(series_of(rates), turnover=1e9)
    assert run.reported_share == pytest.approx(0.5, abs=0.01)
    assert run.verdict.overall == "unmeasured"
    assert "published rate" in run.readability()


def test_capacity_is_deployable_capital_in_the_dollars_the_gate_names():
    book = E.Settings()
    rich = E.simulate(flat_year(), settings=book, turnover=683_959_205.0)
    assert rich.capacity_usdt == pytest.approx(683_959_205.0 * 0.01 * 4.0 / 3.0)
    rich_states = {item.name: item.verdict for item in rich.verdict.criteria}
    assert rich_states["capacity_usdt"] == E.MET
    thin = E.simulate(flat_year(), settings=book, turnover=1_000_000.0)
    thin_states = {item.name: item.verdict for item in thin.verdict.criteria}
    assert thin_states["capacity_usdt"] == E.MISSED
    unread = E.simulate(flat_year(), settings=book)
    line = next(item for item in unread.verdict.criteria if item.name == "capacity_usdt")
    assert line.verdict == E.UNMEASURED
    assert "no turnover was read" in line.reason


def test_a_book_larger_than_the_contract_can_carry_is_missed_even_above_the_floor():
    """$50k is the gate; a million-dollar book on a $50M day is a different, larger question."""
    big = E.Settings(capital=1_000_000.0)
    run = E.simulate(flat_year(), settings=big, turnover=50_000_000.0)
    line = next(item for item in run.verdict.criteria if item.name == "capacity_usdt")
    assert line.measured == pytest.approx(50_000_000.0 * 0.01 * 4.0 / 3.0)
    assert line.verdict == E.MISSED
    assert "this book actually deploys" in line.reason


def test_leverage_moves_the_notional_and_the_capital_used_stays_the_capital():
    low = E.Settings(leverage=1.0)
    high = E.Settings(leverage=3.0)
    assert low.notional == pytest.approx(low.capital / 2.0)
    assert high.notional == pytest.approx(high.capital * 0.75)
    assert low.capital_used == pytest.approx(low.capital)
    assert high.capital_used == pytest.approx(high.capital)
    slow = E.simulate(flat_year(), settings=low)
    fast = E.simulate(flat_year(), settings=high)
    assert fast.gross_usdt / slow.gross_usdt == pytest.approx(1.5)
    assert fast.net_apy / slow.net_apy == pytest.approx(1.5)


def test_a_book_above_the_declared_leverage_is_refused_before_it_is_priced():
    with pytest.raises(ValueError) as caught:
        E.Settings(leverage=5.0)
    assert "3x" in str(caught.value)
    with pytest.raises(ValueError):
        E.Settings(capital=0.0)
    with pytest.raises(ValueError):
        E.Settings(participation=0.5)
    with pytest.raises(ValueError):
        E.Settings(spot_fee=0.5)



def test_basis_marks_join_the_two_feeds_and_skip_a_bar_off_the_grid():
    perp = CandleSeries(
        symbol="TEST-USDT",
        rows=[
            Candle(at=0.0, open=100.0, high=100.0, low=100.0, close=100.0),
            Candle(at=14400.0, open=101.0, high=101.0, low=101.0, close=101.0),
            Candle(at=28800.0, open=102.0, high=102.0, low=102.0, close=102.0),
        ],
    )
    spot = CandleSeries(
        symbol="TESTUSDT",
        rows=[
            Candle(at=0.0, open=100.0, high=100.0, low=100.0, close=100.0),
            Candle(at=14400.0, open=100.0, high=100.0, low=100.0, close=100.0),
            Candle(at=29400.0, open=100.0, high=100.0, low=100.0, close=100.0),
        ],
    )
    marks = E.basis_marks(perp, spot)
    assert [at for at, _ in marks] == [0.0, 14400.0]
    assert marks[1][1] == pytest.approx(100.0)
    assert E.basis_marks(CandleSeries(), spot) == []



def test_the_basis_is_reported_beside_the_funding_and_never_inside_it():
    series = flat_year()
    plain = E.simulate(series, turnover=1e9)
    marks = [(series.records[0].at, 5.0), (series.records[-1].at, 25.0)]
    marked = E.simulate(series, turnover=1e9, marks=marks)
    assert marked.net_apy == pytest.approx(plain.net_apy)
    assert marked.net_usdt == pytest.approx(plain.net_usdt)
    assert marked.marks["basis_drift_bps"] == pytest.approx(20.0)
    assert marked.marks["basis_pnl_usdt"] == pytest.approx(-20.0 / 1e4 * marked.settings.notional)
    # Only the first held period and the last one sit within a period of a mark.
    assert marked.marks["covered_share"] == pytest.approx(2.0 / 1095.0)
    assert "different clock" in marked.marks["note"]



def test_a_single_mark_is_not_a_basis_path():
    series = flat_year()
    run = E.simulate(series, marks=[(series.records[0].at, 5.0)])
    assert run.marks["marked_periods"] == 1
    assert run.marks["basis_drift_bps"] is None
    assert run.marks["drawdown"] is None
    assert "fewer than two" in run.marks["note"]
    assert any("cover" in note for note in run.warnings)


def test_a_series_with_nothing_to_hold_is_unmeasured_and_says_why():
    run = E.simulate(series_of([0.0001]))
    assert run.periods == 0
    assert run.net_apy is None
    assert run.verdict.overall == "unmeasured"
    assert any("fewer than two funding periods" in note for note in run.warnings)



def test_the_gate_is_the_terms_the_operator_declared():
    assert E.CARRY_GATE == {
        "net_apy": 0.12,
        "max_drawdown": 0.03,
        "positive_share": 0.60,
        "capacity_usdt": 50_000.0,
        "max_leverage": 3.0,
    }
    criteria = E.criteria_of()
    assert E.verify(criteria.document()) == ""
    assert criteria.fingerprint() == E.criteria_of().fingerprint()



def test_the_report_is_hashed_and_the_verdict_travels_inside_it(tmp_path):
    run = E.simulate(flat_year(), turnover=1e9)
    written = E.write_carry(tmp_path / "engine.json", run)
    document = json.loads(written["json"].read_text(encoding="utf-8"))
    assert E.verify(document) == ""
    assert document["run"]["carry"]["verdict"]["overall"] == run.verdict.overall
    assert written["criteria"].is_file()
    disclaimers = " ".join(document["criteria"]["does_not_show"])
    assert "liquidation" in disclaimers
    assert "lower bound" in disclaimers
    document["criteria"]["gate"]["net_apy"] = 0.02
    assert "changed after the run" in E.verify(document)



def test_two_reports_over_the_same_series_are_the_same_bytes(tmp_path):
    run = E.simulate(flat_year(), turnover=1e9)
    first = E.write_carry(tmp_path / "a" / "engine.json", run)["json"].read_bytes()
    second = E.write_carry(tmp_path / "b" / "engine.json", run)["json"].read_bytes()
    assert first == second



def test_the_cli_answers_from_the_cache_without_opening_a_socket(tmp_path, monkeypatch):
    from agent.backtest import carry

    def refuse(*args, **kwargs):
        raise AssertionError("the engine opened a socket for a cache it already had")

    monkeypatch.setattr(carry, "_default_transport", refuse)
    series = flat_year()
    cache = tmp_path / "funding"
    write_funding(cache / series.contract, series)
    out = tmp_path / "engine.json"
    code = E.main(
        [
            "--contract",
            series.contract,
            "--cache",
            str(cache),
            "--out",
            str(out),
            "--turnover",
            "683959205",
            "--quiet",
        ]
    )
    assert code == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert E.verify(document) == ""
    carry_block = document["run"]["carry"]
    assert carry_block["money"]["net_apy"] == pytest.approx(
        (carry_block["money"]["net_usdt"] / 50_000.0) / carry_block["window"]["years"]
    )
    assert carry_block["verdict"]["overall"] == "fail"
    assert carry_block["verdict"]["missed"] == ["net_apy"]



def test_a_gate_without_a_line_is_refused_rather_than_defaulted():
    run = E.simulate(flat_year(), turnover=1e9)
    partial = {key: value for key, value in E.CARRY_GATE.items() if key != "max_drawdown"}
    with pytest.raises(ValueError) as caught:
        E.judge(run, partial)
    assert "max_drawdown" in str(caught.value)
    assert "never declared" in str(caught.value)

