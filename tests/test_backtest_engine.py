"""The engine, on a chart whose every bar was written by hand.

The fixture is a quiet market of forty alternating bars, a fortnight of falls on falling
turnover, and a recovery. Two properties of it matter: the fall *is* a panic by the hunter's
own definition (the decline stands far above its rest while turnover drains), and the
recovery *is* a calm, so the same series exercises both the entry and the take-profit.

What this file proves, and each one is a way a backtest lies if it is wrong:

* **no lookahead.** The detector is wrapped and every prefix it is handed is recorded; the
  lengths must be ``1, 2, 3, …`` with no gap and no jump.
* **the lag.** The bar a panic is measured on is not the bar it is bought on.
* **the table.** ``PANIC`` buys, ``CALM`` takes profit only for a position that was bought
  on a panic, a conviction closes, and ``UNMEASURED`` does nothing — the live loop's table,
  compared against the live loop's own constants rather than restated.
* **the costs.** A stop is a loss of the stop plus the slippage plus two fees, and the
  metrics account for exactly that and for nothing else.
* **the interface.** The package has no path to a chain, a key or the live state file, and
  a test reads the source to say so.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.backtest import engine as E
from agent.backtest import reporter as R
from agent.backtest.data import Candle, CandleSeries
from agent.backtest.run import EXIT_FEED, EXIT_SHORT, EXIT_USAGE, main
from agent.physics.risk_assessment import HONEST, SUSPECT
from agent.physics.volatility_hunter import (
    CALM,
    DISHONEST,
    FORMULA,
    GROWTH,
    PANIC,
    THIN,
    UNMEASURED,
    PanicDetector,
)

START = 1_600_000_000.0
PERIOD = 14400.0


def chart(changes, volumes) -> CandleSeries:
    """A series from a list of per-bar returns and the turnover of each bar."""
    rows = []
    price = 100.0
    for index, (change, volume) in enumerate(zip(changes, volumes)):
        opening = price
        price = price * (1.0 + float(change))
        rows.append(
            Candle(
                at=START + index * PERIOD,
                open=opening,
                high=max(opening, price) * 1.0005,
                low=min(opening, price) * 0.9995,
                close=price,
                volume=float(volume),
            )
        )
    return CandleSeries(symbol="btcusdt", period="4hour", rows=rows, source="fixture")


#: The falls of the crash, in order. Chosen by hand so that the position survives the first
#: stop check and dies on the second — a fixture that stops out on the entry bar would test
#: the stop and nothing else.
CRASH = (
    -0.019,
    -0.017,
    -0.021,
    -0.018,
    -0.020,
    -0.019,
    -0.017,
    -0.021,
    -0.018,
    -0.020,
    -0.019,
    -0.017,
    -0.021,
    -0.018,
)


def panic_chart() -> CandleSeries:
    """A quiet market, a capitulation on draining turnover, and a recovery."""
    changes = [0.004 if index % 2 == 0 else -0.004 for index in range(40)]
    volumes = [1.0e6] * 40
    changes += list(CRASH)
    volumes += [4.0e5] * len(CRASH)
    changes += [0.012] * 12
    volumes += [7.0e5] * 12
    return chart(changes, volumes)


def signal(verdict: str, *, confidence: float = 0.5, resting: str = HONEST):
    """A stand-in for a :class:`PanicSignal`: the engine only reads these four fields."""
    return SimpleNamespace(
        verdict=verdict, confidence=confidence, resting_verdict=resting, reason=""
    )


def test_a_panic_is_bought_and_the_fill_waits_for_the_next_bar():
    result = E.Backtester(
        panic_chart(), E.Settings(min_points=20, log_decisions=True)
    ).run()

    first_panic = next(
        decision for decision in result.decisions if decision["verdict"] == PANIC
    )
    first_trade = result.trades[0]

    assert first_trade.entry_verdict == PANIC
    # The decision was made on the close of the panic bar; the fill is the next bar's open,
    # because the venue cannot sell you a price that has already happened.
    assert first_trade.entry_bar == first_panic["bar"] + 1
    assert result.verdicts[PANIC] >= 1


def test_no_panic_is_bought_below_the_confidence_floor():
    result = E.Backtester(panic_chart(), E.Settings(min_points=20, confidence=0.99)).run()

    # The panics are still measured — the floor is a floor, not a missing market.
    assert result.verdicts[PANIC] >= 1
    assert result.trades == []
    assert result.final_equity == pytest.approx(1000.0)


def test_a_panic_position_is_taken_off_on_the_calm_that_follows_it():
    result = E.Backtester(panic_chart(), E.Settings(min_points=20)).run()

    taken = [trade for trade in result.trades if trade.exit_reason == "take-profit"]

    assert taken, [trade.as_dict() for trade in result.trades]
    assert taken[0].entry_verdict == PANIC
    assert taken[0].exit_verdict == CALM
    assert taken[0].pnl > 0.0
    assert taken[0].bars > 0


def test_a_stop_carries_no_verdict_because_it_is_not_a_reading():
    result = E.Backtester(panic_chart(), E.Settings(min_points=20)).run()

    stopped = [trade for trade in result.trades if trade.exit_reason == "stop"]

    assert stopped, [trade.as_dict() for trade in result.trades]
    assert stopped[0].exit_verdict == ""
    assert stopped[0].pnl < 0.0
    # The stop, the slippage against the seller and two fees: about 2.3 %, never exactly 2 %.
    assert stopped[0].pnl_pct == pytest.approx(-0.0229, abs=0.002)


def test_the_market_is_not_re_entered_inside_the_cooldown_after_an_exit():
    result = E.Backtester(
        panic_chart(), E.Settings(min_points=20, cooldown_bars=100)
    ).run()

    assert len(result.trades) == 1


def test_a_vertically_too_short_series_is_refused_rather_than_measured():
    with pytest.raises(ValueError, match="at least three"):
        E.Backtester(chart([0.0, 0.0], [1.0, 1.0]), E.Settings()).run()


# ── the table, which is the live loop's table ───────────────────────────────


def test_the_verdict_sets_are_the_live_loops_sets():
    from agent.trade import loop as live

    assert E.EXIT_VERDICTS == live.EXIT_VERDICTS
    assert E.TAKE_PROFIT_VERDICTS == live.TAKE_PROFIT_VERDICTS
    assert PANIC in E.ENTRY_VERDICTS and PANIC in live.ENTRY_VERDICTS
    # Blindness is not a verdict: no set in either file may act on UNMEASURED.
    for table in (E.ENTRY_VERDICTS, E.EXIT_VERDICTS, E.TAKE_PROFIT_VERDICTS):
        assert UNMEASURED not in table


def test_a_calm_takes_profit_only_from_a_position_bought_on_a_panic():
    backtester = E.Backtester(panic_chart(), E.Settings(min_points=20))

    assert backtester.exit_reason(signal(CALM), PANIC) == "take-profit"
    assert backtester.exit_reason(signal(GROWTH), PANIC) == "take-profit"
    # A position bought for carry has no dislocation to have passed, so a calm bar is not
    # a reason to sell it — the live loop's own argument, kept here.
    assert backtester.exit_reason(signal(CALM), CALM) is None
    assert backtester.exit_reason(signal(CALM), GROWTH) is None


def test_a_conviction_closes_and_a_blind_bar_does_not():
    backtester = E.Backtester(panic_chart(), E.Settings(min_points=20))

    for verdict in (DISHONEST, FORMULA, THIN):
        assert backtester.exit_reason(signal(verdict), PANIC) == "convicted"
    assert backtester.exit_reason(signal(UNMEASURED), PANIC) is None


def test_an_organic_entry_needs_an_honest_record_and_the_organic_mode():
    chase = E.Backtester(panic_chart(), E.Settings(min_points=20))
    carry = E.Backtester(panic_chart(), E.Settings(min_points=20, entries="organic"))
    strict = E.Backtester(panic_chart(), E.Settings(min_points=20, confidence=0.5))

    assert chase.wants_entry(signal(CALM, resting=HONEST)) is False
    assert carry.wants_entry(signal(CALM, resting=HONEST)) is True
    assert carry.wants_entry(signal(CALM, resting=SUSPECT)) is False
    assert carry.wants_entry(signal(PANIC, confidence=0.4)) is True
    assert strict.wants_entry(signal(PANIC, confidence=0.4)) is False


# ── the honesty that a refactor can break silently ─────────────────────────


class Watcher:
    """A detector that records the prefix it was asked about, and answers honestly.

    The engine is handed this instead of a :class:`PanicDetector`, which is the whole reason
    the detector is a constructor argument: the property under test is not a verdict, it is
    the *length of the history* every verdict was measured on.
    """

    def __init__(self, min_points: int = 20) -> None:
        self.inner = PanicDetector(min_points=min_points, min_depth=0.0)
        self.recent = self.inner.recent
        self.lengths = []

    def measure(self, apy, tvl=None, **kwargs):
        self.lengths.append(len(list(apy)))
        return self.inner.measure(apy, tvl, **kwargs)


def test_the_engine_never_reads_a_bar_it_has_not_reached():
    series = panic_chart()
    watcher = Watcher()

    E.Backtester(series, E.Settings(min_points=20), detector=watcher).run()

    # One measurement per bar, in order, each prefix ending on the bar being decided.
    assert watcher.lengths == list(range(1, len(series) + 1))


# ── the ledger, and the readings of it ─────────────────────────────────────


def test_the_report_agrees_with_the_ledger_it_was_built_from():
    result = E.Backtester(panic_chart(), E.Settings(min_points=20)).run()
    metrics = R.metrics_of(result)

    closed = result.closed
    assert metrics.trades == len(result.trades)
    assert metrics.closed == len(closed)
    assert metrics.wins + metrics.losses + metrics.breakeven == metrics.closed
    assert metrics.fees_paid == pytest.approx(sum(trade.fees for trade in result.trades))
    assert metrics.final_equity == pytest.approx(result.equity[-1]["equity"])
    assert metrics.roi == pytest.approx(metrics.final_equity / metrics.capital - 1.0)
    # Cash in the book plus every realized trade must account for the equity to the cent:
    # a position is open at the end or it is not, and either way the curve says so.
    if not result.open_trades:
        assert metrics.pnl == pytest.approx(sum(trade.pnl for trade in closed), abs=0.02)


def _curve(values, prices=None):
    prices = prices or [100.0] * len(values)
    return [
        {
            "bar": index,
            "at": START + index * PERIOD,
            "price": float(prices[index]),
            "cash": float(values[index]),
            "units": 0.0,
            "equity": float(values[index]),
            "exposure": index % 2 == 0,
            "verdict": CALM,
        }
        for index in range(len(values))
    ]


def _trade(entry_bar: int, exit_bar: int, paid: float, got: float) -> E.Trade:
    return E.Trade(
        entry_bar=entry_bar,
        entry_at=START + entry_bar * PERIOD,
        entry_price=100.0,
        entry_usd=paid,
        units=1.0,
        exit_bar=exit_bar,
        exit_at=START + exit_bar * PERIOD,
        exit_price=got / 1.0,
        exit_usd=got,
    )


def test_the_metrics_are_read_from_the_curve_and_the_trades():
    result = E.BacktestResult(
        settings=E.Settings(capital=1000.0),
        series=chart([0.0] * 4, [1.0] * 4),
        venue={},
        trades=[_trade(0, 3, 100.0, 200.0), _trade(1, 2, 100.0, 50.0), E.Trade(
            entry_bar=3, entry_at=START, entry_price=1.0, entry_usd=10.0, units=1.0
        )],
        equity=_curve([1000.0, 1100.0, 950.0, 1200.0], [100.0, 110.0, 90.0, 120.0]),
    )

    metrics = R.Report(result).metrics

    assert metrics.final_equity == 1200.0
    assert metrics.roi == pytest.approx(0.2)
    assert metrics.pnl == pytest.approx(200.0)
    assert metrics.buy_hold_roi == pytest.approx(0.2)
    # 1100 → 950 is the deepest fall of the book, in per cent of its own peak.
    assert metrics.max_drawdown == pytest.approx(150.0 / 1100.0)
    assert metrics.max_drawdown_at == START + 2 * PERIOD
    assert (metrics.closed, metrics.open_trades) == (2, 1)
    assert (metrics.wins, metrics.losses) == (1, 1)
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.profit_factor == pytest.approx(2.0)
    assert metrics.avg_hold_bars == pytest.approx(2.0)  # 3 bars and 1 bar
    assert metrics.avg_hold_seconds == pytest.approx(2.0 * PERIOD)
    assert metrics.exposure == pytest.approx(0.5)
    assert metrics.best_trade == 100.0 and metrics.worst_trade == -50.0


# ── the report, and the files it leaves behind ─────────────────────────────


def test_the_report_is_json_that_carries_its_own_reading():
    result = E.Backtester(panic_chart(), E.Settings(min_points=20)).run()

    document = json.loads(json.dumps(R.Report(result).as_dict(), allow_nan=False))

    assert document["reading"].startswith("yield = max(0, -bar return)")
    assert document["metrics"]["trades"] == len(result.trades)
    assert document["settings"]["min_points"] == 20
    assert document["series"]["bars"] == len(result.series)
    assert document["window"]["first_stamp"].endswith("Z")
    assert document["verdicts"]["PANIC"] >= 1


def test_the_report_writes_three_files_and_the_csv_is_the_ledger(tmp_path):
    result = E.Backtester(panic_chart(), E.Settings(min_points=20)).run()

    written = R.Report(result).write(tmp_path)

    assert set(written) == {"json", "trades", "equity"}
    trades_csv = written["trades"].read_text(encoding="utf-8").splitlines()
    equity_csv = written["equity"].read_text(encoding="utf-8").splitlines()
    assert len(trades_csv) == len(result.trades) + 1  # the header, then one row per trade
    assert trades_csv[0].startswith("entry_time,entry_bar,entry_price")
    assert len(equity_csv) == len(result.equity) + 1
    assert json.loads(written["json"].read_text(encoding="utf-8"))["metrics"]["closed"] == len(
        result.closed
    )


def test_a_semicolon_report_is_written_for_the_locales_that_want_one(tmp_path):
    result = E.Backtester(panic_chart(), E.Settings(min_points=20)).run()

    written = R.Report(result).write(tmp_path, delimiter=";")

    assert written["trades"].read_text(encoding="utf-8").splitlines()[0].count(";") > 5


# ── the command line, which is how this is actually used ───────────────────


def _write_csv(series: CandleSeries, path) -> None:
    lines = ["timestamp,open,high,low,close,volume"]
    for row in series:
        lines.append(
            f"{int(row.at)},{row.open},{row.high},{row.low},{row.close},{row.volume}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_cli_runs_a_chart_from_a_file_and_writes_the_report(tmp_path, capsys):
    source = tmp_path / "chart.csv"
    _write_csv(panic_chart(), source)
    out = tmp_path / "out"

    code = main(
        [
            "--exchange",
            "csv",
            "--csv",
            str(source),
            "--min-points",
            "20",
            "--capital",
            "1000",
            "--out",
            str(out),
        ]
    )

    printed = capsys.readouterr().out
    assert code == 0
    assert "ROI" in printed and "buy & hold" in printed
    assert (out / "backtest_report.json").exists()
    assert (out / "backtest_trades.csv").exists()
    assert (out / "backtest_equity.csv").exists()


def test_the_cli_refuses_a_window_the_detector_cannot_judge(tmp_path, capsys):
    source = tmp_path / "chart.csv"
    quiet = chart([0.004 if index % 2 else -0.004 for index in range(30)], [1.0e6] * 30)
    _write_csv(quiet, path=source)

    code = main(["--exchange", "csv", "--csv", str(source), "--out", str(tmp_path / "o")])

    assert code == EXIT_SHORT
    assert "not enough for a verdict" in capsys.readouterr().err


def test_the_cli_refuses_a_capital_that_cannot_mean_anything(tmp_path, capsys):
    source = tmp_path / "chart.csv"
    _write_csv(panic_chart(), source)

    code = main(
        [
            "--exchange",
            "csv",
            "--csv",
            str(source),
            "--min-points",
            "20",
            "--capital",
            "-5",
            "--out",
            str(tmp_path / "o"),
        ]
    )

    assert code == EXIT_USAGE
    assert "capital must be positive" in capsys.readouterr().err


def test_the_cli_answers_a_dead_feed_with_an_exit_code(monkeypatch, capsys):
    from agent.backtest import run as runner
    from agent.backtest.data import HtxError

    def dead(args):
        raise HtxError("the feed did not answer")

    monkeypatch.setattr(runner, "_series", dead)

    code = runner.main(["--out", "/tmp/never-written"])

    assert code == EXIT_FEED
    assert "the feed did not answer" in capsys.readouterr().err
    assert code != EXIT_SHORT


def test_the_package_entry_point_is_the_same_function():
    from agent.backtest import __main__ as entry
    from agent.backtest import run as runner

    assert entry.main is runner.main


# ── the wall: no chain, no key, no state file ──────────────────────────────


def _imports_of(path):
    """Only the imports a module makes *at module scope* — an indented one is not one."""
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("import ", "from "))
    ]


def test_the_package_cannot_reach_a_chain_a_key_or_the_live_state():
    import agent.backtest as package

    folder = Path(package.__file__).parent
    modules = sorted(folder.glob("*.py"))
    assert {path.name for path in modules} == {
        "__init__.py",
        "__main__.py",
        "data.py",
        "engine.py",
        "reporter.py",
        "run.py",
        "simulated_provider.py",
        "sweep.py",
    }
    for path in modules:
        imported = "\n".join(_imports_of(path))
        for forbidden in (
            "web3",
            "agent.trade",
            "real_testnet_provider",
            "private_key",
            "eth_send",
            "subprocess",
            "socket",
        ):
            assert forbidden not in imported, f"{path.name} imports {forbidden}"
        if path.name != "data.py":
            assert "urllib" not in imported, f"{path.name} can open a socket"


def test_the_only_socket_is_the_default_transport_and_it_is_not_imported_at_module_scope():
    import agent.backtest.data as data_module

    source = Path(data_module.__file__).read_text(encoding="utf-8")

    # `urllib` appears once, inside `_http_get`, so importing the package cannot open a
    # connection and passing `fetch=`/`--csv` keeps even a run offline.
    assert "urllib" not in "\n".join(_imports_of(Path(data_module.__file__)))
    assert source.count("from urllib.request import") == 1
    assert "def _http_get(" in source


# ── settings that cannot mean anything are refused with a sentence ─────────


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"capital": 0.0}, "capital must be positive"),
        ({"share": 1.5}, "share must be between"),
        ({"stop": 1.0}, "stop must be in"),
        ({"fee": 0.6}, "fee must be in"),
        ({"latency_bars": -1}, "latency_bars must not be negative"),
        ({"window": 0}, "window must be at least one"),
        ({"min_points": 1}, "min_points must be at least 2"),
        ({"cooldown_bars": -3}, "cooldown_bars must not be negative"),
        ({"entries": "moon"}, "entries must be one of"),
        ({"confidence": 2.0}, "confidence must be in"),
    ],
)
def test_a_setting_that_cannot_mean_anything_is_refused(overrides, message):
    with pytest.raises(ValueError, match=message):
        E.Settings(**overrides)


def test_the_settings_are_described_for_the_report():
    described = E.Settings(stop=0.03, entries="both").as_dict()

    assert described["stop"] == 0.03
    assert described["entries"] == "both"
    assert described["min_depth"] == 0.0
    assert described["min_points"] >= 2
