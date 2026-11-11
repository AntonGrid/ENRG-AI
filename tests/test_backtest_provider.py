"""The simulated venue: the same operations, and every one of them measured.

A backtest is only as honest as its fills, so the fills are tested one assumption at a time
— the settlement lag, the slippage, the fee, the stop inside a bar, the gap through it, and
the refusals — on a series whose every bar is written out by hand.

What the file proves, and why each of these is worth a test of its own:

* **the lag.** A decision made on a close cannot be filled at that close. The first test
  watches a buy sit in ``orders`` for one bar and then fill at the *next* bar's open.
* **the costs.** 0.1 % of slippage and 0.1 % of fee per side, on both sides, so a strategy
  that only works before costs cannot look profitable here.
* **the stop, twice.** Inside the bar that crosses it (fill at the stop) and through a gap
  (fill at the open, which is worse) — the difference is the whole reason a gap is not
  smoothed away.
* **the refusals.** One position at a time, nothing to sell, nothing to spend: each is a
  ``False`` and a :class:`Refusal` with a sentence, never an exception and never a fill.
* **the interface.** The names the live provider answers are all here, because a loop that
  runs against one must run against the other.
"""
from __future__ import annotations

import pytest

from agent.backtest.data import Candle, CandleSeries
from agent.backtest.simulated_provider import SimulatedProvider

START = 1_600_000_000.0
PERIOD = 14400.0


def venue(*bars, **kwargs) -> SimulatedProvider:
    """A provider over ``(open, high, low, close, volume)`` bars, written out by hand."""
    rows = [
        Candle(
            at=START + index * PERIOD,
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
        )
        for index, (open_, high, low, close, volume) in enumerate(bars)
    ]
    series = CandleSeries(symbol="btcusdt", period="4hour", rows=rows, source="fixture")
    return SimulatedProvider(series, **kwargs)


def test_a_buy_is_filled_at_the_next_open_and_not_at_the_close_that_decided_it():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (110.0, 112.0, 109.0, 111.0, 1000.0),
        (111.0, 113.0, 110.0, 112.0, 1000.0),
    )

    assert provider.execute_stake("sim:btcusdt", 1000.0) is True
    # The order is taken, and nothing has happened yet: the bar it will settle in has not
    # opened, which is exactly the state a network delay leaves a live loop in.
    assert provider.busy is True
    assert provider.holding is False
    assert provider.fills == []

    provider.advance()

    assert provider.holding is True
    assert provider.fills[0].bar == 1
    assert provider.fills[0].price == pytest.approx(110.0 * 1.001)


def test_a_zero_latency_venue_fills_at_the_close_it_was_told_about():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (110.0, 112.0, 109.0, 111.0, 1000.0),
        latency_bars=0,
    )

    assert provider.execute_stake("sim:btcusdt", 1000.0) is True

    assert provider.holding is True
    assert provider.fills[0].price == pytest.approx(100.0 * 1.001)


def test_the_ticket_carries_its_own_stop():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.5, 100.5, 1000.0),
    )

    provider.execute_stake("sim:btcusdt", 500.0, stop_pct=0.02)
    provider.advance()

    assert provider.stop_price == pytest.approx(provider.entry_price * 0.98)


def test_the_fee_and_the_slippage_are_charged_on_both_sides():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (120.0, 121.0, 119.0, 120.0, 1000.0),
    )

    provider.execute_stake("sim:btcusdt", 1000.0)
    provider.advance()
    units = provider.units
    provider.withdraw("sim:btcusdt", units)
    provider.advance()

    bought = provider.fills[0]
    sold = provider.fills[1]
    assert bought.price == pytest.approx(100.0 * 1.001)
    assert bought.units == pytest.approx((1000.0 * (1 - 0.001)) / bought.price)
    assert sold.price == pytest.approx(120.0 * 0.999)
    assert sold.usd == pytest.approx(units * sold.price * (1 - 0.001))
    assert provider.cash == pytest.approx(sold.usd)
    assert provider.units == 0.0
    assert bought.fee == pytest.approx(1000.0 * 0.001)


def test_a_stop_fills_inside_the_bar_that_crosses_it():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 100.5, 90.0, 92.0, 1000.0),
        (92.0, 93.0, 91.0, 92.5, 1000.0),
    )

    provider.execute_stake("sim:btcusdt", 1000.0, stop_pct=0.02)
    provider.advance()

    stop = provider.fills[0].price * 0.98
    assert len(provider.fills) == 2
    assert provider.fills[1].side == "sell"
    assert provider.fills[1].reason == "stop"
    # The stop is a resting order, so it is paid inside the bar rather than at its close —
    # with the slippage against the seller, which is what a submitted stop costs.
    assert provider.fills[1].price == pytest.approx(stop * (1 - 0.001))
    assert provider.holding is False
    assert provider.stop_price == 0.0


def test_a_gap_through_the_stop_fills_at_the_open_and_not_at_the_stop():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.5, 100.0, 1000.0),
        (90.0, 91.0, 88.0, 89.0, 1000.0),
    )

    provider.execute_stake("sim:btcusdt", 1000.0, stop_pct=0.02)
    provider.advance()
    stop = provider.stop_price
    provider.advance()

    sold = provider.fills[1]
    assert sold.price == pytest.approx(90.0 * 0.999)
    # The bar opened below the stop: the order filled at the open, which is worse than the
    # price the position named. A backtest that filled this at `stop` would invent 8 %.
    assert sold.price < stop
    assert stop == pytest.approx(100.0 * 1.001 * 0.98)


def test_the_venue_holds_one_position_at_a_time_and_says_why():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
    )

    assert provider.execute_stake("sim:btcusdt", 500.0) is True
    assert provider.execute_stake("sim:btcusdt", 500.0) is False
    provider.advance()
    assert provider.execute_stake("sim:btcusdt", 500.0) is False

    assert len(provider.refusals) == 2
    assert "waiting" in provider.refusals[0].reason
    assert "one position" in provider.refusals[1].reason
    assert provider.refusals[1].usd == 500.0


def test_selling_more_than_is_held_sells_what_is_held():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
    )

    provider.execute_stake("sim:btcusdt", 1000.0)
    provider.advance()
    held = provider.units
    provider.withdraw("sim:btcusdt", held * 3.0)
    provider.advance()

    assert provider.fills[1].units == pytest.approx(held)
    assert provider.units == 0.0


def test_a_withdrawal_with_no_position_is_refused_and_not_signed():
    provider = venue((100.0, 101.0, 99.0, 100.0, 1000.0))

    assert provider.withdraw("sim:btcusdt", 1.0) is False

    assert provider.refusals[0].side == "sell"
    assert "no position" in provider.refusals[0].reason
    assert provider.fills == []


def test_a_buy_with_nothing_to_spend_is_refused_with_a_sentence():
    provider = venue((100.0, 101.0, 99.0, 100.0, 1000.0), cash=0.0)

    assert provider.execute_stake("sim:btcusdt", 100.0) is False

    assert "nothing to buy" in provider.refusals[0].reason


def test_the_balance_reads_cash_position_and_equity_at_the_close():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 140.0, 99.0, 130.0, 1000.0),
    )
    provider.execute_stake("sim:btcusdt", 1000.0)
    provider.advance()

    balance = provider.balance()

    assert balance.ok is True
    assert balance.cash == pytest.approx(0.0)
    assert balance.price == 130.0
    assert balance.equity == pytest.approx(provider.units * 130.0)
    assert balance.as_dict()["bar"] == 1


def test_the_clock_only_moves_forward_and_stops_at_the_last_bar():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 101.0, 99.0, 100.0, 1000.0),
    )

    assert provider.step == 0
    assert provider.advance(10) == 2
    assert provider.step == 2
    assert provider.to_bar(0) == 2  # the clock does not rewind


def test_the_provider_answers_every_operation_the_live_one_does():
    provider = venue((100.0, 101.0, 99.0, 100.0, 1000.0))

    for name in (
        "name",
        "step",
        "whitelist",
        "get_protocol_list",
        "fetch_protocol_history",
        "submit_validation_proof",
        "execute_stake",
        "withdraw",
        "balance",
        "advance",
    ):
        assert hasattr(provider, name), name

    assert provider.whitelist.as_list() == []
    assert provider.submit_validation_proof("sim:btcusdt", {"verdict": "PANIC"}) is False
    assert provider.get_protocol_list()[0]["address"] == "sim:btcusdt"


def test_the_history_it_publishes_stops_at_its_own_clock():
    provider = venue(
        (100.0, 101.0, 99.0, 100.0, 1000.0),
        (100.0, 99.0, 97.0, 98.0, 1000.0),
        (98.0, 97.0, 95.0, 96.0, 1000.0),
        (96.0, 95.0, 93.0, 94.0, 1000.0),
    )
    provider.to_bar(3)

    published = provider.fetch_protocol_history("sim:btcusdt")

    # Four bars, one of which has no previous close, so three rates — and never a fourth,
    # because the fourth bar has not happened.
    assert published == [
        pytest.approx(2.0),
        pytest.approx((1 - 96.0 / 98.0) * 100.0),
        pytest.approx((1 - 94.0 / 96.0) * 100.0),
    ]
