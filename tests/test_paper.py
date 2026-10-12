from datetime import UTC, datetime

import pytest

from bot.backtest.costs import CostModel
from bot.core.config import InstrumentConfig
from bot.core.models import OrderStatus, Position, Side, Signal
from bot.execution.broker import (
    DuplicateOrderError,
    OrderRejectedError,
    close_request,
    entry_request,
)
from bot.execution.paper import PaperBroker

T = datetime(2026, 10, 12, 11, 0, tzinfo=UTC)
COSTS = CostModel(taker_fee=0.00055, slippage={"BTCUSDT": 0.0003})
BTC = {"BTCUSDT": InstrumentConfig(tick_size=0.1, qty_step=0.001, min_qty=0.001)}


def make_signal(side: Side = Side.LONG, stop: float = 65000, tp: float | None = 66000) -> Signal:
    return Signal.create(
        strategy="breakout",
        strategy_version="1",
        symbol="BTCUSDT",
        timeframe="1h",
        bar_time=datetime(2026, 10, 12, 10, 0, tzinfo=UTC),
        side=side,
        entry_price=65450,
        stop_loss=stop,
        take_profit=tp,
    )


def broker() -> PaperBroker:
    b = PaperBroker(COSTS, BTC, 10_000, now=lambda: T)
    b.on_minute("BTCUSDT", 65400, 65460, 65390, 65450, T)
    return b


def test_market_entry_with_slippage_and_fee() -> None:
    b = broker()
    order = b.place_order(entry_request(make_signal(), 0.1))
    # 65450 × 1,0003 = 65469,635; комиссия 0,1 × 65469,635 × 0,00055 = 3,6008
    assert order.status is OrderStatus.FILLED
    assert order.avg_fill_price == pytest.approx(65469.635)
    assert b.cash == pytest.approx(10_000 - 3.6008, abs=1e-3)
    [pos] = b.get_positions()
    assert pos.stop_loss == 65000 and pos.take_profit == 66000
    assert pos.signal_id == make_signal().signal_id


def test_stop_hit_inside_candle_with_double_slippage() -> None:
    b = broker()
    b.place_order(entry_request(make_signal(), 0.1))
    info = b.on_minute("BTCUSDT", 65300, 65310, 64900, 64950, T)
    assert info is not None and info.reason == "stop"
    # стоп 65000 × (1 − 0,0006) = 64961; gross (64961 − 65469,635) × 0,1 = −50,8635
    # комиссии 3,6008 + 0,1 × 64961 × 0,00055 = 3,6008 + 3,5729 = 7,1737
    assert info.exit_price == pytest.approx(64961)
    assert info.net_pnl == pytest.approx(-50.8635 - 7.1737, abs=1e-3)
    assert b.get_positions() == []
    assert b.cash == pytest.approx(10_000 + info.net_pnl, abs=1e-6)


def test_gap_fills_at_open_and_stop_wins_tie() -> None:
    b = broker()
    b.place_order(entry_request(make_signal(), 0.1))
    info = b.on_minute("BTCUSDT", 64800, 66500, 64700, 66000, T)
    assert info is not None and info.reason == "stop"
    assert info.exit_price == pytest.approx(64800 * (1 - 0.0006))


def test_take_profit_and_reduce_only_close() -> None:
    b = broker()
    b.place_order(entry_request(make_signal(), 0.1))
    info = b.on_minute("BTCUSDT", 65800, 66100, 65790, 66050, T)
    assert info is not None and info.reason == "take"
    with pytest.raises(OrderRejectedError, match="нет позиции"):
        b.place_order(close_request("BTCUSDT", Side.LONG, 0.1, "x"))


def test_manual_close_and_trailing_stop() -> None:
    b = broker()
    sig = make_signal(tp=None)
    b.place_order(entry_request(sig, 0.1))
    b.set_trading_stop("BTCUSDT", Side.LONG, 65400)
    assert b.on_minute("BTCUSDT", 65450, 65500, 65420, 65480, T) is None
    order = b.place_order(close_request("BTCUSDT", Side.LONG, 0.1, sig.signal_id))
    assert order.reduce_only and b.get_positions() == []
    info = b.last_exit("BTCUSDT")
    assert info is not None and info.reason == "manual"


def test_rejections() -> None:
    b = broker()
    b.place_order(entry_request(make_signal(), 0.1))
    with pytest.raises(DuplicateOrderError):
        b.place_order(entry_request(make_signal(), 0.1))
    with pytest.raises(OrderRejectedError, match="меньше минимального"):
        PaperBroker(COSTS, BTC, 10_000).place_order(entry_request(make_signal(), 0.0001))


def test_funding_short_receives_on_positive_rate() -> None:
    b = broker()
    b.place_order(entry_request(make_signal(Side.SHORT, stop=66000, tp=65000), 0.1))
    # 0,1 × 65450 × 0,0001 = 0,6545 получает шорт
    assert b.apply_funding("BTCUSDT", 0.0001) == pytest.approx(0.6545)


def test_restore_after_restart() -> None:
    b = PaperBroker(COSTS, BTC, 10_000, now=lambda: T)
    b.restore(
        9_990,
        [Position(symbol="BTCUSDT", side=Side.LONG, size=0.1, entry_price=65000, stop_loss=64000)],
    )
    assert b.get_balance().wallet_balance == 9_990
    assert b.get_positions()[0].stop_loss == 64000
