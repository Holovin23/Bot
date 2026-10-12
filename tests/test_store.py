from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from bot.core.db import connect
from bot.core.models import Order, OrderStatus, Side, Signal
from bot.execution.broker import entry_request
from bot.execution.store import Store
from bot.risk.limits import RiskState

NOW = datetime(2026, 10, 12, 11, 0, tzinfo=UTC)


def make_signal() -> Signal:
    return Signal.create(
        strategy="breakout",
        strategy_version="1",
        symbol="BTCUSDT",
        timeframe="1h",
        bar_time=datetime(2026, 10, 12, 10, 0, tzinfo=UTC),
        side=Side.LONG,
        entry_price=65450,
        stop_loss=65000,
    )


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(connect(tmp_path / "bot.sqlite"), mode="paper")


def test_trade_lifecycle(store: Store) -> None:
    sig = make_signal()
    trade = store.add_trade(sig, qty=0.1, risk_amount=48.9)
    assert trade.status == "pending" and trade.signal == sig
    store.update_trade(sig.signal_id, status="open", entry_price=65460.0)
    assert [t.signal_id for t in store.active_trades()] == [sig.signal_id]
    store.update_trade(sig.signal_id, status="closed", net_pnl=-50.0, exit_reason="stop")
    assert store.active_trades() == []
    assert store.trades("closed")[0].net_pnl == -50.0
    with pytest.raises(ValueError, match="неизвестные"):
        store.update_trade(sig.signal_id, mode="demo")


def test_other_mode_is_separate(store: Store) -> None:
    store.add_trade(make_signal(), 0.1, 48.9)
    other = Store(store.conn, mode="demo")
    assert other.trades() == []


def test_order_state_machine(store: Store) -> None:
    request = entry_request(make_signal(), 0.1)
    store.order_sent(request)
    assert store.order_status(request.order_link_id) == "SENT"
    assert store.unconfirmed_orders() == [(request.order_link_id, "BTCUSDT")]
    store.order_result(
        Order(
            order_link_id=request.order_link_id,
            symbol="BTCUSDT",
            side=request.side,
            order_type=request.order_type,
            qty=0.1,
            status=OrderStatus.FILLED,
            filled_qty=0.1,
            avg_fill_price=65460,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    assert store.order_status(request.order_link_id) == "FILLED"
    assert store.unconfirmed_orders() == []
    store.order_failed(request.order_link_id, "rejected")
    assert store.order_status(request.order_link_id) == "REJECTED"


def test_risk_state_roundtrip_and_kill(store: Store) -> None:
    assert store.get_risk_state() is None
    st = RiskState(10_000, 10_500, date(2026, 10, 12), 10_100, loss_streak=2, paused="x")
    store.set_risk_state(st)
    assert store.get_risk_state() == st
    assert store.kill_reason() is None
    store.request_kill("кнопка в интерфейсе")
    assert store.kill_reason() == "кнопка в интерфейсе"
    store.clear_kill()
    assert store.kill_reason() is None


def test_events_newest_first(store: Store) -> None:
    store.add_event("start", "бот запущен")
    store.add_event("order", "ордер отклонён", level="error")
    events = store.events()
    assert [e["kind"] for e in events] == ["order", "start"]
