from datetime import UTC, datetime
from pathlib import Path

import pytest

from bot.backtest.costs import CostModel
from bot.core.config import InstrumentConfig
from bot.core.db import connect
from bot.core.models import Position, Side, Signal
from bot.execution.broker import ExitInfo, entry_request
from bot.execution.paper import PaperBroker
from bot.execution.reconcile import emergency_stop, exit_reason, reconcile
from bot.execution.store import Store

T = datetime(2026, 10, 12, 11, 0, tzinfo=UTC)
BTC = {
    "BTCUSDT": InstrumentConfig(tick_size=0.1, qty_step=0.001, min_qty=0.001),
    "ETHUSDT": InstrumentConfig(tick_size=0.01, qty_step=0.01, min_qty=0.01),
}


def make_signal(symbol: str = "BTCUSDT") -> Signal:
    return Signal.create(
        strategy="breakout",
        strategy_version="1",
        symbol=symbol,
        timeframe="1h",
        bar_time=datetime(2026, 10, 12, 10, 0, tzinfo=UTC),
        side=Side.LONG,
        entry_price=65450,
        stop_loss=65000,
        take_profit=66350,
    )


@pytest.fixture
def setup(tmp_path: Path) -> tuple[PaperBroker, Store]:
    broker = PaperBroker(CostModel(), BTC, 10_000, now=lambda: T)
    broker.on_minute("BTCUSDT", 65450, 65450, 65450, 65450, T)
    broker.on_minute("ETHUSDT", 2500, 2500, 2500, 2500, T)
    return broker, Store(connect(tmp_path / "bot.sqlite"), "paper")


def test_pending_trade_with_position_becomes_open(setup: tuple[PaperBroker, Store]) -> None:
    broker, store = setup
    sig = make_signal()
    store.add_trade(sig, 0.1, 50)
    request = entry_request(sig, 0.1)
    store.order_sent(request)
    broker.place_order(request)  # ответ «потерялся»: процесс упал после отправки
    report = reconcile(broker, store)
    trade = store.get_trade(sig.signal_id)
    assert trade is not None and trade.status == "open"
    assert store.order_status(sig.signal_id) == "FILLED"
    assert report.warnings == []


def test_pending_without_order_fails(setup: tuple[PaperBroker, Store]) -> None:
    broker, store = setup
    sig = make_signal()
    store.add_trade(sig, 0.1, 50)
    store.order_sent(entry_request(sig, 0.1))
    reconcile(broker, store)
    trade = store.get_trade(sig.signal_id)
    assert trade is not None and trade.status == "failed"
    assert store.order_status(sig.signal_id) == "REJECTED"


def test_open_trade_closed_by_exchange_stop(setup: tuple[PaperBroker, Store]) -> None:
    broker, store = setup
    sig = make_signal()
    store.add_trade(sig, 0.1, 50)
    broker.place_order(entry_request(sig, 0.1))
    store.update_trade(sig.signal_id, status="open")
    broker.on_minute("BTCUSDT", 65100, 65100, 64900, 64950, T)  # стоп, пока бот «спал»
    reconcile(broker, store)
    trade = store.get_trade(sig.signal_id)
    assert trade is not None and trade.status == "closed" and trade.exit_reason == "stop"
    assert trade.net_pnl is not None and trade.net_pnl < 0


def test_unknown_position_warns_and_gets_emergency_stop(setup: tuple[PaperBroker, Store]) -> None:
    broker, store = setup
    broker.restore(10_000, [Position(symbol="ETHUSDT", side=Side.SHORT, size=1, entry_price=2500)])
    report = reconcile(broker, store, unknown_stop_pct=0.02)
    assert any("неизвестная позиция ETHUSDT" in w for w in report.warnings)
    [pos] = broker.get_positions()
    assert pos.stop_loss == pytest.approx(2550)  # 2500 × 1,02
    assert store.events()[0]["level"] == "warning"


def test_unknown_position_closed_when_allowed(setup: tuple[PaperBroker, Store]) -> None:
    broker, store = setup
    broker.restore(10_000, [Position(symbol="ETHUSDT", side=Side.LONG, size=1, entry_price=2500)])
    reconcile(broker, store, close_unknown=True)
    assert broker.get_positions() == []


def test_known_position_without_stop_gets_trade_stop(setup: tuple[PaperBroker, Store]) -> None:
    broker, store = setup
    sig = make_signal()
    store.add_trade(sig, 0.1, 50)
    store.update_trade(sig.signal_id, status="open")
    broker.restore(
        10_000,
        [Position(symbol="BTCUSDT", side=Side.LONG, size=0.1, entry_price=65450, signal_id=None)],
    )
    report = reconcile(broker, store)
    assert broker.get_positions()[0].stop_loss == 65000
    assert broker.get_positions()[0].take_profit == 66350
    assert report.warnings == []


def test_helpers() -> None:
    pos = Position(symbol="BTCUSDT", side=Side.LONG, size=1, entry_price=100)
    assert emergency_stop(pos, 0.02) == pytest.approx(98)
    store_trade = type("T", (), {"stop": 95.0, "take": 110.0})()
    info = ExitInfo("BTCUSDT", 109.5, T, 9.0, "unknown")
    assert exit_reason(store_trade, info) == "take"  # type: ignore[arg-type]
