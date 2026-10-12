"""Тесты сбоев из docs/05 (раздел «Тесты сбоев») на подставном брокере.

Ручные проверки на демо-счёте — в docs/USER_GUIDE.md.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from bot.core.models import Order, OrderRequest, OrderStatus, Position
from bot.execution.broker import TransientBrokerError
from bot.execution.paper import PaperBroker
from bot.runner.live import LiveBot

from .test_live_runner import Clock, feed, make_bot


class FlakyBroker(PaperBroker):
    """Бумажная биржа, до которой можно «отключить сеть». Биржа при этом работает:
    стопы срабатывают (on_minute), но бот не может ни спросить, ни отправить."""

    def __init__(self, base: PaperBroker) -> None:
        super().__init__(base.costs, base.instruments, base.cash, base.now)
        self.down = False
        self.lose_response = False  # ордер доходит, а ответ теряется

    def _check(self) -> None:
        if self.down:
            raise TransientBrokerError("сеть недоступна")

    def place_order(self, request: OrderRequest) -> Order:
        self._check()
        order = super().place_order(request)
        if self.lose_response:
            self.down = True  # процесс «упал» сразу после отправки
            raise TransientBrokerError("таймаут ответа")
        return order

    def find_order(self, symbol: str, order_link_id: str) -> Order | None:
        self._check()
        return super().find_order(symbol, order_link_id)

    def get_positions(self) -> list[Position]:
        self._check()
        return super().get_positions()

    def get_balance(self) -> Any:
        self._check()
        return super().get_balance()


def flaky_bot(tmp_path: Path) -> tuple[LiveBot, FlakyBroker, Any, Clock, pd.DataFrame]:
    bot, storage, clock, _, rest = make_bot(tmp_path)
    assert isinstance(bot.broker, PaperBroker)
    broker = FlakyBroker(bot.broker)
    bot.broker = broker
    return bot, broker, storage, clock, rest


def crash_minute(storage: Any, clock: Clock, price: float) -> pd.DataFrame:
    t = clock.now.floor("1min")
    row = pd.DataFrame(
        {
            "open_time": [t],
            "open": [price],
            "high": [price * 1.001],
            "low": [price * 0.999],
            "close": [price],
            "volume": [1.0],
            "turnover": [1.0],
        }
    )
    storage.write_candles("BTCUSDT", "1m", row)
    clock.now = t + pd.Timedelta(minutes=1, seconds=5)
    return row


def test_1_network_down_with_open_position(tmp_path: Path) -> None:
    bot, broker, storage, clock, rest = flaky_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=1)
    [trade] = bot.store.active_trades()
    broker.down = True
    feed(bot, storage, clock, rest.iloc[60:], hours=1)  # бот не падает
    row = crash_minute(storage, clock, trade.stop * 0.995)  # цена прошла стоп без бота
    bot.on_candles("BTCUSDT", row)
    lost = [e for e in bot.store.events() if e["kind"] == "connection"]
    assert len(lost) == 1 and "нет связи" in lost[0]["message"]  # одно сообщение, не 60
    assert broker.positions == {}  # стоп жил на бирже и сработал
    broker.down = False
    row = crash_minute(storage, clock, trade.stop * 0.995)
    bot.on_candles("BTCUSDT", row)
    assert any("восстановлена" in e["message"] for e in bot.store.events())
    closed = bot.store.get_trade(trade.signal_id)
    assert closed is not None and closed.status == "closed" and closed.exit_reason == "stop"


def test_2_crash_between_send_and_response(tmp_path: Path) -> None:
    bot, broker, storage, clock, rest = flaky_bot(tmp_path)
    bot.start()
    broker.lose_response = True
    feed(bot, storage, clock, rest, hours=1)
    [trade] = bot.store.active_trades()
    assert trade.status == "pending"  # бот не знает, дошёл ли ордер
    broker.lose_response = False
    broker.down = False
    restarted = LiveBot(
        bot.cfg,
        bot.settings,
        broker,
        bot.store,
        storage,
        notify=lambda _: None,
        now=clock,
        strategies=bot.running,
    )
    restarted.start()  # сверка при старте
    again = restarted.store.get_trade(trade.signal_id)
    assert again is not None and again.status == "open"
    assert len(broker.orders) == 1  # второго ордера не было


def test_4_same_signal_twice(tmp_path: Path) -> None:
    bot, storage, clock, _, rest = make_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=1)
    [trade] = bot.store.active_trades()
    running = bot.running[0]
    built = bot.context(running, clock.now.floor("1h"))
    assert built is not None
    ctx, atr_value = built
    bot.try_open(trade.signal, ctx, atr_value)  # тот же сигнал ещё раз
    assert len(bot.store.trades()) == 1
    assert isinstance(bot.broker, PaperBroker) and len(bot.broker.orders) == 1
    assert any("5: позиция по этому сигналу уже есть" in e["message"] for e in bot.store.events())


def test_5_partial_fill_uses_actual_size(tmp_path: Path) -> None:
    bot, broker, storage, clock, rest = flaky_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=1)
    [trade] = bot.store.active_trades()
    # Биржа исполнила только половину: позиция меньше плана, сделка ещё pending
    bot.store.update_trade(trade.signal_id, status="pending")
    broker.positions["BTCUSDT"].qty = round(trade.qty / 2, 3)
    bot.store.order_result(
        broker.orders[trade.signal_id].model_copy(update={"status": OrderStatus.PARTIALLY_FILLED})
    )
    bot.sync()
    fixed = bot.store.get_trade(trade.signal_id)
    assert fixed is not None and fixed.status == "open"
    assert fixed.qty == round(trade.qty / 2, 3)


def test_6_qty_is_rounded_to_exchange_step(tmp_path: Path) -> None:
    bot, storage, clock, _, rest = make_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=1)
    [trade] = bot.store.active_trades()
    step = bot.cfg.instruments["BTCUSDT"].qty_step
    assert abs(trade.qty / step - round(trade.qty / step)) < 1e-9


def test_7_gap_through_stop(tmp_path: Path) -> None:
    bot, storage, clock, _, rest = make_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=1)
    [trade] = bot.store.active_trades()
    gap_open = trade.stop * 0.99  # следующая минута открылась сразу за стопом
    bot.on_candles("BTCUSDT", crash_minute(storage, clock, gap_open))
    closed = bot.store.get_trade(trade.signal_id)
    assert closed is not None and closed.status == "closed" and closed.exit_reason == "stop"
    # исполнение по open гэпа (хуже стопа) с двойным проскальзыванием стопа
    slip = bot.costs.slippage_rate("BTCUSDT", stop=True)
    assert closed.exit_price is not None
    assert abs(closed.exit_price - gap_open * (1 - slip)) < 1e-6
