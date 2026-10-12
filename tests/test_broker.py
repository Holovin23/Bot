from datetime import UTC, datetime

import pytest

from bot.core.models import (
    Balance,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Signal,
)
from bot.execution.broker import (
    BrokerError,
    DuplicateOrderError,
    OrderRejectedError,
    TransientBrokerError,
    close_link_id,
    close_request,
    entry_request,
    submit_with_retry,
)

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
        take_profit=66350,
    )


class ScriptedBroker:
    """place_order по очереди выполняет сценарий: исключение или «принято»."""

    name = "fake"

    def __init__(self, script: list[Exception | None], lands: bool = False) -> None:
        self.script = script
        self.lands = lands  # ордер дошёл до биржи, хотя ответ потерялся
        self.placed: list[str] = []
        self.on_exchange: dict[str, Order] = {}

    def _order(self, r: OrderRequest) -> Order:
        return Order(
            order_link_id=r.order_link_id,
            symbol=r.symbol,
            side=r.side,
            order_type=r.order_type,
            qty=r.qty,
            status=OrderStatus.FILLED,
            filled_qty=r.qty,
            created_at=NOW,
            updated_at=NOW,
        )

    def place_order(self, request: OrderRequest) -> Order:
        self.placed.append(request.order_link_id)
        step = self.script.pop(0)
        if step is None or self.lands:
            self.on_exchange[request.order_link_id] = self._order(request)
        if step is not None:
            raise step
        return self.on_exchange[request.order_link_id]

    def find_order(self, symbol: str, order_link_id: str) -> Order | None:
        return self.on_exchange.get(order_link_id)

    def cancel_all(self, symbol: str) -> None: ...

    def get_positions(self) -> list[Position]:
        return []

    def get_balance(self) -> Balance:
        return Balance(equity=0, wallet_balance=0, available=0, updated_at=NOW)

    def set_trading_stop(
        self, symbol: str, side: Side, stop_loss: float, take_profit: float | None = None
    ) -> None: ...

    def set_leverage(self, symbol: str, leverage: float) -> None: ...

    def last_exit(self, symbol: str) -> None:
        return None


def no_sleep(_: float) -> None:
    pass


def test_entry_request_uses_signal_id_and_exchange_stops() -> None:
    sig = make_signal()
    r = entry_request(sig, 0.1)
    assert r.order_link_id == sig.signal_id
    assert r.side is OrderSide.BUY and r.order_type is OrderType.MARKET
    assert r.stop_loss == 65000 and r.take_profit == 66350
    assert not r.reduce_only


def test_close_request_is_reduce_only_and_opposite() -> None:
    r = close_request("BTCUSDT", Side.LONG, 0.1, "b-BTCUSDT-1h-20261012T1000-L")
    assert r.reduce_only and r.side is OrderSide.SELL
    assert r.order_link_id == "b-BTCUSDT-1h-20261012T1000-L-X1"
    assert close_link_id("x" * 36, 2) == "x" * 33 + "-X2"


def test_retry_after_network_error_without_order() -> None:
    broker = ScriptedBroker([TransientBrokerError("timeout"), None])
    order = submit_with_retry(broker, entry_request(make_signal(), 0.1), sleep=no_sleep)
    assert order.status is OrderStatus.FILLED
    assert len(broker.placed) == 2


def test_no_second_order_when_first_reached_exchange() -> None:
    broker = ScriptedBroker([TransientBrokerError("timeout")], lands=True)
    order = submit_with_retry(broker, entry_request(make_signal(), 0.1), sleep=no_sleep)
    assert order.filled_qty == 0.1
    assert len(broker.placed) == 1  # второй раз не отправляли


def test_duplicate_returns_existing_order() -> None:
    broker = ScriptedBroker([DuplicateOrderError("dup")], lands=True)
    order = submit_with_retry(broker, entry_request(make_signal(), 0.1), sleep=no_sleep)
    assert order.order_link_id == make_signal().signal_id


def test_rejection_is_not_retried() -> None:
    broker = ScriptedBroker([OrderRejectedError("qty too small"), None])
    with pytest.raises(OrderRejectedError):
        submit_with_retry(broker, entry_request(make_signal(), 0.1), sleep=no_sleep)
    assert len(broker.placed) == 1


def test_gives_up_after_three_attempts_with_backoff() -> None:
    pauses: list[float] = []
    broker = ScriptedBroker([TransientBrokerError("down")] * 3)
    with pytest.raises(BrokerError, match="не отправлен"):
        submit_with_retry(broker, entry_request(make_signal(), 0.1), sleep=pauses.append)
    assert pauses == [1.0, 2.0, 4.0]
