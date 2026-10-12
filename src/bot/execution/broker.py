"""Интерфейс брокера: один и тот же для paper, демо и (в будущем) реального счёта.

Стратегии и риск-менеджер о брокере не знают — с ним работает только runner.
Интерфейс синхронный (см. docs/tasks/stage-7.md, «Решения по умолчанию»).

Повтор после сбоя (docs/05): при ошибке сети ордер мог дойти до биржи. Поэтому перед
повтором ищем ордер по orderLinkId; повторяем, только если его нет. Ошибки валидации
(биржа отклонила ордер) не повторяются.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from bot.core.models import (
    ORDER_LINK_ID_MAX_LEN,
    Balance,
    Order,
    OrderRequest,
    OrderSide,
    OrderType,
    Position,
    Side,
    Signal,
)

RETRY_PAUSES_S = (1.0, 2.0, 4.0)


class BrokerError(Exception):
    """Любая ошибка брокера."""


class TransientBrokerError(BrokerError):
    """Сеть, таймаут, 5xx: неизвестно, дошёл ли ордер. Повторять можно после поиска."""


class OrderRejectedError(BrokerError):
    """Биржа отклонила запрос (валидация, баланс, параметры). Не повторять."""


class DuplicateOrderError(BrokerError):
    """Ордер с таким orderLinkId уже есть — значит, прошлая попытка дошла."""


class Broker(Protocol):
    name: str  # paper | demo | live

    def place_order(self, request: OrderRequest) -> Order: ...

    def find_order(self, symbol: str, order_link_id: str) -> Order | None: ...

    def cancel_all(self, symbol: str) -> None: ...

    def get_positions(self) -> list[Position]: ...

    def get_balance(self) -> Balance: ...

    def set_trading_stop(
        self, symbol: str, side: Side, stop_loss: float, take_profit: float | None = None
    ) -> None: ...

    def set_leverage(self, symbol: str, leverage: float) -> None: ...


def entry_side(side: Side) -> OrderSide:
    return OrderSide.BUY if side is Side.LONG else OrderSide.SELL


def close_side(side: Side) -> OrderSide:
    return OrderSide.SELL if side is Side.LONG else OrderSide.BUY


def entry_request(signal: Signal, qty: float) -> OrderRequest:
    """Рыночный вход со стопом и тейком на стороне биржи. orderLinkId = signal_id."""
    return OrderRequest(
        symbol=signal.symbol,
        side=entry_side(signal.side),
        order_type=OrderType.MARKET,
        qty=qty,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        order_link_id=signal.signal_id,
        signal_id=signal.signal_id,
    )


def close_link_id(signal_id: str, attempt: int = 1) -> str:
    """id закрывающего ордера: '<signal_id>-X1'. Новая попытка после отказа — X2 и т. д."""
    suffix = f"-X{attempt}"
    return signal_id[: ORDER_LINK_ID_MAX_LEN - len(suffix)] + suffix


def close_request(
    symbol: str, side: Side, qty: float, signal_id: str, attempt: int = 1
) -> OrderRequest:
    """Рыночное закрытие позиции: всегда reduceOnly, чтобы не открыть встречную."""
    return OrderRequest(
        symbol=symbol,
        side=close_side(side),
        order_type=OrderType.MARKET,
        qty=qty,
        reduce_only=True,
        order_link_id=close_link_id(signal_id, attempt),
        signal_id=signal_id,
    )


def submit_with_retry(
    broker: Broker,
    request: OrderRequest,
    sleep: Callable[[float], None] = time.sleep,
    pauses: tuple[float, ...] = RETRY_PAUSES_S,
) -> Order:
    """До len(pauses) попыток. После каждой ошибки сети — пауза и поиск ордера по
    orderLinkId: нашёлся — возвращаем его, не отправляя второй раз."""
    last_error: BrokerError | None = None
    for pause in pauses:
        try:
            return broker.place_order(request)
        except DuplicateOrderError:
            found = broker.find_order(request.symbol, request.order_link_id)
            if found is not None:
                return found
            raise
        except TransientBrokerError as exc:
            last_error = exc
            sleep(pause)
            try:
                found = broker.find_order(request.symbol, request.order_link_id)
            except TransientBrokerError as lookup_error:
                last_error = lookup_error
                continue  # повтор безопасен: дубль orderLinkId биржа отклонит
            if found is not None:
                return found
    raise BrokerError(f"ордер {request.order_link_id} не отправлен: {last_error}")


def utc_now() -> datetime:
    return datetime.now(UTC)
