"""BybitBroker: ордера на Bybit Demo Trading (api-demo.bybit.com) через pybit.

- orderLinkId = signal_id (закрытие — signal_id-X1), повтор после сбоя — только после
  поиска ордера (submit_with_retry в broker.py);
- стоп-лосс и тейк-профит передаются вместе с ордером входа и живут на бирже
  (tpslMode=Full, срабатывание по LastPrice);
- закрывающие ордера — reduceOnly;
- one-way режим позиций (positionIdx=0).

Реальный счёт (этап 8) — только через BybitBroker.live(): BYBIT_ENV=live, ALLOW_LIVE=true
и подтверждение при старте. По умолчанию всё это выключено.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any

import requests
from pybit.exceptions import FailedRequestError, InvalidRequestError
from pybit.unified_trading import HTTP

from bot.core.config import InstrumentConfig, Secrets
from bot.core.models import (
    Balance,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Side,
)
from bot.execution.broker import (
    BrokerError,
    DuplicateOrderError,
    ExitInfo,
    OrderRejectedError,
    TransientBrokerError,
)

CATEGORY = "linear"
# Коды ответа Bybit v5
DUPLICATE_LINK_ID = 110072
NOT_MODIFIED = {34040, 110043}  # стоп / плечо не изменились — не ошибка
TRANSIENT_CODES = {10000, 10002, 10006, 10016, 10018}  # сервер, время, лимит запросов

STATUS_MAP = {
    "Created": OrderStatus.SENT,
    "New": OrderStatus.ACKNOWLEDGED,
    "Untriggered": OrderStatus.ACKNOWLEDGED,
    "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
    "Filled": OrderStatus.FILLED,
    "Cancelled": OrderStatus.CANCELLED,
    "PartiallyFilledCanceled": OrderStatus.CANCELLED,
    "Deactivated": OrderStatus.CANCELLED,
    "Rejected": OrderStatus.REJECTED,
}

# pybit на уровне DEBUG пишет заголовки запроса, а в них ключ API. Не допускаем.
logging.getLogger("pybit").setLevel(logging.WARNING)


def format_step(value: float, step: float, down: bool = True) -> str:
    """Число кратно шагу биржи, строкой без экспоненты: 0.1029, шаг 0.001 → '0.102'."""
    step_d = Decimal(str(step))
    units = (Decimal(str(value)) / step_d).quantize(
        Decimal(1), rounding=ROUND_DOWN if down else ROUND_HALF_UP
    )
    return format((units * step_d).normalize(), "f")


def _ms(value: Any) -> datetime:
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=UTC)
    except (TypeError, ValueError):
        return datetime.now(UTC)


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number else None


def parse_order(raw: dict[str, Any]) -> Order:
    """Элемент result.list ответа /v5/order/realtime или /v5/order/history."""
    return Order(
        order_id=raw.get("orderId") or None,
        order_link_id=raw.get("orderLinkId", ""),
        symbol=raw["symbol"],
        side=OrderSide(raw["side"]),
        order_type=OrderType(raw.get("orderType", "Market")),
        qty=float(raw.get("qty") or 0),
        price=_float(raw.get("price")),
        status=STATUS_MAP.get(raw.get("orderStatus", ""), OrderStatus.ACKNOWLEDGED),
        filled_qty=float(raw.get("cumExecQty") or 0),
        avg_fill_price=_float(raw.get("avgPrice")),
        reduce_only=bool(raw.get("reduceOnly", False)),
        created_at=_ms(raw.get("createdTime")),
        updated_at=_ms(raw.get("updatedTime")),
    )


def parse_position(raw: dict[str, Any]) -> Position | None:
    """Элемент /v5/position/list. Пустая позиция (size 0) → None."""
    size = float(raw.get("size") or 0)
    if size <= 0 or raw.get("side") not in ("Buy", "Sell"):
        return None
    return Position(
        symbol=raw["symbol"],
        side=Side.LONG if raw["side"] == "Buy" else Side.SHORT,
        size=size,
        entry_price=float(raw["avgPrice"]),
        leverage=float(raw.get("leverage") or 1),
        stop_loss=_float(raw.get("stopLoss")),
        take_profit=_float(raw.get("takeProfit")),
        liq_price=_float(raw.get("liqPrice")),
        unrealised_pnl=float(raw.get("unrealisedPnl") or 0),
        opened_at=_ms(raw.get("createdTime")),
    )


def parse_balance(raw: dict[str, Any]) -> Balance:
    """Первый элемент /v5/account/wallet-balance (UNIFIED)."""
    return Balance(
        coin="USD",
        equity=float(raw.get("totalEquity") or 0),
        wallet_balance=float(raw.get("totalWalletBalance") or 0),
        available=float(raw.get("totalAvailableBalance") or 0),
        updated_at=datetime.now(UTC),
    )


def _client(secrets: Secrets, demo: bool, timeout: int) -> HTTP:
    key = secrets.bybit_api_key.get_secret_value()
    secret = secrets.bybit_api_secret.get_secret_value()
    if not key or not secret:
        raise BrokerError("Не заданы BYBIT_API_KEY и BYBIT_API_SECRET")
    # max_retries=1, force_retry=False: pybit сам не повторяет — повтор только после
    # поиска ордера (submit_with_retry).
    return HTTP(
        testnet=False,
        demo=demo,
        api_key=key,
        api_secret=secret,
        timeout=timeout,
        max_retries=1,
        force_retry=False,
    )


class BybitBroker:
    def __init__(
        self,
        client: HTTP,
        instruments: dict[str, InstrumentConfig],
        name: str = "demo",
    ) -> None:
        self.client = client
        self.instruments = instruments
        self.name = name

    @classmethod
    def demo(
        cls, secrets: Secrets, instruments: dict[str, InstrumentConfig], timeout: int = 10
    ) -> BybitBroker:
        if not secrets.is_demo:
            raise BrokerError(
                "Режим demo требует BYBIT_ENV=demo. Реальный счёт — только --mode live (этап 8)."
            )
        return cls(_client(secrets, demo=True, timeout=timeout), instruments, name="demo")

    @classmethod
    def live(
        cls,
        secrets: Secrets,
        instruments: dict[str, InstrumentConfig],
        confirmed: bool,
        timeout: int = 10,
    ) -> BybitBroker:
        """Реальный счёт (этап 8). Три условия, все — решение владельца: BYBIT_ENV=live,
        ALLOW_LIVE=true (в отдельном файле .env.live) и подтверждение при старте."""
        if secrets.bybit_env != "live" or not secrets.allow_live:
            raise BrokerError("Реальный счёт выключен: нужны BYBIT_ENV=live и ALLOW_LIVE=true")
        if not confirmed:
            raise BrokerError("Реальный счёт не подтверждён при запуске")
        return cls(_client(secrets, demo=False, timeout=timeout), instruments, name="live")

    # --- вызов API с переводом ошибок -------------------------------------------------

    def _call(self, method: str, **params: Any) -> dict[str, Any]:
        try:
            response: dict[str, Any] = getattr(self.client, method)(**params)
        except InvalidRequestError as exc:
            code = int(exc.status_code)
            if code == DUPLICATE_LINK_ID:
                raise DuplicateOrderError(f"{code}: {exc.message}") from None
            if code in TRANSIENT_CODES:
                raise TransientBrokerError(f"{code}: {exc.message}") from None
            if code in NOT_MODIFIED:
                return {"result": {}}
            raise OrderRejectedError(f"{code}: {exc.message}") from None
        except FailedRequestError as exc:  # HTTP не 200, ответ не разобран
            raise TransientBrokerError(f"HTTP {exc.status_code}: {exc.message}") from None
        except requests.exceptions.RequestException as exc:
            raise TransientBrokerError(f"сеть: {type(exc).__name__}") from None
        return response

    def _qty(self, symbol: str, qty: float) -> str:
        instrument = self.instruments.get(symbol)
        return format_step(qty, instrument.qty_step) if instrument else str(qty)

    def _price(self, symbol: str, price: float) -> str:
        instrument = self.instruments.get(symbol)
        return format_step(price, instrument.tick_size, down=False) if instrument else str(price)

    # --- интерфейс Broker ---------------------------------------------------------------

    def place_order(self, request: OrderRequest) -> Order:
        params: dict[str, Any] = {
            "category": CATEGORY,
            "symbol": request.symbol,
            "side": request.side.value,
            "orderType": request.order_type.value,
            "qty": self._qty(request.symbol, request.qty),
            "orderLinkId": request.order_link_id,
            "positionIdx": 0,
        }
        if request.price is not None:
            params["price"] = self._price(request.symbol, request.price)
        if request.reduce_only:
            params["reduceOnly"] = True
        if request.stop_loss is not None or request.take_profit is not None:
            params["tpslMode"] = "Full"
        if request.stop_loss is not None:
            params["stopLoss"] = self._price(request.symbol, request.stop_loss)
            params["slTriggerBy"] = "LastPrice"
        if request.take_profit is not None:
            params["takeProfit"] = self._price(request.symbol, request.take_profit)
            params["tpTriggerBy"] = "LastPrice"
        response = self._call("place_order", **params)
        result = response.get("result", {})
        # Ответ подтверждает только приём; исполнение узнаём поиском ордера.
        found = self.find_order(request.symbol, request.order_link_id)
        if found is not None:
            return found
        now = datetime.now(UTC)
        return Order(
            order_id=result.get("orderId"),
            order_link_id=request.order_link_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            qty=request.qty,
            status=OrderStatus.ACKNOWLEDGED,
            reduce_only=request.reduce_only,
            created_at=now,
            updated_at=now,
        )

    def find_order(self, symbol: str, order_link_id: str) -> Order | None:
        for method in ("get_open_orders", "get_order_history"):
            response = self._call(
                method, category=CATEGORY, symbol=symbol, orderLinkId=order_link_id
            )
            items = response.get("result", {}).get("list", [])
            if items:
                return parse_order(items[0])
        return None

    def cancel_all(self, symbol: str) -> None:
        self._call("cancel_all_orders", category=CATEGORY, symbol=symbol)

    def get_positions(self) -> list[Position]:
        response = self._call("get_positions", category=CATEGORY, settleCoin="USDT")
        items = response.get("result", {}).get("list", [])
        return [p for p in (parse_position(i) for i in items) if p is not None]

    def get_balance(self) -> Balance:
        response = self._call("get_wallet_balance", accountType="UNIFIED")
        accounts = response.get("result", {}).get("list", [])
        if not accounts:
            raise BrokerError("Bybit не вернул баланс")
        return parse_balance(accounts[0])

    def set_trading_stop(
        self, symbol: str, side: Side, stop_loss: float, take_profit: float | None = None
    ) -> None:
        params: dict[str, Any] = {
            "category": CATEGORY,
            "symbol": symbol,
            "tpslMode": "Full",
            "positionIdx": 0,
            "stopLoss": self._price(symbol, stop_loss),
            "slTriggerBy": "LastPrice",
        }
        if take_profit is not None:
            params["takeProfit"] = self._price(symbol, take_profit)
        self._call("set_trading_stop", **params)

    def set_leverage(self, symbol: str, leverage: float) -> None:
        value = format_step(leverage, 0.01)
        self._call(
            "set_leverage",
            category=CATEGORY,
            symbol=symbol,
            buyLeverage=value,
            sellLeverage=value,
        )

    def last_exit(self, symbol: str) -> ExitInfo | None:
        response = self._call("get_closed_pnl", category=CATEGORY, symbol=symbol, limit=1)
        items = response.get("result", {}).get("list", [])
        if not items:
            return None
        raw = items[0]
        return ExitInfo(
            symbol=symbol,
            exit_price=float(raw.get("avgExitPrice") or 0),
            exit_time=_ms(raw.get("updatedTime")),
            net_pnl=float(raw.get("closedPnl") or 0),  # комиссии учтены, funding — нет
            reason="unknown",
        )
