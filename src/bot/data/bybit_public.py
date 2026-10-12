"""Клиенты Bybit v5 (pybit).

Публичные рыночные данные берём с основного API (api.bybit.com): демо-счёт торгует
по тем же ценам. Приватные запросы (баланс, ордера) идут на демо-домен
api-demo.bybit.com. Реальный счёт — только при BYBIT_ENV=live и ALLOW_LIVE=true.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pybit.unified_trading import HTTP

from bot.core.config import Secrets


def make_public_client(timeout: int = 10) -> HTTP:
    return HTTP(testnet=False, timeout=timeout)


def make_private_client(secrets: Secrets, timeout: int = 10) -> HTTP:
    key = secrets.bybit_api_key.get_secret_value()
    secret = secrets.bybit_api_secret.get_secret_value()
    if not key or not secret:
        raise ValueError("В .env не заданы BYBIT_API_KEY и BYBIT_API_SECRET")
    return HTTP(
        testnet=False,
        demo=secrets.is_demo,
        api_key=key,
        api_secret=secret,
        timeout=timeout,
    )


@dataclass(frozen=True)
class InstrumentInfo:
    symbol: str
    tick_size: float  # шаг цены
    qty_step: float  # шаг объёма (лота)
    min_order_qty: float
    min_notional: float  # минимальная стоимость ордера в USDT
    max_leverage: float


def parse_instrument(raw: dict[str, Any]) -> InstrumentInfo:
    """Разобрать один элемент result.list ответа /v5/market/instruments-info."""
    lot = raw["lotSizeFilter"]
    return InstrumentInfo(
        symbol=raw["symbol"],
        tick_size=float(raw["priceFilter"]["tickSize"]),
        qty_step=float(lot["qtyStep"]),
        min_order_qty=float(lot["minOrderQty"]),
        min_notional=float(lot.get("minNotionalValue") or 0),
        max_leverage=float(raw["leverageFilter"]["maxLeverage"]),
    )


def get_instrument(client: HTTP, symbol: str, category: str = "linear") -> InstrumentInfo:
    response = client.get_instruments_info(category=category, symbol=symbol)
    items = response["result"]["list"]
    if not items:
        raise ValueError(f"Bybit не знает инструмент {symbol}")
    return parse_instrument(items[0])
