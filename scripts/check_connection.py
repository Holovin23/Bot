"""Проверка подключения к Bybit (этап 0.4). Ордера НЕ отправляет.

Запуск: uv run python scripts/check_connection.py
"""

from __future__ import annotations

import time

from pybit.exceptions import FailedRequestError, InvalidRequestError

from bot.core.config import load_config
from bot.data.bybit_public import get_instrument, make_private_client, make_public_client


def main() -> None:
    cfg = load_config()
    public = make_public_client()

    print("== Серверное время (api.bybit.com) ==")
    local_ms = time.time() * 1000
    server = public.get_server_time()
    server_ms = int(server["result"]["timeNano"]) / 1_000_000
    print(
        f"Сервер: {int(server_ms)} мс, разница с локальными часами: {server_ms - local_ms:+.0f} мс"
    )
    print("(больше ±1000 мс — синхронизируйте часы Windows)")

    print("\n== Инструменты ==")
    for symbol in cfg.data.symbols:
        info = get_instrument(public, symbol)
        print(
            f"{symbol}: шаг цены {info.tick_size}, шаг лота {info.qty_step}, "
            f"мин. объём {info.min_order_qty}, мин. сумма {info.min_notional} USDT, "
            f"макс. плечо {info.max_leverage}x"
        )

    domain = "api-demo.bybit.com" if cfg.secrets.is_demo else "api.bybit.com (РЕАЛЬНЫЙ СЧЁТ)"
    print(f"\n== Баланс ({domain}) ==")
    private = make_private_client(cfg.secrets)
    wallet = private.get_wallet_balance(accountType="UNIFIED")
    account = wallet["result"]["list"][0]
    print(f"Equity: {account['totalEquity']} USD, доступно: {account['totalAvailableBalance']} USD")
    for coin in account.get("coin", []):
        if float(coin.get("walletBalance") or 0):
            print(f"  {coin['coin']}: {coin['walletBalance']}")

    print("\n== Комиссии аккаунта ==")
    for symbol in cfg.data.symbols:
        try:
            fees = private.get_fee_rates(category="linear", symbol=symbol)
            rate = fees["result"]["list"][0]
            print(f"{symbol}: taker {rate['takerFeeRate']}, maker {rate['makerFeeRate']}")
        except (InvalidRequestError, FailedRequestError) as error:
            print(f"{symbol}: демо-API не отдаёт комиссии ({error.status_code}: {error.message})")

    print("\nГотово. Ордера не отправлялись.")


if __name__ == "__main__":
    main()
