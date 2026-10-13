"""Список монет интерфейса и добавление новой монеты.

Новая монета проверяется на Bybit (/v5/market/instruments-info, категория linear):
она должна существовать и торговаться. Шаг цены, шаг и минимум объёма берутся оттуда
и сохраняются в config/symbols.local.yaml, проскальзывание для бэктеста — 0,1%
(NEW_SYMBOL_SLIPPAGE). Монеты из config/default.yaml удалить отсюда нельзя.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bot.core.config import (
    LOCAL_SYMBOLS_NAME,
    NEW_SYMBOL_SLIPPAGE,
    local_symbols,
    save_local_symbols,
)
from bot.data.storage import Storage

SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,30}$")

# (символ) -> элемент result.list ответа instruments-info или None, если такого нет
InstrumentLookup = Callable[[str], dict[str, Any] | None]


def normalize_symbol(text: str) -> str:
    symbol = text.strip().upper().replace("/", "").replace("-", "")
    if symbol and not symbol.endswith(("USDT", "USDC", "PERP")):
        symbol += "USDT"  # «sol» → SOLUSDT: бот торгует бессрочными USDT-контрактами
    if not SYMBOL_RE.match(symbol):
        raise ValueError(f"Недопустимое имя монеты: {text!r}")
    return symbol


class SymbolBook:
    def __init__(
        self, symbols: list[str], config_dir: Path, lookup: InstrumentLookup | None = None
    ) -> None:
        self.symbols = symbols  # тот же список, что видит остальной интерфейс
        self.path = config_dir / LOCAL_SYMBOLS_NAME
        self.lookup = lookup
        self._lock = threading.Lock()

    def local(self) -> set[str]:
        return set(local_symbols(self.path))

    def add(self, text: str) -> dict[str, Any]:
        symbol = normalize_symbol(text)
        if symbol in self.symbols:
            return {"symbol": symbol, "added": False}
        if self.lookup is None:
            raise RuntimeError("Проверка монеты на Bybit не настроена")
        raw = self.lookup(symbol)
        if raw is None:
            raise ValueError(f"На Bybit нет бессрочного контракта {symbol}")
        status = raw.get("status", "Trading")
        if status != "Trading":
            raise ValueError(f"{symbol} на Bybit сейчас не торгуется (статус {status})")
        lot, price = raw["lotSizeFilter"], raw["priceFilter"]
        item = {
            "tick_size": float(price["tickSize"]),
            "qty_step": float(lot["qtyStep"]),
            "min_qty": float(lot["minOrderQty"]),
            "slippage": NEW_SYMBOL_SLIPPAGE,
        }
        with self._lock:
            data = local_symbols(self.path)
            data[symbol] = item
            save_local_symbols(self.path, data)
            if symbol not in self.symbols:
                self.symbols.append(symbol)
        return {"symbol": symbol, "added": True, **item}

    def remove(self, symbol: str) -> None:
        with self._lock:
            data = local_symbols(self.path)
            if symbol not in data:
                raise ValueError(f"{symbol} задан в config/default.yaml — уберите его там")
            del data[symbol]
            save_local_symbols(self.path, data)
            if symbol in self.symbols:
                self.symbols.remove(symbol)


def coverage(storage: Storage, symbol: str) -> dict[str, Any]:
    """С какого и по какое время есть 1m-свечи (секунды UTC)."""
    first = storage.first_candle_time(symbol, "1m")
    last = storage.last_candle_time(symbol, "1m")
    return {
        "symbol": symbol,
        "first": int(first.timestamp()) if first is not None else None,
        "last": int(last.timestamp()) if last is not None else None,
    }
