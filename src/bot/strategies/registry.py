"""Список стратегий, доступных из командной строки по имени."""

from __future__ import annotations

from typing import Any

from bot.strategies.base import BaseStrategy
from bot.strategies.random_entry import RandomEntry

STRATEGIES: dict[str, type[BaseStrategy]] = {
    RandomEntry.name: RandomEntry,
}


def create_strategy(
    name: str, timeframe: str, params: dict[str, Any] | None = None
) -> BaseStrategy:
    try:
        cls = STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Неизвестная стратегия {name!r}. Есть: {known}") from None
    return cls(timeframe, params)
