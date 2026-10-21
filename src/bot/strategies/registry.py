"""Список стратегий, доступных по имени.

Торговая стратегия одна — сценарии из Telegram-канала. random_entry — случайные входы
для сравнения в исследованиях, торговать ею не нужно.
"""

from __future__ import annotations

from typing import Any

from bot.strategies.base import BaseStrategy
from bot.strategies.random_entry import RandomEntry
from bot.strategies.telegram_scenario import TelegramScenario

STRATEGIES: dict[str, type[BaseStrategy]] = {
    RandomEntry.name: RandomEntry,
    TelegramScenario.name: TelegramScenario,
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
