"""Список стратегий, доступных из командной строки по имени."""

from __future__ import annotations

from typing import Any

from bot.strategies.base import BaseStrategy
from bot.strategies.composite import Composite
from bot.strategies.mean_reversion import MeanReversion
from bot.strategies.random_entry import RandomEntry
from bot.strategies.telegram_scenario import TelegramScenario
from bot.strategies.time_series_momentum import TimeSeriesMomentum
from bot.strategies.trend_following import TrendFollowing

STRATEGIES: dict[str, type[BaseStrategy]] = {
    Composite.name: Composite,
    MeanReversion.name: MeanReversion,
    RandomEntry.name: RandomEntry,
    TelegramScenario.name: TelegramScenario,
    TimeSeriesMomentum.name: TimeSeriesMomentum,
    TrendFollowing.name: TrendFollowing,
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
