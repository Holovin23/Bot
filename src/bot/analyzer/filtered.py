"""Стратегия с ML-фильтром. Один и тот же код для бэктеста, paper и демо.

Фильтр только убирает сигналы базовой стратегии с вероятностью успеха ниже порога.
Он не создаёт сигналы и не меняет стоп, тейк и размер (размер считает риск-менеджер).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bot.analyzer.dataset import FEATURE_COLUMNS, market_features
from bot.analyzer.ml_filter import FilterModel
from bot.core.models import Position, Side, Signal
from bot.strategies.base import MarketContext, Strategy

PREFIX = "ml_"
MARKET_COLUMNS = [c for c in FEATURE_COLUMNS if c != "side"]


class FilteredStrategy:
    def __init__(self, base: Strategy, model: FilterModel) -> None:
        self.base = base
        self.model = model
        self.name = f"{base.name}_ml"
        self.version = base.version
        self.timeframe = base.timeframe
        self.params: dict[str, Any] = {
            **base.params,
            "ml_model": model.kind,
            "ml_threshold": round(model.threshold, 4),
        }

    def required_history(self) -> int:
        return self.base.required_history()

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        own = self.base.prepare(bars)
        market = market_features(bars)[MARKET_COLUMNS].add_prefix(PREFIX)
        return pd.concat([own, market], axis=1)

    def probability(self, ctx: MarketContext, side: Side) -> float:
        row = {c: ctx.value(PREFIX + c) for c in MARKET_COLUMNS}
        row["side"] = 1.0 if side is Side.LONG else -1.0
        return float(self.model.probability(pd.DataFrame([row]))[0])

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        kept = []
        for signal in self.base.on_bar(ctx):
            p = self.probability(ctx, signal.side)
            if p >= self.model.threshold:
                features = {**signal.features, "p_success": round(p, 4)}
                kept.append(signal.model_copy(update={"features": features}))
        return kept

    def update_stop(self, ctx: MarketContext, position: Position) -> float | None:
        return self.base.update_stop(ctx, position)
