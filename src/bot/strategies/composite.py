"""Составная стратегия: выбор стратегии по режиму рынка (гибрид C, этап 6, шаг 7).

mode=regime — каждая стратегия торгует только в «своём» режиме:
    mean_reversion  — RANGE;
    trend_following — TREND_UP, TREND_DOWN;
mode=all — те же стратегии без учёта режима (база для сравнения).
Если на одной свече сигналят несколько стратегий, берётся первая по порядку в members.
Параметры входящих стратегий — по умолчанию.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from bot.core.models import Position, Signal
from bot.features.regime import Regime, regime_frame
from bot.strategies.base import BaseStrategy, ChartSeries, MarketContext
from bot.strategies.mean_reversion import MeanReversion
from bot.strategies.trend_following import TrendFollowing

MEMBERS: dict[str, type[BaseStrategy]] = {
    MeanReversion.name: MeanReversion,
    TrendFollowing.name: TrendFollowing,
}
ALLOWED: dict[str, set[int]] = {
    MeanReversion.name: {Regime.RANGE},
    TrendFollowing.name: {Regime.TREND_UP, Regime.TREND_DOWN},
}
DEFAULT_PARAMS: dict[str, Any] = {
    "mode": "regime",
    "members": "mean_reversion,trend_following",
}


def _short(name: str) -> str:
    """Тот же префикс, что в signal_id: mean_reversion → mr."""
    return "".join(part[0] for part in name.split("_") if part)


class Composite(BaseStrategy):
    name = "composite"
    version = "1.0.0"

    def __init__(self, timeframe: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(timeframe, {**DEFAULT_PARAMS, **(params or {})})
        if self.params["mode"] not in ("regime", "all"):
            raise ValueError("mode должен быть regime или all")
        names = [n.strip() for n in str(self.params["members"]).split(",") if n.strip()]
        unknown = [n for n in names if n not in MEMBERS]
        if unknown:
            raise ValueError(f"Неизвестные стратегии в members: {unknown}")
        self.members = [MEMBERS[n](timeframe) for n in names]

    def required_history(self) -> int:
        return max(m.required_history() for m in self.members)

    def chart_series(self) -> list[ChartSeries]:
        series = [
            ChartSeries(
                "regime",
                "Режим: 0 боковик, 1 тренд вверх, 2 тренд вниз, 3 высокая волатильность",
                "Режим",
                "#4a90e2",
            )
        ]
        for m in self.members:
            prefix = f"{_short(m.name)}__"
            for s in m.chart_series():
                column = s.column if s.column == "volume" else prefix + s.column
                pane = s.pane if s.pane == "price" else f"{_short(m.name)}: {s.pane}"
                series.append(replace(s, column=column, pane=pane, label=f"{m.name}: {s.label}"))
        return series

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        parts = [regime_frame(bars)[["regime"]]]
        for m in self.members:
            parts.append(m.prepare(bars).add_prefix(f"{_short(m.name)}__"))
        return pd.concat(parts, axis=1)

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        regime = ctx.value("regime")
        signals: list[Signal] = []
        for m in self.members:
            if self.params["mode"] == "regime" and regime not in ALLOWED[m.name]:
                continue
            signals.extend(m.on_bar(ctx.view(f"{_short(m.name)}__")))
        return signals

    def update_stop(self, ctx: MarketContext, position: Position) -> float | None:
        prefix = (position.signal_id or "").split("-")[0]
        for m in self.members:
            if _short(m.name) == prefix:
                return m.update_stop(ctx.view(f"{prefix}__"), position)
        return None
