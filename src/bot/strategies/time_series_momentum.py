"""Дневной импульс (time-series momentum), по мотивам «черепах» (система 2).

Рассчитана на таймфрейм 1d: сделок мало (единицы в год), поэтому комиссии почти не
влияют на результат. Идея: цена, обновившая максимум 55 дней, чаще продолжает расти.

LONG: close выше максимума high 55 свечей до текущей. Стоп: close − 2 × ATR(20).
Выход — трейлинг: минимум low последних 20 свечей (стоп только подтягивается).
SHORT — зеркально (минимум 55 свечей, стоп по максимуму 20 свечей).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bot.core.models import Position, Side, Signal
from bot.features.indicators import atr, rolling_max, rolling_min
from bot.strategies.base import BaseStrategy, ChartSeries, MarketContext

DEFAULT_PARAMS: dict[str, Any] = {
    "entry_length": 55,
    "exit_length": 20,
    "atr_length": 20,
    "stop_atr": 2.0,
    "allow_short": True,
}


class TimeSeriesMomentum(BaseStrategy):
    name = "time_series_momentum"
    version = "1.0.0"

    def __init__(self, timeframe: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(timeframe, {**DEFAULT_PARAMS, **(params or {})})

    def required_history(self) -> int:
        p = self.params
        return int(max(p["entry_length"] + 1, p["exit_length"], p["atr_length"] + 1))

    def chart_series(self) -> list[ChartSeries]:
        p = self.params
        n, m = p["entry_length"], p["exit_length"]
        return [
            ChartSeries("channel_high", f"Максимум {n} свечей (вход)", color="#1a7f37"),
            ChartSeries("channel_low", f"Минимум {n} свечей (вход)", color="#cf222e"),
            ChartSeries(
                "exit_low", f"Минимум {m} свечей (выход лонга)", color="#8b949e", dashed=True
            ),
            ChartSeries(
                "exit_high", f"Максимум {m} свечей (выход шорта)", color="#8b949e", dashed=True
            ),
            ChartSeries("atr", f"ATR({p['atr_length']}) — стоп {p['stop_atr']}×", "ATR"),
        ]

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        entry, exit_ = int(p["entry_length"]), int(p["exit_length"])
        return pd.DataFrame(
            {
                # Канал входа — по свечам ДО текущей: пробой считается по её close
                "channel_high": rolling_max(bars["high"], entry).shift(1),
                "channel_low": rolling_min(bars["low"], entry).shift(1),
                # Канал выхода — включая текущую свечу
                "exit_low": rolling_min(bars["low"], exit_),
                "exit_high": rolling_max(bars["high"], exit_),
                "atr": atr(bars, int(p["atr_length"])),
            }
        )

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        p = self.params
        close = ctx.value("close")
        high, low, value = ctx.value("channel_high"), ctx.value("channel_low"), ctx.value("atr")
        if any(pd.isna(x) for x in (high, low)) or not value > 0:
            return []
        features = {"channel_high": high, "channel_low": low, "atr": value}
        stop_distance = float(p["stop_atr"]) * value
        n = int(p["entry_length"])
        if close > high:
            return [
                self.signal(
                    ctx,
                    Side.LONG,
                    stop_loss=close - stop_distance,
                    reason=f"закрытие выше максимума {n} свечей ({high:g})",
                    features=features,
                )
            ]
        if p["allow_short"] and close < low:
            return [
                self.signal(
                    ctx,
                    Side.SHORT,
                    stop_loss=close + stop_distance,
                    reason=f"закрытие ниже минимума {n} свечей ({low:g})",
                    features=features,
                )
            ]
        return []

    def update_stop(self, ctx: MarketContext, position: Position) -> float | None:
        """Стоп по каналу выхода: минимум (для шорта — максимум) последних свечей."""
        column = "exit_low" if position.side is Side.LONG else "exit_high"
        level = ctx.value(column)
        return None if pd.isna(level) else level
