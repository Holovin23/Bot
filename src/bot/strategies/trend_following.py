"""Trend Following v1 (docs/09-start-checklist.md, «Правила стратегий v1»).

LONG: тренд — EMA50 > EMA200 и close > EMA200; откат — low <= EMA20 и close > EMA20.
Стоп: close − 2 × ATR(14). Тейка нет: выход по трейлингу Chandelier — максимум high
с момента входа − 3 × ATR(22), стоп только подтягивается. SHORT — зеркально.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bot.core.models import Position, Side, Signal
from bot.features.indicators import atr, ema
from bot.strategies.base import BaseStrategy, MarketContext

DEFAULT_PARAMS: dict[str, Any] = {
    "ema_fast": 20,
    "ema_mid": 50,
    "ema_slow": 200,
    "atr_length": 14,
    "stop_atr": 2.0,
    "trail_atr_length": 22,
    "trail_atr": 3.0,
    "allow_short": True,
}


class TrendFollowing(BaseStrategy):
    name = "trend_following"
    version = "1.0.0"

    def __init__(self, timeframe: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(timeframe, {**DEFAULT_PARAMS, **(params or {})})

    def required_history(self) -> int:
        p = self.params
        return int(max(p["ema_slow"], p["atr_length"], p["trail_atr_length"])) + 1

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        close = bars["close"]
        return pd.DataFrame(
            {
                "ema_fast": ema(close, int(p["ema_fast"])),
                "ema_mid": ema(close, int(p["ema_mid"])),
                "ema_slow": ema(close, int(p["ema_slow"])),
                "atr": atr(bars, int(p["atr_length"])),
                "trail_atr": atr(bars, int(p["trail_atr_length"])),
            }
        )

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        close, high, low = ctx.value("close"), ctx.value("high"), ctx.value("low")
        fast, mid, slow = ctx.value("ema_fast"), ctx.value("ema_mid"), ctx.value("ema_slow")
        value = ctx.value("atr")
        if not value > 0 or any(pd.isna(x) for x in (fast, mid, slow)):
            return []
        features = {"ema20": fast, "ema50": mid, "ema200": slow, "atr": value}
        stop_distance = float(self.params["stop_atr"]) * value

        if mid > slow and close > slow and low <= fast < close:
            return [
                self.signal(
                    ctx,
                    Side.LONG,
                    stop_loss=close - stop_distance,
                    reason="тренд вверх (EMA50 > EMA200), откат к EMA20 и закрытие выше",
                    features=features,
                )
            ]
        if self.params["allow_short"] and mid < slow and close < slow and high >= fast > close:
            return [
                self.signal(
                    ctx,
                    Side.SHORT,
                    stop_loss=close + stop_distance,
                    reason="тренд вниз (EMA50 < EMA200), откат к EMA20 и закрытие ниже",
                    features=features,
                )
            ]
        return []

    def update_stop(self, ctx: MarketContext, position: Position) -> float | None:
        """Chandelier: от экстремума с момента входа на trail_atr × ATR(22)."""
        if position.opened_at is None:
            return None
        bars = ctx.bars
        since_entry = bars[bars["open_time"] >= pd.Timestamp(position.opened_at).floor("min")]
        if since_entry.empty:
            return None
        distance = float(self.params["trail_atr"]) * ctx.value("trail_atr")
        if position.side is Side.LONG:
            return float(since_entry["high"].max()) - distance
        return float(since_entry["low"].min()) + distance
