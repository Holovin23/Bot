"""Breakout v1 (docs/09-start-checklist.md, «Правила стратегий v1»).

Сжатие: ширина Bollinger(20, 2) на предыдущей свече в нижних 10% за последние 100 свечей.
LONG: close выше максимума high 20 свечей до текущей и объём > 1,5 × среднего объёма
20 свечей до текущей. Стоп — минимум low последних 20 свечей (включая текущую),
тейк — 2R, выход по времени через 48 свечей. SHORT — зеркально.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bot.core.models import Side, Signal
from bot.features.indicators import bollinger_width, percentile_rank, rolling_max, rolling_min
from bot.strategies.base import BaseStrategy, MarketContext

DEFAULT_PARAMS: dict[str, Any] = {
    "bb_length": 20,
    "bb_std": 2.0,
    "squeeze_lookback": 100,
    "squeeze_percentile": 0.10,
    "channel_length": 20,
    "volume_length": 20,
    "volume_mult": 1.5,
    "take_r": 2.0,
    "time_stop_bars": 48,
    "allow_short": True,
}


class Breakout(BaseStrategy):
    name = "breakout"
    version = "1.0.0"

    def __init__(self, timeframe: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(timeframe, {**DEFAULT_PARAMS, **(params or {})})

    def required_history(self) -> int:
        p = self.params
        return int(p["bb_length"] + p["squeeze_lookback"] + 1)

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        width = bollinger_width(bars["close"], int(p["bb_length"]), float(p["bb_std"]))
        channel = int(p["channel_length"])
        return pd.DataFrame(
            {
                # Сжатие и канал считаются по свечам ДО текущей (shift(1) — сдвиг в прошлое)
                "squeeze_rank": percentile_rank(width, int(p["squeeze_lookback"])).shift(1),
                "channel_high": rolling_max(bars["high"], channel).shift(1),
                "channel_low": rolling_min(bars["low"], channel).shift(1),
                "avg_volume": bars["volume"].rolling(int(p["volume_length"])).mean().shift(1),
                # Стоп — экстремум последних 20 свечей, включая текущую
                "stop_low": rolling_min(bars["low"], channel),
                "stop_high": rolling_max(bars["high"], channel),
            }
        )

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        p = self.params
        names = (
            "squeeze_rank",
            "channel_high",
            "channel_low",
            "avg_volume",
            "stop_low",
            "stop_high",
        )
        f = {name: ctx.value(name) for name in names}
        if any(pd.isna(v) for v in f.values()):
            return []
        if f["squeeze_rank"] > float(p["squeeze_percentile"]):
            return []
        close, volume = ctx.value("close"), ctx.value("volume")
        if not volume > float(p["volume_mult"]) * f["avg_volume"]:
            return []

        if close > f["channel_high"]:
            side, stop = Side.LONG, f["stop_low"]
        elif p["allow_short"] and close < f["channel_low"]:
            side, stop = Side.SHORT, f["stop_high"]
        else:
            return []
        risk = abs(close - stop)
        if risk <= 0:
            return []
        sign = 1.0 if side is Side.LONG else -1.0
        return [
            self.signal(
                ctx,
                side,
                stop_loss=stop,
                take_profit=close + sign * float(p["take_r"]) * risk,
                time_stop_bars=int(p["time_stop_bars"]),
                reason=(
                    f"пробой {'вверх' if side is Side.LONG else 'вниз'} после сжатия "
                    f"(ранг ширины BB {f['squeeze_rank']:.2f}), объём ×"
                    f"{volume / f['avg_volume']:.1f}"
                ),
                features={**f, "volume_ratio": volume / f["avg_volume"]},
            )
        ]
