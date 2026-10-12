"""Mean Reversion v1 (docs/09-start-checklist.md, «Правила стратегий v1»).

LONG: боковик ADX(14) < 20, close ниже нижней полосы Bollinger(20, 2) и RSI(14) < 30.
Тейк — средняя линия Bollinger на свече сигнала, стоп — close − 1,5 × ATR(14),
выход по времени через 24 свечи. Сигнал не создаётся, если R:R < 1,0.
SHORT — зеркально (выше верхней полосы, RSI > 70).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bot.core.models import Side, Signal
from bot.features.indicators import adx, atr, bollinger, rsi
from bot.strategies.base import BaseStrategy, MarketContext

DEFAULT_PARAMS: dict[str, Any] = {
    "bb_length": 20,
    "bb_std": 2.0,
    "rsi_length": 14,
    "rsi_low": 30.0,
    "rsi_high": 70.0,
    "adx_length": 14,
    "adx_max": 20.0,
    "atr_length": 14,
    "stop_atr": 1.5,
    "time_stop_bars": 24,
    "min_rr": 1.0,
    "allow_short": True,
}


class MeanReversion(BaseStrategy):
    name = "mean_reversion"
    version = "1.0.0"

    def __init__(self, timeframe: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(timeframe, {**DEFAULT_PARAMS, **(params or {})})

    def required_history(self) -> int:
        p = self.params
        # ADX сглаживается дважды: нужно ~2 × длина свечей
        return int(max(p["bb_length"], p["rsi_length"] + 1, 2 * p["adx_length"], p["atr_length"]))

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        bands = bollinger(bars["close"], int(p["bb_length"]), float(p["bb_std"]))
        return pd.DataFrame(
            {
                "bb_lower": bands["bb_lower"],
                "bb_middle": bands["bb_middle"],
                "bb_upper": bands["bb_upper"],
                "rsi": rsi(bars["close"], int(p["rsi_length"])),
                "adx": adx(bars, int(p["adx_length"]))["adx"],
                "atr": atr(bars, int(p["atr_length"])),
            }
        )

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        p = self.params
        close = ctx.value("close")
        lower, middle, upper = ctx.value("bb_lower"), ctx.value("bb_middle"), ctx.value("bb_upper")
        rsi_value, adx_value, atr_value = ctx.value("rsi"), ctx.value("adx"), ctx.value("atr")
        values = (lower, middle, upper, rsi_value, adx_value, atr_value)
        if any(pd.isna(v) for v in values) or not atr_value > 0:
            return []
        if adx_value >= float(p["adx_max"]):
            return []
        features = {"adx": adx_value, "rsi": rsi_value, "atr": atr_value, "bb_middle": middle}
        stop_distance = float(p["stop_atr"]) * atr_value

        if close < lower and rsi_value < float(p["rsi_low"]):
            side, stop, reason = Side.LONG, close - stop_distance, "ниже нижней полосы BB, RSI < 30"
        elif p["allow_short"] and close > upper and rsi_value > float(p["rsi_high"]):
            side, stop, reason = (
                Side.SHORT,
                close + stop_distance,
                "выше верхней полосы BB, RSI > 70",
            )
        else:
            return []

        reward_to_risk = abs(middle - close) / stop_distance
        if reward_to_risk < float(p["min_rr"]):
            return []
        features["rr"] = reward_to_risk
        return [
            self.signal(
                ctx,
                side,
                stop_loss=stop,
                take_profit=middle,
                time_stop_bars=int(p["time_stop_bars"]),
                reason=f"боковик (ADX {adx_value:.1f}), {reason}",
                features=features,
            )
        ]
