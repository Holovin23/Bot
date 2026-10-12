"""Базовая линия «случайные входы».

На каждой свече с вероятностью probability открывается сделка в случайную сторону
со стопом stop_atr × ATR(14), тейком tp_atr × ATR и выходом по времени.
Случайность зависит только от seed и времени свечи, поэтому результат воспроизводим
и не зависит от того, сколько данных после свечи загружено.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from bot.core.models import Side, Signal
from bot.features.indicators import atr
from bot.strategies.base import BaseStrategy, MarketContext

DEFAULT_PARAMS: dict[str, Any] = {
    "probability": 0.05,
    "stop_atr": 1.5,
    "tp_atr": 2.0,
    "time_stop_bars": 24,
    "atr_length": 14,
    "seed": 1,
}


class RandomEntry(BaseStrategy):
    name = "random_entry"
    version = "1.0.0"

    def __init__(self, timeframe: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(timeframe, {**DEFAULT_PARAMS, **(params or {})})

    def required_history(self) -> int:
        return int(self.params["atr_length"]) + 1

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"atr": atr(bars, int(self.params["atr_length"]))})

    def _draw(self, ctx: MarketContext) -> tuple[float, float]:
        """Два «случайных» числа в [0, 1), зависящих только от seed и времени свечи."""
        key = f"{self.params['seed']}:{ctx.time.isoformat()}".encode()
        digest = hashlib.blake2b(key, digest_size=16).digest()
        enter = int.from_bytes(digest[:8], "big") / 2**64
        direction = int.from_bytes(digest[8:], "big") / 2**64
        return enter, direction

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        value = ctx.value("atr")
        if not value > 0:
            return []
        enter, direction = self._draw(ctx)
        if enter >= float(self.params["probability"]):
            return []
        side = Side.LONG if direction < 0.5 else Side.SHORT
        sign = 1.0 if side is Side.LONG else -1.0
        close = ctx.value("close")
        stop = close - sign * float(self.params["stop_atr"]) * value
        tp_atr = self.params["tp_atr"]
        take = None if tp_atr is None else close + sign * float(tp_atr) * value
        return [
            self.signal(
                ctx,
                side,
                stop_loss=stop,
                take_profit=take,
                time_stop_bars=int(self.params["time_stop_bars"]),
                reason=f"случайный вход, ATR {value:.2f}",
                features={"atr": value},
            )
        ]
