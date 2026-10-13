"""Trend Following v1 (docs/09-start-checklist.md, «Правила стратегий v1»).

LONG: тренд — EMA50 > EMA200 и close > EMA200; откат — low <= EMA20 и close > EMA20.
Стоп: close − 2 × ATR(14). Тейка нет: выход по трейлингу Chandelier — максимум high
с момента входа − 3 × ATR(22), стоп только подтягивается. SHORT — зеркально.

Опции (по умолчанию выключены, пока проверяются только в бэктесте):
- entry=limit — вход лимитным ордером по close − limit_offset_atr × ATR (для SHORT
  выше close), действует одну свечу; комиссия maker вместо taker, без проскальзывания.
- funding_max > 0 — не открывать LONG, если последняя ставка funding выше funding_max
  (толпа уже в лонгах и платит за это), и SHORT, если ниже −funding_max.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bot.core.models import Position, Side, Signal
from bot.features.indicators import atr, ema
from bot.strategies.base import BaseStrategy, ChartSeries, MarketContext

DEFAULT_PARAMS: dict[str, Any] = {
    "ema_fast": 20,
    "ema_mid": 50,
    "ema_slow": 200,
    "atr_length": 14,
    "stop_atr": 2.0,
    "trail_atr_length": 22,
    "trail_atr": 3.0,
    "allow_short": True,
    "entry": "market",  # market | limit
    "limit_offset_atr": 0.0,
    "funding_max": 0.0,  # 0 — фильтр выключен; 0.0003 = 0,03% за 8 часов
}


class TrendFollowing(BaseStrategy):
    name = "trend_following"
    version = "1.0.0"

    def __init__(self, timeframe: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(timeframe, {**DEFAULT_PARAMS, **(params or {})})
        if self.params["entry"] not in ("market", "limit"):
            raise ValueError("entry должен быть market или limit")

    def required_history(self) -> int:
        p = self.params
        return int(max(p["ema_slow"], p["atr_length"], p["trail_atr_length"])) + 1

    def chart_series(self) -> list[ChartSeries]:
        p = self.params
        return [
            ChartSeries("ema_fast", f"EMA{p['ema_fast']}", color="#f5a623"),
            ChartSeries("ema_mid", f"EMA{p['ema_mid']}", color="#4a90e2"),
            ChartSeries("ema_slow", f"EMA{p['ema_slow']}", color="#bd10e0"),
            ChartSeries("atr", f"ATR({p['atr_length']}) — стоп {p['stop_atr']}×", "ATR"),
            ChartSeries("trail_atr", f"ATR({p['trail_atr_length']}) — трейлинг", "ATR", "#8b949e"),
        ]

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
                "funding_rate": bars["funding_rate"]
                if "funding_rate" in bars
                else pd.Series(float("nan"), index=bars.index),
            }
        )

    def backtest_only(self) -> list[str]:
        """Опции, которые живой бот пока не исполняет (лимитный вход, funding)."""
        out = []
        if self.params["entry"] != "market":
            out.append("лимитный вход (entry=limit)")
        if float(self.params["funding_max"]) > 0:
            out.append("фильтр по funding (funding_max)")
        return out

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        close, high, low = ctx.value("close"), ctx.value("high"), ctx.value("low")
        fast, mid, slow = ctx.value("ema_fast"), ctx.value("ema_mid"), ctx.value("ema_slow")
        value = ctx.value("atr")
        if not value > 0 or any(pd.isna(x) for x in (fast, mid, slow)):
            return []
        p = self.params
        try:
            funding = ctx.value("funding_rate")
        except KeyError:  # признаки посчитаны без funding (например, в тестах)
            funding = float("nan")
        features = {"ema20": fast, "ema50": mid, "ema200": slow, "atr": value}
        if not pd.isna(funding):
            features["funding_rate"] = funding
        stop_distance = float(p["stop_atr"]) * value
        offset = float(p["limit_offset_atr"]) * value
        funding_max = float(p["funding_max"])
        crowded_long = funding_max > 0 and funding > funding_max  # NaN — не фильтруем
        crowded_short = funding_max > 0 and funding < -funding_max

        if mid > slow and close > slow and low <= fast < close and not crowded_long:
            return [
                self.signal(
                    ctx,
                    Side.LONG,
                    stop_loss=close - stop_distance,
                    reason="тренд вверх (EMA50 > EMA200), откат к EMA20 и закрытие выше",
                    features=features,
                    limit_price=close - offset if p["entry"] == "limit" else None,
                )
            ]
        short_setup = mid < slow and close < slow and high >= fast > close
        if p["allow_short"] and short_setup and not crowded_short:
            return [
                self.signal(
                    ctx,
                    Side.SHORT,
                    stop_loss=close + stop_distance,
                    reason="тренд вниз (EMA50 < EMA200), откат к EMA20 и закрытие ниже",
                    features=features,
                    limit_price=close + offset if p["entry"] == "limit" else None,
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
