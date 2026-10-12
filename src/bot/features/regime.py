"""Режим рынка правилами (docs/02-architecture.md, модуль 3, слой A). Базовая линия,
с которой сравнивается любой ИИ.

    HIGH_VOL   — ATR% выше 90-го перцентиля за последние 500 свечей;
    TREND_UP   — ADX > 25, close > EMA50, EMA50 выше, чем 10 свечей назад;
    TREND_DOWN — ADX > 25, close < EMA50, EMA50 ниже, чем 10 свечей назад;
    RANGE      — всё остальное.
Все значения считаются только по прошлым свечам.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd

from bot.features.indicators import adx, atr, ema, percentile_rank

VOL_LOOKBACK = 500
VOL_PERCENTILE = 0.9
ADX_TREND = 25.0
EMA_LENGTH = 50
SLOPE_BARS = 10


class Regime(IntEnum):
    """Числовой код: признаки стратегий и ML должны быть числами."""

    UNKNOWN = -1  # мало истории
    RANGE = 0
    TREND_UP = 1
    TREND_DOWN = 2
    HIGH_VOL = 3


def classify(
    atr_rank: pd.Series,
    adx_value: pd.Series,
    close: pd.Series,
    ema_value: pd.Series,
    ema_slope: pd.Series,
) -> pd.Series:
    """Режим по готовым индикаторам. NaN в любом входе → UNKNOWN."""
    known = ~(atr_rank.isna() | adx_value.isna() | ema_value.isna() | ema_slope.isna())
    trend = adx_value > ADX_TREND
    conditions = [
        ~known,
        atr_rank > VOL_PERCENTILE,
        trend & (close > ema_value) & (ema_slope > 0),
        trend & (close < ema_value) & (ema_slope < 0),
    ]
    choices = [Regime.UNKNOWN, Regime.HIGH_VOL, Regime.TREND_UP, Regime.TREND_DOWN]
    codes = np.select(conditions, [int(c) for c in choices], default=int(Regime.RANGE))
    return pd.Series(codes, index=close.index, name="regime")


def regime_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Индикаторы режима и сам режим для каждой свечи."""
    close = bars["close"]
    atr_pct = atr(bars, 14) / close
    ema50 = ema(close, EMA_LENGTH)
    frame = pd.DataFrame(
        {
            "atr_pct": atr_pct,
            "atr_rank": percentile_rank(atr_pct, VOL_LOOKBACK),
            "adx": adx(bars, 14)["adx"],
            "ema50": ema50,
            "ema50_slope": ema50 - ema50.shift(SLOPE_BARS),
        },
        index=bars.index,
    )
    frame["regime"] = classify(
        frame["atr_rank"], frame["adx"], close, frame["ema50"], frame["ema50_slope"]
    )
    return frame
