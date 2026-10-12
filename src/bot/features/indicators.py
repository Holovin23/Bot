"""Индикаторы. Формулы как на TradingView (графики Bybit построены на нём).

Каждое значение в строке i считается только по строкам 0..i — без заглядывания вперёд.
Пока данных не хватает, значение NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _smooth(values: pd.Series, length: int, alpha: float) -> pd.Series:
    """Рекурсивное сглаживание с затравкой простым средним (как ta.ema / ta.rma в Pine):
    первое значение — среднее первых length значений,
    дальше s = alpha * текущее + (1 - alpha) * предыдущее s."""
    if length < 1:
        raise ValueError("length должен быть >= 1")
    data = values.to_numpy(dtype="float64")
    out = np.full(len(data), np.nan)
    valid = np.flatnonzero(~np.isnan(data))
    if len(valid) < length:
        return pd.Series(out, index=values.index)
    start = valid[0]
    seed_end = start + length
    if np.isnan(data[start:]).any():
        raise ValueError("Пропуски (NaN) внутри данных")
    out[seed_end - 1] = data[start:seed_end].mean()
    for i in range(seed_end, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=values.index)


def rma(values: pd.Series, length: int) -> pd.Series:
    """Сглаживание Уайлдера (для ATR, RSI, ADX): alpha = 1 / length."""
    return _smooth(values, length, 1 / length)


def ema(values: pd.Series, length: int) -> pd.Series:
    """Экспоненциальная средняя: alpha = 2 / (length + 1)."""
    return _smooth(values, length, 2 / (length + 1))


def sma(values: pd.Series, length: int) -> pd.Series:
    result: pd.Series = values.rolling(length).mean()
    return result


def bollinger(values: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Полосы Боллинджера: средняя — SMA, ширина — std * стандартное отклонение
    по генеральной совокупности (ddof=0, как на TradingView)."""
    middle = sma(values, length)
    deviation = values.rolling(length).std(ddof=0)
    return pd.DataFrame(
        {
            "bb_lower": middle - std * deviation,
            "bb_middle": middle,
            "bb_upper": middle + std * deviation,
        }
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    """TR = max(high - low, |high - prev_close|, |low - prev_close|).

    Для первой свечи — high - low."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    result: pd.Series = ranges.max(axis=1, skipna=True)
    return result


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return rma(true_range(df), length)
