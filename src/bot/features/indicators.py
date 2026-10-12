"""Индикаторы. Формулы как на TradingView (графики Bybit построены на нём).

Каждое значение в строке i считается только по строкам 0..i — без заглядывания вперёд.
Пока данных не хватает, значение NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rma(values: pd.Series, length: int) -> pd.Series:
    """Сглаживание Уайлдера (RMA): первое значение — простое среднее первых length
    значений, дальше rma = (предыдущее * (length - 1) + текущее) / length."""
    data = values.to_numpy(dtype="float64")
    out = np.full(len(data), np.nan)
    valid = np.flatnonzero(~np.isnan(data))
    if len(valid) < length:
        return pd.Series(out, index=values.index)
    start = valid[0]
    seed_end = start + length
    if np.isnan(data[start:seed_end]).any():
        raise ValueError("Пропуски внутри данных для RMA")
    out[seed_end - 1] = data[start:seed_end].mean()
    for i in range(seed_end, len(data)):
        out[i] = (out[i - 1] * (length - 1) + data[i]) / length
    return pd.Series(out, index=values.index)


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
