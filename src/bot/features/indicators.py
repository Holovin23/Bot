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


def rsi(values: pd.Series, length: int = 14) -> pd.Series:
    """RSI Уайлдера (как ta.rsi в Pine): 100 − 100 / (1 + RMA(рост) / RMA(падение)).
    Нет падений — 100, нет ростов — 0."""
    change = values.diff()
    up = rma(change.clip(lower=0), length)
    down = rma((-change).clip(lower=0), length)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 100 - 100 / (1 + up / down)
    result = result.where(down != 0, 100.0).where(up != 0, 0.0)
    out: pd.Series = result.where(up.notna() & down.notna())
    return out


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ADX и +DI/−DI по Уайлдеру (как ta.dmi в Pine). Колонки: plus_di, minus_di, adx."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    plus_dm.iloc[0] = minus_dm.iloc[0] = np.nan  # у первой свечи нет предыдущей
    tr = true_range(df)
    tr.iloc[0] = np.nan
    smoothed_tr = rma(tr, length)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100 * rma(plus_dm, length) / smoothed_tr
        minus_di = 100 * rma(minus_dm, length) / smoothed_tr
    total = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / total.where(total != 0, 1.0)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": rma(dx, length)})


def bollinger_width(values: pd.Series, length: int = 20, std: float = 2.0) -> pd.Series:
    """Относительная ширина полос: (верх − низ) / средняя."""
    bands = bollinger(values, length, std)
    width: pd.Series = (bands["bb_upper"] - bands["bb_lower"]) / bands["bb_middle"]
    return width


def percentile_rank(values: pd.Series, length: int) -> pd.Series:
    """Доля значений среди последних length (включая текущее), которые <= текущего.
    1.0 — текущее самое большое, 1/length — самое маленькое."""
    data = values.to_numpy(dtype="float64")
    out = np.full(len(data), np.nan)
    chunk = 20_000
    for start in range(length - 1, len(data), chunk):
        stop = min(start + chunk, len(data))
        window = np.lib.stride_tricks.sliding_window_view(data[start - length + 1 : stop], length)
        current = window[:, -1:]
        rank = (window <= current).mean(axis=1)
        rank[np.isnan(window).any(axis=1)] = np.nan
        out[start:stop] = rank
    return pd.Series(out, index=values.index)


def rolling_max(values: pd.Series, length: int) -> pd.Series:
    result: pd.Series = values.rolling(length).max()
    return result


def rolling_min(values: pd.Series, length: int) -> pd.Series:
    result: pd.Series = values.rolling(length).min()
    return result
