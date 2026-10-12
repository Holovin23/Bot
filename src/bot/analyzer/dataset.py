"""Набор данных для ML-фильтра: признаки на закрытии сигнальной свечи и метка сделки.

Метка — тройной барьер с барьерами самой стратегии (стоп, тейк, выход по времени):
1, если сделка бэктеста закрылась с R > 0, иначе 0. Признаки считаются только по
свечам до сигнальной включительно (проверяется обрезкой в тестах).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot.backtest.engine import BacktestResult
from bot.core.models import Side
from bot.core.timeframes import timeframe_delta
from bot.data.resample import resample_candles
from bot.features.indicators import bollinger_width, rsi
from bot.features.regime import regime_frame

FEATURE_COLUMNS = [
    "regime",
    "adx",
    "rsi",
    "atr_pct",
    "bb_width",
    "ret_1",
    "ret_6",
    "ret_24",
    "volume_ratio",
    "ema50_slope_pct",
    "hour",
    "side",
]


def market_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Признаки рынка для каждой свечи (без стороны сделки)."""
    close = bars["close"]
    regime = regime_frame(bars)
    volume = bars["volume"]
    avg_volume = volume.shift(1).rolling(20).mean()
    return pd.DataFrame(
        {
            "regime": regime["regime"].astype("float64"),
            "adx": regime["adx"],
            "rsi": rsi(close, 14),
            "atr_pct": regime["atr_pct"] * 100,
            "bb_width": bollinger_width(close, 20, 2.0),
            "ret_1": close.pct_change(1) * 100,
            "ret_6": close.pct_change(6) * 100,
            "ret_24": close.pct_change(24) * 100,
            "volume_ratio": volume / avg_volume.replace(0, np.nan),
            "ema50_slope_pct": regime["ema50_slope"] / close * 100,
            "hour": pd.to_datetime(bars["open_time"], utc=True).dt.hour.astype("float64"),
        },
        index=bars.index,
    )


def build_dataset(result: BacktestResult, candles_1m: pd.DataFrame) -> pd.DataFrame:
    """Строка на сделку: признаки FEATURE_COLUMNS, label, r_multiple, signal_time, exit_time."""
    columns = [*FEATURE_COLUMNS, "label", "r_multiple", "signal_time", "exit_time"]
    if not result.trades:
        return pd.DataFrame(columns=columns)
    bars = resample_candles(candles_1m, result.timeframe)
    features = market_features(bars)
    features.index = pd.DatetimeIndex(bars["open_time"])
    step = timeframe_delta(result.timeframe)
    rows = []
    for t in result.trades:
        bar_time = t.signal_time - step  # открытие сигнальной свечи
        if bar_time not in features.index:
            continue
        row = features.loc[bar_time].to_dict()
        row["side"] = 1.0 if t.side is Side.LONG else -1.0
        row["label"] = int(t.r_multiple > 0)
        row["r_multiple"] = t.r_multiple
        row["signal_time"] = t.signal_time
        row["exit_time"] = t.exit_time
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
