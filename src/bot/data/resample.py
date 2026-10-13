"""Сборка старших таймфреймов (5m, 15m, 1h, 4h) из 1m-свечей.

open — первый, high — максимум, low — минимум, close — последний, volume и turnover —
сумма. Время свечи — время открытия. Границы периодов от 00:00 UTC (как на Bybit).
Неполные периоды на краях данных отбрасываются: незакрытая последняя свеча не должна
попасть в стратегию.
"""

from __future__ import annotations

import pandas as pd

from bot.core.timeframes import timeframe_minutes


def resample_candles(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    minutes = timeframe_minutes(timeframe)
    columns = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
    if df_1m.empty:
        return df_1m[columns].copy()
    df = df_1m.sort_values("open_time")
    if minutes == 1:
        return df[columns].reset_index(drop=True)

    period = pd.Timedelta(minutes=minutes)
    key = df["open_time"].dt.floor(period)
    grouped = df.groupby(key, sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        turnover=("turnover", "sum"),
    )
    grouped.index.name = "open_time"
    out = grouped.reset_index()

    first_minute = df["open_time"].iloc[0]
    last_minute = df["open_time"].iloc[-1]
    starts_inside = out["open_time"] >= first_minute
    ends_inside = out["open_time"] + period - pd.Timedelta(minutes=1) <= last_minute
    result: pd.DataFrame = out[starts_inside & ends_inside].reset_index(drop=True)[columns]
    return result


def with_funding_rate(
    bars: pd.DataFrame, funding: pd.DataFrame | None, timeframe: str
) -> pd.DataFrame:
    """Копия свечей с колонкой funding_rate: последняя выплаченная ставка funding,
    известная на закрытии свечи (время выплаты <= закрытия). Нет данных — NaN."""
    out = bars.copy()
    if funding is None or funding.empty or bars.empty:
        out["funding_rate"] = float("nan")
        return out
    period = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    closes = pd.DataFrame({"close_time": (out["open_time"] + period).astype("datetime64[ns, UTC]")})
    rates = funding[["time", "funding_rate"]].sort_values("time")
    rates = rates.assign(time=rates["time"].astype("datetime64[ns, UTC]"))
    merged = pd.merge_asof(closes, rates, left_on="close_time", right_on="time")
    out["funding_rate"] = merged["funding_rate"].to_numpy()
    return out
