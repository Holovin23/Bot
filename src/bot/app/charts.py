"""Данные для графиков веб-интерфейса: свечи нужного таймфрейма и индикаторы.

Окно графика задаётся либо «последние limit свечей» (с until — до этого момента), либо
периодом start…end (даты, выбранные в интерфейсе). Слева всегда берётся запас свечей
(разогрев), чтобы индикаторы на первой видимой свече уже были посчитаны правильно.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import pandas as pd

from bot.core.timeframes import timeframe_minutes
from bot.data.resample import resample_candles
from bot.data.storage import Storage
from bot.features.indicators import bollinger, ema
from bot.strategies.registry import create_strategy

EMA_LENGTHS = (20, 50, 200)
# Лишние свечи слева, чтобы EMA200 «разогрелась» и совпадала с графиком Bybit.
WARMUP_BARS = 1000
MAX_BARS = 20_000  # больше браузер рисует медленно: для длинного периода — крупнее таймфрейм
INDICATOR_COLUMNS = ["ema20", "ema50", "ema200", "bb_lower", "bb_middle", "bb_upper"]


def _clean(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else round(number, 8)


def chart_bars(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Свечи для графика: как resample_candles, но последняя, ещё не закрытая свеча
    таймфрейма (например, текущий час) остаётся — как на графике биржи. Стратегии
    этой функцией не пользуются: они видят только закрытые свечи."""
    bars = resample_candles(df_1m, timeframe)
    if df_1m.empty or timeframe == "1m":
        return bars
    period = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    first = df_1m["open_time"].min()
    tail_start = bars["open_time"].iloc[-1] + period if len(bars) else first.ceil(period)
    tail = df_1m[df_1m["open_time"] >= tail_start]
    if tail.empty:
        return bars
    partial = (
        tail.groupby(tail["open_time"].dt.floor(period), sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            turnover=("turnover", "sum"),
        )
        .rename_axis("open_time")
        .reset_index()
    )
    result: pd.DataFrame = pd.concat([bars, partial[bars.columns]], ignore_index=True)
    return result


def chart_window(
    storage: Storage,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    until: int | None = None,
    start: int | None = None,
    end: int | None = None,
    warmup: int = WARMUP_BARS,
) -> tuple[pd.DataFrame, int, bool]:
    """Свечи с разогревом слева, сколько из них показывать (хвост) и обрезан ли период.

    start/end/until — секунды UTC. С start показывается период [start, end); без end —
    до последней сохранённой свечи."""
    minutes = timeframe_minutes(timeframe)
    step = f"{minutes}min"
    last = storage.last_candle_time(symbol, "1m")
    if last is None:
        return pd.DataFrame(), 0, False
    truncated = False
    if start is not None:
        begin = pd.Timestamp(start, unit="s", tz="UTC").floor(step)
        stop = (
            pd.Timestamp(end, unit="s", tz="UTC").ceil(step)
            if end is not None
            else last + pd.Timedelta(minutes=1)
        )
        wanted = max(int((stop - begin) / pd.Timedelta(minutes=minutes)), 1)
        if wanted > MAX_BARS:
            begin, truncated = stop - pd.Timedelta(minutes=minutes * MAX_BARS), True
            wanted = MAX_BARS
        read_from = begin - pd.Timedelta(minutes=minutes * warmup)
        df = chart_bars(storage.read_candles(symbol, "1m", start=read_from, end=stop), timeframe)
        visible = int((df["open_time"] >= begin).sum()) if len(df) else 0
        return df, visible, truncated

    stop_ts = None
    if until is not None:
        stop_ts = pd.Timestamp(until, unit="s", tz="UTC").ceil(step)
        last = min(last, stop_ts - pd.Timedelta(minutes=1))
    read_from = (last - pd.Timedelta(minutes=minutes * (limit + warmup))).floor(step)
    df = chart_bars(storage.read_candles(symbol, "1m", start=read_from, end=stop_ts), timeframe)
    return df, min(limit, len(df)), truncated


def _candles(df: pd.DataFrame, extra: list[str]) -> list[dict[str, Any]]:
    epoch = pd.Timestamp(0, tz="UTC")  # единица хранения времени (мс или нс) не важна
    times = ((df["open_time"] - epoch) // pd.Timedelta(seconds=1)).tolist()
    out: list[dict[str, Any]] = []
    columns = ["open", "high", "low", "close", "volume", *extra]
    records = df[columns].to_dict(orient="records")
    for t, record in zip(times, records, strict=True):
        item: dict[str, Any] = {"time": int(t)}
        for key in ("open", "high", "low", "close", "volume"):
            item[key] = float(record[key])
        for key in extra:
            item[key] = _clean(record[key])
        out.append(item)
    return out


def _meta(storage: Storage, symbol: str, timeframe: str, truncated: bool) -> dict[str, Any]:
    last_1m = storage.last_candle_time(symbol, "1m")
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        # время открытия последней сохранённой 1m-свечи (секунды UTC) — насколько свежие данные
        "last_minute": int(last_1m.timestamp()) if last_1m is not None else None,
        "truncated": truncated,
    }


def build_chart_data(
    storage: Storage,
    symbol: str,
    timeframe: str,
    limit: int,
    until: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> dict[str, Any]:
    """Общий график: свечи, EMA 20/50/200 и Bollinger (20, 2)."""
    df, visible, truncated = chart_window(storage, symbol, timeframe, limit, until, start, end)
    meta = _meta(storage, symbol, timeframe, truncated)
    if df.empty or visible == 0:
        return {**meta, "candles": []}
    for length in EMA_LENGTHS:
        df[f"ema{length}"] = ema(df["close"], length)
    df = pd.concat([df, bollinger(df["close"], 20, 2.0)], axis=1)
    return {**meta, "candles": _candles(df.tail(visible), INDICATOR_COLUMNS)}


def build_strategy_chart(
    storage: Storage,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    params: dict[str, Any] | None = None,
    limit: int = 500,
    until: int | None = None,
    start: int | None = None,
    end: int | None = None,
) -> dict[str, Any]:
    """График стратегии: свечи и её собственные индикаторы (prepare + chart_series).
    Признаки считаются тем же кодом, что в бэктесте и у бота."""
    strategy = create_strategy(strategy_name, timeframe, params)
    warmup = max(WARMUP_BARS, 2 * strategy.required_history())
    df, visible, truncated = chart_window(
        storage, symbol, timeframe, limit, until, start, end, warmup
    )
    specs = strategy.chart_series()
    meta = {
        **_meta(storage, symbol, timeframe, truncated),
        "strategy": strategy.name,
        "params": strategy.params,
    }
    series_meta = [{k: v for k, v in asdict(s).items()} for s in specs]
    if df.empty or visible == 0:
        return {**meta, "candles": [], "series": [{**m, "values": []} for m in series_meta]}
    bars = df.reset_index(drop=True)
    features = strategy.prepare(bars).reset_index(drop=True)
    tail = slice(len(bars) - visible, len(bars))
    series = []
    for spec, info in zip(specs, series_meta, strict=True):
        source = features if spec.column in features.columns else bars
        if spec.column not in source.columns:
            continue
        values = source[spec.column].iloc[tail] * spec.scale
        series.append({**info, "values": [_clean(v) for v in values.tolist()]})
    return {**meta, "candles": _candles(bars.iloc[tail], []), "series": series}
