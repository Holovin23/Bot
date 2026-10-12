"""Хранилище рыночных данных в Parquet: один файл на символ (и таймфрейм) и месяц.

Пути (от папки data/):
    klines/BTCUSDT/1m/2024-03.parquet          свечи; ключ — open_time
    funding/BTCUSDT/2024-03.parquet            funding; ключ — time
    open_interest/BTCUSDT/5min/2024-03.parquet open interest; ключ — time

Все времена — pandas datetime64[ms, UTC]. Повторная запись тех же строк не создаёт
дублей: строка с тем же ключом заменяется новой.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

CANDLE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "turnover"]
FUNDING_COLUMNS = ["time", "funding_rate"]
OPEN_INTEREST_COLUMNS = ["time", "open_interest"]

TIME_DTYPE = "datetime64[ms, UTC]"


def _to_utc_timestamp(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"Время без таймзоны: {value!r}")
    return ts.tz_convert("UTC")


def _normalize(df: pd.DataFrame, columns: list[str], key: str) -> pd.DataFrame:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Не хватает колонок: {sorted(missing)}")
    out = df[columns].copy()
    if out[key].dt.tz is None:
        raise ValueError(f"Колонка {key} должна быть с таймзоной UTC")
    out[key] = out[key].dt.tz_convert("UTC").astype(TIME_DTYPE)
    for column in columns:
        if column != key:
            out[column] = out[column].astype("float64")
    return out


class _MonthlyParquet:
    """Набор месячных Parquet-файлов в одной папке с ключевой колонкой времени."""

    def __init__(self, folder: Path, columns: list[str], key: str) -> None:
        self.folder = folder
        self.columns = columns
        self.key = key

    def _path(self, month: pd.Period) -> Path:
        return self.folder / f"{month.strftime('%Y-%m')}.parquet"

    def _read_file(self, path: Path) -> pd.DataFrame:
        return _normalize(pd.read_parquet(path), self.columns, self.key)

    def _empty(self) -> pd.DataFrame:
        return _normalize(
            pd.DataFrame({c: pd.Series(dtype=TIME_DTYPE if c == self.key else "float64")
                          for c in self.columns}),
            self.columns,
            self.key,
        )

    def write(self, df: pd.DataFrame) -> int:
        """Дописать строки. Возвращает число новых строк (без учёта заменённых)."""
        if df.empty:
            return 0
        new = _normalize(df, self.columns, self.key)
        self.folder.mkdir(parents=True, exist_ok=True)
        added = 0
        months = new[self.key].dt.tz_localize(None).dt.to_period("M")
        for month, part in new.groupby(months):
            path = self._path(pd.Period(month, freq="M"))
            old = self._read_file(path) if path.exists() else self._empty()
            merged = pd.concat([old, part], ignore_index=True)
            merged = merged.drop_duplicates(subset=self.key, keep="last")
            merged = merged.sort_values(self.key, ignore_index=True)
            added += len(merged) - len(old)
            tmp = path.with_suffix(".parquet.tmp")
            merged.to_parquet(tmp, index=False)
            os.replace(tmp, path)  # атомарная замена: файл не повредится при сбое
        return added

    def read(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        """Строки с key в [start, end). Без границ — всё, что есть."""
        start_ts = _to_utc_timestamp(start) if start is not None else None
        end_ts = _to_utc_timestamp(end) if end is not None else None
        frames = []
        for path in sorted(self.folder.glob("*.parquet")):
            month = pd.Period(path.stem, freq="M")
            month_start = month.start_time.tz_localize("UTC")
            month_end = (month + 1).start_time.tz_localize("UTC")
            if start_ts is not None and month_end <= start_ts:
                continue
            if end_ts is not None and month_start >= end_ts:
                continue
            frames.append(self._read_file(path))
        if not frames:
            return self._empty()
        df = pd.concat(frames, ignore_index=True)
        if start_ts is not None:
            df = df[df[self.key] >= start_ts]
        if end_ts is not None:
            df = df[df[self.key] < end_ts]
        return df.sort_values(self.key, ignore_index=True)

    def last_time(self) -> pd.Timestamp | None:
        files = sorted(self.folder.glob("*.parquet"))
        for path in reversed(files):
            df = self._read_file(path)
            if not df.empty:
                return pd.Timestamp(df[self.key].max())
        return None


class Storage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _klines(self, symbol: str, timeframe: str) -> _MonthlyParquet:
        return _MonthlyParquet(self.root / "klines" / symbol / timeframe, CANDLE_COLUMNS, "open_time")

    def _funding(self, symbol: str) -> _MonthlyParquet:
        return _MonthlyParquet(self.root / "funding" / symbol, FUNDING_COLUMNS, "time")

    def _open_interest(self, symbol: str, interval: str) -> _MonthlyParquet:
        return _MonthlyParquet(
            self.root / "open_interest" / symbol / interval, OPEN_INTEREST_COLUMNS, "time"
        )

    # Свечи
    def write_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        return self._klines(symbol, timeframe).write(df)

    def read_candles(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        return self._klines(symbol, timeframe).read(start, end)

    def last_candle_time(self, symbol: str, timeframe: str) -> pd.Timestamp | None:
        return self._klines(symbol, timeframe).last_time()

    # Funding
    def write_funding(self, symbol: str, df: pd.DataFrame) -> int:
        return self._funding(symbol).write(df)

    def read_funding(
        self, symbol: str, start: datetime | str | None = None, end: datetime | str | None = None
    ) -> pd.DataFrame:
        return self._funding(symbol).read(start, end)

    def last_funding_time(self, symbol: str) -> pd.Timestamp | None:
        return self._funding(symbol).last_time()

    # Open interest
    def write_open_interest(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        return self._open_interest(symbol, interval).write(df)

    def read_open_interest(
        self,
        symbol: str,
        interval: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> pd.DataFrame:
        return self._open_interest(symbol, interval).read(start, end)

    def last_open_interest_time(self, symbol: str, interval: str) -> pd.Timestamp | None:
        return self._open_interest(symbol, interval).last_time()
