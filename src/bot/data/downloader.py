"""Загрузка истории с Bybit v5: 1m-свечи, funding, open interest.

Запуск:
    uv run python -m bot.data.downloader --symbol BTCUSDT --from 2021-01-01

Повторный запуск продолжает с последней сохранённой записи; дублей не будет
(хранилище заменяет строки с тем же временем). Сохраняется только закрытая история:
текущая незакрытая свеча отбрасывается.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd
import requests
from pybit.exceptions import FailedRequestError

from bot.core.config import load_config
from bot.core.logging import get_logger, setup_logging
from bot.core.timeframes import bybit_interval, timeframe_minutes
from bot.data.bybit_public import make_public_client
from bot.data.storage import Storage

log = get_logger(__name__)

MINUTE_MS = 60_000
KLINE_LIMIT = 1000  # максимум свечей за запрос /v5/market/kline
FUNDING_LIMIT = 200  # максимум записей /v5/market/funding/history
OI_LIMIT = 200  # максимум записей /v5/market/open-interest
OI_INTERVALS = {"5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240}

# Лимит Bybit — 600 запросов за 5 секунд с одного IP. Держимся далеко ниже: 10 в секунду.
MIN_REQUEST_INTERVAL_S = 0.1
MAX_ATTEMPTS = 6
IP_BAN_PAUSE_S = 600  # при HTTP 403 Bybit просит подождать не меньше 10 минут


class MarketClient(Protocol):
    def get_kline(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_funding_rate_history(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_open_interest(self, **kwargs: Any) -> dict[str, Any]: ...


def to_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def from_ms(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC")


class RestCaller:
    """Вызов REST с паузой между запросами и повторами при сетевых ошибках."""

    def __init__(
        self,
        min_interval_s: float = MIN_REQUEST_INTERVAL_S,
        max_attempts: int = MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.clock = clock
        self._last_call = float("-inf")

    def call(self, fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        for attempt in range(1, self.max_attempts + 1):
            wait = self._last_call + self.min_interval_s - self.clock()
            if wait > 0:
                self.sleep(wait)
            self._last_call = self.clock()
            try:
                return fn(**kwargs)
            except (FailedRequestError, requests.RequestException) as error:
                if attempt == self.max_attempts:
                    raise
                status = getattr(error, "status_code", None)
                pause = IP_BAN_PAUSE_S if status == 403 else min(2.0**attempt, 60.0)
                log.warning(
                    "request_failed_retrying", attempt=attempt, pause_s=pause, error=str(error)
                )
                self.sleep(pause)
        raise AssertionError("unreachable")


def parse_klines(response: dict[str, Any]) -> pd.DataFrame:
    rows = response["result"]["list"]
    df = pd.DataFrame(
        rows, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"]
    )
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        df[column] = df[column].astype("float64")
    return df.sort_values("open_time", ignore_index=True)


def parse_funding(response: dict[str, Any]) -> pd.DataFrame:
    rows = response["result"]["list"]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                [int(r["fundingRateTimestamp"]) for r in rows], unit="ms", utc=True
            ),
            "funding_rate": [float(r["fundingRate"]) for r in rows],
        }
    ).sort_values("time", ignore_index=True)


def parse_open_interest(response: dict[str, Any]) -> pd.DataFrame:
    rows = response["result"]["list"]
    return pd.DataFrame(
        {
            "time": pd.to_datetime([int(r["timestamp"]) for r in rows], unit="ms", utc=True),
            "open_interest": [float(r["openInterest"]) for r in rows],
        }
    ).sort_values("time", ignore_index=True)


def missing_spans(
    have: pd.Series, start: pd.Timestamp, end: pd.Timestamp, step: pd.Timedelta
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Участки [a, b) внутри [start, end), где нет ни одной свечи. Соседние участки,
    между которыми меньше KLINE_LIMIT свечей, склеиваются: так меньше запросов, а
    повторно скачанные свечи хранилище просто заменит."""
    times = pd.DatetimeIndex(have).sort_values()
    times = times[(times >= start) & (times < end)]
    # Границы: «свеча» до start и после end, чтобы края тоже считались пропусками.
    edges = pd.DatetimeIndex([start - step]).append(times).append(pd.DatetimeIndex([end]))
    gap_starts = edges[:-1] + step
    gap_ends = edges[1:]
    holes = gap_ends > gap_starts
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for a, b in zip(gap_starts[holes], gap_ends[holes], strict=True):
        if spans and a - spans[-1][1] < step * KLINE_LIMIT:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


class Downloader:
    def __init__(
        self,
        client: MarketClient,
        storage: Storage,
        caller: RestCaller | None = None,
        now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
        category: str = "linear",
        progress_every: int = 20,
        say: Callable[[str], None] = print,
    ) -> None:
        self.client = client
        self.storage = storage
        self.caller = caller or RestCaller()
        self.now = now
        self.category = category
        self.progress_every = progress_every
        self.say = say  # куда писать ход загрузки (консоль или задача веб-интерфейса)
        self._listing: dict[str, pd.Timestamp | None] = {}

    def download_klines(self, symbol: str, start: pd.Timestamp, timeframe: str = "1m") -> int:
        """Свечи от start (или от последней сохранённой) до последней закрытой."""
        step = timeframe_minutes(timeframe)
        last = self.storage.last_candle_time(symbol, timeframe)
        begin = max(start, last + pd.Timedelta(minutes=step)) if last is not None else start
        # Свеча закрыта, если open_time + длительность <= сейчас.
        cutoff = self.now().floor(f"{step}min")  # open_time текущей (незакрытой) свечи
        self.say(f"{symbol} {timeframe}: загрузка с {begin:%Y-%m-%d %H:%M} UTC")
        total = self._klines_window(symbol, timeframe, begin, cutoff)
        self.say(f"{symbol} {timeframe}: готово, новых свечей {total}")
        return total

    def _klines_window(
        self, symbol: str, timeframe: str, begin: pd.Timestamp, stop: pd.Timestamp
    ) -> int:
        """Свечи с open_time в [begin, stop); stop не позже текущей незакрытой свечи."""
        step = timeframe_minutes(timeframe)
        window_ms = KLINE_LIMIT * step * MINUTE_MS
        cursor = to_ms(begin)
        total, requests_done = 0, 0
        while cursor < to_ms(stop):
            end = min(cursor + window_ms, to_ms(stop)) - 1
            response = self.caller.call(
                self.client.get_kline,
                category=self.category,
                symbol=symbol,
                interval=bybit_interval(timeframe),
                start=cursor,
                end=end,
                limit=KLINE_LIMIT,
            )
            df = parse_klines(response)
            df = df[(df["open_time"] >= from_ms(cursor)) & (df["open_time"] < stop)]
            total += self.storage.write_candles(symbol, timeframe, df)
            cursor = end + 1
            requests_done += 1
            if requests_done % self.progress_every == 0:
                self._progress(symbol, timeframe, begin, from_ms(cursor), stop, total)
        return total

    def listing_time(self, symbol: str) -> pd.Timestamp | None:
        """Когда инструмент появился на Bybit (launchTime). None — неизвестно."""
        if symbol not in self._listing:
            value: pd.Timestamp | None = None
            get_info = getattr(self.client, "get_instruments_info", None)
            if get_info is not None:
                try:
                    response = self.caller.call(get_info, category=self.category, symbol=symbol)
                    items = response["result"]["list"]
                    if items and items[0].get("launchTime"):
                        value = from_ms(int(items[0]["launchTime"]))
                except (FailedRequestError, requests.RequestException, KeyError, ValueError):
                    value = None
            self._listing[symbol] = value
        return self._listing[symbol]

    def fill_klines(
        self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str = "1m"
    ) -> int:
        """Догрузить все пропуски в [start, end) — и в начале, и в середине, и в конце.

        download_klines продолжает только с последней сохранённой свечи: если сначала
        загрузились свежие дни (например, графиком), старую историю он уже не догрузит.
        """
        step = pd.Timedelta(minutes=timeframe_minutes(timeframe))
        listed = self.listing_time(symbol)
        begin = max(start, listed.ceil(step)) if listed is not None else start
        stop = min(end, self.now().floor(step))
        if begin >= stop:
            return 0
        have = self.storage.read_candles(symbol, timeframe, begin, stop)["open_time"]
        spans = missing_spans(have, begin, stop, step)
        if not spans:
            return 0
        missing = sum((b - a) // step for a, b in spans)
        self.say(
            f"{symbol} {timeframe}: не хватает {missing} свечей за {begin:%Y-%m-%d} — "
            f"{stop:%Y-%m-%d}, загружаю ({len(spans)} участков)"
        )
        total = 0
        for a, b in spans:
            total += self._klines_window(symbol, timeframe, a, b)
        self.say(f"{symbol} {timeframe}: готово, новых свечей {total}")
        return total

    def fill_funding(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """Funding за [start, end): страницы назад от end (повторы хранилище заменит)."""
        end_ms = to_ms(min(end, self.now())) - 1
        pages = []
        while end_ms >= to_ms(start):
            response = self.caller.call(
                self.client.get_funding_rate_history,
                category=self.category,
                symbol=symbol,
                endTime=end_ms,
                limit=FUNDING_LIMIT,
            )
            df = parse_funding(response)
            df = df[df["time"] >= start]
            if df.empty:
                break
            pages.append(df)
            if len(response["result"]["list"]) < FUNDING_LIMIT:
                break
            end_ms = to_ms(df["time"].min()) - 1
        total = 0
        if pages:
            total = self.storage.write_funding(symbol, pd.concat(pages, ignore_index=True))
        self.say(f"{symbol} funding: готово, новых записей {total}")
        return total

    def ensure_history(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
        """1m-свечи и funding за период — всё, что нужно бэктесту и боту."""
        added = self.fill_klines(symbol, start, end, "1m")
        self.fill_funding(symbol, start, end)
        return added

    def download_funding(self, symbol: str, start: pd.Timestamp) -> int:
        """Funding по фактическим временам. Bybit отдаёт историю страницами назад от endTime,
        поэтому сначала собираем всё в памяти и пишем один раз (прерывание не оставит дыр)."""
        last = self.storage.last_funding_time(symbol)
        begin = max(start, last + pd.Timedelta(milliseconds=1)) if last is not None else start
        end_ms = to_ms(self.now())
        pages = []
        while True:
            response = self.caller.call(
                self.client.get_funding_rate_history,
                category=self.category,
                symbol=symbol,
                endTime=end_ms,  # только endTime: Bybit отдаёт 200 записей до этого времени
                limit=FUNDING_LIMIT,
            )
            df = parse_funding(response)
            df = df[df["time"] >= begin]
            if df.empty:
                break
            pages.append(df)
            oldest = to_ms(df["time"].min())
            if len(response["result"]["list"]) < FUNDING_LIMIT or oldest <= to_ms(begin):
                break
            end_ms = oldest - 1
        total = 0
        if pages:
            total = self.storage.write_funding(symbol, pd.concat(pages, ignore_index=True))
        self.say(f"{symbol} funding: готово, новых записей {total}")
        return total

    def download_open_interest(
        self, symbol: str, start: pd.Timestamp, interval: str = "5min"
    ) -> int:
        step = OI_INTERVALS[interval]
        last = self.storage.last_open_interest_time(symbol, interval)
        begin = max(start, last + pd.Timedelta(minutes=step)) if last is not None else start
        cutoff = self.now().floor(f"{step}min")
        window_ms = OI_LIMIT * step * MINUTE_MS
        cursor = to_ms(begin)
        total, requests_done = 0, 0
        first_seen: pd.Timestamp | None = None
        self.say(f"{symbol} open interest {interval}: загрузка с {begin:%Y-%m-%d %H:%M} UTC")
        while cursor < to_ms(cutoff):
            end = min(cursor + window_ms, to_ms(cutoff)) - 1
            response = self.caller.call(
                self.client.get_open_interest,
                category=self.category,
                symbol=symbol,
                intervalTime=interval,
                startTime=cursor,
                endTime=end,
                limit=OI_LIMIT,
            )
            df = parse_open_interest(response)
            df = df[(df["time"] >= from_ms(cursor)) & (df["time"] <= from_ms(end))]
            if first_seen is None and not df.empty:
                first_seen = df["time"].min()
                if last is None and first_seen > begin + pd.Timedelta(days=1):
                    self.say(f"{symbol} open interest: история начинается только с {first_seen}")
            total += self.storage.write_open_interest(symbol, interval, df)
            cursor = end + 1
            requests_done += 1
            if requests_done % self.progress_every == 0:
                self._progress(symbol, f"OI {interval}", begin, from_ms(cursor), cutoff, total)
        self.say(f"{symbol} open interest {interval}: готово, новых записей {total}")
        return total

    def _progress(
        self,
        symbol: str,
        what: str,
        begin: pd.Timestamp,
        position: pd.Timestamp,
        cutoff: pd.Timestamp,
        total: int,
    ) -> None:
        span = (cutoff - begin).total_seconds() or 1.0
        percent = 100 * (position - begin).total_seconds() / span
        self.say(f"  {symbol} {what}: до {position:%Y-%m-%d %H:%M} UTC ({percent:5.1f}%), +{total}")


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Загрузка истории Bybit в data/")
    parser.add_argument("--symbol", action="append", help="символ; можно несколько раз")
    parser.add_argument("--from", dest="start", default=cfg.data.history_start)
    parser.add_argument("--to", dest="end", help="по какую дату включительно (иначе — сейчас)")
    parser.add_argument(
        "--only",
        choices=["klines", "funding", "oi"],
        action="append",
        help="загрузить только это (по умолчанию всё)",
    )
    args = parser.parse_args(argv)

    setup_logging(cfg.resolve(cfg.logging.dir), cfg.logging.level, cfg.logging.retention_days)
    storage = Storage(cfg.resolve(cfg.data.dir))
    downloader = Downloader(make_public_client(), storage, category=cfg.data.category)
    start = pd.Timestamp(args.start, tz="UTC")
    now = pd.Timestamp.now(tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) if args.end else now
    parts = args.only or ["klines", "funding", "oi"]

    for symbol in args.symbol or cfg.data.symbols:
        # fill_*: догружаются все пропуски периода, а не только хвост после последней свечи
        if "klines" in parts:
            downloader.fill_klines(symbol, start, end, cfg.data.base_interval)
        if "funding" in parts:
            downloader.fill_funding(symbol, start, end)
        if "oi" in parts:
            downloader.download_open_interest(symbol, start, cfg.data.open_interest_interval)


if __name__ == "__main__":
    main()
