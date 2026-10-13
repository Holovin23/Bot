from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from bot.data.downloader import Downloader, RestCaller, missing_spans, to_ms
from bot.data.storage import Storage

MIN = 60_000
LISTED = pd.Timestamp("2024-01-01", tz="UTC")
NOW = pd.Timestamp("2024-01-03 12:34:30", tz="UTC")  # середина свечи 12:34


class FakeBybit:
    """Имитация публичного API: свеча на каждую минуту с LISTED до текущей (незакрытой)."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def _tick(self) -> None:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise requests.ConnectionError("сеть пропала")

    def get_kline(self, **kw: Any) -> dict[str, Any]:
        self._tick()
        first = max(kw["start"], to_ms(LISTED))
        last = min(kw["end"], to_ms(NOW))  # биржа отдаёт и незакрытую свечу
        times = [t for t in range(first - first % MIN, last + 1, MIN) if t >= kw["start"]]
        times = times[-kw["limit"] :][::-1]  # новые первыми
        rows = [[str(t), "100", "101", "99", "100.5", "2", "200"] for t in times]
        return {"result": {"list": rows}}

    def get_funding_rate_history(self, **kw: Any) -> dict[str, Any]:
        self._tick()
        assert "startTime" not in kw
        every = 8 * 60 * MIN
        times = [t for t in range(to_ms(LISTED), to_ms(NOW) + 1, every) if t <= kw["endTime"]]
        times = times[-kw["limit"] :][::-1]
        rows = [{"fundingRate": "0.0001", "fundingRateTimestamp": str(t)} for t in times]
        return {"result": {"list": rows}}

    def get_open_interest(self, **kw: Any) -> dict[str, Any]:
        self._tick()
        step = 5 * MIN
        first = max(kw["startTime"], to_ms(LISTED))
        times = [t for t in range(first - first % step, kw["endTime"] + 1, step)]
        times = [t for t in times if kw["startTime"] <= t <= to_ms(NOW)][-kw["limit"] :][::-1]
        rows = [{"openInterest": "1000", "timestamp": str(t)} for t in times]
        return {"result": {"list": rows}}


def make(tmp_path: Path, client: FakeBybit) -> tuple[Downloader, Storage]:
    storage = Storage(tmp_path)
    caller = RestCaller(min_interval_s=0, max_attempts=2, sleep=lambda _: None)
    return Downloader(client, storage, caller=caller, now=lambda: NOW), storage


def expected_candles() -> int:
    # 01.01 00:00 ... 03.01 12:33 включительно: 2 суток + 12 ч 34 мин
    return 2 * 1440 + 12 * 60 + 34


def test_downloads_only_closed_candles(tmp_path: Path) -> None:
    downloader, storage = make(tmp_path, FakeBybit())
    assert downloader.download_klines("BTCUSDT", LISTED) == expected_candles()
    df = storage.read_candles("BTCUSDT", "1m")
    assert df["open_time"].iloc[-1] == pd.Timestamp("2024-01-03 12:33", tz="UTC")
    assert df["open_time"].is_unique
    assert df["open_time"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()


def test_second_run_adds_nothing(tmp_path: Path) -> None:
    downloader, _ = make(tmp_path, FakeBybit())
    downloader.download_klines("BTCUSDT", LISTED)
    assert downloader.download_klines("BTCUSDT", LISTED) == 0


def test_resume_after_interruption(tmp_path: Path) -> None:
    downloader, storage = make(tmp_path, FakeBybit(fail_after=2))
    with pytest.raises(requests.ConnectionError):
        downloader.download_klines("BTCUSDT", LISTED)
    assert len(storage.read_candles("BTCUSDT", "1m")) == 2000  # успели 2 пачки по 1000

    client = FakeBybit()
    downloader, storage = make(tmp_path, client)
    assert downloader.download_klines("BTCUSDT", LISTED) == expected_candles() - 2000
    assert client.calls == 2  # продолжили с 2001-й свечи, а не с начала
    df = storage.read_candles("BTCUSDT", "1m")
    assert len(df) == expected_candles()
    assert df["open_time"].is_unique


def test_retry_on_network_error() -> None:
    attempts = []
    pauses: list[float] = []

    def flaky(**_: Any) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.ConnectionError("timeout")
        return {"ok": True}

    caller = RestCaller(min_interval_s=0, max_attempts=5, sleep=pauses.append)
    assert caller.call(flaky) == {"ok": True}
    assert pauses == [2.0, 4.0]  # пауза растёт: 2, 4 секунды


def test_funding_pages_back_through_history(tmp_path: Path) -> None:
    downloader, storage = make(tmp_path, FakeBybit())
    # С 01.01 00:00 до 03.01 12:34 — выплаты в 00, 08, 16: 3 + 3 + 2 = 8
    assert downloader.download_funding("BTCUSDT", LISTED) == 8
    downloader.caller = RestCaller(min_interval_s=0, sleep=lambda _: None)
    assert downloader.download_funding("BTCUSDT", LISTED) == 0
    assert len(storage.read_funding("BTCUSDT")) == 8


def test_funding_needs_several_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bot.data.downloader.FUNDING_LIMIT", 3)
    downloader, storage = make(tmp_path, FakeBybit())
    assert downloader.download_funding("BTCUSDT", LISTED) == 8
    assert storage.read_funding("BTCUSDT")["time"].is_unique


def test_open_interest(tmp_path: Path) -> None:
    downloader, storage = make(tmp_path, FakeBybit())
    # Точки каждые 5 минут с 00:00 01.01 до 12:25 03.01 (12:30 — интервал ещё не закрыт)
    expected = 2 * 288 + 12 * 12 + 6
    assert downloader.download_open_interest("BTCUSDT", LISTED) == expected
    assert downloader.download_open_interest("BTCUSDT", LISTED) == 0
    assert storage.read_open_interest("BTCUSDT", "5min")["time"].is_unique


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def test_missing_spans_finds_head_middle_and_tail() -> None:
    step = pd.Timedelta(minutes=1)
    # Есть 00:10–00:19 и 03:00–03:04 (времена open). Период 00:00–05:00.
    have = pd.Series(
        list(pd.date_range("2024-01-01 00:10", periods=10, freq="1min", tz="UTC"))
        + list(pd.date_range("2024-01-01 03:00", periods=5, freq="1min", tz="UTC"))
    )
    start, end = ts("2024-01-01 00:00"), ts("2024-01-01 05:00")
    # Пропуски: 00:00–00:10, 00:20–03:00, 03:05–05:00. Между ними < 1000 минут — склеиваются.
    assert missing_spans(have, start, end, step) == [(start, end)]
    # Без склейки (порог 1 свеча) — три отдельных участка
    import bot.data.downloader as d

    old, d.KLINE_LIMIT = d.KLINE_LIMIT, 1
    try:
        assert missing_spans(have, start, end, step) == [
            (start, ts("2024-01-01 00:10")),
            (ts("2024-01-01 00:20"), ts("2024-01-01 03:00")),
            (ts("2024-01-01 03:05"), end),
        ]
    finally:
        d.KLINE_LIMIT = old


def test_missing_spans_nothing_missing_or_nothing_stored() -> None:
    step = pd.Timedelta(minutes=1)
    start, end = ts("2024-01-01 00:00"), ts("2024-01-01 00:05")
    full = pd.Series(pd.date_range(start, periods=5, freq="1min", tz="UTC"))
    assert missing_spans(full, start, end, step) == []
    assert missing_spans(pd.Series([], dtype="datetime64[ns, UTC]"), start, end, step) == [
        (start, end)
    ]


def test_fill_klines_backfills_older_history(tmp_path: Path) -> None:
    """Сначала загрузились только свежие сутки (как это делает график), потом нужна
    вся история: download_klines её бы пропустил, fill_klines догружает."""
    downloader, storage = make(tmp_path, FakeBybit())
    downloader.download_klines("BTCUSDT", ts("2024-01-03"))
    assert downloader.download_klines("BTCUSDT", LISTED) == 0  # старое поведение: ничего
    added = downloader.fill_klines("BTCUSDT", LISTED, ts("2024-01-04"))
    assert added == 2 * 1440
    df = storage.read_candles("BTCUSDT", "1m")
    assert len(df) == expected_candles()
    assert df["open_time"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()
    assert downloader.fill_klines("BTCUSDT", LISTED, ts("2024-01-04")) == 0


def test_fill_klines_starts_at_listing(tmp_path: Path) -> None:
    class Listed(FakeBybit):
        def get_instruments_info(self, **_: Any) -> dict[str, Any]:
            return {"result": {"list": [{"launchTime": str(to_ms(LISTED))}]}}

    client = Listed()
    downloader, _ = make(tmp_path, client)
    assert downloader.fill_klines("BTCUSDT", ts("2021-01-01"), ts("2024-01-02")) == 1440
    calls = client.calls
    # Повтор: до листинга свечей нет, но запросов туда больше нет
    assert downloader.fill_klines("BTCUSDT", ts("2021-01-01"), ts("2024-01-02")) == 0
    assert client.calls == calls


def test_fill_funding_range(tmp_path: Path) -> None:
    downloader, storage = make(tmp_path, FakeBybit())
    # [02.01 00:00, 03.01 00:00): выплаты в 00, 08, 16 второго числа
    assert downloader.fill_funding("BTCUSDT", ts("2024-01-02"), ts("2024-01-03")) == 3
    times = storage.read_funding("BTCUSDT")["time"]
    assert times.min() == ts("2024-01-02")
