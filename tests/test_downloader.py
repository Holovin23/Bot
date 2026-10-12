from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from bot.data.downloader import Downloader, RestCaller, to_ms
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
