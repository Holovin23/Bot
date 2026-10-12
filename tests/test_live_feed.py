from pathlib import Path
from typing import Any

import pandas as pd

from bot.data.live_feed import STALE_AFTER, LiveFeed, parse_confirmed
from bot.data.storage import Storage

T0 = pd.Timestamp("2024-03-01 10:00", tz="UTC")


def ws_message(minute: int, confirm: bool, symbol: str = "BTCUSDT") -> dict[str, Any]:
    start = int((T0 + pd.Timedelta(minutes=minute)).value // 1_000_000)
    return {
        "topic": f"kline.1.{symbol}",
        "type": "snapshot",
        "data": [
            {
                "start": start,
                "end": start + 59_999,
                "interval": "1",
                "open": "100",
                "close": "101",
                "high": "102",
                "low": "99",
                "volume": "3",
                "turnover": "300",
                "confirm": confirm,
                "timestamp": start + 59_999,
            }
        ],
    }


class FakeBackfiller:
    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        self.calls: list[tuple[str, pd.Timestamp]] = []

    def download_klines(self, symbol: str, start: pd.Timestamp, timeframe: str = "1m") -> int:
        self.calls.append((symbol, start))
        # «REST» отдаёт пропущенные свечи 10:01 и 10:02
        times = pd.date_range(start, T0 + pd.Timedelta(minutes=2), freq="1min")
        df = pd.DataFrame(
            {
                "open_time": times,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
                "turnover": 1.0,
            }
        )
        return self.storage.write_candles(symbol, timeframe, df)


Clock = dict[str, pd.Timestamp]


def make(tmp_path: Path) -> tuple[LiveFeed, Storage, FakeBackfiller, Clock]:
    storage = Storage(tmp_path)
    backfiller = FakeBackfiller(storage)
    clock = {"now": T0}
    feed = LiveFeed(storage, backfiller, ["BTCUSDT"], now=lambda: clock["now"])
    return feed, storage, backfiller, clock


def test_parse_keeps_only_confirmed() -> None:
    _, df = parse_confirmed(ws_message(0, confirm=False))
    assert df.empty
    symbol, df = parse_confirmed(ws_message(0, confirm=True))
    assert symbol == "BTCUSDT"
    assert df["open_time"].item() == T0
    assert df["close"].item() == 101


def test_unclosed_candle_is_not_stored(tmp_path: Path) -> None:
    feed, storage, _, _ = make(tmp_path)
    feed.handle_message(ws_message(0, confirm=False))
    assert storage.read_candles("BTCUSDT", "1m").empty
    feed.handle_message(ws_message(0, confirm=True))
    assert len(storage.read_candles("BTCUSDT", "1m")) == 1


def test_gap_is_filled_from_rest(tmp_path: Path) -> None:
    feed, storage, backfiller, _ = make(tmp_path)
    feed.handle_message(ws_message(0, confirm=True))  # 10:00
    feed.handle_message(ws_message(3, confirm=True))  # 10:03 — пропущены 10:01, 10:02
    assert backfiller.calls == [("BTCUSDT", T0 + pd.Timedelta(minutes=1))]
    times = storage.read_candles("BTCUSDT", "1m")["open_time"]
    assert list(times.dt.minute) == [0, 1, 2, 3]


def test_no_backfill_without_gap(tmp_path: Path) -> None:
    feed, _, backfiller, _ = make(tmp_path)
    feed.handle_message(ws_message(0, confirm=True))
    feed.handle_message(ws_message(1, confirm=True))
    feed.handle_message(ws_message(1, confirm=True))  # повтор той же свечи — не дубль
    assert backfiller.calls == []


def test_silent_stream_triggers_backfill(tmp_path: Path) -> None:
    feed, _, backfiller, clock = make(tmp_path)
    feed.handle_message(ws_message(0, confirm=True))
    assert not feed.check_stale()
    clock["now"] = T0 + STALE_AFTER + pd.Timedelta(seconds=1)
    assert feed.check_stale()
    assert backfiller.calls[0][0] == "BTCUSDT"
