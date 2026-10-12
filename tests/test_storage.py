from pathlib import Path

import pandas as pd
import pytest

from bot.data.storage import Storage


def candles(start: str, count: int, price: float = 100.0) -> pd.DataFrame:
    times = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": times,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": 10.0,
            "turnover": 1000.0,
        }
    )


def test_write_and_read_back(tmp_path: Path) -> None:
    store = Storage(tmp_path)
    assert store.write_candles("BTCUSDT", "1m", candles("2024-03-01", 5)) == 5
    assert (tmp_path / "klines" / "BTCUSDT" / "1m" / "2024-03.parquet").exists()
    df = store.read_candles("BTCUSDT", "1m")
    assert len(df) == 5
    assert str(df["open_time"].dtype) == "datetime64[ms, UTC]"
    assert df["open_time"].iloc[0] == pd.Timestamp("2024-03-01", tz="UTC")


def test_rewrite_does_not_duplicate(tmp_path: Path) -> None:
    store = Storage(tmp_path)
    store.write_candles("BTCUSDT", "1m", candles("2024-03-01", 5))
    # Те же 5 минут ещё раз + 3 новые (пересекаются 00:03 и 00:04)
    added = store.write_candles("BTCUSDT", "1m", candles("2024-03-01 00:03", 5, price=200))
    assert added == 3
    df = store.read_candles("BTCUSDT", "1m")
    assert len(df) == 8
    assert df["open_time"].is_unique
    assert df["open_time"].is_monotonic_increasing
    # Повторная строка заменена новой версией
    assert (
        df.loc[df["open_time"] == pd.Timestamp("2024-03-01 00:03", tz="UTC"), "close"].item() == 200
    )


def test_split_by_month_and_read_range(tmp_path: Path) -> None:
    store = Storage(tmp_path)
    # 2024-02-29 23:58 ... 2024-03-01 00:02 -> 2 свечи в феврале, 3 в марте
    store.write_candles("BTCUSDT", "1m", candles("2024-02-29 23:58", 5))
    folder = tmp_path / "klines" / "BTCUSDT" / "1m"
    assert sorted(p.name for p in folder.iterdir()) == ["2024-02.parquet", "2024-03.parquet"]

    part = store.read_candles("BTCUSDT", "1m", "2024-02-29 23:59Z", "2024-03-01 00:01Z")
    assert list(part["open_time"].dt.strftime("%H:%M")) == ["23:59", "00:00"]  # end не включается


def test_last_candle_time(tmp_path: Path) -> None:
    store = Storage(tmp_path)
    assert store.last_candle_time("BTCUSDT", "1m") is None
    store.write_candles("BTCUSDT", "1m", candles("2024-02-29 23:58", 5))
    assert store.last_candle_time("BTCUSDT", "1m") == pd.Timestamp("2024-03-01 00:02", tz="UTC")


def test_naive_time_is_rejected(tmp_path: Path) -> None:
    df = candles("2024-03-01", 2)
    df["open_time"] = df["open_time"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        Storage(tmp_path).write_candles("BTCUSDT", "1m", df)


def test_funding_and_open_interest(tmp_path: Path) -> None:
    store = Storage(tmp_path)
    funding = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-03-01 00:00", "2024-03-01 08:00"], utc=True),
            "funding_rate": [0.0001, -0.0002],
        }
    )
    assert store.write_funding("BTCUSDT", funding) == 2
    assert store.write_funding("BTCUSDT", funding) == 0
    assert store.read_funding("BTCUSDT")["funding_rate"].tolist() == [0.0001, -0.0002]

    oi = pd.DataFrame(
        {"time": pd.to_datetime(["2024-03-01 00:05"], utc=True), "open_interest": [55000.5]}
    )
    store.write_open_interest("BTCUSDT", "5min", oi)
    assert store.last_open_interest_time("BTCUSDT", "5min") == pd.Timestamp(
        "2024-03-01 00:05", tz="UTC"
    )
