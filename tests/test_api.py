from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from bot.app.api import create_app
from bot.data.storage import Storage


def make_client(tmp_path: Path) -> TestClient:
    storage = Storage(tmp_path)
    times = pd.date_range("2024-03-01", periods=3 * 24 * 60, freq="1min", tz="UTC")
    prices = pd.Series(range(len(times)), dtype="float64") / 100 + 100
    storage.write_candles(
        "BTCUSDT",
        "1m",
        pd.DataFrame(
            {
                "open_time": times,
                "open": prices,
                "high": prices + 0.5,
                "low": prices - 0.5,
                "close": prices,
                "volume": 1.0,
                "turnover": prices,
            }
        ),
    )
    return TestClient(create_app(storage, ["BTCUSDT", "ETHUSDT"]))


def test_candles_with_indicators(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/candles", params={"symbol": "BTCUSDT", "timeframe": "1h"})
    assert response.status_code == 200
    candles = response.json()["candles"]
    assert len(candles) == 72  # 3 суток часовых свечей
    first, last = candles[0], candles[-1]
    assert first["time"] == int(pd.Timestamp("2024-03-01", tz="UTC").timestamp())
    assert first["open"] == 100.0
    assert last["ema20"] is not None
    assert last["ema200"] is None  # 72 свечей мало для EMA200
    assert last["bb_lower"] < last["bb_middle"] < last["bb_upper"]


def test_unknown_symbol_and_timeframe(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    assert client.get("/api/candles", params={"symbol": "DOGEUSDT"}).status_code == 404
    bad_tf = {"symbol": "BTCUSDT", "timeframe": "2h"}
    assert client.get("/api/candles", params=bad_tf).status_code == 400


def test_empty_symbol_and_page(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/candles", params={"symbol": "ETHUSDT"})
    assert response.json()["candles"] == []
    page = client.get("/")
    assert page.status_code == 200
    assert "TradingView" in page.text
    js = client.get("/static/vendor/lightweight-charts.standalone.production.js")
    assert js.status_code == 200
