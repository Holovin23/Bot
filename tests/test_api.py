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


def write_backtest(root: Path, name: str) -> None:
    folder = root / name
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "signal_id": ["rnd-1"],
            "side": ["LONG"],
            "entry_time": ["2024-03-01 11:00:00+00:00"],
            "exit_time": ["2024-03-01 15:04:00+00:00"],
            "entry_price": [101.0],
            "exit_price": [99.5],
            "initial_stop": [99.5],
            "stop": [99.5],
            "take": [None],
            "exit_reason": ["stop"],
            "r_multiple": [-1.0],
            "net_pnl": [-50.0],
        }
    ).to_csv(folder / "trades.csv", index=False)


def test_backtests_list_and_trades(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    root = tmp_path / "backtests"
    write_backtest(root, "20261012-100000_mean_reversion_BTCUSDT_1h")
    write_backtest(root, "20261012-110000_breakout_ETHUSDT_5m")
    (root / "20261012-120000_grid_breakout_BTCUSDT_1h").mkdir()  # перебор: нет trades.csv

    items = client.get("/api/backtests").json()["backtests"]
    assert [b["id"] for b in items] == [
        "20261012-110000_breakout_ETHUSDT_5m",
        "20261012-100000_mean_reversion_BTCUSDT_1h",
    ]
    assert items[1]["strategy"] == "mean_reversion"
    assert items[1]["timeframe"] == "1h"

    data = client.get("/api/backtests/20261012-100000_mean_reversion_BTCUSDT_1h").json()
    trade = data["trades"][0]
    assert trade["number"] == 1
    assert trade["entry_time"] == int(pd.Timestamp("2024-03-01 11:00", tz="UTC").timestamp())
    assert trade["take"] is None
    assert trade["r_multiple"] == -1.0
    assert client.get("/api/backtests/..").status_code == 404
    assert client.get("/api/backtests/20261012-000000_x_BTCUSDT_1h").status_code == 404


def test_candles_until(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    until = int(pd.Timestamp("2024-03-02 05:30", tz="UTC").timestamp())
    params = {"symbol": "BTCUSDT", "timeframe": "1h", "limit": 10, "until": until}
    candles = client.get("/api/candles", params=params).json()["candles"]
    assert len(candles) == 10
    # Свеча 05:00 содержит момент until — она последняя
    assert candles[-1]["time"] == int(pd.Timestamp("2024-03-02 05:00", tz="UTC").timestamp())


def test_thin_equity_keeps_extremes() -> None:
    from bot.app.backtests import thin_equity

    times = list(range(10))
    values = [5.0, 4.0, 9.0, 6.0, 1.0, 7.0, 8.0, 2.0, 3.0, 6.5]
    assert len(thin_equity(times, values, 20)) == 10
    # 2 отрезка по 5 точек: в первом мин 1 (t=4) и макс 9 (t=2), во втором мин 2 (t=7)
    # и макс 8 (t=6), плюс первая и последняя точки
    thinned = thin_equity(times, values, 4)
    assert [p["time"] for p in thinned] == [0, 2, 4, 6, 7, 9]


def test_equity_and_report(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    name = "20261012-100000_mean_reversion_BTCUSDT_1h"
    write_backtest(tmp_path / "backtests", name)
    folder = tmp_path / "backtests" / name
    pd.DataFrame(
        {
            "time": ["2024-03-01 01:00:00+00:00", "2024-03-01 02:00:00+00:00"],
            "equity": [10000.0, 9950.123],
            "in_position": [False, True],
        }
    ).to_csv(folder / "equity.csv", index=False)
    (folder / "report.md").write_text("# Бэктест", encoding="utf-8")
    equity = client.get(f"/api/backtests/{name}/equity").json()["equity"]
    assert equity[1] == {
        "time": int(pd.Timestamp("2024-03-01 02:00", tz="UTC").timestamp()),
        "value": 9950.12,
    }
    assert client.get(f"/api/backtests/{name}/report").json()["report"] == "# Бэктест"
    assert client.get("/api/backtests/20261012-000000_x_BTCUSDT_1h/equity").status_code == 404
