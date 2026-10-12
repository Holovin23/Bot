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


def test_strategies_and_run_job(tmp_path: Path) -> None:
    import time

    calls = []

    def fake_run(strategy, symbol, timeframe, start, end, params):  # type: ignore[no-untyped-def]
        calls.append((strategy, symbol, timeframe, start, end, params))
        return "20261012-120000_breakout_BTCUSDT_1h"

    storage = Storage(tmp_path)
    client = TestClient(create_app(storage, ["BTCUSDT"], run=fake_run))
    catalog = client.get("/api/strategies").json()["strategies"]
    names = [s["name"] for s in catalog]
    assert "mean_reversion" in names
    mr = next(s for s in catalog if s["name"] == "mean_reversion")
    assert mr["params"]["adx_max"] == 20.0

    body = {"strategy": "breakout", "symbol": "BTCUSDT", "params": {"take_r": 3}}
    job_id = client.post("/api/backtests/run", json=body).json()["job_id"]
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job == {
        "status": "done",
        "backtest_id": "20261012-120000_breakout_BTCUSDT_1h",
        "error": None,
    }
    assert calls == [("breakout", "BTCUSDT", "1h", "2021-01-01", "2024-06-30", {"take_r": 3})]

    bad = {"strategy": "breakout", "symbol": "BTCUSDT", "end": "2025-08-01"}
    assert client.post("/api/backtests/run", json=bad).status_code == 400  # hold-out
    assert (
        client.post("/api/backtests/run", json={"strategy": "x", "symbol": "BTCUSDT"}).status_code
        == 400
    )


def test_job_runner_reports_errors() -> None:
    from bot.app.jobs import JobRunner

    def boom(*_args):  # type: ignore[no-untyped-def]
        raise ValueError("Нет 1m-свечей")

    runner = JobRunner(boom)
    request = {
        "strategy": "s",
        "symbol": "X",
        "timeframe": "1h",
        "start": "",
        "end": "",
        "params": {},
    }
    job = runner.start(request, wait=True)
    assert job.status == "error"
    assert job.error == "Нет 1m-свечей"
    assert not runner.busy()


def test_backtest_without_trades(tmp_path: Path) -> None:
    client = make_client(tmp_path)
    folder = tmp_path / "backtests" / "20261012-100000_breakout_BTCUSDT_1h"
    folder.mkdir(parents=True)
    (folder / "trades.csv").write_text("\n", encoding="utf-8")
    data = client.get("/api/backtests/20261012-100000_breakout_BTCUSDT_1h").json()
    assert data["trades"] == []


def test_telegram_feed(tmp_path: Path) -> None:
    import json as _json
    from datetime import UTC, datetime

    from bot.analyzer.llm_parser import ParseOutcome, store_parse
    from bot.analyzer.telegram_reader import PostRecord, save_posts
    from bot.core.db import connect

    client = make_client(tmp_path)
    assert client.get("/api/telegram").json()["posts"] == []
    conn = connect(tmp_path / "bot.sqlite")
    t = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
    save_posts(conn, [PostRecord("c", 1, t, "BTC вверх"), PostRecord("c", 2, t, "мем")])
    view = {
        "symbol": "BTCUSDT",
        "bias": "BULLISH",
        "horizon": "SWING",
        "levels": {"support": [], "resistance": [], "invalidation": None},
        "conviction": "LOW",
        "is_trade_call": False,
    }
    result = {"instruments": [view], "summary_ru": "рост", "parse_confidence": "HIGH"}
    store_parse(conn, "c", 1, t.isoformat(), ParseOutcome("parsed", result), "m")
    posts = {p["post_id"]: p for p in client.get("/api/telegram").json()["posts"]}
    assert posts[1]["summary"] == "рост"
    assert posts[1]["views"] == [{"symbol": "BTCUSDT", "bias": "BULLISH", "horizon": "SWING"}]
    assert posts[2]["summary"] is None
    assert _json.dumps(posts[1])
