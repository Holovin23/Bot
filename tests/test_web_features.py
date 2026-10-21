"""Новые возможности веб-интерфейса: график стратегии, период графика, монеты,
команды терминала, JSONL-журналы, несколько paper-ботов."""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bot.app.api import create_app
from bot.app.commands import BY_ID, Arg, Command, CommandRunner, build_argv
from bot.app.jobs import JobRunner
from bot.app.symbols import SymbolBook, normalize_symbol
from bot.core.config import load_config, local_symbols
from bot.core.journal import Journal, read_journal
from bot.data.storage import Storage
from bot.runner.host import BotHost, history_days

from .strategy_helpers import minute_walk


def storage_with_walk(tmp_path: Path, days: int = 20) -> Storage:
    storage = Storage(tmp_path / "data")
    storage.write_candles("BTCUSDT", "1m", minute_walk(days))
    return storage


# --- график стратегии --------------------------------------------------------------


def test_strategy_chart_has_own_indicators(tmp_path: Path) -> None:
    client = TestClient(create_app(storage_with_walk(tmp_path), ["BTCUSDT"]))
    params = {"strategy": "mean_reversion", "symbol": "BTCUSDT", "timeframe": "1h", "limit": 100}
    data = client.get("/api/strategy-chart", params=params).json()
    assert len(data["candles"]) == 100
    panes = {s["pane"] for s in data["series"]}
    assert panes == {"price", "RSI", "ADX"}
    rsi = next(s for s in data["series"] if s["column"] == "rsi")
    assert rsi["levels"] == [30.0, 70.0]
    assert len(rsi["values"]) == 100
    assert all(v is None or 0 <= v <= 100 for v in rsi["values"])

    # Параметры меняют индикатор: уровень ADX берётся из adx_max
    params["params"] = json.dumps({"adx_max": 30})
    data = client.get("/api/strategy-chart", params=params).json()
    adx = next(s for s in data["series"] if s["column"] == "adx")
    assert adx["levels"] == [30.0]
    params["params"] = "[1]"
    assert client.get("/api/strategy-chart", params=params).status_code == 400


def test_strategy_chart_matches_prepare(tmp_path: Path) -> None:
    """Значения на графике — ровно prepare() стратегии на тех же свечах."""
    from bot.app.charts import chart_bars
    from bot.strategies.registry import create_strategy

    storage = storage_with_walk(tmp_path)
    client = TestClient(create_app(storage, ["BTCUSDT"]))
    params = {"strategy": "breakout", "symbol": "BTCUSDT", "timeframe": "1h", "limit": 50}
    data = client.get("/api/strategy-chart", params=params).json()
    bars = chart_bars(storage.read_candles("BTCUSDT", "1m"), "1h")
    features = create_strategy("breakout", "1h").prepare(bars)
    rank = next(s for s in data["series"] if s["column"] == "squeeze_rank")["values"]
    expected = features["squeeze_rank"].tail(50).round(8).tolist()
    assert rank == pytest.approx([None if pd.isna(v) else v for v in expected])
    volume = next(s for s in data["series"] if s["column"] == "avg_volume")
    assert volume["values"][-1] == pytest.approx(features["avg_volume"].iloc[-1] * 1.5)


def test_composite_chart_series_use_member_prefixes() -> None:
    from bot.strategies.composite import Composite

    columns = {s.column for s in Composite("1h").chart_series()}
    assert {"regime", "mr__rsi", "tf__ema_fast", "b__channel_high", "volume"} <= columns


def test_candles_for_date_range(tmp_path: Path) -> None:
    client = TestClient(create_app(storage_with_walk(tmp_path), ["BTCUSDT"]))
    start = int(pd.Timestamp("2024-01-05", tz="UTC").timestamp())
    end = int(pd.Timestamp("2024-01-07", tz="UTC").timestamp())
    params = {"symbol": "BTCUSDT", "timeframe": "1h", "start": start, "end": end}
    data = client.get("/api/candles", params=params).json()
    assert len(data["candles"]) == 48
    assert data["candles"][0]["time"] == start
    assert data["candles"][-1]["time"] == end - 3600
    # EMA20 на первой свече периода уже посчитана: слева взяты свечи для разогрева
    assert data["candles"][0]["ema20"] is not None


# --- монеты -------------------------------------------------------------------------


INSTRUMENT = {
    "symbol": "SOLUSDT",
    "status": "Trading",
    "priceFilter": {"tickSize": "0.01"},
    "lotSizeFilter": {"qtyStep": "0.1", "minOrderQty": "0.1"},
}


def test_normalize_symbol() -> None:
    assert normalize_symbol(" sol ") == "SOLUSDT"
    assert normalize_symbol("XRP/USDT") == "XRPUSDT"
    with pytest.raises(ValueError):
        normalize_symbol("bad name!")


def test_add_symbol_saves_local_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text(
        "data: {symbols: [BTCUSDT]}\ninstruments:\n"
        "  BTCUSDT: {tick_size: 0.1, qty_step: 0.001, min_qty: 0.001}\n",
        encoding="utf-8",
    )
    symbols = ["BTCUSDT"]
    asked: list[str] = []

    def lookup(symbol: str) -> dict[str, Any] | None:
        asked.append(symbol)
        return INSTRUMENT if symbol == "SOLUSDT" else None

    book = SymbolBook(symbols, config_dir, lookup)
    client = TestClient(create_app(Storage(tmp_path / "data"), symbols, symbol_book=book))
    added = client.post("/api/symbols", json={"symbol": "sol"}).json()
    assert added["symbol"] == "SOLUSDT" and added["added"] is True
    assert symbols == ["BTCUSDT", "SOLUSDT"]
    assert client.post("/api/symbols", json={"symbol": "NOPEUSDT"}).status_code == 400

    cfg = load_config(config_dir / "default.yaml", env_file=None)
    assert cfg.data.symbols == ["BTCUSDT", "SOLUSDT"]
    assert cfg.instruments["SOLUSDT"].qty_step == 0.1
    assert cfg.costs.slippage["SOLUSDT"] == 0.001
    assert set(local_symbols(config_dir / "symbols.local.yaml")) == {"SOLUSDT"}

    # Монету из default.yaml убрать нельзя, добавленную — можно
    assert client.delete("/api/symbols/BTCUSDT").status_code == 400
    assert client.delete("/api/symbols/SOLUSDT").status_code == 200
    assert symbols == ["BTCUSDT"]


# --- команды терминала --------------------------------------------------------------


def test_build_argv_validates_values() -> None:
    backtest = BY_ID["backtest"]
    argv = build_argv(
        backtest,
        {
            "--strategy": "breakout",
            "--symbol": "BTCUSDT",
            "--timeframe": "1h",
            "--from": "2023-01-01",
            "--to": "2023-06-30",
            "--param": ["take_r=3", "volume_mult=2"],
        },
        ["BTCUSDT"],
    )
    assert argv == [
        "--strategy", "breakout", "--symbol", "BTCUSDT", "--timeframe", "1h",
        "--from", "2023-01-01", "--to", "2023-06-30",
        "--param", "take_r=3", "--param", "volume_mult=2",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="Не заполнено"):
        build_argv(backtest, {"--strategy": "breakout"}, ["BTCUSDT"])
    with pytest.raises(ValueError, match="недопустимое"):
        build_argv(BY_ID["quality"], {"--symbol": ["DOGEUSDT"]}, ["BTCUSDT"])
    with pytest.raises(ValueError, match="ГГГГ"):
        build_argv(BY_ID["quality"], {"--from": "01.01.2024"}, ["BTCUSDT"])
    with pytest.raises(ValueError, match="текст"):
        build_argv(BY_ID["telegram"], {"--channel": "--all"}, [])  # не даст подставить флаг


def test_command_runner_streams_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    command = Command(
        "echo", "Проверка", "Тест", ("-c", "import sys; print('привет', sys.argv[1:])"),
        (Arg("--name", "Имя"),),
    )
    monkeypatch.setitem(BY_ID, "echo", command)
    jobs = JobRunner(tmp_path / "jobs")
    runner = CommandRunner(jobs, lambda: [], python=sys.executable, cwd=str(tmp_path))
    client = TestClient(create_app(Storage(tmp_path), [], jobs=jobs, commands=runner))
    job_id = client.post("/api/commands/echo", json={"values": {"--name": "мир"}}).json()["job_id"]
    import time

    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}", params={"log": 20}).json()
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job["status"] == "done", job
    assert any("привет ['--name', 'мир']" in line for line in job["log"])
    assert list((tmp_path / "jobs").glob("*_command.log"))
    names = [c["id"] for c in client.get("/api/commands").json()["commands"]]
    assert {"download", "backtest", "experiment", "validate", "live_bot"} <= set(names)


# --- журналы ------------------------------------------------------------------------


def test_journal_writes_json_lines(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "j.jsonl", account="paper")
    journal.write("signal", price=float("nan"), at=pd.Timestamp("2024-01-01", tz="UTC"))
    records = read_journal(tmp_path / "j.jsonl")
    assert records[0]["kind"] == "signal"
    assert records[0]["account"] == "paper"
    assert records[0]["price"] is None
    assert records[0]["at"] == "2024-01-01T00:00:00+00:00"


def test_backtest_writes_jsonl_files(tmp_path: Path) -> None:
    from bot.backtest.report import run_report

    storage = storage_with_walk(tmp_path, days=30)
    cfg = load_config(env_file=None)
    cfg = cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"dir": tmp_path / "data"}),
            "logging": cfg.logging.model_copy(update={"dir": tmp_path / "logs"}),
        }
    )
    lines: list[str] = []
    folder, metrics = run_report(
        cfg,
        "random_entry",
        "BTCUSDT",
        "1h",
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-31", tz="UTC"),
        {"probability": 0.3},
        say=lines.append,
    )
    assert storage.root == tmp_path / "data"
    assert metrics.trades > 0
    trades = read_journal(folder / "trades.jsonl")
    assert len(trades) == metrics.trades
    first = trades[0]
    assert {"reason", "features", "entry_price", "exit_reason", "net_pnl", "stop_moves"} <= set(
        first
    )
    assert first["net_pnl"] == pytest.approx(
        first["gross_pnl"] - first["entry_fee"] - first["exit_fee"] + first["funding"]
    )
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert run["metrics"]["trades"] == metrics.trades
    assert run["params"]["probability"] == 0.3
    summary = read_journal(tmp_path / "logs" / "journal" / "backtests.jsonl")
    assert summary[-1]["backtest_id"] == folder.name
    assert any("сделок" in line for line in lines)


# --- несколько paper-ботов -----------------------------------------------------------


class NoFeed:
    def run(self, hours: float, stop: Any = None) -> None:
        return None


def test_history_days_covers_strategy_warmup() -> None:
    assert history_days("breakout", "1h", {}) == 30
    # EMA200 на 4h — 200 свечей × 4 ч × 1,5 = 50 дней
    assert history_days("trend_following", "4h", {}) == 52


def test_two_paper_bots_run_side_by_side(tmp_path: Path) -> None:
    data = tmp_path / "data"
    storage = Storage(data)
    walk = minute_walk(10)
    cut = len(walk) - 120  # последние 2 часа придут «вживую»
    storage.write_candles("BTCUSDT", "1m", walk.iloc[:cut])
    clock = {"now": walk["open_time"].iloc[cut - 1] + pd.Timedelta(minutes=1)}
    feeds: list[list[str]] = []

    def feed_factory(symbols: list[str], _cb: Any) -> NoFeed:
        feeds.append(symbols)
        return NoFeed()

    host = BotHost(
        data,
        storage,
        lambda: load_config(env_file=None),
        tmp_path / "journal",
        feed_factory=feed_factory,
        notify=lambda _text: None,
        now=lambda: clock["now"],
        background=False,
    )
    client = TestClient(create_app(storage, ["BTCUSDT"], host=host))
    body = {"strategy": "random_entry", "symbol": "BTCUSDT", "timeframe": "1h"}
    a = client.post("/api/bots", json={**body, "params": {"probability": 1.0}}).json()["id"]
    b = client.post("/api/bots", json={**body, "capital": 5000}).json()["id"]
    assert a != b
    bots = {x["id"]: x for x in client.get("/api/bots").json()["bots"]}
    assert bots[a]["state"] == bots[b]["state"] == "running"
    assert feeds[-1] == ["BTCUSDT"]

    # Пришли 2 часа свечей: бот a на закрытии часа получает сигнал (probability=1)
    for k in range(cut, len(walk)):
        storage.write_candles("BTCUSDT", "1m", walk.iloc[k : k + 1])
        clock["now"] = walk["open_time"].iloc[k] + pd.Timedelta(minutes=1)
        host.on_candles("BTCUSDT", walk.iloc[k : k + 1])

    kinds = [r["kind"] for r in read_journal(tmp_path / "journal" / f"paper-{a}.jsonl", 5000)]
    assert "bar" in kinds and "signal" in kinds and "risk_decision" in kinds
    detail = client.get(f"/api/bots/{a}").json()
    assert detail["status"]["mode"] == f"paper:{a}"
    # Счета раздельные: у бота b свой капитал
    detail_b = client.get(f"/api/bots/{b}").json()
    assert detail_b["equity"] == pytest.approx(5000)

    assert client.post(f"/api/bots/{a}/stop").status_code == 200
    assert client.get(f"/api/bots/{a}").json()["state"] == "stopped"
    assert host.get(a) is not None and host.get(a).enabled is False  # type: ignore[union-attr]
    assert client.delete(f"/api/bots/{b}").status_code == 200
    assert [x["id"] for x in client.get("/api/bots").json()["bots"]] == [a]
    assert host._feed_symbols == []  # никого не осталось — поток свечей остановлен
    host.shutdown()


def _report_cfg(tmp_path: Path) -> Any:
    cfg = load_config(env_file=None)
    return cfg.model_copy(
        update={
            "data": cfg.data.model_copy(update={"dir": tmp_path / "data"}),
            "logging": cfg.logging.model_copy(update={"dir": tmp_path / "logs"}),
        }
    )


def test_backtest_refuses_incomplete_history(tmp_path: Path) -> None:
    from bot.backtest.report import run_report

    storage_with_walk(tmp_path, days=10)  # свечи 01.01–10.01, просим до 31.01
    with pytest.raises(ValueError, match="неполных данных"):
        run_report(
            _report_cfg(tmp_path),
            "random_entry",
            "BTCUSDT",
            "1h",
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-31", tz="UTC"),
        )


def test_backtest_allows_period_before_listing(tmp_path: Path) -> None:
    from bot.backtest.report import run_report

    storage_with_walk(tmp_path, days=30)  # свечи с 01.01.2024, период с 01.12.2023
    lines: list[str] = []
    _, metrics = run_report(
        _report_cfg(tmp_path),
        "random_entry",
        "BTCUSDT",
        "1h",
        pd.Timestamp("2023-12-01", tz="UTC"),
        pd.Timestamp("2024-01-31", tz="UTC"),
        {"probability": 0.3},
        say=lines.append,
    )
    assert metrics.trades > 0
    assert any("начинаются с 2024-01-01" in line for line in lines)
