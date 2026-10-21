import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from bot.analyzer.telegram_reader import MediaRecord, PostRecord, save_media, save_posts
from bot.core.config import Config, DataConfig, InstrumentConfig, ScenariosConfig, Secrets
from bot.core.db import connect, db_path
from bot.data.storage import Storage
from bot.research.scenario_backtest import post_versions, run_post
from bot.scenarios.__main__ import import_answer, parse_pending, pending_payload
from bot.scenarios.models import PostScenarios
from bot.scenarios.parser import parse_with_api, response_schema, system_prompt
from bot.scenarios.store import active_cells

TEXT = "BTC: если закрепимся выше 100к — лонг. Стоп 97к."
ANSWER: dict[str, Any] = {
    "post_kind": "ANALYSIS",
    "images": [{"index": 1, "kind": "CHART"}],
    "summary_ru": "Автор ждёт роста выше 100к",
    "existing": [],
    "new_scenarios": [
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "trigger": {
                "type": "close_above",
                "level": 100000,
                "zone_low": None,
                "zone_high": None,
            },
            "stop": 97000,
            "targets": [],
            "level_source": "TEXT",
            "quotes": ["если закрепимся выше 100к"],
            "conviction": "MEDIUM",
            "note_ru": "",
        }
    ],
}


def make_cfg(tmp_path: Path) -> Config:
    return Config(
        data=DataConfig(dir=tmp_path),
        scenarios=ScenariosConfig(channel="chan", symbols=["BTCUSDT"]),
        instruments={"BTCUSDT": InstrumentConfig(tick_size=0.1, qty_step=0.001, min_qty=0.001)},
        secrets=Secrets(),
    )


def seed(tmp_path: Path) -> Any:
    conn = connect(db_path(tmp_path))
    t = datetime(2025, 3, 1, 2, 0, tzinfo=UTC)
    save_posts(
        conn,
        [
            PostRecord("chan", 1, t, TEXT),
            PostRecord("chan", 2, t.replace(hour=5), "BTC: итоги недели +12%"),
        ],
    )
    save_media(conn, [MediaRecord("chan", 1, None, "telegram/media/chan/1.jpg")])
    return conn


def test_schema_matches_model_and_prompt_lists_symbols() -> None:
    schema = response_schema(["BTCUSDT", "SOLUSDT"])
    assert schema["properties"]["new_scenarios"]["items"]["properties"]["symbol"]["enum"] == [
        "BTCUSDT",
        "SOLUSDT",
    ]
    PostScenarios.model_validate(ANSWER)  # ответ по схеме читается моделью
    prompt = system_prompt(["BTCUSDT"])
    assert "PNL_SCREENSHOT" in prompt and "НЕ анализируй" in prompt


def test_pending_and_import_in_order(tmp_path: Path) -> None:
    cfg, conn = make_cfg(tmp_path), seed(tmp_path)
    payload = pending_payload(cfg, conn, limit=5, since=None)
    assert [p["post_id"] for p in payload["posts"]] == [1, 2]
    assert payload["posts"][0]["images"] == [
        str((tmp_path / "telegram/media/chan/1.jpg").resolve())
    ]
    report = import_answer(cfg, conn, 1, json.dumps(ANSWER), "claude-code")
    assert report.created == ["tg-BTCUSDT-202503010200-1"]
    with pytest.raises(SystemExit, match="уже разобран"):
        import_answer(cfg, conn, 1, json.dumps(ANSWER), "claude-code")
    results = {**ANSWER, "post_kind": "RESULTS", "new_scenarios": []}
    assert import_answer(cfg, conn, 2, json.dumps(results), "claude-code").post_kind == "RESULTS"
    assert pending_payload(cfg, conn, 5, None)["posts"] == []
    assert len(active_cells(conn)) == 1


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def reply(text: str, stop: str = "end_turn") -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=5000, output_tokens=1500, cache_read_input_tokens=0)
    return SimpleNamespace(
        stop_reason=stop, content=[SimpleNamespace(type="text", text=text)], usage=usage
    )


def test_parse_with_api_sends_images_and_handles_errors(tmp_path: Path) -> None:
    img = tmp_path / "chart.png"
    img.write_bytes(b"\x89PNG")
    client = FakeClient([reply(json.dumps(ANSWER)), reply("{", "max_tokens"), RuntimeError("сеть")])
    ok = parse_with_api(client, "claude-opus-5-5", ["BTCUSDT"], "2025-03-01", TEXT, [img], [])
    assert ok.result is not None and ok.usage["input_tokens"] == 5000
    call = client.calls[0]
    assert call["messages"][0]["content"][0]["source"]["media_type"] == "image/png"
    assert call["fallbacks"] == "default"
    bad = parse_with_api(client, "claude-haiku-5-5", ["BTCUSDT"], "2025-03-01", TEXT, [], [])
    assert bad.error == "ответ обрезан (max_tokens)" and "fallbacks" not in client.calls[1]
    failed = parse_with_api(client, "claude-opus-5-5", ["BTCUSDT"], "2025-03-01", TEXT, [], [])
    assert failed.error is not None and failed.error.startswith("ошибка API")


def test_parse_pending_records_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg, conn = make_cfg(tmp_path), seed(tmp_path)
    (tmp_path / "telegram/media/chan").mkdir(parents=True)
    (tmp_path / "telegram/media/chan/1.jpg").write_bytes(b"jpg")
    client = FakeClient([reply(json.dumps(ANSWER)), reply("", "refusal")])
    parse_pending(cfg, conn, client, "claude-opus-5-5", 5)
    out = capsys.readouterr().out
    assert "создан tg-BTCUSDT-202503010200-1" in out
    assert "пост 2: не разобран — модель отказалась" in out
    assert pending_payload(cfg, conn, 5, None)["posts"] == []


def minutes(start: str, closes: np.ndarray, wick: float = 5) -> pd.DataFrame:
    times = pd.date_range(start, periods=len(closes), freq="1min", tz="UTC")
    open_ = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": np.maximum(open_, closes) + wick,
            "low": np.minimum(open_, closes) - wick,
            "close": closes,
            "volume": 1.0,
            "turnover": closes,
        }
    )


def test_single_post_backtest(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    published = pd.Timestamp("2025-03-02 00:00", tz="UTC")
    answer = PostScenarios.model_validate(ANSWER)
    versions, rejected = post_versions(answer, TEXT, published, ["BTCUSDT"])
    assert rejected == [] and len(versions) == 1
    # 2 дня у 99 000 (до и после поста), рывок выше 100 000 и рост до 110 000
    flat = np.full(2 * 1440, 99_000.0)
    up = np.linspace(99_000, 110_000, 2 * 1440)
    candles = minutes("2025-03-01 00:00", np.r_[flat, up])
    Storage(tmp_path).write_candles("BTCUSDT", "1m", candles)
    [run] = run_post(cfg, Storage(tmp_path), versions, published, days=3, timeframe="1h")
    assert run.error == "" and run.result is not None
    assert next(iter(run.outcome.values())).startswith("сработал")
    # Стоп автора 97 000 — это больше 4 ATR часовых свечей: риск-менеджер отклоняет вход
    assert run.result.trades == []
    assert "стоп" in run.result.skipped[0].reason and "ATR" in run.result.skipped[0].reason
    # Цена у 99 800 (тени ±100), стоп автора 99 300 (≈ 3–4 ATR): вход проходит,
    # цель по умолчанию 2R достигается на росте
    other = tmp_path / "other"
    flat = np.full(2 * 1440, 99_800.0)
    up = np.linspace(99_800, 110_000, 2 * 1440)
    Storage(other).write_candles("BTCUSDT", "1m", minutes("2025-03-01 00:00", np.r_[flat, up], 100))
    near = answer.new_scenarios[0].model_copy(update={"stop": 99_300})
    versions, _ = post_versions(
        answer.model_copy(update={"new_scenarios": [near]}), TEXT, published, ["BTCUSDT"]
    )
    [run] = run_post(cfg, Storage(other), versions, published, days=3, timeframe="1h")
    assert run.result is not None
    [trade] = run.result.trades
    assert trade.initial_stop == 99_300
    assert trade.take is not None and trade.take > trade.entry_price
    assert trade.exit_reason == "take"
    results = post_versions(
        answer.model_copy(update={"post_kind": "RESULTS"}), TEXT, published, ["BTCUSDT"]
    )
    assert results == ([], ["пост типа RESULTS — сценарии не создаются"])
