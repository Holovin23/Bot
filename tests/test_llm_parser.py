import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bot.analyzer.llm_parser import (
    DEFAULT_MODEL,
    POST_SCHEMA,
    compare_with_labels,
    label_posts,
    parse_pending,
    parse_post,
)
from bot.analyzer.telegram_reader import PostRecord, save_posts
from bot.core.db import connect

GOOD = {
    "instruments": [
        {
            "symbol": "BTCUSDT",
            "bias": "BULLISH",
            "horizon": "SWING",
            "levels": {"support": [64000], "resistance": [66500], "invalidation": 63500},
            "conviction": "MEDIUM",
            "is_trade_call": False,
        }
    ],
    "summary_ru": "Автор ждёт роста при удержании 64000",
    "parse_confidence": "HIGH",
}


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def reply(text: str, stop: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)])


def client(*responses: Any) -> SimpleNamespace:
    return SimpleNamespace(messages=FakeMessages(list(responses)))


def test_parse_post_ok_and_request_shape() -> None:
    c = client(reply(json.dumps(GOOD)))
    outcome = parse_post(c, "BTC держит 64к", "2026-10-01T08:00:00+00:00")
    assert outcome.status == "parsed"
    assert outcome.result is not None
    assert outcome.result["instruments"][0]["bias"] == "BULLISH"
    call = c.messages.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": POST_SCHEMA}
    assert call["output_config"]["effort"] == "low"
    assert "2026-10-01T08:00:00+00:00" in call["messages"][0]["content"]


def test_parse_post_failures_do_not_raise() -> None:
    assert parse_post(client(reply("", "refusal")), "x", "t").error == (
        "модель отказалась разбирать пост"
    )
    assert parse_post(client(reply("{", "max_tokens")), "x", "t").status == "failed"
    bad = dict(GOOD, parse_confidence="SURE")
    assert "не по схеме" in (parse_post(client(reply(json.dumps(bad))), "x", "t").error or "")
    assert "ошибка API" in (parse_post(client(RuntimeError("timeout")), "x", "t").error or "")


def test_pending_parse_label_and_compare(tmp_path: Path) -> None:
    conn = connect(tmp_path / "bot.sqlite")
    t = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
    save_posts(conn, [PostRecord("c", 1, t, "BTC вверх"), PostRecord("c", 2, t, "мем")])
    c = client(reply(json.dumps(GOOD)), reply("", "refusal"))
    assert parse_pending(conn, c, DEFAULT_MODEL, 10) == (1, 1)
    # Удачно разобранный пост больше не отправляется; неудачный — пробуем снова
    c2 = client(reply(json.dumps(dict(GOOD, instruments=[]))))
    assert parse_pending(conn, c2, DEFAULT_MODEL, 10) == (1, 0)
    stored = conn.execute("SELECT result_json FROM post_parses WHERE post_id = 1").fetchone()
    assert json.loads(stored["result_json"])["post_id"] == 1

    answers = iter(["b", "-", "x", "n", "-"])  # «x» — неверный ввод, спросит снова
    assert label_posts(conn, 5, ask=lambda _: next(answers), show=lambda _: None) == 2
    agree, total, mismatches = compare_with_labels(conn, DEFAULT_MODEL)
    assert total == 4
    labels = {
        (r["post_id"], r["symbol"]): r["bias"] for r in conn.execute("SELECT * FROM post_labels")
    }
    expected_agree = sum(
        1
        for (post, sym), bias in labels.items()
        if bias == ("BULLISH" if (post, sym) == (1, "BTCUSDT") else "NONE")
    )
    assert agree == expected_agree
    assert len(mismatches) == total - agree
