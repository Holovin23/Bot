"""Разбор постов Telegram-канала через Claude API (docs/02-architecture.md, слой C).

Модель только **извлекает мнение автора** в JSON по схеме; она не прогнозирует цену
сама и не влияет на сделки. Результат хранится вместе с версией промпта и моделью.

    uv run python -m bot.analyzer.llm_parser --label 40      # ручная разметка (эталон)
    uv run python -m bot.analyzer.llm_parser --parse --limit 20
    uv run python -m bot.analyzer.llm_parser --compare       # сверка с разметкой
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from bot.core.config import load_config
from bot.core.db import connect, db_path

DEFAULT_MODEL = "claude-opus-5-5"
CHEAP_MODEL = "claude-haiku-5-5"
PROMPT_VERSION = "v1"
MAX_TOKENS = 4000
SYMBOLS = ("BTCUSDT", "ETHUSDT")

Bias = Literal["BULLISH", "BEARISH", "NEUTRAL"]

_NUMBER_OR_NULL = {"anyOf": [{"type": "number"}, {"type": "null"}]}

POST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["instruments", "summary_ru", "parse_confidence"],
    "properties": {
        "instruments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["symbol", "bias", "horizon", "levels", "conviction", "is_trade_call"],
                "properties": {
                    "symbol": {"type": "string", "enum": list(SYMBOLS)},
                    "bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL"]},
                    "horizon": {
                        "type": "string",
                        "enum": ["INTRADAY", "SWING", "POSITION", "UNKNOWN"],
                    },
                    "levels": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["support", "resistance", "invalidation"],
                        "properties": {
                            "support": {"type": "array", "items": {"type": "number"}},
                            "resistance": {"type": "array", "items": {"type": "number"}},
                            "invalidation": _NUMBER_OR_NULL,
                        },
                    },
                    "conviction": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                    "is_trade_call": {"type": "boolean"},
                },
            },
        },
        "summary_ru": {"type": "string"},
        "parse_confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
    },
}

SYSTEM_PROMPT = f"""Ты разбираешь аналитические посты Telegram-канала о криптовалютах.
Задача — извлечь мнение АВТОРА поста, а не дать собственный прогноз.

Правила:
- Учитывай только инструменты {", ".join(SYMBOLS)}. Биткоин, BTC, BTCUSD → BTCUSDT; эфир,
  ETH, ETHUSD → ETHUSDT. Если пост не о них, верни пустой список instruments.
- bias — куда, по мнению автора, пойдёт цена: BULLISH (вверх), BEARISH (вниз),
  NEUTRAL (боковик или автор не определился).
- horizon — горизонт, если автор его называет или он ясен из контекста
  (INTRADAY — в течение дня, SWING — дни–недели, POSITION — месяцы), иначе UNKNOWN.
- levels — только уровни, которые автор назвал явно; invalidation — уровень, при котором
  автор считает свой сценарий неверным, или null.
- conviction — насколько уверенно пишет автор (а не насколько уверен ты).
- is_trade_call — автор прямо предлагает открыть сделку.
- summary_ru — одно предложение на русском: что думает автор.
- parse_confidence — насколько однозначно удалось разобрать текст.
Не добавляй ничего, чего нет в тексте."""


class Levels(BaseModel):
    model_config = ConfigDict(extra="forbid")
    support: list[float]
    resistance: list[float]
    invalidation: float | None


class InstrumentView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: Literal["BTCUSDT", "ETHUSDT"]
    bias: Bias
    horizon: Literal["INTRADAY", "SWING", "POSITION", "UNKNOWN"]
    levels: Levels
    conviction: Literal["LOW", "MEDIUM", "HIGH"]
    is_trade_call: bool


class PostAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruments: list[InstrumentView]
    summary_ru: str
    parse_confidence: Literal["LOW", "MEDIUM", "HIGH"]


@dataclass(frozen=True)
class ParseOutcome:
    status: Literal["parsed", "failed"]
    result: dict[str, Any] | None
    error: str | None = None


def parse_post(
    client: Any, text: str, published_at: str, model: str = DEFAULT_MODEL
) -> ParseOutcome:
    """Один вызов Claude API. Любая ошибка → «не разобран», бот работает дальше."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Пост опубликован {published_at} (UTC):\n\n{text}"}
            ],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": POST_SCHEMA},
            },
        )
    except Exception as exc:  # сеть, лимиты, неверный ключ — пост просто не разобран
        return ParseOutcome("failed", None, f"ошибка API: {type(exc).__name__}: {exc}")
    if response.stop_reason == "refusal":
        return ParseOutcome("failed", None, "модель отказалась разбирать пост")
    if response.stop_reason == "max_tokens":
        return ParseOutcome("failed", None, "ответ обрезан (max_tokens)")
    text_out = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    try:
        analysis = PostAnalysis.model_validate(json.loads(text_out))
    except (json.JSONDecodeError, ValidationError) as exc:
        return ParseOutcome("failed", None, f"ответ не по схеме: {exc}"[:500])
    return ParseOutcome("parsed", analysis.model_dump())


def store_parse(
    conn: sqlite3.Connection,
    channel: str,
    post_id: int,
    published_at: str,
    outcome: ParseOutcome,
    model: str,
) -> None:
    result = None
    if outcome.result is not None:
        result = json.dumps(
            {"post_id": post_id, "published_at": published_at, **outcome.result},
            ensure_ascii=False,
        )
    conn.execute(
        "INSERT OR REPLACE INTO post_parses (channel, post_id, model, prompt_version, status, "
        "result_json, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            channel,
            post_id,
            model,
            PROMPT_VERSION,
            outcome.status,
            result,
            outcome.error,
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()


def pending_posts(conn: sqlite3.Connection, model: str, limit: int) -> list[sqlite3.Row]:
    """Посты без удачного разбора этой моделью и версией промпта, от старых к новым."""
    return conn.execute(
        "SELECT p.channel, p.post_id, p.published_at, p.text FROM telegram_posts p "
        "LEFT JOIN post_parses r ON r.channel = p.channel AND r.post_id = p.post_id "
        "AND r.model = ? AND r.prompt_version = ? AND r.status = 'parsed' "
        "WHERE r.post_id IS NULL ORDER BY p.published_at LIMIT ?",
        (model, PROMPT_VERSION, limit),
    ).fetchall()


def parse_pending(conn: sqlite3.Connection, client: Any, model: str, limit: int) -> tuple[int, int]:
    parsed = failed = 0
    for row in pending_posts(conn, model, limit):
        outcome = parse_post(client, row["text"], row["published_at"], model)
        store_parse(conn, row["channel"], row["post_id"], row["published_at"], outcome, model)
        if outcome.status == "parsed":
            parsed += 1
        else:
            failed += 1
            print(f"  пост {row['post_id']}: {outcome.error}")
    return parsed, failed


LABEL_KEYS = {"b": "BULLISH", "s": "BEARISH", "n": "NEUTRAL", "-": "NONE"}


def label_posts(
    conn: sqlite3.Connection,
    count: int,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> int:
    """Ручная разметка: для каждого поста и символа — мнение автора. Это эталон."""
    rows = conn.execute(
        "SELECT p.channel, p.post_id, p.published_at, p.text FROM telegram_posts p "
        "WHERE NOT EXISTS (SELECT 1 FROM post_labels l WHERE l.channel = p.channel "
        "AND l.post_id = p.post_id) ORDER BY RANDOM() LIMIT ?",
        (count,),
    ).fetchall()
    done = 0
    for row in rows:
        show(f"\n— Пост {row['post_id']} ({row['published_at']} UTC) —\n{row['text']}\n")
        for symbol in SYMBOLS:
            answer = ""
            while answer not in LABEL_KEYS:
                answer = ask(f"{symbol}: b — рост, s — падение, n — нейтрально, - — нет мнения: ")
                answer = answer.strip().lower()
            conn.execute(
                "INSERT OR REPLACE INTO post_labels VALUES (?, ?, ?, ?, ?)",
                (
                    row["channel"],
                    row["post_id"],
                    symbol,
                    LABEL_KEYS[answer],
                    datetime.now(UTC).isoformat(),
                ),
            )
        conn.commit()
        done += 1
    return done


def model_bias(result_json: str | None, symbol: str) -> str:
    if not result_json:
        return "NONE"
    for item in json.loads(result_json).get("instruments", []):
        if item.get("symbol") == symbol:
            return str(item.get("bias", "NONE"))
    return "NONE"


def compare_with_labels(conn: sqlite3.Connection, model: str) -> tuple[int, int, list[str]]:
    """(совпало, всего, список расхождений) по bias для размеченных постов."""
    rows = conn.execute(
        "SELECT l.post_id, l.symbol, l.bias, r.result_json FROM post_labels l "
        "JOIN post_parses r ON r.channel = l.channel AND r.post_id = l.post_id "
        "AND r.model = ? AND r.prompt_version = ? AND r.status = 'parsed'",
        (model, PROMPT_VERSION),
    ).fetchall()
    agree, mismatches = 0, []
    for row in rows:
        predicted = model_bias(row["result_json"], row["symbol"])
        if predicted == row["bias"]:
            agree += 1
        else:
            mismatches.append(
                f"пост {row['post_id']} {row['symbol']}: вы {row['bias']}, модель {predicted}"
            )
    return agree, len(rows), mismatches


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Разбор постов Telegram через Claude API")
    parser.add_argument("--label", type=int, metavar="N", help="разметить вручную N постов")
    parser.add_argument("--parse", action="store_true", help="разобрать новые посты")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--compare", action="store_true", help="сверить с ручной разметкой")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"например, {CHEAP_MODEL}")
    args = parser.parse_args(argv)
    conn = connect(db_path(cfg.resolve(cfg.data.dir)))

    if args.label:
        print(f"Размечено постов: {label_posts(conn, args.label)}")
    if args.parse:
        key = cfg.secrets.anthropic_api_key.get_secret_value()
        if not key:
            raise SystemExit("Впишите ANTHROPIC_API_KEY в .env")
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        parsed, failed = parse_pending(conn, client, args.model, args.limit)
        print(f"Разобрано: {parsed}, не разобрано: {failed}")
    if args.compare:
        agree, total, mismatches = compare_with_labels(conn, args.model)
        if not total:
            print("Нет постов, которые и размечены, и разобраны моделью.")
        else:
            print(f"Совпадение bias с вашей разметкой: {agree} из {total} ({agree / total:.0%})")
            for line in mismatches[:30]:
                print("  " + line)
    if not (args.label or args.parse or args.compare):
        parser.print_help()


if __name__ == "__main__":
    main()
