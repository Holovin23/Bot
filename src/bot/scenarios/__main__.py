"""Команды для сценариев из Telegram (docs/10-telegram-scenarios-plan.md).

    uv run python -m bot.scenarios pending                 # следующий неразобранный пост (JSON)
    uv run python -m bot.scenarios import ответ.json --post 123   # план Б: ответ Claude Code
    uv run python -m bot.scenarios parse --limit 20        # план А: разбор через Claude API
    uv run python -m bot.scenarios list                    # активные ячейки
    uv run python -m bot.scenarios approve tg-BTCUSDT-202610131200-1   # уровни с картинки проверены

Посты разбираются строго по порядку публикации: каждый следующий видит ячейки,
созданные предыдущими.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from bot.analyzer.telegram_reader import post_images
from bot.core.config import Config, load_config
from bot.core.db import connect, db_path
from bot.scenarios.models import PostScenarios
from bot.scenarios.parser import PROMPT_VERSION, parse_with_api, response_schema, system_prompt
from bot.scenarios.store import (
    ApplyReport,
    active_cells,
    apply_post,
    approve,
    is_processed,
    record_post,
)


def pending_rows(
    conn: sqlite3.Connection, channel: str, limit: int, since: str | None = None
) -> list[sqlite3.Row]:
    """Неразобранные посты канала, от старых к новым."""
    return conn.execute(
        "SELECT p.post_id, p.published_at, p.text FROM telegram_posts p "
        "WHERE p.channel = ? AND p.published_at >= ? AND NOT EXISTS (SELECT 1 FROM "
        "scenario_posts s WHERE s.channel = p.channel AND s.post_id = p.post_id) "
        "ORDER BY p.published_at, p.post_id LIMIT ?",
        (channel, since or "", limit),
    ).fetchall()


def pending_payload(
    cfg: Config, conn: sqlite3.Connection, limit: int, since: str | None
) -> dict[str, Any]:
    channel, symbols = cfg.scenarios.channel, cfg.scenarios.symbols
    data_dir = cfg.resolve(cfg.data.dir)
    posts = [
        {
            "post_id": int(r["post_id"]),
            "published_at": r["published_at"],
            "text": r["text"],
            "images": [
                str((data_dir / p).resolve()) for p in post_images(conn, channel, int(r["post_id"]))
            ],
        }
        for r in pending_rows(conn, channel, limit, since)
    ]
    return {
        "channel": channel,
        "prompt_version": PROMPT_VERSION,
        "rules": system_prompt(symbols),
        "schema": response_schema(symbols),
        "active_cells": active_cells(conn, symbols),
        "posts": posts,
    }


def _post(conn: sqlite3.Connection, channel: str, post_id: int) -> sqlite3.Row:
    row: sqlite3.Row | None = conn.execute(
        "SELECT post_id, published_at, text FROM telegram_posts WHERE channel = ? AND post_id = ?",
        (channel, post_id),
    ).fetchone()
    if row is None:
        raise SystemExit(f"Поста {post_id} нет в базе. Сначала выгрузите канал (telegram_reader).")
    return row


def import_answer(
    cfg: Config, conn: sqlite3.Connection, post_id: int, answer: str, source: str
) -> ApplyReport:
    channel = cfg.scenarios.channel
    row = _post(conn, channel, post_id)
    if is_processed(conn, channel, post_id):
        raise SystemExit(f"Пост {post_id} уже разобран.")
    older = pending_rows(conn, channel, 1)
    if older and int(older[0]["post_id"]) != post_id:
        print(f"Внимание: есть более ранний неразобранный пост {older[0]['post_id']}.")
    try:
        result = PostScenarios.model_validate_json(answer)
    except ValidationError as exc:
        raise SystemExit(f"Ответ не по схеме:\n{exc}") from None
    return apply_post(
        conn,
        channel=channel,
        post_id=post_id,
        published_at=row["published_at"],
        post_text=row["text"],
        result=result,
        symbols=cfg.scenarios.symbols,
        source=source,
    )


def parse_pending(
    cfg: Config, conn: sqlite3.Connection, client: Any, model: str, limit: int
) -> None:
    channel, symbols = cfg.scenarios.channel, cfg.scenarios.symbols
    data_dir = cfg.resolve(cfg.data.dir)
    for row in pending_rows(conn, channel, limit):
        post_id = int(row["post_id"])
        images = [data_dir / p for p in post_images(conn, channel, post_id)]
        outcome = parse_with_api(
            client,
            model,
            symbols,
            row["published_at"],
            row["text"],
            images,
            active_cells(conn, symbols),
        )
        if outcome.result is None:
            failed = ApplyReport("FAILED", rejected=[outcome.error or ""], usage=outcome.usage)
            record_post(conn, channel, post_id, model, "FAILED", None, failed)
            print(f"пост {post_id}: не разобран — {outcome.error}")
            continue
        report = apply_post(
            conn,
            channel=channel,
            post_id=post_id,
            published_at=row["published_at"],
            post_text=row["text"],
            result=outcome.result,
            symbols=symbols,
            source=model,
            usage=outcome.usage,
        )
        print(f"пост {post_id}: " + "; ".join(report.lines()) + f"; токены {outcome.usage}")


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Сценарии из Telegram-канала")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pending", help="неразобранные посты, правила и схема (JSON)")
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--since", help="не раньше даты, например 2025-01-01")
    p = sub.add_parser("import", help="применить ответ по схеме (план Б)")
    p.add_argument("file", help="JSON-файл с ответом или - для stdin")
    p.add_argument("--post", type=int, required=True)
    p.add_argument("--source", default="claude-code")
    p = sub.add_parser("parse", help="разобрать посты через Claude API (план А)")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--model", default=None, help="по умолчанию scenarios.model из конфига")
    sub.add_parser("list", help="активные ячейки")
    p = sub.add_parser("approve", help="уровни с картинки проверены — ячейку можно торговать")
    p.add_argument("scenario_id")
    args = parser.parse_args(argv)
    conn = connect(db_path(cfg.resolve(cfg.data.dir)))

    if args.command == "pending":
        payload = pending_payload(cfg, conn, args.limit, args.since)
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        print(json.dumps(payload, ensure_ascii=False, indent=1))
    elif args.command == "import":
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
        report = import_answer(cfg, conn, args.post, text, args.source)
        print("\n".join(report.lines()))
    elif args.command == "parse":
        key = cfg.secrets.anthropic_api_key.get_secret_value()
        if not key:
            raise SystemExit("Впишите ANTHROPIC_API_KEY в .env (или используйте план Б)")
        import anthropic

        parse_pending(
            cfg,
            conn,
            anthropic.Anthropic(api_key=key),
            args.model or cfg.scenarios.model,
            args.limit,
        )
    elif args.command == "list":
        for cell in active_cells(conn, cfg.scenarios.symbols):
            s = cell["scenario"]
            t = s["trigger"]
            where = t["level"] if t["level"] is not None else f"{t['zone_low']}–{t['zone_high']}"
            review = " [проверить уровни с картинки]" if cell["needs_review"] else ""
            print(
                f"{cell['scenario_id']}: {s['side']} {t['type']} {where}, стоп {s['stop']}, "
                f"цели {s['targets']}{review}"
            )
    elif args.command == "approve":
        print("одобрено" if approve(conn, args.scenario_id) else "нет такой ячейки")


if __name__ == "__main__":
    main()
