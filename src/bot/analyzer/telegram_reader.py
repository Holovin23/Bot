"""Выгрузка постов Telegram-канала в SQLite (Telethon, пользовательский клиент:
Bot API не читает чужие каналы).

    uv run python -m bot.analyzer.telegram_reader --channel имя_канала

При первом запуске Telegram спросит номер телефона и код из приложения; сессия
сохраняется в data/telegram.session (не в git). Повторный запуск докачивает новые посты.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from bot.core.config import load_config
from bot.core.db import connect, db_path


@dataclass(frozen=True)
class PostRecord:
    channel: str
    post_id: int
    published_at: datetime  # UTC
    text: str


class MessageSource(Protocol):
    def iter_messages(self, entity: str, **kwargs: Any) -> AsyncIterator[Any]: ...


def save_posts(conn: sqlite3.Connection, posts: Iterable[PostRecord]) -> int:
    """Сохраняет новые посты; уже сохранённые (тот же канал и id) пропускаются."""
    inserted = 0
    for p in posts:
        cur = conn.execute(
            "INSERT OR IGNORE INTO telegram_posts (channel, post_id, published_at, text) "
            "VALUES (?, ?, ?, ?)",
            (p.channel, p.post_id, p.published_at.astimezone(UTC).isoformat(), p.text),
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def last_post_id(conn: sqlite3.Connection, channel: str) -> int:
    row = conn.execute(
        "SELECT MAX(post_id) AS m FROM telegram_posts WHERE channel = ?", (channel,)
    ).fetchone()
    return int(row["m"] or 0)


async def fetch_posts(source: MessageSource, channel: str, min_id: int = 0) -> list[PostRecord]:
    """Посты с id > min_id, от старых к новым. Сообщения без текста пропускаются."""
    posts = []
    async for message in source.iter_messages(channel, min_id=min_id, reverse=True):
        text = (getattr(message, "message", None) or getattr(message, "text", None) or "").strip()
        if not text:
            continue
        posts.append(PostRecord(channel, int(message.id), message.date.astimezone(UTC), text))
    return posts


def main(argv: list[str] | None = None) -> None:
    from telethon import TelegramClient

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Выгрузка постов Telegram-канала")
    parser.add_argument("--channel", required=True, help="имя канала без @ или ссылка")
    args = parser.parse_args(argv)
    secrets = cfg.secrets
    if not secrets.tg_api_id or not secrets.tg_api_hash.get_secret_value():
        raise SystemExit("Впишите TG_API_ID и TG_API_HASH в .env (получить на my.telegram.org)")
    data_dir = cfg.resolve(cfg.data.dir)
    channel = args.channel.removeprefix("https://t.me/").lstrip("@")
    conn = connect(db_path(data_dir))
    start = last_post_id(conn, channel)

    async def run() -> list[PostRecord]:
        client = TelegramClient(
            str(data_dir / "telegram"), secrets.tg_api_id, secrets.tg_api_hash.get_secret_value()
        )
        await client.start()  # при первом запуске спросит телефон и код
        try:
            return await fetch_posts(client, channel, start)
        finally:
            await client.disconnect()

    posts = asyncio.run(run())
    added = save_posts(conn, posts)
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM telegram_posts WHERE channel = ?", (channel,)
    ).fetchone()["n"]
    print(f"Новых постов: {added}. Всего в базе по каналу {channel}: {total}")


if __name__ == "__main__":
    main()
