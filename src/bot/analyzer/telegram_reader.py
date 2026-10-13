"""Выгрузка постов Telegram-канала в SQLite (Telethon, пользовательский клиент:
Bot API не читает чужие каналы).

    uv run python -m bot.analyzer.telegram_reader --channel имя_канала
    uv run python -m bot.analyzer.telegram_reader --channel имя_канала --every 10
    uv run python -m bot.analyzer.telegram_reader --since 2025-01-01   # только с этой даты

При первом запуске Telegram спросит номер телефона и код из приложения; сессия
сохраняется в data/telegram.session (не в git). Повторный запуск докачивает новые посты.
Картинки сохраняются в data/telegram/media/<канал>/ (таблица telegram_media); в альбоме
текст и картинки — разные сообщения с общим grouped_id. --every N — проверять канал
каждые N минут, пока не остановите (Ctrl+C). Посты и картинки сохраняются в базу по ходу
выгрузки (каждые 20 сообщений), поэтому остановленная выгрузка продолжится с того же места.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from bot.core.config import load_config
from bot.core.db import connect, db_path

PROGRESS_EVERY = 20


@dataclass(frozen=True)
class PostRecord:
    channel: str
    post_id: int
    published_at: datetime  # UTC
    text: str


@dataclass(frozen=True)
class MediaRecord:
    channel: str
    post_id: int
    grouped_id: int | None
    path: str  # относительно папки data


class MessageSource(Protocol):
    def iter_messages(self, entity: str, **kwargs: Any) -> AsyncIterator[Any]: ...


class MediaSource(MessageSource, Protocol):
    async def download_media(self, message: Any, file: str) -> str | None: ...


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
        "SELECT MAX(m) AS m FROM (SELECT MAX(post_id) AS m FROM telegram_posts WHERE channel = ? "
        "UNION ALL SELECT MAX(post_id) FROM telegram_media WHERE channel = ?)",
        (channel, channel),
    ).fetchone()
    return int(row["m"] or 0)


async def fetch_posts(source: MessageSource, channel: str, min_id: int = 0) -> list[PostRecord]:
    """Посты с id > min_id, от старых к новым. Сообщения без текста пропускаются."""
    posts, _ = await fetch_channel(source, channel, min_id)
    return posts


async def fetch_channel(
    source: MessageSource,
    channel: str,
    min_id: int = 0,
    data_dir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
    since: datetime | None = None,
    on_batch: Callable[[list[PostRecord], list[MediaRecord]], None] | None = None,
) -> tuple[list[PostRecord], list[MediaRecord]]:
    """Посты и картинки с id > min_id. data_dir=None — картинки не скачиваются.

    Постом считается сообщение с текстом или одиночная картинка без текста; картинки
    альбома без подписи — это не отдельные посты, они привязаны к подписи по grouped_id.
    since — только сообщения не раньше этой даты (UTC). progress(сообщений, картинок)
    и on_batch(новые посты, новые картинки) — каждые PROGRESS_EVERY сообщений и в конце:
    так скачанное сохраняется по ходу, а не только после всей выгрузки.
    """
    posts: list[PostRecord] = []
    media: list[MediaRecord] = []
    flushed = [0, 0]  # сколько постов и картинок уже отдано в on_batch

    def flush() -> None:
        if on_batch is not None:
            on_batch(posts[flushed[0] :], media[flushed[1] :])
            flushed[:] = [len(posts), len(media)]

    seen = 0
    kwargs: dict[str, Any] = {"min_id": min_id, "reverse": True}
    if since is not None:
        kwargs["offset_date"] = since  # при reverse=True — сообщения после этой даты
    async for message in source.iter_messages(channel, **kwargs):
        if since is not None and message.date < since:
            continue
        seen += 1
        if seen % PROGRESS_EVERY == 0:
            flush()
            if progress is not None:
                progress(seen, len(media))
        text = (getattr(message, "message", None) or getattr(message, "text", None) or "").strip()
        photo = getattr(message, "photo", None) is not None
        grouped = getattr(message, "grouped_id", None)
        if photo and data_dir is not None:
            folder = data_dir / "telegram" / "media" / channel
            folder.mkdir(parents=True, exist_ok=True)
            saved = await source.download_media(message, file=str(folder / str(message.id)))  # type: ignore[attr-defined]
            if saved:
                rel = Path(saved).resolve().relative_to(data_dir.resolve()).as_posix()
                media.append(MediaRecord(channel, int(message.id), grouped, rel))
        if text or (photo and grouped is None and data_dir is not None):
            posts.append(PostRecord(channel, int(message.id), message.date.astimezone(UTC), text))
    flush()
    return posts, media


def save_media(conn: sqlite3.Connection, media: Iterable[MediaRecord]) -> int:
    inserted = 0
    for m in media:
        cur = conn.execute(
            "INSERT OR IGNORE INTO telegram_media VALUES (?, ?, ?, ?)",
            (m.channel, m.post_id, m.grouped_id, m.path),
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def post_images(conn: sqlite3.Connection, channel: str, post_id: int) -> list[str]:
    """Картинки поста (пути относительно папки data): своя и картинки его альбома."""
    rows = conn.execute(
        "SELECT path FROM telegram_media WHERE channel = ? AND (post_id = ? OR grouped_id IN "
        "(SELECT grouped_id FROM telegram_media WHERE channel = ? AND post_id = ? "
        "AND grouped_id IS NOT NULL)) ORDER BY post_id",
        (channel, post_id, channel, post_id),
    ).fetchall()
    return [str(r["path"]) for r in rows]


def main(argv: list[str] | None = None) -> None:
    from telethon import TelegramClient

    cfg = load_config()
    parser = argparse.ArgumentParser(description="Выгрузка постов Telegram-канала")
    parser.add_argument("--channel", help="имя канала без @ или ссылка (по умолчанию из конфига)")
    parser.add_argument("--every", type=float, default=0, help="повторять каждые N минут")
    parser.add_argument("--since", help="только посты не раньше даты (UTC), например 2025-01-01")
    args = parser.parse_args(argv)
    since = datetime.fromisoformat(args.since).replace(tzinfo=UTC) if args.since else None
    secrets = cfg.secrets
    if not secrets.tg_api_id or not secrets.tg_api_hash.get_secret_value():
        raise SystemExit("Впишите TG_API_ID и TG_API_HASH в .env (получить на my.telegram.org)")
    data_dir = cfg.resolve(cfg.data.dir)
    channel = (args.channel or cfg.scenarios.channel).removeprefix("https://t.me/").lstrip("@")
    conn = connect(db_path(data_dir))

    async def run(start: int) -> tuple[list[PostRecord], list[MediaRecord]]:
        client = TelegramClient(
            str(data_dir / "telegram"), secrets.tg_api_id, secrets.tg_api_hash.get_secret_value()
        )
        await client.start()  # при первом запуске спросит телефон и код
        try:

            def show(messages: int, images: int) -> None:
                print(
                    f"  просмотрено сообщений: {messages}, скачано картинок: {images}", flush=True
                )

            def save(posts: list[PostRecord], media: list[MediaRecord]) -> None:
                save_media(conn, media)
                save_posts(conn, posts)

            after = f" с {since:%Y-%m-%d}" if since else ""
            print(f"Канал {channel}: качаю сообщения после №{start}{after}…", flush=True)
            return await fetch_channel(client, channel, start, data_dir, show, since, save)
        finally:
            await client.disconnect()

    while True:
        before = conn.execute(
            "SELECT COUNT(*) AS n FROM telegram_posts WHERE channel = ?", (channel,)
        ).fetchone()["n"]
        _, media = asyncio.run(run(last_post_id(conn, channel)))
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM telegram_posts WHERE channel = ?", (channel,)
        ).fetchone()["n"]
        print(
            f"Новых постов: {total - before}, картинок: {len(media)}. "
            f"Всего в базе по каналу {channel}: {total}"
        )
        if args.every <= 0:
            break
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
