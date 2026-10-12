"""SQLite: data/bot.sqlite. Таблицы создаются при первом подключении.

Здесь только то, что нужно хранить надёжно и читать из интерфейса: посты Telegram и
их разбор, а с этапа 7 — сигналы, ордера, позиции и события живого бота.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS telegram_posts (
        channel TEXT NOT NULL,
        post_id INTEGER NOT NULL,
        published_at TEXT NOT NULL,   -- ISO 8601, UTC
        text TEXT NOT NULL,
        PRIMARY KEY (channel, post_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS post_parses (
        channel TEXT NOT NULL,
        post_id INTEGER NOT NULL,
        model TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        status TEXT NOT NULL,         -- parsed | failed
        result_json TEXT,             -- JSON по схеме из 02 или NULL
        error TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (channel, post_id, model, prompt_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS post_labels (
        channel TEXT NOT NULL,
        post_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        bias TEXT NOT NULL,           -- BULLISH | BEARISH | NEUTRAL | NONE
        labeled_at TEXT NOT NULL,
        PRIMARY KEY (channel, post_id, symbol)
    )
    """,
]


def connect(path: Path) -> sqlite3.Connection:
    """Подключение с созданием таблиц. Строки доступны по имени колонки."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    for statement in SCHEMA:
        conn.execute(statement)
    conn.commit()
    return conn


def db_path(data_dir: Path) -> Path:
    return data_dir / "bot.sqlite"
