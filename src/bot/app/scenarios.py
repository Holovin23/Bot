"""Данные вкладки «Сценарии»: посты канала с картинками и итогом разбора, ячейки.

Только чтение из SQLite; ячейки меняет разбор постов (python -m bot.scenarios),
из интерфейса можно лишь одобрить уровни с картинки.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from bot.analyzer.telegram_reader import post_images
from bot.core.db import connect, db_path

# Картинка канала: data/telegram/media/<канал>/<файл>
MEDIA_RE = re.compile(r"^telegram/media/([A-Za-z0-9_]{1,64})/([A-Za-z0-9_.-]{1,120})$")


def media_url(path: str) -> str | None:
    """Путь картинки из базы → адрес для браузера (None — путь не из папки канала)."""
    match = MEDIA_RE.match(path.replace("\\", "/"))
    if match is None:
        return None
    return f"/api/telegram/media/{match.group(1)}/{match.group(2)}"


def _ts(value: str | None) -> int | None:
    return int(pd.Timestamp(value).timestamp()) if value else None


def posts_feed(data_dir: Path, limit: int, channel: str | None = None) -> list[dict[str, Any]]:
    """Последние посты: текст, картинки (свои и альбома) и что с ними сделал разбор."""
    path = db_path(data_dir)
    if not path.exists():
        return []
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT p.channel, p.post_id, p.published_at, p.text, s.source, s.post_kind, "
            "s.result_json, s.report_json FROM telegram_posts p LEFT JOIN scenario_posts s "
            "ON s.channel = p.channel AND s.post_id = p.post_id "
            "WHERE (? IS NULL OR p.channel = ?) ORDER BY p.published_at DESC LIMIT ?",
            (channel, channel, limit),
        ).fetchall()
        feed = []
        for r in rows:
            images = [media_url(p) for p in post_images(conn, r["channel"], int(r["post_id"]))]
            result = json.loads(r["result_json"]) if r["result_json"] else None
            report = json.loads(r["report_json"]) if r["report_json"] else None
            feed.append(
                {
                    "channel": r["channel"],
                    "post_id": int(r["post_id"]),
                    "time": _ts(r["published_at"]),
                    "published_at": r["published_at"],
                    "text": r["text"],
                    "images": [u for u in images if u],
                    "parsed": r["post_kind"] is not None,
                    "source": r["source"],
                    "kind": r["post_kind"],
                    "summary": (result or {}).get("summary_ru"),
                    "result": result,  # ответ разбора — можно сразу отдать в бэктест поста
                    "report": report,
                }
            )
        return feed
    finally:
        conn.close()


def cells(data_dir: Path, include_cancelled: bool = False) -> list[dict[str, Any]]:
    """Ячейки с последней версией: сначала активные, затем (по желанию) отменённые."""
    path = db_path(data_dir)
    if not path.exists():
        return []
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT s.scenario_id, s.channel, s.symbol, s.side, s.status, s.needs_review, "
            "s.created_at, s.updated_at, v.version, v.valid_from, v.valid_to, v.post_id, "
            "v.scenario_json FROM scenarios s JOIN scenario_versions v "
            "ON v.scenario_id = s.scenario_id AND v.version = (SELECT MAX(version) "
            "FROM scenario_versions WHERE scenario_id = s.scenario_id) "
            "WHERE (? OR s.status = 'ACTIVE') "
            "ORDER BY s.status = 'ACTIVE' DESC, s.updated_at DESC",
            (include_cancelled,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "scenario_id": r["scenario_id"],
            "channel": r["channel"],
            "symbol": r["symbol"],
            "side": r["side"],
            "status": r["status"],
            "needs_review": bool(r["needs_review"]),
            "version": int(r["version"]),
            "post_id": int(r["post_id"]),
            "created": _ts(r["created_at"]),
            "updated": _ts(r["updated_at"]),
            "scenario": json.loads(r["scenario_json"]),
        }
        for r in rows
    ]
