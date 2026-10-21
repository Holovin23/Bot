"""Ячейки сценариев в SQLite и применение разбора поста к ним (docs/10).

Новый пост меняет только активные ячейки: создаёт, обновляет (новая версия) или
отменяет. Открытые позиции он не трогает — у них свой стоп и тейк на бирже.
Каждая версия хранит время поста, с которого она действует: бэктест видит ячейку
только после публикации.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from bot.scenarios.models import PostScenarios, Scenario, validate_scenario


@dataclass(frozen=True)
class ScenarioVersion:
    scenario_id: str
    version: int
    valid_from: pd.Timestamp
    valid_to: pd.Timestamp | None
    needs_review: bool
    scenario: Scenario


@dataclass
class ApplyReport:
    post_kind: str
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)  # «что — почему»
    usage: dict[str, int] = field(default_factory=dict)  # токены Claude API, если был вызов

    def lines(self) -> list[str]:
        out = [f"тип поста: {self.post_kind}"]
        out += [f"создан {x}" for x in self.created]
        out += [f"обновлён {x}" for x in self.updated]
        out += [f"отменён {x}" for x in self.cancelled]
        out += [f"отклонено: {x}" for x in self.rejected]
        return out


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _iso(ts: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


def is_processed(conn: sqlite3.Connection, channel: str, post_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM scenario_posts WHERE channel = ? AND post_id = ?", (channel, post_id)
    ).fetchone()
    return row is not None


def active_cells(
    conn: sqlite3.Connection, symbols: list[str] | None = None
) -> list[dict[str, Any]]:
    """Активные ячейки с их текущей версией — это ИИ видит при разборе нового поста."""
    rows = conn.execute(
        "SELECT s.scenario_id, s.symbol, s.needs_review, s.created_at, v.scenario_json "
        "FROM scenarios s JOIN scenario_versions v ON v.scenario_id = s.scenario_id "
        "AND v.valid_to IS NULL WHERE s.status = 'ACTIVE' ORDER BY s.created_at"
    ).fetchall()
    return [
        {
            "scenario_id": r["scenario_id"],
            "created_at": r["created_at"],
            "needs_review": bool(r["needs_review"]),
            "scenario": json.loads(r["scenario_json"]),
        }
        for r in rows
        if symbols is None or r["symbol"] in symbols
    ]


def _new_id(conn: sqlite3.Connection, symbol: str, published: pd.Timestamp) -> str:
    base = f"tg-{symbol}-{published:%Y%m%d%H%M}"
    n = 1
    while conn.execute(
        "SELECT 1 FROM scenarios WHERE scenario_id = ?", (f"{base}-{n}",)
    ).fetchone():
        n += 1
    return f"{base}-{n}"


def _close_version(conn: sqlite3.Connection, scenario_id: str, at: str) -> int:
    row = conn.execute(
        "SELECT MAX(version) AS v FROM scenario_versions WHERE scenario_id = ?", (scenario_id,)
    ).fetchone()
    conn.execute(
        "UPDATE scenario_versions SET valid_to = ? WHERE scenario_id = ? AND valid_to IS NULL",
        (at, scenario_id),
    )
    return int(row["v"] or 0)


def apply_post(
    conn: sqlite3.Connection,
    *,
    channel: str,
    post_id: int,
    published_at: datetime | pd.Timestamp,
    post_text: str,
    result: PostScenarios,
    symbols: list[str],
    source: str,
    usage: dict[str, int] | None = None,
) -> ApplyReport:
    """Применить разбор поста к ячейкам. Ошибочные сценарии отклоняются по одному,
    остальные применяются. Пост помечается обработанным."""
    published = pd.Timestamp(published_at).tz_convert("UTC")
    at = _iso(published)
    report = ApplyReport(result.post_kind, usage=dict(usage or {}))
    charts = sum(1 for img in result.images if img.kind == "CHART")
    if result.post_kind == "ANALYSIS":
        for decision in result.existing:
            sid = decision.scenario_id
            row = conn.execute(
                "SELECT symbol, status FROM scenarios WHERE scenario_id = ?", (sid,)
            ).fetchone()
            if row is None or row["status"] != "ACTIVE":
                if decision.action != "KEEP":
                    report.rejected.append(f"{sid} — нет такой активной ячейки")
                continue
            if decision.action == "CANCEL":
                _close_version(conn, sid, at)
                conn.execute(
                    "UPDATE scenarios SET status = 'CANCELLED', updated_at = ? "
                    "WHERE scenario_id = ?",
                    (at, sid),
                )
                report.cancelled.append(f"{sid}: {decision.reason_ru}")
            elif decision.action == "UPDATE":
                new = decision.scenario
                errors = ["нет новой версии"] if new is None else []
                if new is not None:
                    if new.symbol != row["symbol"]:
                        errors.append("другая монета — это новая ячейка, а не обновление")
                    errors += validate_scenario(new, post_text, symbols, charts)
                if errors or new is None:
                    report.rejected.append(f"обновление {sid} — {'; '.join(errors)}")
                    continue
                version = _close_version(conn, sid, at) + 1
                conn.execute(
                    "INSERT INTO scenario_versions VALUES (?, ?, ?, NULL, ?, ?)",
                    (sid, version, at, post_id, new.model_dump_json()),
                )
                conn.execute(
                    "UPDATE scenarios SET side = ?, needs_review = ?, updated_at = ? "
                    "WHERE scenario_id = ?",
                    (new.side, int(new.level_source == "IMAGE"), at, sid),
                )
                report.updated.append(f"{sid} (версия {version}): {decision.reason_ru}")
        for scenario in result.new_scenarios:
            errors = validate_scenario(scenario, post_text, symbols, charts)
            if errors:
                what = f"{scenario.symbol} {scenario.side} {scenario.trigger.type}"
                report.rejected.append(f"{what} — {'; '.join(errors)}")
                continue
            sid = _new_id(conn, scenario.symbol, published)
            conn.execute(
                "INSERT INTO scenarios VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, ?)",
                (
                    sid,
                    channel,
                    scenario.symbol,
                    scenario.side,
                    int(scenario.level_source == "IMAGE"),
                    at,
                    at,
                ),
            )
            conn.execute(
                "INSERT INTO scenario_versions VALUES (?, 1, ?, NULL, ?, ?)",
                (sid, at, post_id, scenario.model_dump_json()),
            )
            report.created.append(sid)
    record_post(conn, channel, post_id, source, result.post_kind, result.model_dump_json(), report)
    return report


def record_post(
    conn: sqlite3.Connection,
    channel: str,
    post_id: int,
    source: str,
    post_kind: str,
    result_json: str | None,
    report: ApplyReport | None,
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO scenario_posts VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            channel,
            post_id,
            source,
            post_kind,
            result_json,
            json.dumps(report.__dict__ if report else None, ensure_ascii=False),
            _now(),
        ),
    )
    conn.commit()


def approve(conn: sqlite3.Connection, scenario_id: str) -> bool:
    """Владелец проверил уровни с картинки: ячейку можно торговать."""
    cur = conn.execute(
        "UPDATE scenarios SET needs_review = 0 WHERE scenario_id = ?", (scenario_id,)
    )
    conn.commit()
    return cur.rowcount > 0


def load_versions(conn: sqlite3.Connection, symbol: str) -> list[ScenarioVersion]:
    """Все версии ячеек монеты по времени — то, что видит стратегия."""
    rows = conn.execute(
        "SELECT v.*, s.needs_review FROM scenario_versions v "
        "JOIN scenarios s ON s.scenario_id = v.scenario_id "
        "WHERE s.symbol = ? ORDER BY v.valid_from, v.scenario_id, v.version",
        (symbol,),
    ).fetchall()
    return [
        ScenarioVersion(
            scenario_id=r["scenario_id"],
            version=int(r["version"]),
            valid_from=pd.Timestamp(r["valid_from"]),
            valid_to=pd.Timestamp(r["valid_to"]) if r["valid_to"] else None,
            needs_review=bool(r["needs_review"]),
            scenario=Scenario.model_validate_json(r["scenario_json"]),
        )
        for r in rows
    ]
