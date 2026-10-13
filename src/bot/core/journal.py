"""Структурированный журнал в формате JSON Lines: одна строка — одно событие.

Такой файл удобно читать и человеку (каждая строка — понятный словарь), и ИИ: можно
приложить его к вопросу «проверь, правильно ли бот открыл и закрыл сделки». Пишется
только то, что нужно для разбора: сигнал с причиной и признаками, решение риск-менеджера,
ордера и ответы биржи, перенос стопа, закрытие с результатом. Ключей API здесь нет.

Формат строки: {"time": "...UTC", "account": "...", "kind": "...", ...поля события}.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Привести значение к тому, что понимает json: даты — ISO-строки, NaN — null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):  # числа numpy
        return to_jsonable(value.item())
    return str(value)


def dumps(record: dict[str, Any]) -> str:
    return json.dumps(to_jsonable(record), ensure_ascii=False, separators=(",", ":"))


class Journal:
    """Дописывает строки в файл. Можно писать из разных потоков."""

    def __init__(self, path: Path, **base: Any) -> None:
        self.path = path
        self.base = base
        self._lock = threading.Lock()

    def write(self, kind: str, **fields: Any) -> None:
        record = {"time": datetime.now(UTC).isoformat(), **self.base, "kind": kind, **fields}
        line = dumps(record)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


def read_journal(path: Path, limit: int = 500) -> list[dict[str, Any]]:
    """Последние limit строк журнала (испорченные строки пропускаются)."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps(r) + "\n" for r in records), encoding="utf-8")
