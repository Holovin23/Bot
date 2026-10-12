"""Журнал экспериментов research/experiments.md (шаблон — docs/04-backtesting-guide.md, раздел 1).

Каждый эксперимент получает номер EXP-NNN. Записи только дописываются в конец файла;
решение «принять / отклонить» владелец вписывает руками.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADER = (
    "# Журнал экспериментов\n\n"
    "Записи добавляет `python -m bot.research.experiment`. Строку «Решение владельца» "
    "заполняет владелец. Число проверенных вариантов пишется всегда.\n"
)

_EXP_RE = re.compile(r"^## EXP-(\d+)", re.MULTILINE)


@dataclass(frozen=True)
class JournalEntry:
    title: str
    date: str
    hypothesis: str
    rationale: str
    period: str
    params: str
    variants: int
    criteria: str
    result: str
    verdict: str
    report: str


def next_number(text: str) -> int:
    """Следующий номер EXP: максимум из уже записанных + 1."""
    numbers = [int(m) for m in _EXP_RE.findall(text)]
    return max(numbers, default=0) + 1


def render_entry(number: int, entry: JournalEntry) -> str:
    return "\n".join(
        [
            f"## EXP-{number:03d}  {entry.title}",
            f"Дата: {entry.date}",
            f"Гипотеза: {entry.hypothesis}",
            f"Почему это может работать: {entry.rationale}",
            f"Период: {entry.period}",
            f"Параметры: {entry.params}",
            f"Проверено вариантов: {entry.variants}",
            f"Критерий успеха: {entry.criteria}",
            f"Результат: {entry.result}",
            f"Предварительный вердикт: {entry.verdict}",
            "Решение владельца: ___ (принять / отклонить / доработать гипотезу, НЕ параметры)",
            f"Отчёт: {entry.report}",
            "",
        ]
    )


def append_entry(path: Path, entry: JournalEntry) -> int:
    """Дописывает запись в журнал и возвращает её номер."""
    text = path.read_text(encoding="utf-8") if path.exists() else HEADER
    number = next_number(text)
    if not text.endswith("\n"):
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n" + render_entry(number, entry), encoding="utf-8")
    return number
