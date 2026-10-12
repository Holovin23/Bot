"""Отрезки истории (docs/09-start-checklist.md) и защита hold-out.

Hold-out открывается один раз: каждый запуск записывается в research/holdout-log.md,
и повторный запуск той же стратегии на том же символе и таймфрейме требует явного флага.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp  # включительно
    end: pd.Timestamp  # не включительно

    def label(self) -> str:
        last_day = self.end - pd.Timedelta(days=1)
        return f"{self.name} {self.start:%Y-%m-%d} — {last_day:%Y-%m-%d}"


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


IN_SAMPLE = Period("in-sample", _ts("2021-01-01"), _ts("2024-07-01"))
VALIDATION = Period("validation", _ts("2024-07-01"), _ts("2025-07-01"))
HOLDOUT = Period("holdout", _ts("2025-07-01"), _ts("2026-10-01"))
# Walk-forward идёт внутри in-sample + validation
WALK_FORWARD = Period("walk-forward", IN_SAMPLE.start, VALIDATION.end)
PERIODS = {p.name: p for p in (IN_SAMPLE, VALIDATION, HOLDOUT, WALK_FORWARD)}

HOLDOUT_LOG_HEADER = (
    "# Журнал открытий hold-out\n\n"
    "Каждая строка — один взгляд на hold-out. Если после взгляда стратегия менялась, "
    "hold-out для неё «сгорел».\n\n"
    "| Дата (UTC) | Стратегия | Символ | Таймфрейм | Параметры |\n"
    "|---|---|---|---|---|\n"
)


class HoldoutAlreadyOpenedError(RuntimeError):
    pass


def touches_holdout(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Пересекается ли [start, end) с hold-out."""
    return start < HOLDOUT.end and end > HOLDOUT.start


def holdout_opened(log: Path, strategy: str, symbol: str, timeframe: str) -> bool:
    if not log.exists():
        return False
    needle = f"| {strategy} | {symbol} | {timeframe} |"
    return any(needle in line for line in log.read_text(encoding="utf-8").splitlines())


def open_holdout(
    log: Path,
    strategy: str,
    symbol: str,
    timeframe: str,
    params: str,
    force: bool = False,
    now: datetime | None = None,
) -> None:
    """Записать открытие hold-out. Повторное открытие без force — ошибка."""
    if holdout_opened(log, strategy, symbol, timeframe) and not force:
        raise HoldoutAlreadyOpenedError(
            f"Hold-out для {strategy} {symbol} {timeframe} уже открывался (см. {log}). "
            "Повторный взгляд превращает его в in-sample."
        )
    text = log.read_text(encoding="utf-8") if log.exists() else HOLDOUT_LOG_HEADER
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        text + f"| {stamp} | {strategy} | {symbol} | {timeframe} | {params} |\n", encoding="utf-8"
    )
