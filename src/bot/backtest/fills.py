"""Срабатывание стопа и тейка внутри 1m-свечи.

Правила (docs/tasks/stage-2.md): свеча открылась за стопом или тейком (гэп) —
исполнение по open; стоп и тейк задеты в одной свече — считаем, что сработал стоп.
Возвращается цена срабатывания до проскальзывания.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bot.core.models import Side


class ExitReason(StrEnum):
    STOP = "stop"
    TAKE = "take"
    TIME = "time"
    END = "end_of_data"


@dataclass(frozen=True)
class Hit:
    reason: ExitReason
    price: float  # до проскальзывания
    gap: bool  # исполнение по open из-за гэпа


def check_exit(
    side: Side,
    stop: float,
    take: float | None,
    open_: float,
    high: float,
    low: float,
) -> Hit | None:
    if side is Side.LONG:
        if open_ <= stop:
            return Hit(ExitReason.STOP, open_, gap=True)
        if take is not None and open_ >= take:
            return Hit(ExitReason.TAKE, open_, gap=True)
        if low <= stop:
            return Hit(ExitReason.STOP, stop, gap=False)
        if take is not None and high >= take:
            return Hit(ExitReason.TAKE, take, gap=False)
        return None

    if open_ >= stop:
        return Hit(ExitReason.STOP, open_, gap=True)
    if take is not None and open_ <= take:
        return Hit(ExitReason.TAKE, open_, gap=True)
    if high >= stop:
        return Hit(ExitReason.STOP, stop, gap=False)
    if take is not None and low <= take:
        return Hit(ExitReason.TAKE, take, gap=False)
    return None
