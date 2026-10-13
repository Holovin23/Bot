"""Стоп и цель сделки по сценарию. Автор редко называет их, поэтому есть значения по
умолчанию (решение владельца 2026-10-13, docs/10). Чистые функции, без сети и без ИИ.

Стоп: уровень автора; если его нет — за уровнем входа на stop_buffer_atr × ATR
(для зоны — за её дальним краем), для входа `now` — now_stop_atr × ATR от входа.
Цель: первая цель автора с R:R ≥ min_rr; если цели автора есть, но все ближе — сделки
нет (автор ждёт меньшего движения, чем нам нужно); если целей нет — вход ± default_rr × риск.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bot.scenarios.models import Scenario, reference_levels


@dataclass(frozen=True)
class LevelParams:
    stop_buffer_atr: float = 0.5
    now_stop_atr: float = 1.5
    default_rr: float = 2.0
    min_rr: float = 2.0


@dataclass(frozen=True)
class TradeLevels:
    stop: float
    take: float
    stop_from_author: bool
    take_from_author: bool


def trade_levels(
    scenario: Scenario, entry: float, atr_value: float, params: LevelParams | None = None
) -> TradeLevels | str:
    """Стоп и тейк для входа по цене entry или строка с причиной отказа."""
    p = params or LevelParams()
    long = scenario.side == "LONG"
    sign = 1.0 if long else -1.0
    low, high = reference_levels(scenario)
    if scenario.stop is not None:
        stop, stop_from_author = scenario.stop, True
    elif not (atr_value > 0):
        return "ATR неизвестен — стоп по умолчанию не посчитать"
    elif scenario.trigger.type == "now" or math.isnan(low):
        stop, stop_from_author = entry - sign * p.now_stop_atr * atr_value, False
    else:
        edge = low if long else high
        stop, stop_from_author = edge - sign * p.stop_buffer_atr * atr_value, False
    risk = sign * (entry - stop)
    if risk <= 0:
        return "цена входа уже за стопом"
    ahead = sorted((t for t in scenario.targets if sign * (t - entry) > 0), key=lambda t: sign * t)
    for target in ahead:
        if sign * (target - entry) / risk >= p.min_rr - 1e-9:
            return TradeLevels(stop, target, stop_from_author, True)
    if scenario.targets:
        return f"цели автора ближе {p.min_rr:g}R от входа"
    return TradeLevels(stop, entry + sign * p.default_rr * risk, stop_from_author, False)
