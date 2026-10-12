"""Расчёт объёма позиции (docs/05-risk-and-execution.md, «Расчёт размера позиции»).

    риск_в_деньгах  = капитал × риск_на_сделку
    дистанция_стопа = |вход − стоп| + оценка проскальзывания
    объём           = риск_в_деньгах / дистанция_стопа, вниз до шага лота
Плечо — следствие расчёта; лимит плеча только ограничивает объём сверху.
Чистые функции, без сети и без ИИ.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal


@dataclass(frozen=True)
class SizeResult:
    qty: float  # 0 — сделку открывать нельзя (см. reason)
    risk_amount: float  # сколько USDT потеряем при срабатывании стопа (с проскальзыванием)
    notional: float
    leverage: float
    capped_by_leverage: bool
    reason: str = ""


def round_down(value: float, step: float) -> float:
    """Округление вниз до шага: 0.10416 при шаге 0.001 -> 0.104."""
    if step <= 0:
        raise ValueError("Шаг должен быть > 0")
    steps = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
    return float(steps * Decimal(str(step)))


def position_size(
    *,
    equity: float,
    risk_fraction: float,
    entry_price: float,
    stop_loss: float,
    slippage_per_unit: float,
    qty_step: float,
    min_qty: float,
    max_leverage: float,
) -> SizeResult:
    if equity <= 0:
        return SizeResult(0.0, 0.0, 0.0, 0.0, False, "капитал не положительный")
    distance = abs(entry_price - stop_loss) + slippage_per_unit
    if abs(entry_price - stop_loss) <= 0:
        return SizeResult(0.0, 0.0, 0.0, 0.0, False, "стоп совпадает с входом")

    raw_qty = equity * risk_fraction / distance
    max_qty = equity * max_leverage / entry_price
    capped = raw_qty > max_qty
    qty = round_down(min(raw_qty, max_qty), qty_step)
    if qty < min_qty:
        return SizeResult(0.0, 0.0, 0.0, 0.0, capped, f"объём {qty} меньше минимального {min_qty}")
    notional = qty * entry_price
    return SizeResult(
        qty=qty,
        risk_amount=qty * distance,
        notional=notional,
        leverage=notional / equity,
        capped_by_leverage=capped,
    )
