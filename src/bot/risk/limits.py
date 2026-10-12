"""Проверки перед каждой сделкой и состояние риска (docs/05-risk-and-execution.md).

Чистые функции: без сети, без ИИ, без чтения файлов. Номера проверок — как в таблице 05.
Проверки самого сигнала (1, 2, 3, 12) — в signal_checks: они же работают в бэктесте,
чтобы бэктест и живой бот отклоняли одни и те же сигналы.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

import pandas as pd

from bot.backtest.costs import CostModel
from bot.core.config import InstrumentConfig, LimitsConfig, RiskConfig
from bot.core.models import Side, Signal
from bot.core.timeframes import timeframe_delta
from bot.risk.sizing import position_size

ATR_TARGET_WITHOUT_TP = 2.0  # для проверки 12 без тейка: потенциал = 2 ATR


@dataclass(frozen=True)
class OpenRisk:
    signal_id: str
    symbol: str
    side: Side
    risk_amount: float


@dataclass(frozen=True)
class RiskState:
    """Состояние счёта для лимитов 9–11 и kill switch. Хранится в SQLite."""

    equity: float
    peak_equity: float
    day: date
    day_start_equity: float
    loss_streak: int = 0
    halted: str | None = None  # причина остановки; снимается только вручную
    paused: str | None = None  # пауза до следующего дня (дневной убыток, серия убытков)


@dataclass
class Decision:
    approved: bool
    qty: float = 0.0
    risk_amount: float = 0.0
    reasons: list[str] = field(default_factory=list)  # почему отклонено
    notes: list[str] = field(default_factory=list)  # что изменено (уменьшен объём и т. п.)


def min_rr_for(strategy: str, limits: LimitsConfig) -> float:
    return limits.min_rr.get(strategy, limits.default_min_rr)


def signal_checks(
    signal: Signal,
    atr_value: float,
    costs: CostModel,
    limits: LimitsConfig,
) -> list[str]:
    """Проверки 1, 2, 3, 12 — только по самому сигналу. Пустой список — всё в порядке."""
    reasons = []
    entry, stop = signal.entry_price, signal.stop_loss
    long = signal.side is Side.LONG
    # 1. Стоп есть и с правильной стороны (Signal не создаётся иначе — проверка на всякий случай)
    if (long and stop >= entry) or (not long and stop <= entry):
        reasons.append("1: стоп с неправильной стороны от входа")
        return reasons
    risk = abs(entry - stop)
    # 2. R:R — только если есть тейк (стратегии с трейлингом без тейка не проверяются)
    rr = signal.reward_to_risk
    min_rr = min_rr_for(signal.strategy, limits)
    if rr is not None and rr < min_rr - 1e-9:
        reasons.append(f"2: R:R {rr:.2f} < {min_rr:g}")
    # 3. Стоп не ближе 0,5 ATR и не дальше 4 ATR
    if atr_value > 0:
        in_atr = risk / atr_value
        if in_atr < limits.min_stop_atr:
            reasons.append(f"3: стоп {in_atr:.2f} ATR ближе {limits.min_stop_atr:g} ATR")
        elif in_atr > limits.max_stop_atr:
            reasons.append(f"3: стоп {in_atr:.2f} ATR дальше {limits.max_stop_atr:g} ATR")
    else:
        reasons.append("3: ATR неизвестен")
    # 12. Издержки < 20% потенциальной прибыли
    tp = signal.take_profit
    potential = abs(tp - entry) if tp is not None else ATR_TARGET_WITHOUT_TP * atr_value
    rate = costs.slippage_rate(signal.symbol)
    cost = entry * 2 * (costs.taker_fee + rate)
    if potential <= 0 or cost / potential >= limits.max_cost_share:
        share = cost / potential if potential > 0 else float("inf")
        reasons.append(
            f"12: издержки {share:.0%} потенциальной прибыли (≥ {limits.max_cost_share:.0%})"
        )
    return reasons


def liquidation_ok(entry: float, stop: float, leverage: float, limits: LimitsConfig) -> bool:
    """Проверка 4 для isolated-маржи. Цена ликвидации ≈ вход × (1 ∓ 1/плечо ± ММ);
    расстояние до неё должно быть не меньше liq_buffer × расстояние до стопа. Объём на
    цену ликвидации в isolated не влияет, поэтому при нарушении сделка отклоняется."""
    liq_distance = 1.0 / leverage - limits.maintenance_margin
    stop_distance = abs(entry - stop) / entry
    return liq_distance >= limits.liq_buffer * stop_distance


def block_reason(state: RiskState, limits: LimitsConfig) -> str | None:
    """Проверки 9–11 и kill switch: можно ли вообще открывать сделки."""
    if state.halted:
        return f"торговля остановлена: {state.halted}"
    if state.paused:
        return f"пауза до следующего дня: {state.paused}"
    return None


def update_state(state: RiskState, equity: float, today: date, limits: LimitsConfig) -> RiskState:
    """Новый капитал → пик, новый день, дневной убыток (9), просадка от пика (10)."""
    if today != state.day:
        state = replace(state, day=today, day_start_equity=equity, paused=None)
    state = replace(state, equity=equity, peak_equity=max(state.peak_equity, equity))
    if state.day_start_equity > 0:
        day_loss = (state.day_start_equity - equity) / state.day_start_equity
        if day_loss >= limits.daily_loss and not state.paused:
            state = replace(state, paused=f"дневной убыток {day_loss:.1%}")
    if state.peak_equity > 0:
        drawdown = (state.peak_equity - equity) / state.peak_equity
        if drawdown >= limits.max_drawdown and not state.halted:
            state = replace(state, halted=f"просадка {drawdown:.1%} от пика (kill switch)")
    return state


def record_trade(state: RiskState, net_pnl: float, limits: LimitsConfig) -> RiskState:
    """Проверка 11: серия убытков подряд."""
    streak = state.loss_streak + 1 if net_pnl <= 0 else 0
    state = replace(state, loss_streak=streak)
    if streak >= limits.max_loss_streak and not state.paused:
        state = replace(state, paused=f"{streak} убытков подряд")
    return state


def _correlated(symbol: str, other: str, limits: LimitsConfig) -> bool:
    return any(symbol in g and other in g for g in limits.correlated_groups)


def evaluate(
    signal: Signal,
    *,
    state: RiskState,
    open_risks: list[OpenRisk],
    atr_value: float,
    last_candle_open: pd.Timestamp,
    now: pd.Timestamp,
    entry_price: float,
    instrument: InstrumentConfig,
    costs: CostModel,
    risk: RiskConfig,
    limits: LimitsConfig,
) -> Decision:
    """Все проверки 1–13 и размер позиции. entry_price — текущая цена для расчёта."""
    decision = Decision(approved=False)
    blocked = block_reason(state, limits)
    if blocked:
        decision.reasons.append(blocked)
        return decision
    decision.reasons += signal_checks(signal, atr_value, costs, limits)
    # 5. Дубль
    if any(r.signal_id == signal.signal_id for r in open_risks):
        decision.reasons.append("5: позиция по этому сигналу уже есть (дубль)")
    # 6. Лимит позиций
    if len(open_risks) >= limits.max_positions:
        decision.reasons.append(
            f"6: открыто {len(open_risks)} позиций (максимум {limits.max_positions})"
        )
    same_symbol = sum(1 for r in open_risks if r.symbol == signal.symbol)
    if same_symbol >= limits.max_positions_per_symbol:
        decision.reasons.append(f"6: по {signal.symbol} уже есть позиция")
    # 13. Свежесть данных: последняя закрытая свеча не старше stale_bars интервалов
    bar = timeframe_delta(signal.timeframe)
    age = now - (last_candle_open + bar)
    if age > bar * limits.stale_bars:
        decision.reasons.append(f"13: данные устарели на {age}")
    # 4. Ликвидация
    if not liquidation_ok(entry_price, signal.stop_loss, risk.max_leverage, limits):
        decision.reasons.append("4: стоп слишком близко к цене ликвидации")
    if decision.reasons:
        return decision

    # Размер: 8 — коррелированная позиция в ту же сторону уменьшает риск в 1,5 раза
    fraction = risk.risk_per_trade
    if any(
        r.side is signal.side
        and r.symbol != signal.symbol
        and _correlated(signal.symbol, r.symbol, limits)
        for r in open_risks
    ):
        fraction /= limits.correlated_weight
        decision.notes.append(f"8: коррелированная позиция — риск ÷ {limits.correlated_weight:g}")
    size = position_size(
        equity=state.equity,
        risk_fraction=fraction,
        entry_price=entry_price,
        stop_loss=signal.stop_loss,
        slippage_per_unit=signal.stop_loss * costs.slippage_rate(signal.symbol, stop=True),
        qty_step=instrument.qty_step,
        min_qty=instrument.min_qty,
        max_leverage=risk.max_leverage,
    )
    if size.qty <= 0:
        decision.reasons.append(f"размер: {size.reason}")
        return decision
    if size.capped_by_leverage:
        decision.notes.append(f"объём ограничен плечом {risk.max_leverage:g}x")
    # 7. Суммарный риск открытых позиций
    total = sum(r.risk_amount for r in open_risks) + size.risk_amount
    if total > limits.max_total_risk * state.equity + 1e-9:
        decision.reasons.append(
            f"7: суммарный риск {total / state.equity:.2%} > {limits.max_total_risk:.2%}"
        )
        return decision
    decision.approved = True
    decision.qty = size.qty
    decision.risk_amount = size.risk_amount
    return decision
