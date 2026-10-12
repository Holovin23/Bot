"""Сверка состояния бота с биржей (docs/05, Reconciliation). При старте и каждые 5 минут.

1. Ордера, ушедшие без ответа (SENT): найти по orderLinkId и записать итог.
2. Сделки pending: есть позиция на бирже → open; нет → вход не состоялся (failed)
   или позиция уже закрылась стопом (closed).
3. Сделки open, которых нет на бирже: закрыты стопом/тейком, пока бот не следил → closed.
4. Позиции на бирже без стопа → стоп немедленно (из сделки; у неизвестной позиции —
   аварийный стоп unknown_stop_pct от цены входа).
5. Неизвестная позиция (нет сделки в базе) → предупреждение; закрытие — только при
   close_unknown=true.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from bot.core.models import Position, Side
from bot.execution.broker import (
    Broker,
    BrokerError,
    ExitInfo,
    close_request,
    submit_with_retry,
)
from bot.execution.store import LiveTrade, Store


@dataclass
class ReconcileReport:
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.actions and not self.warnings


def exit_reason(trade: LiveTrade, info: ExitInfo) -> str:
    """Причина по цене выхода: ближе к стопу — stop, к тейку — take."""
    if info.reason != "unknown":
        return info.reason
    if trade.take is None:
        return "stop"
    to_stop = abs(info.exit_price - trade.stop)
    to_take = abs(info.exit_price - trade.take)
    return "stop" if to_stop <= to_take else "take"


def emergency_stop(position: Position, pct: float) -> float:
    if position.side is Side.LONG:
        return position.entry_price * (1 - pct)
    return position.entry_price * (1 + pct)


def mark_closed(store: Store, broker: Broker, trade: LiveTrade) -> str:
    info = broker.last_exit(trade.symbol)
    if info is None:
        store.update_trade(trade.signal_id, status="closed", exit_reason="unknown")
        return f"{trade.signal_id}: позиции нет на бирже, итог неизвестен"
    store.update_trade(
        trade.signal_id,
        status="closed",
        exit_price=info.exit_price,
        exit_time=info.exit_time.isoformat(),
        exit_reason=exit_reason(trade, info),
        net_pnl=info.net_pnl,
    )
    return f"{trade.signal_id}: закрыта на бирже ({info.net_pnl:+.2f} USDT)"


def reconcile(
    broker: Broker,
    store: Store,
    *,
    close_unknown: bool = False,
    unknown_stop_pct: float = 0.02,
) -> ReconcileReport:
    report = ReconcileReport()
    trades = store.active_trades()

    # 1. Ордера без ответа
    for link_id, symbol in store.unconfirmed_orders():
        found = broker.find_order(symbol, link_id)
        if found is not None:
            store.order_result(found)
            report.actions.append(f"ордер {link_id}: на бирже {found.status.value}")
        else:
            store.order_failed(link_id, "не найден на бирже при сверке")
            report.actions.append(f"ордер {link_id}: не дошёл до биржи")

    positions = {p.symbol: p for p in broker.get_positions()}

    # 2–3. Сделки бота против позиций биржи
    for trade in trades:
        pos = positions.get(trade.symbol)
        same_side = pos is not None and pos.side.value == trade.side
        if trade.status == "pending":
            if same_side:
                assert pos is not None
                store.update_trade(
                    trade.signal_id,
                    status="open",
                    entry_price=pos.entry_price,
                    opened_at=(pos.opened_at.isoformat() if pos.opened_at else None),
                )
                report.actions.append(f"{trade.signal_id}: вход подтверждён")
            elif store.order_status(trade.signal_id) == "FILLED":
                report.actions.append(mark_closed(store, broker, trade))
            else:
                store.update_trade(trade.signal_id, status="failed")
                report.actions.append(f"{trade.signal_id}: вход не состоялся")
        elif not same_side:
            report.actions.append(mark_closed(store, broker, trade))

    # 4–5. Позиции биржи: стоп и неизвестные
    active = {t.symbol: t for t in store.active_trades()}
    for symbol, pos in positions.items():
        known: LiveTrade | None = active.get(symbol)
        if known is None:
            report.warnings.append(
                f"неизвестная позиция {symbol} {pos.side.value} {pos.size} "
                f"по {pos.entry_price} — бот её не открывал"
            )
            if close_unknown:
                try:
                    submit_with_retry(
                        broker,
                        close_request(
                            symbol, pos.side, pos.size, f"rc-{symbol}-{int(time.time())}"
                        ),
                    )
                    report.actions.append(f"неизвестная позиция {symbol} закрыта")
                    continue
                except BrokerError as exc:
                    report.warnings.append(f"не удалось закрыть {symbol}: {exc}")
        if pos.stop_loss is None:
            stop = known.stop if known is not None else emergency_stop(pos, unknown_stop_pct)
            take = known.take if known is not None else None
            try:
                broker.set_trading_stop(symbol, pos.side, stop, take)
                report.actions.append(f"{symbol}: позиция была без стопа — стоп {stop:.2f}")
            except BrokerError as exc:
                report.warnings.append(f"{symbol}: позиция без стопа, стоп не выставлен: {exc}")

    for text in report.actions:
        store.add_event("reconcile", text)
    for text in report.warnings:
        store.add_event("reconcile", text, level="warning")
    return report
