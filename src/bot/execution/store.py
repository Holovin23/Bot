"""Состояние живого бота в SQLite (data/bot.sqlite): сделки, ордера, события, состояние
риска и флаг аварийной остановки.

Правило из docs/05: запись в базу — до и после каждого сетевого вызова. Тогда после
перезапуска бот знает, какие ордера могли уйти на биржу, и сверяет их (reconcile).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from typing import Any

from bot.core.models import Order, OrderRequest, OrderStatus, Signal
from bot.risk.limits import RiskState

KILL_KEY = "kill"
RISK_STATE_KEY = "risk_state"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class LiveTrade:
    signal_id: str
    mode: str
    strategy: str
    symbol: str
    timeframe: str
    side: str
    qty: float
    entry_price: float | None
    stop: float
    take: float | None
    time_stop_bars: int | None
    bars_held: int
    risk_amount: float
    status: str  # pending | open | closed | failed
    opened_at: str | None
    exit_price: float | None
    exit_time: str | None
    exit_reason: str | None
    net_pnl: float | None
    signal_json: str

    @property
    def signal(self) -> Signal:
        return Signal.model_validate_json(self.signal_json)


class Store:
    def __init__(self, conn: sqlite3.Connection, mode: str) -> None:
        self.conn = conn
        self.mode = mode

    # --- сделки ---------------------------------------------------------------

    def add_trade(self, signal: Signal, qty: float, risk_amount: float) -> LiveTrade:
        """Запись ДО отправки ордера: статус pending."""
        self.conn.execute(
            "INSERT INTO live_trades (signal_id, mode, strategy, symbol, timeframe, side, qty, "
            "stop, take, time_stop_bars, risk_amount, status, signal_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (
                signal.signal_id,
                self.mode,
                signal.strategy,
                signal.symbol,
                signal.timeframe,
                signal.side.value,
                qty,
                signal.stop_loss,
                signal.take_profit,
                signal.time_stop_bars,
                risk_amount,
                signal.model_dump_json(),
            ),
        )
        self.conn.commit()
        trade = self.get_trade(signal.signal_id)
        assert trade is not None
        return trade

    def update_trade(self, signal_id: str, **fields: Any) -> None:
        allowed = set(LiveTrade.__dataclass_fields__) - {"signal_id", "mode", "signal_json"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"неизвестные поля сделки: {sorted(unknown)}")
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE live_trades SET {sets} WHERE signal_id = ?",  # имена полей — только из allowed
            (*fields.values(), signal_id),
        )
        self.conn.commit()

    def get_trade(self, signal_id: str) -> LiveTrade | None:
        row = self.conn.execute(
            "SELECT * FROM live_trades WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        return LiveTrade(**dict(row)) if row else None

    def trades(self, status: str | None = None, limit: int = 1000) -> list[LiveTrade]:
        if status is None:
            rows = self.conn.execute(
                "SELECT * FROM live_trades WHERE mode = ? ORDER BY rowid DESC LIMIT ?",
                (self.mode, limit),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM live_trades WHERE mode = ? AND status = ? ORDER BY rowid DESC "
                "LIMIT ?",
                (self.mode, status, limit),
            )
        return [LiveTrade(**dict(r)) for r in rows.fetchall()]

    def active_trades(self) -> list[LiveTrade]:
        """Открытые и ещё не подтверждённые (pending) сделки."""
        return [t for t in self.trades() if t.status in ("pending", "open")]

    # --- ордера: машина состояний ------------------------------------------------

    def order_sent(self, request: OrderRequest) -> None:
        """Перед сетевым вызовом: NEW → SENT."""
        now = _now()
        self.conn.execute(
            "INSERT INTO orders (order_link_id, signal_id, symbol, side, order_type, qty, "
            "reduce_only, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(order_link_id) DO UPDATE SET status = excluded.status, "
            "updated_at = excluded.updated_at",
            (
                request.order_link_id,
                request.signal_id,
                request.symbol,
                request.side.value,
                request.order_type.value,
                request.qty,
                int(request.reduce_only),
                OrderStatus.SENT.value,
                now,
                now,
            ),
        )
        self.conn.commit()

    def order_result(self, order: Order) -> None:
        """После ответа биржи: ACKNOWLEDGED / FILLED / ..."""
        self.conn.execute(
            "UPDATE orders SET status = ?, order_id = ?, filled_qty = ?, avg_price = ?, "
            "updated_at = ? WHERE order_link_id = ?",
            (
                order.status.value,
                order.order_id,
                order.filled_qty,
                order.avg_fill_price,
                _now(),
                order.order_link_id,
            ),
        )
        self.conn.commit()

    def order_failed(self, order_link_id: str, error: str) -> None:
        self.conn.execute(
            "UPDATE orders SET status = ?, error = ?, updated_at = ? WHERE order_link_id = ?",
            (OrderStatus.REJECTED.value, error[:500], _now(), order_link_id),
        )
        self.conn.commit()

    def order_status(self, order_link_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM orders WHERE order_link_id = ?", (order_link_id,)
        ).fetchone()
        return str(row["status"]) if row else None

    def unconfirmed_orders(self) -> list[tuple[str, str]]:
        """(orderLinkId, символ) ордеров, ушедших на биржу без ответа (SENT):
        их надо найти при reconcile."""
        rows = self.conn.execute(
            "SELECT order_link_id, symbol FROM orders WHERE status = ?",
            (OrderStatus.SENT.value,),
        ).fetchall()
        return [(str(r["order_link_id"]), str(r["symbol"])) for r in rows]

    # --- события ---------------------------------------------------------------

    def add_event(self, kind: str, message: str, level: str = "info") -> None:
        self.conn.execute(
            "INSERT INTO events (time, level, kind, message) VALUES (?, ?, ?, ?)",
            (_now(), level, kind, message),
        )
        self.conn.commit()

    def events(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT time, level, kind, message FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- состояние риска и kill switch ------------------------------------------

    def get_value(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_value(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_risk_state(self) -> RiskState | None:
        raw = self.get_value(f"{RISK_STATE_KEY}:{self.mode}")
        if raw is None:
            return None
        data = json.loads(raw)
        data["day"] = date.fromisoformat(data["day"])
        return RiskState(**data)

    def set_risk_state(self, state: RiskState) -> None:
        data = asdict(state)
        data["day"] = state.day.isoformat()
        self.set_value(f"{RISK_STATE_KEY}:{self.mode}", json.dumps(data, ensure_ascii=False))

    def request_kill(self, reason: str) -> None:
        self.set_value(KILL_KEY, reason or "аварийная остановка")

    def kill_reason(self) -> str | None:
        value = self.get_value(KILL_KEY)
        return value or None

    def clear_kill(self) -> None:
        self.conn.execute("DELETE FROM kv WHERE key = ?", (KILL_KEY,))
        self.conn.commit()
