"""Состояние живого бота для вкладки «Бот» (читается из data/bot.sqlite).

Интерфейс и бот — разные процессы, поэтому команды передаются файлами в data/:
KILL — аварийная остановка (бот закроет позиции на следующей минуте), RESUME —
возобновить торговлю. Это тот же механизм, что ручной файл data/KILL.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from bot.core.db import connect, db_path
from bot.execution.store import Store

KILL_FILE = "KILL"
RESUME_FILE = "RESUME"


def bot_status(data_dir: Path, mode: str) -> dict[str, Any]:
    path = db_path(data_dir)
    kill_file = (data_dir / KILL_FILE).exists()
    if not path.exists():
        return {
            "mode": mode,
            "running": False,
            "kill_file": kill_file,
            "trades": [],
            "closed": [],
            "events": [],
            "risk": None,
            "kill": None,
        }
    conn = connect(path)
    try:
        store = Store(conn, mode)
        state = store.get_risk_state()
        risk = None
        if state is not None:
            risk = {**asdict(state), "day": state.day.isoformat()}
        fields = [
            "signal_id",
            "strategy",
            "symbol",
            "timeframe",
            "side",
            "qty",
            "entry_price",
            "stop",
            "take",
            "bars_held",
            "status",
            "opened_at",
            "exit_price",
            "exit_time",
            "exit_reason",
            "net_pnl",
        ]
        active = [{f: getattr(t, f) for f in fields} for t in store.active_trades()]
        closed = [{f: getattr(t, f) for f in fields} for t in store.trades("closed", limit=50)]
        return {
            "mode": mode,
            "running": state is not None,
            "kill_file": kill_file,
            "kill": store.kill_reason(),
            "risk": risk,
            "trades": active,
            "closed": closed,
            "events": store.events(100),
        }
    finally:
        conn.close()


def request_kill(data_dir: Path, reason: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / KILL_FILE).write_text(reason, encoding="utf-8")


def request_resume(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / RESUME_FILE).write_text("", encoding="utf-8")
