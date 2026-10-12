"""Сохранённые бэктесты (data/backtests/<время>_<стратегия>_<символ>_<таймфрейм>/trades.csv)
для показа сделок на графике."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

FOLDER_RE = re.compile(
    r"^(?P<stamp>\d{8}-\d{6})_(?P<strategy>.+)_(?P<symbol>[A-Z0-9]+)_(?P<tf>\w+)$"
)


def _meta(folder: Path) -> dict[str, Any] | None:
    match = FOLDER_RE.match(folder.name)
    if match is None or not (folder / "trades.csv").is_file():
        return None
    return {
        "id": folder.name,
        "created": match["stamp"],
        "strategy": match["strategy"],
        "symbol": match["symbol"],
        "timeframe": match["tf"],
    }


def list_backtests(root: Path) -> list[dict[str, Any]]:
    """Бэктесты со сделками, новые сверху. Папки перебора (grid) без trades.csv пропускаются."""
    if not root.is_dir():
        return []
    items = [m for m in (_meta(p) for p in root.iterdir() if p.is_dir()) if m is not None]
    return sorted(items, key=lambda m: m["id"], reverse=True)


def _seconds(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(pd.Timestamp(value).timestamp())


def _number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 8)


def load_backtest(root: Path, backtest_id: str) -> dict[str, Any] | None:
    """Метаданные и сделки бэктеста; None — нет такого. Время — секунды UTC."""
    if FOLDER_RE.match(backtest_id) is None:  # заодно не пускает «..» и «/»
        return None
    meta = _meta(root / backtest_id)
    if meta is None:
        return None
    table = pd.read_csv(root / backtest_id / "trades.csv")
    trades = []
    for number, row in enumerate(table.to_dict(orient="records"), start=1):
        trades.append(
            {
                "number": number,
                "signal_id": str(row["signal_id"]),
                "side": str(row["side"]),
                "entry_time": _seconds(row["entry_time"]),
                "exit_time": _seconds(row["exit_time"]),
                "entry_price": _number(row["entry_price"]),
                "exit_price": _number(row["exit_price"]),
                "initial_stop": _number(row["initial_stop"]),
                "stop": _number(row["stop"]),
                "take": _number(row["take"]),
                "exit_reason": str(row["exit_reason"]),
                "r_multiple": _number(row["r_multiple"]),
                "net_pnl": _number(row["net_pnl"]),
            }
        )
    return {**meta, "trades": trades}
