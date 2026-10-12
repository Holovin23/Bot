"""Этап 8: реальное исполнение против демо (проскальзывание, частичные исполнения).

    uv run python -m bot.research.live_vs_demo

Бот запускается одновременно на демо и на реальном счёте с одинаковыми стратегиями,
поэтому один и тот же сигнал (signal_id) торгуется на обоих счетах. Сравниваем цены
входа и выхода в базисных пунктах (1 б.п. = 0,01%) и долю частично исполненных ордеров.
Если реальное исполнение систематически хуже демо, оценки бэктеста надо ухудшить
(параметр slippage в config/default.yaml), а не наоборот.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from bot.core.config import load_config
from bot.core.db import connect, db_path
from bot.execution.store import LiveTrade, Store


@dataclass(frozen=True)
class ExecutionDiff:
    signal_id: str
    side: str
    entry_bps: float  # >0 — на реальном счёте вход хуже
    exit_bps: float | None  # >0 — на реальном счёте выход хуже


@dataclass
class ExecutionReport:
    pairs: list[ExecutionDiff]
    only_live: list[str]
    only_demo: list[str]
    partial_live: int
    orders_live: int
    partial_demo: int
    orders_demo: int

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @property
    def mean_entry_bps(self) -> float | None:
        return self._mean([p.entry_bps for p in self.pairs])

    @property
    def mean_exit_bps(self) -> float | None:
        return self._mean([p.exit_bps for p in self.pairs if p.exit_bps is not None])


def worse_bps(real: float, demo: float, buy: bool) -> float:
    """Насколько цена реального счёта хуже демо: при покупке хуже — дороже."""
    diff = (real - demo) / demo * 10_000
    return diff if buy else -diff


def partial_fills(orders: list[dict[str, Any]]) -> int:
    return sum(1 for o in orders if 0 < float(o["filled_qty"] or 0) < float(o["qty"]))


def compare_execution(
    live: list[LiveTrade],
    demo: list[LiveTrade],
    live_orders: list[dict[str, Any]],
    demo_orders: list[dict[str, Any]],
) -> ExecutionReport:
    real = {t.signal_id: t for t in live if t.entry_price}
    paper = {t.signal_id: t for t in demo if t.entry_price}
    pairs = []
    for signal_id in sorted(real.keys() & paper.keys()):
        r, d = real[signal_id], paper[signal_id]
        assert r.entry_price is not None and d.entry_price is not None
        long = r.side == "LONG"
        exit_bps = None
        if r.exit_price and d.exit_price:
            exit_bps = worse_bps(r.exit_price, d.exit_price, buy=not long)
        pairs.append(
            ExecutionDiff(
                signal_id=signal_id,
                side=r.side,
                entry_bps=worse_bps(r.entry_price, d.entry_price, buy=long),
                exit_bps=exit_bps,
            )
        )
    return ExecutionReport(
        pairs=pairs,
        only_live=sorted(real.keys() - paper.keys()),
        only_demo=sorted(paper.keys() - real.keys()),
        partial_live=partial_fills(live_orders),
        orders_live=len(live_orders),
        partial_demo=partial_fills(demo_orders),
        orders_demo=len(demo_orders),
    )


def render(report: ExecutionReport) -> str:
    def bps(value: float | None) -> str:
        return "—" if value is None else f"{value:+.1f} б.п."

    lines = [
        "# Реальный счёт против демо",
        "",
        f"- Общих сделок: {len(report.pairs)}; только на реальном: {len(report.only_live)}, "
        f"только на демо: {len(report.only_demo)}",
        f"- Вход на реальном хуже демо в среднем на {bps(report.mean_entry_bps)}",
        f"- Выход на реальном хуже демо в среднем на {bps(report.mean_exit_bps)}",
        f"- Частичные исполнения: реальный {report.partial_live} из {report.orders_live} "
        f"ордеров, демо {report.partial_demo} из {report.orders_demo}",
        "",
    ]
    if report.pairs:
        lines += ["| signal_id | Сторона | Вход | Выход |", "|---|---|---|---|"]
        lines += [
            f"| {p.signal_id} | {p.side} | {bps(p.entry_bps)} | {bps(p.exit_bps)} |"
            for p in report.pairs
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Реальное исполнение против демо").parse_args(argv)
    cfg = load_config()
    conn = connect(db_path(cfg.resolve(cfg.data.dir)))
    live, demo = Store(conn, "live"), Store(conn, "demo")
    report = compare_execution(
        live.trades(limit=100_000),
        demo.trades(limit=100_000),
        live.orders(100_000),
        demo.orders(100_000),
    )
    print(render(report))


if __name__ == "__main__":
    main()
