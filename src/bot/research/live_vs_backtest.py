"""Сравнение сделок живого бота с бэктестом на тех же свечах (этап 7, 03).

    uv run python -m bot.research.live_vs_backtest --mode paper

Для каждой стратегии из config/live.yaml: бэктест за период работы бота по сохранённым
1m-свечам, сопоставление сделок по signal_id. Показывает, сколько сигналов совпало,
насколько отличаются цены входа и выхода и результат. Большие расхождения — повод
искать ошибку (заглядывание в будущее в бэктесте, задержки или проскальзывание живого
исполнения), а не подгонять параметры.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from bot.backtest.engine import BacktestTrade
from bot.backtest.walk_forward import strategy_window
from bot.core.config import load_config
from bot.core.db import connect, db_path
from bot.data.storage import Storage
from bot.execution.store import LiveTrade, Store
from bot.runner.live import build_strategy
from bot.runner.settings import load_live_settings


@dataclass(frozen=True)
class MatchedTrade:
    signal_id: str
    live_entry: float
    bt_entry: float
    live_exit: float | None
    bt_exit: float
    live_pnl: float | None
    bt_pnl: float
    live_reason: str | None
    bt_reason: str

    @property
    def entry_diff_bps(self) -> float:
        """Насколько живой вход хуже (+) или лучше (−) бэктеста, в б.п. (0,01%)."""
        return (self.live_entry - self.bt_entry) / self.bt_entry * 10_000


@dataclass
class Comparison:
    matched: list[MatchedTrade] = field(default_factory=list)
    only_live: list[str] = field(default_factory=list)
    only_backtest: list[str] = field(default_factory=list)

    @property
    def match_share(self) -> float:
        total = len(self.matched) + len(self.only_live) + len(self.only_backtest)
        return len(self.matched) / total if total else 0.0

    @property
    def same_exit_share(self) -> float:
        if not self.matched:
            return 0.0
        same = sum(1 for m in self.matched if m.live_reason == m.bt_reason)
        return same / len(self.matched)

    @property
    def pnl_diff(self) -> float:
        return sum((m.live_pnl or 0.0) - m.bt_pnl for m in self.matched)


def compare(live: list[LiveTrade], backtest: list[BacktestTrade]) -> Comparison:
    """Сопоставление по signal_id. Учитываются только закрытые живые сделки."""
    closed = {t.signal_id: t for t in live if t.status == "closed"}
    bt = {t.signal_id: t for t in backtest}
    result = Comparison()
    for signal_id in sorted(closed.keys() | bt.keys()):
        lt, bt_trade = closed.get(signal_id), bt.get(signal_id)
        if lt is None:
            result.only_backtest.append(signal_id)
        elif bt_trade is None:
            result.only_live.append(signal_id)
        else:
            result.matched.append(
                MatchedTrade(
                    signal_id=signal_id,
                    live_entry=lt.entry_price or 0.0,
                    bt_entry=bt_trade.entry_price,
                    live_exit=lt.exit_price,
                    bt_exit=bt_trade.exit_price,
                    live_pnl=lt.net_pnl,
                    bt_pnl=bt_trade.net_pnl,
                    live_reason=lt.exit_reason,
                    bt_reason=bt_trade.exit_reason,
                )
            )
    return result


def render(title: str, c: Comparison) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- Совпало сигналов: {len(c.matched)} ({c.match_share:.0%}); только у бота: "
        f"{len(c.only_live)}, только в бэктесте: {len(c.only_backtest)}",
        f"- Одинаковая причина выхода: {c.same_exit_share:.0%}",
        f"- Разница результата (бот − бэктест): {c.pnl_diff:+.2f} USDT",
    ]
    if c.matched:
        mean_bps = sum(m.entry_diff_bps for m in c.matched) / len(c.matched)
        lines.append(f"- Средняя разница цены входа: {mean_bps:+.1f} б.п.")
        lines += [
            "",
            "| signal_id | Вход бот / бэктест | Выход бот / бэктест | Результат бот / бэктест |",
            "|---|---|---|---|",
        ]
        for m in c.matched:
            lines.append(
                f"| {m.signal_id} | {m.live_entry:.2f} / {m.bt_entry:.2f} | "
                f"{m.live_exit or 0:.2f} ({m.live_reason}) / {m.bt_exit:.2f} ({m.bt_reason}) | "
                f"{m.live_pnl or 0:+.2f} / {m.bt_pnl:+.2f} |"
            )
    if c.only_live:
        lines.append(f"\nТолько у бота: {', '.join(c.only_live)}")
    if c.only_backtest:
        lines.append(f"\nТолько в бэктесте: {', '.join(c.only_backtest)}")
    return [*lines, ""]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Сделки бота против бэктеста")
    parser.add_argument("--mode", choices=["paper", "demo"], default="paper")
    parser.add_argument("--config", type=Path, help="настройки бота (config/live.yaml)")
    parser.add_argument("--out", type=Path, help="сохранить отчёт в файл .md")
    args = parser.parse_args(argv)
    cfg = load_config()
    settings = load_live_settings(args.config) if args.config else load_live_settings()
    data_dir = cfg.resolve(cfg.data.dir)
    storage = Storage(data_dir)
    store = Store(connect(db_path(data_dir)), args.mode)
    lines = [f"# Бот ({args.mode}) против бэктеста", ""]
    for instance in settings.instances:
        strategy = build_strategy(instance, data_dir / "models")
        live = [
            t
            for t in store.trades(limit=100_000)
            if t.symbol == instance.symbol and t.strategy == strategy.name
        ]
        title = f"{strategy.name} {instance.symbol} {instance.timeframe}"
        if not live:
            lines += [f"## {title}", "", "Сделок бота пока нет.", ""]
            continue
        start = min(pd.Timestamp(t.signal.created_at) for t in live)
        end = pd.Timestamp.now(tz="UTC")
        candles = storage.read_candles(instance.symbol, "1m", start - pd.Timedelta(days=31), end)
        funding = storage.read_funding(instance.symbol, start - pd.Timedelta(days=31), end)
        backtest = strategy_window(strategy, instance.symbol, candles, cfg, start, end, funding)
        lines += render(title, compare(live, backtest))
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
