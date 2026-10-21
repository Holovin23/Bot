"""Проверка стратегии на новых данных (этап 4).

    uv run python -m bot.research.validate --strategy telegram_scenario --symbol BTCUSDT \\
        --timeframe 1h --stage validation
    ... --stage walk-forward --grid adx_max=15,20,25 --grid bb_std=2,2.5
    ... --stage holdout            (один раз!)

Отчёт: data/backtests/<время>_<stage>-<стратегия>_<символ>_<таймфрейм>/validation.md
и trades.csv со сделками вне выборки (их можно открыть на графике).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bot.backtest.engine import BacktestResult, BacktestTrade
from bot.backtest.report import _f, _parse_params, trades_frame
from bot.backtest.walk_forward import backtest_window, make_windows, run_walk_forward
from bot.core.config import PROJECT_ROOT, Config, load_config
from bot.data.storage import Storage
from bot.research.grid import parse_grid, to_markdown
from bot.research.journal import JournalEntry, append_entry
from bot.research.periods import (
    HOLDOUT,
    IN_SAMPLE,
    VALIDATION,
    WALK_FORWARD,
    HoldoutAlreadyOpenedError,
    Period,
    open_holdout,
)
from bot.research.robustness import MonteCarlo, TradeStats, monte_carlo_drawdown, trade_stats
from bot.strategies.registry import create_strategy

# Критерии OOS (docs/01-goals.md)
OOS_MIN_PF = 1.1
OOS_MAX_DD_RATIO = 1.5
OOS_MIN_TRADES = 30  # меньше — вывод почти ничего не значит (04, раздел 5)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def oos_checks(is_stats: TradeStats, oos: TradeStats) -> list[Check]:
    """Критерии из 01: PF > 1,1, Expectancy > 0, просадка не хуже 1,5 × IS."""
    dd_limit = is_stats.max_drawdown_pct * OOS_MAX_DD_RATIO
    return [
        Check(
            f"PF > {OOS_MIN_PF:g}",
            oos.profit_factor > OOS_MIN_PF,
            f"PF {_f(oos.profit_factor)}",
        ),
        Check("Expectancy > 0", oos.expectancy_r > 0, f"{oos.expectancy_r:.3f}R"),
        Check(
            f"Max DD ≤ {OOS_MAX_DD_RATIO:g} × IS",
            oos.max_drawdown_pct <= dd_limit,
            f"{_f(oos.max_drawdown_pct, 1)}% при IS {_f(is_stats.max_drawdown_pct, 1)}% "
            f"(предел {_f(dd_limit, 1)}%)",
        ),
        Check(
            f"≥ {OOS_MIN_TRADES} сделок",
            oos.trades >= OOS_MIN_TRADES,
            f"{oos.trades} сделок",
        ),
    ]


def _stats_line(name: str, s: TradeStats) -> str:
    return (
        f"| {name} | {s.trades} | {_f(s.profit_factor)} | {s.expectancy_r:.3f} | "
        f"{_f(s.net_pnl)} | {_f(s.max_drawdown_pct, 1)}% |"
    )


def render_validation(
    title: str,
    rows: list[tuple[str, TradeStats]],
    checks: list[Check],
    mc: MonteCarlo,
    windows: pd.DataFrame | None = None,
) -> str:
    lines = [
        f"# Проверка вне выборки: {title}",
        "",
        "| Отрезок | Сделок | PF | Expectancy, R | Результат, USDT | Max DD |",
        "|---|---|---|---|---|---|",
        *[_stats_line(name, s) for name, s in rows],
        "",
        "Max DD здесь считается по капиталу после каждой сделки (без колебаний внутри сделки).",
        "",
        "## Критерии (docs/01-goals.md)",
        "",
        *[f"- {'✓' if c.passed else '✗'} {c.name}: {c.detail}" for c in checks],
        "",
        f"Итог: {'все критерии выполнены' if all(c.passed for c in checks) else 'не прошла'}. "
        "Решение принимает владелец.",
        "",
        "## Монте-Карло (перестановка сделок вне выборки)",
        "",
        f"- {mc.runs} перестановок. Max DD в исходном порядке {_f(mc.original_dd_pct, 1)}%, "
        f"медиана {_f(mc.median_dd_pct, 1)}%, 95% — {_f(mc.p95_dd_pct, 1)}%, "
        f"худшая {_f(mc.worst_dd_pct, 1)}%.",
        "- Ориентир для лимитов: просадка 95-го перцентиля должна укладываться в лимит 10% "
        "из docs/05.",
        "",
    ]
    if windows is not None and not windows.empty:
        lines += ["## Окна walk-forward", "", to_markdown(windows), ""]
    return "\n".join(lines)


def _period_trades(
    cfg: Config,
    storage: Storage,
    name: str,
    tf: str,
    symbol: str,
    params: dict[str, Any],
    period: Period,
) -> list[BacktestTrade]:
    candles = storage.read_candles(symbol, "1m", period.start - pd.Timedelta(days=31), period.end)
    if candles.empty:
        raise SystemExit(f"Нет 1m-свечей {symbol} за {period.label()}. Загрузите историю.")
    funding = storage.read_funding(symbol, period.start - pd.Timedelta(days=31), period.end)
    return backtest_window(
        name, tf, symbol, candles, cfg, params, period.start, period.end, funding
    )


def _save(
    cfg: Config,
    folder_name: str,
    report: str,
    trades: list[BacktestTrade],
    meta: tuple[str, str, str],
) -> Path:
    folder = cfg.resolve(cfg.data.dir) / "backtests" / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "validation.md").write_text(report, encoding="utf-8")
    name, symbol, tf = meta
    holder = BacktestResult(
        name, "", {}, symbol, tf, cfg.costs.initial_capital, trades, pd.DataFrame()
    )
    trades_frame(holder).to_csv(folder / "trades.csv", index=False)
    return folder / "validation.md"


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Проверка стратегии вне выборки")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument(
        "--stage", choices=["validation", "walk-forward", "holdout"], default="validation"
    )
    parser.add_argument("--param", action="append", help="параметр стратегии: ключ=значение")
    parser.add_argument("--grid", action="append", help="для walk-forward: ключ=v1,v2,...")
    parser.add_argument("--other-symbol", help="второй инструмент (по умолчанию другой из конфига)")
    parser.add_argument("--journal", default="research/experiments.md")
    parser.add_argument("--holdout-log", default="research/holdout-log.md")
    parser.add_argument(
        "--i-know-holdout-is-burned",
        dest="force_holdout",
        action="store_true",
        help="разрешить повторный взгляд на hold-out",
    )
    args = parser.parse_args(argv)

    params = _parse_params(args.param)
    grid = parse_grid(args.grid)
    if grid and args.stage != "walk-forward":
        parser.error("--grid используется только с --stage walk-forward")
    if args.symbol not in cfg.instruments:
        raise SystemExit(f"Нет параметров инструмента {args.symbol} в config/default.yaml")
    storage = Storage(cfg.resolve(cfg.data.dir))
    name, symbol, tf = args.strategy, args.symbol, args.timeframe
    full_params = create_strategy(name, tf, params).params
    params_text = ", ".join(f"{k}={v}" for k, v in full_params.items())

    if args.stage == "holdout":
        try:
            open_holdout(
                cfg.resolve(Path(args.holdout_log)),
                name,
                symbol,
                tf,
                params_text,
                force=args.force_holdout,
            )
        except HoldoutAlreadyOpenedError as exc:
            raise SystemExit(str(exc)) from None
        print("Hold-out открыт и записан в журнал. Больше не меняйте эту стратегию по его итогам.")

    capital = cfg.costs.initial_capital
    print(f"In-sample {name} {symbol} {tf}…", flush=True)
    is_stats = trade_stats(
        _period_trades(cfg, storage, name, tf, symbol, params, IN_SAMPLE), capital
    )
    windows_table = None
    if args.stage == "walk-forward":
        candles = storage.read_candles(symbol, "1m", WALK_FORWARD.start, WALK_FORWARD.end)
        funding = storage.read_funding(symbol, WALK_FORWARD.start, WALK_FORWARD.end)
        windows = make_windows(WALK_FORWARD.start, WALK_FORWARD.end)
        if not windows:
            raise SystemExit("Нет ни одного окна walk-forward")
        period = Period("walk-forward OOS", windows[0].test_start, windows[-1].test_end)
        print(f"Walk-forward: {len(windows)} окон", flush=True)
        wf = run_walk_forward(
            name,
            tf,
            symbol,
            candles,
            cfg,
            windows,
            grid,
            params,
            funding,
            progress=lambda d, n: print(f"  окно {d}/{n}", flush=True),
        )
        oos_trades, windows_table = wf.trades, wf.windows
    else:
        period = VALIDATION if args.stage == "validation" else HOLDOUT
        print(f"{period.label()}…", flush=True)
        oos_trades = _period_trades(cfg, storage, name, tf, symbol, params, period)
    oos = trade_stats(oos_trades, capital)
    rows = [(IN_SAMPLE.label(), is_stats), (period.label(), oos)]

    other = args.other_symbol or next((s for s in cfg.data.symbols if s != symbol), None)
    if other and other in cfg.instruments and args.stage != "walk-forward":
        try:
            other_trades = _period_trades(cfg, storage, name, tf, other, params, period)
            rows.append((f"{other}, {period.name}", trade_stats(other_trades, capital)))
        except SystemExit as exc:
            print(f"  {other}: {exc}")

    checks = oos_checks(is_stats, oos)
    mc = monte_carlo_drawdown([t.net_pnl for t in oos_trades], capital)
    title = f"{name} {symbol} {tf}, {args.stage}"
    report_text = render_validation(title, rows, checks, mc, windows_table)
    if grid:
        report_text += "\nСетка: " + "; ".join(f"{k} ∈ {v}" for k, v in grid.items()) + "\n"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report = _save(
        cfg,
        f"{stamp}_{args.stage}-{name}_{symbol}_{tf}",
        report_text,
        oos_trades,
        (name, symbol, tf),
    )

    passed = all(c.passed for c in checks)
    shown = report.relative_to(PROJECT_ROOT) if report.is_relative_to(PROJECT_ROOT) else report
    variants = 1
    if grid:
        variants = math.prod(len(v) for v in grid.values())
    number = append_entry(
        cfg.resolve(Path(args.journal)),
        JournalEntry(
            title=f"{name} {symbol} {tf} — {args.stage}",
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            hypothesis="та же, что в записи in-sample этой стратегии",
            rationale="проверка вне выборки (этап 4)",
            period=period.label(),
            params=params_text + (f"; сетка {grid}" if grid else ""),
            variants=variants,
            criteria=f"PF > {OOS_MIN_PF:g}, Expectancy > 0, Max DD ≤ {OOS_MAX_DD_RATIO:g} × IS",
            result=(
                f"сделок {oos.trades}, PF {_f(oos.profit_factor)}, "
                f"Expectancy {oos.expectancy_r:.3f}R, DD {_f(oos.max_drawdown_pct, 1)}%; "
                f"Монте-Карло DD 95% {_f(mc.p95_dd_pct, 1)}%"
            ),
            verdict="прошла"
            if passed
            else "не прошла: " + "; ".join(c.name for c in checks if not c.passed),
            report=shown.as_posix(),
        ),
    )
    for c in checks:
        print(f"  {'✓' if c.passed else '✗'} {c.name}: {c.detail}")
    print(f"EXP-{number:03d}. Отчёт: {report}")


if __name__ == "__main__":
    main()
