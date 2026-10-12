"""Эксперимент по стратегии на in-sample: бэктест, проверка на заглядывание вперёд,
сравнение со случайными входами и buy & hold, отчёт и запись в журнал.

    uv run python -m bot.research.experiment --strategy mean_reversion --symbol BTCUSDT \\
        --timeframe 1h
    uv run python -m bot.research.experiment --all

Validation и hold-out здесь не используются: период по умолчанию — in-sample
(docs/tasks/stage-3.md).
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestResult, run_backtest
from bot.backtest.lookahead import find_lookahead
from bot.backtest.metrics import Metrics, compute_metrics
from bot.backtest.report import ReportContext, _f, _parse_params, save_report
from bot.core.config import PROJECT_ROOT, Config, load_config
from bot.data.resample import resample_candles
from bot.data.storage import Storage
from bot.research.baseline import BaselineComparison, compare_with_random
from bot.research.grid import count_variants, grid_summary, parse_grid, render_grid, run_grid
from bot.research.journal import JournalEntry, append_entry
from bot.strategies.base import Strategy
from bot.strategies.registry import create_strategy

IN_SAMPLE_START = "2021-01-01"
IN_SAMPLE_END = "2024-06-30"  # включительно
ALL_STRATEGIES = ("trend_following", "mean_reversion", "breakout")
ALL_TIMEFRAMES = ("5m", "15m", "1h")

# Критерии in-sample (docs/01-goals.md, docs/04-backtesting-guide.md)
MIN_TRADES = 100
MIN_PROFIT_FACTOR = 1.2
MIN_RANDOM_PERCENTILE = 0.95  # лучше 95% случайных прогонов
CRITERIA_TEXT = (
    f"≥ {MIN_TRADES} сделок, PF > {MIN_PROFIT_FACTOR:g} после расходов на IS, "
    f"PF лучше {MIN_RANDOM_PERCENTILE:.0%} случайных входов, нет заглядывания вперёд; "
    "на OOS (этап 4) PF > 1.1"
)

LOOKAHEAD_BARS = 5000  # окно последних свечей для проверки обрезкой
LOOKAHEAD_CUT = 0.8  # обрезка на 80% окна

HYPOTHESES: dict[str, tuple[str, str]] = {
    "trend_following": (
        "в восходящем тренде (EMA50 > EMA200, close > EMA200) откат к EMA20 с закрытием "
        "выше неё продолжается движением по тренду; выход трейлингом 3 × ATR(22)",
        "тренды в крипте длятся долго, откат даёт вход с коротким стопом",
    ),
    "mean_reversion": (
        "в боковике (ADX < 20) закрытие за нижней полосой BB(20, 2) при RSI < 30 "
        "возвращается к средней линии в течение 24 свечей",
        "краткосрочные перекосы ликвидности и ликвидации без смены режима",
    ),
    "breakout": (
        "после сжатия (ширина BB в нижних 10% за 100 свечей) пробой максимума 20 свечей "
        "на объёме > 1,5 × среднего продолжается на 2R в течение 48 свечей",
        "сжатие волатильности сменяется её расширением, объём подтверждает пробой",
    ),
}


@dataclass
class Experiment:
    result: BacktestResult
    metrics: Metrics
    comparison: BaselineComparison | None
    lookahead: list[str]
    verdict: str = ""
    reasons: list[str] = field(default_factory=list)


def preliminary_verdict(
    metrics: Metrics, comparison: BaselineComparison | None, lookahead: list[str]
) -> tuple[str, list[str]]:
    """«кандидат» — все критерии IS выполнены, иначе «отклонить» с причинами.
    Окончательное решение принимает владелец."""
    reasons = []
    if lookahead:
        reasons.append(f"заглядывание вперёд ({len(lookahead)} расхождений)")
    if metrics.trades < MIN_TRADES:
        reasons.append(f"мало сделок: {metrics.trades} < {MIN_TRADES}")
    if not metrics.profit_factor > MIN_PROFIT_FACTOR:
        reasons.append(f"PF {_f(metrics.profit_factor)} ≤ {MIN_PROFIT_FACTOR:g}")
    if comparison is None or not comparison.profit_factors:
        reasons.append("нет сравнения со случайными входами")
    elif comparison.pf_percentile < MIN_RANDOM_PERCENTILE:
        reasons.append(
            f"PF лучше только {comparison.pf_percentile:.0%} случайных прогонов "
            f"(нужно {MIN_RANDOM_PERCENTILE:.0%})"
        )
    return ("кандидат" if not reasons else "отклонить"), reasons


def check_lookahead(strategy: Strategy, symbol: str, bars: pd.DataFrame) -> list[str]:
    """Проверка обрезкой на последних LOOKAHEAD_BARS свечах реальных данных."""
    window = bars.iloc[-LOOKAHEAD_BARS:].reset_index(drop=True)
    if len(window) < 2:
        return []
    cut = window["open_time"].iloc[int(len(window) * LOOKAHEAD_CUT)]
    return find_lookahead(strategy, symbol, window, cut)


def run_experiment(
    strategy: Strategy,
    symbol: str,
    candles_1m: pd.DataFrame,
    cfg: Config,
    funding: pd.DataFrame | None = None,
    runs: int = 200,
    progress: bool = False,
) -> Experiment:
    costs = CostModel.from_config(cfg.costs)
    common: dict[str, Any] = {
        "instrument": cfg.instruments[symbol],
        "costs": costs,
        "risk": cfg.risk,
        "funding": funding,
    }
    result = run_backtest(
        strategy, symbol, candles_1m, initial_capital=cfg.costs.initial_capital, **common
    )
    metrics = compute_metrics(result, candles_1m["close"])
    lookahead = check_lookahead(strategy, symbol, resample_candles(candles_1m, strategy.timeframe))

    def show(done: int, total: int) -> None:
        if progress and (done % 10 == 0 or done == total):
            print(f"  случайные входы: {done}/{total}", flush=True)

    comparison = compare_with_random(result, candles_1m, runs=runs, progress=show, **common)
    verdict, reasons = preliminary_verdict(metrics, comparison, lookahead)
    return Experiment(result, metrics, comparison, lookahead, verdict, reasons)


def render_baselines(exp: Experiment) -> str:
    m, c = exp.metrics, exp.comparison
    buy_hold = "—" if m.buy_and_hold_pct is None else f"{_f(m.buy_and_hold_pct, 1)}%"
    lines = [
        "",
        "## Сравнение с базовыми линиями",
        "",
        f"- Buy & hold за период: {buy_hold}; стратегия: {_f(m.total_return_pct, 1)}%.",
    ]
    if c is None or not c.profit_factors:
        lines.append("- Случайные входы: сравнение не выполнено (нет сделок).")
    else:
        params = ", ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in c.random_params.items()
        )
        pfs = sorted(c.profit_factors)
        lines += [
            f"- Случайные входы: {c.runs} прогонов с параметрами {params}.",
            f"- PF случайных: медиана {_f(statistics.median(pfs))}, "
            f"5% — {_f(pfs[int(len(pfs) * 0.05)])}, 95% — {_f(pfs[int(len(pfs) * 0.95)])}.",
            f"- PF стратегии {_f(c.strategy_profit_factor)} лучше {c.pf_percentile:.0%} "
            f"случайных прогонов; Expectancy {c.strategy_expectancy_r:.3f}R лучше "
            f"{c.expectancy_percentile:.0%}.",
        ]
    lines += ["", "## Проверка на заглядывание вперёд", ""]
    if exp.lookahead:
        lines += [f"- {p}" for p in exp.lookahead[:20]]
    else:
        lines.append("Расхождений нет: признаки и сигналы до точки обрезки не изменились.")
    lines += ["", f"## Предварительный вердикт: {exp.verdict}", ""]
    lines += [f"- {r}" for r in exp.reasons] or ["Все критерии in-sample выполнены."]
    lines += ["", f"Критерии: {CRITERIA_TEXT}. Окончательное решение — за владельцем.", ""]
    return "\n".join(lines)


def journal_entry(
    exp: Experiment, period: str, report_path: str, variants: int = 1, date: str | None = None
) -> JournalEntry:
    r, m = exp.result, exp.metrics
    hypothesis, rationale = HYPOTHESES.get(r.strategy, ("—", "—"))
    c = exp.comparison
    random_part = "—" if c is None or not c.profit_factors else f"{c.pf_percentile:.0%}"
    buy_hold = "—" if m.buy_and_hold_pct is None else f"{_f(m.buy_and_hold_pct, 1)}%"
    reasons = f" ({'; '.join(exp.reasons)})" if exp.reasons else ""
    return JournalEntry(
        title=f"{r.strategy} {r.strategy_version} {r.symbol} {r.timeframe}",
        date=date or datetime.now(UTC).strftime("%Y-%m-%d"),
        hypothesis=hypothesis,
        rationale=rationale,
        period=period,
        params=", ".join(f"{k}={v}" for k, v in r.params.items()) or "—",
        variants=variants,
        criteria=CRITERIA_TEXT,
        result=(
            f"сделок {m.trades}, PF {_f(m.profit_factor)}, Expectancy {m.expectancy_r:.3f}R, "
            f"доходность {_f(m.total_return_pct, 1)}% (buy & hold {buy_hold}), "
            f"Max DD {_f(m.max_drawdown_pct, 1)}%, лучше случайных: {random_part}"
        ),
        verdict=exp.verdict + reasons,
        report=report_path,
    )


def _load(
    cfg: Config, storage: Storage, symbol: str, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candles = storage.read_candles(symbol, "1m", start, end)
    if candles.empty:
        raise SystemExit(f"Нет 1m-свечей {symbol} за период. Сначала загрузите историю.")
    if symbol not in cfg.instruments:
        raise SystemExit(f"Нет параметров инструмента {symbol} в config/default.yaml")
    return candles, storage.read_funding(symbol, start, end)


def _period(start: pd.Timestamp, end: pd.Timestamp) -> str:
    return f"{start:%Y-%m-%d} — {end - pd.Timedelta(minutes=1):%Y-%m-%d} (in-sample)"


def _shown(path: Path) -> str:
    return (
        path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
    ).as_posix()


def _run_grid(
    cfg: Config,
    storage: Storage,
    name: str,
    symbol: str,
    timeframe: str,
    params: dict[str, Any],
    grid: dict[str, list[Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    journal: Path,
) -> pd.DataFrame:
    candles, funding = _load(cfg, storage, symbol, start, end)
    total = count_variants(grid)
    print(f"Перебор {name} {symbol} {timeframe}: {total} вариантов", flush=True)

    def show(done: int, count: int) -> None:
        print(f"  вариант {done}/{count}", flush=True)

    table = run_grid(name, timeframe, symbol, candles, cfg, grid, params, funding, show)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    folder = cfg.resolve(cfg.data.dir) / "backtests" / f"{stamp}_grid_{name}_{symbol}_{timeframe}"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "grid.csv", index=False)
    report = folder / "grid.md"
    report.write_text(render_grid(table, grid, f"{name} {symbol} {timeframe}"), encoding="utf-8")

    strategy = create_strategy(name, timeframe, params)
    hypothesis, rationale = HYPOTHESES.get(name, ("—", "—"))
    grid_text = "; ".join(f"{k} ∈ {{{', '.join(map(str, v))}}}" for k, v in grid.items())
    entry = JournalEntry(
        title=f"{name} {strategy.version} {symbol} {timeframe} — перебор",
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        hypothesis=hypothesis,
        rationale=rationale,
        period=_period(start, end),
        params=f"перебор {grid_text}; остальные по умолчанию"
        + (f", {', '.join(f'{k}={v}' for k, v in params.items())}" if params else ""),
        variants=total,
        criteria=CRITERIA_TEXT,
        result=grid_summary(table, MIN_PROFIT_FACTOR, MIN_TRADES),
        verdict="выбор варианта из центра «плато» — за владельцем; выбранный вариант "
        f"прогнать без --grid с --param ... --variants {total}",
        report=_shown(report),
    )
    number = append_entry(journal, entry)
    print(f"  EXP-{number:03d}: {entry.result}")
    print(f"  Таблица: {report}")
    return table


def _run_one(
    cfg: Config,
    storage: Storage,
    name: str,
    symbol: str,
    timeframe: str,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    runs: int,
    journal: Path,
    variants: int = 1,
) -> Experiment:
    candles, funding = _load(cfg, storage, symbol, start, end)
    strategy = create_strategy(name, timeframe, params)
    print(f"Эксперимент {name} {symbol} {timeframe}: {len(candles)} минутных свечей", flush=True)
    exp = run_experiment(strategy, symbol, candles, cfg, funding, runs=runs, progress=True)

    costs = CostModel.from_config(cfg.costs)
    ctx = ReportContext(start=start, end=end, costs=costs, risk=cfg.risk)
    folder = save_report(exp.result, exp.metrics, ctx, cfg.resolve(cfg.data.dir) / "backtests")
    report = folder / "report.md"
    with report.open("a", encoding="utf-8") as fh:
        fh.write(render_baselines(exp))
    number = append_entry(
        journal, journal_entry(exp, _period(start, end), _shown(report), variants)
    )
    m = exp.metrics
    print(
        f"  EXP-{number:03d}: сделок {m.trades}, PF {_f(m.profit_factor)}, "
        f"Expectancy {m.expectancy_r:.3f}R → {exp.verdict}"
    )
    print(f"  Отчёт: {report}")
    return exp


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Эксперимент на in-sample")
    parser.add_argument("--strategy", choices=ALL_STRATEGIES)
    parser.add_argument("--symbol")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--all", action="store_true", help="3 стратегии × 3 ТФ × все символы")
    parser.add_argument("--from", dest="start", default=IN_SAMPLE_START)
    parser.add_argument("--to", dest="end", default=IN_SAMPLE_END, help="включительно")
    parser.add_argument("--param", action="append", help="параметр стратегии: ключ=значение")
    parser.add_argument("--runs", type=int, default=200, help="прогонов случайных входов")
    parser.add_argument("--journal", default="research/experiments.md")
    parser.add_argument(
        "--grid", action="append", help="перебор параметра: ключ=v1,v2,... (до 3 параметров)"
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="сколько вариантов было проверено до этого запуска (для журнала)",
    )
    args = parser.parse_args(argv)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)  # конец дня включительно
    if end > pd.Timestamp(IN_SAMPLE_END, tz="UTC") + pd.Timedelta(days=1):
        print("Внимание: период выходит за in-sample — validation/hold-out «сгорают».")
    storage = Storage(cfg.resolve(cfg.data.dir))
    journal = cfg.resolve(Path(args.journal))
    params = _parse_params(args.param)
    try:
        grid = parse_grid(args.grid)
    except ValueError as exc:
        parser.error(str(exc))
    if args.all and grid:
        parser.error("--grid запускается для одной стратегии, без --all")

    if args.all and params:
        parser.error("--param нельзя вместе с --all: у стратегий разные параметры")
    if args.all:
        jobs = [
            (name, symbol, tf)
            for symbol in cfg.data.symbols
            for name in ALL_STRATEGIES
            for tf in ALL_TIMEFRAMES
        ]
    elif args.strategy and args.symbol:
        jobs = [(args.strategy, args.symbol, args.timeframe)]
    else:
        parser.error("нужны --strategy и --symbol, или --all")
    for name, symbol, tf in jobs:
        if grid:
            _run_grid(cfg, storage, name, symbol, tf, params, grid, start, end, journal)
        else:
            _run_one(
                cfg,
                storage,
                name,
                symbol,
                tf,
                params,
                start,
                end,
                args.runs,
                journal,
                args.variants,
            )
    print(f"Журнал: {journal}")


if __name__ == "__main__":
    main()
