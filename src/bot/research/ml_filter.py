"""Обучение ML-фильтра на in-sample и проверка пользы на validation (этап 6, шаги 2–5).

    uv run python -m bot.research.ml_filter --strategy mean_reversion --symbol BTCUSDT \\
        --timeframe 1h

Модель сохраняется в data/models/<стратегия>_<символ>_<таймфрейм>.joblib, отчёт —
в data/backtests/<время>_mlfilter-<стратегия>_<символ>_<таймфрейм>/validation.md.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bot.analyzer.calibration import reliability_table
from bot.analyzer.dataset import build_dataset
from bot.analyzer.filtered import FilteredStrategy
from bot.analyzer.ml_filter import FilterModel, ModelKind, train_filter
from bot.backtest.costs import CostModel
from bot.backtest.engine import run_backtest
from bot.backtest.report import _f, _parse_params
from bot.backtest.walk_forward import strategy_window
from bot.core.config import PROJECT_ROOT, Config, load_config
from bot.data.storage import Storage
from bot.research.journal import JournalEntry, append_entry
from bot.research.periods import IN_SAMPLE, VALIDATION
from bot.research.robustness import TradeStats, trade_stats
from bot.strategies.registry import create_strategy

MIN_TRADES_SHARE = 0.5  # «сравнимое число сделок» (01): фильтр оставляет ≥ 50%


MIN_RANDOM_PERCENTILE = 0.95


def random_removal_percentile(
    base_r: list[float], filtered_r: list[float], runs: int = 1000, seed: int = 0
) -> float:
    """Доля случайных подмножеств сделок базовой стратегии (того же размера, что после
    фильтра), у которых средний R ниже, чем у фильтра. Отвечает на вопрос: лучше ли
    фильтр, чем просто выбросить столько же сделок наугад."""
    if not base_r or not filtered_r or len(filtered_r) > len(base_r):
        return 0.0
    rng = np.random.default_rng(seed)
    values = np.asarray(base_r, dtype="float64")
    target = float(np.mean(filtered_r))
    means = np.array(
        [rng.choice(values, size=len(filtered_r), replace=False).mean() for _ in range(runs)]
    )
    return float((means < target).mean())


def filter_is_useful(
    base: TradeStats, filtered: TradeStats, random_percentile: float = 1.0
) -> tuple[bool, str]:
    """Критерий из 01: Expectancy или Max DD лучше при сравнимом числе сделок.
    Дополнительно: фильтр должен быть лучше 95% случайных отборов того же числа сделок."""
    if base.trades == 0:
        return False, "у стратегии нет сделок на validation"
    share = filtered.trades / base.trades
    if share < MIN_TRADES_SHARE:
        return False, f"фильтр оставил {share:.0%} сделок (< {MIN_TRADES_SHARE:.0%})"
    better_exp = filtered.expectancy_r > base.expectancy_r
    better_dd = filtered.max_drawdown_pct < base.max_drawdown_pct
    if random_percentile < MIN_RANDOM_PERCENTILE:
        return False, (
            f"лучше только {random_percentile:.0%} случайных отборов того же числа сделок "
            f"(нужно {MIN_RANDOM_PERCENTILE:.0%}) — улучшение похоже на случайность"
        )
    if better_exp or better_dd:
        parts = []
        if better_exp:
            parts.append(f"Expectancy {base.expectancy_r:.3f} → {filtered.expectancy_r:.3f}R")
        if better_dd:
            parts.append(
                f"Max DD {_f(base.max_drawdown_pct, 1)} → {_f(filtered.max_drawdown_pct, 1)}%"
            )
        return True, "; ".join(parts)
    return False, "ни Expectancy, ни Max DD не стали лучше"


def render_model(model: FilterModel) -> list[str]:
    r = model.report
    lines = [
        f"### {model.kind}",
        "",
        f"- Сделок in-sample: {r['trades']}, доля прибыльных {r['base_rate']:.1%}",
        f"- AUC вне блока (purged k-fold): {r['auc_oof']:.3f} (0.5 — не лучше монетки)",
        f"- Brier: базовый {r['brier_base']:.4f}, модель {r['brier_raw']:.4f}, "
        f"после калибровки {r['brier_calibrated']:.4f} (меньше — лучше)",
        f"- Порог {r['threshold']:.3f}: остаётся {r['kept_share']:.0%} сделок, "
        f"Expectancy {r['expectancy_all']:.3f}R → {r['expectancy_kept']:.3f}R (на OOF)",
        "",
        "| Корзина p | Сделок | Средний прогноз | Доля успехов |",
        "|---|---|---|---|",
    ]
    for b in reliability_table(r["oof_calibrated"], r["labels"]):
        if b.count:
            lines.append(
                f"| {b.low:.1f}–{b.high:.1f} | {b.count} | {b.mean_predicted:.2f} | "
                f"{b.observed:.2f} |"
            )
    return [*lines, ""]


def _stats_row(name: str, s: TradeStats) -> str:
    return (
        f"| {name} | {s.trades} | {_f(s.profit_factor)} | {s.expectancy_r:.3f} | "
        f"{_f(s.net_pnl)} | {_f(s.max_drawdown_pct, 1)}% |"
    )


def main(argv: list[str] | None = None) -> None:
    cfg: Config = load_config()
    parser = argparse.ArgumentParser(
        description="ML-фильтр: обучение на IS, проверка на validation"
    )
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--param", action="append", help="параметр стратегии: ключ=значение")
    parser.add_argument("--model", choices=["logreg", "lgbm", "best"], default="best")
    parser.add_argument("--journal", default="research/experiments.md")
    args = parser.parse_args(argv)
    name, symbol, tf = args.strategy, args.symbol, args.timeframe
    params: dict[str, Any] = _parse_params(args.param)
    if symbol not in cfg.instruments:
        raise SystemExit(f"Нет параметров инструмента {symbol} в config/default.yaml")

    storage = Storage(cfg.resolve(cfg.data.dir))
    candles = storage.read_candles(symbol, "1m", IN_SAMPLE.start, IN_SAMPLE.end)
    if candles.empty:
        raise SystemExit("Нет 1m-свечей за in-sample. Сначала загрузите историю.")
    funding = storage.read_funding(symbol, IN_SAMPLE.start, IN_SAMPLE.end)
    base = create_strategy(name, tf, params)
    print(f"Бэктест {name} {symbol} {tf} на in-sample…", flush=True)
    result = run_backtest(
        base,
        symbol,
        candles,
        instrument=cfg.instruments[symbol],
        costs=CostModel.from_config(cfg.costs),
        risk=cfg.risk,
        initial_capital=cfg.costs.initial_capital,
        funding=funding,
    )
    data = build_dataset(result, candles)
    if len(data) < 50:
        raise SystemExit(f"Сделок in-sample {len(data)} — слишком мало для обучения фильтра.")

    kinds: list[ModelKind] = ["logreg", "lgbm"] if args.model == "best" else [args.model]
    models = []
    for kind in kinds:
        print(f"Обучение {kind}…", flush=True)
        models.append(train_filter(data, kind))
    chosen = min(models, key=lambda m: m.report["brier_calibrated"])
    model_path = cfg.resolve(cfg.data.dir) / "models" / f"{name}_{symbol}_{tf}.joblib"
    chosen.save(model_path)

    print("Сравнение на validation…", flush=True)
    warm = VALIDATION.start - pd.Timedelta(days=31)
    v_candles = storage.read_candles(symbol, "1m", warm, VALIDATION.end)
    if v_candles.empty:
        raise SystemExit("Нет 1m-свечей за validation.")
    v_funding = storage.read_funding(symbol, warm, VALIDATION.end)
    capital = cfg.costs.initial_capital
    window = (v_candles, cfg, VALIDATION.start, VALIDATION.end, v_funding)
    plain_trades = strategy_window(base, symbol, *window)
    plain = trade_stats(plain_trades, capital)
    filtered_strategy = FilteredStrategy(create_strategy(name, tf, params), chosen)
    filtered_trades = strategy_window(filtered_strategy, symbol, *window)
    filtered = trade_stats(filtered_trades, capital)
    percentile = random_removal_percentile(
        [t.r_multiple for t in plain_trades], [t.r_multiple for t in filtered_trades]
    )
    useful, why = filter_is_useful(plain, filtered, percentile)

    lines = [
        f"# ML-фильтр: {name} {symbol} {tf}",
        "",
        f"Обучение: {IN_SAMPLE.label()}. Проверка: {VALIDATION.label()}.",
        "",
        "## Модели (in-sample, вне блока)",
        "",
    ]
    for m in models:
        lines += render_model(m)
    lines += [
        f"Выбрана модель **{chosen.kind}** (меньший Brier после калибровки), "
        f"порог {chosen.threshold:.3f}. Файл: `{model_path.name}`.",
        "",
        "## Validation: стратегия против стратегии с фильтром",
        "",
        "| Вариант | Сделок | PF | Expectancy, R | Результат, USDT | Max DD |",
        "|---|---|---|---|---|---|",
        _stats_row("без фильтра", plain),
        _stats_row("с фильтром", filtered),
        "",
        f"Средний R с фильтром лучше, чем у {percentile:.0%} случайных отборов того же "
        "числа сделок без фильтра.",
        "",
        f"Итог: фильтр {'полезен' if useful else 'не полезен'} — {why}.",
        "Если фильтр не полезен на validation, он остаётся только справочным мнением "
        "и не включается в торговлю (03, этап 6).",
        "",
    ]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    folder = cfg.resolve(cfg.data.dir) / "backtests" / f"{stamp}_mlfilter-{name}_{symbol}_{tf}"
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / "validation.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    (folder / "trades.csv").write_text("\n", encoding="utf-8")

    shown = report.relative_to(PROJECT_ROOT) if report.is_relative_to(PROJECT_ROOT) else report
    number = append_entry(
        cfg.resolve(Path(args.journal)),
        JournalEntry(
            title=f"{name} {symbol} {tf} + ML-фильтр ({chosen.kind})",
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            hypothesis="признаки рынка на момент сигнала отличают прибыльные сделки от убыточных",
            rationale="режим рынка и волатильность влияют на исход сделки",
            period=f"обучение {IN_SAMPLE.label()}, проверка {VALIDATION.label()}",
            params=", ".join(f"{k}={v}" for k, v in base.params.items()),
            variants=len(models),
            criteria="Expectancy или Max DD лучше на validation при ≥ 50% сделок (01)",
            result=(
                f"без фильтра {plain.trades} сделок, {plain.expectancy_r:.3f}R; "
                f"с фильтром {filtered.trades}, {filtered.expectancy_r:.3f}R; "
                f"AUC {chosen.report['auc_oof']:.3f}"
            ),
            verdict=("полезен: " if useful else "не полезен: ") + why,
            report=shown.as_posix(),
        ),
    )
    print(f"Фильтр {'полезен' if useful else 'не полезен'}: {why}")
    print(f"EXP-{number:03d}. Отчёт: {report}")
    if math.isnan(chosen.report["auc_oof"]):
        print("Внимание: AUC не посчитан (в данных одна метка).")


if __name__ == "__main__":
    main()
