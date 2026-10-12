"""Конвейер исследований (этап 9): стратегии × символы × таймфреймы × сетка параметров,
каждая комбинация — через walk-forward, итог — с поправкой на число попыток (DSR).

    uv run python -m bot.research.pipeline                 # всё из config/research.yaml
    uv run python -m bot.research.pipeline --dry-run       # только посчитать попытки

Hold-out конвейер не трогает. Прошедший кандидат проверяется вручную один раз:
    uv run python -m bot.research.validate --stage holdout ...
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field

from bot.backtest.engine import BacktestTrade
from bot.backtest.report import _f
from bot.backtest.walk_forward import make_windows, run_walk_forward
from bot.core.config import PROJECT_ROOT, Config, load_config
from bot.data.storage import Storage
from bot.research.dsr import bonferroni, deflated_sharpe, mean_p_value, sharpe
from bot.research.grid import count_variants
from bot.research.journal import JournalEntry, append_entry
from bot.research.periods import WALK_FORWARD
from bot.research.robustness import TradeStats, trade_stats
from bot.research.validate import OOS_MIN_PF, OOS_MIN_TRADES

DEFAULT_RESEARCH_PATH = PROJECT_ROOT / "config" / "research.yaml"
TRIALS_PATH = Path("research/trials.json")


class StrategySpace(BaseModel):
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class ResearchConfig(BaseModel):
    symbols: list[str]
    timeframes: list[str] = Field(default_factory=lambda: ["1h"])
    min_dsr: float = Field(default=0.95, ge=0.5, lt=1.0)
    strategies: dict[str, StrategySpace]

    def combos(self) -> list[tuple[str, str, str]]:
        return [
            (name, symbol, tf)
            for name in self.strategies
            for symbol in self.symbols
            for tf in self.timeframes
        ]

    def trials(self) -> int:
        """Число попыток в одном запуске: варианты сетки × символы × таймфреймы."""
        return sum(
            max(count_variants(self.strategies[name].grid), 1) for name, _, _ in self.combos()
        )


def load_research(path: Path = DEFAULT_RESEARCH_PATH) -> ResearchConfig:
    return ResearchConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class Candidate:
    strategy: str
    symbol: str
    timeframe: str
    variants: int
    r_multiples: list[float]
    stats: TradeStats
    sharpe: float = 0.0
    dsr: float = 0.0
    p_bonferroni: float = 1.0
    passed: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.strategy} {self.symbol} {self.timeframe}"


def sr_variance(candidates: list[Candidate]) -> float:
    """Разброс Sharpe между попытками. Если кандидатов мало (< 3) — теоретическая
    дисперсия оценки Sharpe при нулевом настоящем: 1/n (n — число сделок)."""
    srs = [c.sharpe for c in candidates if len(c.r_multiples) >= 2]
    if len(srs) >= 3:
        return float(np.var(srs, ddof=1))
    counts = [len(c.r_multiples) for c in candidates if c.r_multiples]
    return float(np.mean([1 / n for n in counts])) if counts else 0.0


def score(candidates: list[Candidate], total_trials: int, min_dsr: float) -> None:
    """DSR и вердикт для каждого кандидата с учётом всех попыток (накопительно)."""
    for c in candidates:
        c.sharpe = sharpe(c.r_multiples)
    variance = sr_variance(candidates)
    for c in candidates:
        c.dsr = deflated_sharpe(c.r_multiples, total_trials, variance)
        c.p_bonferroni = bonferroni(mean_p_value(c.r_multiples), total_trials)
        c.reasons = []
        if c.stats.trades < OOS_MIN_TRADES:
            c.reasons.append(f"сделок {c.stats.trades} < {OOS_MIN_TRADES}")
        if c.stats.profit_factor <= OOS_MIN_PF:
            c.reasons.append(f"PF {_f(c.stats.profit_factor)} ≤ {OOS_MIN_PF:g}")
        if c.stats.expectancy_r <= 0:
            c.reasons.append("Expectancy ≤ 0")
        if c.dsr < min_dsr:
            c.reasons.append(f"DSR {c.dsr:.2f} < {min_dsr:g}")
        c.passed = not c.reasons


def read_trials(path: Path) -> int:
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("total", 0))


def add_trials(path: Path, count: int, note: str) -> int:
    """Накопительный счётчик попыток: каждый запуск конвейера только увеличивает его."""
    data: dict[str, Any] = {"total": 0, "runs": []}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data["total"] = int(data.get("total", 0)) + count
    data.setdefault("runs", []).append(
        {"date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"), "trials": count, "note": note}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return int(data["total"])


WalkForwardFn = Callable[[str, str, str, dict[str, list[Any]], dict[str, Any]], list[BacktestTrade]]


def storage_walk_forward(cfg: Config, storage: Storage) -> WalkForwardFn:
    """Walk-forward на сохранённых свечах (как validate --stage walk-forward)."""
    windows = make_windows(WALK_FORWARD.start, WALK_FORWARD.end)
    cache: dict[str, tuple[Any, Any]] = {}

    def run(
        name: str, symbol: str, tf: str, grid: dict[str, list[Any]], params: dict[str, Any]
    ) -> list[BacktestTrade]:
        if symbol not in cache:
            cache.clear()  # держим в памяти свечи одного символа
            cache[symbol] = (
                storage.read_candles(symbol, "1m", WALK_FORWARD.start, WALK_FORWARD.end),
                storage.read_funding(symbol, WALK_FORWARD.start, WALK_FORWARD.end),
            )
        candles, funding = cache[symbol]
        if candles.empty:
            raise SystemExit(f"Нет 1m-свечей {symbol}. Сначала загрузите историю.")
        wf = run_walk_forward(
            name, tf, symbol, candles, cfg, windows, grid or None, params, funding
        )
        return wf.trades

    return run


def run_pipeline(
    research: ResearchConfig,
    walk_forward: WalkForwardFn,
    capital: float,
    prior_trials: int,
    progress: Callable[[str], None] = print,
) -> tuple[list[Candidate], int]:
    candidates = []
    combos = research.combos()
    for number, (name, symbol, tf) in enumerate(combos, start=1):
        space = research.strategies[name]
        progress(f"[{number}/{len(combos)}] {name} {symbol} {tf}…")
        trades = walk_forward(name, symbol, tf, space.grid, space.params)
        candidates.append(
            Candidate(
                strategy=name,
                symbol=symbol,
                timeframe=tf,
                variants=max(count_variants(space.grid), 1),
                r_multiples=[t.r_multiple for t in sorted(trades, key=lambda t: t.entry_time)],
                stats=trade_stats(trades, capital),
            )
        )
    total = prior_trials + research.trials()
    score(candidates, total, research.min_dsr)
    return candidates, total


def render(candidates: list[Candidate], total_trials: int, run_trials: int, min_dsr: float) -> str:
    lines = [
        "# Конвейер исследований",
        "",
        f"Walk-forward: тестовые окна внутри {WALK_FORWARD.label()}. Hold-out не использовался.",
        f"Попыток в этом запуске: {run_trials}; всего за все запуски: {total_trials} "
        "(учтены в DSR).",
        "",
        "| Кандидат | Вариантов | Сделок OOS | PF | Expectancy, R | Sharpe/сделку | DSR | "
        "p (Бонферрони) | Итог |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in sorted(candidates, key=lambda c: c.dsr, reverse=True):
        verdict = "**кандидат**" if c.passed else "отклонён: " + "; ".join(c.reasons)
        lines.append(
            f"| {c.title} | {c.variants} | {c.stats.trades} | {_f(c.stats.profit_factor)} | "
            f"{c.stats.expectancy_r:.3f} | {c.sharpe:.3f} | {c.dsr:.2f} | "
            f"{c.p_bonferroni:.3f} | {verdict} |"
        )
    passed = [c for c in candidates if c.passed]
    lines += [
        "",
        f"DSR — вероятность, что настоящий Sharpe > 0 с учётом числа попыток. Порог {min_dsr:g}.",
        "",
    ]
    if passed:
        lines.append("Следующий шаг для кандидата — один раз hold-out:")
        for c in passed:
            lines.append(
                f"`uv run python -m bot.research.validate --strategy {c.strategy} "
                f"--symbol {c.symbol} --timeframe {c.timeframe} --stage holdout` "
                "(параметры — из walk-forward отчёта validate --stage walk-forward)"
            )
    else:
        lines.append(
            "Ни один кандидат не прошёл. Это нормальный результат: новые варианты добавляйте "
            "только с новой гипотезой — каждая попытка поднимает планку DSR."
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Конвейер исследований с поправкой на перебор")
    parser.add_argument("--config", type=Path, default=DEFAULT_RESEARCH_PATH)
    parser.add_argument("--journal", default="research/experiments.md")
    parser.add_argument("--dry-run", action="store_true", help="только посчитать попытки")
    args = parser.parse_args(argv)
    cfg = load_config()
    research = load_research(args.config)
    trials_path = cfg.resolve(TRIALS_PATH)
    prior = read_trials(trials_path)
    print(f"Комбинаций: {len(research.combos())}, попыток: {research.trials()}, уже было: {prior}")
    if args.dry_run:
        return
    for symbol in research.symbols:
        if symbol not in cfg.instruments:
            raise SystemExit(f"Нет параметров инструмента {symbol} в config/default.yaml")
    storage = Storage(cfg.resolve(cfg.data.dir))
    candidates, total = run_pipeline(
        research, storage_walk_forward(cfg, storage), cfg.costs.initial_capital, prior
    )
    add_trials(trials_path, research.trials(), f"{len(candidates)} комбинаций")
    text = render(candidates, total, research.trials(), research.min_dsr)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    folder = cfg.resolve(cfg.data.dir) / "backtests" / f"{stamp}_pipeline_ALL_mixed"
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / "validation.md"
    report.write_text(text, encoding="utf-8")
    (folder / "trades.csv").write_text("\n", encoding="utf-8")
    shown = report.relative_to(PROJECT_ROOT) if report.is_relative_to(PROJECT_ROOT) else report
    passed = [c.title for c in candidates if c.passed]
    number = append_entry(
        cfg.resolve(Path(args.journal)),
        JournalEntry(
            title=f"Конвейер: {len(candidates)} комбинаций",
            date=datetime.now(UTC).strftime("%Y-%m-%d"),
            hypothesis="хотя бы одна стратегия из config/research.yaml устойчиво прибыльна",
            rationale="систематический перебор с поправкой на число попыток (этап 9)",
            period=f"walk-forward, {WALK_FORWARD.label()}",
            params="; ".join(f"{k}: {v.grid}" for k, v in research.strategies.items()),
            variants=research.trials(),
            criteria=f"PF > {OOS_MIN_PF:g}, Expectancy > 0, ≥ {OOS_MIN_TRADES} сделок, "
            f"DSR ≥ {research.min_dsr:g} при {total} попытках",
            result=", ".join(f"{c.title}: DSR {c.dsr:.2f}" for c in candidates),
            verdict=("кандидаты: " + ", ".join(passed)) if passed else "кандидатов нет",
            report=shown.as_posix(),
        ),
    )
    print(text)
    print(f"EXP-{number:03d}. Отчёт: {report}")


if __name__ == "__main__":
    main()
