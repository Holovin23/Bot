"""Устойчивость результата (docs/04-backtesting-guide.md, раздел 7).

- trade_stats — метрики по одному списку сделок (для склейки окон walk-forward);
- plateau_scores — «плато» по сетке: средний PF варианта и его соседей;
- monte_carlo_drawdown — распределение Max Drawdown при перемешивании порядка сделок.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from bot.backtest.engine import BacktestTrade

PF_CAP = 10.0  # бесконечный PF (нет убытков) при усреднении считается как 10


@dataclass(frozen=True)
class TradeStats:
    trades: int
    profit_factor: float
    expectancy_r: float
    net_pnl: float
    max_drawdown_pct: float  # по капиталу после каждой сделки


def max_drawdown_of_pnls(pnls: Sequence[float], capital: float) -> float:
    """Максимальная просадка (в % от пика) по последовательности результатов сделок."""
    equity = capital + np.cumsum(np.asarray(pnls, dtype="float64"))
    equity = np.concatenate([[capital], equity])
    peaks = np.maximum.accumulate(equity)
    return float(((peaks - equity) / peaks).max() * 100) if len(equity) else 0.0


def trade_stats(trades: Sequence[BacktestTrade], capital: float) -> TradeStats:
    ordered = sorted(trades, key=lambda t: t.entry_time)
    pnls = [t.net_pnl for t in ordered]
    wins = sum(p for p in pnls if p > 0)
    losses = -sum(p for p in pnls if p <= 0)
    n = len(pnls)
    return TradeStats(
        trades=n,
        profit_factor=wins / losses if losses > 0 else math.inf,
        expectancy_r=sum(t.r_multiple for t in ordered) / n if n else 0.0,
        net_pnl=sum(pnls),
        max_drawdown_pct=max_drawdown_of_pnls(pnls, capital),
    )


def plateau_scores(
    table: pd.DataFrame,
    keys: list[str],
    value: str = "profit_factor",
    min_trades: int = 0,
) -> pd.Series:
    """Для каждой строки — среднее value по ней и соседям (±1 шаг по каждому параметру,
    все комбинации). Варианты с trades < min_trades получают NaN и в среднее соседей
    не входят — плато из «пустых» вариантов не бывает."""
    levels = {k: sorted(table[k].unique().tolist()) for k in keys}
    position = {k: {v: i for i, v in enumerate(levels[k])} for k in keys}
    values = table[value].astype("float64").clip(upper=PF_CAP)
    if "trades" in table:
        values = values.where(table["trades"] >= min_trades)
    by_key: dict[tuple[Any, ...], float] = {
        tuple(row[k] for k in keys): float(v)
        for (_, row), v in zip(table.iterrows(), values, strict=True)
    }
    scores = []
    for (_, row), own in zip(table.iterrows(), values, strict=True):
        if math.isnan(own):
            scores.append(math.nan)
            continue
        neighbours = []
        for offsets in np.ndindex(*([3] * len(keys))):
            point = []
            for k, off in zip(keys, offsets, strict=True):
                i = position[k][row[k]] + off - 1
                if not 0 <= i < len(levels[k]):
                    break
                point.append(levels[k][i])
            else:
                v = by_key.get(tuple(point), math.nan)
                if not math.isnan(v):
                    neighbours.append(v)
        scores.append(sum(neighbours) / len(neighbours))
    return pd.Series(scores, index=table.index, name="plateau")


@dataclass(frozen=True)
class MonteCarlo:
    runs: int
    original_dd_pct: float
    median_dd_pct: float
    p95_dd_pct: float
    worst_dd_pct: float


def monte_carlo_drawdown(
    pnls: Sequence[float], capital: float, runs: int = 1000, seed: int = 0
) -> MonteCarlo:
    rng = np.random.default_rng(seed)
    values = np.asarray(pnls, dtype="float64")
    dds = np.array(
        [max_drawdown_of_pnls(rng.permutation(values).tolist(), capital) for _ in range(runs)]
    )
    if not len(dds):
        dds = np.zeros(1)
    return MonteCarlo(
        runs=runs,
        original_dd_pct=max_drawdown_of_pnls(values.tolist(), capital),
        median_dd_pct=float(np.median(dds)),
        p95_dd_pct=float(np.percentile(dds, 95)),
        worst_dd_pct=float(dds.max()),
    )
