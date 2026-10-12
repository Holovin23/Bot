"""Ограниченный перебор параметров на in-sample (docs/04-backtesting-guide.md, раздел 1).

Перебираются 2–3 параметра, десятки вариантов, не тысячи. Результат — таблица всех
вариантов и «тепловая карта» PF по двум параметрам. Вариант выбирается из центра
«плато» (соседние значения тоже хороши), а не по максимуму; выбор делает владелец.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import pandas as pd
import yaml

from bot.backtest.costs import CostModel
from bot.backtest.engine import run_backtest
from bot.backtest.metrics import compute_metrics
from bot.core.config import Config
from bot.strategies.registry import create_strategy

MAX_KEYS = 3
MAX_VARIANTS = 100
METRIC_COLUMNS = [
    "trades",
    "profit_factor",
    "expectancy_r",
    "total_return_pct",
    "max_drawdown_pct",
    "sharpe",
]


def parse_grid(items: list[str] | None) -> dict[str, list[Any]]:
    """['bb_std=2,2.5', 'adx_max=15,20,25'] -> {'bb_std': [2, 2.5], 'adx_max': [15, 20, 25]}."""
    grid: dict[str, list[Any]] = {}
    for item in items or []:
        key, sep, values = item.partition("=")
        if not sep or not values:
            raise ValueError(f"Перебор задаётся как ключ=v1,v2,...: {item!r}")
        grid[key.strip()] = [yaml.safe_load(v) for v in values.split(",")]
    if len(grid) > MAX_KEYS:
        raise ValueError(f"Не больше {MAX_KEYS} параметров в переборе, задано {len(grid)}")
    total = count_variants(grid)
    if total > MAX_VARIANTS:
        raise ValueError(f"{total} вариантов — слишком много (не больше {MAX_VARIANTS})")
    return grid


def count_variants(grid: dict[str, list[Any]]) -> int:
    return math.prod(len(v) for v in grid.values()) if grid else 0


def grid_variants(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]


def run_grid(
    strategy_name: str,
    timeframe: str,
    symbol: str,
    candles_1m: pd.DataFrame,
    cfg: Config,
    grid: dict[str, list[Any]],
    base_params: dict[str, Any] | None = None,
    funding: pd.DataFrame | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Бэктест каждого варианта. Строка таблицы — параметры варианта и его метрики."""
    costs = CostModel.from_config(cfg.costs)
    variants = grid_variants(grid)
    rows = []
    for number, variant in enumerate(variants, start=1):
        strategy = create_strategy(strategy_name, timeframe, {**(base_params or {}), **variant})
        result = run_backtest(
            strategy,
            symbol,
            candles_1m,
            instrument=cfg.instruments[symbol],
            costs=costs,
            risk=cfg.risk,
            initial_capital=cfg.costs.initial_capital,
            funding=funding,
        )
        m = compute_metrics(result, candles_1m["close"])
        rows.append({**variant, **{c: getattr(m, c) for c in METRIC_COLUMNS}})
        if progress is not None:
            progress(number, len(variants))
    return pd.DataFrame(rows, columns=[*grid, *METRIC_COLUMNS])


def grid_summary(table: pd.DataFrame, min_pf: float, min_trades: int) -> str:
    """Одна строка для журнала: разброс PF и сколько вариантов проходят критерии."""
    if table.empty:
        return "нет вариантов"
    pf = table["profit_factor"]
    passed = int(((pf > min_pf) & (table["trades"] >= min_trades)).sum())
    return (
        f"PF по вариантам: мин {_cell(float(pf.min()))}, медиана {_cell(float(pf.median()))}, "
        f"макс {_cell(float(pf.max()))}; PF > {min_pf:g} и ≥ {min_trades} сделок: "
        f"{passed} из {len(table)}"
    )


def heatmap(
    table: pd.DataFrame, rows: str, columns: str, value: str = "profit_factor"
) -> pd.DataFrame:
    """Таблица value по двум параметрам. Если параметров три — среднее по третьему."""
    return table.pivot_table(index=rows, columns=columns, values=value, aggfunc="mean")


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return "∞" if math.isinf(value) else "—" if math.isnan(value) else f"{value:.2f}"
    return str(value)


def to_markdown(frame: pd.DataFrame, index_name: str | None = None) -> str:
    """Простая Markdown-таблица. index_name — показать индекс первым столбцом."""
    if index_name is not None:
        frame = frame.copy()
        frame.columns = [str(c) for c in frame.columns]
        frame = frame.reset_index(names=index_name)
    header = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(lines)


def render_grid(table: pd.DataFrame, grid: dict[str, list[Any]], title: str) -> str:
    keys = list(grid)
    lines = [
        f"# Перебор параметров: {title}",
        "",
        f"Вариантов: {len(table)}. Параметры: "
        + "; ".join(f"{k} ∈ {{{', '.join(map(str, v))}}}" for k, v in grid.items()),
        "",
        "Выбирайте вариант из центра «плато», где соседние значения тоже прибыльны, "
        "а не максимум таблицы. Одиночный пик среди убыточных соседей — признак подгонки.",
        "",
    ]
    if len(keys) >= 2:
        title_pf = f"## PF: строки {keys[0]}, столбцы {keys[1]}"
        if len(keys) == 3:
            title_pf += f" (среднее по {keys[2]})"
        lines += [title_pf, "", to_markdown(heatmap(table, keys[0], keys[1]), keys[0]), ""]
    lines += ["## Все варианты", "", to_markdown(table), ""]
    return "\n".join(lines)
