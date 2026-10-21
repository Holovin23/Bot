"""Walk-forward (docs/04-backtesting-guide.md, раздел 3).

    Окно 1: [подбор 12 мес][проверка 3 мес]
    Окно 2:      [подбор 12 мес][проверка 3 мес]   (шаг 3 мес)

На окне подбора перебирается сетка, выбирается вариант с лучшей «оценкой плато»,
он проверяется на следующем окне. Результат — склейка всех окон проверки.
Окна проверки не перекрываются, поэтому каждая сделка склейки — вне выборки.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestTrade, run_backtest
from bot.core.config import Config
from bot.core.timeframes import timeframe_minutes
from bot.research.grid import run_grid
from bot.research.robustness import TradeStats, plateau_scores, trade_stats
from bot.strategies.base import Strategy
from bot.strategies.registry import create_strategy

WARMUP = pd.Timedelta(days=30)
MIN_TRAIN_TRADES = 30


@dataclass(frozen=True)
class Window:
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # не включительно; = test_start
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # не включительно


def make_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3,
) -> list[Window]:
    windows = []
    train_start = start
    while True:
        test_start = train_start + pd.DateOffset(months=train_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        windows.append(Window(train_start, test_start, test_start, test_end))
        train_start = train_start + pd.DateOffset(months=step_months)
    return windows


def _slice(df: pd.DataFrame, column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df[column] >= start) & (df[column] < end)].reset_index(drop=True)


def backtest_window(
    strategy_name: str,
    timeframe: str,
    symbol: str,
    candles_1m: pd.DataFrame,
    cfg: Config,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding: pd.DataFrame | None = None,
) -> list[BacktestTrade]:
    """Сделки с сигналом внутри [start, end). Бэктест начинается на WARMUP раньше,
    чтобы индикаторы успели посчитаться; сделка, не закрытая к end, закрывается по
    последней цене окна (end_of_data)."""
    strategy = create_strategy(strategy_name, timeframe, params)
    return strategy_window(strategy, symbol, candles_1m, cfg, start, end, funding)


def warmup_for(strategy: Strategy) -> pd.Timedelta:
    """Запас истории перед окном: не меньше WARMUP и не меньше 1,5 × required_history
    свечей стратегии (EMA200 на 4h — 50 дней, канал 55 дней на 1d — 84 дня)."""
    bars = strategy.required_history() * 1.5
    return max(WARMUP, pd.Timedelta(minutes=timeframe_minutes(strategy.timeframe) * bars))


def strategy_window(
    strategy: Strategy,
    symbol: str,
    candles_1m: pd.DataFrame,
    cfg: Config,
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding: pd.DataFrame | None = None,
) -> list[BacktestTrade]:
    """То же, что backtest_window, для готового объекта стратегии (например, с фильтром)."""
    warmup = warmup_for(strategy)
    candles = _slice(candles_1m, "open_time", start - warmup, end)
    fund = None if funding is None else _slice(funding, "time", start - warmup, end)
    result = run_backtest(
        strategy,
        symbol,
        candles,
        instrument=cfg.instruments[symbol],
        costs=CostModel.from_config(cfg.costs),
        risk=cfg.risk,
        initial_capital=cfg.costs.initial_capital,
        funding=fund,
        limits=cfg.limits,
    )
    return [t for t in result.trades if start <= t.signal_time < end]


def select_variant(
    table: pd.DataFrame, keys: list[str], min_trades: int = MIN_TRAIN_TRADES
) -> dict[str, Any] | None:
    """Вариант с лучшей оценкой плато; None — ни у одного нет min_trades сделок."""
    if table.empty:
        return None
    scores = plateau_scores(table, keys, min_trades=min_trades)
    if scores.isna().all():
        return None
    row = table.iloc[_best(scores)]
    return {k: row[k].item() if hasattr(row[k], "item") else row[k] for k in keys}


def _best(scores: pd.Series) -> int:
    """Позиция максимума без NaN; при равенстве — первая."""
    values = scores.fillna(float("-inf")).to_numpy()
    return int(values.argmax())


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame
    trades: list[BacktestTrade] = field(default_factory=list)
    stats: TradeStats | None = None


def run_walk_forward(
    strategy_name: str,
    timeframe: str,
    symbol: str,
    candles_1m: pd.DataFrame,
    cfg: Config,
    windows: list[Window],
    grid: dict[str, list[Any]] | None = None,
    base_params: dict[str, Any] | None = None,
    funding: pd.DataFrame | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> WalkForwardResult:
    base = dict(base_params or {})
    rows = []
    all_trades: list[BacktestTrade] = []
    for number, w in enumerate(windows, start=1):
        chosen: dict[str, Any] | None = {}
        train_pf = float("nan")
        if grid:
            train = _slice(candles_1m, "open_time", w.train_start, w.train_end)
            fund = None if funding is None else _slice(funding, "time", w.train_start, w.train_end)
            table = run_grid(strategy_name, timeframe, symbol, train, cfg, grid, base, fund)
            chosen = select_variant(table, list(grid))
            if chosen is not None:
                match = (table[list(grid)] == pd.Series(chosen)).all(axis=1)
                train_pf = float(table.loc[match, "profit_factor"].iloc[0])
        params = {**base, **(chosen or {})}
        trades = backtest_window(
            strategy_name,
            timeframe,
            symbol,
            candles_1m,
            cfg,
            params,
            w.test_start,
            w.test_end,
            funding,
        )
        stats = trade_stats(trades, cfg.costs.initial_capital)
        rows.append(
            {
                "train": f"{w.train_start:%Y-%m-%d} — {w.train_end:%Y-%m-%d}",
                "test": f"{w.test_start:%Y-%m-%d} — {w.test_end:%Y-%m-%d}",
                "params": "по умолчанию"
                if not chosen
                else ", ".join(f"{k}={v}" for k, v in chosen.items()),
                "train_pf": train_pf,
                "test_trades": stats.trades,
                "test_pf": stats.profit_factor,
                "test_expectancy_r": stats.expectancy_r,
                "test_pnl": stats.net_pnl,
            }
        )
        all_trades.extend(trades)
        if progress is not None:
            progress(number, len(windows))
    return WalkForwardResult(
        windows=pd.DataFrame(rows),
        trades=all_trades,
        stats=trade_stats(all_trades, cfg.costs.initial_capital),
    )
