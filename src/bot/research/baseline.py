"""Сравнение стратегии со случайными входами той же частоты и с теми же стопом/тейком
(docs/04-backtesting-guide.md, раздел 7).

Параметры случайных входов берутся из сделок стратегии:
- вероятность входа = число сделок / число свечей без позиции;
- стоп и тейк — медианные расстояния от входа в ATR(14) на свече сигнала;
- выход по времени — медианный time_stop_bars из сигналов стратегии; если стратегия
  его не задаёт, — самая длинная сделка (медиана длительности занизила бы удержание:
  многие сделки закрываются стопом раньше).
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestResult, BacktestTrade, run_backtest
from bot.backtest.metrics import compute_metrics
from bot.core.config import InstrumentConfig, RiskConfig
from bot.core.timeframes import timeframe_delta
from bot.data.resample import resample_candles
from bot.features.indicators import atr
from bot.strategies.random_entry import RandomEntry

ATR_LENGTH = 14
MIN_SHARE_WITH_TAKE = 0.5  # если тейк есть меньше чем у половины сделок — без тейка


@dataclass(frozen=True)
class BaselineComparison:
    random_params: dict[str, Any]
    runs: int
    profit_factors: list[float] = field(default_factory=list)
    expectancies_r: list[float] = field(default_factory=list)
    strategy_profit_factor: float = 0.0
    strategy_expectancy_r: float = 0.0

    @property
    def pf_percentile(self) -> float:
        """Доля случайных прогонов, у которых PF ниже, чем у стратегии (0..1)."""
        return _share_below(self.profit_factors, self.strategy_profit_factor)

    @property
    def expectancy_percentile(self) -> float:
        return _share_below(self.expectancies_r, self.strategy_expectancy_r)


def _share_below(values: list[float], value: float) -> float:
    if not values:
        return 0.0
    return sum(v < value for v in values) / len(values)


def matched_random_params(
    result: BacktestResult, candles_1m: pd.DataFrame
) -> dict[str, Any] | None:
    """Параметры RandomEntry, похожие на стратегию. None — у стратегии нет сделок."""
    trades = result.trades
    flat_bars = int((~result.equity["in_position"]).sum()) if len(result.equity) else 0
    if not trades or flat_bars == 0:
        return None

    bars = resample_candles(candles_1m, result.timeframe)
    atr_by_time = pd.Series(atr(bars, ATR_LENGTH).to_numpy(), index=bars["open_time"])
    step = timeframe_delta(result.timeframe)

    stops, takes = [], []
    for t in trades:
        value = atr_by_time.get(t.signal_time - step)
        if value is None or not value > 0:
            continue
        stops.append(abs(t.entry_price_raw - t.initial_stop) / value)
        if t.take is not None:
            takes.append(abs(t.take - t.entry_price_raw) / value)
    if not stops:
        return None
    with_take = len(takes) / len(stops) >= MIN_SHARE_WITH_TAKE
    return {
        "probability": min(len(trades) / flat_bars, 1.0),
        "stop_atr": statistics.median(stops),
        "tp_atr": statistics.median(takes) if with_take and takes else None,
        "time_stop_bars": _time_stop(trades),
        "atr_length": ATR_LENGTH,
    }


def _time_stop(trades: list[BacktestTrade]) -> int:
    planned = [t.time_stop_bars for t in trades if t.time_stop_bars is not None]
    if planned:
        return max(1, round(statistics.median(planned)))
    return max(1, max(t.bars_held for t in trades))


def compare_with_random(
    result: BacktestResult,
    candles_1m: pd.DataFrame,
    *,
    instrument: InstrumentConfig,
    costs: CostModel,
    risk: RiskConfig,
    funding: pd.DataFrame | None = None,
    runs: int = 50,
    progress: Callable[[int, int], None] | None = None,
) -> BaselineComparison | None:
    params = matched_random_params(result, candles_1m)
    if params is None:
        return None
    strategy_metrics = compute_metrics(result)
    pfs, exps = [], []
    for seed in range(1, runs + 1):
        random_result = run_backtest(
            RandomEntry(result.timeframe, {**params, "seed": seed}),
            result.symbol,
            candles_1m,
            instrument=instrument,
            costs=costs,
            risk=risk,
            initial_capital=result.initial_capital,
            funding=funding,
        )
        m = compute_metrics(random_result)
        if m.trades:
            pfs.append(min(m.profit_factor, 100.0))
            exps.append(m.expectancy_r)
        if progress is not None:
            progress(seed, runs)
    return BaselineComparison(
        random_params=params,
        runs=runs,
        profit_factors=pfs,
        expectancies_r=exps,
        strategy_profit_factor=min(strategy_metrics.profit_factor, 100.0),
        strategy_expectancy_r=strategy_metrics.expectancy_r,
    )
