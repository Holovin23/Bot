import pandas as pd
import pytest

from bot.backtest.costs import CostModel
from bot.backtest.engine import run_backtest
from bot.core.config import InstrumentConfig, RiskConfig
from bot.research.baseline import compare_with_random, matched_random_params
from bot.strategies.random_entry import RandomEntry

from .strategy_helpers import minute_walk as walk

INSTRUMENT = InstrumentConfig(tick_size=0.1, qty_step=0.001, min_qty=0.001)
COSTS = CostModel(slippage={"BTCUSDT": 0.0003})


def backtest(strategy: RandomEntry, candles: pd.DataFrame):  # type: ignore[no-untyped-def]
    return run_backtest(
        strategy, "BTCUSDT", candles, instrument=INSTRUMENT, costs=COSTS, risk=RiskConfig()
    )


def test_matched_params_recover_random_strategy() -> None:
    # «Стратегия» — сама случайная: стоп 1.5 ATR, тейк 2 ATR, 24 свечи, вероятность 0.05.
    candles = walk()
    result = backtest(RandomEntry("1h", {"seed": 99}), candles)
    params = matched_random_params(result, candles)
    assert params is not None
    # Стоп и тейк считаются от open свечи входа, а не от close сигнальной — небольшое отличие
    assert params["stop_atr"] == pytest.approx(1.5, abs=0.15)
    assert params["tp_atr"] == pytest.approx(2.0, abs=0.15)
    assert 0.02 < params["probability"] < 0.09
    assert params["time_stop_bars"] == 24


def test_random_strategy_lands_in_the_middle() -> None:
    candles = walk()
    result = backtest(RandomEntry("1h", {"seed": 99}), candles)
    comparison = compare_with_random(
        result, candles, instrument=INSTRUMENT, costs=COSTS, risk=RiskConfig(), runs=20
    )
    assert comparison is not None
    assert len(comparison.profit_factors) == 20
    assert comparison.random_params["time_stop_bars"] == 24
    # Перцентиль — доля случайных PF ниже PF стратегии
    below = sum(pf < comparison.strategy_profit_factor for pf in comparison.profit_factors)
    assert comparison.pf_percentile == pytest.approx(below / 20)


def test_share_below_hand_example() -> None:
    from bot.research.baseline import BaselineComparison

    c = BaselineComparison(
        random_params={},
        runs=4,
        profit_factors=[0.8, 1.0, 1.2, 1.5],
        expectancies_r=[-0.1, 0.0, 0.1, 0.2],
        strategy_profit_factor=1.1,
        strategy_expectancy_r=0.25,
    )
    assert c.pf_percentile == 0.5  # 0.8 и 1.0 ниже 1.1
    assert c.expectancy_percentile == 1.0


def test_no_trades_no_comparison() -> None:
    candles = walk(days=3)
    result = backtest(RandomEntry("1h", {"probability": 0.0}), candles)
    assert matched_random_params(result, candles) is None
