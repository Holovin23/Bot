import pandas as pd

from bot.analyzer.dataset import FEATURE_COLUMNS, build_dataset, market_features
from bot.backtest.costs import CostModel
from bot.backtest.engine import run_backtest
from bot.backtest.lookahead import find_lookahead
from bot.core.config import InstrumentConfig, RiskConfig
from bot.strategies.base import BaseStrategy
from bot.strategies.random_entry import RandomEntry

from .strategy_helpers import minute_walk, random_bars


class FeaturesOnly(BaseStrategy):
    name = "features_only"

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return market_features(bars)


def test_market_features_hand_values_and_no_lookahead() -> None:
    bars = random_bars(700)
    f = market_features(bars)
    # ret_1 в % — (close_i / close_{i−1} − 1) × 100
    expected = (bars["close"].iloc[10] / bars["close"].iloc[9] - 1) * 100
    assert f["ret_1"].iloc[10] == expected
    assert f["hour"].iloc[1] == 1.0  # часовые свечи с 00:00
    cut = bars["open_time"].iloc[600]
    assert find_lookahead(FeaturesOnly("1h"), "BTCUSDT", bars, cut) == []


def test_dataset_matches_trades() -> None:
    candles = minute_walk(days=60)
    result = run_backtest(
        RandomEntry("1h", {"seed": 4}),
        "BTCUSDT",
        candles,
        instrument=InstrumentConfig(tick_size=0.1, qty_step=0.001, min_qty=0.001),
        costs=CostModel(slippage={"BTCUSDT": 0.0003}),
        risk=RiskConfig(),
    )
    data = build_dataset(result, candles)
    assert len(data) == len(result.trades)
    assert list(data.columns[: len(FEATURE_COLUMNS)]) == FEATURE_COLUMNS
    first = result.trades[0]
    assert data["label"].iloc[0] == int(first.r_multiple > 0)
    assert data["side"].iloc[0] == (1.0 if first.side.value == "LONG" else -1.0)
    # Признак часа — час открытия сигнальной свечи
    assert data["hour"].iloc[0] == (first.signal_time - pd.Timedelta(hours=1)).hour
