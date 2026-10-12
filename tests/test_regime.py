import numpy as np
import pandas as pd

from bot.backtest.lookahead import find_lookahead
from bot.features.regime import Regime, classify, regime_frame
from bot.strategies.base import BaseStrategy

from .strategy_helpers import random_bars


def test_classify_hand_example() -> None:
    s = lambda *v: pd.Series(v, dtype="float64")  # noqa: E731
    codes = classify(
        atr_rank=s(0.95, 0.5, 0.5, 0.5, 0.5, np.nan),
        adx_value=s(40, 30, 30, 30, 15, 30),
        close=s(100, 105, 95, 95, 100, 100),
        ema_value=s(100, 100, 100, 100, 100, 100),
        ema_slope=s(1, 1, -1, 1, 1, 1),
    )
    # 0: высокая волатильность важнее тренда; 1: рост; 2: падение;
    # 3: close ниже EMA, но EMA растёт — не тренд; 4: ADX 15 — боковик; 5: NaN
    assert codes.tolist() == [
        Regime.HIGH_VOL,
        Regime.TREND_UP,
        Regime.TREND_DOWN,
        Regime.RANGE,
        Regime.RANGE,
        Regime.UNKNOWN,
    ]


class RegimeOnly(BaseStrategy):
    name = "regime_only"

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return regime_frame(bars)


def test_regime_has_no_lookahead() -> None:
    bars = random_bars(900)
    frame = regime_frame(bars)
    assert set(frame["regime"].unique()) >= {Regime.UNKNOWN, Regime.RANGE}
    assert (frame["regime"].iloc[:499] == Regime.UNKNOWN).all()  # 500 свечей на перцентиль
    cut = bars["open_time"].iloc[700]
    assert find_lookahead(RegimeOnly("1h"), "BTCUSDT", bars, cut) == []
