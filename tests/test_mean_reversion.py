import pandas as pd
import pytest

from bot.backtest.lookahead import find_lookahead
from bot.core.models import Side
from bot.strategies.base import MarketContext, generate_signals
from bot.strategies.mean_reversion import MeanReversion

from .strategy_helpers import random_bars

T0 = pd.Timestamp("2024-01-01", tz="UTC")
# Полосы 95 / 100 / 105, RSI 25, ADX 15, ATR 2
BASE = {
    "bb_lower": 95.0,
    "bb_middle": 100.0,
    "bb_upper": 105.0,
    "rsi": 25.0,
    "adx": 15.0,
    "atr": 2.0,
}


def ctx_for(close: float, **features: float) -> MarketContext:
    bars = pd.DataFrame(
        {
            "open_time": [T0],
            "open": [close],
            "high": [close + 1],
            "low": [close - 1],
            "close": [close],
            "volume": [1.0],
        }
    )
    return MarketContext("BTCUSDT", "1h", bars, pd.DataFrame([{**BASE, **features}]), 0)


def test_long_below_lower_band() -> None:
    # close 94 < 95, RSI 25 < 30, ADX 15 < 20 -> LONG
    # стоп 94 − 1.5 × 2 = 91, тейк 100, R:R = 6 / 3 = 2
    (signal,) = MeanReversion("1h").on_bar(ctx_for(94))
    assert signal.side is Side.LONG
    assert signal.stop_loss == 91
    assert signal.take_profit == 100
    assert signal.time_stop_bars == 24
    assert signal.features["rr"] == 2


@pytest.mark.parametrize(
    ("close", "features"),
    [
        (96.0, {}),  # выше нижней полосы
        (94.0, {"rsi": 35.0}),  # RSI не перепродан
        (94.0, {"adx": 25.0}),  # тренд, не боковик
        (94.0, {"adx": 20.0}),  # граница: ADX должен быть строго < 20
        (94.0, {"bb_middle": 96.0}),  # R:R = 2 / 3 < 1
    ],
)
def test_no_long(close: float, features: dict[str, float]) -> None:
    assert MeanReversion("1h").on_bar(ctx_for(close, **features)) == []


def test_short_is_mirror() -> None:
    # close 106 > 105, RSI 75 > 70 -> SHORT, стоп 109, тейк 100, R:R 6 / 3 = 2
    (signal,) = MeanReversion("1h").on_bar(ctx_for(106, rsi=75.0))
    assert signal.side is Side.SHORT
    assert signal.stop_loss == 109
    assert signal.take_profit == 100


def test_random_data_rules_and_no_lookahead() -> None:
    bars = random_bars(3000)
    strategy = MeanReversion("1h", {"adx_max": 30.0})
    signals = generate_signals(strategy, "BTCUSDT", bars)
    assert signals
    for s in signals:
        assert s.features["adx"] < 30
        assert s.features["rr"] >= 1.0
    assert find_lookahead(strategy, "BTCUSDT", bars, bars["open_time"].iloc[1700]) == []
