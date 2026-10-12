import numpy as np
import pandas as pd
import pytest

from bot.backtest.lookahead import find_lookahead
from bot.core.models import Side
from bot.strategies.base import MarketContext, generate_signals
from bot.strategies.breakout import Breakout

from .strategy_helpers import random_bars

T0 = pd.Timestamp("2024-01-01", tz="UTC")
BASE = {
    "squeeze_rank": 0.05,
    "channel_high": 105.0,
    "channel_low": 95.0,
    "avg_volume": 10.0,
    "stop_low": 96.0,
    "stop_high": 104.0,
}


def ctx_for(close: float, volume: float = 20.0, **features: float) -> MarketContext:
    bars = pd.DataFrame(
        {
            "open_time": [T0],
            "open": [100.0],
            "high": [max(close, 100.0)],
            "low": [min(close, 100.0)],
            "close": [close],
            "volume": [volume],
        }
    )
    return MarketContext("BTCUSDT", "1h", bars, pd.DataFrame([{**BASE, **features}]), 0)


def test_long_breakout_by_hand() -> None:
    # close 106 > 105, объём 20 > 1.5 × 10, сжатие 0.05 <= 0.10
    # стоп 96, риск 10, тейк 106 + 2 × 10 = 126
    (signal,) = Breakout("1h").on_bar(ctx_for(106))
    assert signal.side is Side.LONG
    assert signal.stop_loss == 96
    assert signal.take_profit == 126
    assert signal.time_stop_bars == 48


def test_short_breakout_by_hand() -> None:
    # close 94 < 95, стоп 104, риск 10, тейк 94 − 20 = 74
    (signal,) = Breakout("1h").on_bar(ctx_for(94))
    assert signal.side is Side.SHORT
    assert signal.stop_loss == 104
    assert signal.take_profit == 74


@pytest.mark.parametrize(
    ("close", "volume", "features"),
    [
        (104.0, 20.0, {}),  # внутри канала
        (106.0, 15.0, {}),  # объём ровно 1.5× — нужно строго больше
        (106.0, 20.0, {"squeeze_rank": 0.2}),  # нет сжатия
    ],
)
def test_no_signal(close: float, volume: float, features: dict[str, float]) -> None:
    assert Breakout("1h").on_bar(ctx_for(close, volume, **features)) == []


def squeeze_then_breakout() -> pd.DataFrame:
    """200 спокойных свечей с затуханием колебаний, затем резкий рост с большим объёмом."""
    n = 200
    amplitude = np.linspace(2.0, 0.2, n)
    close = 100 + amplitude * np.sin(np.arange(n))
    close = np.r_[close, 103.0]
    open_ = np.r_[close[0], close[:-1]]
    volume = np.r_[np.full(n, 10.0), 40.0]
    return pd.DataFrame(
        {
            "open_time": pd.date_range(T0, periods=n + 1, freq="1h"),
            "open": open_,
            "high": np.maximum(open_, close) + 0.05,
            "low": np.minimum(open_, close) - 0.05,
            "close": close,
            "volume": volume,
        }
    )


def test_end_to_end_squeeze_breakout() -> None:
    bars = squeeze_then_breakout()
    signals = generate_signals(Breakout("1h"), "BTCUSDT", bars)
    assert [s.side for s in signals] == [Side.LONG]
    signal = signals[0]
    assert signal.entry_price == 103
    assert signal.stop_loss == pytest.approx(bars["low"].iloc[-20:].min())


def test_random_data_no_lookahead() -> None:
    bars = random_bars(2000)
    bars["volume"] = np.random.default_rng(3).lognormal(0, 0.6, len(bars))
    strategy = Breakout("1h", {"squeeze_percentile": 0.3})
    assert generate_signals(strategy, "BTCUSDT", bars)
    assert find_lookahead(strategy, "BTCUSDT", bars, bars["open_time"].iloc[1200]) == []
