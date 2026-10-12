import pytest

from bot.backtest.lookahead import find_lookahead
from bot.core.models import Side
from bot.strategies.base import generate_signals
from bot.strategies.random_entry import RandomEntry

from .strategy_helpers import random_bars


def test_reproducible_with_same_seed() -> None:
    bars = random_bars(2000)
    first = generate_signals(RandomEntry("1h", {"seed": 5}), "BTCUSDT", bars)
    second = generate_signals(RandomEntry("1h", {"seed": 5}), "BTCUSDT", bars)
    other = generate_signals(RandomEntry("1h", {"seed": 6}), "BTCUSDT", bars)
    assert [s.signal_id for s in first] == [s.signal_id for s in second]
    assert [s.signal_id for s in first] != [s.signal_id for s in other]


def test_frequency_and_both_sides() -> None:
    bars = random_bars(4000)
    signals = generate_signals(RandomEntry("1h", {"probability": 0.05}), "BTCUSDT", bars)
    # ожидаем около 0.05 * 3985 ≈ 199
    assert 140 < len(signals) < 260
    sides = {s.side for s in signals}
    assert sides == {Side.LONG, Side.SHORT}


def test_stop_and_take_from_atr() -> None:
    bars = random_bars(500)
    signal = generate_signals(RandomEntry("1h", {"probability": 0.2}), "BTCUSDT", bars)[0]
    value = signal.features["atr"]
    sign = 1 if signal.side is Side.LONG else -1
    assert signal.stop_loss == pytest.approx(signal.entry_price - sign * 1.5 * value)
    assert signal.take_profit == pytest.approx(signal.entry_price + sign * 2.0 * value)
    assert signal.time_stop_bars == 24


def test_no_lookahead() -> None:
    bars = random_bars(600)
    strategy = RandomEntry("1h", {"probability": 0.2})
    for i in (100, 333, 599):
        assert find_lookahead(strategy, "BTCUSDT", bars, bars["open_time"].iloc[i]) == []
