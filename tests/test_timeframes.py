from datetime import timedelta

import pytest

from bot.core.timeframes import bybit_interval, timeframe_delta, timeframe_minutes


def test_timeframe_minutes() -> None:
    assert timeframe_minutes("1m") == 1
    assert timeframe_minutes("15m") == 15
    assert timeframe_minutes("1h") == 60
    assert timeframe_minutes("4h") == 240
    assert timeframe_delta("5m") == timedelta(minutes=5)


@pytest.mark.parametrize("bad", ["", "m", "0m", "5x", "h1", "1.5h"])
def test_bad_timeframe(bad: str) -> None:
    with pytest.raises(ValueError):
        timeframe_minutes(bad)


def test_bybit_interval() -> None:
    assert bybit_interval("1m") == "1"
    assert bybit_interval("1h") == "60"
    assert bybit_interval("1d") == "D"
