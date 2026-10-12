"""Таймфреймы: '1m', '5m', '15m', '1h', '4h', '1d'."""

from __future__ import annotations

from datetime import timedelta

_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440}

# Код интервала в Bybit v5 (/v5/market/kline, WS kline.{interval}.{symbol}).
_BYBIT_INTERVAL = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1d": "D",
}


def timeframe_minutes(timeframe: str) -> int:
    """'15m' -> 15, '1h' -> 60, '4h' -> 240."""
    if len(timeframe) < 2 or timeframe[-1] not in _UNIT_MINUTES or not timeframe[:-1].isdigit():
        raise ValueError(f"Неизвестный таймфрейм: {timeframe!r}")
    value = int(timeframe[:-1])
    if value <= 0:
        raise ValueError(f"Неизвестный таймфрейм: {timeframe!r}")
    return value * _UNIT_MINUTES[timeframe[-1]]


def timeframe_delta(timeframe: str) -> timedelta:
    return timedelta(minutes=timeframe_minutes(timeframe))


def bybit_interval(timeframe: str) -> str:
    try:
        return _BYBIT_INTERVAL[timeframe]
    except KeyError:
        raise ValueError(f"Bybit не поддерживает таймфрейм {timeframe!r}") from None
