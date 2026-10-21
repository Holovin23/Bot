import pandas as pd

from bot.backtest.lookahead import find_lookahead
from bot.core.models import Position, Side
from bot.strategies.base import MarketContext, generate_signals
from bot.strategies.registry import create_strategy
from bot.strategies.time_series_momentum import TimeSeriesMomentum

from .strategy_helpers import random_bars

SMALL = {"entry_length": 3, "exit_length": 2, "atr_length": 2}


def daily(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    bars = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    bars.insert(0, "open_time", pd.date_range("2024-01-01", periods=len(rows), freq="1D", tz="UTC"))
    bars["volume"] = 1.0
    return bars


# Пять дневных свечей (open, high, low, close). Канал входа — 3 свечи до текущей.
ROWS = [
    (100, 102, 98, 101),
    (101, 103, 99, 102),
    (102, 104, 100, 103),
    (103, 104, 101, 103.5),
    (103.5, 108, 103, 107),  # close 107 > максимума high трёх прошлых свечей (104)
]


def ctx_at(strategy: TimeSeriesMomentum, bars: pd.DataFrame, index: int) -> MarketContext:
    features = strategy.prepare(bars)
    return MarketContext("BTCUSDT", "1d", bars, features, index)


def test_long_on_new_high_by_hand() -> None:
    strategy = TimeSeriesMomentum("1d", SMALL)
    bars = daily(ROWS)
    # На 4-й свече (индекс 3) close 103.5 < максимума 104 — сигнала нет
    assert strategy.on_bar(ctx_at(strategy, bars, 3)) == []
    (signal,) = strategy.on_bar(ctx_at(strategy, bars, 4))
    assert signal.side is Side.LONG
    assert signal.entry_price == 107
    assert signal.signal_id.startswith("tsm-BTCUSDT-1d-")
    # TR по свечам: 4, 4, 4, 3 (104−101), 5 (108−103). ATR(2) по Уайлдеру: затравка —
    # среднее первых двух 4; дальше ½·TR + ½·прошлое: 4, ½·3 + ½·4 = 3.5, ½·5 + ½·3.5 = 4.25.
    # Стоп 107 − 2 × 4.25 = 98.5
    assert signal.stop_loss == 98.5


def test_short_and_trailing_exit_channel() -> None:
    strategy = TimeSeriesMomentum("1d", SMALL)
    mirrored = [(200 - o, 200 - low, 200 - h, 200 - c) for o, h, low, c in ROWS]
    (signal,) = strategy.on_bar(ctx_at(strategy, daily(mirrored), 4))
    assert signal.side is Side.SHORT
    assert (
        TimeSeriesMomentum("1d", {**SMALL, "allow_short": False}).on_bar(
            ctx_at(strategy, daily(mirrored), 4)
        )
        == []
    )
    # Трейлинг лонга — минимум low двух последних свечей (101 и 103) = 101;
    # шорта — максимум high двух последних свечей (зеркально: 200 − 101 = 99)
    position = Position(symbol="BTCUSDT", side=Side.LONG, size=1, entry_price=107, stop_loss=95)
    assert strategy.update_stop(ctx_at(strategy, daily(ROWS), 4), position) == 101
    short = position.model_copy(update={"side": Side.SHORT, "entry_price": 93, "stop_loss": 105})
    assert strategy.update_stop(ctx_at(strategy, daily(mirrored), 4), short) == 99


def test_registered_and_no_lookahead() -> None:
    strategy = create_strategy("time_series_momentum", "1d")
    assert strategy.required_history() == 56
    bars = random_bars(600, timeframe_minutes=1440)
    assert generate_signals(strategy, "BTCUSDT", bars)
    assert find_lookahead(strategy, "BTCUSDT", bars, bars["open_time"].iloc[400]) == []
