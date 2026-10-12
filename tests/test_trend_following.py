from datetime import UTC, datetime

import pandas as pd
import pytest

from bot.backtest.lookahead import find_lookahead
from bot.core.models import Position, Side
from bot.strategies.base import MarketContext, generate_signals
from bot.strategies.trend_following import TrendFollowing

from .strategy_helpers import random_bars

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def ctx_for(rows: list[dict[str, float]], features: list[dict[str, float]]) -> MarketContext:
    """Контекст на последней строке с заданными вручную свечами и индикаторами."""
    bars = pd.DataFrame(rows)
    bars.insert(0, "open_time", pd.date_range(T0, periods=len(rows), freq="1h"))
    return MarketContext("BTCUSDT", "1h", bars, pd.DataFrame(features), len(rows) - 1)


UP = {"ema_fast": 100.0, "ema_mid": 95.0, "ema_slow": 90.0, "atr": 2.0, "trail_atr": 3.0}
DOWN = {"ema_fast": 100.0, "ema_mid": 105.0, "ema_slow": 110.0, "atr": 2.0, "trail_atr": 3.0}


def bar(o: float, h: float, low: float, c: float) -> dict[str, float]:
    return {"open": o, "high": h, "low": low, "close": c, "volume": 1.0}


def test_long_on_pullback_to_ema20() -> None:
    # Тренд вверх, low 99.5 <= EMA20 100, close 101 > 100 -> LONG, стоп 101 − 2 × 2 = 97
    (signal,) = TrendFollowing("1h").on_bar(ctx_for([bar(100, 102, 99.5, 101)], [UP]))
    assert signal.side is Side.LONG
    assert signal.entry_price == 101
    assert signal.stop_loss == 97
    assert signal.take_profit is None


@pytest.mark.parametrize(
    ("candle", "features"),
    [
        (bar(101, 102, 100.5, 101.5), UP),  # не коснулась EMA20
        (bar(100, 101, 98, 99), UP),  # закрылась ниже EMA20
        (bar(100, 102, 99.5, 101), {**UP, "ema_mid": 89.0}),  # EMA50 < EMA200: нет тренда
        (bar(91, 92, 85, 89), {**UP, "ema_fast": 88.0}),  # close ниже EMA200
    ],
)
def test_no_long_signal(candle: dict[str, float], features: dict[str, float]) -> None:
    signals = TrendFollowing("1h", {"allow_short": False}).on_bar(ctx_for([candle], [features]))
    assert signals == []


def test_short_is_mirror() -> None:
    # Тренд вниз, high 100.5 >= EMA20 100, close 99 < 100 -> SHORT, стоп 99 + 4 = 103
    (signal,) = TrendFollowing("1h").on_bar(ctx_for([bar(100, 100.5, 98, 99)], [DOWN]))
    assert signal.side is Side.SHORT
    assert signal.stop_loss == 103
    assert (
        TrendFollowing("1h", {"allow_short": False}).on_bar(
            ctx_for([bar(100, 100.5, 98, 99)], [DOWN])
        )
        == []
    )


def test_chandelier_trailing() -> None:
    # Вход на свече 1 (10:00 -> 01:00). Максимум high с входа = 110, ATR(22) = 3
    # стоп = 110 − 3 × 3 = 101. Свеча 0 (high 120) была до входа и не учитывается.
    rows = [bar(100, 120, 99, 100), bar(100, 105, 99, 104), bar(104, 110, 103, 108)]
    ctx = ctx_for(rows, [UP, UP, UP])
    position = Position(
        symbol="BTCUSDT",
        side=Side.LONG,
        size=1,
        entry_price=100,
        stop_loss=96,
        opened_at=datetime(2024, 1, 1, 1, 0, tzinfo=UTC),
    )
    assert TrendFollowing("1h").update_stop(ctx, position) == 101

    short = position.model_copy(update={"side": Side.SHORT, "stop_loss": 110})
    # минимум low с входа = 99 -> 99 + 9 = 108
    assert TrendFollowing("1h").update_stop(ctx, short) == 108


def test_signals_on_random_data_follow_rules_and_no_lookahead() -> None:
    bars = random_bars(1500)
    strategy = TrendFollowing("1h")
    signals = generate_signals(strategy, "BTCUSDT", bars)
    assert signals
    for s in signals:
        f = s.features
        if s.side is Side.LONG:
            assert f["ema50"] > f["ema200"] and s.entry_price > f["ema20"]
        else:
            assert f["ema50"] < f["ema200"] and s.entry_price < f["ema20"]
    assert find_lookahead(strategy, "BTCUSDT", bars, bars["open_time"].iloc[900]) == []
