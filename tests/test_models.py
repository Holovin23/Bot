from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from bot.core.models import Candle, Side, Signal, Trade

BAR = datetime(2026, 10, 12, 14, 0, tzinfo=UTC)


def make_signal(**overrides: object) -> Signal:
    params: dict[str, object] = {
        "strategy": "telegram_scenario",
        "strategy_version": "1.0.0",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "bar_time": BAR,
        "side": Side.LONG,
        "entry_price": 65450,
        "stop_loss": 65000,
        "take_profit": 66300,
    }
    params.update(overrides)
    return Signal.create(**params)  # type: ignore[arg-type]


def test_same_input_gives_same_signal_id() -> None:
    assert make_signal().signal_id == make_signal().signal_id
    assert make_signal().signal_id == "ts-BTCUSDT-1h-20261012T1400-L"


def test_signal_id_differs_by_bar_side_and_symbol() -> None:
    ids = {
        make_signal().signal_id,
        make_signal(bar_time=BAR + timedelta(hours=1)).signal_id,
        make_signal(side=Side.SHORT, stop_loss=66000, take_profit=64000).signal_id,
        make_signal(symbol="ETHUSDT").signal_id,
    }
    assert len(ids) == 4


def test_signal_id_ignores_timezone_of_input() -> None:
    kyiv = timezone(timedelta(hours=3))
    same_moment = BAR.astimezone(kyiv)
    assert make_signal(bar_time=same_moment).signal_id == make_signal().signal_id


def test_signal_id_fits_bybit_order_link_id() -> None:
    signal = make_signal(strategy="random_entry", symbol="ETHUSDT", timeframe="15m")
    assert len(signal.signal_id) <= 36


def test_created_at_is_close_of_signal_bar() -> None:
    assert make_signal().created_at == datetime(2026, 10, 12, 15, 0, tzinfo=UTC)


def test_signal_without_stop_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_signal(stop_loss=None)


def test_stop_on_wrong_side_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_signal(stop_loss=66000)  # LONG со стопом выше входа
    with pytest.raises(ValidationError):
        make_signal(side=Side.SHORT, stop_loss=65000, take_profit=None)


def test_reward_to_risk_from_docs_example() -> None:
    # Пример из docs/02-architecture.md: риск 450, цель 850, R:R = 850 / 450 ≈ 1.89
    signal = make_signal()
    assert signal.risk_per_unit == 450
    assert signal.reward_to_risk == pytest.approx(1.8889, abs=1e-4)


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Candle(
            symbol="BTCUSDT",
            timeframe="1m",
            open_time=datetime(2026, 1, 1),
            open=1,
            high=1,
            low=1,
            close=1,
            volume=0,
        )


def test_trade_pnl_short_by_hand() -> None:
    # SHORT 0.1 BTC: вход 65000, выход 64000 -> (65000 - 64000) * 0.1 = 100
    # комиссии 7, funding -2 (заплатили) -> чистый результат 100 - 7 - 2 = 91
    trade = Trade(
        signal_id="x",
        symbol="BTCUSDT",
        side=Side.SHORT,
        qty=0.1,
        entry_time=BAR,
        entry_price=65000,
        exit_time=BAR + timedelta(hours=5),
        exit_price=64000,
        fees=7,
        funding=-2,
    )
    assert trade.gross_pnl == pytest.approx(100)
    assert trade.net_pnl == pytest.approx(91)
