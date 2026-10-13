"""Стоп и цель по сценарию. Все числа посчитаны вручную."""

import pytest

from bot.scenarios.levels import LevelParams, TradeLevels, trade_levels
from bot.scenarios.models import Scenario, Trigger


def sc(
    side: str, trigger: Trigger, stop: float | None = None, targets: list[float] | None = None
) -> Scenario:
    return Scenario(symbol="BTCUSDT", side=side, trigger=trigger, stop=stop, targets=targets or [])  # type: ignore[arg-type]


ABOVE_100 = Trigger(type="close_above", level=100)


def test_default_stop_and_target_long() -> None:
    # стоп = 100 − 0,5 × 2 = 99; риск = 101 − 99 = 2; цель = 101 + 2 × 2 = 105
    levels = trade_levels(sc("LONG", ABOVE_100), entry=101, atr_value=2)
    assert levels == TradeLevels(99, 105, stop_from_author=False, take_from_author=False)


def test_first_author_target_with_enough_rr() -> None:
    # риск 2: цель 103 даёт 1R (мало), 106 — 2,5R → берём 106
    levels = trade_levels(sc("LONG", ABOVE_100, targets=[106, 103]), entry=101, atr_value=2)
    assert isinstance(levels, TradeLevels)
    assert (levels.take, levels.take_from_author) == (106, True)


def test_author_targets_too_close_means_no_trade() -> None:
    result = trade_levels(sc("LONG", ABOVE_100, targets=[103]), entry=101, atr_value=2)
    assert result == "цели автора ближе 2R от входа"


def test_author_stop_short() -> None:
    # SHORT: риск = 102 − 99 = 3; цель = 99 − 2 × 3 = 93
    s = sc("SHORT", Trigger(type="close_below", level=100), stop=102)
    assert trade_levels(s, entry=99, atr_value=2) == TradeLevels(102, 93, True, False)


def test_now_entry_uses_wider_atr_stop() -> None:
    # стоп = 50 − 1,5 × 1 = 48,5; риск 1,5; цель = 50 + 3 = 53
    levels = trade_levels(sc("LONG", Trigger(type="now")), entry=50, atr_value=1)
    assert levels == TradeLevels(48.5, 53, False, False)


def test_zone_stop_behind_far_edge() -> None:
    # LONG от зоны 95–96: стоп = 95 − 0,5 × 2 = 94; риск = 96,5 − 94 = 2,5; цель = 101,5
    s = sc("LONG", Trigger(type="touch_zone", zone_low=95, zone_high=96))
    levels = trade_levels(s, entry=96.5, atr_value=2)
    assert isinstance(levels, TradeLevels)
    assert levels.stop == 94
    assert levels.take == pytest.approx(101.5)


def test_entry_beyond_stop_and_unknown_atr() -> None:
    assert trade_levels(sc("LONG", ABOVE_100, stop=99), entry=98, atr_value=2) == (
        "цена входа уже за стопом"
    )
    assert "ATR" in str(trade_levels(sc("LONG", ABOVE_100), entry=101, atr_value=float("nan")))


def test_params_change_defaults() -> None:
    # буфер 1 ATR: стоп 98, риск 3; R:R 3 → цель 110
    p = LevelParams(stop_buffer_atr=1.0, default_rr=3.0, min_rr=3.0)
    assert trade_levels(sc("LONG", ABOVE_100), 101, 2, p) == TradeLevels(98, 110, False, False)
