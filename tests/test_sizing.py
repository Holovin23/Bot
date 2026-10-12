import pytest

from bot.core.config import load_config
from bot.risk.sizing import position_size, round_down

BTC = {"qty_step": 0.001, "min_qty": 0.001, "max_leverage": 3.0}


def test_example_from_docs() -> None:
    # docs/05: капитал 10 000, риск 0.5% = 50, вход 65 450, стоп 65 000,
    # дистанция 450 + 30 = 480 -> 50 / 480 = 0.10416 -> 0.104 BTC,
    # номинал 0.104 × 65 450 = 6 806.8, плечо 0.68x
    result = position_size(
        equity=10_000,
        risk_fraction=0.005,
        entry_price=65_450,
        stop_loss=65_000,
        slippage_per_unit=30,
        **BTC,
    )
    assert result.qty == 0.104
    assert result.notional == pytest.approx(6_806.8)
    assert result.leverage == pytest.approx(0.68068)
    assert result.risk_amount == pytest.approx(0.104 * 480)  # 49.92 <= 50
    assert not result.capped_by_leverage


def test_short_is_symmetric() -> None:
    result = position_size(
        equity=10_000,
        risk_fraction=0.005,
        entry_price=65_000,
        stop_loss=65_450,
        slippage_per_unit=30,
        **BTC,
    )
    assert result.qty == 0.104


def test_leverage_cap() -> None:
    # Стоп в 10 USDT: 50 / 10 = 5 BTC (≈ 325 000 USDT, 32x) -> ограничение 3x:
    # 10 000 × 3 / 65 000 = 0.4615 -> 0.461
    result = position_size(
        equity=10_000,
        risk_fraction=0.005,
        entry_price=65_000,
        stop_loss=64_990,
        slippage_per_unit=0,
        **BTC,
    )
    assert result.capped_by_leverage
    assert result.qty == 0.461
    assert result.leverage <= 3.0


def test_below_min_qty_rejected() -> None:
    # 10 USDT капитала: 0.05 / 480 = 0.0001 BTC < 0.001
    result = position_size(
        equity=10,
        risk_fraction=0.005,
        entry_price=65_450,
        stop_loss=65_000,
        slippage_per_unit=30,
        **BTC,
    )
    assert result.qty == 0
    assert "меньше минимального" in result.reason


def test_round_down_avoids_float_errors() -> None:
    assert round_down(0.3, 0.1) == 0.3  # 0.3 / 0.1 в float = 2.9999999999999996
    assert round_down(1.239, 0.01) == 1.23
    assert round_down(5, 0.001) == 5.0


def test_risk_limits_from_config() -> None:
    cfg = load_config(env_file=None)
    assert cfg.risk.risk_per_trade == 0.005
    assert cfg.risk.max_leverage == 3.0
    assert cfg.instruments["BTCUSDT"].qty_step == 0.001
