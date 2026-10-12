import pandas as pd
import pytest

from bot.backtest.costs import (
    CostModel,
    entry_fill,
    exit_fill,
    funding_events,
    funding_payment,
)
from bot.core.config import load_config
from bot.core.models import Side

COSTS = CostModel(slippage={"BTCUSDT": 0.0003, "ETHUSDT": 0.0005})


def test_values_from_config(tmp_path: object) -> None:
    model = CostModel.from_config(load_config(env_file=None).costs)
    assert model.taker_fee == 0.00055
    assert model.slippage_rate("BTCUSDT") == 0.0003
    assert model.slippage_rate("BTCUSDT", stop=True) == pytest.approx(0.0006)


def test_fee_by_hand() -> None:
    # 0.1 BTC по 65 000 = 6 500 USDT; taker 0.055% = 3.575; maker 0.02% = 1.3
    assert COSTS.fee(0.1 * 65_000) == pytest.approx(3.575)
    assert COSTS.fee(0.1 * 65_000, maker=True) == pytest.approx(1.3)


def test_slippage_direction_by_hand() -> None:
    # LONG входит покупкой: 65 000 × 1.0003 = 65 019.5; выходит продажей: 65 000 × 0.9997 = 64 980.5
    assert entry_fill(65_000, Side.LONG, 0.0003) == pytest.approx(65_019.5)
    assert exit_fill(65_000, Side.LONG, 0.0003) == pytest.approx(64_980.5)
    # SHORT наоборот
    assert entry_fill(65_000, Side.SHORT, 0.0003) == pytest.approx(64_980.5)
    assert exit_fill(65_000, Side.SHORT, 0.0003) == pytest.approx(65_019.5)


def test_funding_sign_by_hand() -> None:
    # 0.1 BTC × 65 000 × 0.0001 = 0.65 USDT
    assert funding_payment(Side.LONG, 0.1, 65_000, 0.0001) == pytest.approx(-0.65)
    assert funding_payment(Side.SHORT, 0.1, 65_000, 0.0001) == pytest.approx(0.65)
    # отрицательная ставка: лонг получает
    assert funding_payment(Side.LONG, 0.1, 65_000, -0.0001) == pytest.approx(0.65)


def test_funding_events_window() -> None:
    times = pd.to_datetime(["2024-03-01 00:00", "2024-03-01 08:00", "2024-03-01 16:00"], utc=True)
    funding = pd.DataFrame({"time": times, "funding_rate": [0.0001, 0.0002, 0.0003]})
    # Вход 00:00 (ровно в момент выплаты — её не платим), выход 16:00 (платим)
    events = funding_events(funding, times[0], times[2])
    assert events["funding_rate"].tolist() == [0.0002, 0.0003]
    # Позиция 01:00–07:59 не застала выплат
    inside = funding_events(
        funding,
        pd.Timestamp("2024-03-01 01:00", tz="UTC"),
        pd.Timestamp("2024-03-01 07:59", tz="UTC"),
    )
    assert inside.empty
