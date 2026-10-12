import numpy as np
import pandas as pd
import pytest

from bot.research.channel_study import forward_returns, study


def candles(prices: list[float]) -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=len(prices), freq="1h", tz="UTC")
    return pd.DataFrame({"open_time": times, "open": prices})


def test_forward_returns_hand_example() -> None:
    df = candles([100, 110, 99, 121])
    times = df["open_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    opens = df["open"].to_numpy(dtype="float64")
    # Пост в 00:30 → вход по open 01:00 (110); через 1 ч → open 02:00 (99): −10%
    event = np.array([pd.Timestamp("2026-01-01 00:30", tz="UTC").value])
    assert forward_returns(times, opens, event, 1)[0] == pytest.approx(-10.0)
    # Через 2 ч → open 03:00 (121): +10%
    assert forward_returns(times, opens, event, 2)[0] == pytest.approx(10.0)
    # Через 3 ч данных нет
    assert np.isnan(forward_returns(times, opens, event, 3)[0])


def test_study_finds_a_perfect_channel() -> None:
    rng = np.random.default_rng(0)
    n = 24 * 400
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = candles(prices.tolist())
    # «Идеальный» канал: знает, куда пойдёт цена за 4 часа
    idx = rng.choice(np.arange(10, n - 10), size=150, replace=False)
    sign = np.sign(prices[idx + 4] - prices[idx])
    events = pd.DataFrame({"time": df["open_time"].iloc[idx].to_numpy(), "sign": sign})
    result = study(df, events, horizons=(4,), runs=200)[0]
    assert result.posts == 150
    assert result.hit_rate == 1.0
    assert result.percentile == 1.0
    assert result.mean_signed_pct > result.random_mean_pct
