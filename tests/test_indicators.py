import math

import numpy as np
import pandas as pd
import pytest

from bot.features.indicators import (
    adx,
    atr,
    bollinger,
    bollinger_width,
    ema,
    percentile_rank,
    rma,
    rsi,
    true_range,
)


def test_rma_by_hand() -> None:
    # length 3: первое значение — среднее (1 + 2 + 3) / 3 = 2
    # дальше: (2 * 2 + 6) / 3 = 3.3333; (3.3333 * 2 + 0) / 3 = 2.2222
    out = rma(pd.Series([1.0, 2.0, 3.0, 6.0, 0.0]), 3)
    assert math.isnan(out[0]) and math.isnan(out[1])
    assert out[2:].tolist() == pytest.approx([2.0, 3.3333333, 2.2222222])


def test_true_range_uses_previous_close() -> None:
    df = pd.DataFrame(
        {
            "high": [10.0, 12.0, 11.0],
            "low": [8.0, 11.0, 7.0],
            "close": [9.0, 11.5, 8.0],
        }
    )
    # 1: high-low = 2
    # 2: max(12-11=1, |12-9|=3, |11-9|=2) = 3 (гэп вверх от закрытия 9)
    # 3: max(11-7=4, |11-11.5|=0.5, |7-11.5|=4.5) = 4.5
    assert true_range(df).tolist() == [2.0, 3.0, 4.5]
    # ATR(2): (2 + 3) / 2 = 2.5; потом (2.5 * 1 + 4.5) / 2 = 3.5
    assert atr(df, 2).tolist()[1:] == [2.5, 3.5]


def test_ema_by_hand() -> None:
    # length 3, alpha = 2 / 4 = 0.5. Затравка: (2 + 4 + 6) / 3 = 4
    # дальше: 0.5 * 10 + 0.5 * 4 = 7; 0.5 * 0 + 0.5 * 7 = 3.5
    out = ema(pd.Series([2.0, 4.0, 6.0, 10.0, 0.0]), 3)
    assert out[2:].tolist() == [4.0, 7.0, 3.5]


def test_bollinger_by_hand() -> None:
    # Окно 4: [1, 2, 3, 6], среднее 3, отклонения -2, -1, 0, 3,
    # дисперсия (4 + 1 + 0 + 9) / 4 = 3.5, std = 1.8708; полосы 3 ± 2 * 1.8708
    out = bollinger(pd.Series([1.0, 2.0, 3.0, 6.0]), length=4, std=2.0)
    last = out.iloc[-1]
    assert last["bb_middle"] == 3.0
    assert last["bb_upper"] == pytest.approx(3 + 2 * 1.8708287)
    assert last["bb_lower"] == pytest.approx(3 - 2 * 1.8708287)
    assert out["bb_middle"][:3].isna().all()


def random_candles(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 1, rows).cumsum()
    high = close + rng.uniform(0, 2, rows)
    low = close - rng.uniform(0, 2, rows)
    return pd.DataFrame({"high": high, "low": low, "close": close})


def test_matches_pandas_ta() -> None:
    ta = pytest.importorskip("pandas_ta")
    df = random_candles()
    close = df["close"]

    for length in (20, 50, 200):
        expected = ta.ema(close, length)
        assert np.allclose(ema(close, length), expected, equal_nan=True)

    bands = bollinger(close, 20, 2.0)
    # TradingView считает std по генеральной совокупности (ddof=0); у pandas-ta по умолчанию 1.
    expected_bb = ta.bbands(close, length=20, std=2.0, ddof=0, talib=False)
    assert np.allclose(bands["bb_lower"], expected_bb.iloc[:, 0], equal_nan=True)
    assert np.allclose(bands["bb_middle"], expected_bb.iloc[:, 1], equal_nan=True)
    assert np.allclose(bands["bb_upper"], expected_bb.iloc[:, 2], equal_nan=True)

    expected_atr = ta.atr(df["high"], df["low"], close, length=14)
    ours = atr(df, 14)
    assert np.allclose(ours, expected_atr, equal_nan=True)


def test_rsi_by_hand() -> None:
    # length 2. Изменения: +2, −1, +3, −2
    # рост: [2, 0, 3, 0] -> RMA(2): среднее (2, 0) = 1; (1 + 3) / 2 = 2; (2 + 0) / 2 = 1
    # падение: [0, 1, 0, 2] -> 0.5; 0.25; 1.125
    # RSI на последней: 100 − 100 / (1 + 1 / 1.125) = 47.0588
    out = rsi(pd.Series([10.0, 12.0, 11.0, 14.0, 12.0]), 2)
    assert math.isnan(out[1])
    assert out[2] == pytest.approx(100 - 100 / (1 + 1 / 0.5))
    assert out[4] == pytest.approx(47.0588235)


def test_rsi_extremes() -> None:
    assert rsi(pd.Series([1.0, 2, 3, 4, 5]), 2).iloc[-1] == 100
    assert rsi(pd.Series([5.0, 4, 3, 2, 1]), 2).iloc[-1] == 0


def test_adx_by_hand_strong_uptrend() -> None:
    # Каждая свеча выше предыдущей: +DM > 0, −DM = 0 -> −DI = 0, +DI > 0, DX = 100, ADX -> 100
    df = pd.DataFrame(
        {
            "high": [11.0, 12, 13, 14, 15, 16, 17],
            "low": [9.0, 10, 11, 12, 13, 14, 15],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5],
        }
    )
    out = adx(df, 2)
    # TR (кроме первой) = max(2, |high − prev close| = 1.5, |low − prev close| = 0.5) = 2
    # +DM = 1 -> +DI = 100 × 1 / 2 = 50
    assert out["plus_di"].iloc[-1] == pytest.approx(50)
    assert out["minus_di"].iloc[-1] == 0
    assert out["adx"].iloc[-1] == pytest.approx(100)


def test_percentile_rank_and_width() -> None:
    values = pd.Series([5.0, 3.0, 4.0, 1.0, 2.0, 6.0])
    rank = percentile_rank(values, 3)
    # окно [5, 3, 4]: <= 4 два из трёх; [3, 4, 1]: 1 — минимум, 1/3; [1, 2, 6]: 6 — максимум
    assert rank.iloc[2] == pytest.approx(2 / 3)
    assert rank.iloc[3] == pytest.approx(1 / 3)
    assert rank.iloc[5] == 1.0
    assert rank.iloc[:2].isna().all()
    # ширина BB по [1, 2, 3, 6] (см. выше): 4 × 1.8708 / 3
    width = bollinger_width(pd.Series([1.0, 2.0, 3.0, 6.0]), length=4)
    assert width.iloc[-1] == pytest.approx(4 * 1.8708287 / 3)


def test_rsi_adx_match_pandas_ta() -> None:
    ta = pytest.importorskip("pandas_ta")
    df = random_candles(800)
    # pandas-ta начинает RSI с другой затравки; к 300-й свече разница исчезает
    expected_rsi = ta.rsi(df["close"], length=14)
    assert np.allclose(rsi(df["close"], 14).iloc[300:], expected_rsi.iloc[300:], atol=1e-6)
    expected = ta.adx(df["high"], df["low"], df["close"], length=14)
    ours = adx(df, 14)
    assert np.allclose(ours["adx"].iloc[300:], expected["ADX_14"].iloc[300:], rtol=1e-4)
    assert np.allclose(ours["plus_di"].iloc[300:], expected["DMP_14"].iloc[300:], rtol=1e-4)
