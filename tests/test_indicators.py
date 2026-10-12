import math

import numpy as np
import pandas as pd
import pytest

from bot.features.indicators import atr, bollinger, ema, rma, true_range


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
