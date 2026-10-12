import math

import pandas as pd
import pytest

from bot.features.indicators import atr, rma, true_range


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
