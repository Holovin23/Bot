import math

import pandas as pd
import pytest

from bot.research.robustness import (
    max_drawdown_of_pnls,
    monte_carlo_drawdown,
    plateau_scores,
)


def test_drawdown_hand_example() -> None:
    # 1000 → 1100 (пик) → 990 → 1045: просадка (1100 − 990) / 1100 = 10%
    assert max_drawdown_of_pnls([100, -110, 55], 1000) == pytest.approx(10.0)
    assert max_drawdown_of_pnls([], 1000) == 0.0


def test_monte_carlo_bounds() -> None:
    pnls = [100, -110, 55, -20, 30]
    mc = monte_carlo_drawdown(pnls, 1000, runs=200, seed=1)
    assert mc.original_dd_pct == pytest.approx(10.0)
    # Хуже всего — все убытки подряд в начале: 1000 → 870, просадка 13%
    assert mc.worst_dd_pct <= 13.0 + 1e-9
    assert mc.median_dd_pct <= mc.p95_dd_pct <= mc.worst_dd_pct


def test_plateau_hand_example() -> None:
    # Сетка 3×1: PF 1.0, 2.0, 1.2 → у средней точки соседи 1.0 и 1.2
    table = pd.DataFrame({"a": [1, 2, 3], "b": [5, 5, 5], "profit_factor": [1.0, 2.0, 1.2]})
    scores = plateau_scores(table, ["a", "b"])
    assert scores.tolist() == pytest.approx([1.5, (1.0 + 2.0 + 1.2) / 3, 1.6])


def test_plateau_ignores_thin_variants() -> None:
    table = pd.DataFrame({"a": [1, 2, 3], "profit_factor": [5.0, 1.3, 1.1], "trades": [3, 50, 50]})
    scores = plateau_scores(table, ["a"], min_trades=30)
    assert math.isnan(scores[0])
    assert scores[1] == pytest.approx((1.3 + 1.1) / 2)
    # Бесконечный PF ограничивается 10
    inf = pd.DataFrame({"a": [1, 2], "profit_factor": [math.inf, 1.0]})
    assert plateau_scores(inf, ["a"])[0] == pytest.approx(5.5)
