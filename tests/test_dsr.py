import math

import pytest

from bot.research.dsr import (
    bonferroni,
    deflated_sharpe,
    expected_max_sharpe,
    mean_p_value,
    moments,
    probabilistic_sharpe,
    sharpe,
)


def test_sharpe_by_hand() -> None:
    # R: 1, −1, 2, 0 → среднее 0,5; отклонения 0,5 −1,5 1,5 −0,5 → сумма квадратов 5;
    # дисперсия 5/3, std 1,29099 → Sharpe 0,38730
    assert sharpe([1, -1, 2, 0]) == pytest.approx(0.5 / math.sqrt(5 / 3))
    assert sharpe([1]) == 0.0 and sharpe([1, 1]) == 0.0


def test_moments_of_symmetric_sample() -> None:
    skew, kurt = moments([-1, 1, -1, 1])
    assert skew == pytest.approx(0) and kurt == pytest.approx(1)


def test_psr_by_hand() -> None:
    # SR 0,1, n 101, нормальные хвосты: z = 0,1 × 10 / √1,005 = 0,99751 → Φ = 0,84074
    assert probabilistic_sharpe(0.1, 0.0, 101) == pytest.approx(0.84074, abs=1e-5)
    # Толстые хвосты (куртозис 9): z = 1 / √(1 + 8/4 × 0,01) = 1 / √1,02 = 0,99015 → Φ = 0,83895
    fat = probabilistic_sharpe(0.1, 0.0, 101, kurt=9)
    assert fat == pytest.approx(0.83895, abs=1e-5) and fat < 0.84074


def test_expected_max_sharpe_by_hand() -> None:
    # N = 10, V = 1: 0,42278 × Φ⁻¹(0,9) + 0,57722 × Φ⁻¹(1 − 1/(10e))
    #              = 0,42278 × 1,28155 + 0,57722 × 1,78966 ≈ 1,5746
    assert expected_max_sharpe(10, 1.0) == pytest.approx(1.5746, abs=1e-4)
    assert expected_max_sharpe(1, 1.0) == 0.0
    # больше попыток — выше планка
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01)


def test_deflated_sharpe_drops_with_more_trials() -> None:
    r = [1.5, -1, 2, -1, 0.5, -1, 3, -1, 1, -1] * 5  # 50 сделок, средний R 0,3
    one = deflated_sharpe(r, 1, 0.02)
    many = deflated_sharpe(r, 200, 0.02)
    assert one > 0.9 and many < one


def test_bonferroni_and_p_value() -> None:
    assert bonferroni(0.01, 20) == pytest.approx(0.2)
    assert bonferroni(0.2, 20) == 1.0
    # Sharpe 0,3873 × √4 = 0,7746 → p = 1 − Φ(0,7746) = 0,2193
    assert mean_p_value([1, -1, 2, 0]) == pytest.approx(0.2193, abs=1e-4)
