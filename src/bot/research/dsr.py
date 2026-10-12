"""Поправка на множественные проверки (этап 9).

Чем больше вариантов перебрано, тем выше лучший Sharpe чисто случайно. Deflated Sharpe
Ratio (Bailey, López de Prado, 2014) отвечает на вопрос: какова вероятность, что
настоящий Sharpe стратегии больше нуля, с учётом того, сколько вариантов мы пробовали,
длины выборки и «толстых хвостов» распределения результатов.

Sharpe здесь — по сделкам: среднее R / стандартное отклонение R (без годового пересчёта).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_NORMAL = NormalDist()


def sharpe(values: Sequence[float]) -> float:
    """Среднее / стандартное отклонение (выборочное, ddof=1). Меньше 2 значений — 0."""
    x = np.asarray(values, dtype="float64")
    if len(x) < 2:
        return 0.0
    std = float(x.std(ddof=1))
    return float(x.mean()) / std if std > 0 else 0.0


def moments(values: Sequence[float]) -> tuple[float, float]:
    """Асимметрия и куртозис (обычный, у нормального = 3)."""
    x = np.asarray(values, dtype="float64")
    if len(x) < 3 or x.std() == 0:
        return 0.0, 3.0
    z = (x - x.mean()) / x.std()
    return float((z**3).mean()), float((z**4).mean())


def probabilistic_sharpe(
    sr: float, sr_benchmark: float, n: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """PSR: вероятность, что настоящий Sharpe больше sr_benchmark.
    PSR = Φ((SR − SR*) × √(n − 1) / √(1 − γ3·SR + (γ4 − 1)/4 · SR²))."""
    if n < 2:
        return 0.0
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    if denom <= 0:
        return 0.0
    return _NORMAL.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Ожидаемый максимум Sharpe среди n_trials вариантов, у которых настоящий Sharpe = 0:
    √V × ((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)))."""
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    a = _NORMAL.inv_cdf(1 - 1 / n_trials)
    b = _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(values: Sequence[float], n_trials: int, sr_variance: float) -> float:
    """DSR = PSR с порогом «ожидаемый максимум Sharpe при n_trials попытках»."""
    skew, kurt = moments(values)
    benchmark = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe(sharpe(values), benchmark, len(values), skew, kurt)


def bonferroni(p_value: float, n_trials: int) -> float:
    """Самая простая поправка: p × число проверок (не больше 1)."""
    return min(1.0, p_value * max(n_trials, 1))


def mean_p_value(values: Sequence[float]) -> float:
    """Односторонний p-value гипотезы «средний R ≤ 0» (нормальное приближение t-теста)."""
    n = len(values)
    if n < 2:
        return 1.0
    t = sharpe(values) * math.sqrt(n)
    return 1 - _NORMAL.cdf(t)
