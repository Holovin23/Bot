"""Калибровка вероятностей и её проверка (docs/02-architecture.md, слой B).

Brier score = среднее (p − y)². Таблица надёжности: прогнозы делятся на корзины по p,
в каждой сравнивается средний прогноз и фактическая доля успехов. У откалиброванной
модели эти числа близки.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.isotonic import IsotonicRegression


def brier_score(probs: ArrayLike, labels: ArrayLike) -> float:
    p = np.asarray(probs, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


@dataclass(frozen=True)
class ReliabilityBin:
    low: float
    high: float
    count: int
    mean_predicted: float
    observed: float


def reliability_table(probs: ArrayLike, labels: ArrayLike, bins: int = 10) -> list[ReliabilityBin]:
    p = np.asarray(probs, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    table = []
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        table.append(
            ReliabilityBin(
                low=float(edges[b]),
                high=float(edges[b + 1]),
                count=n,
                mean_predicted=float(p[mask].mean()) if n else float("nan"),
                observed=float(y[mask].mean()) if n else float("nan"),
            )
        )
    return table


class IsotonicCalibrator:
    """Монотонное отображение «сырая вероятность → откалиброванная»."""

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.fitted = False

    def fit(self, probs: ArrayLike, labels: ArrayLike) -> IsotonicCalibrator:
        self._model.fit(np.asarray(probs, dtype="float64"), np.asarray(labels, dtype="float64"))
        self.fitted = True
        return self

    def transform(self, probs: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(probs, dtype="float64")
        if not self.fitted:
            return p
        return np.asarray(self._model.predict(p), dtype="float64")
