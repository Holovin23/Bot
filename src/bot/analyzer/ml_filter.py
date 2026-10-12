"""ML-фильтр сигналов (docs/02-architecture.md, слой B).

Модель оценивает вероятность, что сделка закроется в плюс. Фильтр может только
пропустить сигнал (p < порога) — никогда не создаёт сделку и не меняет её размер и стоп.

Кросс-валидация по времени с «очисткой» (purged k-fold): из обучения убираются сделки,
чей интервал [сигнал, выход] пересекается с блоком проверки, и сделки в течение эмбарго
сразу после него. Иначе модель «видит» исход соседних сделок.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bot.analyzer.calibration import IsotonicCalibrator, brier_score
from bot.analyzer.dataset import FEATURE_COLUMNS

ModelKind = Literal["logreg", "lgbm"]
MIN_KEEP_SHARE = 0.5


def purged_kfold(
    signal_times: pd.Series,
    exit_times: pd.Series,
    n_splits: int = 5,
    embargo: float = 0.01,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Индексы (обучение, проверка). Данные должны быть отсортированы по signal_time."""
    n = len(signal_times)
    signals = pd.Series(pd.to_datetime(signal_times.to_numpy(), utc=True))
    exits = pd.Series(pd.to_datetime(exit_times.to_numpy(), utc=True))
    span = exits.max() - signals.min() if n else pd.Timedelta(0)
    embargo_delta = span * embargo
    splits = []
    for block in np.array_split(np.arange(n), n_splits):
        if not len(block):
            continue
        t0 = signals.iloc[int(block[0])]
        t1 = max(exits.iloc[int(i)] for i in block)
        before = exits < t0
        after = signals > t1 + embargo_delta
        train = np.flatnonzero((before | after).to_numpy())
        splits.append((train.astype(np.int64), block.astype(np.int64)))
    return splits


def make_model(kind: ModelKind) -> Any:
    if kind == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000),
        )
    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=0,
        verbose=-1,
    )


def _fit_predict(
    kind: ModelKind, x_train: pd.DataFrame, y_train: NDArray[Any], x_test: pd.DataFrame
) -> NDArray[np.float64]:
    if len(np.unique(y_train)) < 2:  # одна метка — модель не обучить, прогноз = доля
        base = float(y_train.mean()) if len(y_train) else 0.5
        return np.full(len(x_test), base)
    model = make_model(kind)
    model.fit(x_train, y_train)
    return np.asarray(model.predict_proba(x_test)[:, 1], dtype="float64")


def oof_predictions(
    data: pd.DataFrame, kind: ModelKind, n_splits: int = 5, embargo: float = 0.01
) -> NDArray[np.float64]:
    """Вероятности «вне блока»: каждая сделка оценена моделью, которая её не видела."""
    data = data.sort_values("signal_time").reset_index(drop=True)
    x = data[FEATURE_COLUMNS]
    y = data["label"].to_numpy()
    out = np.full(len(data), np.nan)
    for train, test in purged_kfold(data["signal_time"], data["exit_time"], n_splits, embargo):
        out[test] = _fit_predict(kind, x.iloc[train], y[train], x.iloc[test])
    return out


def choose_threshold(
    probs: NDArray[np.float64], r_multiples: NDArray[np.float64], min_keep: float = MIN_KEEP_SHARE
) -> float:
    """Порог p, при котором средний R оставшихся сделок максимален, но остаётся
    не меньше min_keep сделок. 0 — фильтр не нужен (лучше всех оставить)."""
    best_threshold, best_mean = 0.0, float(np.mean(r_multiples)) if len(r_multiples) else 0.0
    for q in np.linspace(0.0, 1.0 - min_keep, 11)[1:]:
        threshold = float(np.quantile(probs, q))
        kept = probs >= threshold
        if kept.mean() < min_keep or not kept.any():
            continue
        mean = float(r_multiples[kept].mean())
        if mean > best_mean:
            best_threshold, best_mean = threshold, mean
    return best_threshold


@dataclass
class FilterModel:
    kind: ModelKind
    model: Any
    calibrator: IsotonicCalibrator
    threshold: float
    features: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    report: dict[str, Any] = field(default_factory=dict)

    def probability(self, x: pd.DataFrame) -> NDArray[np.float64]:
        raw = np.asarray(self.model.predict_proba(x[self.features])[:, 1], dtype="float64")
        return self.calibrator.transform(raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> FilterModel:
        model = joblib.load(path)
        if not isinstance(model, FilterModel):
            raise TypeError(f"{path} — не модель фильтра")
        return model


def _auc(labels: NDArray[Any], probs: NDArray[np.float64]) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def train_filter(data: pd.DataFrame, kind: ModelKind, n_splits: int = 5) -> FilterModel:
    """Обучение на всех сделках in-sample. Калибровка и порог — по OOF-прогнозам."""
    data = data.sort_values("signal_time").reset_index(drop=True)
    y = data["label"].to_numpy()
    r = data["r_multiple"].to_numpy(dtype="float64")
    oof = oof_predictions(data, kind, n_splits)
    calibrator = IsotonicCalibrator().fit(oof, y)
    calibrated = calibrator.transform(oof)
    threshold = choose_threshold(calibrated, r)
    model = make_model(kind)
    model.fit(data[FEATURE_COLUMNS], y)
    kept = calibrated >= threshold
    report = {
        "trades": len(data),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "auc_oof": _auc(y, oof),
        "brier_raw": brier_score(oof, y),
        "brier_calibrated": brier_score(calibrated, y),
        "brier_base": brier_score(np.full(len(y), y.mean()), y) if len(y) else float("nan"),
        "threshold": threshold,
        "kept_share": float(kept.mean()) if len(kept) else 0.0,
        "expectancy_all": float(r.mean()) if len(r) else 0.0,
        "expectancy_kept": float(r[kept].mean()) if kept.any() else 0.0,
        "oof_calibrated": calibrated.tolist(),
        "labels": y.tolist(),
    }
    return FilterModel(kind, model, calibrator, threshold, report=report)
