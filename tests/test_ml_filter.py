import numpy as np
import pandas as pd
import pytest

from bot.analyzer.dataset import FEATURE_COLUMNS
from bot.analyzer.ml_filter import (
    FilterModel,
    choose_threshold,
    oof_predictions,
    purged_kfold,
    train_filter,
)


def times(*hours: float) -> pd.Series:
    base = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.Series([base + pd.Timedelta(hours=h) for h in hours])


def test_purged_kfold_hand_example() -> None:
    # 4 сделки по 2 часа, сигнал каждые 3 часа; вторая сделка длинная: 3 → 8 ч
    signals = times(0, 3, 6, 9)
    exits = times(2, 8, 8, 11)
    splits = purged_kfold(signals, exits, n_splits=2, embargo=0.0)
    (train1, test1), (train2, test2) = splits
    assert test1.tolist() == [0, 1]
    # Блок 1 заканчивается в 8 ч (выход сделки 1) — сделка 2 с сигналом в 6 ч пересекается
    assert train1.tolist() == [3]
    assert test2.tolist() == [2, 3]
    # Блок 2 начинается в 6 ч — сделка 1 (выход в 8 ч) убирается из обучения
    assert train2.tolist() == [0]


def test_embargo_removes_trades_right_after_block() -> None:
    signals = times(0, 1, 2, 3, 100)
    exits = times(0.5, 1.5, 2.5, 3.5, 101)
    (train, _), _ = purged_kfold(signals, exits, n_splits=2, embargo=0.0)
    assert 3 in train.tolist()
    (train_e, _), _ = purged_kfold(signals, exits, n_splits=2, embargo=0.05)  # 5% от 101 ч
    assert 3 not in train_e.tolist()


def test_choose_threshold_hand_example() -> None:
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    r = np.array([-1, -1, -1, -1, 1, 1, 1, -1, 1, 1], dtype="float64")
    threshold = choose_threshold(probs, r)
    kept = probs >= threshold
    # Все 10 сделок: средний R = 0. Оставить 6 с p ≥ 0.5: (1+1+1−1+1+1)/6 = 0.667 — лучше;
    # оставить 5 с p ≥ 0.6: (1+1−1+1+1)/5 = 0.6; 7 с p ≥ 0.4: 3/7 = 0.43
    assert kept.sum() == 6
    assert r[kept].mean() == pytest.approx(4 / 6)


def synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    p = 1 / (1 + np.exp(-1.5 * data["rsi"]))  # исход зависит от одного признака
    data["label"] = (rng.uniform(size=n) < p).astype(int)
    data["r_multiple"] = np.where(data["label"] == 1, 1.0, -1.0)
    start = pd.Timestamp("2022-01-01", tz="UTC")
    data["signal_time"] = [start + pd.Timedelta(hours=6 * i) for i in range(n)]
    data["exit_time"] = data["signal_time"] + pd.Timedelta(hours=4)
    return data


def test_models_learn_signal_and_filter_helps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data = synthetic()
    for kind in ("logreg", "lgbm"):
        oof = oof_predictions(data, kind)  # type: ignore[arg-type]
        assert not np.isnan(oof).any()
        model = train_filter(data, kind)  # type: ignore[arg-type]
        assert model.report["auc_oof"] > 0.7
        assert model.report["expectancy_kept"] > model.report["expectancy_all"]
        assert model.report["brier_calibrated"] < model.report["brier_base"]
    path = tmp_path / "f.joblib"
    model.save(path)
    loaded = FilterModel.load(path)
    p = loaded.probability(data.head(5))
    assert ((p >= 0) & (p <= 1)).all()
