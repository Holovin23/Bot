import math

import numpy as np
import pytest

from bot.analyzer.calibration import IsotonicCalibrator, brier_score, reliability_table


def test_brier_hand_example() -> None:
    # (0.8−1)² + (0.3−0)² + (0.6−1)² = 0.04 + 0.09 + 0.16 = 0.29; / 3
    assert brier_score([0.8, 0.3, 0.6], [1, 0, 1]) == pytest.approx(0.29 / 3)


def test_reliability_hand_example() -> None:
    table = reliability_table([0.05, 0.15, 0.12, 0.95], [0, 1, 0, 1], bins=10)
    assert table[0].count == 1
    assert table[1].count == 2
    assert table[1].mean_predicted == pytest.approx(0.135)
    assert table[1].observed == pytest.approx(0.5)
    assert table[9].count == 1  # 0.95 → последняя корзина
    assert math.isnan(table[5].observed)


def test_isotonic_fixes_overconfidence() -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.3, 0.7, 4000)
    labels = (rng.uniform(size=4000) < true_p).astype(int)
    raw = np.clip((true_p - 0.5) * 2 + 0.5, 0, 1)  # «самоуверенная» модель
    calibrated = IsotonicCalibrator().fit(raw[:2000], labels[:2000]).transform(raw[2000:])
    assert brier_score(calibrated, labels[2000:]) < brier_score(raw[2000:], labels[2000:])
