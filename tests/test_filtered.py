import numpy as np
import pandas as pd

from bot.analyzer.calibration import IsotonicCalibrator
from bot.analyzer.filtered import FilteredStrategy
from bot.analyzer.ml_filter import FilterModel
from bot.strategies.base import generate_signals
from bot.strategies.random_entry import RandomEntry

from .strategy_helpers import random_bars


class ByHour:
    """«Модель»: вероятность 0.9 для чётных часов, 0.1 для нечётных."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.where(x["hour"].to_numpy() % 2 == 0, 0.9, 0.1)
        return np.column_stack([1 - p, p])


def test_filter_only_removes_signals() -> None:
    bars = random_bars(600)
    base = RandomEntry("1h", {"seed": 1, "probability": 0.3})
    model = FilterModel("lgbm", ByHour(), IsotonicCalibrator(), threshold=0.5)
    filtered = FilteredStrategy(base, model)
    plain = generate_signals(base, "BTCUSDT", bars)
    kept = generate_signals(filtered, "BTCUSDT", bars)
    assert 0 < len(kept) < len(plain)
    plain_by_id = {s.signal_id: s for s in plain}
    for s in kept:
        original = plain_by_id[s.signal_id]  # фильтр не создаёт новых сигналов
        assert s.stop_loss == original.stop_loss
        assert s.take_profit == original.take_profit
        assert s.features["p_success"] == 0.9
        assert (s.created_at - pd.Timedelta(hours=1)).hour % 2 == 0
    assert filtered.params["ml_threshold"] == 0.5
