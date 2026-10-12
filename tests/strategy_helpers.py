"""Синтетические свечи для тестов стратегий и движка."""

import numpy as np
import pandas as pd


def random_bars(count: int, timeframe_minutes: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(rng.normal(0, 0.01, count).cumsum())  # всегда > 0
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.001, 0.01, count) * close
    low = np.minimum(open_, close) - rng.uniform(0.001, 0.01, count) * close
    times = pd.date_range("2024-01-01", periods=count, freq=f"{timeframe_minutes}min", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
            "turnover": close,
        }
    )


def minute_walk(days: int = 60, seed: int = 0) -> pd.DataFrame:
    n = days * 1440
    rng = np.random.default_rng(seed)
    close = 40_000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) * 1.0003,
            "low": np.minimum(open_, close) * 0.9997,
            "close": close,
            "volume": 1.0,
            "turnover": close,
        }
    )
