import itertools

import pandas as pd

from bot.backtest.walk_forward import (
    backtest_window,
    make_windows,
    run_walk_forward,
    select_variant,
)
from bot.core.config import load_config

from .strategy_helpers import minute_walk


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def test_windows_12_3_3() -> None:
    windows = make_windows(ts("2021-01-01"), ts("2025-07-01"))
    # Первое окно проверки 2022-01 — 2022-04, последнее заканчивается 2025-07-01
    assert windows[0].train_start == ts("2021-01-01")
    assert windows[0].test_start == ts("2022-01-01")
    assert windows[0].test_end == ts("2022-04-01")
    assert windows[-1].test_end == ts("2025-07-01")
    assert len(windows) == 14  # (54 − 12) / 3
    for a, b in itertools.pairwise(windows):
        assert a.test_end == b.test_start  # проверки идут подряд и не перекрываются


def test_select_variant_prefers_plateau() -> None:
    table = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "profit_factor": [1.0, 3.0, 1.0, 1.4],
            "trades": [50, 50, 50, 50],
        }
    )
    # Плато: a=1 → 2.0, a=2 → 1.67, a=3 → 1.8, a=4 → 1.2. Пик a=2 стоит среди слабых соседей
    assert select_variant(table, ["a"]) == {"a": 1}
    thin = table.assign(trades=5)
    assert select_variant(thin, ["a"]) is None


def test_window_trades_stay_inside_window() -> None:
    cfg = load_config()
    candles = minute_walk(days=90)
    start, end = ts("2024-02-01"), ts("2024-03-01")
    trades = backtest_window("random_entry", "1h", "BTCUSDT", candles, cfg, {"seed": 3}, start, end)
    assert trades
    assert all(start <= t.signal_time < end for t in trades)
    assert all(t.exit_time is not None and t.exit_time < end for t in trades)


def test_walk_forward_on_synthetic() -> None:
    cfg = load_config()
    candles = minute_walk(days=122)
    windows = make_windows(ts("2024-01-01"), ts("2024-05-01"), 2, 1, 1)
    assert len(windows) == 2
    result = run_walk_forward(
        "random_entry",
        "1h",
        "BTCUSDT",
        candles,
        cfg,
        windows,
        grid={"stop_atr": [1.0, 1.5, 2.0]},
        base_params={"seed": 1, "probability": 0.2},
    )
    assert len(result.windows) == 2
    assert result.stats is not None
    assert result.stats.trades == sum(result.windows["test_trades"])
    assert result.windows["params"].iloc[0].startswith("stop_atr=")


def test_warmup_covers_strategy_history() -> None:
    from bot.backtest.walk_forward import WARMUP, warmup_for
    from bot.strategies.registry import create_strategy

    # mean_reversion 1h: истории нужно меньше 30 дней — остаётся WARMUP
    assert warmup_for(create_strategy("mean_reversion", "1h")) == WARMUP
    # trend_following 4h: 201 свеча × 1,5 × 4 ч = 1206 ч = 50,25 дня
    assert warmup_for(create_strategy("trend_following", "4h")) == pd.Timedelta(hours=1206)
    # дневной импульс: 56 свечей × 1,5 = 84 дня
    assert warmup_for(create_strategy("time_series_momentum", "1d")) == pd.Timedelta(days=84)
