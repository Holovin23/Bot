from bot.research.ml_filter import filter_is_useful, random_removal_percentile
from bot.research.robustness import TradeStats


def stats(trades: int, exp: float, dd: float) -> TradeStats:
    return TradeStats(
        trades=trades, profit_factor=1.0, expectancy_r=exp, net_pnl=0.0, max_drawdown_pct=dd
    )


def test_filter_is_useful_hand_examples() -> None:
    base = stats(100, 0.05, 10.0)
    assert filter_is_useful(base, stats(60, 0.10, 12.0))[0]  # Expectancy лучше
    assert filter_is_useful(base, stats(60, 0.01, 8.0))[0]  # просадка лучше
    useful, why = filter_is_useful(base, stats(40, 0.30, 5.0))  # 40% сделок — мало
    assert not useful
    assert "40%" in why
    assert not filter_is_useful(base, stats(80, 0.02, 11.0))[0]
    assert not filter_is_useful(stats(0, 0, 0), stats(0, 0, 0))[0]


def test_random_filter_is_not_useful() -> None:
    base = stats(100, 0.05, 10.0)
    useful, why = filter_is_useful(base, stats(60, 0.10, 12.0), random_percentile=0.7)
    assert not useful
    assert "70%" in why


def test_random_removal_percentile_hand_example() -> None:
    # Базовые R: 1, −1, 1, −1. Фильтр оставил оба плюса: средний 1.0. Случайная пара
    # даёт 1.0 только в 1 случае из 6, иначе меньше → перцентиль ≈ 5/6
    p = random_removal_percentile([1, -1, 1, -1], [1, 1], runs=3000, seed=1)
    assert abs(p - 5 / 6) < 0.03
    assert random_removal_percentile([], [1.0]) == 0.0
