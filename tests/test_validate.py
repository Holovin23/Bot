from bot.research.robustness import MonteCarlo, TradeStats
from bot.research.validate import oos_checks, render_validation


def stats(trades: int, pf: float, exp: float, dd: float) -> TradeStats:
    return TradeStats(
        trades=trades, profit_factor=pf, expectancy_r=exp, net_pnl=0.0, max_drawdown_pct=dd
    )


def test_oos_checks_hand_example() -> None:
    is_stats = stats(200, 1.4, 0.2, 8.0)
    good = oos_checks(is_stats, stats(60, 1.15, 0.05, 12.0))  # 12 ≤ 1.5 × 8
    assert all(c.passed for c in good)
    bad = oos_checks(is_stats, stats(20, 1.1, -0.01, 12.5))
    assert [c.passed for c in bad] == [False, False, False, False]


def test_render_validation() -> None:
    mc = MonteCarlo(runs=1000, original_dd_pct=5, median_dd_pct=6, p95_dd_pct=9, worst_dd_pct=14)
    s = stats(60, 1.2, 0.1, 5.0)
    text = render_validation("x", [("in-sample", s), ("validation", s)], oos_checks(s, s), mc)
    assert "| validation | 60 | 1.20 | 0.100 |" in text
    assert "✓ PF > 1.1" in text
    assert "95% — 9.0%" in text
