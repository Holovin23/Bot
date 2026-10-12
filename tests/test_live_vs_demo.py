import pytest

from bot.execution.store import LiveTrade
from bot.research.live_vs_demo import compare_execution, partial_fills, render, worse_bps


def trade(signal_id: str, side: str, entry: float, exit_: float | None) -> LiveTrade:
    return LiveTrade(
        signal_id=signal_id,
        mode="x",
        strategy="breakout",
        symbol="BTCUSDT",
        timeframe="1h",
        side=side,
        qty=0.1,
        entry_price=entry,
        stop=1,
        take=None,
        time_stop_bars=None,
        bars_held=0,
        risk_amount=1,
        status="closed",
        opened_at=None,
        exit_price=exit_,
        exit_time=None,
        exit_reason="stop",
        net_pnl=0,
        signal_json="{}",
    )


def test_worse_bps_by_side() -> None:
    # покупка: реальный 100,05 против демо 100 → на 5 б.п. дороже = хуже
    assert worse_bps(100.05, 100, buy=True) == pytest.approx(5)
    # продажа: реальный 99,98 против 100 → на 2 б.п. дешевле = хуже
    assert worse_bps(99.98, 100, buy=False) == pytest.approx(2)


def test_compare_execution() -> None:
    live = [
        trade("a", "LONG", 100.05, 98.98),
        trade("b", "SHORT", 50, None),
        trade("c", "LONG", 1, 1),
    ]
    demo = [
        trade("a", "LONG", 100.0, 99.0),
        trade("b", "SHORT", 50.01, None),
        trade("d", "LONG", 1, 1),
    ]
    orders = [
        {"qty": 0.1, "filled_qty": 0.05},
        {"qty": 0.1, "filled_qty": 0.1},
        {"qty": 0.1, "filled_qty": 0},
    ]
    report = compare_execution(live, demo, orders, orders[1:])
    a, b = report.pairs
    assert a.entry_bps == pytest.approx(5)
    # выход лонга = продажа: 98,98 против 99 → ≈2,02 б.п. хуже
    assert a.exit_bps == pytest.approx(2.0202, abs=1e-3)
    # вход шорта = продажа: 50 против 50,01 → ≈2 б.п. хуже
    assert b.entry_bps == pytest.approx(1.9996, abs=1e-3) and b.exit_bps is None
    assert report.only_live == ["c"] and report.only_demo == ["d"]
    assert (report.partial_live, report.orders_live, report.partial_demo) == (1, 3, 0)
    assert report.mean_entry_bps == pytest.approx((5 + 1.9996) / 2, abs=1e-3)
    assert "Общих сделок: 2" in render(report)
    assert partial_fills([]) == 0
