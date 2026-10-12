import math

import pandas as pd
import pytest

from bot.backtest.engine import BacktestResult, BacktestTrade
from bot.backtest.metrics import compute_metrics, max_drawdown
from bot.core.models import Side

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def trade(day: int, entry: float, exit_: float, side: Side = Side.LONG) -> BacktestTrade:
    """1 единица актива, без комиссий и проскальзывания, риск 10 USDT."""
    t = BacktestTrade(
        signal_id=f"t{day}",
        symbol="TEST",
        side=side,
        qty=1.0,
        signal_time=T0 + pd.Timedelta(days=day),
        entry_time=T0 + pd.Timedelta(days=day),
        entry_price_raw=entry,
        entry_price=entry,
        initial_stop=entry - 10 if side is Side.LONG else entry + 10,
        stop=entry - 10,
        take=None,
        risk_amount=10.0,
        entry_fee=0.0,
    )
    t.exit_time = T0 + pd.Timedelta(days=day, hours=12)
    t.exit_price_raw = exit_
    t.exit_price = exit_
    t.exit_reason = "take"
    return t


def make_result() -> BacktestResult:
    # Сделки: +20, −10, +30, −10 (в USDT); капитал 1 000
    trades = [
        trade(0, 100, 120),
        trade(1, 100, 90),
        trade(2, 130, 100, Side.SHORT),
        trade(3, 100, 90),
    ]
    # Кривая капитала по дням: 1000 -> 1020 -> 1010 -> 1040 -> 1030
    equity = pd.DataFrame(
        {
            "time": [T0 + pd.Timedelta(days=d) for d in range(5)],
            "equity": [1000.0, 1020.0, 1010.0, 1040.0, 1030.0],
            "in_position": [True, True, False, True, False],
        }
    )
    return BacktestResult(
        strategy="test",
        strategy_version="1",
        params={},
        symbol="TEST",
        timeframe="1h",
        initial_capital=1000.0,
        trades=trades,
        equity=equity,
    )


def test_trade_stats_by_hand() -> None:
    m = compute_metrics(make_result())
    assert m.trades == 4
    assert m.win_rate == 0.5  # 2 из 4
    assert m.avg_win == 25  # (20 + 30) / 2
    assert m.avg_loss == 10  # (10 + 10) / 2
    assert m.profit_factor == 2.5  # 50 / 20
    assert m.expectancy_usd == 7.5  # 30 / 4
    assert m.expectancy_r == 0.75  # (2 − 1 + 3 − 1) / 4
    # % от капитала перед сделкой: 20/1000, −10/1020, 30/1010, −10/1040
    expected_pct = (2.0 - 10 / 10.2 + 30 / 10.1 - 10 / 10.4) / 4
    assert m.expectancy_pct == pytest.approx(expected_pct)


def test_equity_stats_by_hand() -> None:
    m = compute_metrics(make_result())
    assert m.total_return_pct == pytest.approx(3.0)  # 1030 / 1000
    # Просадки: 1020 -> 1010 (0.98%), 1040 -> 1030 (0.96%): максимум 10 / 1020
    assert m.max_drawdown_pct == pytest.approx(10 / 1020 * 100)
    assert m.max_drawdown_duration == pd.Timedelta(days=1)
    assert m.exposure_pct == 60  # 3 из 5 отметок
    # 4 дня, рост 3%: CAGR = 1.03^(365/4) − 1
    assert m.cagr_pct == pytest.approx((1.03 ** (365 / 4) - 1) * 100)
    assert m.calmar == pytest.approx(m.cagr_pct / 100 / (10 / 1020))


def test_sharpe_by_hand() -> None:
    m = compute_metrics(make_result())
    returns = pd.Series([1020 / 1000, 1010 / 1020, 1040 / 1010, 1030 / 1040]) - 1
    expected = returns.mean() / returns.std(ddof=1) * math.sqrt(365)
    assert m.sharpe == pytest.approx(expected)
    downside = math.sqrt((returns.clip(upper=0) ** 2).mean())
    assert m.sortino == pytest.approx(returns.mean() / downside * math.sqrt(365))


def test_costs_and_buy_and_hold() -> None:
    result = make_result()
    t = result.trades[0]
    # Издержки первой сделки: комиссии 1 + 1, проскальзывание 0.5 на входе (100.5 вместо 100)
    # и 0.5 на выходе (119.5 вместо 120), funding −1
    t.entry_price = 100.5
    t.exit_price = 119.5
    t.entry_fee = 1.0
    t.exit_fee = 1.0
    t.funding = -1.0
    m = compute_metrics(result, closes=pd.Series([100.0, 90.0, 110.0]))
    assert m.fees == 2
    assert m.slippage == pytest.approx(1.0)
    assert m.funding == -1
    # Валовая прибыль до издержек: 20 − 10 + 30 − 10 = 30; издержки 2 + 1 + 1 = 4 -> 13.3%
    assert m.cost_share_pct == pytest.approx(4 / 30 * 100)
    assert m.buy_and_hold_pct == pytest.approx(10.0)


def test_by_year() -> None:
    result = make_result()
    m = compute_metrics(result)
    assert [(y.year, y.trades) for y in m.by_year] == [(2024, 4)]
    assert m.by_year[0].return_pct == pytest.approx(3.0)


def test_max_drawdown_duration_until_recovery() -> None:
    times = pd.Series([T0 + pd.Timedelta(days=d) for d in range(5)])
    dd, duration = max_drawdown(pd.Series([100.0, 80.0, 90.0, 101.0, 95.0]), times)
    assert dd == pytest.approx(0.2)
    assert duration == pd.Timedelta(days=2)  # пик в день 0, восстановление в день 3


def test_empty_result() -> None:
    result = make_result()
    result.trades = []
    m = compute_metrics(result)
    assert m.trades == 0
    assert m.profit_factor == math.inf
