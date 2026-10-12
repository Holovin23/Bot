from datetime import UTC, datetime

import pandas as pd
import pytest

from bot.backtest.engine import BacktestTrade
from bot.core.models import Side, Signal
from bot.execution.store import LiveTrade
from bot.research.live_vs_backtest import compare, render

SIG = Signal.create(
    strategy="breakout",
    strategy_version="1",
    symbol="BTCUSDT",
    timeframe="1h",
    bar_time=datetime(2026, 10, 12, 10, 0, tzinfo=UTC),
    side=Side.LONG,
    entry_price=100,
    stop_loss=95,
)


def live(signal_id: str, entry: float, pnl: float, reason: str = "stop") -> LiveTrade:
    return LiveTrade(
        signal_id=signal_id,
        mode="paper",
        strategy="breakout",
        symbol="BTCUSDT",
        timeframe="1h",
        side="LONG",
        qty=1,
        entry_price=entry,
        stop=95,
        take=None,
        time_stop_bars=None,
        bars_held=1,
        risk_amount=5,
        status="closed",
        opened_at=None,
        exit_price=95,
        exit_time=None,
        exit_reason=reason,
        net_pnl=pnl,
        signal_json=SIG.model_dump_json(),
    )


def bt(signal_id: str, entry: float, exit_: float, reason: str = "stop") -> BacktestTrade:
    t = pd.Timestamp("2026-10-12 11:00", tz="UTC")
    return BacktestTrade(
        signal_id=signal_id,
        symbol="BTCUSDT",
        side=Side.LONG,
        qty=1,
        signal_time=t,
        entry_time=t,
        entry_price_raw=entry,
        entry_price=entry,
        initial_stop=95,
        stop=95,
        take=None,
        risk_amount=5,
        entry_fee=0,
        exit_time=t,
        exit_price_raw=exit_,
        exit_price=exit_,
        exit_reason=reason,
    )


def test_compare_by_signal_id() -> None:
    result = compare(
        [live("a", 100.1, -5.2), live("b", 50, 1.0), live("c", 10, 0, "take")],
        [bt("a", 100.0, 95.0), bt("c", 10, 11, "stop"), bt("d", 20, 21)],
    )
    assert [m.signal_id for m in result.matched] == ["a", "c"]
    assert result.only_live == ["b"] and result.only_backtest == ["d"]
    # вход a: (100,1 − 100) / 100 × 10 000 = 10 б.п. хуже
    assert result.matched[0].entry_diff_bps == pytest.approx(10)
    # результат: a −5,2 против −5; c 0 против +1 → разница −1,2
    assert result.pnl_diff == pytest.approx(-1.2)
    assert result.match_share == pytest.approx(0.5)
    assert result.same_exit_share == pytest.approx(0.5)
    text = "\n".join(render("breakout", result))
    assert "Совпало сигналов: 2 (50%)" in text


def test_pending_live_trades_are_ignored() -> None:
    trade = live("a", 100, 0)
    open_trade = LiveTrade(**{**trade.__dict__, "status": "open"})
    assert compare([open_trade], []).only_live == []
