import pandas as pd
import pytest

from bot.backtest.lookahead import find_lookahead
from bot.features.regime import Regime
from bot.strategies.base import generate_signals, iter_contexts
from bot.strategies.composite import Composite
from bot.strategies.mean_reversion import MeanReversion
from bot.strategies.trend_following import TrendFollowing

from .strategy_helpers import random_bars


def test_all_mode_equals_union_of_members() -> None:
    bars = random_bars(1500, seed=11)
    composite = Composite("1h", {"mode": "all", "members": "mean_reversion,trend_following"})
    ids = {s.signal_id for s in generate_signals(composite, "BTCUSDT", bars)}
    # Составная стратегия ждёт самую длинную историю из входящих (EMA200 у TF)
    start = bars["open_time"].iloc[composite.required_history() - 1]
    mr = generate_signals(MeanReversion("1h"), "BTCUSDT", bars)
    tf = generate_signals(TrendFollowing("1h"), "BTCUSDT", bars)
    union = {s.signal_id for s in mr + tf if s.created_at > start}
    assert ids == union
    assert ids  # на случайных данных сигналы есть


def test_regime_mode_respects_regimes() -> None:
    bars = random_bars(1500, seed=11)
    composite = Composite("1h", {"mode": "regime"})
    features = composite.prepare(bars.reset_index(drop=True))
    regime_at = dict(
        zip(bars["open_time"] + pd.Timedelta(hours=1), features["regime"], strict=True)
    )
    signals = generate_signals(composite, "BTCUSDT", bars)
    assert signals
    for s in signals:
        regime = regime_at[pd.Timestamp(s.created_at)]
        if s.strategy == "trend_following":
            assert regime in (Regime.TREND_UP, Regime.TREND_DOWN)
        else:
            assert regime == Regime.RANGE


def test_no_lookahead_and_bad_params() -> None:
    bars = random_bars(900, seed=3)
    cut = bars["open_time"].iloc[700]
    assert find_lookahead(Composite("1h", {"mode": "all"}), "BTCUSDT", bars, cut) == []
    with pytest.raises(ValueError):
        Composite("1h", {"members": "nope"})
    with pytest.raises(ValueError):
        Composite("1h", {"mode": "sometimes"})


def test_view_reads_prefixed_columns() -> None:
    bars = random_bars(300)
    composite = Composite("1h", {"members": "mean_reversion"})
    _, ctx = next(iter(iter_contexts(composite, "BTCUSDT", bars)))
    assert ctx.view("mr__").value("atr") == ctx.value("mr__atr")
    assert ctx.view("mr__").value("close") == ctx.value("close")
