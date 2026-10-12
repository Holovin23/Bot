import pandas as pd

from bot.backtest.lookahead import find_lookahead
from bot.core.models import Side, Signal
from bot.strategies.base import BaseStrategy, MarketContext, generate_signals, iter_contexts

from .strategy_helpers import random_bars


class CloseAboveSma(BaseStrategy):
    """Честная стратегия: LONG, когда close выше SMA(5)."""

    name = "close_above_sma"
    version = "1.0.0"

    def required_history(self) -> int:
        return 5

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"sma": bars["close"].rolling(5).mean()})

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        prev_close = ctx.bars["close"].iloc[-2]
        prev_sma = ctx.features["sma"].iloc[-2]
        if ctx.bar["close"] > ctx.feature["sma"] and not prev_close > prev_sma:
            return [self.signal(ctx, Side.LONG, stop_loss=float(ctx.bar["low"]) - 1)]
        return []


class PeekingStrategy(CloseAboveSma):
    """Подглядывает: использует close СЛЕДУЮЩЕЙ свечи (shift(-1))."""

    name = "peeking"

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"sma": bars["close"].shift(-1).rolling(5).mean()})


class NormalizingStrategy(CloseAboveSma):
    """Нормализует по всему датасету — классическая утечка будущего."""

    name = "normalizing"

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        z = (bars["close"] - bars["close"].mean()) / bars["close"].std()
        return pd.DataFrame({"sma": z.rolling(5).mean() * 0 + bars["close"].mean()})


def test_context_sees_only_closed_bars() -> None:
    bars = random_bars(20)
    strategy = CloseAboveSma("1h")
    contexts = list(iter_contexts(strategy, "BTCUSDT", bars))
    assert contexts[0][0] == 4  # первые 4 свечи — разгон
    i, ctx = contexts[3]
    assert len(ctx.bars) == i + 1
    assert ctx.bar["open_time"] == bars["open_time"].iloc[i]
    assert ctx.time == bars["open_time"].iloc[i].to_pydatetime()


def test_signal_has_entry_at_close_and_id() -> None:
    signals = generate_signals(CloseAboveSma("1h"), "BTCUSDT", random_bars(200))
    assert signals
    first = signals[0]
    assert first.signal_id.startswith("cas-BTCUSDT-1h-")
    again = generate_signals(CloseAboveSma("1h"), "BTCUSDT", random_bars(200))
    assert [s.signal_id for s in again] == [s.signal_id for s in signals]


def test_honest_strategy_passes_lookahead_check() -> None:
    bars = random_bars(300)
    cut = bars["open_time"].iloc[150]
    assert find_lookahead(CloseAboveSma("1h"), "BTCUSDT", bars, cut) == []


def test_peeking_strategy_is_caught() -> None:
    bars = random_bars(300)
    cut = bars["open_time"].iloc[150]
    assert find_lookahead(PeekingStrategy("1h"), "BTCUSDT", bars, cut)


def test_full_dataset_normalization_is_caught() -> None:
    bars = random_bars(300)
    cut = bars["open_time"].iloc[150]
    assert find_lookahead(NormalizingStrategy("1h"), "BTCUSDT", bars, cut)
