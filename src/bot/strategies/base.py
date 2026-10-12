"""Интерфейс стратегии. Один и тот же код работает в бэктесте, paper и на демо.

Стратегия видит только закрытые свечи своего таймфрейма и ничего не знает о балансе,
объёме позиции и бирже. Она говорит: «здесь вход, здесь я неправа (стоп), вот цель».
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import pandas as pd

from bot.core.models import EntryType, Position, Side, Signal


@dataclass(frozen=True)
class MarketContext:
    """Всё, что стратегия знает на закрытии текущей свечи.

    bars и features содержат строки только до текущей свечи включительно,
    последняя строка — текущая (только что закрытая) свеча.
    """

    symbol: str
    timeframe: str
    bars: pd.DataFrame
    features: pd.DataFrame

    @property
    def bar(self) -> pd.Series:
        return self.bars.iloc[-1]

    @property
    def feature(self) -> pd.Series:
        return self.features.iloc[-1]

    @property
    def time(self) -> datetime:
        """Время открытия текущей свечи (UTC)."""
        ts: pd.Timestamp = self.bars["open_time"].iloc[-1]
        return ts.to_pydatetime()


class Strategy(Protocol):
    name: str
    version: str
    timeframe: str
    params: dict[str, Any]

    def required_history(self) -> int:
        """Сколько закрытых свечей нужно, прежде чем стратегия может дать сигнал."""
        ...

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Числовые признаки (индикаторы) для всех свечей. Значение в строке i зависит
        только от строк 0..i — это проверяет bot.backtest.lookahead."""
        ...

    def on_bar(self, ctx: MarketContext) -> list[Signal]: ...

    def update_stop(self, ctx: MarketContext, position: Position) -> float | None:
        """Новый уровень стопа для открытой позиции (трейлинг) или None."""
        ...


class BaseStrategy:
    """Удобная основа: значения по умолчанию и построение сигнала."""

    name = "base"
    version = "0.0.0"

    def __init__(self, timeframe: str, params: dict[str, Any] | None = None) -> None:
        self.timeframe = timeframe
        self.params: dict[str, Any] = dict(params or {})

    def required_history(self) -> int:
        return 1

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=bars.index)

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        return []

    def update_stop(self, ctx: MarketContext, position: Position) -> float | None:
        return None

    def signal(
        self,
        ctx: MarketContext,
        side: Side,
        stop_loss: float,
        take_profit: float | None = None,
        time_stop_bars: int | None = None,
        reason: str = "",
        features: dict[str, Any] | None = None,
    ) -> Signal:
        """Рыночный сигнал на закрытии текущей свечи; entry_price — её close
        (оценка: реальный вход будет по открытию следующей свечи)."""
        return Signal.create(
            strategy=self.name,
            strategy_version=self.version,
            symbol=ctx.symbol,
            timeframe=self.timeframe,
            bar_time=ctx.time,
            side=side,
            entry_price=float(ctx.bar["close"]),
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_type=EntryType.MARKET,
            time_stop_bars=time_stop_bars,
            reason=reason,
            features=features or {},
        )


def iter_contexts(
    strategy: Strategy, symbol: str, bars: pd.DataFrame
) -> Iterator[tuple[int, MarketContext]]:
    """Контексты по закрытым свечам: (номер свечи, контекст)."""
    bars = bars.reset_index(drop=True)
    features = strategy.prepare(bars).reset_index(drop=True)
    if len(features) != len(bars):
        raise ValueError("prepare() должен вернуть по строке на каждую свечу")
    start = max(strategy.required_history() - 1, 0)
    for i in range(start, len(bars)):
        yield (
            i,
            MarketContext(
                symbol=symbol,
                timeframe=strategy.timeframe,
                bars=bars.iloc[: i + 1],
                features=features.iloc[: i + 1],
            ),
        )


def generate_signals(strategy: Strategy, symbol: str, bars: pd.DataFrame) -> list[Signal]:
    """Все сигналы стратегии на истории (без учёта позиций)."""
    signals: list[Signal] = []
    for _, ctx in iter_contexts(strategy, symbol, bars):
        signals.extend(strategy.on_bar(ctx))
    return signals
