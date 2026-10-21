"""Интерфейс стратегии. Один и тот же код работает в бэктесте, paper и на демо.

Стратегия видит только закрытые свечи своего таймфрейма и ничего не знает о балансе,
объёме позиции и бирже. Она говорит: «здесь вход, здесь я неправа (стоп), вот цель».
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd
from numpy.typing import NDArray

from bot.core.models import EntryType, Position, Side, Signal


class MarketContext:
    """Всё, что стратегия знает на закрытии текущей свечи.

    bars и features содержат строки только до текущей свечи включительно,
    последняя строка — текущая (только что закрытая) свеча. bar и feature —
    значения текущей свечи (словарь «колонка -> число»), value() — значение
    колонки на lag свечей назад. Будущие строки недоступны.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        all_bars: pd.DataFrame,
        all_features: pd.DataFrame,
        index: int,
        columns: dict[str, NDArray[Any]] | None = None,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.index = index
        self._all_bars = all_bars
        self._all_features = all_features
        self._columns = columns if columns is not None else _numpy_columns(all_bars, all_features)
        self._bars: pd.DataFrame | None = None
        self._features: pd.DataFrame | None = None
        self._prefix = ""

    def view(self, prefix: str) -> MarketContext:
        """Тот же момент времени, но признаки читаются с префиксом: value("atr")
        вернёт колонку prefix + "atr", если она есть. Нужно составной стратегии, чтобы
        у каждой входящей стратегии были свои признаки."""
        other = MarketContext(
            self.symbol,
            self.timeframe,
            self._all_bars,
            self._all_features,
            self.index,
            self._columns,
        )
        other._prefix = prefix
        return other

    @property
    def bars(self) -> pd.DataFrame:
        if self._bars is None:
            self._bars = self._all_bars.iloc[: self.index + 1]
        return self._bars

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            self._features = self._all_features.iloc[: self.index + 1]
        return self._features

    def value(self, column: str, lag: int = 0) -> float:
        """Значение колонки свечи или признака: lag=0 — текущая, 1 — предыдущая."""
        if lag < 0:
            raise ValueError("Будущие свечи недоступны")
        if lag > self.index:
            return float("nan")
        if self._prefix and self._prefix + column in self._columns:
            column = self._prefix + column
        return float(self._columns[column][self.index - lag])

    @property
    def bar(self) -> dict[str, float]:
        return {c: self.value(c) for c in _OHLCV if c in self._columns}

    @property
    def feature(self) -> dict[str, float]:
        return {c: self.value(c) for c in self._all_features.columns}

    @property
    def time(self) -> datetime:
        """Время открытия текущей свечи (UTC)."""
        ns = int(self._columns["open_time"][self.index])
        return datetime.fromtimestamp(ns / 1e9, tz=UTC)


_OHLCV = ("open", "high", "low", "close", "volume", "turnover")


def _numpy_columns(bars: pd.DataFrame, features: pd.DataFrame) -> dict[str, NDArray[Any]]:
    columns: dict[str, NDArray[Any]] = {
        c: bars[c].to_numpy(dtype="float64") for c in _OHLCV if c in bars
    }
    columns["open_time"] = bars["open_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    for c in features.columns:
        columns[str(c)] = features[c].to_numpy(dtype="float64")
    return columns


@dataclass(frozen=True)
class ChartSeries:
    """Как рисовать признак стратегии на графике веб-интерфейса.

    column — колонка из prepare() (или колонка свечи, например volume);
    pane — "price" рисуется поверх свечей, любое другое имя — отдельная панель под
    графиком; levels — горизонтальные уровни в панели (например, RSI 30 и 70);
    scale — множитель значения (порог объёма = средний объём × 1,5).
    """

    column: str
    label: str
    pane: str = "price"
    color: str = "#4a90e2"
    kind: str = "line"  # line | histogram
    scale: float = 1.0
    levels: tuple[float, ...] = ()
    dashed: bool = False


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

    def chart_series(self) -> list[ChartSeries]:
        """Индикаторы, с которыми работает стратегия, — для графика."""
        return []

    def backtest_only(self) -> list[str]:
        """Включённые опции, которые пока есть только в бэктесте: живой бот с ними
        не запустится, чтобы paper и демо не расходились с бэктестом."""
        return []

    def signal(
        self,
        ctx: MarketContext,
        side: Side,
        stop_loss: float,
        take_profit: float | None = None,
        time_stop_bars: int | None = None,
        reason: str = "",
        features: dict[str, Any] | None = None,
        limit_price: float | None = None,
    ) -> Signal:
        """Сигнал на закрытии текущей свечи. Без limit_price — рыночный вход, entry_price —
        close (оценка: реальный вход будет по открытию следующей свечи). С limit_price —
        лимитный ордер по этой цене, действует одну следующую свечу."""
        return Signal.create(
            strategy=self.name,
            strategy_version=self.version,
            symbol=ctx.symbol,
            timeframe=self.timeframe,
            bar_time=ctx.time,
            side=side,
            entry_price=ctx.value("close") if limit_price is None else limit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_type=EntryType.MARKET if limit_price is None else EntryType.LIMIT,
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
    columns = _numpy_columns(bars, features)
    start = max(strategy.required_history() - 1, 0)
    for i in range(start, len(bars)):
        yield i, MarketContext(symbol, strategy.timeframe, bars, features, i, columns)


def generate_signals(strategy: Strategy, symbol: str, bars: pd.DataFrame) -> list[Signal]:
    """Все сигналы стратегии на истории (без учёта позиций)."""
    signals: list[Signal] = []
    for _, ctx in iter_contexts(strategy, symbol, bars):
        signals.extend(strategy.on_bar(ctx))
    return signals
