"""Модели данных, общие для бэктеста, paper и демо.

Все времена — UTC, datetime с таймзоной. Время свечи — время её открытия.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from bot.core.timeframes import timeframe_delta

# Bybit принимает orderLinkId длиной до 36 символов.
ORDER_LINK_ID_MAX_LEN = 36


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
PositiveFloat = Annotated[float, Field(gt=0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Side(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class EntryType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderSide(StrEnum):
    """Сторона ордера в терминах Bybit."""

    BUY = "Buy"
    SELL = "Sell"


class OrderType(StrEnum):
    MARKET = "Market"
    LIMIT = "Limit"


class OrderStatus(StrEnum):
    """Машина состояний ордера из docs/05-risk-and-execution.md."""

    NEW = "NEW"
    SENT = "SENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Candle(_Frozen):
    symbol: str
    timeframe: str
    open_time: UtcDatetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float = 0.0

    @property
    def close_time(self) -> datetime:
        return self.open_time + timeframe_delta(self.timeframe)


class Signal(_Frozen):
    """Сигнал стратегии. Без стоп-лосса не создаётся."""

    signal_id: str = Field(max_length=ORDER_LINK_ID_MAX_LEN)
    created_at: UtcDatetime
    strategy: str
    strategy_version: str
    symbol: str
    timeframe: str
    side: Side
    entry_type: EntryType = EntryType.MARKET
    entry_price: PositiveFloat
    stop_loss: PositiveFloat
    take_profit: PositiveFloat | None = None
    time_stop_bars: int | None = Field(default=None, gt=0)
    reason: str = ""
    features: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_levels(self) -> Signal:
        if self.side is Side.LONG:
            stop_ok = self.stop_loss < self.entry_price
            tp_ok = self.take_profit is None or self.take_profit > self.entry_price
        else:
            stop_ok = self.stop_loss > self.entry_price
            tp_ok = self.take_profit is None or self.take_profit < self.entry_price
        if not stop_ok:
            raise ValueError("стоп-лосс должен быть с правильной стороны от входа")
        if not tp_ok:
            raise ValueError("тейк-профит должен быть с правильной стороны от входа")
        return self

    @staticmethod
    def build_id(strategy: str, symbol: str, timeframe: str, bar_time: datetime, side: Side) -> str:
        """Детерминированный id: одинаковые входные данные -> одинаковый id.

        bar_time — время открытия свечи, на закрытии которой появился сигнал.
        Пример: trend_following, BTCUSDT, 1h, 2026-10-12 14:00 UTC, LONG
        -> 'tf-BTCUSDT-1h-20261012T1400-L' (используется как orderLinkId).
        """
        if bar_time.tzinfo is None:
            raise ValueError("bar_time должен быть с таймзоной")
        short = "".join(part[0] for part in strategy.split("_") if part)
        stamp = bar_time.astimezone(UTC).strftime("%Y%m%dT%H%M")
        signal_id = f"{short}-{symbol}-{timeframe}-{stamp}-{side.value[0]}"
        if len(signal_id) > ORDER_LINK_ID_MAX_LEN:
            raise ValueError(f"signal_id длиннее {ORDER_LINK_ID_MAX_LEN}: {signal_id}")
        return signal_id

    @classmethod
    def create(
        cls,
        *,
        strategy: str,
        strategy_version: str,
        symbol: str,
        timeframe: str,
        bar_time: datetime,
        side: Side,
        entry_price: float,
        stop_loss: float,
        take_profit: float | None = None,
        entry_type: EntryType = EntryType.MARKET,
        time_stop_bars: int | None = None,
        reason: str = "",
        features: dict[str, Any] | None = None,
    ) -> Signal:
        """Создать сигнал на закрытии свечи bar_time: id и created_at считаются сами."""
        return cls(
            signal_id=cls.build_id(strategy, symbol, timeframe, bar_time, side),
            created_at=bar_time + timeframe_delta(timeframe),
            strategy=strategy,
            strategy_version=strategy_version,
            symbol=symbol,
            timeframe=timeframe,
            side=side,
            entry_type=entry_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_stop_bars=time_stop_bars,
            reason=reason,
            features=features or {},
        )

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_to_risk(self) -> float | None:
        if self.take_profit is None:
            return None
        return abs(self.take_profit - self.entry_price) / self.risk_per_unit


class OrderRequest(_Frozen):
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: PositiveFloat
    price: PositiveFloat | None = None
    stop_loss: PositiveFloat | None = None
    take_profit: PositiveFloat | None = None
    reduce_only: bool = False
    order_link_id: str = Field(max_length=ORDER_LINK_ID_MAX_LEN)
    signal_id: str | None = None

    @model_validator(mode="after")
    def _check_price(self) -> OrderRequest:
        if self.order_type is OrderType.LIMIT and self.price is None:
            raise ValueError("лимитному ордеру нужна цена")
        return self


class Order(_Frozen):
    order_id: str | None = None
    order_link_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: float
    price: float | None = None
    status: OrderStatus
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    reduce_only: bool = False
    created_at: UtcDatetime
    updated_at: UtcDatetime


class Position(_Frozen):
    symbol: str
    side: Side
    size: PositiveFloat
    entry_price: PositiveFloat
    leverage: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    liq_price: float | None = None
    unrealised_pnl: float = 0.0
    signal_id: str | None = None
    opened_at: UtcDatetime | None = None


class Trade(_Frozen):
    """Закрытая сделка: вход и выход с издержками."""

    signal_id: str
    symbol: str
    side: Side
    qty: PositiveFloat
    entry_time: UtcDatetime
    entry_price: PositiveFloat
    exit_time: UtcDatetime
    exit_price: PositiveFloat
    fees: float = 0.0
    funding: float = 0.0  # >0 — получили, <0 — заплатили
    exit_reason: str = ""

    @property
    def gross_pnl(self) -> float:
        direction = 1.0 if self.side is Side.LONG else -1.0
        return direction * (self.exit_price - self.entry_price) * self.qty

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees + self.funding


class Balance(_Frozen):
    coin: str = "USDT"
    equity: float
    wallet_balance: float
    available: float
    updated_at: UtcDatetime
