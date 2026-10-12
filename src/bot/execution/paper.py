"""PaperBroker: виртуальное исполнение по живым ценам (этап 7, режим paper).

Правила исполнения — те же, что в бэктесте (docs/tasks/stage-2.md):
- рыночный ордер исполняется по последней известной цене (close последней закрытой
  1m-свечи ≈ open следующей) с проскальзыванием и комиссией taker;
- стоп и тейк проверяются по каждой новой 1m-свече функцией fills.check_exit:
  гэп — исполнение по open, стоп и тейк в одной свече — считается стоп;
- проскальзывание на стопе — с множителем stop_slippage_multiplier;
- funding начисляется в моменты выплат (runner вызывает apply_funding).
Сеть не используется. Состояние в памяти: после перезапуска runner сверяет его с базой.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from bot.backtest.costs import CostModel, entry_fill, exit_fill, funding_payment
from bot.backtest.fills import ExitReason, check_exit
from bot.core.config import InstrumentConfig
from bot.core.models import (
    Balance,
    Order,
    OrderRequest,
    OrderSide,
    OrderStatus,
    Position,
    Side,
)
from bot.execution.broker import DuplicateOrderError, ExitInfo, OrderRejectedError


@dataclass
class _PaperPosition:
    symbol: str
    side: Side
    qty: float
    entry_price: float
    stop: float | None
    take: float | None
    signal_id: str | None
    opened_at: datetime
    entry_fee: float
    funding: float = 0.0


class PaperBroker:
    name = "paper"

    def __init__(
        self,
        costs: CostModel,
        instruments: dict[str, InstrumentConfig],
        initial_capital: float,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.costs = costs
        self.instruments = instruments
        self.cash = initial_capital
        self.now = now
        self.prices: dict[str, float] = {}
        self.positions: dict[str, _PaperPosition] = {}
        self.orders: dict[str, Order] = {}
        self.exits: dict[str, ExitInfo] = {}
        self.leverage: dict[str, float] = {}

    # --- рынок -------------------------------------------------------------------

    def on_minute(
        self, symbol: str, open_: float, high: float, low: float, close: float, time: datetime
    ) -> ExitInfo | None:
        """Новая закрытая 1m-свеча: проверить стоп и тейк, запомнить цену."""
        exit_info = None
        pos = self.positions.get(symbol)
        if pos is not None and pos.stop is not None:
            hit = check_exit(pos.side, pos.stop, pos.take, open_, high, low)
            if hit is not None:
                exit_info = self._close(
                    symbol, hit.price, hit.reason.value, stop=hit.reason is ExitReason.STOP, at=time
                )
        self.prices[symbol] = close
        return exit_info

    def apply_funding(self, symbol: str, rate: float) -> float:
        pos = self.positions.get(symbol)
        price = self.prices.get(symbol)
        if pos is None or price is None:
            return 0.0
        amount = funding_payment(pos.side, pos.qty, price, rate)
        pos.funding += amount
        self.cash += amount
        return amount

    # --- интерфейс Broker ----------------------------------------------------------

    def place_order(self, request: OrderRequest) -> Order:
        if request.order_link_id in self.orders:
            raise DuplicateOrderError(f"orderLinkId {request.order_link_id} уже есть")
        instrument = self.instruments.get(request.symbol)
        if instrument is not None and request.qty < instrument.min_qty:
            raise OrderRejectedError(f"объём {request.qty} меньше минимального")
        price = self.prices.get(request.symbol)
        if price is None:
            raise OrderRejectedError(f"нет цены {request.symbol}")
        pos = self.positions.get(request.symbol)
        buy = request.side is OrderSide.BUY
        if request.reduce_only:
            if pos is None or (pos.side is Side.LONG) == buy:
                raise OrderRejectedError("reduceOnly: нет позиции для закрытия")
            if request.qty < pos.qty:
                raise OrderRejectedError("частичное закрытие в paper не поддерживается")
            qty = pos.qty
            info = self._close(request.symbol, price, "manual", stop=False, at=self.now())
            fill = info.exit_price
        else:
            if pos is not None:
                raise OrderRejectedError(f"по {request.symbol} уже есть позиция")
            side = Side.LONG if buy else Side.SHORT
            qty = request.qty
            fill = entry_fill(price, side, self.costs.slippage_rate(request.symbol))
            fee = self.costs.fee(qty * fill)
            self.cash -= fee
            self.positions[request.symbol] = _PaperPosition(
                symbol=request.symbol,
                side=side,
                qty=qty,
                entry_price=fill,
                stop=request.stop_loss,
                take=request.take_profit,
                signal_id=request.signal_id,
                opened_at=self.now(),
                entry_fee=fee,
            )
        now = self.now()
        order = Order(
            order_id=f"paper-{len(self.orders) + 1}",
            order_link_id=request.order_link_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            qty=qty,
            status=OrderStatus.FILLED,
            filled_qty=qty,
            avg_fill_price=fill,
            reduce_only=request.reduce_only,
            created_at=now,
            updated_at=now,
        )
        self.orders[request.order_link_id] = order
        return order

    def find_order(self, symbol: str, order_link_id: str) -> Order | None:
        return self.orders.get(order_link_id)

    def cancel_all(self, symbol: str) -> None:
        """В paper нет висящих ордеров: все рыночные и исполняются сразу."""

    def get_positions(self) -> list[Position]:
        result = []
        for pos in self.positions.values():
            price = self.prices.get(pos.symbol, pos.entry_price)
            direction = 1.0 if pos.side is Side.LONG else -1.0
            result.append(
                Position(
                    symbol=pos.symbol,
                    side=pos.side,
                    size=pos.qty,
                    entry_price=pos.entry_price,
                    leverage=self.leverage.get(pos.symbol, 1.0),
                    stop_loss=pos.stop,
                    take_profit=pos.take,
                    unrealised_pnl=direction * (price - pos.entry_price) * pos.qty,
                    signal_id=pos.signal_id,
                    opened_at=pos.opened_at,
                )
            )
        return result

    def get_balance(self) -> Balance:
        unrealised = sum(p.unrealised_pnl for p in self.get_positions())
        return Balance(
            equity=self.cash + unrealised,
            wallet_balance=self.cash,
            available=self.cash,
            updated_at=self.now(),
        )

    def set_trading_stop(
        self, symbol: str, side: Side, stop_loss: float, take_profit: float | None = None
    ) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            raise OrderRejectedError(f"нет позиции {symbol}")
        pos.stop = stop_loss
        if take_profit is not None:
            pos.take = take_profit

    def set_leverage(self, symbol: str, leverage: float) -> None:
        self.leverage[symbol] = leverage

    def last_exit(self, symbol: str) -> ExitInfo | None:
        return self.exits.get(symbol)

    # --- внутреннее --------------------------------------------------------------

    def _close(self, symbol: str, raw: float, reason: str, stop: bool, at: datetime) -> ExitInfo:
        pos = self.positions.pop(symbol)
        rate = self.costs.slippage_rate(symbol, stop=stop)
        fill = exit_fill(raw, pos.side, rate)
        fee = self.costs.fee(pos.qty * fill)
        direction = 1.0 if pos.side is Side.LONG else -1.0
        gross = direction * (fill - pos.entry_price) * pos.qty
        self.cash += gross - fee
        info = ExitInfo(
            symbol=symbol,
            exit_price=fill,
            exit_time=at,
            net_pnl=gross - fee - pos.entry_fee + pos.funding,
            reason=reason,
        )
        self.exits[symbol] = info
        return info

    def restore(self, cash: float, positions: list[Position]) -> None:
        """После перезапуска: капитал и открытые позиции из базы (комиссия входа уже
        вычтена из капитала, поэтому entry_fee = 0)."""
        self.cash = cash
        for p in positions:
            self.positions[p.symbol] = _PaperPosition(
                symbol=p.symbol,
                side=p.side,
                qty=p.size,
                entry_price=p.entry_price,
                stop=p.stop_loss,
                take=p.take_profit,
                signal_id=p.signal_id,
                opened_at=p.opened_at or self.now(),
                entry_fee=0.0,
            )
            self.prices.setdefault(p.symbol, p.entry_price)
