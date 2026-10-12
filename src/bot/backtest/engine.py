"""Событийный бэктест-движок для одной стратегии на одном символе.

Порядок на каждой свече таймфрейма стратегии (правила — docs/tasks/stage-2.md):
1. На открытии свечи: исполнить отложенный выход (по времени) или вход по сигналу
   прошлой свечи — рыночным ордером по open первой 1m-свечи, с проскальзыванием.
2. Внутри свечи по 1m: funding в моменты выплат, затем проверка стопа и тейка.
3. На закрытии свечи: отметка капитала; для открытой позиции — счётчик свечей,
   выход по времени, трейлинг-стоп; без позиции — вызов стратегии.
Стратегия видит только закрытые свечи; исполнение всегда позже решения.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.backtest.costs import CostModel, entry_fill, exit_fill, funding_payment
from bot.backtest.fills import ExitReason, check_exit
from bot.core.config import InstrumentConfig, RiskConfig
from bot.core.models import Position, Side, Signal
from bot.core.timeframes import timeframe_delta
from bot.data.resample import resample_candles
from bot.risk.sizing import position_size
from bot.strategies.base import MarketContext, Strategy, iter_contexts


@dataclass
class BacktestTrade:
    signal_id: str
    symbol: str
    side: Side
    qty: float
    signal_time: pd.Timestamp  # закрытие сигнальной свечи
    entry_time: pd.Timestamp
    entry_price_raw: float  # open свечи входа
    entry_price: float  # с проскальзыванием
    initial_stop: float
    stop: float  # последний стоп (после трейлинга)
    take: float | None
    risk_amount: float  # плановый риск в USDT с проскальзыванием стопа
    entry_fee: float
    time_stop_bars: int | None = None
    exit_time: pd.Timestamp | None = None
    exit_price_raw: float = 0.0
    exit_price: float = 0.0
    exit_fee: float = 0.0
    exit_reason: str = ""
    funding: float = 0.0  # >0 — получили, <0 — заплатили
    funding_events: int = 0
    bars_held: int = 0
    gap_exit: bool = False

    @property
    def direction(self) -> float:
        return 1.0 if self.side is Side.LONG else -1.0

    @property
    def gross_pnl(self) -> float:
        """Результат по ценам исполнения (проскальзывание уже внутри)."""
        return self.direction * (self.exit_price - self.entry_price) * self.qty

    @property
    def fees(self) -> float:
        return self.entry_fee + self.exit_fee

    @property
    def slippage_cost(self) -> float:
        entry = abs(self.entry_price - self.entry_price_raw) * self.qty
        exit_ = abs(self.exit_price - self.exit_price_raw) * self.qty
        return entry + exit_

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees + self.funding

    @property
    def r_multiple(self) -> float:
        return self.net_pnl / self.risk_amount if self.risk_amount else 0.0


@dataclass
class SkippedSignal:
    signal_id: str
    reason: str


@dataclass
class BacktestResult:
    strategy: str
    strategy_version: str
    params: dict[str, object]
    symbol: str
    timeframe: str
    initial_capital: float
    trades: list[BacktestTrade]
    equity: pd.DataFrame  # time (закрытие свечи), equity, in_position
    skipped: list[SkippedSignal] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        if self.equity.empty:
            return self.initial_capital
        return float(self.equity["equity"].iloc[-1])


class _Market:
    """1m-свечи и funding в виде массивов numpy для быстрого прохода."""

    def __init__(self, candles_1m: pd.DataFrame, funding: pd.DataFrame | None) -> None:
        df = candles_1m.sort_values("open_time").reset_index(drop=True)
        self.times = df["open_time"].to_numpy(dtype="datetime64[ns]")
        self.open = df["open"].to_numpy(dtype="float64")
        self.high = df["high"].to_numpy(dtype="float64")
        self.low = df["low"].to_numpy(dtype="float64")
        self.close = df["close"].to_numpy(dtype="float64")
        if funding is None or funding.empty:
            self.f_times = np.array([], dtype="datetime64[ns]")
            self.f_rates = np.array([], dtype="float64")
        else:
            f = funding.sort_values("time")
            self.f_times = f["time"].to_numpy(dtype="datetime64[ns]")
            self.f_rates = f["funding_rate"].to_numpy(dtype="float64")

    def index_at(self, ts: pd.Timestamp) -> int:
        """Первая 1m-свеча с open_time >= ts."""
        return int(np.searchsorted(self.times, np.datetime64(ts.tz_convert(None), "ns")))

    def time(self, k: int) -> pd.Timestamp:
        return pd.Timestamp(self.times[k]).tz_localize("UTC")


def run_backtest(
    strategy: Strategy,
    symbol: str,
    candles_1m: pd.DataFrame,
    *,
    instrument: InstrumentConfig,
    costs: CostModel,
    risk: RiskConfig,
    initial_capital: float = 10_000.0,
    funding: pd.DataFrame | None = None,
) -> BacktestResult:
    market = _Market(candles_1m, funding)
    bars = resample_candles(candles_1m, strategy.timeframe)
    step = timeframe_delta(strategy.timeframe)

    cash = initial_capital
    trades: list[BacktestTrade] = []
    skipped: list[SkippedSignal] = []
    equity_rows: list[tuple[pd.Timestamp, float, bool]] = []
    position: BacktestTrade | None = None
    pending_entry: Signal | None = None
    pending_exit: ExitReason | None = None
    f_ptr = 0

    def close_position(k_time: pd.Timestamp, raw: float, reason: ExitReason, stop: bool) -> None:
        nonlocal cash, position
        assert position is not None
        rate = costs.slippage_rate(symbol, stop=stop)
        position.exit_time = k_time
        position.exit_price_raw = raw
        position.exit_price = exit_fill(raw, position.side, rate)
        position.exit_fee = costs.fee(position.qty * position.exit_price)
        position.exit_reason = reason.value
        cash += position.gross_pnl - position.exit_fee
        trades.append(position)
        position = None

    def apply_funding_until(k: int) -> None:
        """Выплаты funding с временем <= времени свечи k (позиция открыта раньше)."""
        nonlocal cash, f_ptr
        while f_ptr < len(market.f_times) and market.f_times[f_ptr] <= market.times[k]:
            if position is not None:
                amount = funding_payment(
                    position.side, position.qty, market.open[k], float(market.f_rates[f_ptr])
                )
                position.funding += amount
                position.funding_events += 1
                cash += amount
            f_ptr += 1

    def try_enter(signal: Signal, k: int) -> None:
        nonlocal cash, position, f_ptr
        raw = float(market.open[k])
        fill = entry_fill(raw, signal.side, costs.slippage_rate(symbol))
        long = signal.side is Side.LONG
        beyond_stop = fill <= signal.stop_loss if long else fill >= signal.stop_loss
        tp = signal.take_profit
        beyond_take = tp is not None and (fill >= tp if long else fill <= tp)
        if beyond_stop or beyond_take:
            skipped.append(SkippedSignal(signal.signal_id, "цена входа уже за стопом или тейком"))
            return
        size = position_size(
            equity=cash,
            risk_fraction=risk.risk_per_trade,
            entry_price=fill,
            stop_loss=signal.stop_loss,
            slippage_per_unit=signal.stop_loss * costs.slippage_rate(symbol, stop=True),
            qty_step=instrument.qty_step,
            min_qty=instrument.min_qty,
            max_leverage=risk.max_leverage,
        )
        if size.qty <= 0:
            skipped.append(SkippedSignal(signal.signal_id, size.reason))
            return
        entry_time = market.time(k)
        fee = costs.fee(size.qty * fill)
        cash -= fee
        position = BacktestTrade(
            signal_id=signal.signal_id,
            symbol=symbol,
            side=signal.side,
            qty=size.qty,
            signal_time=pd.Timestamp(signal.created_at),
            entry_time=entry_time,
            entry_price_raw=raw,
            entry_price=fill,
            initial_stop=signal.stop_loss,
            stop=signal.stop_loss,
            take=tp,
            risk_amount=size.risk_amount,
            entry_fee=fee,
            time_stop_bars=signal.time_stop_bars,
        )
        # Выплата ровно в момент входа позицию не затрагивает.
        while f_ptr < len(market.f_times) and market.f_times[f_ptr] <= market.times[k]:
            f_ptr += 1

    for i, ctx in iter_contexts(strategy, symbol, bars):
        bar_open = pd.Timestamp(bars["open_time"].iloc[i])
        bar_close = bar_open + step
        k_start = market.index_at(bar_open)
        k_end = market.index_at(bar_close)

        # 1. Открытие свечи: отложенный выход или вход.
        if k_start < k_end:
            if pending_exit is not None and position is not None:
                apply_funding_until(k_start)
                close_position(
                    market.time(k_start), float(market.open[k_start]), pending_exit, False
                )
            elif pending_entry is not None:
                try_enter(pending_entry, k_start)
        pending_entry = None
        pending_exit = None

        # 2. Внутри свечи: funding, стоп и тейк по 1m.
        if position is not None:
            for k in range(k_start, k_end):
                apply_funding_until(k)
                hit = check_exit(
                    position.side,
                    position.stop,
                    position.take,
                    float(market.open[k]),
                    float(market.high[k]),
                    float(market.low[k]),
                )
                if hit is not None:
                    position.gap_exit = hit.gap
                    close_position(
                        market.time(k), hit.price, hit.reason, stop=hit.reason is ExitReason.STOP
                    )
                    break

        # 3. Закрытие свечи.
        close_price = float(ctx.bar["close"])
        if position is not None:
            position.bars_held += 1
            time_stop = position.time_stop_bars
            if time_stop is not None and position.bars_held >= time_stop:
                pending_exit = ExitReason.TIME
            else:
                _trail(strategy, ctx, position)
        else:
            signals = [s for s in strategy.on_bar(ctx) if s.symbol == symbol]
            if signals:
                pending_entry = signals[0]
                for extra in signals[1:]:
                    skipped.append(SkippedSignal(extra.signal_id, "второй сигнал на той же свече"))

        unrealized = 0.0
        if position is not None:
            unrealized = position.direction * (close_price - position.entry_price) * position.qty
        equity_rows.append((bar_close, cash + unrealized, position is not None))

    if position is not None and len(market.close):
        last = len(market.close) - 1
        close_position(market.time(last), float(market.close[last]), ExitReason.END, False)
        if equity_rows:
            time, _, _ = equity_rows[-1]
            equity_rows[-1] = (time, cash, False)

    equity = pd.DataFrame(equity_rows, columns=["time", "equity", "in_position"])
    return BacktestResult(
        strategy=strategy.name,
        strategy_version=strategy.version,
        params=dict(strategy.params),
        symbol=symbol,
        timeframe=strategy.timeframe,
        initial_capital=initial_capital,
        trades=trades,
        equity=equity,
        skipped=skipped,
    )


def _trail(strategy: Strategy, ctx: MarketContext, trade: BacktestTrade) -> None:
    position = Position(
        symbol=trade.symbol,
        side=trade.side,
        size=trade.qty,
        entry_price=trade.entry_price,
        stop_loss=trade.stop,
        take_profit=trade.take,
        signal_id=trade.signal_id,
        opened_at=trade.entry_time.to_pydatetime(),
    )
    new_stop = strategy.update_stop(ctx, position)
    if new_stop is None:
        return
    tighter = new_stop > trade.stop if trade.side is Side.LONG else new_stop < trade.stop
    if tighter:
        trade.stop = float(new_stop)
