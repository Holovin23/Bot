"""Движок на синтетике: каждая цифра посчитана вручную в комментариях.

Общие условия: символ TEST, 1m-свечи с ценой 100 (open=high=low=close), стратегия на 5m.
Проскальзывание 0.1% (стоп-ордер 0.2%), комиссия taker 0.05%, капитал 10 000, риск 0.5% = 50.
Сигнал на свече 00:00–00:05 -> вход по open 1m-свечи 00:05.
"""

from typing import Any

import pandas as pd
import pytest

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestResult, run_backtest
from bot.core.config import InstrumentConfig, RiskConfig
from bot.core.models import Position, Side, Signal
from bot.strategies.base import BaseStrategy, MarketContext

T0 = pd.Timestamp("2024-03-01 00:00", tz="UTC")
COSTS = CostModel(taker_fee=0.0005, slippage={"TEST": 0.001}, stop_slippage_multiplier=2.0)
INSTRUMENT = InstrumentConfig(tick_size=0.01, qty_step=0.001, min_qty=0.001)
RISK = RiskConfig(risk_per_trade=0.005, max_leverage=3.0)

# Вход LONG по 100: цена с проскальзыванием 100 × 1.001 = 100.1.
# Стоп 98: оценка проскальзывания стопа 98 × 0.002 = 0.196.
# Дистанция 100.1 − 98 + 0.196 = 2.296; объём 50 / 2.296 = 21.7770 -> 21.777.
QTY = 21.777
ENTRY = 100.1
ENTRY_FEE = QTY * ENTRY * 0.0005  # 2 179.8777 × 0.0005 = 1.0899


Ohlc = tuple[float, float, float, float]


def minutes(count: int = 60, changes: dict[int, Ohlc] | None = None) -> pd.DataFrame:
    rows = []
    for m in range(count):
        o, h, low, c = (changes or {}).get(m, (100.0, 100.0, 100.0, 100.0))
        rows.append((T0 + pd.Timedelta(minutes=m), o, h, low, c, 1.0, 100.0))
    return pd.DataFrame(
        rows, columns=["open_time", "open", "high", "low", "close", "volume", "turnover"]
    )


class Scripted(BaseStrategy):
    """Даёт один сигнал на свече T0 (5m) и, если задано, двигает стоп."""

    name = "scripted"
    version = "1.0.0"

    def __init__(self, signal: dict[str, Any], trail_to: float | None = None) -> None:
        super().__init__("5m")
        self.spec = signal
        self.trail_to = trail_to

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        if pd.Timestamp(ctx.time) != T0:
            return []
        return [self.signal(ctx, **self.spec)]

    def update_stop(self, ctx: MarketContext, position: Position) -> float | None:
        return self.trail_to


def run(
    strategy: BaseStrategy, candles: pd.DataFrame, funding: pd.DataFrame | None = None
) -> BacktestResult:
    return run_backtest(
        strategy,
        "TEST",
        candles,
        instrument=INSTRUMENT,
        costs=COSTS,
        risk=RISK,
        initial_capital=10_000,
        funding=funding,
    )


LONG = {"side": Side.LONG, "stop_loss": 98.0, "take_profit": 104.0, "time_stop_bars": 10}


def test_take_profit_long() -> None:
    # 00:12: high 105 задевает тейк 104 -> выход 104 × 0.999 = 103.896
    result = run(Scripted(LONG), minutes(changes={12: (100, 105, 100, 104)}))
    (trade,) = result.trades
    assert trade.entry_time == T0 + pd.Timedelta(minutes=5)
    assert trade.qty == QTY
    assert trade.entry_price == pytest.approx(ENTRY)
    assert trade.exit_reason == "take"
    assert trade.exit_time == T0 + pd.Timedelta(minutes=12)
    assert trade.exit_price == pytest.approx(103.896)
    # Валовой: (103.896 − 100.1) × 21.777 = 82.6655
    assert trade.gross_pnl == pytest.approx(82.665492)
    # Комиссии: 1.0899 + 21.777 × 103.896 × 0.0005 = 1.0899 + 1.1313 = 2.2212
    assert trade.fees == pytest.approx(ENTRY_FEE + QTY * 103.896 * 0.0005)
    assert trade.net_pnl == pytest.approx(82.665492 - 2.221212, abs=1e-4)
    assert result.final_equity == pytest.approx(10_000 + trade.net_pnl)
    # R: прибыль / плановый риск 21.777 × 2.296 = 50.0
    assert trade.risk_amount == pytest.approx(49.999992)
    assert trade.r_multiple == pytest.approx(trade.net_pnl / 49.999992)


def test_stop_with_gap_long() -> None:
    # 00:08 открывается на 97 — ниже стопа 98: исполнение по 97 × (1 − 0.002) = 96.806
    result = run(Scripted(LONG), minutes(changes={8: (97, 97, 96, 96.5)}))
    (trade,) = result.trades
    assert trade.exit_reason == "stop"
    assert trade.gap_exit
    assert trade.exit_price == pytest.approx(96.806)
    # (96.806 − 100.1) × 21.777 = −71.7334
    assert trade.gross_pnl == pytest.approx(-71.733438)
    assert trade.r_multiple < -1  # гэп сделал убыток больше планового риска


def test_stop_and_take_in_one_candle_counts_as_stop() -> None:
    result = run(Scripted(LONG), minutes(changes={9: (100, 105, 97, 101)}))
    (trade,) = result.trades
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(98 * 0.998)  # 97.804


def test_time_stop_and_funding() -> None:
    # time_stop 2: вход на открытии свечи 00:05, свечи 00:05 и 00:10 закрыты ->
    # выход по open 00:15 = 100 × 0.999 = 99.9.
    # Funding в 00:10 по ставке 0.1%: лонг платит 21.777 × 100 × 0.001 = 2.1777.
    # Funding в 00:05 (момент входа) и в 00:20 (после выхода) не считаются.
    funding = pd.DataFrame(
        {
            "time": [T0 + pd.Timedelta(minutes=m) for m in (5, 10, 20)],
            "funding_rate": [0.01, 0.001, 0.01],
        }
    )
    spec = {**LONG, "time_stop_bars": 2}
    result = run(Scripted(spec), minutes(), funding)
    (trade,) = result.trades
    assert trade.exit_reason == "time"
    assert trade.exit_time == T0 + pd.Timedelta(minutes=15)
    assert trade.bars_held == 2
    assert trade.exit_price == pytest.approx(99.9)
    assert trade.funding_events == 1
    assert trade.funding == pytest.approx(-2.1777)
    # (99.9 − 100.1) × 21.777 = −4.3554; комиссии 1.0899 + 21.777 × 99.9 × 0.0005 = 2.1777
    assert trade.net_pnl == pytest.approx(-4.3554 - (ENTRY_FEE + QTY * 99.9 * 0.0005) - 2.1777)
    assert result.final_equity == pytest.approx(10_000 + trade.net_pnl)


def test_short_take_profit() -> None:
    # SHORT по 100: вход 100 × 0.999 = 99.9; стоп 102, тейк 96.
    # Дистанция 102 − 99.9 + 102 × 0.002 = 2.304; объём 50 / 2.304 = 21.7013 -> 21.701
    # 00:07: low 95 -> тейк 96, выкуп 96 × 1.001 = 96.096;
    # валовой (99.9 − 96.096) × 21.701 = 82.5506
    spec = {"side": Side.SHORT, "stop_loss": 102.0, "take_profit": 96.0}
    result = run(Scripted(spec), minutes(changes={7: (100, 100, 95, 96)}))
    (trade,) = result.trades
    assert trade.qty == 21.701
    assert trade.entry_price == pytest.approx(99.9)
    assert trade.exit_price == pytest.approx(96.096)
    assert trade.gross_pnl == pytest.approx(82.550604)


def test_trailing_stop_moves_only_towards_profit() -> None:
    # После закрытия свечи 00:05 стратегия ставит стоп 99.5; в 00:13 low 99.4 -> стоп 99.5 × 0.998
    result = run(Scripted(LONG, trail_to=99.5), minutes(changes={13: (100, 100, 99.4, 99.6)}))
    (trade,) = result.trades
    assert trade.initial_stop == 98
    assert trade.stop == 99.5
    assert trade.exit_price == pytest.approx(99.5 * 0.998)

    # Стоп «назад» (97 < 98) игнорируется
    loose = run(Scripted(LONG, trail_to=97.0), minutes(changes={13: (100, 100, 97.5, 99)}))
    assert loose.trades[0].stop == 98
    assert loose.trades[0].exit_price == pytest.approx(98 * 0.998)


def test_entry_skipped_when_open_beyond_stop() -> None:
    # Свеча входа 00:05 открывается на 97.5 — уже ниже стопа 98: сделки нет
    result = run(Scripted(LONG), minutes(changes={5: (97.5, 98, 97, 97.5)}))
    assert result.trades == []
    assert result.skipped[0].reason.startswith("цена входа уже за стопом")
    assert result.final_equity == 10_000


def test_open_position_closed_at_end_of_data() -> None:
    result = run(Scripted({**LONG, "time_stop_bars": None}), minutes(count=30))
    (trade,) = result.trades
    assert trade.exit_reason == "end_of_data"
    assert trade.exit_price == pytest.approx(99.9)


def test_equity_curve_marks_open_position() -> None:
    # Цена 102 c 00:06 и далее: на закрытии свечи 00:05–00:10 нереализованная прибыль
    # (102 − 100.1) × 21.777 = 41.3763 минус уже уплаченная комиссия входа 1.0899
    changes = {m: (102.0, 102.0, 102.0, 102.0) for m in range(6, 30)}
    result = run(Scripted({**LONG, "time_stop_bars": None}), minutes(count=30, changes=changes))
    row = result.equity[result.equity["time"] == T0 + pd.Timedelta(minutes=10)].iloc[0]
    assert row["in_position"]
    assert row["equity"] == pytest.approx(10_000 - ENTRY_FEE + 41.37630)


# --- лимитный вход ---------------------------------------------------------------------
# Комиссия maker по умолчанию 0.02%. Лимит LONG 99.5, стоп 98: проскальзывания на входе
# нет, оценка проскальзывания стопа 98 × 0.002 = 0.196.
# Дистанция 99.5 − 98 + 0.196 = 1.696; объём 50 / 1.696 = 29.4811 -> 29.481.
LIMIT_QTY = 29.481
LIMIT_LONG = {"side": Side.LONG, "stop_loss": 98.0, "take_profit": 104.0, "limit_price": 99.5}


def test_limit_entry_fills_when_price_goes_through() -> None:
    # Свеча входа 00:05–00:10. В 00:07 low 99.4 < 99.5 — лимит исполнен по 99.5.
    # В той же минуте high 104.5 выше тейка, но порядок цен внутри минуты неизвестен —
    # тейк не засчитан; дальше цена 100, позиция закрывается в конце данных.
    candles = minutes(20, {7: (100.0, 104.5, 99.4, 99.6)})
    (trade,) = run(Scripted(LIMIT_LONG), candles).trades
    assert trade.entry_time == T0 + pd.Timedelta(minutes=7)
    assert trade.entry_price == trade.entry_price_raw == 99.5
    assert trade.qty == pytest.approx(LIMIT_QTY)
    # 29.481 × 99.5 = 2 933.3595; × 0.0002 = 0.58667
    assert trade.entry_fee == pytest.approx(0.58667, abs=1e-5)
    assert trade.exit_reason == "end_of_data"


def test_limit_entry_cancelled_when_only_touched() -> None:
    # low ровно 99.5 — касание без прохода: в очереди стакана ордер мог не исполниться
    candles = minutes(20, {7: (100.0, 100.0, 99.5, 99.6)})
    result = run(Scripted(LIMIT_LONG), candles)
    assert result.trades == []
    assert result.skipped[0].reason == "лимитный ордер не исполнился за свечу"


def test_limit_entry_opened_below_limit_is_taker_but_not_worse() -> None:
    # Лимит 100.5, свеча открылась по 100: исполнение сразу как рыночное —
    # 100 × 1.001 = 100.1 (не хуже 100.5), комиссия taker.
    spec = {**LIMIT_LONG, "limit_price": 100.5}
    (trade,) = run(Scripted(spec), minutes(20)).trades
    assert trade.entry_time == T0 + pd.Timedelta(minutes=5)
    assert trade.entry_price == pytest.approx(ENTRY)
    assert trade.qty == pytest.approx(QTY)
    assert trade.entry_fee == pytest.approx(ENTRY_FEE)


def test_limit_at_open_price_waits_in_queue() -> None:
    # Лимит 100 = цене открытия: ордер не пересекает стакан, ждёт прохода ниже 100.
    spec = {**LIMIT_LONG, "limit_price": 100.0}
    assert run(Scripted(spec), minutes(20)).trades == []
    # В 00:08 low 99.9 — исполнение по 100 с комиссией maker.
    # Дистанция 100 − 98 + 0.196 = 2.196; объём 50 / 2.196 = 22.7686 -> 22.768
    (trade,) = run(Scripted(spec), minutes(20, {8: (100.0, 100.0, 99.9, 100.0)})).trades
    assert trade.entry_time == T0 + pd.Timedelta(minutes=8)
    assert trade.entry_price == 100.0
    assert trade.qty == pytest.approx(22.768)
    assert trade.entry_fee == pytest.approx(22.768 * 100 * 0.0002)
