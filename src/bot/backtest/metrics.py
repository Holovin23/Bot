"""Метрики бэктеста (docs/04-backtesting-guide.md, раздел 6)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from bot.backtest.engine import BacktestResult, BacktestTrade

DAYS_PER_YEAR = 365  # крипта торгуется каждый день


@dataclass(frozen=True)
class YearStats:
    year: int
    trades: int
    net_pnl: float
    return_pct: float  # изменение капитала за год, %


@dataclass(frozen=True)
class Metrics:
    trades: int
    win_rate: float  # доля прибыльных, 0..1
    avg_win: float  # USDT
    avg_loss: float  # USDT, положительное число
    profit_factor: float  # сумма прибылей / сумма убытков
    expectancy_usd: float
    expectancy_r: float
    expectancy_pct: float  # средний результат сделки в % от капитала перед ней
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float  # положительное число
    max_drawdown_duration: pd.Timedelta
    sharpe: float
    sortino: float
    calmar: float
    exposure_pct: float
    fees: float
    slippage: float
    funding: float  # >0 — получили, <0 — заплатили
    cost_share_pct: float | None  # издержки / валовая прибыль до издержек
    buy_and_hold_pct: float | None
    by_year: list[YearStats] = field(default_factory=list)


def _raw_pnl(trade: BacktestTrade) -> float:
    """Результат по ценам свечей, до проскальзывания, комиссий и funding."""
    return trade.direction * (trade.exit_price_raw - trade.entry_price_raw) * trade.qty


def max_drawdown(equity: pd.Series, times: pd.Series) -> tuple[float, pd.Timedelta]:
    """Максимальная просадка (доля от пика) и самая долгая просадка по времени."""
    if equity.empty:
        return 0.0, pd.Timedelta(0)
    values = equity.to_numpy(dtype="float64")
    stamps = list(times)
    peak = values[0]
    peak_time = stamps[0]
    worst = 0.0
    longest = pd.Timedelta(0)
    for value, stamp in zip(values, stamps, strict=True):
        if value >= peak:
            peak, peak_time = value, stamp
        else:
            worst = max(worst, (peak - value) / peak)
            longest = max(longest, stamp - peak_time)
    return worst, longest


def daily_returns(equity: pd.DataFrame) -> pd.Series:
    daily = equity.set_index("time")["equity"].resample("1D").last().dropna()
    returns: pd.Series = daily.pct_change().dropna()
    return returns


def compute_metrics(result: BacktestResult, closes: pd.Series | None = None) -> Metrics:
    """closes — цены закрытия за тот же период для buy & hold (необязательно)."""
    trades = result.trades
    pnl = [t.net_pnl for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [-p for p in pnl if p <= 0]
    n = len(trades)

    capital = result.initial_capital
    pct_per_trade = []
    for t in sorted(trades, key=lambda t: t.entry_time):
        pct_per_trade.append(t.net_pnl / capital * 100)
        capital += t.net_pnl

    equity = result.equity
    dd, dd_duration = max_drawdown(equity["equity"], equity["time"])
    total_return = result.final_equity / result.initial_capital - 1

    days = 0.0
    if len(equity) > 1:
        days = (equity["time"].iloc[-1] - equity["time"].iloc[0]).total_seconds() / 86_400
    growth = result.final_equity / result.initial_capital
    cagr = growth ** (DAYS_PER_YEAR / days) - 1 if days > 0 and growth > 0 else 0.0

    returns = daily_returns(equity) if len(equity) else pd.Series(dtype="float64")
    sharpe = sortino = 0.0
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(DAYS_PER_YEAR)
    downside = math.sqrt((returns.clip(upper=0) ** 2).mean()) if len(returns) else 0.0
    if downside > 0:
        sortino = returns.mean() / downside * math.sqrt(DAYS_PER_YEAR)

    fees = sum(t.fees for t in trades)
    slippage = sum(t.slippage_cost for t in trades)
    funding = sum(t.funding for t in trades)
    raw_total = sum(_raw_pnl(t) for t in trades)
    costs = fees + slippage + max(0.0, -funding)
    cost_share = costs / raw_total * 100 if raw_total > 0 else None

    buy_hold = None
    if closes is not None and len(closes) > 1:
        buy_hold = (float(closes.iloc[-1]) / float(closes.iloc[0]) - 1) * 100

    return Metrics(
        trades=n,
        win_rate=len(wins) / n if n else 0.0,
        avg_win=sum(wins) / len(wins) if wins else 0.0,
        avg_loss=sum(losses) / len(losses) if losses else 0.0,
        profit_factor=(sum(wins) / sum(losses)) if sum(losses) > 0 else math.inf,
        expectancy_usd=sum(pnl) / n if n else 0.0,
        expectancy_r=sum(t.r_multiple for t in trades) / n if n else 0.0,
        expectancy_pct=sum(pct_per_trade) / n if n else 0.0,
        total_return_pct=total_return * 100,
        cagr_pct=cagr * 100,
        max_drawdown_pct=dd * 100,
        max_drawdown_duration=dd_duration,
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=cagr / dd if dd > 0 else 0.0,
        exposure_pct=float(equity["in_position"].mean() * 100) if len(equity) else 0.0,
        fees=fees,
        slippage=slippage,
        funding=funding,
        cost_share_pct=cost_share,
        buy_and_hold_pct=buy_hold,
        by_year=_by_year(result),
    )


def _by_year(result: BacktestResult) -> list[YearStats]:
    equity = result.equity
    if equity.empty:
        return []
    stats = []
    start_value = result.initial_capital
    for year, part in equity.groupby(equity["time"].dt.year):
        end_value = float(part["equity"].iloc[-1])
        year_trades = [
            t for t in result.trades if t.exit_time is not None and t.exit_time.year == year
        ]
        stats.append(
            YearStats(
                year=int(year),
                trades=len(year_trades),
                net_pnl=sum(t.net_pnl for t in year_trades),
                return_pct=(end_value / start_value - 1) * 100,
            )
        )
        start_value = end_value
    return stats
