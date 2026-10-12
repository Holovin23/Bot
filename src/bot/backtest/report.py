"""Отчёт бэктеста и команда запуска.

    uv run python -m bot.backtest.report --strategy random_entry --symbol BTCUSDT \\
        --timeframe 1h --from 2021-01-01 --to 2024-06-30

Результат: data/backtests/<время>_<стратегия>_<символ>_<таймфрейм>/
report.md (метрики и 10 сделок с пошаговым расчётом), trades.csv, equity.csv.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestResult, BacktestTrade, run_backtest
from bot.backtest.metrics import Metrics, compute_metrics
from bot.core.config import Config, RiskConfig, load_config
from bot.core.models import Side
from bot.data.storage import Storage
from bot.strategies.registry import create_strategy

SAMPLE_TRADES = 10


@dataclass(frozen=True)
class ReportContext:
    start: pd.Timestamp
    end: pd.Timestamp
    costs: CostModel
    risk: RiskConfig


def _f(value: float, digits: int = 2) -> str:
    if math.isinf(value):
        return "∞"
    return f"{value:,.{digits}f}".replace(",", " ")


def _t(ts: pd.Timestamp | None) -> str:
    return "—" if ts is None else f"{ts:%Y-%m-%d %H:%M}"


def trades_frame(result: BacktestResult) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        rows.append(
            {
                "signal_id": t.signal_id,
                "side": t.side.value,
                "qty": t.qty,
                "entry_time": t.entry_time,
                "entry_price_raw": t.entry_price_raw,
                "entry_price": t.entry_price,
                "initial_stop": t.initial_stop,
                "stop": t.stop,
                "take": t.take,
                "exit_time": t.exit_time,
                "exit_price_raw": t.exit_price_raw,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
                "fees": t.fees,
                "slippage": t.slippage_cost,
                "funding": t.funding,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "risk_amount": t.risk_amount,
                "r_multiple": t.r_multiple,
            }
        )
    return pd.DataFrame(rows)


def explain_trade(
    trade: BacktestTrade, number: int, equity_before: float, ctx: ReportContext, symbol: str
) -> list[str]:
    """Пошаговый расчёт одной сделки для ручной сверки."""
    long = trade.side is Side.LONG
    slip = ctx.costs.slippage_rate(symbol)
    stop_slip = ctx.costs.slippage_rate(symbol, stop=True)
    exit_slip = stop_slip if trade.exit_reason == "stop" else slip
    sign_in = "+" if long else "−"
    sign_out = "−" if long else "+"
    risk_money = equity_before * ctx.risk.risk_per_trade
    distance = abs(trade.entry_price - trade.initial_stop) + trade.initial_stop * stop_slip
    entry_s, exit_s = _f(trade.entry_price, 4), _f(trade.exit_price, 4)
    move = f"({exit_s} − {entry_s})" if long else f"({entry_s} − {exit_s})"
    stop0 = _f(trade.initial_stop, 4)
    take = "нет" if trade.take is None else _f(trade.take, 4)
    risk_pct = ctx.risk.risk_per_trade
    fee = ctx.costs.taker_fee
    lines = [
        f"### {number}. {trade.signal_id} — {trade.side.value}, {trade.exit_reason}",
        "",
        f"1. Сигнал на закрытии свечи {_t(trade.signal_time)} UTC: стоп {stop0}, тейк {take}.",
        f"2. Вход {_t(trade.entry_time)} UTC по open 1m-свечи {_f(trade.entry_price_raw, 4)}; "
        f"с проскальзыванием {sign_in}{slip * 100:g}%: "
        f"{_f(trade.entry_price_raw, 4)} × (1 {sign_in} {slip:g}) = {_f(trade.entry_price, 4)}.",
        f"3. Объём: капитал {_f(equity_before)} × {risk_pct:g} = {_f(risk_money)} USDT риска; "
        f"дистанция |{_f(trade.entry_price, 4)} − {_f(trade.initial_stop, 4)}| + "
        f"{_f(trade.initial_stop, 4)} × {stop_slip:g} = {_f(distance, 4)}; "
        f"{_f(risk_money)} / {_f(distance, 4)} = {risk_money / distance:.6f} → "
        f"вниз до шага лота (и не больше плеча {ctx.risk.max_leverage:g}x) = {trade.qty:g}.",
        f"4. Комиссия входа: {trade.qty:g} × {_f(trade.entry_price, 4)} × {fee:g} = "
        f"{_f(trade.entry_fee, 4)}.",
        f"5. Выход {_t(trade.exit_time)} UTC ({trade.exit_reason}"
        f"{', гэп — по open' if trade.gap_exit else ''}): цена {_f(trade.exit_price_raw, 4)}; "
        f"с проскальзыванием {sign_out}{exit_slip * 100:g}% = {_f(trade.exit_price, 4)}. "
        f"Стоп на момент выхода {_f(trade.stop, 4)}.",
        f"6. Комиссия выхода: {trade.qty:g} × {_f(trade.exit_price, 4)} × {fee:g} = "
        f"{_f(trade.exit_fee, 4)}.",
        f"7. Funding: {trade.funding_events} выплат, итого {_f(trade.funding, 4)} "
        f"({'получили' if trade.funding > 0 else 'заплатили' if trade.funding < 0 else '0'}).",
        f"8. Результат: {move} × {trade.qty:g} = {_f(trade.gross_pnl, 4)}; "
        f"− комиссии {_f(trade.fees, 4)} + funding {_f(trade.funding, 4)} = "
        f"**{_f(trade.net_pnl, 4)} USDT**, {trade.r_multiple:.2f}R "
        f"(плановый риск {_f(trade.risk_amount, 4)}).",
        "",
    ]
    return lines


def render_report(
    result: BacktestResult, metrics: Metrics, ctx: ReportContext, sample_seed: int = 0
) -> str:
    m = metrics
    params = ", ".join(f"{k}={v}" for k, v in result.params.items()) or "—"
    cost_share = "—" if m.cost_share_pct is None else f"{_f(m.cost_share_pct, 1)}%"
    buy_hold = "—" if m.buy_and_hold_pct is None else f"{_f(m.buy_and_hold_pct, 1)}%"
    lines = [
        f"# Бэктест {result.strategy} {result.strategy_version} — "
        f"{result.symbol} {result.timeframe}",
        "",
        f"- Период: {_t(ctx.start)} — {_t(ctx.end)} UTC",
        f"- Параметры: {params}",
        f"- Капитал: {_f(result.initial_capital)} USDT, "
        f"риск на сделку {ctx.risk.risk_per_trade:g}, "
        f"плечо не выше {ctx.risk.max_leverage:g}x",
        f"- Издержки: taker {ctx.costs.taker_fee:g}, проскальзывание "
        f"{ctx.costs.slippage_rate(result.symbol):g} "
        f"(стоп ×{ctx.costs.stop_slippage_multiplier:g}), "
        "funding по истории",
        f"- Пропущено сигналов: {len(result.skipped)}",
        "",
        "## Метрики",
        "",
        "| Метрика | Значение | Ориентир |",
        "|---|---|---|",
        f"| Сделок | {m.trades} | ≥ 100 |",
        f"| Win Rate | {_f(m.win_rate * 100, 1)}% | вместе с avg win/loss |",
        f"| Средний выигрыш / проигрыш | {_f(m.avg_win)} / {_f(m.avg_loss)} USDT | |",
        f"| Profit Factor | {_f(m.profit_factor)} | > 1,2 после расходов |",
        f"| Expectancy | {m.expectancy_r:.3f}R, {_f(m.expectancy_pct, 3)}%, "
        f"{_f(m.expectancy_usd)} USDT | > 0,1R |",
        f"| Доходность | {_f(m.total_return_pct, 1)}% (годовых {_f(m.cagr_pct, 1)}%) | "
        f"buy & hold {buy_hold} |",
        f"| Max Drawdown | {_f(m.max_drawdown_pct, 1)}%, самая долгая просадка "
        f"{m.max_drawdown_duration.days} дн. | |",
        f"| Sharpe / Sortino | {_f(m.sharpe)} / {_f(m.sortino)} | > 1 на OOS |",
        f"| Calmar | {_f(m.calmar)} | |",
        f"| Время в позиции | {_f(m.exposure_pct, 1)}% | |",
        f"| Комиссии / проскальзывание / funding | {_f(m.fees)} / {_f(m.slippage)} / "
        f"{_f(m.funding)} USDT | |",
        f"| Доля расходов в валовой прибыли | {cost_share} | < 30–40% |",
        "",
        "## По годам",
        "",
        "| Год | Сделок | Результат, USDT | Капитал за год |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {y.year} | {y.trades} | {_f(y.net_pnl)} | {_f(y.return_pct, 1)}% |" for y in m.by_year
    ]

    lines += ["", "## Сделки для ручной проверки", ""]
    lines += [
        "Случайные сделки с пошаговым расчётом. Пересчитайте на калькуляторе и сверьте цены "
        "открытия свечей с графиком Bybit (время UTC).",
        "",
    ]
    equity_before = {}
    capital = result.initial_capital
    for t in sorted(result.trades, key=lambda t: t.entry_time):
        equity_before[id(t)] = capital
        capital += t.net_pnl
    sample = random.Random(sample_seed).sample(
        result.trades, min(SAMPLE_TRADES, len(result.trades))
    )
    for number, t in enumerate(sorted(sample, key=lambda t: t.entry_time), start=1):
        lines += explain_trade(t, number, equity_before[id(t)], ctx, result.symbol)
    return "\n".join(lines)


def save_report(
    result: BacktestResult, metrics: Metrics, ctx: ReportContext, out_root: Path
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    folder = out_root / f"{stamp}_{result.strategy}_{result.symbol}_{result.timeframe}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.md").write_text(render_report(result, metrics, ctx), encoding="utf-8")
    trades_frame(result).to_csv(folder / "trades.csv", index=False)
    result.equity.to_csv(folder / "equity.csv", index=False)
    return folder


def _parse_params(items: list[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Параметр должен быть в виде ключ=значение: {item!r}")
        params[key.strip()] = yaml.safe_load(value)
    return params


def run_report(
    cfg: Config,
    strategy_name: str,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    params: dict[str, Any] | None = None,
) -> tuple[Path, Metrics]:
    """Бэктест на сохранённых данных и отчёт в data/backtests. Общий код для командной
    строки и для запуска из веб-интерфейса."""
    data_dir = cfg.resolve(cfg.data.dir)
    storage = Storage(data_dir)
    if symbol not in cfg.instruments:
        raise ValueError(f"Нет параметров инструмента {symbol} в config/default.yaml")
    candles = storage.read_candles(symbol, "1m", start, end)
    if candles.empty:
        raise ValueError(f"Нет 1m-свечей {symbol} за период. Сначала загрузите историю.")
    funding = storage.read_funding(symbol, start, end)
    strategy = create_strategy(strategy_name, timeframe, params)
    costs = CostModel.from_config(cfg.costs)
    result = run_backtest(
        strategy,
        symbol,
        candles,
        instrument=cfg.instruments[symbol],
        costs=costs,
        risk=cfg.risk,
        initial_capital=cfg.costs.initial_capital,
        funding=funding,
    )
    metrics = compute_metrics(result, candles["close"])
    ctx = ReportContext(start=start, end=end, costs=costs, risk=cfg.risk)
    return save_report(result, metrics, ctx, data_dir / "backtests"), metrics


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Бэктест стратегии")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", required=True)
    parser.add_argument("--param", action="append", help="параметр стратегии: ключ=значение")
    args = parser.parse_args(argv)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    print(f"Бэктест {args.strategy} {args.symbol} {args.timeframe}…", flush=True)
    try:
        folder, metrics = run_report(
            cfg, args.strategy, args.symbol, args.timeframe, start, end, _parse_params(args.param)
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(
        f"Сделок {metrics.trades}, PF {_f(metrics.profit_factor)}, "
        f"Expectancy {metrics.expectancy_r:.3f}R, доходность {_f(metrics.total_return_pct, 1)}%, "
        f"Max DD {_f(metrics.max_drawdown_pct, 1)}%"
    )
    print(f"Отчёт: {folder / 'report.md'}")


if __name__ == "__main__":
    main()
