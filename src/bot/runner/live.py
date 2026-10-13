"""Живой бот: свечи → стратегии → риск-менеджер → брокер (этап 7).

    uv run python -m bot.runner.live --mode paper     # виртуальные сделки
    uv run python -m bot.runner.live --mode demo      # Bybit Demo Trading

Порядок на каждой новой закрытой 1m-свече (свечи приходят из LiveFeed):
1. paper: проверка стопа и тейка по этой минуте (как в бэктесте), funding;
2. аварийная остановка: файл data/KILL или команда из интерфейса/Telegram;
3. сверка с биржей (reconcile) раз в reconcile_minutes;
4. если закрылась свеча таймфрейма стратегии: для открытой сделки — счётчик свечей,
   выход по времени (reduceOnly) или трейлинг стопа на бирже; без сделки — on_bar,
   проверки 1–13 риск-менеджера и рыночный вход со стопом и тейком на бирже.
Стратегии и риск-менеджер — тот же код, что в бэктесте.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bot.backtest.costs import CostModel
from bot.core.config import PROJECT_ROOT, Config, load_config
from bot.core.db import connect, db_path
from bot.core.journal import Journal
from bot.core.logging import get_logger, setup_logging
from bot.core.models import Order, OrderStatus, Position, Side, Signal
from bot.core.timeframes import timeframe_delta
from bot.data.resample import resample_candles
from bot.data.storage import Storage
from bot.execution.broker import (
    Broker,
    BrokerError,
    OrderRejectedError,
    close_link_id,
    close_request,
    entry_request,
    submit_with_retry,
)
from bot.execution.paper import PaperBroker
from bot.execution.reconcile import reconcile
from bot.execution.store import LiveTrade, Store
from bot.features.indicators import atr
from bot.risk.limits import OpenRisk, RiskState, evaluate, record_trade, update_state
from bot.runner.settings import Instance, LiveSettings, load_live_settings
from bot.strategies.base import MarketContext, Strategy
from bot.strategies.registry import create_strategy

log = get_logger(__name__)

KILL_FILE = "KILL"
RESUME_FILE = "RESUME"
PAPER_CASH_KEY = "paper_cash"
FUNDING_HOURS = (0, 8, 16)
MINUTE = pd.Timedelta(minutes=1)

Notify = Callable[[str], None]


@dataclass
class Running:
    instance: Instance
    strategy: Strategy


def build_strategy(instance: Instance, models_dir: Path) -> Strategy:
    strategy: Strategy = create_strategy(instance.strategy, instance.timeframe, instance.params)
    if instance.ml_filter:
        from bot.analyzer.filtered import FilteredStrategy
        from bot.analyzer.ml_filter import FilterModel

        path = models_dir / f"{instance.strategy}_{instance.symbol}_{instance.timeframe}.joblib"
        if not path.exists():
            raise SystemExit(f"Нет модели {path}. Обучите: python -m bot.research.ml_filter")
        strategy = FilteredStrategy(strategy, FilterModel.load(path))
    return strategy


def _trade_fields(trade: LiveTrade) -> dict[str, Any]:
    """Сделка для журнала: всё, кроме сырого JSON сигнала (он уже записан в signal)."""
    return {k: v for k, v in asdict(trade).items() if k != "signal_json"}


def bar_closes_at(minute_open: pd.Timestamp, timeframe: str) -> bool:
    """Закрылась ли свеча таймфрейма вместе с 1m-свечой minute_open."""
    end = minute_open + MINUTE
    step = timeframe_delta(timeframe)
    return end == end.floor(step)


class LiveBot:
    def __init__(
        self,
        cfg: Config,
        settings: LiveSettings,
        broker: Broker,
        store: Store,
        storage: Storage,
        notify: Notify = print,
        now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
        strategies: list[Running] | None = None,
        journal: Journal | None = None,
        control_dir: Path | None = None,
    ) -> None:
        """journal — подробный JSONL-журнал для разбора сделок; control_dir — папка файлов
        KILL и RESUME (по умолчанию data/; у ботов из веб-интерфейса — своя папка)."""
        self.cfg = cfg
        self.settings = settings
        self.broker = broker
        self.store = store
        self.storage = storage
        self.notify = notify
        self.now = now
        self.costs = CostModel.from_config(cfg.costs)
        models_dir = cfg.resolve(cfg.data.dir) / "models"
        self.running = strategies or [
            Running(i, build_strategy(i, models_dir)) for i in settings.instances
        ]
        self.journal = journal
        self.kill_file = (control_dir or storage.root) / KILL_FILE
        self.last_minute: dict[str, pd.Timestamp] = {}
        self.minutes_seen = 0
        self.state: RiskState | None = None
        # Свечи приходят из потока WebSocket, команды — из потока Telegram.
        self.lock = threading.RLock()
        self.offline_since: pd.Timestamp | None = None

    # --- запуск и состояние риска ----------------------------------------------------

    def start(self) -> None:
        for symbol in self.settings.symbols:
            last = self.storage.last_candle_time(symbol, "1m")
            if last is not None:
                self.last_minute[symbol] = last
            try:
                self.broker.set_leverage(symbol, self.cfg.risk.max_leverage)
            except BrokerError as exc:
                self.event("leverage", f"{symbol}: плечо не установлено: {exc}", "warning")
        if isinstance(self.broker, PaperBroker):
            self._restore_paper()
        self.sync()
        names = ", ".join(
            f"{r.strategy.name} {r.instance.symbol} {r.strategy.timeframe}" for r in self.running
        )
        self.event("start", f"бот запущен ({self.broker.name}): {names}")
        self.notify(f"Бот запущен в режиме {self.broker.name}: {names}")

    @property
    def _cash_key(self) -> str:
        return f"{PAPER_CASH_KEY}:{self.store.mode}"

    def _restore_paper(self) -> None:
        assert isinstance(self.broker, PaperBroker)
        cash = self.store.get_value(self._cash_key)
        if cash is None and self.store.mode == "paper":
            cash = self.store.get_value(PAPER_CASH_KEY)  # ключ до появления нескольких ботов
        positions = [
            Position(
                symbol=t.symbol,
                side=Side(t.side),
                size=t.qty,
                entry_price=t.entry_price or t.signal.entry_price,
                stop_loss=t.stop,
                take_profit=t.take,
                signal_id=t.signal_id,
            )
            for t in self.store.trades("open")
        ]
        self.broker.restore(float(cash) if cash else self.settings.paper_capital, positions)

    def refresh_state(self) -> RiskState:
        equity = self.broker.get_balance().equity
        today = self.now().date()
        state = self.state or self.store.get_risk_state()
        if state is None:
            state = RiskState(equity, equity, today, equity)
        before = state
        state = update_state(state, equity, today, self.cfg.limits)
        self._save_state(state)  # до kill: kill закрывает сделки и снова читает состояние
        if state.halted and not before.halted and not self.store.kill_reason():
            self.kill(state.halted)
        elif state.paused and not before.paused:
            self.event("risk", f"пауза: {state.paused}", "warning")
            self.notify(f"Пауза в торговле до следующего дня: {state.paused}")
        return state

    def record(self, net_pnl: float) -> None:
        """Итог закрытой сделки — в серию убытков (проверка 11)."""
        state = self.state or self.refresh_state()
        new = record_trade(state, net_pnl, self.cfg.limits)
        self._save_state(new)
        if new.paused and not state.paused:
            self.event("risk", f"пауза: {new.paused}", "warning")
            self.notify(f"Пауза в торговле до следующего дня: {new.paused}")

    def _save_state(self, state: RiskState) -> None:
        self.state = state
        self.store.set_risk_state(state)
        self._save_paper_cash()

    def _save_paper_cash(self) -> None:
        if isinstance(self.broker, PaperBroker):
            self.store.set_value(self._cash_key, repr(self.broker.cash))

    def event(self, kind: str, message: str, level: str = "info") -> None:
        self.store.add_event(kind, message, level)
        getattr(log, "warning" if level != "info" else "info")(kind, message=message)
        self.log("event", event=kind, level=level, message=message)

    def log(self, kind: str, **fields: Any) -> None:
        """Строка в JSONL-журнал (если он подключён)."""
        if self.journal is not None:
            self.journal.write(kind, **fields)

    # --- поток свечей -----------------------------------------------------------------

    def on_candles(self, symbol: str, _: pd.DataFrame) -> None:
        """Колбэк LiveFeed. Читаем из хранилища всё после последней обработанной минуты:
        так не теряются свечи, дозагруженные после обрыва связи."""
        with self.lock:
            self._process_new_minutes(symbol)

    def _process_new_minutes(self, symbol: str) -> None:
        last = self.last_minute.get(symbol)
        start = last + MINUTE if last is not None else self.now().floor("1min") - MINUTE * 2
        minutes = self.storage.read_candles(symbol, "1m", start=start)
        for record in minutes.to_dict(orient="records"):
            row = {str(k): v for k, v in record.items()}
            open_time = pd.Timestamp(row["open_time"])
            try:
                self.on_minute(symbol, row)
                if self.offline_since is not None:
                    self.event("connection", f"связь с биржей восстановлена ({open_time})")
                    self.notify("Связь с биржей восстановлена")
                    self.offline_since = None
            except BrokerError as exc:
                # Нет связи: минуту пропускаем (новых сделок нет — проверка 13 «API
                # отвечает»), стопы живут на бирже. Сообщаем один раз, а не каждую минуту.
                if self.offline_since is None:
                    self.offline_since = open_time
                    self.event("connection", f"нет связи с биржей: {exc}", "warning")
                    self.notify(f"Нет связи с биржей: {exc}. Стопы остаются на бирже.")
            except Exception as exc:  # одна ошибка не должна останавливать бота
                log.exception("minute_failed", symbol=symbol)
                self.event("error", f"{symbol} {open_time}: {exc}", "error")
            self.last_minute[symbol] = open_time

    def on_minute(self, symbol: str, row: dict[str, Any]) -> None:
        open_time = pd.Timestamp(row["open_time"])
        close_time = open_time + MINUTE
        if isinstance(self.broker, PaperBroker):
            self.broker.on_minute(
                symbol,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                close_time.to_pydatetime(),
            )
            if close_time.minute == 0 and close_time.hour in FUNDING_HOURS:
                self._paper_funding(symbol, close_time)
        self.check_kill_file()
        self.minutes_seen += 1
        if self.minutes_seen % self.settings.reconcile_minutes == 0:
            self.sync()
        for running in self.running:
            closed = bar_closes_at(open_time, running.strategy.timeframe)
            if running.instance.symbol == symbol and closed:
                self.on_bar_close(running, close_time)
        self._save_paper_cash()

    def _paper_funding(self, symbol: str, at: pd.Timestamp) -> None:
        assert isinstance(self.broker, PaperBroker)
        rates = self.storage.read_funding(symbol, at - pd.Timedelta(days=3), at + MINUTE)
        if rates.empty:
            return
        amount = self.broker.apply_funding(symbol, float(rates["funding_rate"].iloc[-1]))
        if amount:
            self.event("funding", f"{symbol}: funding {amount:+.4f} USDT")

    # --- сверка и закрытые сделки ----------------------------------------------------

    def sync(self) -> None:
        """Сверка с биржей; сделки, закрывшиеся с прошлой сверки, — в состояние риска."""
        before = {t.signal_id for t in self.store.active_trades()}
        report = reconcile(
            self.broker,
            self.store,
            close_unknown=self.settings.close_unknown,
            unknown_stop_pct=self.settings.unknown_stop_pct,
        )
        for warning in report.warnings:
            self.notify(f"Сверка: {warning}")
        self.refresh_state()
        for signal_id in before:
            trade = self.store.get_trade(signal_id)
            if trade is not None and trade.status == "closed":
                self.log("trade_closed", source="reconcile", trade=_trade_fields(trade))
                self.notify(
                    f"Закрыта {trade.symbol} {trade.side} ({trade.exit_reason}): "
                    f"{(trade.net_pnl or 0):+.2f} USDT"
                )
                self.record(trade.net_pnl or 0.0)

    # --- свеча стратегии ----------------------------------------------------------------

    def context(
        self, running: Running, close_time: pd.Timestamp
    ) -> tuple[MarketContext, float] | None:
        symbol, tf = running.instance.symbol, running.strategy.timeframe
        start = close_time - pd.Timedelta(days=self.settings.history_days)
        bars = resample_candles(self.storage.read_candles(symbol, "1m", start, close_time), tf)
        bars = bars.reset_index(drop=True)
        if len(bars) < running.strategy.required_history():
            self.event("data", f"{symbol} {tf}: мало истории ({len(bars)} свечей)", "warning")
            return None
        features = running.strategy.prepare(bars).reset_index(drop=True)
        ctx = MarketContext(symbol, tf, bars, features, len(bars) - 1)
        return ctx, float(atr(bars, 14).iloc[-1])

    def on_bar_close(self, running: Running, close_time: pd.Timestamp) -> None:
        if self.store.kill_reason():
            return
        built = self.context(running, close_time)
        if built is None:
            return
        ctx, atr_value = built
        trade = self._trade_for(running)
        self.log(
            "bar",
            strategy=running.strategy.name,
            symbol=running.instance.symbol,
            timeframe=running.strategy.timeframe,
            bar_open=ctx.time,
            bar=ctx.bar,
            features=ctx.feature,
            atr=atr_value,
            open_trade=trade.signal_id if trade is not None else None,
        )
        if trade is not None:
            self.manage_open(running, ctx, trade)
            return
        signals = [s for s in running.strategy.on_bar(ctx) if s.symbol == running.instance.symbol]
        for signal in signals:
            self.log("signal", signal=signal, used=signal is signals[0])
        if signals:
            self.try_open(signals[0], ctx, atr_value)

    def _trade_for(self, running: Running) -> LiveTrade | None:
        for trade in self.store.active_trades():
            if trade.symbol == running.instance.symbol and trade.strategy == running.strategy.name:
                return trade
        return None

    def manage_open(self, running: Running, ctx: MarketContext, trade: LiveTrade) -> None:
        if trade.status != "open":
            return
        bars_held = trade.bars_held + 1
        self.store.update_trade(trade.signal_id, bars_held=bars_held)
        if trade.time_stop_bars is not None and bars_held >= trade.time_stop_bars:
            self.log("time_exit", signal_id=trade.signal_id, bars_held=bars_held)
            self.close(trade, "time")
            return
        position = Position(
            symbol=trade.symbol,
            side=Side(trade.side),
            size=trade.qty,
            entry_price=trade.entry_price or trade.signal.entry_price,
            stop_loss=trade.stop,
            take_profit=trade.take,
            signal_id=trade.signal_id,
        )
        new_stop = running.strategy.update_stop(ctx, position)
        if new_stop is None:
            return
        long = trade.side == Side.LONG.value
        tighter = new_stop > trade.stop if long else new_stop < trade.stop
        if not tighter:
            return
        try:
            self.broker.set_trading_stop(trade.symbol, Side(trade.side), new_stop, trade.take)
        except BrokerError as exc:
            self.event("trail", f"{trade.signal_id}: стоп не перенесён: {exc}", "error")
            return
        self.store.update_trade(trade.signal_id, stop=float(new_stop))
        self.log("stop_moved", signal_id=trade.signal_id, old=trade.stop, new=float(new_stop))
        self.event("trail", f"{trade.signal_id}: стоп {trade.stop:g} → {new_stop:g}")

    def try_open(self, signal: Signal, ctx: MarketContext, atr_value: float) -> None:
        state = self.state or self.refresh_state()
        open_risks = [
            OpenRisk(t.signal_id, t.symbol, Side(t.side), t.risk_amount)
            for t in self.store.active_trades()
        ]
        if self.store.get_trade(signal.signal_id) is not None:
            open_risks.append(OpenRisk(signal.signal_id, signal.symbol, signal.side, 0.0))
        last_bar_open = pd.Timestamp(ctx.time)
        decision = evaluate(
            signal,
            state=state,
            open_risks=open_risks,
            atr_value=atr_value,
            last_candle_open=last_bar_open,
            now=self.now(),
            entry_price=ctx.value("close"),
            instrument=self.cfg.instruments[signal.symbol],
            costs=self.costs,
            risk=self.cfg.risk,
            limits=self.cfg.limits,
        )
        self.log(
            "risk_decision",
            signal_id=signal.signal_id,
            approved=decision.approved,
            reasons=decision.reasons,
            notes=decision.notes,
            qty=decision.qty,
            risk_amount=decision.risk_amount,
            equity=state.equity,
            open_risks=open_risks,
            atr=atr_value,
        )
        if not decision.approved:
            self.event("risk", f"{signal.signal_id} отклонён: {'; '.join(decision.reasons)}")
            return
        for note in decision.notes:
            self.event("risk", f"{signal.signal_id}: {note}")
        self.open(signal, decision.qty, decision.risk_amount)

    def open(self, signal: Signal, qty: float, risk_amount: float) -> None:
        self.store.add_trade(signal, qty, risk_amount)
        request = entry_request(signal, qty)
        self.store.order_sent(request)
        self.log("order_sent", purpose="entry", request=request)
        try:
            order = submit_with_retry(self.broker, request)
        except OrderRejectedError as exc:
            self.log("order_rejected", order_link_id=request.order_link_id, error=str(exc))
            self.store.order_failed(request.order_link_id, str(exc))
            self.store.update_trade(signal.signal_id, status="failed")
            self.event("order", f"{signal.signal_id}: биржа отклонила вход: {exc}", "error")
            self.notify(f"Ордер отклонён {signal.symbol}: {exc}")
            return
        except BrokerError as exc:
            self.log("order_unknown", order_link_id=request.order_link_id, error=str(exc))
            # Неизвестно, дошёл ли ордер: сделка остаётся pending, решит сверка.
            self.event("order", f"{signal.signal_id}: нет ответа биржи: {exc}", "error")
            self.notify(f"Нет ответа биржи по {signal.symbol}, сверка уточнит: {exc}")
            return
        self.store.order_result(order)
        self.log("order_result", order=order)
        if order.status is OrderStatus.FILLED:
            self._mark_open(signal, order)
            self.log(
                "trade_opened",
                signal_id=signal.signal_id,
                fill_price=order.avg_fill_price,
                signal_price=signal.entry_price,
                qty=qty,
                stop=signal.stop_loss,
                take=signal.take_profit,
                risk_amount=risk_amount,
            )
        self.event(
            "order",
            f"вход {signal.symbol} {signal.side.value} {qty:g} по {order.avg_fill_price or '?'}, "
            f"стоп {signal.stop_loss:g}, тейк {signal.take_profit or '—'}",
        )
        self.notify(
            f"Вход {signal.symbol} {signal.side.value} {qty:g} ({signal.strategy}), "
            f"стоп {signal.stop_loss:g}, тейк {signal.take_profit or '—'}"
        )

    def _mark_open(self, signal: Signal, order: Order) -> None:
        self.store.update_trade(
            signal.signal_id,
            status="open",
            entry_price=order.avg_fill_price,
            opened_at=order.updated_at.isoformat(),
        )

    def close(self, trade: LiveTrade, reason: str) -> None:
        """Закрытие рыночным reduceOnly-ордером (выход по времени, kill switch)."""
        attempt = 1  # новая попытка — новый orderLinkId (X1, X2, ...)
        while self.store.order_status(close_link_id(trade.signal_id, attempt)) is not None:
            attempt += 1
        request = close_request(trade.symbol, Side(trade.side), trade.qty, trade.signal_id, attempt)
        self.store.order_sent(request)
        self.log("order_sent", purpose=f"close:{reason}", request=request)
        try:
            order = submit_with_retry(self.broker, request)
        except BrokerError as exc:
            self.log("order_failed", order_link_id=request.order_link_id, error=str(exc))
            self.store.order_failed(request.order_link_id, str(exc))
            self.event("order", f"{trade.signal_id}: не закрыта ({reason}): {exc}", "error")
            self.notify(f"Не удалось закрыть {trade.symbol}: {exc}")
            return
        self.store.order_result(order)
        info = self.broker.last_exit(trade.symbol)
        self.store.update_trade(
            trade.signal_id,
            status="closed",
            exit_price=info.exit_price if info else order.avg_fill_price,
            exit_time=(info.exit_time if info else datetime.now(UTC)).isoformat(),
            exit_reason=reason,
            net_pnl=info.net_pnl if info else None,
        )
        self.log("order_result", order=order)
        closed = self.store.get_trade(trade.signal_id)
        if closed is not None:
            self.log("trade_closed", source="bot", trade=_trade_fields(closed))
        self.refresh_state()
        self.record(info.net_pnl if info else 0.0)
        pnl = f"{info.net_pnl:+.2f} USDT" if info else "итог уточнит сверка"
        self.event("order", f"{trade.signal_id}: закрыта ({reason}), {pnl}")
        self.notify(f"Закрыта {trade.symbol} ({reason}): {pnl}")

    # --- аварийная остановка ------------------------------------------------------------

    def check_kill_file(self) -> None:
        """Команды из интерфейса и ручные: файлы data/KILL и data/RESUME."""
        resume_file = self.kill_file.with_name(RESUME_FILE)
        if resume_file.exists():
            resume_file.unlink()
            self.resume()
        if self.kill_file.exists() and not self.store.kill_reason():
            text = self.kill_file.read_text(encoding="utf-8").strip()
            self.kill(text or "файл data/KILL")

    def kill(self, reason: str) -> None:
        """Отменить ордера, закрыть позиции, не открывать новых до /resume."""
        with self.lock:
            self._kill(reason)

    def _kill(self, reason: str) -> None:
        self.store.request_kill(reason)
        self.event("kill", f"аварийная остановка: {reason}", "error")
        for symbol in self.settings.symbols:
            try:
                self.broker.cancel_all(symbol)
            except BrokerError as exc:
                self.event("kill", f"{symbol}: ордера не отменены: {exc}", "error")
        trades = {t.symbol: t for t in self.store.active_trades() if t.status == "open"}
        for position in self.broker.get_positions():
            trade = trades.get(position.symbol)
            if trade is not None:
                self.close(trade, "kill")
            else:
                stamp = int(self.now().timestamp())
                link = f"kill-{position.symbol}-{stamp}"
                request = close_request(position.symbol, position.side, position.size, link)
                try:
                    submit_with_retry(self.broker, request)
                except BrokerError as exc:
                    self.event("kill", f"{position.symbol}: не закрыта: {exc}", "error")
        left = self.broker.get_positions()
        status = "все позиции закрыты" if not left else f"ОСТАЛИСЬ позиции: {len(left)}"
        self.notify(f"АВАРИЙНАЯ ОСТАНОВКА ({reason}): {status}. Возобновить: /resume")

    def resume(self) -> None:
        """Ручное снятие остановки: флаг, файл data/KILL, остановка по просадке и пауза."""
        with self.lock:
            self._resume()

    def _resume(self) -> None:
        self.store.clear_kill()
        if self.kill_file.exists():
            self.kill_file.unlink()
        if self.state is not None:
            st = self.state
            self._save_state(
                replace(st, halted=None, paused=None, loss_streak=0, peak_equity=st.equity)
            )
        self.event("kill", "торговля возобновлена вручную")
        self.notify("Торговля возобновлена")

    def status_text(self) -> str:
        with self.lock:
            return self._status_text()

    def _status_text(self) -> str:
        balance = self.broker.get_balance()
        lines = [
            f"Режим: {self.broker.name}. Капитал: {balance.equity:.2f}",
            f"Остановка: {self.store.kill_reason() or 'нет'}",
        ]
        if self.state is not None and (self.state.paused or self.state.halted):
            lines.append(f"Риск: {self.state.halted or self.state.paused}")
        for p in self.broker.get_positions():
            lines.append(
                f"{p.symbol} {p.side.value} {p.size:g} по {p.entry_price:g}, стоп {p.stop_loss}, "
                f"PnL {p.unrealised_pnl:+.2f}"
            )
        return "\n".join(lines)


ENV_LIVE = PROJECT_ROOT / ".env.live"


def confirm_live(equity: float, ask: Callable[[str], str] = input) -> bool:
    """Подтверждение запуска на реальном счёте: ввести LIVE и капитал (целые USDT)."""
    expected = f"LIVE {int(equity)}"
    print("ВНИМАНИЕ: запуск на РЕАЛЬНОМ счёте Bybit. Сделки будут с настоящими деньгами.")
    print(f"Капитал на бирже: {equity:.2f} USDT. Для запуска введите: {expected}")
    return ask("> ").strip() == expected


def load_live_account(env_file: Path = ENV_LIVE) -> Config:
    """Настройки реального счёта: ключи из .env.live и строгий профиль риска."""
    if not env_file.exists():
        raise SystemExit(f"Нет файла {env_file.name}: реальный счёт не настроен (этап 8)")
    cfg = load_config(env_file=env_file)
    if cfg.secrets.bybit_env != "live" or not cfg.secrets.allow_live:
        raise SystemExit(f"В {env_file.name} нужны BYBIT_ENV=live и ALLOW_LIVE=true")
    return cfg.for_live()


def main(argv: list[str] | None = None) -> None:
    from bot.data.bybit_public import make_public_client
    from bot.data.downloader import Downloader
    from bot.data.live_feed import LiveFeed
    from bot.execution.bybit import BybitBroker

    parser = argparse.ArgumentParser(description="Живой бот: paper, Bybit Demo или live")
    parser.add_argument(
        "--mode",
        choices=["paper", "demo", "live"],
        help="вместо mode из live.yaml; live — реальный счёт (этап 8, нужен .env.live)",
    )
    parser.add_argument("--config", type=Path, help="настройки (по умолчанию config/live.yaml)")
    parser.add_argument("--hours", type=float, default=24 * 365, help="сколько часов работать")
    args = parser.parse_args(argv)

    live = args.mode == "live"
    cfg = load_live_account() if live else load_config()
    settings = load_live_settings(args.config) if args.config else load_live_settings()
    if args.mode and not live:
        settings = settings.model_copy(update={"mode": args.mode})
    if not settings.instances:
        raise SystemExit("В config/live.yaml нет ни одной стратегии (instances)")
    setup_logging(cfg.resolve(cfg.logging.dir), cfg.logging.level, cfg.logging.retention_days)

    data_dir = cfg.resolve(cfg.data.dir)
    storage = Storage(data_dir)
    store = Store(connect(db_path(data_dir), threads=True), "live" if live else settings.mode)
    broker: Broker
    if live:
        probe = BybitBroker.live(cfg.secrets, cfg.instruments, confirmed=True)
        if not confirm_live(probe.get_balance().equity):
            raise SystemExit("Запуск на реальном счёте отменён")
        broker = probe
    elif settings.mode == "demo":
        broker = BybitBroker.demo(cfg.secrets, cfg.instruments)
    else:
        costs = CostModel.from_config(cfg.costs)
        broker = PaperBroker(costs, cfg.instruments, settings.paper_capital)

    from bot.notify.telegram import TelegramNotifier

    notifier = TelegramNotifier.from_secrets(cfg.secrets)
    account = "live" if live else settings.mode
    journal_path = cfg.resolve(cfg.logging.dir) / "journal" / f"{account}.jsonl"
    journal = Journal(journal_path, account=account)
    bot = LiveBot(cfg, settings, broker, store, storage, notify=notifier.send, journal=journal)
    downloader = Downloader(make_public_client(), storage, category=cfg.data.category)
    feed = LiveFeed(storage, downloader, settings.symbols, on_candle=bot.on_candles)
    for symbol in settings.symbols:  # сначала догрузить историю, потом стартовать
        feed.backfill(symbol)
    bot.start()
    notifier.start_commands(bot)
    print(f"Бот работает в режиме {broker.name}. Остановить: Ctrl+C")
    print("После остановки позиции остаются на бирже со стопами.")
    try:
        feed.run(args.hours)
    finally:
        notifier.stop()


if __name__ == "__main__":
    main()
