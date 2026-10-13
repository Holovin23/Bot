"""Несколько paper-ботов внутри процесса веб-интерфейса.

Каждый бот — отдельный виртуальный счёт со своим капиталом, сделками, состоянием риска
и аварийной остановкой: в data/bot.sqlite его записи хранятся с режимом
«paper:<id>», настройки — в data/bots/<id>/bot.json, подробный журнал — в
logs/journal/paper-<id>.jsonl. Стратегии, риск-менеджер и paper-брокер — тот же код,
что у `python -m bot.runner.live --mode paper` и в бэктесте.

Свечи для всех ботов приходят из одного живого потока (LiveFeed) по всем нужным
символам. Если закрыть веб-интерфейс, боты останавливаются; при следующем запуске
интерфейса включённые боты стартуют снова и продолжают со своими позициями.
Здесь только paper: демо- и реальный счёт запускаются командой в терминале.
"""

from __future__ import annotations

import json
import math
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from pydantic import BaseModel, Field

from bot.app.bot_status import bot_status
from bot.backtest.costs import CostModel
from bot.core.config import Config
from bot.core.db import connect, db_path
from bot.core.journal import Journal
from bot.core.logging import get_logger
from bot.core.timeframes import timeframe_minutes
from bot.data.storage import Storage
from bot.execution.paper import PaperBroker
from bot.execution.store import Store
from bot.runner.live import LiveBot, Notify
from bot.runner.settings import Instance, LiveSettings
from bot.strategies.registry import create_strategy

log = get_logger(__name__)

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MIN_HISTORY_DAYS = 30


class BotSpec(BaseModel):
    id: str
    strategy: str
    symbol: str
    timeframe: str = "1h"
    params: dict[str, Any] = Field(default_factory=dict)
    capital: float = Field(default=10_000, gt=0)
    enabled: bool = True
    created: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def mode(self) -> str:
        return f"paper:{self.id}"


class Feed(Protocol):
    def run(self, hours: float, stop: threading.Event | None = None) -> None: ...


# (символы, колбэк новой свечи) -> поток свечей
FeedFactory = Callable[[list[str], Callable[[str, pd.DataFrame], None]], Feed]
# (символ, с какого, по какое) -> загрузить недостающую историю
History = Callable[[str, pd.Timestamp, pd.Timestamp], int]


@dataclass
class _Running:
    spec: BotSpec
    bot: LiveBot
    close: Callable[[], None]


def history_days(strategy_name: str, timeframe: str, params: dict[str, Any]) -> int:
    """Сколько дней 1m-истории нужно боту: запас ×1,5 к required_history стратегии."""
    strategy = create_strategy(strategy_name, timeframe, params)
    minutes = strategy.required_history() * timeframe_minutes(timeframe) * 1.5
    return int(min(365, max(MIN_HISTORY_DAYS, math.ceil(minutes / 1440) + 1)))


class BotHost:
    def __init__(
        self,
        data_dir: Path,
        storage: Storage,
        load_cfg: Callable[[], Config],
        journal_dir: Path,
        feed_factory: FeedFactory | None = None,
        history: History | None = None,
        notify: Notify = print,
        now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
        background: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.root = data_dir / "bots"
        self.storage = storage
        self.load_cfg = load_cfg
        self.journal_dir = journal_dir
        self.feed_factory = feed_factory
        self.history = history
        self.notify = notify
        self.now = now
        self.background = background
        self._lock = threading.RLock()
        self._running: dict[str, _Running] = {}
        self._state: dict[str, str] = {}  # starting | running | error
        self._errors: dict[str, str] = {}
        self._feed_symbols: list[str] = []
        self._feed_stop: threading.Event | None = None
        self._feed_thread: threading.Thread | None = None

    # --- настройки ботов -----------------------------------------------------------

    def _spec_path(self, bot_id: str) -> Path:
        if not ID_RE.match(bot_id):
            raise ValueError(f"Недопустимый id бота: {bot_id!r}")
        return self.root / bot_id / "bot.json"

    def specs(self) -> list[BotSpec]:
        if not self.root.is_dir():
            return []
        out = []
        for path in sorted(self.root.glob("*/bot.json")):
            try:
                out.append(BotSpec.model_validate_json(path.read_text(encoding="utf-8")))
            except ValueError:
                log.warning("bot_spec_broken", path=str(path))
        return sorted(out, key=lambda s: s.created)

    def get(self, bot_id: str) -> BotSpec | None:
        path = self._spec_path(bot_id)
        if not path.is_file():
            return None
        return BotSpec.model_validate_json(path.read_text(encoding="utf-8"))

    def _save(self, spec: BotSpec) -> None:
        path = self._spec_path(spec.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(spec.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _new_id(self, strategy: str, symbol: str, timeframe: str) -> str:
        base = f"{strategy}-{symbol}-{timeframe}".lower()
        taken = {s.id for s in self.specs()}
        n = 1
        while f"{base}-{n}" in taken:
            n += 1
        return f"{base}-{n}"

    def create(
        self,
        strategy: str,
        symbol: str,
        timeframe: str,
        params: dict[str, Any] | None = None,
        capital: float = 10_000,
        start: bool = True,
    ) -> BotSpec:
        cfg = self.load_cfg()
        if symbol not in cfg.instruments:
            raise ValueError(f"Нет параметров инструмента {symbol}: добавьте монету в «Данных»")
        create_strategy(strategy, timeframe, params)  # проверка имени и параметров
        spec = BotSpec(
            id=self._new_id(strategy, symbol, timeframe),
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            params=dict(params or {}),
            capital=capital,
            enabled=start,
        )
        self._save(spec)
        if start:
            self.start(spec.id)
        return spec

    # --- запуск и остановка ---------------------------------------------------------

    def autostart(self) -> None:
        """Запустить ботов, которые работали, когда интерфейс закрыли."""
        for spec in self.specs():
            if spec.enabled:
                self.start(spec.id)

    def start(self, bot_id: str) -> None:
        spec = self.get(bot_id)
        if spec is None:
            raise KeyError(bot_id)
        with self._lock:
            if self._state.get(bot_id) in ("starting", "running"):
                return
            self._state[bot_id] = "starting"
            self._errors.pop(bot_id, None)
        if not spec.enabled:
            self._save(spec.model_copy(update={"enabled": True}))
        if self.background:
            threading.Thread(target=self._start, args=(bot_id,), daemon=True).start()
        else:
            self._start(bot_id)

    def _start(self, bot_id: str) -> None:
        spec = self.get(bot_id)
        try:
            if spec is None:
                raise KeyError(bot_id)
            running = self._build(spec)
        except Exception as exc:  # бот не запустился — показать причину в интерфейсе
            log.exception("bot_start_failed", bot=bot_id)
            with self._lock:
                self._state[bot_id] = "error"
                self._errors[bot_id] = str(exc) or type(exc).__name__
            return
        with self._lock:
            if self._state.get(bot_id) != "starting":  # пока запускался, его остановили
                running.close()
                return
            self._running[bot_id] = running
            self._state[bot_id] = "running"
        self._refresh_feed()

    def _build(self, spec: BotSpec) -> _Running:
        cfg = self.load_cfg()
        days = history_days(spec.strategy, spec.timeframe, spec.params)
        if self.history is not None:
            now = self.now()
            self.history(spec.symbol, (now - pd.Timedelta(days=days)).floor("1D"), now)
        settings = LiveSettings(
            mode="paper",
            paper_capital=spec.capital,
            history_days=days,
            instances=[
                Instance(
                    strategy=spec.strategy,
                    symbol=spec.symbol,
                    timeframe=spec.timeframe,
                    params=spec.params,
                )
            ],
        )
        conn = connect(db_path(self.data_dir), threads=True)
        store = Store(conn, spec.mode)
        broker = PaperBroker(CostModel.from_config(cfg.costs), cfg.instruments, spec.capital)
        account = f"paper-{spec.id}"
        journal = Journal(self.journal_dir / f"{account}.jsonl", account=account)

        def notify(text: str) -> None:
            self.notify(f"[paper {spec.id}] {text}")

        bot = LiveBot(
            cfg,
            settings,
            broker,
            store,
            self.storage,
            notify=notify,
            now=self.now,
            journal=journal,
            control_dir=self.root / spec.id,
        )
        bot.start()
        return _Running(spec, bot, conn.close)

    def stop(self, bot_id: str) -> None:
        """Остановить бота. Виртуальные позиции сохраняются и продолжатся после запуска."""
        spec = self.get(bot_id)
        if spec is None:
            raise KeyError(bot_id)
        self._save(spec.model_copy(update={"enabled": False}))
        with self._lock:
            running = self._running.pop(bot_id, None)
            self._state.pop(bot_id, None)
        if running is not None:
            with running.bot.lock:
                running.bot.event("stop", "бот остановлен из интерфейса")
                running.close()
        self._refresh_feed()

    def delete(self, bot_id: str) -> None:
        """Убрать бота из списка. Его сделки и журнал остаются в базе и в logs/journal."""
        self.stop(bot_id)
        path = self._spec_path(bot_id)
        path.unlink(missing_ok=True)
        for leftover in ("KILL", "RESUME"):
            (path.parent / leftover).unlink(missing_ok=True)
        if path.parent.is_dir() and not any(path.parent.iterdir()):
            path.parent.rmdir()

    def kill(self, bot_id: str, reason: str) -> None:
        running = self._running.get(bot_id)
        if running is None:
            raise KeyError(bot_id)
        running.bot.kill(reason)

    def resume(self, bot_id: str) -> None:
        running = self._running.get(bot_id)
        if running is None:
            raise KeyError(bot_id)
        running.bot.resume()

    def kill_all(self, reason: str) -> int:
        with self._lock:
            bots = list(self._running.values())
        for running in bots:
            running.bot.kill(reason)
        return len(bots)

    def shutdown(self) -> None:
        with self._lock:
            bots = list(self._running.values())
            self._running.clear()
            self._state.clear()
        for running in bots:
            running.close()
        self._stop_feed()

    # --- свечи ----------------------------------------------------------------------

    def on_candles(self, symbol: str, df: pd.DataFrame) -> None:
        with self._lock:
            bots = [r for r in self._running.values() if r.spec.symbol == symbol]
        for running in bots:
            try:
                running.bot.on_candles(symbol, df)
            except Exception:  # один бот не должен мешать остальным
                log.exception("bot_candle_failed", bot=running.spec.id)

    def _refresh_feed(self) -> None:
        if self.feed_factory is None:
            return
        with self._lock:
            symbols = sorted({r.spec.symbol for r in self._running.values()})
            if symbols == self._feed_symbols:
                return
            self._stop_feed()
            self._feed_symbols = symbols
            if not symbols:
                return
            feed = self.feed_factory(symbols, self.on_candles)
            stop = threading.Event()
            thread = threading.Thread(target=self._run_feed, args=(feed, stop), daemon=True)
            self._feed_stop, self._feed_thread = stop, thread
            thread.start()

    @staticmethod
    def _run_feed(feed: Feed, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                feed.run(24 * 365, stop)
            except Exception:  # обрыв или ошибка — пауза и новое подключение
                log.exception("bot_feed_failed")
            stop.wait(30)  # поток закончился сам — переподключиться, но не чаще раза в 30 с

    def _stop_feed(self) -> None:
        if self._feed_stop is not None:
            self._feed_stop.set()
        self._feed_stop, self._feed_thread = None, None
        self._feed_symbols = []

    # --- состояние для интерфейса ---------------------------------------------------

    def state(self, bot_id: str) -> str:
        return self._state.get(bot_id, "stopped")

    def summary(self, spec: BotSpec) -> dict[str, Any]:
        status = bot_status(self.data_dir, spec.mode)
        return {
            **spec.model_dump(),
            "mode": spec.mode,
            "state": self.state(spec.id),
            "error": self._errors.get(spec.id),
            "equity": (status["risk"] or {}).get("equity"),
            "kill": status["kill"],
            "open_trades": len(status["trades"]),
            **self._totals(spec.mode),
        }

    def _totals(self, mode: str) -> dict[str, Any]:
        path = db_path(self.data_dir)
        if not path.exists():
            return {"closed_trades": 0, "closed_pnl": 0.0}
        conn = connect(path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(net_pnl), 0) AS pnl FROM live_trades "
                "WHERE mode = ? AND status = 'closed'",
                (mode,),
            ).fetchone()
        finally:
            conn.close()
        return {"closed_trades": int(row["n"]), "closed_pnl": round(float(row["pnl"]), 2)}

    def list(self) -> list[dict[str, Any]]:
        return [self.summary(spec) for spec in self.specs()]

    def detail(self, bot_id: str) -> dict[str, Any]:
        spec = self.get(bot_id)
        if spec is None:
            raise KeyError(bot_id)
        return {
            **self.summary(spec),
            "status": bot_status(self.data_dir, spec.mode),
            "journal": str(self.journal_dir / f"paper-{spec.id}.jsonl"),
        }
