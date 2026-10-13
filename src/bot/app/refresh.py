"""Догрузка свежих свечей для графика (почему график «отстаёт» — см. USER_GUIDE, раздел 5).

График строится из файлов data/klines, а не из интернета. Эти файлы пополняют только
загрузчик истории, живой поток и бот. Если они не запущены, график останавливается на
последней загрузке. Поэтому интерфейс сам догружает недостающие закрытые 1m-свечи с
публичного API Bybit (ключи не нужны) — в фоне, не чаще раза в MIN_INTERVAL на символ.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from bot.core.logging import get_logger

log = get_logger(__name__)

MIN_INTERVAL = pd.Timedelta(seconds=30)
FRESH_AFTER = pd.Timedelta(minutes=2)  # последняя закрытая 1m-свеча не старше — данные свежие

Download = Callable[[str, pd.Timestamp], int]  # (символ, с какого времени) → новых свечей


@dataclass
class RefreshState:
    running: bool = False
    error: str | None = None
    last_try: pd.Timestamp | None = None
    added: int = 0


class CandleRefresher:
    def __init__(
        self,
        download: Download,
        last_time: Callable[[str], pd.Timestamp | None],
        now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
        background: bool = True,
    ) -> None:
        self.download = download
        self.last_time = last_time
        self.now = now
        self.background = background
        self.states: dict[str, RefreshState] = {}
        self._lock = threading.Lock()

    def stale(self, symbol: str) -> bool:
        last = self.last_time(symbol)
        return last is None or self.now() - (last + pd.Timedelta(minutes=1)) > FRESH_AFTER

    def request(self, symbol: str) -> RefreshState:
        """Запустить догрузку, если данные устарели и с прошлой попытки прошло время."""
        with self._lock:
            state = self.states.setdefault(symbol, RefreshState())
            if state.running or not self.stale(symbol):
                return state
            if state.last_try is not None and self.now() - state.last_try < MIN_INTERVAL:
                return state
            state.running, state.last_try = True, self.now()
        if self.background:
            threading.Thread(target=self._run, args=(symbol,), daemon=True).start()
        else:
            self._run(symbol)
        return state

    def _run(self, symbol: str) -> None:
        state = self.states[symbol]
        last = self.last_time(symbol)
        start = (
            last + pd.Timedelta(minutes=1)
            if last is not None
            else self.now() - pd.Timedelta(days=7)
        )
        try:
            state.added = self.download(symbol, start.floor("1min"))
            state.error = None
        except Exception as exc:  # нет интернета, Bybit недоступен — график покажет старое
            log.warning("chart_refresh_failed", symbol=symbol, error=type(exc).__name__)
            state.error = f"не удалось догрузить свечи с Bybit: {type(exc).__name__}"
        finally:
            state.running = False
