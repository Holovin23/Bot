"""Живой поток 1m-свечей с WebSocket Bybit (основной API, публичный канал linear).

Сохраняются только закрытые свечи (confirm=true). Если между сохранённой и пришедшей
свечой есть пропуск (обрыв связи, переподключение), недостающее дозагружается через REST.
Если поток молчит дольше STALE_AFTER, бот сам дозагружает свечи и переподключается.

Запуск на 24 часа:
    uv run python -m bot.data.live_feed --hours 24
"""

from __future__ import annotations

import argparse
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import pandas as pd
from pybit.unified_trading import WebSocket

from bot.core.config import load_config
from bot.core.logging import get_logger, setup_logging
from bot.data.bybit_public import make_public_client
from bot.data.downloader import Downloader
from bot.data.storage import Storage

log = get_logger(__name__)

STALE_AFTER = pd.Timedelta(minutes=3)
CHECK_EVERY_S = 30.0
NEW_SYMBOL_LOOKBACK = pd.Timedelta(days=1)  # если истории нет совсем, берём последние сутки


class Backfiller(Protocol):
    def download_klines(self, symbol: str, start: pd.Timestamp, timeframe: str = "1m") -> int: ...


def parse_confirmed(message: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    """Из сообщения kline.1.<symbol> взять только закрытые свечи."""
    symbol = str(message["topic"]).split(".")[-1]
    rows = [item for item in message.get("data", []) if item.get("confirm")]
    df = pd.DataFrame(
        {
            "open_time": pd.to_datetime([int(r["start"]) for r in rows], unit="ms", utc=True),
            "open": [float(r["open"]) for r in rows],
            "high": [float(r["high"]) for r in rows],
            "low": [float(r["low"]) for r in rows],
            "close": [float(r["close"]) for r in rows],
            "volume": [float(r["volume"]) for r in rows],
            "turnover": [float(r["turnover"]) for r in rows],
        }
    )
    return symbol, df


class LiveFeed:
    def __init__(
        self,
        storage: Storage,
        backfiller: Backfiller,
        symbols: list[str],
        on_candle: Callable[[str, pd.DataFrame], None] | None = None,
        now: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
    ) -> None:
        self.storage = storage
        self.backfiller = backfiller
        self.symbols = symbols
        self.on_candle = on_candle
        self.now = now
        self._lock = threading.Lock()  # сообщения WS приходят из другого потока
        self.last_message_at = now()

    def backfill(self, symbol: str) -> int:
        """Дозагрузить через REST всё закрытое после последней сохранённой свечи."""
        last = self.storage.last_candle_time(symbol, "1m")
        start = (
            last + pd.Timedelta(minutes=1) if last is not None else self.now() - NEW_SYMBOL_LOOKBACK
        )
        added = self.backfiller.download_klines(symbol, start.floor("1min"), "1m")
        if added:
            log.info("backfilled", symbol=symbol, candles=added)
        return added

    def handle_message(self, message: dict[str, Any]) -> None:
        with self._lock:
            self.last_message_at = self.now()
            symbol, df = parse_confirmed(message)
            if df.empty:
                return
            last = self.storage.last_candle_time(symbol, "1m")
            first_new = df["open_time"].min()
            if last is not None and first_new > last + pd.Timedelta(minutes=1):
                log.warning("gap_detected", symbol=symbol, last=str(last), received=str(first_new))
                self.backfill(symbol)
            self.storage.write_candles(symbol, "1m", df)
            if self.on_candle is not None:
                self.on_candle(symbol, df)

    def check_stale(self) -> bool:
        """True, если поток молчит слишком долго (тогда дозагрузили через REST)."""
        with self._lock:
            silent = self.now() - self.last_message_at
            if silent <= STALE_AFTER:
                return False
            log.warning("stream_silent", seconds=int(silent.total_seconds()))
            for symbol in self.symbols:
                self.backfill(symbol)
            self.last_message_at = self.now()
            return True

    def run(self, hours: float, stop: threading.Event | None = None) -> None:
        """stop — событие для остановки из другого потока (боты веб-интерфейса)."""
        stop = stop or threading.Event()
        for symbol in self.symbols:
            self.backfill(symbol)
        ws = self._connect()
        deadline = time.monotonic() + hours * 3600
        try:
            while time.monotonic() < deadline and not stop.is_set():
                if stop.wait(CHECK_EVERY_S):
                    break
                if self.check_stale():
                    ws.exit()
                    ws = self._connect()
        except KeyboardInterrupt:
            print("Остановлено пользователем")
        finally:
            ws.exit()
        print("Живой поток завершён")

    def _connect(self) -> WebSocket:
        ws = WebSocket(testnet=False, channel_type="linear")
        ws.kline_stream(interval=1, symbol=self.symbols, callback=self._safe_handle)
        log.info("ws_connected", symbols=self.symbols)
        return ws

    def _safe_handle(self, message: dict[str, Any]) -> None:
        try:
            self.handle_message(message)
        except Exception:  # ошибка в одной свече не должна ронять поток
            log.exception("ws_message_failed")


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Живой поток закрытых 1m-свечей")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--symbol", action="append")
    args = parser.parse_args(argv)

    setup_logging(cfg.resolve(cfg.logging.dir), cfg.logging.level, cfg.logging.retention_days)
    storage = Storage(cfg.resolve(cfg.data.dir))
    downloader = Downloader(make_public_client(), storage, category=cfg.data.category)

    def show(symbol: str, df: pd.DataFrame) -> None:
        row = df.iloc[-1]
        print(f"{row['open_time']:%Y-%m-%d %H:%M} UTC {symbol} close {row['close']}")

    feed = LiveFeed(storage, downloader, args.symbol or cfg.data.symbols, on_candle=show)
    print(f"Живой поток на {args.hours} ч. Остановить: Ctrl+C")
    feed.run(args.hours)


if __name__ == "__main__":
    main()
