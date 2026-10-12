from pathlib import Path

import pandas as pd
import pytest

from bot.backtest.costs import CostModel
from bot.core.config import Config, load_config
from bot.core.db import connect
from bot.core.models import Side, Signal
from bot.data.storage import Storage
from bot.execution.paper import PaperBroker
from bot.execution.store import Store
from bot.features.indicators import atr
from bot.runner.live import LiveBot, Running, bar_closes_at
from bot.runner.settings import Instance, LiveSettings
from bot.strategies.base import BaseStrategy, MarketContext

from .strategy_helpers import minute_walk


class AlwaysLong(BaseStrategy):
    """Тестовая стратегия: лонг на каждой свече, стоп 1,5 ATR, тейк 3 ATR, 2 свечи."""

    name = "always_long"
    version = "1"

    def required_history(self) -> int:
        return 20

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"atr": atr(bars, 14)})

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        a, close = ctx.value("atr"), ctx.value("close")
        return [self.signal(ctx, Side.LONG, close - 1.5 * a, close + 3 * a, time_stop_bars=2)]


class Clock:
    def __init__(self) -> None:
        self.now = pd.Timestamp("2024-01-03", tz="UTC")

    def __call__(self) -> pd.Timestamp:
        return self.now


def make_bot(tmp_path: Path) -> tuple[LiveBot, Storage, Clock, list[str], pd.DataFrame]:
    cfg: Config = load_config(env_file=None)
    candles = minute_walk(days=3, seed=3)
    storage = Storage(tmp_path)
    storage.write_candles("BTCUSDT", "1m", candles[candles["open_time"] < "2024-01-03"])
    clock = Clock()
    broker = PaperBroker(CostModel.from_config(cfg.costs), cfg.instruments, 10_000)
    store = Store(connect(tmp_path / "bot.sqlite"), "paper")
    instance = Instance(strategy="always_long", symbol="BTCUSDT", timeframe="1h")
    settings = LiveSettings(instances=[instance], history_days=3)
    messages: list[str] = []
    bot = LiveBot(
        cfg,
        settings,
        broker,
        store,
        storage,
        notify=messages.append,
        now=clock,
        strategies=[Running(instance, AlwaysLong("1h"))],
    )
    rest = candles[candles["open_time"] >= "2024-01-03"].reset_index(drop=True)
    return bot, storage, clock, messages, rest


def feed(bot: LiveBot, storage: Storage, clock: Clock, rest: pd.DataFrame, hours: int) -> None:
    start = clock.now
    for h in range(hours):
        chunk = rest[
            (rest["open_time"] >= start + pd.Timedelta(hours=h))
            & (rest["open_time"] < start + pd.Timedelta(hours=h + 1))
        ]
        storage.write_candles("BTCUSDT", "1m", chunk)
        clock.now = chunk["open_time"].iloc[-1] + pd.Timedelta(minutes=1, seconds=5)
        bot.on_candles("BTCUSDT", chunk)


def test_bar_closes_at() -> None:
    assert bar_closes_at(pd.Timestamp("2024-01-01 10:59", tz="UTC"), "1h")
    assert not bar_closes_at(pd.Timestamp("2024-01-01 10:58", tz="UTC"), "1h")
    assert bar_closes_at(pd.Timestamp("2024-01-01 10:14", tz="UTC"), "15m")


def test_paper_day_opens_and_closes_trades(tmp_path: Path) -> None:
    bot, storage, clock, messages, rest = make_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=12)
    trades = bot.store.trades()
    closed = [t for t in trades if t.status == "closed"]
    assert len(closed) >= 2
    assert all(t.exit_reason in ("stop", "take", "time") for t in closed)
    assert all(t.net_pnl is not None for t in closed)
    assert len(bot.store.active_trades()) <= 1  # одна позиция на символ
    # капитал бумажного брокера = старт + сумма закрытых (открытых нет или учтены отдельно)
    open_fee = sum(
        t.qty * (t.entry_price or 0) * bot.costs.taker_fee for t in bot.store.active_trades()
    )
    total = sum(t.net_pnl or 0 for t in closed)
    assert isinstance(bot.broker, PaperBroker)
    assert bot.broker.cash == pytest.approx(10_000 + total - open_fee, abs=1e-6)
    state = bot.store.get_risk_state()
    assert state is not None and state.day.isoformat() == "2024-01-03"
    assert any("Вход BTCUSDT" in m for m in messages)


def test_time_stop_closes_after_two_bars(tmp_path: Path) -> None:
    bot, storage, clock, _, rest = make_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=12)
    by_time = [t for t in bot.store.trades("closed") if t.exit_reason == "time"]
    assert by_time and all(t.bars_held == 2 for t in by_time)


def test_kill_file_closes_positions_and_blocks_entries(tmp_path: Path) -> None:
    bot, storage, clock, messages, rest = make_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=1)
    assert bot.store.active_trades()  # вход на закрытии первой свечи
    (storage.root / "KILL").write_text("")
    feed(bot, storage, clock, rest.iloc[60:], hours=3)
    assert bot.store.active_trades() == []
    assert bot.broker.get_positions() == []
    assert bot.store.kill_reason() == "файл data/KILL"
    assert any("АВАРИЙНАЯ ОСТАНОВКА" in m for m in messages)
    count = len(bot.store.trades())
    feed(bot, storage, clock, rest.iloc[240:], hours=2)
    assert len(bot.store.trades()) == count  # новых входов нет
    bot.resume()
    assert bot.store.kill_reason() is None and not (storage.root / "KILL").exists()


def test_stale_data_rejects_signal(tmp_path: Path) -> None:
    bot, storage, clock, _, rest = make_bot(tmp_path)
    bot.start()
    chunk = rest.iloc[:60]
    storage.write_candles("BTCUSDT", "1m", chunk)
    clock.now = pd.Timestamp("2024-01-03 05:00", tz="UTC")  # данные отстали на 4 часа
    bot.on_candles("BTCUSDT", chunk)
    assert bot.store.trades() == []
    assert any("13: данные устарели" in e["message"] for e in bot.store.events())


def test_restart_restores_paper_position(tmp_path: Path) -> None:
    bot, storage, clock, _, rest = make_bot(tmp_path)
    bot.start()
    feed(bot, storage, clock, rest, hours=1)
    [trade] = bot.store.active_trades()
    assert isinstance(bot.broker, PaperBroker)
    cash = bot.broker.cash
    fresh = PaperBroker(bot.costs, bot.cfg.instruments, 10_000)
    again = LiveBot(
        bot.cfg,
        bot.settings,
        fresh,
        bot.store,
        storage,
        notify=lambda _: None,
        now=clock,
        strategies=bot.running,
    )
    again.start()
    assert fresh.cash == pytest.approx(cash)
    [pos] = fresh.get_positions()
    assert pos.signal_id == trade.signal_id and pos.stop_loss == trade.stop
    assert again.store.active_trades()[0].status == "open"


def test_live_confirmation_and_account(tmp_path: Path) -> None:
    from bot.runner.live import confirm_live, load_live_account

    assert confirm_live(1234.9, ask=lambda _: "LIVE 1234")
    assert not confirm_live(1234.9, ask=lambda _: "yes")
    with pytest.raises(SystemExit, match="не настроен"):
        load_live_account(tmp_path / ".env.live")
    env = tmp_path / ".env.live"
    env.write_text("BYBIT_ENV=demo\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="ALLOW_LIVE"):
        load_live_account(env)
    env.write_text("BYBIT_ENV=live\nALLOW_LIVE=true\n", encoding="utf-8")
    cfg = load_live_account(env)
    assert cfg.risk.risk_per_trade == 0.0025 and cfg.limits.daily_loss_halts
