"""Стратегия по сценариям из Telegram-канала (docs/10-telegram-scenarios-plan.md).

Сценарии (ячейки) хранятся в SQLite, их создаёт разбор постов. Стратегия на закрытых
свечах ищет первую свечу после публикации, на которой выполнилось условие сценария, и
даёт рыночный сигнал (вход — открытие следующей свечи). Стоп и цель — от автора или
по умолчанию (scenarios/levels.py), дальше обычные проверки риск-менеджера.

Сценарий срабатывает один раз. До срабатывания он «умирает», если цена дошла до стопа
автора (сценарий отменён рынком) или до его первой цели (движение упущено). Срок жизни
не ограничен (решение владельца), max_age_days > 0 его включает. Но пост, опубликованный
раньше первой загруженной свечи, не торгуется: по такой истории не проверить, что он ещё
не сработал. Живой бот читает history_days дней (config/live.yaml, у paper-ботов — 30).

Параметры: latency_minutes — через сколько минут после поста ячейка начинает работать
(время на разбор); stop_buffer_atr, now_stop_atr, default_rr, min_rr — см. levels.py;
allow_image_levels — торговать ячейки, у которых уровни только с картинки и которые
владелец ещё не одобрил (для сравнения в бэктесте).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from bot.core.models import Side, Signal
from bot.core.timeframes import timeframe_delta
from bot.features.indicators import atr
from bot.scenarios.levels import LevelParams, TradeLevels, trade_levels
from bot.scenarios.store import ScenarioVersion
from bot.strategies.base import BaseStrategy, MarketContext

LATENCY = pd.Timedelta(minutes=5)
VersionsProvider = Callable[[str], list[ScenarioVersion]]

DEFAULT_PARAMS: dict[str, Any] = {
    "latency_minutes": 5,
    "stop_buffer_atr": 0.5,
    "now_stop_atr": 1.5,
    "default_rr": 2.0,
    "min_rr": 2.0,
    "allow_image_levels": False,
    "max_age_days": 0,
}


@dataclass(frozen=True)
class Fire:
    """Сценарий выполнился на закрытии свечи bar_index."""

    bar_index: int
    version: ScenarioVersion


def _db_versions(symbol: str) -> list[ScenarioVersion]:
    from bot.core.config import load_config
    from bot.core.db import connect, db_path
    from bot.scenarios.store import load_versions

    cfg = load_config()
    conn = connect(db_path(cfg.resolve(cfg.data.dir)))
    try:
        return load_versions(conn, symbol)
    finally:
        conn.close()


def find_fires(
    bars: pd.DataFrame,
    timeframe: str,
    versions: Sequence[ScenarioVersion],
    latency: pd.Timedelta = LATENCY,
    max_age: pd.Timedelta | None = None,
) -> tuple[list[Fire], dict[str, str]]:
    """Срабатывания сценариев на истории и итог по каждому несработавшему.

    Причинно: срабатывание на свече i зависит только от свечей 0..i и версий,
    опубликованных до неё, поэтому на обрезанной истории получаются те же события.
    """
    opens = pd.to_datetime(bars["open_time"], utc=True).reset_index(drop=True)
    step = timeframe_delta(timeframe)
    h = bars["high"].to_numpy(dtype="float64")
    low_ = bars["low"].to_numpy(dtype="float64")
    c = bars["close"].to_numpy(dtype="float64")
    fires: list[Fire] = []
    outcome: dict[str, str] = {}
    by_id: dict[str, list[ScenarioVersion]] = {}
    for v in versions:
        by_id.setdefault(v.scenario_id, []).append(v)
    for sid, chain in by_id.items():
        fired = False
        for v in sorted(chain, key=lambda x: x.version):
            if fired:
                break
            s = v.scenario
            t = s.trigger
            long = s.side == "LONG"
            start = v.valid_from + latency
            if len(opens) == 0 or start < opens[0]:
                # Пост раньше загруженной истории: неизвестно, сработал ли он уже и не
                # отменён ли рынком, поэтому не торгуем (живой бот видит history_days дней).
                outcome[sid] = "пост раньше загруженной истории — не проверяется"
                break
            first = int(opens.searchsorted(start))
            stop_at = v.valid_to
            if max_age is not None:
                limit = v.valid_from + max_age
                stop_at = limit if stop_at is None else min(stop_at, limit)
            armed = False
            if t.type == "retest" and t.level is not None and first > 0:
                armed = c[first - 1] > t.level if long else c[first - 1] < t.level
            target = None
            if s.targets:
                target = min(s.targets) if long else max(s.targets)
            for i in range(first, len(c)):
                if stop_at is not None and opens[i] + step > stop_at:
                    outcome[sid] = "заменён или отменён до срабатывания"
                    break
                hit = False
                if t.type == "now":
                    hit = True
                elif t.type == "close_above" and t.level is not None:
                    hit = c[i] > t.level and i > 0 and c[i - 1] <= t.level
                elif t.type == "close_below" and t.level is not None:
                    hit = c[i] < t.level and i > 0 and c[i - 1] >= t.level
                elif t.type == "retest" and t.level is not None:
                    if armed:
                        hit = low_[i] <= t.level < c[i] if long else h[i] >= t.level > c[i]
                    armed = armed or (c[i] > t.level if long else c[i] < t.level)
                elif t.type == "touch_zone" and t.zone_low is not None and t.zone_high is not None:
                    if long:
                        hit = low_[i] <= t.zone_high and c[i] >= t.zone_low
                    else:
                        hit = h[i] >= t.zone_low and c[i] <= t.zone_high
                if hit:
                    fires.append(Fire(i, v))
                    outcome[sid] = f"сработал {opens[i] + step:%Y-%m-%d %H:%M} UTC"
                    fired = True
                    break
                # До срабатывания: стоп автора — сценарий сломан, первая цель — упущен
                if s.stop is not None and (low_[i] <= s.stop if long else h[i] >= s.stop):
                    outcome[sid] = f"цена дошла до стопа автора до входа ({opens[i]:%Y-%m-%d})"
                    break
                if target is not None and (h[i] >= target if long else low_[i] <= target):
                    outcome[sid] = f"цена дошла до цели без входа ({opens[i]:%Y-%m-%d})"
                    break
            else:
                outcome[sid] = "условие ещё не выполнилось"
    fires.sort(key=lambda f: (f.bar_index, f.version.scenario_id))
    return fires, outcome


class TelegramScenario(BaseStrategy):
    name = "telegram_scenario"
    version = "1.0.0"

    def __init__(
        self,
        timeframe: str,
        params: dict[str, Any] | None = None,
        versions: VersionsProvider | Sequence[ScenarioVersion] | None = None,
    ) -> None:
        super().__init__(timeframe, {**DEFAULT_PARAMS, **(params or {})})
        if versions is None:
            self._provider: VersionsProvider = _db_versions
        elif callable(versions):
            self._provider = versions
        else:
            fixed = list(versions)
            self._provider = lambda _symbol: fixed
        p = self.params
        self.levels = LevelParams(
            stop_buffer_atr=float(p["stop_buffer_atr"]),
            now_stop_atr=float(p["now_stop_atr"]),
            default_rr=float(p["default_rr"]),
            min_rr=float(p["min_rr"]),
        )
        self._bars: pd.DataFrame | None = None
        self._cache: tuple[str, int, dict[int, list[Fire]]] | None = None
        self.outcome: dict[str, str] = {}  # итог по сценариям — для отчёта бэктеста поста
        self.notes: list[str] = []  # почему сработавший сценарий не дал сигнала

    def required_history(self) -> int:
        return 15

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        self._bars = bars
        self._cache = None
        return pd.DataFrame({"atr": atr(bars, 14)}, index=bars.index)

    def _fires(self, symbol: str) -> dict[int, list[Fire]]:
        assert self._bars is not None, "сначала prepare()"
        key = (symbol, id(self._bars))
        if self._cache is None or self._cache[:2] != key:
            age = int(self.params["max_age_days"])
            fires, self.outcome = find_fires(
                self._bars,
                self.timeframe,
                self._provider(symbol),
                pd.Timedelta(minutes=int(self.params["latency_minutes"])),
                pd.Timedelta(days=age) if age > 0 else None,
            )
            by_bar: dict[int, list[Fire]] = {}
            for f in fires:
                by_bar.setdefault(f.bar_index, []).append(f)
            self._cache = (symbol, id(self._bars), by_bar)
        return self._cache[2]

    def on_bar(self, ctx: MarketContext) -> list[Signal]:
        for fire in self._fires(ctx.symbol).get(ctx.index, []):
            v = fire.version
            s = v.scenario
            tag = f"{v.scenario_id} v{v.version}"
            if v.needs_review and not self.params["allow_image_levels"]:
                self.notes.append(f"{tag}: уровни только с картинки, ячейка не одобрена")
                continue
            atr_value = ctx.value("atr")
            levels = trade_levels(s, ctx.value("close"), atr_value, self.levels)
            if not isinstance(levels, TradeLevels):
                self.notes.append(f"{tag}: {levels}")
                continue
            if math.isnan(levels.stop) or levels.stop <= 0:
                self.notes.append(f"{tag}: стоп не положительный")
                continue
            origin = "автор" if levels.stop_from_author else "по умолчанию"
            target = "автор" if levels.take_from_author else "по умолчанию"
            return [
                self.signal(
                    ctx,
                    Side(s.side),
                    stop_loss=levels.stop,
                    take_profit=levels.take,
                    reason=f"{tag}: {s.trigger.type}, стоп — {origin}, цель — {target}",
                    features={"scenario_id": v.scenario_id, "scenario_version": v.version},
                )
            ]
        return []
