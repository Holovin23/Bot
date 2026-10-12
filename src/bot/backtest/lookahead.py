"""Проверка «обрезки данных»: если удалить свечи после момента cut, признаки и сигналы
до cut не должны измениться. Если изменились — стратегия заглядывает в будущее
(shift не в ту сторону, нормализация по всему датасету и т. п.)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from bot.core.models import Signal
from bot.strategies.base import Strategy, generate_signals


def _key(signal: Signal) -> tuple[object, ...]:
    return (
        signal.signal_id,
        signal.side,
        round(signal.entry_price, 10),
        round(signal.stop_loss, 10),
        None if signal.take_profit is None else round(signal.take_profit, 10),
        signal.time_stop_bars,
    )


def find_lookahead(
    strategy: Strategy, symbol: str, bars: pd.DataFrame, cut: datetime | pd.Timestamp
) -> list[str]:
    """Список расхождений (пустой — заглядывания не найдено).

    cut — время: в обрезанных данных остаются свечи с open_time < cut.
    """
    cut_ts = pd.Timestamp(cut)
    kept = bars[bars["open_time"] < cut_ts].reset_index(drop=True)
    problems = _compare_features(strategy, bars.reset_index(drop=True), kept)
    full = [s for s in generate_signals(strategy, symbol, bars) if s.created_at <= cut_ts]
    truncated = generate_signals(strategy, symbol, kept)
    full_keys = {_key(s) for s in full}
    cut_keys = {_key(s) for s in truncated}
    problems += [f"только на полных данных: {k}" for k in sorted(full_keys - cut_keys, key=str)]
    problems += [f"только на обрезанных: {k}" for k in sorted(cut_keys - full_keys, key=str)]
    return problems


def _compare_features(strategy: Strategy, full: pd.DataFrame, kept: pd.DataFrame) -> list[str]:
    """Признаки по обрезанным данным должны совпасть с признаками по полным."""
    full_features = strategy.prepare(full).reset_index(drop=True).iloc[: len(kept)]
    cut_features = strategy.prepare(kept).reset_index(drop=True)
    problems = []
    for column in full_features.columns:
        a = full_features[column].to_numpy(dtype="float64")
        b = cut_features[column].to_numpy(dtype="float64")
        same = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True)
        if not same.all():
            row = int(np.flatnonzero(~same)[0])
            problems.append(f"признак {column!r} отличается, первая строка {row}")
    return problems
