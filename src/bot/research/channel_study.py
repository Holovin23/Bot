"""Есть ли польза от мнений Telegram-канала (этап 6, шаг 6).

Для каждого поста с мнением BULLISH/BEARISH по символу считаем доходность через 4, 24
и 72 часа после публикации со знаком мнения (+ для BULLISH, − для BEARISH). Вход — open
первой 1m-свечи, открывшейся не раньше времени поста. Сравниваем со случайными
моментами времени с той же долей «бычьих» и «медвежьих» мнений.

    uv run python -m bot.research.channel_study --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bot.analyzer.llm_parser import DEFAULT_MODEL, PROMPT_VERSION
from bot.core.config import load_config
from bot.core.db import connect, db_path
from bot.data.storage import Storage

HORIZONS_H = (4, 24, 72)
RANDOM_RUNS = 1000


def forward_returns(
    times_ns: NDArray[np.int64],
    opens: NDArray[np.float64],
    events_ns: NDArray[np.int64],
    hours: int,
) -> NDArray[np.float64]:
    """Доходность в % от open первой свечи ≥ события до open первой свечи ≥ события + hours.
    NaN, если данных не хватает."""
    entry = np.searchsorted(times_ns, events_ns, side="left")
    exit_ = np.searchsorted(times_ns, events_ns + np.int64(hours * 3_600_000_000_000), side="left")
    out = np.full(len(events_ns), np.nan)
    ok = exit_ < len(times_ns)
    out[ok] = (opens[exit_[ok]] / opens[entry[ok]] - 1) * 100
    return out


@dataclass(frozen=True)
class HorizonResult:
    hours: int
    posts: int
    mean_signed_pct: float
    hit_rate: float
    random_mean_pct: float
    percentile: float  # доля случайных прогонов со средним ниже, чем у канала


def study(
    candles_1m: pd.DataFrame,
    events: pd.DataFrame,
    horizons: tuple[int, ...] = HORIZONS_H,
    runs: int = RANDOM_RUNS,
    seed: int = 0,
) -> list[HorizonResult]:
    """events: колонки time (UTC) и sign (+1 / −1)."""
    times = candles_1m["open_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    opens = candles_1m["open"].to_numpy(dtype="float64")
    ev_times = pd.to_datetime(events["time"], utc=True).to_numpy(dtype="datetime64[ns]")
    ev_ns = ev_times.astype("int64")
    signs = events["sign"].to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    results = []
    for h in horizons:
        signed = signs * forward_returns(times, opens, ev_ns, h)
        valid = ~np.isnan(signed)
        n = int(valid.sum())
        if n == 0:
            results.append(HorizonResult(h, 0, float("nan"), float("nan"), float("nan"), 0.0))
            continue
        observed = float(signed[valid].mean())
        latest = times[-1] - np.int64(h * 3_600_000_000_000)
        random_means = np.empty(runs)
        share_long = float((signs[valid] > 0).mean())
        for i in range(runs):
            moments = rng.integers(times[0], latest, size=n)
            random_signs = np.where(rng.uniform(size=n) < share_long, 1.0, -1.0)
            random_means[i] = np.nanmean(random_signs * forward_returns(times, opens, moments, h))
        results.append(
            HorizonResult(
                hours=h,
                posts=n,
                mean_signed_pct=observed,
                hit_rate=float((signed[valid] > 0).mean()),
                random_mean_pct=float(random_means.mean()),
                percentile=float((random_means < observed).mean()),
            )
        )
    return results


def load_events(conn: sqlite3.Connection, symbol: str, model: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT p.published_at, r.result_json FROM post_parses r JOIN telegram_posts p "
        "ON p.channel = r.channel AND p.post_id = r.post_id "
        "WHERE r.status = 'parsed' AND r.model = ? AND r.prompt_version = ?",
        (model, PROMPT_VERSION),
    ).fetchall()
    records = []
    for row in rows:
        for item in json.loads(row["result_json"]).get("instruments", []):
            if item.get("symbol") == symbol and item.get("bias") in ("BULLISH", "BEARISH"):
                sign = 1.0 if item["bias"] == "BULLISH" else -1.0
                records.append({"time": row["published_at"], "sign": sign})
    return pd.DataFrame(records, columns=["time", "sign"])


def render(symbol: str, results: list[HorizonResult]) -> str:
    lines = [
        f"# Польза мнений канала: {symbol}",
        "",
        "| Горизонт | Постов | Средняя доходность по мнению | Доля верных | Случайные моменты "
        "| Лучше случайных |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.hours} ч | {r.posts} | {r.mean_signed_pct:+.3f}% | {r.hit_rate:.0%} | "
            f"{r.random_mean_pct:+.3f}% | {r.percentile:.0%} |"
        )
    lines += [
        "",
        "Канал полезен, если на нескольких горизонтах он лучше 95% случайных прогонов и "
        "эффект держится на разных годах. При сотнях постов случайная удача вероятна.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Польза мнений Telegram-канала")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    data_dir = cfg.resolve(cfg.data.dir)
    events = load_events(connect(db_path(data_dir)), args.symbol, args.model)
    if events.empty:
        raise SystemExit("Нет разобранных постов с мнением BULLISH/BEARISH. Сначала --parse.")
    times = pd.to_datetime(events["time"], utc=True)
    candles = Storage(data_dir).read_candles(
        args.symbol, "1m", times.min() - pd.Timedelta(days=1), times.max() + pd.Timedelta(days=4)
    )
    if candles.empty:
        raise SystemExit("Нет 1m-свечей за период постов.")
    text = render(args.symbol, study(candles, events))
    out = data_dir / "research" / f"channel_study_{args.symbol}_{datetime.now(UTC):%Y%m%d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
