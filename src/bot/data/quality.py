"""Проверка качества свечей и отчёт.

Запуск:
    uv run python -m bot.data.quality --symbol BTCUSDT
Отчёт печатается в консоль и сохраняется в data/reports/quality_<symbol>.md.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from bot.core.config import load_config
from bot.core.timeframes import timeframe_minutes
from bot.data.storage import Storage
from bot.features.indicators import atr

OUTLIER_ATR = 10.0  # high/low дальше 10 ATR от close — выброс
ATR_LENGTH = 14
MAX_ROWS_IN_REPORT = 50


@dataclass(frozen=True)
class Gap:
    start: pd.Timestamp  # время открытия первой пропущенной свечи
    end: pd.Timestamp  # время открытия последней пропущенной свечи
    missing: int


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    gaps: list[Gap] = field(default_factory=list)
    duplicates: list[pd.Timestamp] = field(default_factory=list)
    zero_volume: list[pd.Timestamp] = field(default_factory=list)
    outliers: list[pd.Timestamp] = field(default_factory=list)
    high_below_low: list[pd.Timestamp] = field(default_factory=list)

    @property
    def missing_total(self) -> int:
        return sum(g.missing for g in self.gaps)

    @property
    def is_clean(self) -> bool:
        return not (
            self.gaps or self.duplicates or self.zero_volume or self.outliers or self.high_below_low
        )


def find_gaps(times: pd.Series, step: pd.Timedelta) -> list[Gap]:
    ordered = times.drop_duplicates().sort_values().reset_index(drop=True)
    diffs = ordered.diff()
    gaps = []
    for i in diffs.index[diffs > step]:
        prev, nxt = ordered[i - 1], ordered[i]
        missing = int((nxt - prev) / step) - 1
        gaps.append(Gap(start=prev + step, end=nxt - step, missing=missing))
    return gaps


def find_outliers(df: pd.DataFrame) -> pd.Series:
    """Маска выбросов. ATR берётся по предыдущей свече, чтобы выброс не раздул свой порог."""
    prev_atr = atr(df, ATR_LENGTH).shift(1)
    limit = OUTLIER_ATR * prev_atr
    return ((df["high"] - df["close"]) > limit) | ((df["close"] - df["low"]) > limit)


def check_candles(df: pd.DataFrame, symbol: str, timeframe: str) -> QualityReport:
    step = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    report = QualityReport(
        symbol=symbol,
        timeframe=timeframe,
        rows=len(df),
        first=df["open_time"].min() if len(df) else None,
        last=df["open_time"].max() if len(df) else None,
    )
    if df.empty:
        return report
    times = df["open_time"]
    report.duplicates = sorted(times[times.duplicated()].unique())
    clean = df.drop_duplicates("open_time").sort_values("open_time", ignore_index=True)
    report.gaps = find_gaps(clean["open_time"], step)
    report.zero_volume = list(clean.loc[clean["volume"] == 0, "open_time"])
    report.high_below_low = list(clean.loc[clean["high"] < clean["low"], "open_time"])
    report.outliers = list(clean.loc[find_outliers(clean), "open_time"])
    return report


def _fmt(ts: pd.Timestamp) -> str:
    return f"{ts:%Y-%m-%d %H:%M}"


def _times_section(title: str, times: list[pd.Timestamp]) -> list[str]:
    lines = [f"## {title}: {len(times)}", ""]
    lines += [f"- {_fmt(t)}" for t in times[:MAX_ROWS_IN_REPORT]]
    if len(times) > MAX_ROWS_IN_REPORT:
        lines.append(f"- … и ещё {len(times) - MAX_ROWS_IN_REPORT}")
    lines.append("")
    return lines


def render_markdown(report: QualityReport) -> str:
    lines = [f"# Качество данных {report.symbol} {report.timeframe}", ""]
    if report.first is None or report.last is None:
        return "\n".join([*lines, "Данных нет.", ""])
    lines += [
        f"- Период: {_fmt(report.first)} — {_fmt(report.last)} UTC",
        f"- Свечей: {report.rows}",
        f"- Пропущено свечей: {report.missing_total} в {len(report.gaps)} интервалах",
        f"- Итог: {'чисто' if report.is_clean else 'есть замечания'}",
        "",
        f"## Пропуски: {len(report.gaps)}",
        "",
    ]
    if report.gaps:
        lines += ["| С (UTC) | По (UTC) | Свечей |", "|---|---|---|"]
        lines += [
            f"| {_fmt(g.start)} | {_fmt(g.end)} | {g.missing} |"
            for g in report.gaps[:MAX_ROWS_IN_REPORT]
        ]
        if len(report.gaps) > MAX_ROWS_IN_REPORT:
            lines.append(f"| … | … | ещё {len(report.gaps) - MAX_ROWS_IN_REPORT} интервалов |")
    lines.append("")
    lines += _times_section("Дубли", report.duplicates)
    lines += _times_section("Нулевой объём", report.zero_volume)
    lines += _times_section(f"Выбросы (дальше {OUTLIER_ATR:g} ATR от close)", report.outliers)
    lines += _times_section("high < low", report.high_below_low)
    return "\n".join(lines)


def write_report(report: QualityReport, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if report.timeframe == "1m" else f"_{report.timeframe}"
    path = reports_dir / f"quality_{report.symbol}{suffix}.md"
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Отчёт о качестве свечей")
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--timeframe", default=cfg.data.base_interval)
    parser.add_argument("--from", dest="start")
    parser.add_argument("--to", dest="end")
    args = parser.parse_args(argv)

    data_dir = cfg.resolve(cfg.data.dir)
    storage = Storage(data_dir)
    start = pd.Timestamp(args.start, tz="UTC") if args.start else None
    end = pd.Timestamp(args.end, tz="UTC") if args.end else None
    for symbol in args.symbol or cfg.data.symbols:
        df = storage.read_candles(symbol, args.timeframe, start, end)
        report = check_candles(df, symbol, args.timeframe)
        print(render_markdown(report))
        path = write_report(report, data_dir / "reports")
        print(f"Отчёт сохранён: {path}\n")


if __name__ == "__main__":
    main()
