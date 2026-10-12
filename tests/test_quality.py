from pathlib import Path

import pandas as pd

from bot.data.quality import check_candles, write_report


def flat_candles(start: str, count: int) -> pd.DataFrame:
    times = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": times,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 5.0,
            "turnover": 500.0,
        }
    )


def test_clean_data() -> None:
    report = check_candles(flat_candles("2024-03-01", 100), "BTCUSDT", "1m")
    assert report.is_clean
    assert report.rows == 100


def test_detects_every_problem(tmp_path: Path) -> None:
    df = flat_candles("2024-03-01 00:00", 60)
    # Пропуск: удаляем 00:10, 00:11, 00:12 (3 свечи) и 00:40 (1 свеча)
    df = df[~df["open_time"].dt.minute.isin([10, 11, 12, 40])].reset_index(drop=True)
    # Дубль 00:20
    df = pd.concat([df, df[df["open_time"].dt.minute == 20]], ignore_index=True)
    # Нулевой объём 00:30
    df.loc[df["open_time"].dt.minute == 30, "volume"] = 0.0
    # high < low в 00:45
    df.loc[df["open_time"].dt.minute == 45, ["high", "low"]] = [98.0, 99.5]
    # Выброс в 00:50: ATR ≈ 2 (high - low), low 70 — это 30 от close, больше 10 * 2 = 20
    df.loc[df["open_time"].dt.minute == 50, "low"] = 70.0

    report = check_candles(df, "BTCUSDT", "1m")

    assert [(g.start.minute, g.end.minute, g.missing) for g in report.gaps] == [
        (10, 12, 3),
        (40, 40, 1),
    ]
    assert report.missing_total == 4
    assert [t.minute for t in report.duplicates] == [20]
    assert [t.minute for t in report.zero_volume] == [30]
    assert [t.minute for t in report.high_below_low] == [45]
    assert [t.minute for t in report.outliers] == [50]

    path = write_report(report, tmp_path)
    assert path.name == "quality_BTCUSDT.md"
    text = path.read_text(encoding="utf-8")
    assert "| 2024-03-01 00:10 | 2024-03-01 00:12 | 3 |" in text
    assert "Пропущено свечей: 4 в 2 интервалах" in text


def test_empty_data() -> None:
    report = check_candles(flat_candles("2024-03-01", 0), "BTCUSDT", "1m")
    assert report.rows == 0
