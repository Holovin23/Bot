import pandas as pd
import pytest

from bot.data.resample import resample_candles


def minutes(start: str, rows: list[tuple[float, float, float, float, float]]) -> pd.DataFrame:
    times = pd.date_range(start, periods=len(rows), freq="1min", tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])
    df.insert(0, "open_time", times)
    df["turnover"] = df["close"] * df["volume"]
    return df


# 12 минутных свечей 10:00..10:11: две полные 5m-свечи и неполная третья (10:10, 10:11).
ROWS = [
    # open, high, low, close, volume
    (100, 102, 99, 101, 1),  # 10:00
    (101, 105, 100, 104, 2),  # 10:01  <- максимум первой 5m
    (104, 104, 98, 99, 3),  # 10:02  <- минимум первой 5m
    (99, 100, 99, 100, 1),  # 10:03
    (100, 101, 99, 100.5, 1),  # 10:04  <- close первой 5m
    (100.5, 101, 100, 101, 1),  # 10:05
    (101, 110, 101, 109, 5),  # 10:06
    (109, 109, 95, 96, 5),  # 10:07
    (96, 97, 95, 97, 1),  # 10:08
    (97, 98, 96, 97.5, 1),  # 10:09
    (97.5, 99, 97, 98, 1),  # 10:10  неполный период
    (98, 99, 97, 98.5, 1),  # 10:11
]


def test_5m_by_hand() -> None:
    out = resample_candles(minutes("2024-03-01 10:00", ROWS), "5m")
    assert list(out["open_time"].dt.strftime("%H:%M")) == ["10:00", "10:05"]
    first, second = out.iloc[0], out.iloc[1]
    # 10:00: open 100 (первая), high 105, low 98, close 100.5 (последняя), volume 1+2+3+1+1 = 8
    assert (first.open, first.high, first.low, first.close, first.volume) == (
        100,
        105,
        98,
        100.5,
        8,
    )
    # 10:05: open 100.5, high 110, low 95, close 97.5, volume 1+5+5+1+1 = 13
    assert (second.open, second.high, second.low, second.close, second.volume) == (
        100.5,
        110,
        95,
        97.5,
        13,
    )
    # turnover — сумма: 101*1 + 104*2 + 99*3 + 100*1 + 100.5*1 = 806.5
    assert first.turnover == pytest.approx(806.5)


def test_incomplete_last_period_is_dropped() -> None:
    out = resample_candles(minutes("2024-03-01 10:00", ROWS), "15m")
    assert out.empty  # 12 минут — 15m-свеча ещё не закрыта


def test_partial_first_period_is_dropped() -> None:
    # Данные начинаются с 10:03: свеча 10:00 неполная, 10:05 полная
    out = resample_candles(minutes("2024-03-01 10:03", ROWS[3:10]), "5m")
    assert list(out["open_time"].dt.strftime("%H:%M")) == ["10:05"]


def test_1h_and_4h_align_to_utc_midnight() -> None:
    rows = [(1.0, 2.0, 0.5, 1.5, 1.0)] * (8 * 60)
    df = minutes("2024-03-01 00:00", rows)
    hours = resample_candles(df, "1h")
    assert len(hours) == 8
    assert hours["volume"].tolist() == [60.0] * 8
    four = resample_candles(df, "4h")
    assert list(four["open_time"].dt.strftime("%H:%M")) == ["00:00", "04:00"]


def test_gap_inside_period_still_aggregates() -> None:
    df = minutes("2024-03-01 10:00", ROWS[:10]).drop(index=[2])  # нет свечи 10:02
    out = resample_candles(df, "5m")
    assert len(out) == 2
    assert out.iloc[0].low == 99  # минимум 98 был в пропавшей свече
    assert out.iloc[0].volume == 5


def test_with_funding_rate_uses_rate_known_at_bar_close() -> None:
    from bot.data.resample import with_funding_rate

    bars = pd.DataFrame(
        {"open_time": pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC"), "close": 1.0}
    )
    funding = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00"], utc=True),
            "funding_rate": [0.0001, 0.0005],
        }
    )
    # Закрытия 04:00, 08:00, 12:00: ставка 00:00, затем выплаченная ровно в 08:00
    out = with_funding_rate(bars, funding, "4h")
    assert out["funding_rate"].tolist() == [0.0001, 0.0005, 0.0005]
    assert with_funding_rate(bars, None, "4h")["funding_rate"].isna().all()
