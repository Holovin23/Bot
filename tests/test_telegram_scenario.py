import pandas as pd
import pytest

from bot.scenarios.models import Scenario, Trigger
from bot.scenarios.store import ScenarioVersion
from bot.strategies.base import generate_signals
from bot.strategies.registry import STRATEGIES
from bot.strategies.telegram_scenario import TelegramScenario, find_fires

T0 = pd.Timestamp("2025-03-01 00:00", tz="UTC")


def bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Часовые свечи (open, high, low, close) с T0."""
    return pd.DataFrame(
        {
            "open_time": pd.date_range(T0, periods=len(rows), freq="1h"),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "volume": 1.0,
            "turnover": 1.0,
        }
    )


def flat(price: float, n: int) -> list[tuple[float, float, float, float]]:
    return [(price, price + 1, price - 1, price)] * n


def version(
    trigger: Trigger,
    hours: float = 2,
    side: str = "LONG",
    stop: float | None = None,
    targets: list[float] | None = None,
    sid: str = "s1",
    n: int = 1,
    until: float | None = None,
) -> ScenarioVersion:
    s = Scenario(symbol="BTCUSDT", side=side, trigger=trigger, stop=stop, targets=targets or [])  # type: ignore[arg-type]
    to = T0 + pd.Timedelta(hours=until) if until is not None else None
    return ScenarioVersion(sid, n, T0 + pd.Timedelta(hours=hours), to, False, s)


def fired_at(df: pd.DataFrame, versions: list[ScenarioVersion]) -> list[int]:
    fires, _ = find_fires(df, "1h", versions)
    return [f.bar_index for f in fires]


ABOVE = Trigger(type="close_above", level=100)


def test_close_above_fires_on_first_cross_after_post() -> None:
    # Пересечение на свече 1 (до поста в 02:00) не считается; первая подходящая свеча —
    # открытие 03:00 (02:00 + 5 минут на разбор), пересечение на свече 5.
    df = bars(
        [*flat(99, 1), (99, 101, 98, 101), (101, 101, 98, 99), *flat(99, 2), (99, 102, 98, 101.5)]
    )
    assert fired_at(df, [version(ABOVE)]) == [5]
    _, outcome = find_fires(df, "1h", [version(ABOVE)])
    assert outcome["s1"] == "сработал 2025-03-01 06:00 UTC"


def test_author_stop_before_entry_kills_scenario() -> None:
    df = bars([*flat(99, 4), (99, 99.5, 96.5, 98), (98, 102, 97.5, 101)])
    _, outcome = find_fires(df, "1h", [version(ABOVE, stop=97)])
    assert fired_at(df, [version(ABOVE, stop=97)]) == []
    assert outcome["s1"].startswith("цена дошла до стопа автора")


def test_first_target_before_entry_means_missed() -> None:
    zone = Trigger(type="touch_zone", zone_low=95, zone_high=96)
    df = bars([*flat(100, 4), (100, 104.5, 99, 104), (104, 104, 95.5, 96)])
    _, outcome = find_fires(df, "1h", [version(zone, targets=[104])])
    assert outcome["s1"].startswith("цена дошла до цели без входа")


def test_touch_zone_and_retest() -> None:
    zone = Trigger(type="touch_zone", zone_low=95, zone_high=96)
    df = bars([*flat(100, 4), (100, 100, 95.8, 97)])
    assert fired_at(df, [version(zone)]) == [4]
    # Цена уже выше 100 на момент поста → ретест: касание 100 и закрытие выше
    retest = Trigger(type="retest", level=100)
    df = bars([*flat(103, 4), (103, 103, 99.8, 100.5)])
    assert fired_at(df, [version(retest)]) == [4]
    # Закрытие ниже уровня — это не ретест, а пробой вниз
    df = bars([*flat(103, 4), (103, 103, 99, 99.5)])
    assert fired_at(df, [version(retest)]) == []


def test_latency_and_updated_version() -> None:
    # Пост в 03:58: свеча 04:00 начинается раньше 04:03 — она не проверяется
    df = bars([*flat(99, 4), (99, 102, 98, 101), (101, 101, 98, 99), (99, 102, 98, 101)])
    assert fired_at(df, [version(ABOVE, hours=3 + 58 / 60)]) == [6]
    # v1 (уровень 100) действует до 03:00, затем v2 (уровень 105)
    df = bars([*flat(99, 4), (99, 102, 98, 101), (101, 106, 100, 105.5)])
    v1 = version(ABOVE, hours=0, until=3)
    v2 = version(Trigger(type="close_above", level=105), hours=3, n=2)
    fires, _ = find_fires(df, "1h", [v1, v2])
    assert [(f.bar_index, f.version.version) for f in fires] == [(5, 2)]


def test_causal_on_truncated_history() -> None:
    rows = (
        [*flat(99, 3), (99, 102, 98, 101), (101, 101, 98, 99), *flat(99, 3), (99, 102, 98, 101)]
    )
    versions = [version(ABOVE, hours=0), version(ABOVE, hours=5, sid="s2")]
    full = fired_at(bars(rows), versions)
    for cut in range(1, len(rows) + 1):
        assert fired_at(bars(rows[:cut]), versions) == [i for i in full if i < cut]


def test_strategy_signal_uses_default_levels() -> None:
    df = bars([*flat(99, 30), (99, 101.5, 98.5, 101)])
    strategy = TelegramScenario("1h", versions=[version(ABOVE)])
    [signal] = generate_signals(strategy, "BTCUSDT", df)
    atr_value = strategy.prepare(df)["atr"].iloc[-1]
    assert signal.entry_price == 101
    assert signal.stop_loss == pytest.approx(100 - 0.5 * atr_value)
    risk = 101 - signal.stop_loss
    assert signal.take_profit == pytest.approx(101 + 2 * risk)
    assert signal.features["scenario_id"] == "s1"
    assert "стоп — по умолчанию" in signal.reason


def test_image_levels_wait_for_approval() -> None:
    df = bars([*flat(99, 30), (99, 101.5, 98.5, 101)])
    v = version(ABOVE)
    unapproved = ScenarioVersion(v.scenario_id, 1, v.valid_from, None, True, v.scenario)
    strategy = TelegramScenario("1h", versions=[unapproved])
    assert generate_signals(strategy, "BTCUSDT", df) == []
    assert strategy.notes == ["s1 v1: уровни только с картинки, ячейка не одобрена"]
    allowed = TelegramScenario("1h", {"allow_image_levels": True}, versions=[unapproved])
    assert len(generate_signals(allowed, "BTCUSDT", df)) == 1


def test_registered() -> None:
    assert STRATEGIES["telegram_scenario"] is TelegramScenario
