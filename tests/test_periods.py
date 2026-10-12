from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from bot.research.periods import (
    HOLDOUT,
    IN_SAMPLE,
    VALIDATION,
    HoldoutAlreadyOpenedError,
    holdout_opened,
    open_holdout,
    touches_holdout,
)


def test_periods_follow_each_other() -> None:
    assert IN_SAMPLE.end == VALIDATION.start
    assert VALIDATION.end == HOLDOUT.start
    assert IN_SAMPLE.label() == "in-sample 2021-01-01 — 2024-06-30"
    assert HOLDOUT.label() == "holdout 2025-07-01 — 2026-09-30"


def test_touches_holdout() -> None:
    assert not touches_holdout(IN_SAMPLE.start, VALIDATION.end)
    assert touches_holdout(VALIDATION.start, pd.Timestamp("2025-07-02", tz="UTC"))


def test_holdout_opens_once(tmp_path: Path) -> None:
    log = tmp_path / "research" / "holdout-log.md"
    now = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
    assert not holdout_opened(log, "breakout", "BTCUSDT", "1h")
    open_holdout(log, "breakout", "BTCUSDT", "1h", "take_r=2.0", now=now)
    assert holdout_opened(log, "breakout", "BTCUSDT", "1h")
    assert "| 2026-10-12 12:00 | breakout | BTCUSDT | 1h | take_r=2.0 |" in log.read_text()
    with pytest.raises(HoldoutAlreadyOpenedError):
        open_holdout(log, "breakout", "BTCUSDT", "1h", "take_r=2.0")
    # Другой таймфрейм — отдельный hold-out; force записывает второй взгляд
    open_holdout(log, "breakout", "BTCUSDT", "5m", "")
    open_holdout(log, "breakout", "BTCUSDT", "1h", "", force=True)
    assert log.read_text().count("| breakout | BTCUSDT | 1h |") == 2
