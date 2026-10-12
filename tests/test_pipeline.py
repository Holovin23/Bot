from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bot.backtest.engine import BacktestTrade
from bot.core.models import Side
from bot.research.pipeline import (
    Candidate,
    ResearchConfig,
    StrategySpace,
    add_trials,
    read_trials,
    render,
    run_pipeline,
    score,
    sr_variance,
)
from bot.research.robustness import trade_stats


def trade(r: float, i: int) -> BacktestTrade:
    t = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i)
    # риск 10 USDT, результат r × 10 (без комиссий: entry_fee = 0, цены подобраны)
    return BacktestTrade(
        signal_id=f"s{i}",
        symbol="BTCUSDT",
        side=Side.LONG,
        qty=1,
        signal_time=t,
        entry_time=t,
        entry_price_raw=100,
        entry_price=100,
        initial_stop=90,
        stop=90,
        take=None,
        risk_amount=10,
        entry_fee=0,
        exit_time=t,
        exit_price_raw=100 + 10 * r,
        exit_price=100 + 10 * r,
        exit_reason="stop",
    )


GOOD = [1.5, -1, 2, -1, 0.5, -1, 3, -1, 1, -1] * 5  # 50 сделок, средний R 0,3
FLAT = [1, -1] * 25  # 50 сделок, средний R 0


def research() -> ResearchConfig:
    return ResearchConfig(
        symbols=["BTCUSDT"],
        timeframes=["1h"],
        strategies={
            "good": StrategySpace(grid={"a": [1, 2, 3]}),
            "flat": StrategySpace(grid={"b": [1, 2]}),
        },
    )


def test_trials_count() -> None:
    assert research().trials() == 5  # 3 + 2 варианта
    assert len(research().combos()) == 2


def test_pipeline_scores_and_rejects() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def wf(name: str, symbol: str, tf: str, grid: dict[str, list[Any]], params: dict[str, Any]):  # type: ignore[no-untyped-def]
        calls.append((name, grid))
        values = GOOD if name == "good" else FLAT
        return [trade(r, i) for i, r in enumerate(values)]

    candidates, total = run_pipeline(
        research(), wf, 10_000, prior_trials=0, progress=lambda _: None
    )
    assert total == 5 and [c[0] for c in calls] == ["good", "flat"]
    good, flat = candidates
    assert good.stats.trades == 50 and good.stats.expectancy_r == pytest.approx(0.3)
    assert good.dsr > flat.dsr
    assert not flat.passed and any("Expectancy" in r for r in flat.reasons)
    text = render(candidates, total, 5, 0.95)
    assert "всего за все запуски: 5" in text and "good BTCUSDT 1h" in text


def test_more_prior_trials_raise_the_bar() -> None:
    def make() -> list[Candidate]:
        return [
            Candidate(
                "good",
                "BTCUSDT",
                "1h",
                1,
                GOOD,
                trade_stats([trade(r, i) for i, r in enumerate(GOOD)], 10_000),
            ),
            Candidate(
                "flat",
                "BTCUSDT",
                "1h",
                1,
                FLAT,
                trade_stats([trade(r, i) for i, r in enumerate(FLAT)], 10_000),
            ),
        ]

    few, many = make(), make()
    score(few, 2, 0.95)
    score(many, 10_000, 0.95)
    assert many[0].dsr < few[0].dsr
    assert many[0].p_bonferroni >= few[0].p_bonferroni


def test_sr_variance_fallback_uses_trade_count() -> None:
    c = Candidate("x", "BTCUSDT", "1h", 1, [1, -1] * 25, trade_stats([], 10_000))
    assert sr_variance([c]) == pytest.approx(1 / 50)


def test_trials_file_is_cumulative(tmp_path: Path) -> None:
    path = tmp_path / "trials.json"
    assert read_trials(path) == 0
    assert add_trials(path, 5, "первый") == 5
    assert add_trials(path, 7, "второй") == 12
    assert read_trials(path) == 12
