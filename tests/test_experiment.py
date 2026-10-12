import dataclasses
from pathlib import Path

from bot.core.config import load_config
from bot.research.baseline import BaselineComparison
from bot.research.experiment import (
    preliminary_verdict,
    render_baselines,
    run_experiment,
)
from bot.research.journal import JournalEntry, append_entry, next_number
from bot.strategies.random_entry import RandomEntry

from .strategy_helpers import minute_walk as walk


def entry(title: str = "x") -> JournalEntry:
    return JournalEntry(
        title=title,
        date="2026-10-12",
        hypothesis="h",
        rationale="r",
        period="p",
        params="—",
        variants=1,
        criteria="c",
        result="res",
        verdict="v",
        report="rep",
    )


def test_next_number_takes_max_plus_one() -> None:
    assert next_number("") == 1
    assert next_number("## EXP-003  a\n...\n## EXP-007  b\n") == 8


def test_append_entry_creates_journal_and_numbers(tmp_path: Path) -> None:
    path = tmp_path / "research" / "experiments.md"
    assert append_entry(path, entry("первая")) == 1
    assert append_entry(path, entry("вторая")) == 2
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Журнал экспериментов")
    assert "## EXP-001  первая" in text
    assert "## EXP-002  вторая" in text
    assert "Проверено вариантов: 1" in text


def test_experiment_on_synthetic_data() -> None:
    cfg = load_config()
    candles = walk(days=60)
    exp = run_experiment(RandomEntry("1h", {"seed": 99}), "BTCUSDT", candles, cfg, runs=10)
    assert exp.lookahead == []
    assert exp.comparison is not None
    assert len(exp.comparison.profit_factors) == 10
    # 60 дней по 24 свечи с вероятностью 0.05 — около 60 сделок, меньше 100
    assert exp.verdict == "отклонить"
    assert any("мало сделок" in r for r in exp.reasons)
    text = render_baselines(exp)
    assert "## Сравнение с базовыми линиями" in text
    assert "Buy & hold" in text


def test_verdict_hand_example() -> None:
    cfg = load_config()
    exp = run_experiment(RandomEntry("1h", {"seed": 99}), "BTCUSDT", walk(days=60), cfg, runs=4)
    good = dataclasses.replace(exp.metrics, trades=150, profit_factor=1.5)
    better = BaselineComparison(
        random_params={},
        runs=4,
        profit_factors=[0.8, 0.9, 1.0, 1.1],
        expectancies_r=[0.0] * 4,
        strategy_profit_factor=1.5,
    )
    assert preliminary_verdict(good, better, []) == ("кандидат", [])
    # 3 из 4 случайных ниже 1.05 — 75% < 95%
    worse = dataclasses.replace(better, strategy_profit_factor=1.05)
    verdict, reasons = preliminary_verdict(good, worse, ["x"])
    assert verdict == "отклонить"
    assert len(reasons) == 2
    weak = dataclasses.replace(good, profit_factor=1.2)  # ровно 1.2 — не больше порога
    assert preliminary_verdict(weak, better, [])[0] == "отклонить"
