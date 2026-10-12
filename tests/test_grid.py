import pandas as pd
import pytest

from bot.core.config import load_config
from bot.research.grid import (
    count_variants,
    grid_summary,
    grid_variants,
    heatmap,
    parse_grid,
    render_grid,
    run_grid,
    to_markdown,
)

from .strategy_helpers import minute_walk


def test_parse_grid_values_and_count() -> None:
    grid = parse_grid(["stop_atr=1.5,2", "time_stop_bars=12,24,48"])
    assert grid == {"stop_atr": [1.5, 2], "time_stop_bars": [12, 24, 48]}
    assert count_variants(grid) == 6  # 2 × 3
    assert grid_variants(grid)[0] == {"stop_atr": 1.5, "time_stop_bars": 12}
    assert grid_variants(grid)[-1] == {"stop_atr": 2, "time_stop_bars": 48}
    assert parse_grid(None) == {}


def test_parse_grid_limits() -> None:
    with pytest.raises(ValueError, match="Не больше 3"):
        parse_grid(["a=1", "b=1", "c=1", "d=1"])
    with pytest.raises(ValueError, match="слишком много"):
        parse_grid(["a=" + ",".join(map(str, range(11))), "b=" + ",".join(map(str, range(10)))])
    with pytest.raises(ValueError, match="ключ=v1"):
        parse_grid(["a"])


def test_heatmap_and_summary_hand_example() -> None:
    table = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [10, 20, 10, 20],
            "trades": [150, 150, 50, 150],
            "profit_factor": [1.0, 1.3, 1.5, 1.25],
        }
    )
    hm = heatmap(table, "a", "b")
    assert hm.loc[1, 10] == 1.0
    assert hm.loc[2, 20] == 1.25
    # PF > 1.2 и ≥ 100 сделок: (1,20) и (2,20); (2,10) — мало сделок
    summary = grid_summary(table, 1.2, 100)
    assert "2 из 4" in summary
    assert "медиана 1.27" in summary  # (1.25 + 1.3) / 2 = 1.275
    md = to_markdown(hm, "a")
    assert md.splitlines()[0] == "| a | 10 | 20 |"
    assert md.splitlines()[2] == "| 1 | 1.00 | 1.30 |"


def test_run_grid_on_synthetic() -> None:
    cfg = load_config()
    candles = minute_walk(days=30)
    grid = parse_grid(["stop_atr=1,2", "tp_atr=2,3"])
    table = run_grid("random_entry", "1h", "BTCUSDT", candles, cfg, grid, {"seed": 5})
    assert len(table) == 4
    assert list(table.columns[:2]) == ["stop_atr", "tp_atr"]
    assert (table["trades"] > 0).all()
    text = render_grid(table, grid, "random_entry BTCUSDT 1h")
    assert "## PF: строки stop_atr, столбцы tp_atr" in text
    assert "Вариантов: 4" in text
