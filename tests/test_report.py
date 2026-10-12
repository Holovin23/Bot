from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bot.backtest import report
from bot.backtest.costs import CostModel
from bot.backtest.metrics import compute_metrics
from bot.backtest.report import ReportContext, _parse_params, render_report
from bot.core.config import RiskConfig, load_config
from bot.data.storage import Storage

from .test_engine import LONG, T0, Scripted, minutes, run


def test_trade_explanation_has_hand_numbers() -> None:
    result = run(Scripted(LONG), minutes(changes={12: (100, 105, 100, 104)}))
    ctx = ReportContext(
        start=T0,
        end=T0 + pd.Timedelta(hours=1),
        costs=CostModel(taker_fee=0.0005, slippage={"TEST": 0.001}),
        risk=RiskConfig(),
    )
    text = render_report(result, compute_metrics(result), ctx)
    assert "100.0000 × (1 + 0.001) = 100.1000" in text
    assert "капитал 10 000.00 × 0.005 = 50.00 USDT риска" in text
    assert "= 2.2960" in text  # дистанция 2.1 + 0.196
    assert "= 21.777." in text
    assert "103.8960" in text  # выход по тейку 104 × 0.999
    assert "| Сделок | 1 |" in text


def test_parse_params() -> None:
    assert _parse_params(["probability=0.1", "tp_atr=null", "seed=3"]) == {
        "probability": 0.1,
        "tp_atr": None,
        "seed": 3,
    }
    with pytest.raises(ValueError):
        _parse_params(["oops"])


def test_command_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 20 дней 1m-свечей случайного блуждания
    n = 20 * 1440
    rng = np.random.default_rng(1)
    close = 40_000 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    open_ = np.r_[close[0], close[:-1]]
    candles = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
            "open": open_,
            "high": np.maximum(open_, close) * 1.0003,
            "low": np.minimum(open_, close) * 0.9997,
            "close": close,
            "volume": 1.0,
            "turnover": close,
        }
    )
    Storage(tmp_path).write_candles("BTCUSDT", "1m", candles)
    cfg = load_config(env_file=None)
    cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update={"dir": tmp_path})})
    monkeypatch.setattr(report, "load_config", lambda: cfg)

    report.main(
        [
            "--strategy", "random_entry", "--symbol", "BTCUSDT", "--timeframe", "1h",
            "--from", "2024-01-01", "--to", "2024-01-21", "--param", "probability=0.2",
        ]
    )  # fmt: skip

    (folder,) = (tmp_path / "backtests").iterdir()
    text = (folder / "report.md").read_text(encoding="utf-8")
    assert "# Бэктест random_entry 1.0.0 — BTCUSDT 1h" in text
    assert "probability=0.2" in text
    trades = pd.read_csv(folder / "trades.csv")
    assert len(trades) > 10
    assert text.count("### ") == 10  # 10 сделок для ручной проверки
    equity = pd.read_csv(folder / "equity.csv")
    assert len(equity) > 400
