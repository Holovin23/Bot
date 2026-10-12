from datetime import UTC, date, datetime

import pandas as pd
import pytest

from bot.backtest.costs import CostModel
from bot.core.config import InstrumentConfig, LimitsConfig, RiskConfig
from bot.core.models import Side, Signal
from bot.risk.limits import (
    OpenRisk,
    RiskState,
    evaluate,
    liquidation_ok,
    record_trade,
    signal_checks,
    update_state,
)

LIMITS = LimitsConfig()
COSTS = CostModel(slippage={"BTCUSDT": 0.0003, "ETHUSDT": 0.0005})
BTC = InstrumentConfig(tick_size=0.1, qty_step=0.001, min_qty=0.001)
BAR = datetime(2026, 10, 12, 10, 0, tzinfo=UTC)


def signal(
    stop: float = 65000,
    tp: float | None = 66350,
    strategy: str = "breakout",
    side: Side = Side.LONG,
) -> Signal:
    return Signal.create(
        strategy=strategy,
        strategy_version="1",
        symbol="BTCUSDT",
        timeframe="1h",
        bar_time=BAR,
        side=side,
        entry_price=65450,
        stop_loss=stop,
        take_profit=tp,
    )


def state(equity: float = 10_000) -> RiskState:
    return RiskState(
        equity=equity, peak_equity=equity, day=date(2026, 10, 12), day_start_equity=equity
    )


def run(
    sig: Signal,
    open_risks: list[OpenRisk] | None = None,
    now: str = "2026-10-12 11:00:30",
    st: RiskState | None = None,
):  # type: ignore[no-untyped-def]
    return evaluate(
        sig,
        state=st or state(),
        open_risks=open_risks or [],
        atr_value=300,
        last_candle_open=pd.Timestamp(BAR),
        now=pd.Timestamp(now, tz="UTC"),
        entry_price=65450,
        instrument=BTC,
        costs=COSTS,
        risk=RiskConfig(),
        limits=LIMITS,
    )


def test_signal_checks_hand_examples() -> None:
    # Риск 450, цель 900: R:R 2; стоп 1,5 ATR; издержки 65450 × 2 × 0,00085 = 111,3 = 12% от 900
    assert signal_checks(signal(), 300, COSTS, LIMITS) == []
    # Цель 650: R:R 1,44 < 1,5
    assert signal_checks(signal(tp=66100), 300, COSTS, LIMITS)[0].startswith("2:")
    # Для Mean Reversion минимальный R:R 1,0 — та же цель проходит (издержки 111/650 = 17%)
    assert signal_checks(signal(tp=66100, strategy="mean_reversion"), 300, COSTS, LIMITS) == []
    # Цель 550: издержки 111/550 = 20,2% — уже не проходит проверку 12
    assert any(
        r.startswith("12:")
        for r in signal_checks(signal(tp=66000, strategy="mean_reversion"), 300, COSTS, LIMITS)
    )
    # Стоп 450 = 4,5 ATR при ATR 100 — дальше 4 ATR; 0,45 ATR при ATR 1000 — ближе 0,5
    assert any(r.startswith("3:") for r in signal_checks(signal(), 100, COSTS, LIMITS))
    assert any(r.startswith("3:") for r in signal_checks(signal(), 1000, COSTS, LIMITS))
    # Без тейка R:R не проверяется; потенциал 2 ATR = 600: издержки 111/600 = 18,5% < 20%
    assert signal_checks(signal(tp=None), 300, COSTS, LIMITS) == []
    # ATR 250: потенциал 500, издержки 22% — отклонить
    assert any(
        r.startswith("12:") for r in signal_checks(signal(tp=None, stop=65050), 250, COSTS, LIMITS)
    )


def test_sizing_and_correlation_hand_example() -> None:
    d = run(signal())
    # Риск 50 USDT; дистанция 450 + 65000 × 0,0006 = 489; 50 / 489 = 0,10225 → 0,102
    assert d.approved
    assert d.qty == pytest.approx(0.102)
    assert d.risk_amount == pytest.approx(0.102 * 489)
    eth_long = OpenRisk("x", "ETHUSDT", Side.LONG, 40.0)
    d = run(signal(), [eth_long])
    # 33,33 / 489 = 0,0682 → 0,068
    assert d.approved and d.qty == pytest.approx(0.068)
    assert d.notes and d.notes[0].startswith("8:")
    eth_short = OpenRisk("x", "ETHUSDT", Side.SHORT, 40.0)
    assert run(signal(), [eth_short]).qty == pytest.approx(0.102)


def test_rejections() -> None:
    sig = signal()
    assert any(
        r.startswith("5:")
        for r in run(sig, [OpenRisk(sig.signal_id, "BTCUSDT", Side.LONG, 1)]).reasons
    )
    two = [OpenRisk("a", "ETHUSDT", Side.SHORT, 1), OpenRisk("b", "SOLUSDT", Side.SHORT, 1)]
    assert any(r.startswith("6:") for r in run(sig, two).reasons)
    assert any(
        "по BTCUSDT" in r for r in run(sig, [OpenRisk("a", "BTCUSDT", Side.SHORT, 1)]).reasons
    )
    # 7: 100 + 49,9 = 149,9 ≤ 150 — можно; 101 + 49,9 > 150 — нельзя
    assert run(sig, [OpenRisk("a", "ETHUSDT", Side.SHORT, 100)]).approved
    assert run(sig, [OpenRisk("a", "ETHUSDT", Side.SHORT, 101)]).reasons[0].startswith("7:")
    # 13: свеча 10:00–11:00, сейчас 13:00:30 — прошло больше 2 часов
    assert any(r.startswith("13:") for r in run(sig, now="2026-10-12 13:00:30").reasons)
    halted = RiskState(**{**state().__dict__, "halted": "kill switch"})
    assert run(sig, st=halted).reasons == ["торговля остановлена: kill switch"]


def test_liquidation_hand_example() -> None:
    # 3x: ликвидация на 1/3 − 0,5% = 32,8% от входа; стоп 10% × 2 = 20% — можно; 20% × 2 — нельзя
    assert liquidation_ok(100, 90, 3.0, LIMITS)
    assert not liquidation_ok(100, 80, 3.0, LIMITS)


def test_state_limits_hand_example() -> None:
    s = state()
    s = update_state(s, 9_850, date(2026, 10, 12), LIMITS)
    assert s.paused is None  # −1,5%
    s = update_state(s, 9_800, date(2026, 10, 12), LIMITS)
    assert s.paused == "дневной убыток 2.0%"
    s = update_state(s, 9_800, date(2026, 10, 13), LIMITS)  # новый день — пауза снята
    assert s.paused is None and s.day_start_equity == 9_800
    s = update_state(s, 11_000, date(2026, 10, 14), LIMITS)
    s = update_state(s, 9_900, date(2026, 10, 15), LIMITS)  # −10% от пика 11 000
    assert s.halted is not None and "просадка" in s.halted
    s = update_state(s, 12_000, date(2026, 10, 16), LIMITS)
    assert s.halted is not None  # kill switch снимается только вручную


def test_loss_streak() -> None:
    s = state()
    for _ in range(5):
        s = record_trade(s, -10, LIMITS)
    assert s.paused is None and s.loss_streak == 5
    s = record_trade(s, -10, LIMITS)
    assert s.paused == "6 убытков подряд"
    assert record_trade(s, 5, LIMITS).loss_streak == 0


def test_limits_cannot_be_loosened() -> None:
    with pytest.raises(ValueError):
        LimitsConfig(daily_loss=0.05)
    with pytest.raises(ValueError):
        LimitsConfig(max_positions=5)
    with pytest.raises(ValueError):
        LimitsConfig(max_drawdown=0.2)
    assert LimitsConfig(daily_loss=0.01).daily_loss == 0.01  # ужесточить можно


def test_backtest_uses_the_same_signal_checks() -> None:
    from bot.backtest.engine import run_backtest
    from bot.strategies.random_entry import RandomEntry

    from .strategy_helpers import minute_walk

    candles = minute_walk(days=20)
    # Тейк 1 ATR при стопе 1,5 ATR: R:R 0,67 < 1,0 — живой бот такой сигнал не возьмёт
    strategy = RandomEntry("1h", {"seed": 2, "tp_atr": 1.0})
    kwargs = {"instrument": BTC, "costs": COSTS, "risk": RiskConfig()}
    plain = run_backtest(strategy, "BTCUSDT", candles, **kwargs)
    checked = run_backtest(strategy, "BTCUSDT", candles, limits=LIMITS, **kwargs)
    assert plain.trades
    assert not checked.trades
    assert all(s.reason.startswith("риск: 2:") for s in checked.skipped)


def test_live_profile_is_stricter_and_daily_loss_halts() -> None:
    from bot.core.config import Config, LiveRiskConfig, Secrets

    cfg = Config(secrets=Secrets(_env_file=None))  # type: ignore[call-arg]
    live = cfg.for_live()
    assert live.risk.risk_per_trade == 0.0025 and live.limits.max_positions == 1
    assert live.limits.daily_loss == 0.01 and live.limits.daily_loss_halts
    assert cfg.limits.daily_loss_halts is False  # демо-настройки не тронуты
    with pytest.raises(ValueError):
        LiveRiskConfig(risk_per_trade=0.005)  # больше 0,25% нельзя
    # Капитал 10 000 → 9 890 за день: −1,1% ≥ 1% → полная остановка, а не пауза
    st = update_state(state(), 9_890, date(2026, 10, 12), live.limits)
    assert st.halted is not None and "ручной" in st.halted and st.paused is None
    # На демо тот же убыток — ниже 2%, торговля продолжается
    assert update_state(state(), 9_890, date(2026, 10, 12), LIMITS).paused is None
