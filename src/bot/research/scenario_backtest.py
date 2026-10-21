"""Бэктест одного поста: пост и дата задаются вручную (docs/10, п. 4.4).

    # сценарий уже есть (ответ Claude Code или написан руками) — без затрат на ИИ:
    uv run python -m bot.research.scenario_backtest --scenario ответ.json \\
        --post пост.txt --date "2025-03-01 10:00"
    # разобрать пост через Claude API (план А):
    uv run python -m bot.research.scenario_backtest --post пост.txt --image график.png \\
        --date "2025-03-01 10:00"

--scenario — JSON по схеме разбора (как у `python -m bot.scenarios import`). Дата — время
публикации поста в UTC: сценарий начинает работать только после неё. Прогон идёт
--days дней (по умолчанию 30) обычным движком: комиссии, проскальзывание, funding,
проверки риск-менеджера. Отчёт сохраняется в data/backtests, как у любого бэктеста.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from bot.backtest.costs import CostModel
from bot.backtest.engine import BacktestResult, run_backtest
from bot.backtest.metrics import compute_metrics
from bot.backtest.report import ReportContext, save_report
from bot.core.config import Config, load_config
from bot.core.timeframes import timeframe_delta
from bot.data.storage import Storage
from bot.scenarios.models import PostScenarios, validate_scenario
from bot.scenarios.store import ScenarioVersion
from bot.strategies.telegram_scenario import TelegramScenario

WARMUP_BARS = 30  # история до поста для ATR(14)


@dataclass
class SymbolRun:
    symbol: str
    result: BacktestResult | None
    outcome: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    folder: Path | None = None
    error: str = ""


def post_versions(
    answer: PostScenarios, post_text: str, published: pd.Timestamp, symbols: list[str]
) -> tuple[list[ScenarioVersion], list[str]]:
    """Сценарии поста как версии ячеек с началом в момент публикации и причины отказов.
    Те же проверки, что при импорте в базу."""
    if answer.post_kind != "ANALYSIS":
        return [], [f"пост типа {answer.post_kind} — сценарии не создаются"]
    charts = sum(1 for img in answer.images if img.kind == "CHART")
    versions, rejected = [], []
    for n, s in enumerate(answer.new_scenarios, start=1):
        errors = validate_scenario(s, post_text, symbols, charts)
        if errors:
            rejected.append(f"{s.symbol} {s.side} {s.trigger.type}: {'; '.join(errors)}")
            continue
        sid = f"post-{s.symbol}-{published:%Y%m%d%H%M}-{n}"
        versions.append(ScenarioVersion(sid, 1, published, None, s.level_source == "IMAGE", s))
    return versions, rejected


def run_post(
    cfg: Config,
    storage: Storage,
    versions: list[ScenarioVersion],
    published: pd.Timestamp,
    days: float,
    timeframe: str,
    params: dict[str, object] | None = None,
    out_root: Path | None = None,
) -> list[SymbolRun]:
    runs = []
    end = published + pd.Timedelta(days=days)
    start = published - WARMUP_BARS * timeframe_delta(timeframe)
    for symbol in sorted({v.scenario.symbol for v in versions}):
        mine = [v for v in versions if v.scenario.symbol == symbol]
        if symbol not in cfg.instruments:
            runs.append(SymbolRun(symbol, None, error="нет параметров инструмента в конфиге"))
            continue
        candles = storage.read_candles(symbol, "1m", start, end)
        if candles.empty:
            runs.append(
                SymbolRun(symbol, None, error="нет 1m-свечей за период — загрузите историю")
            )
            continue
        strategy = TelegramScenario(timeframe, params, versions=mine)
        costs = CostModel.from_config(cfg.costs)
        result = run_backtest(
            strategy,
            symbol,
            candles,
            instrument=cfg.instruments[symbol],
            costs=costs,
            risk=cfg.risk,
            initial_capital=cfg.costs.initial_capital,
            funding=storage.read_funding(symbol, start, end),
            limits=cfg.limits,
        )
        run = SymbolRun(symbol, result, dict(strategy.outcome), list(strategy.notes))
        if out_root is not None:
            metrics = compute_metrics(result, candles["close"])
            ctx = ReportContext(start=start, end=end, costs=costs, risk=cfg.risk)
            run.folder = save_report(result, metrics, ctx, out_root)
        runs.append(run)
    return runs


def describe(run: SymbolRun) -> list[str]:
    lines = [f"{run.symbol}:"]
    if run.error:
        return [*lines, f"  {run.error}"]
    lines += [f"  {sid}: {text}" for sid, text in run.outcome.items()]
    lines += [f"  не вошли: {n}" for n in run.notes]
    assert run.result is not None
    lines += [f"  пропущен сигнал {s.signal_id}: {s.reason}" for s in run.result.skipped]
    for t in run.result.trades:
        r = t.net_pnl / t.risk_amount if t.risk_amount else 0.0
        lines.append(
            f"  сделка {t.side.value} вход {t.entry_time:%Y-%m-%d %H:%M} по {t.entry_price:.6g}, "
            f"стоп {t.initial_stop:.6g}, цель {t.take if t.take is not None else '—'}; "
            f"выход {t.exit_time:%Y-%m-%d %H:%M} ({t.exit_reason}) → {r:+.2f}R, "
            f"{t.net_pnl:+.2f} USDT"
        )
    if run.folder is not None:
        lines.append(f"  отчёт: {run.folder}")
    return lines


def backtest_post(
    cfg: Config,
    text: str,
    published: pd.Timestamp,
    *,
    answer_json: str | None = None,
    images: list[Path] | None = None,
    days: float = 30,
    timeframe: str = "1h",
    allow_image_levels: bool = False,
    model: str | None = None,
    say: Callable[[str], None] = print,
) -> list[SymbolRun]:
    """Весь бэктест поста: разбор (готовый JSON или Claude API), проверки, прогон.
    Общий код для командной строки и веб-интерфейса. Ошибка ввода — ValueError."""
    symbols = cfg.scenarios.symbols
    if answer_json:
        try:
            answer = PostScenarios.model_validate_json(answer_json)
        except ValidationError as exc:
            raise ValueError(f"Ответ разбора не по схеме: {exc}") from None
    else:
        key = cfg.secrets.anthropic_api_key.get_secret_value()
        if not key:
            raise ValueError("Нет ANTHROPIC_API_KEY: передайте готовый ответ разбора (план Б)")
        import anthropic

        from bot.scenarios.parser import parse_with_api

        outcome = parse_with_api(
            anthropic.Anthropic(api_key=key),
            model or cfg.scenarios.model,
            symbols,
            published.isoformat(),
            text,
            images or [],
            [],
        )
        if outcome.result is None:
            raise ValueError(f"Пост не разобран: {outcome.error}")
        answer = outcome.result
        say(f"Токены: {outcome.usage}")
    say(f"Разбор: {answer.post_kind}. {answer.summary_ru}")
    versions, rejected = post_versions(answer, text, published, symbols)
    for line in rejected:
        say(f"отклонено: {line}")
    if not versions:
        raise ValueError("Нет сценариев для бэктеста.")
    params: dict[str, object] = {"allow_image_levels": allow_image_levels}
    storage = Storage(cfg.resolve(cfg.data.dir))
    out = cfg.resolve(cfg.data.dir) / "backtests"
    runs = run_post(cfg, storage, versions, published, days, timeframe, params, out)
    for run in runs:
        for line in describe(run):
            say(line)
    return runs


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Бэктест одного поста")
    parser.add_argument("--post", required=True, help="файл с текстом поста")
    parser.add_argument("--image", action="append", default=[], help="картинка поста")
    parser.add_argument("--date", required=True, help='время публикации UTC, "2025-03-01 10:00"')
    parser.add_argument("--scenario", help="готовый ответ разбора (JSON) — без Claude API")
    parser.add_argument("--days", type=float, default=30)
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--model", default=None)
    parser.add_argument("--allow-image-levels", action="store_true")
    args = parser.parse_args(argv)
    try:
        backtest_post(
            cfg,
            Path(args.post).read_text(encoding="utf-8").strip(),
            pd.Timestamp(args.date, tz="UTC"),
            answer_json=Path(args.scenario).read_text(encoding="utf-8") if args.scenario else None,
            images=[Path(p) for p in args.image],
            days=args.days,
            timeframe=args.timeframe,
            allow_image_levels=args.allow_image_levels,
            model=args.model,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
