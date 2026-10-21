"""Локальный веб-интерфейс: графики стратегий, бэктесты, загрузка данных, paper-боты
и все команды терминала.

Запуск:
    uv run python -m bot.app.api
Открыть: http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bot.app.backtests import (
    FOLDER_RE,
    list_backtests,
    load_backtest,
    load_equity,
    load_report,
)
from bot.app.bot_status import bot_status, request_kill, request_resume
from bot.app.charts import (  # chart_bars — для совместимости со старыми импортами
    build_chart_data,
    build_strategy_chart,
    chart_bars,
)
from bot.app.commands import CommandRunner, catalog
from bot.app.jobs import Job, JobRunner, RunFn
from bot.app.refresh import CandleRefresher
from bot.app.scenarios import MEDIA_RE, cells, pipeline_status, posts_feed
from bot.app.symbols import SymbolBook, coverage
from bot.core.db import connect, db_path
from bot.core.journal import read_journal
from bot.data.downloader import Downloader
from bot.data.storage import Storage
from bot.research.experiment import IN_SAMPLE_END, IN_SAMPLE_START
from bot.research.periods import HOLDOUT, touches_holdout
from bot.runner.host import BotHost
from bot.scenarios.store import approve
from bot.strategies.registry import STRATEGIES, create_strategy

__all__ = ["chart_bars", "create_app", "main"]

STATIC_DIR = Path(__file__).parent / "static"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
# Файлы бэктеста, которые можно скачать из интерфейса (для разбора и для ИИ)
BACKTEST_FILES = {
    "report.md",
    "trades.csv",
    "trades.jsonl",
    "skipped.jsonl",
    "equity.csv",
    "run.json",
    "validation.md",
}
JOURNAL_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,120}\.jsonl$")

DownloaderFactory = Callable[[Callable[[str], None]], Downloader]
# Бэктест поста из вкладки «Сценарии»: запрос и вывод в журнал задачи → итог одной строкой
PostBacktest = Callable[[dict[str, Any], Callable[[str], None]], str]


class RunRequest(BaseModel):
    strategy: str
    symbol: str
    timeframe: str = "1h"
    start: str = IN_SAMPLE_START
    end: str = IN_SAMPLE_END
    params: dict[str, Any] = Field(default_factory=dict)
    # Период задевает hold-out: запуск только с явного согласия, он запишется в
    # research/holdout-log.md (hold-out после этого считается открытым).
    allow_holdout: bool = False
    # Перед бэктестом догрузить с Bybit недостающие свечи и funding за период
    download: bool = True


class DownloadRequest(BaseModel):
    symbols: list[str]
    start: str
    end: str | None = None
    open_interest: bool = False


class SymbolRequest(BaseModel):
    symbol: str


class BotRequest(BaseModel):
    strategy: str
    symbol: str
    timeframe: str = "1h"
    params: dict[str, Any] = Field(default_factory=dict)
    capital: float = Field(default=10_000, gt=0)


class CommandRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class PostBacktestRequest(BaseModel):
    text: str = Field(min_length=1)
    date: str  # время публикации, UTC
    # Готовый ответ разбора (план Б). Пусто — разбор через Claude API (план А).
    answer_json: str = ""
    post_id: int | None = None  # пост из ленты: взять его картинки для Claude API
    days: float = Field(default=30, gt=0, le=365)
    timeframe: str = "1h"
    allow_image_levels: bool = False


def parse_period(start: str, end: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Даты из формы (ГГГГ-ММ-ДД, end включительно) → [начало, конец). Ошибка — 400."""
    try:
        begin = pd.Timestamp(start, tz="UTC")
        stop = (
            pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            if end
            else pd.Timestamp.now(tz="UTC").ceil("1min")
        )
    except (ValueError, TypeError):
        raise HTTPException(400, f"Неверная дата: {start!r} — {end!r}") from None
    if pd.isna(begin) or pd.isna(stop):
        raise HTTPException(400, "Укажите обе даты периода")
    if stop <= begin:
        raise HTTPException(400, "Дата «по» раньше даты «с»")
    return begin, stop


def strategy_catalog() -> list[dict[str, Any]]:
    """Стратегии, их параметры по умолчанию и индикаторы для графика."""
    out = []
    for name, cls in sorted(STRATEGIES.items()):
        strategy = cls("1h")
        out.append(
            {
                "name": name,
                "version": cls.version,
                "params": strategy.params,
                "description": _doc(cls),
                "indicators": [s.label for s in strategy.chart_series()],
            }
        )
    return out


def _doc(cls: type) -> str:
    """Описание стратегии — docstring её модуля."""
    module = sys.modules.get(cls.__module__)
    text = (module.__doc__ if module is not None else None) or ""
    return text.strip()


def telegram_feed(data_dir: Path, limit: int) -> list[dict[str, Any]]:
    """Последние посты канала и их разбор ИИ (последний удачный, любой моделью)."""
    path = db_path(data_dir)
    if not path.exists():
        return []
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT p.channel, p.post_id, p.published_at, p.text, "
            "(SELECT r.result_json FROM post_parses r WHERE r.channel = p.channel "
            " AND r.post_id = p.post_id AND r.status = 'parsed' "
            " ORDER BY r.created_at DESC LIMIT 1) AS result_json "
            "FROM telegram_posts p ORDER BY p.published_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    feed = []
    for row in rows:
        parsed = json.loads(row["result_json"]) if row["result_json"] else None
        feed.append(
            {
                "channel": row["channel"],
                "post_id": row["post_id"],
                "time": int(pd.Timestamp(row["published_at"]).timestamp()),
                "text": row["text"],
                "summary": parsed.get("summary_ru") if parsed else None,
                "views": [
                    {"symbol": i["symbol"], "bias": i["bias"], "horizon": i["horizon"]}
                    for i in (parsed or {}).get("instruments", [])
                ],
            }
        )
    return feed


def create_app(
    storage: Storage,
    symbols: list[str],
    backtests_dir: Path | None = None,
    run: RunFn | None = None,
    refresher: CandleRefresher | None = None,
    downloader_factory: DownloaderFactory | None = None,
    symbol_book: SymbolBook | None = None,
    host: BotHost | None = None,
    jobs: JobRunner | None = None,
    commands: CommandRunner | None = None,
    journal_dir: Path | None = None,
    post_backtest: PostBacktest | None = None,
    channel: str | None = None,
    scenario_symbols: list[str] | None = None,
) -> FastAPI:
    """channel и scenario_symbols — канал и белый список монет сценариев (для «Старта»)."""
    app = FastAPI(title="Bot")
    backtests_root = backtests_dir if backtests_dir is not None else storage.root / "backtests"
    jobs = jobs or JobRunner()

    def check_symbol(symbol: str, code: int = 400) -> None:
        if symbol not in symbols:
            raise HTTPException(code, f"Неизвестный символ {symbol}: добавьте его в «Данных»")

    def check_timeframe(timeframe: str) -> None:
        if timeframe not in TIMEFRAMES:
            raise HTTPException(400, f"Таймфрейм должен быть одним из {TIMEFRAMES}")

    def check_strategy(name: str) -> None:
        if name not in STRATEGIES:
            raise HTTPException(400, f"Неизвестная стратегия {name}")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # --- монеты и данные ---------------------------------------------------------------

    @app.get("/api/symbols")
    def get_symbols() -> dict[str, Any]:
        local = sorted(symbol_book.local()) if symbol_book is not None else []
        return {"symbols": symbols, "timeframes": TIMEFRAMES, "local": local}

    @app.post("/api/symbols")
    def add_symbol(request: SymbolRequest) -> dict[str, Any]:
        if symbol_book is None:
            raise HTTPException(503, "Добавление монет не настроено")
        try:
            return symbol_book.add(request.symbol)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from None
        except Exception as exc:  # нет интернета, Bybit недоступен
            raise HTTPException(502, f"Не удалось спросить Bybit: {type(exc).__name__}") from None

    @app.delete("/api/symbols/{symbol}")
    def delete_symbol(symbol: str) -> dict[str, str]:
        if symbol_book is None:
            raise HTTPException(503, "Добавление монет не настроено")
        try:
            symbol_book.remove(symbol)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"status": f"{symbol} убран из списка (скачанные данные остались в data/)"}

    @app.get("/api/data/coverage")
    def get_coverage() -> dict[str, Any]:
        return {"coverage": [coverage(storage, s) for s in symbols]}

    @app.post("/api/data/download")
    def download(request: DownloadRequest) -> dict[str, Any]:
        if downloader_factory is None:
            raise HTTPException(503, "Загрузка данных не настроена")
        if not request.symbols:
            raise HTTPException(400, "Выберите хотя бы одну монету")
        for symbol in request.symbols:
            check_symbol(symbol)
        begin, stop = parse_period(request.start, request.end)
        names = ", ".join(request.symbols)

        def work(job: Job) -> str:
            loader = downloader_factory(job.say)
            total = 0
            for symbol in request.symbols:
                total += loader.ensure_history(symbol, begin, stop)
                if request.open_interest:
                    loader.download_open_interest(symbol, begin)
            return f"новых 1m-свечей: {total}"

        title = f"Загрузка {names} {begin:%Y-%m-%d} — {stop - pd.Timedelta(days=1):%Y-%m-%d}"
        try:
            job = jobs.start("download", title, work, request.model_dump())
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"job_id": job.job_id}

    @app.get("/api/candles")
    def get_candles(
        symbol: str,
        timeframe: str = "1h",
        limit: int = Query(default=500, ge=10, le=5000),
        until: int | None = Query(default=None, ge=0),
        start: int | None = Query(default=None, ge=0),
        end: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        check_symbol(symbol, 404)
        check_timeframe(timeframe)
        data = build_chart_data(storage, symbol, timeframe, limit, until, start, end)
        data["refreshing"], data["refresh_error"] = False, None
        if refresher is not None and until is None and start is None:
            state = refresher.request(symbol)
            data["refreshing"], data["refresh_error"] = state.running, state.error
        return data

    @app.get("/api/strategy-chart")
    def get_strategy_chart(
        strategy: str,
        symbol: str,
        timeframe: str = "1h",
        params: str = "{}",
        limit: int = Query(default=500, ge=10, le=5000),
        until: int | None = Query(default=None, ge=0),
        start: int | None = Query(default=None, ge=0),
        end: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        check_strategy(strategy)
        check_symbol(symbol, 404)
        check_timeframe(timeframe)
        try:
            values = json.loads(params)
            if not isinstance(values, dict):
                raise ValueError("параметры должны быть словарём")
            data = build_strategy_chart(
                storage, strategy, symbol, timeframe, values, limit, until, start, end
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"Параметры стратегии: {exc}") from None
        data["refreshing"], data["refresh_error"] = False, None
        if refresher is not None and until is None and start is None:
            state = refresher.request(symbol)
            data["refreshing"], data["refresh_error"] = state.running, state.error
        return data

    # --- бэктесты ----------------------------------------------------------------------

    @app.get("/api/backtests")
    def get_backtests() -> dict[str, list[dict[str, Any]]]:
        return {"backtests": list_backtests(backtests_root)}

    @app.get("/api/backtests/{backtest_id}")
    def get_backtest(backtest_id: str) -> dict[str, Any]:
        data = load_backtest(backtests_root, backtest_id)
        if data is None:
            raise HTTPException(404, f"Нет бэктеста {backtest_id}")
        return data

    @app.get("/api/backtests/{backtest_id}/equity")
    def get_equity(backtest_id: str) -> dict[str, Any]:
        points = load_equity(backtests_root, backtest_id)
        if points is None:
            raise HTTPException(404, f"Нет бэктеста {backtest_id}")
        return {"equity": points}

    @app.get("/api/backtests/{backtest_id}/report")
    def get_report(backtest_id: str) -> dict[str, str]:
        text = load_report(backtests_root, backtest_id)
        if text is None:
            raise HTTPException(404, f"Нет бэктеста {backtest_id}")
        return {"report": text}

    @app.get("/api/backtests/{backtest_id}/files/{name}")
    def get_backtest_file(backtest_id: str, name: str) -> FileResponse:
        path = backtests_root / backtest_id / name
        if FOLDER_RE.match(backtest_id) is None or name not in BACKTEST_FILES:
            raise HTTPException(404, "Нет такого файла")
        if not path.is_file():
            raise HTTPException(404, "Нет такого файла (бэктест сделан до появления журналов)")
        return FileResponse(path, filename=f"{backtest_id}_{name}")

    @app.post("/api/backtests/run")
    def run_backtest_job(request: RunRequest) -> dict[str, Any]:
        if run is None:
            raise HTTPException(503, "Запуск бэктестов не настроен")
        check_strategy(request.strategy)
        check_symbol(request.symbol)
        check_timeframe(request.timeframe)
        begin, stop = parse_period(request.start, request.end)
        try:
            create_strategy(request.strategy, request.timeframe, request.params)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"Параметры стратегии: {exc}") from None
        if touches_holdout(begin, stop) and not request.allow_holdout:
            last = HOLDOUT.end - pd.Timedelta(days=1)
            raise HTTPException(
                409,
                {
                    "holdout": True,
                    "message": (
                        f"Период задевает hold-out ({HOLDOUT.start:%Y-%m-%d} — {last:%Y-%m-%d}). "
                        "Это последняя честная проверка стратегии: её смотрят один раз, в конце. "
                        "Если всё равно запустить, запуск запишется в research/holdout-log.md."
                    ),
                },
            )
        body = request.model_dump()
        title = (
            f"Бэктест {request.strategy} {request.symbol} {request.timeframe} "
            f"{request.start} — {request.end}"
        )
        try:
            job = jobs.start("backtest", title, lambda job: run(body, job.say), body)
        except RuntimeError:
            raise HTTPException(409, "Уже идёт другой бэктест — дождитесь окончания") from None
        return {"job_id": job.job_id}

    # --- фоновые задачи и команды ------------------------------------------------------

    @app.get("/api/jobs")
    def get_jobs(kind: str | None = None) -> dict[str, Any]:
        items = [j.summary() for j in jobs.list(100) if kind is None or j.kind == kind]
        return {"jobs": items}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, log: int = Query(default=0, ge=0, le=5000)) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Нет такого запуска")
        return job.summary(log_tail=log)

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, str]:
        if commands is None or not commands.stop(job_id):
            raise HTTPException(404, "Эту задачу нельзя остановить (или она уже закончилась)")
        return {"status": "остановка отправлена"}

    @app.get("/api/commands")
    def get_commands() -> dict[str, Any]:
        return {"commands": catalog(symbols)}

    @app.post("/api/commands/{command_id}")
    def run_command(command_id: str, request: CommandRequest) -> dict[str, Any]:
        if commands is None:
            raise HTTPException(503, "Запуск команд не настроен")
        try:
            job = commands.start(command_id, request.values)
        except KeyError:
            raise HTTPException(404, f"Нет команды {command_id}") from None
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from None
        return {"job_id": job.job_id}

    # --- журналы (JSONL) ---------------------------------------------------------------

    def journal_path(name: str) -> Path:
        if journal_dir is None or not JOURNAL_RE.match(name) or ".." in name:
            raise HTTPException(404, "Нет такого журнала")
        path = journal_dir / name
        if not path.is_file():
            raise HTTPException(404, "Журнал пока пуст")
        return path

    @app.get("/api/journals")
    def get_journals() -> dict[str, Any]:
        if journal_dir is None or not journal_dir.is_dir():
            return {"journals": []}
        files = sorted(journal_dir.glob("*.jsonl"))
        return {
            "dir": str(journal_dir),
            "journals": [{"name": p.name, "size": p.stat().st_size} for p in files],
        }

    @app.get("/api/journals/{name}")
    def get_journal(name: str, limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
        return {"name": name, "records": read_journal(journal_path(name), limit)}

    @app.get("/api/journals/{name}/download")
    def download_journal(name: str) -> FileResponse:
        return FileResponse(journal_path(name), filename=name)

    # --- Telegram ----------------------------------------------------------------------

    @app.get("/api/telegram")
    def get_telegram(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        return {"posts": telegram_feed(storage.root, limit)}

    # --- сценарии из Telegram-канала (docs/10) -----------------------------------------

    @app.get("/api/scenarios")
    def get_scenarios(
        limit: int = Query(default=50, ge=1, le=500), cancelled: bool = False
    ) -> dict[str, Any]:
        return {
            "cells": cells(storage.root, cancelled),
            "posts": posts_feed(storage.root, limit),
        }

    @app.get("/api/scenarios/status")
    def get_scenarios_status() -> dict[str, Any]:
        return {
            **pipeline_status(storage.root, channel),
            "channel": channel,
            "symbols": scenario_symbols if scenario_symbols is not None else symbols,
        }

    @app.post("/api/scenarios/backtest")
    def run_post_backtest(request: PostBacktestRequest) -> dict[str, Any]:
        if post_backtest is None:
            raise HTTPException(503, "Бэктест поста не настроен")
        check_timeframe(request.timeframe)
        try:
            pd.Timestamp(request.date)
        except ValueError:
            raise HTTPException(400, "Дата: формат 2025-03-01 10:00") from None
        if request.answer_json.strip():
            try:
                json.loads(request.answer_json)
            except ValueError:
                raise HTTPException(400, "Ответ разбора — не JSON") from None
        body = request.model_dump()
        title = f"Бэктест поста {request.date}, {request.days:g} дн., {request.timeframe}"
        try:
            job = jobs.start("post_backtest", title, lambda job: post_backtest(body, job.say), body)
        except RuntimeError:
            raise HTTPException(409, "Уже идёт бэктест поста — дождитесь окончания") from None
        return {"job_id": job.job_id}

    @app.post("/api/scenarios/{scenario_id}/approve")
    def approve_cell(scenario_id: str) -> dict[str, str]:
        path = db_path(storage.root)
        if not path.exists():
            raise HTTPException(404, "Нет такой ячейки")
        conn = connect(path)
        try:
            if not approve(conn, scenario_id):
                raise HTTPException(404, "Нет такой ячейки")
        finally:
            conn.close()
        return {"status": "approved"}

    @app.get("/api/telegram/media/{channel}/{name}")
    def get_media(channel: str, name: str) -> FileResponse:
        rel = f"telegram/media/{channel}/{name}"
        path = storage.root / rel
        if MEDIA_RE.match(rel) is None or ".." in name or not path.is_file():
            raise HTTPException(404, "Нет такой картинки")
        return FileResponse(path)

    @app.get("/api/strategies")
    def get_strategies() -> dict[str, Any]:
        return {"strategies": strategy_catalog(), "symbols": symbols, "timeframes": TIMEFRAMES}

    # --- бот из терминала (config/live.yaml) -------------------------------------------

    @app.get("/api/bot")
    def get_bot(
        mode: str = Query(default="paper", pattern="^(paper|demo|live)$"),
    ) -> dict[str, Any]:
        return bot_status(storage.root, mode)

    @app.post("/api/bot/kill")
    def post_kill() -> dict[str, str]:
        """Аварийная остановка всего: бот из терминала (файл data/KILL) и paper-боты."""
        request_kill(storage.root, "кнопка в интерфейсе")
        stopped = host.kill_all("кнопка в интерфейсе") if host is not None else 0
        return {
            "status": "запрошена аварийная остановка: бот из терминала выполнит её в течение "
            f"минуты; paper-ботов остановлено: {stopped}"
        }

    @app.post("/api/bot/resume")
    def post_resume() -> dict[str, str]:
        request_resume(storage.root)
        return {"status": "запрошено возобновление: бот выполнит его в течение минуты"}

    # --- paper-боты интерфейса ---------------------------------------------------------

    def need_host() -> BotHost:
        if host is None:
            raise HTTPException(503, "Paper-боты не настроены")
        return host

    @app.get("/api/bots")
    def get_bots() -> dict[str, Any]:
        return {"bots": need_host().list()}

    @app.post("/api/bots")
    def create_bot(request: BotRequest) -> dict[str, Any]:
        h = need_host()
        check_strategy(request.strategy)
        check_symbol(request.symbol)
        check_timeframe(request.timeframe)
        try:
            spec = h.create(
                request.strategy, request.symbol, request.timeframe, request.params,
                request.capital,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from None
        return {"id": spec.id}

    def bot_action(bot_id: str, action: Callable[[str], None]) -> dict[str, str]:
        try:
            action(bot_id)
        except KeyError:
            raise HTTPException(404, f"Бот {bot_id} не найден или не запущен") from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"status": "ok"}

    @app.get("/api/bots/{bot_id}")
    def get_bot_detail(bot_id: str) -> dict[str, Any]:
        try:
            return need_host().detail(bot_id)
        except (KeyError, ValueError):
            raise HTTPException(404, f"Бот {bot_id} не найден") from None

    @app.post("/api/bots/{bot_id}/start")
    def start_bot(bot_id: str) -> dict[str, str]:
        return bot_action(bot_id, need_host().start)

    @app.post("/api/bots/{bot_id}/stop")
    def stop_bot(bot_id: str) -> dict[str, str]:
        return bot_action(bot_id, need_host().stop)

    @app.post("/api/bots/{bot_id}/kill")
    def kill_bot(bot_id: str) -> dict[str, str]:
        return bot_action(bot_id, lambda i: need_host().kill(i, "кнопка в интерфейсе"))

    @app.post("/api/bots/{bot_id}/resume")
    def resume_bot(bot_id: str) -> dict[str, str]:
        return bot_action(bot_id, need_host().resume)

    @app.delete("/api/bots/{bot_id}")
    def delete_bot(bot_id: str) -> dict[str, str]:
        return bot_action(bot_id, need_host().delete)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    import uvicorn

    from bot.backtest.report import run_report
    from bot.core.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
    from bot.core.logging import setup_logging
    from bot.data.bybit_public import make_public_client
    from bot.data.live_feed import LiveFeed
    from bot.notify.telegram import TelegramNotifier
    from bot.research.periods import open_holdout

    cfg = load_config()
    setup_logging(cfg.resolve(cfg.logging.dir), cfg.logging.level, cfg.logging.retention_days)
    data_dir = cfg.resolve(cfg.data.dir)
    logs_dir = cfg.resolve(cfg.logging.dir)
    storage = Storage(data_dir)
    public = make_public_client()

    def make_downloader(say: Callable[[str], None]) -> Downloader:
        return Downloader(public, storage, category=cfg.data.category, say=say)

    def run(request: dict[str, Any], say: Callable[[str], None]) -> str:
        current = load_config()  # свежие настройки: монеты могли добавить из интерфейса
        start = pd.Timestamp(request["start"], tz="UTC")
        end = pd.Timestamp(request["end"], tz="UTC") + pd.Timedelta(days=1)
        if request.get("download", True):
            say("проверяю историю за период, недостающее загружаю с Bybit…")
            try:
                make_downloader(say).ensure_history(request["symbol"], start, end)
            except Exception as exc:  # без интернета — бэктест на том, что уже скачано
                say(f"не удалось загрузить с Bybit: {exc}")
                say("считаю на том, что уже скачано")
        if touches_holdout(start, end):
            params = ", ".join(f"{k}={v}" for k, v in request["params"].items()) or "—"
            open_holdout(
                PROJECT_ROOT / "research" / "holdout-log.md",
                request["strategy"],
                request["symbol"],
                request["timeframe"],
                params + " (бэктест из интерфейса)",
                force=True,
            )
            say("запуск записан в research/holdout-log.md")
        folder, _ = run_report(
            current,
            request["strategy"],
            request["symbol"],
            request["timeframe"],
            start,
            end,
            request["params"],
            say=say,
        )
        return folder.name

    def post_backtest(request: dict[str, Any], say: Callable[[str], None]) -> str:
        from bot.analyzer.telegram_reader import post_images
        from bot.research.scenario_backtest import backtest_post

        current = load_config()
        published = pd.Timestamp(request["date"])
        if published.tzinfo is None:
            published = published.tz_localize("UTC")
        images: list[Path] = []
        if request.get("post_id") is not None and not request["answer_json"].strip():
            conn = connect(db_path(data_dir))
            try:
                rel = post_images(conn, current.scenarios.channel, int(request["post_id"]))
            finally:
                conn.close()
            images = [data_dir / p for p in rel]
        end = published + pd.Timedelta(days=float(request["days"]))
        for symbol in current.scenarios.symbols:
            try:
                make_downloader(say).ensure_history(symbol, published - pd.Timedelta(days=3), end)
            except Exception as exc:  # без интернета — на том, что уже скачано
                say(f"{symbol}: не удалось загрузить с Bybit ({type(exc).__name__})")
        runs = backtest_post(  # ошибка ввода (ValueError) покажется в журнале задачи
            current,
            request["text"],
            published,
            answer_json=request["answer_json"].strip() or None,
            images=images,
            days=float(request["days"]),
            timeframe=request["timeframe"],
            allow_image_levels=bool(request["allow_image_levels"]),
            say=say,
        )
        trades = sum(len(r.result.trades) for r in runs if r.result is not None)
        folders = ", ".join(r.folder.name for r in runs if r.folder is not None)
        tail = f"; отчёты во вкладке «Бэктесты»: {folders}" if folders else ""
        return f"сделок: {trades}{tail}"

    refresh_loader = make_downloader(lambda _text: None)
    refresher = CandleRefresher(
        lambda symbol, start: refresh_loader.download_klines(symbol, start, "1m"),
        lambda symbol: storage.last_candle_time(symbol, "1m"),
    )

    def lookup(symbol: str) -> dict[str, Any] | None:
        items = public.get_instruments_info(category=cfg.data.category, symbol=symbol)
        found: list[dict[str, Any]] = items["result"]["list"]
        return found[0] if found else None

    symbols = list(cfg.data.symbols)
    book = SymbolBook(symbols, DEFAULT_CONFIG_PATH.parent, lookup)
    notifier = TelegramNotifier.from_secrets(cfg.secrets)
    journal_dir = logs_dir / "journal"
    host = BotHost(
        data_dir,
        storage,
        load_config,
        journal_dir,
        feed_factory=lambda syms, on_candle: LiveFeed(
            storage, make_downloader(lambda _text: None), syms, on_candle=on_candle
        ),
        history=lambda symbol, start, end: make_downloader(print).ensure_history(
            symbol, start, end
        ),
        notify=notifier.send,
    )
    jobs = JobRunner(logs_dir / "jobs")
    commands = CommandRunner(jobs, lambda: symbols)
    app = create_app(
        storage,
        symbols,
        run=run,
        refresher=refresher,
        downloader_factory=make_downloader,
        symbol_book=book,
        host=host,
        jobs=jobs,
        commands=commands,
        journal_dir=journal_dir,
        post_backtest=post_backtest,
        channel=cfg.scenarios.channel,
        scenario_symbols=cfg.scenarios.symbols,
    )
    host.autostart()
    print(f"Откройте http://{cfg.app.host}:{cfg.app.port}  (остановить: Ctrl+C)")
    try:
        uvicorn.run(app, host=cfg.app.host, port=cfg.app.port)
    finally:
        host.shutdown()


if __name__ == "__main__":
    main()
