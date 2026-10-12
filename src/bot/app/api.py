"""Локальный веб-интерфейс: свечи, индикаторы и сделки бэктестов
на TradingView Lightweight Charts.

Запуск:
    uv run python -m bot.app.api
Открыть: http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bot.app.backtests import list_backtests, load_backtest, load_equity, load_report
from bot.app.bot_status import bot_status, request_kill, request_resume
from bot.app.jobs import JobRunner, RunFn
from bot.backtest.report import run_report
from bot.core.config import load_config
from bot.core.db import connect, db_path
from bot.core.timeframes import timeframe_minutes
from bot.data.resample import resample_candles
from bot.data.storage import Storage
from bot.features.indicators import bollinger, ema
from bot.research.experiment import IN_SAMPLE_END, IN_SAMPLE_START
from bot.research.periods import HOLDOUT
from bot.strategies.registry import STRATEGIES

STATIC_DIR = Path(__file__).parent / "static"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
EMA_LENGTHS = (20, 50, 200)
# Лишние свечи слева, чтобы EMA200 «разогрелась» и совпадала с графиком Bybit.
WARMUP_BARS = 1000
INDICATOR_COLUMNS = ["ema20", "ema50", "ema200", "bb_lower", "bb_middle", "bb_upper"]


def _clean(value: float) -> float | None:
    return None if math.isnan(value) else round(float(value), 8)


def build_chart_data(
    storage: Storage, symbol: str, timeframe: str, limit: int, until: int | None = None
) -> dict[str, Any]:
    """limit последних свечей; until (секунды UTC) — показать свечи до этого момента."""
    last = storage.last_candle_time(symbol, "1m")
    if last is None:
        return {"symbol": symbol, "timeframe": timeframe, "candles": []}
    minutes = timeframe_minutes(timeframe)
    end = None
    if until is not None:
        end = pd.Timestamp(until, unit="s", tz="UTC").ceil(f"{minutes}min")
        last = min(last, end - pd.Timedelta(minutes=1))
    start = (last - pd.Timedelta(minutes=minutes * (limit + WARMUP_BARS))).floor(f"{minutes}min")
    df = resample_candles(storage.read_candles(symbol, "1m", start=start, end=end), timeframe)

    for length in EMA_LENGTHS:
        df[f"ema{length}"] = ema(df["close"], length)
    df = pd.concat([df, bollinger(df["close"], 20, 2.0)], axis=1)
    df = df.tail(limit)

    # Время в секундах UTC (open_time хранится в миллисекундах).
    df["time"] = df["open_time"].astype("int64") // 1000
    columns = ["time", "open", "high", "low", "close", "volume", *INDICATOR_COLUMNS]
    candles: list[dict[str, Any]] = []
    for record in df[columns].to_dict(orient="records"):
        item = {str(key): value for key, value in record.items()}
        item["time"] = int(item["time"])
        for name in INDICATOR_COLUMNS:
            item[name] = _clean(item[name])
        candles.append(item)
    return {"symbol": symbol, "timeframe": timeframe, "candles": candles}


class RunRequest(BaseModel):
    strategy: str
    symbol: str
    timeframe: str = "1h"
    start: str = IN_SAMPLE_START
    end: str = IN_SAMPLE_END
    params: dict[str, Any] = Field(default_factory=dict)


def strategy_catalog() -> list[dict[str, Any]]:
    """Стратегии и их параметры по умолчанию (для формы запуска)."""
    return [
        {"name": name, "version": cls.version, "params": cls("1h").params}
        for name, cls in sorted(STRATEGIES.items())
    ]


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
) -> FastAPI:
    app = FastAPI(title="Bot")
    backtests_root = backtests_dir if backtests_dir is not None else storage.root / "backtests"
    jobs = JobRunner(run) if run is not None else None

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/symbols")
    def get_symbols() -> dict[str, list[str]]:
        return {"symbols": symbols, "timeframes": TIMEFRAMES}

    @app.get("/api/candles")
    def get_candles(
        symbol: str,
        timeframe: str = "1h",
        limit: int = Query(default=500, ge=10, le=5000),
        until: int | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        if symbol not in symbols:
            raise HTTPException(404, f"Неизвестный символ {symbol}")
        if timeframe not in TIMEFRAMES:
            raise HTTPException(400, f"Таймфрейм должен быть одним из {TIMEFRAMES}")
        return build_chart_data(storage, symbol, timeframe, limit, until)

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

    @app.get("/api/telegram")
    def get_telegram(limit: int = Query(default=50, ge=1, le=500)) -> dict[str, Any]:
        return {"posts": telegram_feed(storage.root, limit)}

    @app.get("/api/strategies")
    def get_strategies() -> dict[str, Any]:
        return {"strategies": strategy_catalog(), "symbols": symbols, "timeframes": TIMEFRAMES}

    @app.post("/api/backtests/run")
    def run_backtest_job(request: RunRequest) -> dict[str, Any]:
        if jobs is None:
            raise HTTPException(503, "Запуск бэктестов не настроен")
        if request.strategy not in STRATEGIES:
            raise HTTPException(400, f"Неизвестная стратегия {request.strategy}")
        if request.symbol not in symbols:
            raise HTTPException(400, f"Неизвестный символ {request.symbol}")
        if request.timeframe not in TIMEFRAMES:
            raise HTTPException(400, f"Таймфрейм должен быть одним из {TIMEFRAMES}")
        if pd.Timestamp(request.end, tz="UTC") >= HOLDOUT.start:
            raise HTTPException(400, "Период заходит в hold-out: используйте bot.research.validate")
        try:
            job = jobs.start(request.model_dump(), wait=False)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"job_id": job.job_id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id) if jobs is not None else None
        if job is None:
            raise HTTPException(404, "Нет такого запуска")
        return {"status": job.status, "backtest_id": job.backtest_id, "error": job.error}

    @app.get("/api/bot")
    def get_bot(mode: str = Query(default="paper", pattern="^(paper|demo)$")) -> dict[str, Any]:
        return bot_status(storage.root, mode)

    @app.post("/api/bot/kill")
    def post_kill() -> dict[str, str]:
        request_kill(storage.root, "кнопка в интерфейсе")
        return {"status": "запрошена аварийная остановка: бот выполнит её в течение минуты"}

    @app.post("/api/bot/resume")
    def post_resume() -> dict[str, str]:
        request_resume(storage.root)
        return {"status": "запрошено возобновление: бот выполнит его в течение минуты"}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    import uvicorn

    cfg = load_config()

    def run(
        strategy: str, symbol: str, timeframe: str, start: str, end: str, params: dict[str, Any]
    ) -> str:
        folder, _ = run_report(
            cfg,
            strategy,
            symbol,
            timeframe,
            pd.Timestamp(start, tz="UTC"),
            pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
            params,
        )
        return folder.name

    app = create_app(Storage(cfg.resolve(cfg.data.dir)), cfg.data.symbols, run=run)
    print(f"Откройте http://{cfg.app.host}:{cfg.app.port}  (остановить: Ctrl+C)")
    uvicorn.run(app, host=cfg.app.host, port=cfg.app.port)


if __name__ == "__main__":
    main()
