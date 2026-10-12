"""Локальный веб-интерфейс: свечи и индикаторы на TradingView Lightweight Charts.

Запуск:
    uv run python -m bot.app.api
Открыть: http://127.0.0.1:8000
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bot.core.config import load_config
from bot.core.timeframes import timeframe_minutes
from bot.data.resample import resample_candles
from bot.data.storage import Storage
from bot.features.indicators import bollinger, ema

STATIC_DIR = Path(__file__).parent / "static"
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h"]
EMA_LENGTHS = (20, 50, 200)
# Лишние свечи слева, чтобы EMA200 «разогрелась» и совпадала с графиком Bybit.
WARMUP_BARS = 1000
INDICATOR_COLUMNS = ["ema20", "ema50", "ema200", "bb_lower", "bb_middle", "bb_upper"]


def _clean(value: float) -> float | None:
    return None if math.isnan(value) else round(float(value), 8)


def build_chart_data(storage: Storage, symbol: str, timeframe: str, limit: int) -> dict[str, Any]:
    last = storage.last_candle_time(symbol, "1m")
    if last is None:
        return {"symbol": symbol, "timeframe": timeframe, "candles": []}
    minutes = timeframe_minutes(timeframe)
    start = (last - pd.Timedelta(minutes=minutes * (limit + WARMUP_BARS))).floor(f"{minutes}min")
    df = resample_candles(storage.read_candles(symbol, "1m", start=start), timeframe)

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


def create_app(storage: Storage, symbols: list[str]) -> FastAPI:
    app = FastAPI(title="Bot")

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
    ) -> dict[str, Any]:
        if symbol not in symbols:
            raise HTTPException(404, f"Неизвестный символ {symbol}")
        if timeframe not in TIMEFRAMES:
            raise HTTPException(400, f"Таймфрейм должен быть одним из {TIMEFRAMES}")
        return build_chart_data(storage, symbol, timeframe, limit)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def main() -> None:
    import uvicorn

    cfg = load_config()
    app = create_app(Storage(cfg.resolve(cfg.data.dir)), cfg.data.symbols)
    print(f"Откройте http://{cfg.app.host}:{cfg.app.port}  (остановить: Ctrl+C)")
    uvicorn.run(app, host=cfg.app.host, port=cfg.app.port)


if __name__ == "__main__":
    main()
