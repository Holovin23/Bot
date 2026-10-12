"""Настройки живого бота из config/live.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from bot.core.config import PROJECT_ROOT

DEFAULT_LIVE_PATH = PROJECT_ROOT / "config" / "live.yaml"


class Instance(BaseModel):
    strategy: str
    symbol: str
    timeframe: str = "1h"
    params: dict[str, Any] = Field(default_factory=dict)
    ml_filter: bool = False


class LiveSettings(BaseModel):
    mode: Literal["paper", "demo"] = "paper"
    paper_capital: float = Field(default=10_000, gt=0)
    reconcile_minutes: int = Field(default=1, ge=1, le=60)
    close_unknown: bool = False
    unknown_stop_pct: float = Field(default=0.02, gt=0, le=0.1)
    history_days: int = Field(default=30, ge=3, le=365)
    instances: list[Instance] = Field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return sorted({i.symbol for i in self.instances})


def load_live_settings(path: Path = DEFAULT_LIVE_PATH) -> LiveSettings:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return LiveSettings(**(raw or {}))
