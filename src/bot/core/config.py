"""Загрузка настроек: config/default.yaml (общие параметры) + .env (секреты и режим счёта)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class DataConfig(BaseModel):
    dir: Path = Path("data")
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    category: Literal["linear"] = "linear"
    history_start: str = "2021-01-01"
    base_interval: str = "1m"
    timeframes: list[str] = Field(default_factory=lambda: ["5m", "15m", "1h", "4h"])
    open_interest_interval: str = "5min"


class LoggingConfig(BaseModel):
    dir: Path = Path("logs")
    level: str = "INFO"
    retention_days: int = 30


class AppConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class CostsConfig(BaseModel):
    taker_fee: float = 0.00055
    maker_fee: float = 0.0002
    slippage: dict[str, float] = Field(default_factory=dict)
    stop_slippage_multiplier: float = 2.0
    initial_capital: float = 10_000.0


class InstrumentConfig(BaseModel):
    tick_size: float
    qty_step: float
    min_qty: float


class RiskConfig(BaseModel):
    risk_per_trade: float = Field(default=0.005, gt=0, le=0.02)
    max_leverage: float = Field(default=3.0, gt=0, le=3.0)


class LimitsConfig(BaseModel):
    """Лимиты из docs/05-risk-and-execution.md. Границы полей не дают ослабить их больше
    значений из 05 — для этого нужно решение владельца и правка кода."""

    max_positions: int = Field(default=2, ge=1, le=2)
    max_positions_per_symbol: int = Field(default=1, ge=1, le=1)
    max_total_risk: float = Field(default=0.015, gt=0, le=0.015)
    correlated_groups: list[list[str]] = Field(default_factory=lambda: [["BTCUSDT", "ETHUSDT"]])
    correlated_weight: float = Field(default=1.5, ge=1.5)
    daily_loss: float = Field(default=0.02, gt=0, le=0.02)
    max_drawdown: float = Field(default=0.10, gt=0, le=0.10)
    max_loss_streak: int = Field(default=6, ge=1, le=6)
    min_stop_atr: float = Field(default=0.5, ge=0.5)
    max_stop_atr: float = Field(default=4.0, gt=0, le=4.0)
    liq_buffer: float = Field(default=2.0, ge=2.0)
    maintenance_margin: float = Field(default=0.005, ge=0.005)
    max_cost_share: float = Field(default=0.2, gt=0, le=0.2)
    stale_bars: int = Field(default=2, ge=1, le=2)
    default_min_rr: float = Field(default=1.5, ge=1.0)
    # Для отдельных стратегий (05, проверка 2): Mean Reversion — 1,0. random_entry —
    # исследовательская база, в сравнении получает R:R той стратегии, которую изображает.
    min_rr: dict[str, float] = Field(
        default_factory=lambda: {"mean_reversion": 1.0, "random_entry": 1.0}
    )

    @field_validator("min_rr")
    @classmethod
    def _min_rr_floor(cls, value: dict[str, float]) -> dict[str, float]:
        if any(v < 1.0 for v in value.values()):
            raise ValueError("минимальный R:R не может быть меньше 1,0 (docs/05)")
        return value


class Secrets(BaseSettings):
    """Значения из .env и переменных окружения. Ключи хранятся как SecretStr:
    при печати видно '**********', а не сам ключ."""

    model_config = SettingsConfigDict(
        extra="ignore", env_file_encoding="utf-8", env_ignore_empty=True
    )

    bybit_api_key: SecretStr = SecretStr("")
    bybit_api_secret: SecretStr = SecretStr("")
    bybit_env: Literal["demo", "live"] = "demo"
    allow_live: bool = False
    anthropic_api_key: SecretStr = SecretStr("")
    # Telegram: чтение канала (my.telegram.org) и бот для уведомлений (@BotFather)
    tg_api_id: int | None = None
    tg_api_hash: SecretStr = SecretStr("")
    tg_bot_token: SecretStr = SecretStr("")
    tg_chat_id: str = ""

    @model_validator(mode="after")
    def _live_requires_flag(self) -> Secrets:
        if self.bybit_env == "live" and not self.allow_live:
            raise ValueError("BYBIT_ENV=live требует ALLOW_LIVE=true в .env (решение владельца)")
        return self

    @property
    def is_demo(self) -> bool:
        return self.bybit_env == "demo"


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    app: AppConfig = Field(default_factory=AppConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    instruments: dict[str, InstrumentConfig] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    secrets: Secrets

    def resolve(self, path: Path) -> Path:
        """Относительные пути из конфига считаются от корня проекта."""
        return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    env_file: Path | None = DEFAULT_ENV_PATH,
) -> Config:
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    env = env_file if env_file is not None and env_file.exists() else None
    secrets = Secrets(_env_file=env)  # type: ignore[call-arg]
    return Config(**raw, secrets=secrets)
