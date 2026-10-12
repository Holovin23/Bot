"""Проверка запуска: `uv run python -m bot` печатает настройки (без секретов) и пишет лог."""

from bot.core.config import load_config
from bot.core.logging import get_logger, setup_logging


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.resolve(cfg.logging.dir), cfg.logging.level, cfg.logging.retention_days)
    log = get_logger("bot")
    log.info(
        "bot_started",
        bybit_env=cfg.secrets.bybit_env,
        symbols=cfg.data.symbols,
        keys_loaded=bool(cfg.secrets.bybit_api_key.get_secret_value()),
    )


if __name__ == "__main__":
    main()
