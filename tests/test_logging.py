import json
import logging
from pathlib import Path

from bot.core.logging import get_logger, setup_logging


def test_log_written_to_file_and_secrets_masked(tmp_path: Path) -> None:
    setup_logging(tmp_path, level="INFO")
    log = get_logger("test")
    log.info("connected", api_key="abc123", api_secret="xyz", symbol="BTCUSDT")
    for handler in logging.getLogger().handlers:
        handler.flush()

    text = (tmp_path / "bot.log").read_text(encoding="utf-8")
    record = json.loads(text.strip().splitlines()[-1])
    assert record["event"] == "connected"
    assert record["symbol"] == "BTCUSDT"
    assert record["api_key"] == "***"
    assert "abc123" not in text
    assert "xyz" not in text
    logging.getLogger().handlers.clear()
