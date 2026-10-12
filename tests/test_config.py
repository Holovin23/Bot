from pathlib import Path

import pytest

from bot.core.config import load_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("BYBIT_API_KEY", "BYBIT_API_SECRET", "BYBIT_ENV", "ALLOW_LIVE"):
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path: Path, text: str) -> Path:
    env = tmp_path / ".env"
    env.write_text(text, encoding="utf-8")
    return env


def test_loads_default_yaml_and_env(tmp_path: Path) -> None:
    env = write_env(tmp_path, "BYBIT_API_KEY=abc\nBYBIT_API_SECRET=xyz\n")
    cfg = load_config(env_file=env)
    assert cfg.data.symbols == ["BTCUSDT", "ETHUSDT"]
    assert cfg.costs.taker_fee == 0.00055
    assert cfg.costs.slippage["ETHUSDT"] == 0.0005
    assert cfg.secrets.bybit_api_key.get_secret_value() == "abc"
    assert cfg.secrets.bybit_env == "demo"
    assert cfg.secrets.is_demo


def test_demo_is_default_without_env_file(tmp_path: Path) -> None:
    cfg = load_config(env_file=tmp_path / "missing.env")
    assert cfg.secrets.bybit_env == "demo"


def test_live_without_flag_is_rejected(tmp_path: Path) -> None:
    env = write_env(tmp_path, "BYBIT_ENV=live\n")
    with pytest.raises(ValueError, match="ALLOW_LIVE"):
        load_config(env_file=env)


def test_live_with_flag_is_allowed(tmp_path: Path) -> None:
    env = write_env(tmp_path, "BYBIT_ENV=live\nALLOW_LIVE=true\n")
    cfg = load_config(env_file=env)
    assert cfg.secrets.bybit_env == "live"


def test_secrets_are_hidden_when_printed(tmp_path: Path) -> None:
    env = write_env(tmp_path, "BYBIT_API_KEY=supersecret\n")
    cfg = load_config(env_file=env)
    assert "supersecret" not in repr(cfg)
    assert "supersecret" not in str(cfg.secrets)


def test_env_example_parses_with_empty_values() -> None:
    from pathlib import Path

    from bot.core.config import PROJECT_ROOT, load_config

    cfg = load_config(env_file=Path(PROJECT_ROOT / ".env.example"))
    assert cfg.secrets.is_demo
    assert cfg.secrets.tg_api_id is None
    assert cfg.secrets.tg_chat_id == ""
