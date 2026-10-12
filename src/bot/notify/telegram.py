"""Уведомления и команды через Telegram Bot API (без сторонних библиотек).

Настройка: создать бота у @BotFather → TG_BOT_TOKEN; узнать свой chat id (например,
написать боту и открыть https://api.telegram.org/bot<токен>/getUpdates) → TG_CHAT_ID.
Без этих значений уведомления только печатаются в консоль.

Команды принимаются только из чата TG_CHAT_ID:
    /status — режим, капитал, позиции;  /kill — аварийная остановка;  /resume — возобновить.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from bot.core.config import Secrets
from bot.core.logging import get_logger

log = get_logger(__name__)

API = "https://api.telegram.org"
POLL_TIMEOUT_S = 25
MAX_LENGTH = 4000  # лимит Telegram — 4096 символов

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class Commands(Protocol):
    def status_text(self) -> str: ...

    def kill(self, reason: str) -> None: ...

    def resume(self) -> None: ...


def urllib_transport(token: str) -> Transport:
    def call(method: str, params: dict[str, Any]) -> dict[str, Any]:
        data = urllib.parse.urlencode(params).encode()
        url = f"{API}/bot{token}/{method}"
        timeout = POLL_TIMEOUT_S + 10
        with urllib.request.urlopen(url, data=data, timeout=timeout) as response:
            result: dict[str, Any] = json.loads(response.read().decode())
        return result

    return call


class TelegramNotifier:
    def __init__(self, transport: Transport | None, chat_id: str) -> None:
        self.transport = transport
        self.chat_id = chat_id
        self.offset = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_secrets(cls, secrets: Secrets) -> TelegramNotifier:
        token = secrets.tg_bot_token.get_secret_value()
        if not token or not secrets.tg_chat_id:
            print("Telegram не настроен (TG_BOT_TOKEN, TG_CHAT_ID): уведомления только в консоли")
            return cls(None, "")
        return cls(urllib_transport(token), secrets.tg_chat_id)

    @property
    def enabled(self) -> bool:
        return self.transport is not None

    def send(self, text: str) -> None:
        print(f"[уведомление] {text}")
        if self.transport is None:
            return
        try:
            self.transport("sendMessage", {"chat_id": self.chat_id, "text": text[:MAX_LENGTH]})
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # В сообщении об ошибке нет адреса с токеном: пишем только тип и код.
            log.warning("telegram_send_failed", error=type(exc).__name__)

    def handle(self, update: dict[str, Any], commands: Commands) -> str | None:
        """Одна команда. Чужие чаты игнорируются: управлять ботом может только владелец."""
        message = update.get("message") or {}
        chat = str((message.get("chat") or {}).get("id", ""))
        text = str(message.get("text") or "").strip()
        if chat != self.chat_id or not text.startswith("/"):
            return None
        command = text.split()[0].split("@")[0].lower()
        if command == "/status":
            return commands.status_text()
        if command == "/kill":
            commands.kill("команда /kill в Telegram")
            return None  # kill сам отправляет итог
        if command == "/resume":
            commands.resume()
            return None
        return "Команды: /status, /kill, /resume"

    def poll_once(self, commands: Commands) -> None:
        if self.transport is None:
            return
        response = self.transport("getUpdates", {"offset": self.offset, "timeout": POLL_TIMEOUT_S})
        for update in response.get("result", []):
            self.offset = max(self.offset, int(update["update_id"]) + 1)
            try:
                answer = self.handle(update, commands)
            except Exception as exc:
                log.exception("telegram_command_failed")
                answer = f"Ошибка команды: {exc}"
            if answer:
                self.send(answer)

    def start_commands(self, commands: Commands) -> None:
        if self.transport is None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.poll_once(commands)
                except (urllib.error.URLError, OSError, ValueError) as exc:
                    log.warning("telegram_poll_failed", error=type(exc).__name__)
                    self._stop.wait(5)

        self._thread = threading.Thread(target=loop, name="telegram", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
