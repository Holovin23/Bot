from typing import Any

from bot.core.config import Secrets
from bot.notify.telegram import TelegramNotifier


class FakeCommands:
    def __init__(self) -> None:
        self.killed: list[str] = []
        self.resumed = 0

    def status_text(self) -> str:
        return "Режим: paper"

    def kill(self, reason: str) -> None:
        self.killed.append(reason)

    def resume(self) -> None:
        self.resumed += 1


class FakeTransport:
    def __init__(self, updates: list[dict[str, Any]]) -> None:
        self.updates = updates
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((method, params))
        if method == "getUpdates":
            return {"ok": True, "result": self.updates}
        return {"ok": True}


def update(uid: int, text: str, chat: int = 42) -> dict[str, Any]:
    return {"update_id": uid, "message": {"chat": {"id": chat}, "text": text}}


def test_commands_only_from_owner_chat() -> None:
    transport = FakeTransport(
        [
            update(1, "/status"),
            update(2, "/kill", chat=999),
            update(3, "/kill"),
            update(4, "/resume"),
        ]
    )
    notifier = TelegramNotifier(transport, "42")
    commands = FakeCommands()
    notifier.poll_once(commands)
    assert commands.killed == ["команда /kill в Telegram"]  # чужой чат проигнорирован
    assert commands.resumed == 1
    sent = [p["text"] for m, p in transport.calls if m == "sendMessage"]
    assert sent == ["Режим: paper"]
    assert notifier.offset == 5


def test_unknown_command_gets_help() -> None:
    notifier = TelegramNotifier(FakeTransport([]), "42")
    assert (
        notifier.handle(update(1, "/hello"), FakeCommands()) == "Команды: /status, /kill, /resume"
    )
    assert notifier.handle(update(1, "просто текст"), FakeCommands()) is None


def test_disabled_without_token() -> None:
    notifier = TelegramNotifier.from_secrets(Secrets(_env_file=None))  # type: ignore[call-arg]
    assert not notifier.enabled
    notifier.send("проверка")  # только печать, без ошибок


def test_send_failure_is_swallowed() -> None:
    def broken(method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise OSError("network down")

    TelegramNotifier(broken, "42").send("текст")
