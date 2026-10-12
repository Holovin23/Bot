import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bot.analyzer.telegram_reader import PostRecord, fetch_posts, last_post_id, save_posts
from bot.core.db import connect


class FakeTelegram:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.calls: list[dict[str, Any]] = []

    async def iter_messages(self, entity: str, **kwargs: Any) -> AsyncIterator[Any]:
        self.calls.append({"entity": entity, **kwargs})
        for m in self.messages:
            if m.id > kwargs.get("min_id", 0):
                yield m


def msg(i: int, text: str | None) -> SimpleNamespace:
    kyiv = timezone(timedelta(hours=3))
    return SimpleNamespace(id=i, message=text, date=datetime(2026, 10, 1, 11, i, tzinfo=kyiv))


def test_fetch_and_save(tmp_path: Path) -> None:
    conn = connect(tmp_path / "bot.sqlite")
    source = FakeTelegram([msg(1, "BTC выше 64000"), msg(2, None), msg(3, " ETH вниз ")])
    posts = asyncio.run(fetch_posts(source, "chan"))
    assert [p.post_id for p in posts] == [1, 3]  # пост без текста (картинка) пропущен
    assert posts[1].text == "ETH вниз"
    assert posts[0].published_at == datetime(2026, 10, 1, 8, 1, tzinfo=UTC)  # время в UTC
    assert save_posts(conn, posts) == 2
    assert save_posts(conn, posts) == 0  # повторно не дублируется
    assert last_post_id(conn, "chan") == 3
    # Докачка: только посты новее последнего сохранённого
    source.messages.append(msg(4, "новый"))
    new = asyncio.run(fetch_posts(source, "chan", last_post_id(conn, "chan")))
    assert [p.post_id for p in new] == [4]
    assert source.calls[-1]["min_id"] == 3
    assert source.calls[-1]["reverse"] is True


def test_record_is_plain_data() -> None:
    p = PostRecord("c", 1, datetime(2026, 1, 1, tzinfo=UTC), "t")
    assert p.channel == "c"
