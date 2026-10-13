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


class FakeMediaTelegram(FakeTelegram):
    async def download_media(self, message: Any, file: str) -> str:
        path = Path(file + ".jpg")
        path.write_bytes(b"jpg")
        return str(path)


def photo(i: int, text: str | None, grouped: int | None = None) -> SimpleNamespace:
    m = msg(i, text)
    m.photo = object()
    m.grouped_id = grouped
    return m


def test_images_and_albums(tmp_path: Path) -> None:
    from bot.analyzer.telegram_reader import fetch_channel, post_images, save_media

    conn = connect(tmp_path / "bot.sqlite")
    source = FakeMediaTelegram(
        [
            photo(1, "BTC: график", grouped=77),  # подпись альбома
            photo(2, None, grouped=77),  # вторая картинка альбома — не отдельный пост
            photo(3, None),  # одиночная картинка без текста — пост
            msg(4, "просто текст"),
        ]
    )
    posts, media = asyncio.run(fetch_channel(source, "chan", 0, tmp_path))
    assert [p.post_id for p in posts] == [1, 3, 4]
    assert [m.path for m in media] == [f"telegram/media/chan/{i}.jpg" for i in (1, 2, 3)]
    save_posts(conn, posts)
    assert save_media(conn, media) == 3
    assert post_images(conn, "chan", 1) == [
        "telegram/media/chan/1.jpg",
        "telegram/media/chan/2.jpg",
    ]
    assert post_images(conn, "chan", 3) == ["telegram/media/chan/3.jpg"]
    assert post_images(conn, "chan", 4) == []
    assert last_post_id(conn, "chan") == 4


def test_progress_callback(tmp_path: Path) -> None:
    from bot.analyzer import telegram_reader

    calls: list[tuple[int, int]] = []
    source = FakeMediaTelegram([msg(i, f"пост {i}") for i in range(1, 46)])
    asyncio.run(
        telegram_reader.fetch_channel(
            source, "chan", 0, tmp_path, lambda m, i: calls.append((m, i))
        )
    )
    assert calls == [(20, 0), (40, 0)]
