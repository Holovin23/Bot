from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

from bot.analyzer.telegram_reader import MediaRecord, PostRecord, save_media, save_posts
from bot.app.api import create_app
from bot.app.scenarios import media_url
from bot.core.db import connect
from bot.data.storage import Storage
from bot.scenarios.models import ImageInfo, PostScenarios, Scenario, Trigger
from bot.scenarios.store import apply_post

TEXT = "Биток: если закрепимся выше 100к — идём на 108к. Стоп под 97к."


def setup(tmp_path: Path, post_backtest: Callable[..., str] | None = None) -> TestClient:
    conn = connect(tmp_path / "bot.sqlite")
    t = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
    save_posts(conn, [PostRecord("chan", 1, t, TEXT), PostRecord("chan", 3, t, "мем")])
    folder = tmp_path / "telegram" / "media" / "chan"
    folder.mkdir(parents=True)
    (folder / "2.jpg").write_bytes(b"jpg")
    save_media(
        conn,
        [
            MediaRecord("chan", 1, 9, "telegram/media/chan/1.jpg"),
            MediaRecord("chan", 2, 9, "telegram/media/chan/2.jpg"),
        ],
    )
    text_cell = Scenario(
        symbol="BTCUSDT",
        side="LONG",
        trigger=Trigger(type="close_above", level=100_000),
        stop=97_000,
        targets=[108_000],
        quotes=["закрепимся выше 100к"],
    )
    image_cell = text_cell.model_copy(
        update={"symbol": "ETHUSDT", "level_source": "IMAGE", "quotes": []}
    )
    result = PostScenarios(
        post_kind="ANALYSIS",
        images=[ImageInfo(index=0, kind="CHART")],
        summary_ru="биток вверх",
        new_scenarios=[text_cell, image_cell],
    )
    apply_post(
        conn,
        channel="chan",
        post_id=1,
        published_at=pd.Timestamp(t),
        post_text=TEXT,
        result=result,
        symbols=["BTCUSDT", "ETHUSDT"],
        source="claude-code",
    )
    conn.close()
    app = create_app(Storage(tmp_path), ["BTCUSDT", "ETHUSDT"], post_backtest=post_backtest)
    return TestClient(app)


def test_media_url() -> None:
    assert media_url("telegram/media/chan/12.jpg") == "/api/telegram/media/chan/12.jpg"
    assert media_url("telegram\\media\\chan\\12.jpg") == "/api/telegram/media/chan/12.jpg"
    assert media_url("../secret.txt") is None


def test_cells_posts_and_approve(tmp_path: Path) -> None:
    client = setup(tmp_path)
    data = client.get("/api/scenarios").json()
    cells = {c["symbol"]: c for c in data["cells"]}
    assert cells["BTCUSDT"]["needs_review"] is False
    assert cells["ETHUSDT"]["needs_review"] is True  # уровни только с картинки
    assert cells["BTCUSDT"]["scenario"]["trigger"]["level"] == 100_000
    posts = {p["post_id"]: p for p in data["posts"]}
    assert set(posts) == {1, 3}  # картинка альбома без подписи — не отдельный пост
    assert posts[1]["kind"] == "ANALYSIS" and posts[1]["summary"] == "биток вверх"
    assert posts[1]["images"] == [
        "/api/telegram/media/chan/1.jpg",
        "/api/telegram/media/chan/2.jpg",
    ]
    assert len(posts[1]["report"]["created"]) == 2
    assert posts[1]["result"]["new_scenarios"][0]["symbol"] == "BTCUSDT"
    assert posts[3]["parsed"] is False

    eth = cells["ETHUSDT"]["scenario_id"]
    assert client.post(f"/api/scenarios/{eth}/approve").status_code == 200
    again = {c["symbol"]: c for c in client.get("/api/scenarios").json()["cells"]}
    assert again["ETHUSDT"]["needs_review"] is False
    assert client.post("/api/scenarios/нет/approve").status_code == 404


def test_media_files(tmp_path: Path) -> None:
    client = setup(tmp_path)
    assert client.get("/api/telegram/media/chan/2.jpg").content == b"jpg"
    assert client.get("/api/telegram/media/chan/1.jpg").status_code == 404  # файла нет
    assert client.get("/api/telegram/media/chan/..%2Fbot.sqlite").status_code == 404


def test_post_backtest_job(tmp_path: Path) -> None:
    from tests.test_api import wait_job

    seen: list[dict[str, Any]] = []

    def fake(request: dict[str, Any], say: Callable[[str], None]) -> str:
        seen.append(request)
        say("прогон")
        return "сделок: 1"

    client = setup(tmp_path, fake)
    body = {"text": TEXT, "date": "2026-10-01 08:00", "answer_json": "{}", "post_id": 1}
    job = wait_job(client, client.post("/api/scenarios/backtest", json=body).json()["job_id"])
    assert job["status"] == "done"
    assert seen[0]["days"] == 30 and seen[0]["timeframe"] == "1h" and seen[0]["post_id"] == 1
    bad = client.post("/api/scenarios/backtest", json={**body, "answer_json": "не json"})
    assert bad.status_code == 400
    bad = client.post("/api/scenarios/backtest", json={**body, "timeframe": "2h"})
    assert bad.status_code == 400
    assert setup(tmp_path / "x").post("/api/scenarios/backtest", json=body).status_code == 503


def test_pipeline_status(tmp_path: Path) -> None:
    client = setup(tmp_path)
    status = client.get("/api/scenarios/status").json()
    # Два поста (1 и 3), разобран один; две активные ячейки, одна — уровни с картинки
    assert status["posts"] == 2 and status["unparsed"] == 1 and status["failed"] == 0
    assert status["cells_active"] == 2 and status["cells_review"] == 1
    assert status["last_post"] == int(datetime(2026, 10, 1, 8, 0, tzinfo=UTC).timestamp())
    assert status["symbols"] == ["BTCUSDT", "ETHUSDT"]
    other = TestClient(create_app(Storage(tmp_path), ["BTCUSDT"], channel="другой"))
    assert other.get("/api/scenarios/status").json()["posts"] == 0
    empty = TestClient(create_app(Storage(tmp_path / "empty"), ["BTCUSDT"]))
    assert empty.get("/api/scenarios/status").json()["cells_active"] == 0
