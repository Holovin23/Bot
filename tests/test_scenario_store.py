from pathlib import Path

import pandas as pd

from bot.core.db import connect
from bot.scenarios.models import (
    ExistingDecision,
    ImageInfo,
    PostScenarios,
    Scenario,
    Trigger,
    quote_found,
    validate_scenario,
)
from bot.scenarios.store import active_cells, apply_post, approve, is_processed, load_versions

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TEXT = "Биток: если закрепимся выше 100к — идём на 108к. Стоп под 97к."


def scenario(**kw: object) -> Scenario:
    base: dict[str, object] = {
        "symbol": "BTCUSDT",
        "side": "LONG",
        "trigger": Trigger(type="close_above", level=100_000),
        "stop": 97_000,
        "targets": [108_000],
        "quotes": ["если  закрепимся выше 100к"],
    }
    return Scenario(**{**base, **kw})  # type: ignore[arg-type]


def test_quote_and_validation() -> None:
    assert quote_found("ЕСЛИ закрепимся   выше 100к", TEXT)
    assert not quote_found("пробьём 120к", TEXT)
    assert validate_scenario(scenario(), TEXT, SYMBOLS, 0) == []
    errors = validate_scenario(scenario(symbol="DOGEUSDT", quotes=["выдумка"]), TEXT, SYMBOLS, 0)
    assert errors == ["DOGEUSDT нет в белом списке"]
    errors = validate_scenario(scenario(stop=101_000, quotes=["выдумка"]), TEXT, SYMBOLS, 0)
    assert "стоп автора с неправильной стороны от уровня входа" in errors
    assert "ни одной цитаты из сценария нет в тексте поста" in errors
    assert validate_scenario(scenario(side="SHORT", stop=None, targets=[]), TEXT, SYMBOLS, 0) == [
        "close_above не сочетается с SHORT"
    ]
    image_only = scenario(level_source="IMAGE", quotes=[])
    assert validate_scenario(image_only, TEXT, SYMBOLS, 0) == [
        "уровни с картинки, но в посте нет графика"
    ]
    assert validate_scenario(image_only, TEXT, SYMBOLS, 1) == []


def apply(conn, post_id: int, at: str, result: PostScenarios, text: str = TEXT):  # type: ignore[no-untyped-def]
    return apply_post(
        conn,
        channel="c",
        post_id=post_id,
        published_at=pd.Timestamp(at, tz="UTC"),
        post_text=text,
        result=result,
        symbols=SYMBOLS,
        source="test",
    )


def test_create_update_cancel_and_versions(tmp_path: Path) -> None:
    conn = connect(tmp_path / "bot.sqlite")
    r1 = apply(
        conn,
        1,
        "2025-03-01 10:00",
        PostScenarios(
            post_kind="ANALYSIS",
            new_scenarios=[scenario(), scenario(symbol="SOLUSDT")],
        ),
    )
    sid = "tg-BTCUSDT-202503011000-1"
    assert r1.created == [sid]
    assert r1.rejected == ["SOLUSDT LONG close_above — SOLUSDT нет в белом списке"]
    assert is_processed(conn, "c", 1)
    assert [c["scenario_id"] for c in active_cells(conn)] == [sid]

    # Пост с результатами сделок ячейки не меняет
    r2 = apply(
        conn,
        2,
        "2025-03-02 10:00",
        PostScenarios(
            post_kind="RESULTS",
            images=[ImageInfo(index=1, kind="PNL_SCREENSHOT")],
            existing=[ExistingDecision(scenario_id=sid, action="CANCEL")],
        ),
    )
    assert r2.cancelled == [] and len(active_cells(conn)) == 1

    new_text = "Биток: теперь ждём закрытия выше 102к, цель 110к, стоп 98к"
    r3 = apply(
        conn,
        3,
        "2025-03-03 10:00",
        PostScenarios(
            post_kind="ANALYSIS",
            existing=[
                ExistingDecision(
                    scenario_id=sid,
                    action="UPDATE",
                    reason_ru="уровень поднят",
                    scenario=scenario(
                        trigger=Trigger(type="close_above", level=102_000),
                        stop=98_000,
                        targets=[110_000],
                        quotes=["закрытия выше 102к"],
                    ),
                )
            ],
        ),
        text=new_text,
    )
    assert r3.updated == [f"{sid} (версия 2): уровень поднят"]

    r4 = apply(
        conn,
        4,
        "2025-03-05 10:00",
        PostScenarios(
            post_kind="ANALYSIS",
            existing=[
                ExistingDecision(scenario_id=sid, action="CANCEL", reason_ru="автор передумал")
            ],
        ),
    )
    assert r4.cancelled == [f"{sid}: автор передумал"]
    assert active_cells(conn) == []

    versions = load_versions(conn, "BTCUSDT")
    assert [(v.version, v.scenario.trigger.level) for v in versions] == [(1, 100_000), (2, 102_000)]
    assert versions[0].valid_from == pd.Timestamp("2025-03-01 10:00", tz="UTC")
    assert versions[0].valid_to == pd.Timestamp("2025-03-03 10:00", tz="UTC")
    assert versions[1].valid_to == pd.Timestamp("2025-03-05 10:00", tz="UTC")

    # Отменённую ячейку нельзя обновить
    r5 = apply(
        conn,
        5,
        "2025-03-06 10:00",
        PostScenarios(
            post_kind="ANALYSIS",
            existing=[ExistingDecision(scenario_id=sid, action="UPDATE", scenario=scenario())],
        ),
    )
    assert r5.rejected == [f"{sid} — нет такой активной ячейки"]


def test_image_levels_need_review(tmp_path: Path) -> None:
    conn = connect(tmp_path / "bot.sqlite")
    apply(
        conn,
        1,
        "2025-03-01 10:00",
        PostScenarios(
            post_kind="ANALYSIS",
            images=[ImageInfo(index=1, kind="CHART")],
            new_scenarios=[scenario(level_source="IMAGE", quotes=[])],
        ),
    )
    [cell] = active_cells(conn)
    assert cell["needs_review"] is True
    assert approve(conn, cell["scenario_id"])
    assert load_versions(conn, "BTCUSDT")[0].needs_review is False
