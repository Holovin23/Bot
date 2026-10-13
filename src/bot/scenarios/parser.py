"""Разбор поста в сценарии: правила, схема ответа и вызов Claude API (план А, docs/10).

Правила и схема одни и те же для Claude API и для скилла Claude Code (план Б): скилл
получает их из `python -m bot.scenarios pending`. Модель только переносит в схему то,
что написал автор; проверяет и применяет результат код (models.validate_scenario,
store.apply_post).
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from bot.scenarios.models import PostScenarios

PROMPT_VERSION = "scenarios-v1"
MAX_TOKENS = 16000
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_NUM_OR_NULL = {"anyOf": [{"type": "number"}, {"type": "null"}]}


def scenario_schema(symbols: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "symbol",
            "side",
            "trigger",
            "stop",
            "targets",
            "level_source",
            "quotes",
            "conviction",
            "note_ru",
        ],
        "properties": {
            "symbol": {"type": "string", "enum": symbols},
            "side": {"type": "string", "enum": ["LONG", "SHORT"]},
            "trigger": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "level", "zone_low", "zone_high"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["close_above", "close_below", "retest", "touch_zone", "now"],
                    },
                    "level": _NUM_OR_NULL,
                    "zone_low": _NUM_OR_NULL,
                    "zone_high": _NUM_OR_NULL,
                },
            },
            "stop": _NUM_OR_NULL,
            "targets": {"type": "array", "items": {"type": "number"}},
            "level_source": {"type": "string", "enum": ["TEXT", "IMAGE", "MIXED"]},
            "quotes": {"type": "array", "items": {"type": "string"}},
            "conviction": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
            "note_ru": {"type": "string"},
        },
    }


def response_schema(symbols: list[str]) -> dict[str, Any]:
    scenario = scenario_schema(symbols)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["post_kind", "images", "summary_ru", "existing", "new_scenarios"],
        "properties": {
            "post_kind": {"type": "string", "enum": ["ANALYSIS", "RESULTS", "OTHER"]},
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["index", "kind"],
                    "properties": {
                        "index": {"type": "integer"},
                        "kind": {"type": "string", "enum": ["CHART", "PNL_SCREENSHOT", "OTHER"]},
                    },
                },
            },
            "summary_ru": {"type": "string"},
            "existing": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["scenario_id", "action", "reason_ru", "scenario"],
                    "properties": {
                        "scenario_id": {"type": "string"},
                        "action": {"type": "string", "enum": ["KEEP", "UPDATE", "CANCEL"]},
                        "reason_ru": {"type": "string"},
                        "scenario": {"anyOf": [scenario, {"type": "null"}]},
                    },
                },
            },
            "new_scenarios": {"type": "array", "items": scenario},
        },
    }


def system_prompt(symbols: list[str]) -> str:
    return f"""Ты разбираешь посты Telegram-канала с аналитикой криптовалют. Задача — перенести
в схему сценарии АВТОРА поста. Ты не прогнозируешь цену сам и ничего не добавляешь от себя.

Тип поста (post_kind):
- ANALYSIS — разбор с условиями на будущее («если …, то …», уровни, зоны).
- RESULTS — отчёт о прошлых сделках: профит, убыток, закрытые позиции, скриншоты с
  биржи или торговой платформы. По таким постам сценарии НЕ создаются и ячейки не меняются.
- OTHER — всё остальное (реклама, новости без условий, общение).

Картинки (images, index — номер картинки с 1):
- CHART — график цены с линиями, зонами, уровнями.
- PNL_SCREENSHOT — скриншот платформы с профитом, убытком, позициями, балансом. Содержимое
  таких картинок НЕ анализируй и не используй ни для каких полей.
- OTHER — остальное.

Монеты: только {", ".join(symbols)}. Названия приводи к тикеру Bybit: «биток», BTC → BTCUSDT;
эфир, ETH → ETHUSDT; SOL → SOLUSDT и т. п. Монет не из списка не касайся.
Числа: «64к» = 64000, «64 500» = 64500. Только числа автора или подписи на его графике.

Сценарий (new_scenarios) — одно условие автора с направлением:
- side: LONG (рост) или SHORT (падение).
- trigger.type:
  close_above — «закрепимся / пробьём вверх / закрытие выше уровня» (только LONG);
  close_below — «пробой вниз / закрытие ниже уровня» (только SHORT);
  retest — «после пробоя ждём ретест уровня» (LONG — ретест сверху, SHORT — снизу);
  touch_zone — «покупки / продажи от зоны или уровня», «отскок от поддержки»: zone_low и
  zone_high — края зоны; если автор дал один уровень, зона = этот уровень ± 0,3%;
  now — автор прямо предлагает входить сейчас.
  level — уровень для close_above, close_below, retest; для touch_zone и now — null.
- stop — только если автор назвал стоп или уровень, где сценарий отменяется; иначе null.
- targets — цели автора по порядку; если не названы — пустой список.
- level_source: TEXT — числа из текста; IMAGE — только с графика; MIXED — и то, и то.
- quotes — 1–3 дословных фрагмента текста поста, где описано условие (копируй символ в символ).
- conviction — насколько уверенно пишет автор. note_ru — одна фраза о сценарии.
Если условие расплывчатое и его нельзя выразить этими типами — не создавай сценарий.

Активные ячейки (existing): в сообщении есть список уже сохранённых сценариев. Для каждой
ячейки монеты, о которой пишет пост, реши: KEEP — пост её не меняет; UPDATE — автор изменил
уровни или условие (в scenario — полная новая версия, монета та же); CANCEL — автор отменил
сценарий или он больше не актуален по словам автора. Ячейки монет, о которых пост молчит,
не включай. Новые сценарии, не совпадающие с ячейками, — в new_scenarios.
summary_ru — одно предложение: что думает автор."""


def user_text(published_at: str, text: str, image_count: int, cells: list[dict[str, Any]]) -> str:
    images = f"Картинок в посте: {image_count} (идут перед этим текстом по порядку)."
    return (
        f"Пост опубликован {published_at} (UTC). {images}\n\n"
        f"Текст поста:\n<<<\n{text}\n>>>\n\n"
        f"Активные ячейки:\n{json.dumps(cells, ensure_ascii=False, indent=1)}"
    )


def image_block(path: Path) -> dict[str, Any]:
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


@dataclass(frozen=True)
class ApiOutcome:
    result: PostScenarios | None
    error: str | None
    usage: dict[str, int]


def parse_with_api(
    client: Any,
    model: str,
    symbols: list[str],
    published_at: str,
    text: str,
    images: list[Path],
    cells: list[dict[str, Any]],
) -> ApiOutcome:
    """Один вызов Claude API. Любая ошибка → пост «не разобран», бот работает дальше."""
    content: list[dict[str, Any]] = [image_block(p) for p in images]
    content.append({"type": "text", "text": user_text(published_at, text, len(images), cells)})
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": system_prompt(symbols),
        "messages": [{"role": "user", "content": content}],
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": response_schema(symbols)},
        },
    }
    try:
        if "haiku" in model:
            response = client.messages.create(**kwargs)
        else:  # при отказе модели запрос сам повторяется на запасной модели
            response = client.beta.messages.create(
                **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
    except Exception as exc:  # сеть, лимиты, неверный ключ
        return ApiOutcome(None, f"ошибка API: {type(exc).__name__}: {exc}"[:500], {})
    u = getattr(response, "usage", None)
    usage = {
        k: int(getattr(u, k, 0) or 0)
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")
    }
    if response.stop_reason == "refusal":
        return ApiOutcome(None, "модель отказалась разбирать пост", usage)
    if response.stop_reason == "max_tokens":
        return ApiOutcome(None, "ответ обрезан (max_tokens)", usage)
    raw = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    try:
        return ApiOutcome(PostScenarios.model_validate(json.loads(raw)), None, usage)
    except (json.JSONDecodeError, ValidationError) as exc:
        return ApiOutcome(None, f"ответ не по схеме: {exc}"[:500], usage)
