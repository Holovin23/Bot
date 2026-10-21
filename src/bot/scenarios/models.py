"""Сценарий автора канала и разбор поста (docs/10-telegram-scenarios-plan.md).

ИИ только переносит в эти структуры то, что написал автор. Проверку и решение о сделке
делает код: validate_scenario здесь, стоп и цель по умолчанию — scenarios/levels.py,
объём и лимиты — риск-менеджер.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Side = Literal["LONG", "SHORT"]
TriggerType = Literal["close_above", "close_below", "retest", "touch_zone", "now"]
PostKind = Literal["ANALYSIS", "RESULTS", "OTHER"]
ImageKind = Literal["CHART", "PNL_SCREENSHOT", "OTHER"]
LevelSource = Literal["TEXT", "IMAGE", "MIXED"]
Action = Literal["KEEP", "UPDATE", "CANCEL"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Trigger(_Strict):
    """Условие входа на закрытой свече таймфрейма сценария.

    close_above / close_below — закрытие выше / ниже level (пересечение);
    retest — после пробоя level цена возвращается к нему и закрывается с нужной стороны;
    touch_zone — касание зоны zone_low..zone_high и закрытие не за зоной;
    now — автор предлагает входить сразу (первая свеча после поста).
    """

    type: TriggerType
    level: float | None = None
    zone_low: float | None = None
    zone_high: float | None = None


class Scenario(_Strict):
    symbol: str
    side: Side
    trigger: Trigger
    stop: float | None = None  # уровень отмены сценария, если автор его назвал
    targets: list[float] = Field(default_factory=list)
    level_source: LevelSource = "TEXT"
    quotes: list[str] = Field(default_factory=list)  # дословные цитаты из поста
    conviction: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"
    note_ru: str = ""


class ImageInfo(_Strict):
    index: int
    kind: ImageKind


class ExistingDecision(_Strict):
    """Что делать с уже сохранённой ячейкой монеты после нового поста."""

    scenario_id: str
    action: Action
    reason_ru: str = ""
    scenario: Scenario | None = None  # новая версия для UPDATE


class PostScenarios(_Strict):
    """Ответ разбора одного поста (одинаковый для Claude API и Claude Code)."""

    post_kind: PostKind
    images: list[ImageInfo] = Field(default_factory=list)
    summary_ru: str = ""
    existing: list[ExistingDecision] = Field(default_factory=list)
    new_scenarios: list[Scenario] = Field(default_factory=list)


def reference_levels(scenario: Scenario) -> tuple[float, float]:
    """(нижняя, верхняя) граница уровня входа: для зоны — края, иначе level дважды.
    Для `now` уровня нет — (nan, nan)."""
    t = scenario.trigger
    if t.type == "touch_zone":
        assert t.zone_low is not None and t.zone_high is not None
        return t.zone_low, t.zone_high
    if t.level is None:
        return float("nan"), float("nan")
    return t.level, t.level


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_found(quote: str, post_text: str) -> bool:
    """Цитата есть в посте (без учёта регистра и лишних пробелов)."""
    q = _normalize(quote)
    return bool(q) and q in _normalize(post_text)


def validate_scenario(
    scenario: Scenario, post_text: str, symbols: list[str], chart_images: int
) -> list[str]:
    """Проверки кодом. Пустой список — сценарий принят. chart_images — сколько картинок
    поста признано графиками (скриншоты профита/убытка не считаются)."""
    errors: list[str] = []
    t = scenario.trigger
    if scenario.symbol not in symbols:
        errors.append(f"{scenario.symbol} нет в белом списке")
    if t.type in ("close_above", "close_below", "retest"):
        if t.level is None or t.level <= 0:
            errors.append(f"для {t.type} нужен уровень")
    elif t.type == "touch_zone" and (
        t.zone_low is None or t.zone_high is None or not 0 < t.zone_low < t.zone_high
    ):
        errors.append("для touch_zone нужна зона zone_low < zone_high")
    if errors:
        return errors
    long = scenario.side == "LONG"
    if (t.type == "close_above" and not long) or (t.type == "close_below" and long):
        errors.append(f"{t.type} не сочетается с {scenario.side}")
    low, high = reference_levels(scenario)
    stop = scenario.stop
    if stop is not None and t.type != "now" and (stop >= low if long else stop <= high):
        errors.append("стоп автора с неправильной стороны от уровня входа")
    if t.type != "now":
        wrong = [x for x in scenario.targets if (x <= high if long else x >= low)]
        if wrong:
            errors.append(f"цели с неправильной стороны от уровня входа: {wrong}")
    quoted = any(quote_found(q, post_text) for q in scenario.quotes)
    if scenario.level_source in ("TEXT", "MIXED") and not quoted:
        errors.append("ни одной цитаты из сценария нет в тексте поста")
    if scenario.level_source in ("IMAGE", "MIXED") and chart_images == 0:
        errors.append("уровни с картинки, но в посте нет графика")
    return errors
