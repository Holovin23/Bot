"""Команды терминала из веб-интерфейса.

Каждая команда из docs/USER_GUIDE.md (Часть «Все команды») описана здесь со своими
параметрами. Интерфейс показывает форму, сервер собирает из неё ту же командную строку
и запускает `python -m ...` в папке проекта — то есть делает ровно то, что сделала бы
команда в терминале. Вывод виден в интерфейсе по мере выполнения и сохраняется в
logs/jobs/.

Не запускаются отсюда: реальный счёт (`--mode live` требует ручного подтверждения в
терминале) и ручная разметка постов (`llm_parser --label` спрашивает ответы с клавиатуры).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from bot.app.jobs import Job, JobRunner
from bot.core.config import PROJECT_ROOT
from bot.strategies.registry import STRATEGIES

TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[^\n\r]{0,200}$")
TEXT_RE = re.compile(r"^[^\n\r\-][^\n\r]{0,199}$")  # не с «-»: иначе это был бы новый флаг
MAX_RUNNING = 4


@dataclass(frozen=True)
class Arg:
    flag: str  # например "--symbol"
    label: str
    kind: str = "text"  # text | date | int | float | choice | multi | flag | kv
    choices: tuple[str, ...] = ()  # для choice/multi; "@symbols"/"@strategies" — из настроек
    default: Any = None
    required: bool = False
    help: str = ""


@dataclass(frozen=True)
class Command:
    id: str
    title: str
    group: str
    target: tuple[str, ...]  # что запускать после python: ("-m", "bot.x") или ("scripts/x.py",)
    args: tuple[Arg, ...] = ()
    help: str = ""


def _symbol(flag: str = "--symbol", required: bool = False, multi: bool = False) -> Arg:
    kind = "multi" if multi else "choice"
    hint = "можно несколько; пусто — все из настроек" if multi else ""
    return Arg(flag, "Символ", kind, ("@symbols",), required=required, help=hint)


def _strategy(required: bool = True) -> Arg:
    return Arg("--strategy", "Стратегия", "choice", ("@strategies",), required=required)


def _tf(default: str = "1h") -> Arg:
    return Arg("--timeframe", "Таймфрейм", "choice", TIMEFRAMES, default=default)


PARAM = Arg("--param", "Параметры стратегии", "kv", help="по строке: ключ=значение")

COMMANDS: tuple[Command, ...] = (
    Command("config", "Проверить настройки", "Проверка", ("-m", "bot"),
            help="Читает config/default.yaml и .env, печатает настройки без секретов."),
    Command("connection", "Связь с Bybit", "Проверка", ("scripts/check_connection.py",),
            help="Время сервера, параметры инструментов, баланс демо-счёта."),
    Command("tests", "Автотесты", "Проверка", ("-m", "pytest", "-q"),
            help="Все автоматические тесты проекта (pytest)."),
    Command(
        "download", "Скачать историю", "Данные", ("-m", "bot.data.downloader"),
        (
            _symbol(multi=True),
            Arg("--from", "С даты", "date", default="2021-01-01"),
            Arg("--to", "По дату", "date", help="включительно; пусто — до сейчас"),
            Arg("--only", "Что", "multi", ("klines", "funding", "oi"),
                help="пусто — всё: свечи, funding, open interest"),
        ),
        "Догружает все пропуски в периоде (повторный запуск ничего не скачает заново).",
    ),
    Command(
        "quality", "Качество данных", "Данные", ("-m", "bot.data.quality"),
        (_symbol(multi=True), _tf("1m"), Arg("--from", "С даты", "date"),
         Arg("--to", "По дату", "date")),
        "Пропуски, дубли, нулевой объём, выбросы.",
    ),
    Command(
        "live_feed", "Живой поток свечей", "Данные", ("-m", "bot.data.live_feed"),
        (Arg("--hours", "Часов", "float", default=1.0), _symbol(multi=True)),
        "Проверка WebSocket: печатает каждую закрытую минуту.",
    ),
    Command(
        "backtest", "Бэктест", "Исследования", ("-m", "bot.backtest.report"),
        (_strategy(), _symbol(required=True), _tf(),
         Arg("--from", "С даты", "date", default="2021-01-01", required=True),
         Arg("--to", "По дату", "date", default="2024-06-30", required=True), PARAM),
        "То же, что форма бэктеста на вкладке стратегии.",
    ),
    Command(
        "experiment", "Эксперимент против случайных входов", "Исследования",
        ("-m", "bot.research.experiment"),
        (
            _strategy(required=False), _symbol(), _tf(),
            Arg("--all", "Все стратегии × ТФ × символы", "flag"),
            Arg("--from", "С даты", "date", default="2021-01-01"),
            Arg("--to", "По дату", "date", default="2024-06-30"),
            PARAM,
            Arg("--runs", "Прогонов случайных входов", "int", default=200),
            Arg("--grid", "Перебор", "kv", help="ключ=v1,v2,... (до 3 параметров)"),
            Arg("--variants", "Сколько вариантов уже пробовали", "int", default=1),
        ),
        "Результат и запись EXP-NNN в research/experiments.md.",
    ),
    Command(
        "validate", "Validation / walk-forward / hold-out", "Исследования",
        ("-m", "bot.research.validate"),
        (
            _strategy(), _symbol(required=True), _tf(),
            Arg("--stage", "Этап", "choice", ("validation", "walk-forward", "holdout"),
                default="validation"),
            PARAM,
            Arg("--grid", "Перебор (walk-forward)", "kv", help="ключ=v1,v2,..."),
            _symbol("--other-symbol"),
            Arg("--i-know-holdout-is-burned", "Повторно открыть hold-out", "flag",
                help="только если понимаете, что hold-out уже не чистый"),
        ),
        "Hold-out открывается один раз: запуск пишется в research/holdout-log.md.",
    ),
    Command("pipeline", "Перебор с DSR", "Исследования", ("-m", "bot.research.pipeline"),
            (Arg("--dry-run", "Только посчитать попытки", "flag"),),
            "Комбинации из config/research.yaml через walk-forward и DSR."),
    Command(
        "ml_filter", "ML-фильтр", "Исследования", ("-m", "bot.research.ml_filter"),
        (_strategy(), _symbol(required=True), _tf(), PARAM,
         Arg("--model", "Модель", "choice", ("best", "logreg", "lgbm"), default="best")),
        "Обучить и проверить фильтр сигналов.",
    ),
    Command("telegram", "Скачать посты канала", "Telegram",
            ("-m", "bot.analyzer.telegram_reader"),
            (Arg("--channel", "Канал", "text", help="имя без @ или ссылка; пусто — из настроек"),
             Arg("--since", "Не раньше даты (UTC)", "date", help="пусто — вся история канала"),
             Arg("--every", "Повторять каждые N минут", "float",
                 help="пусто — один раз; задача идёт, пока её не остановить")),
            "Первый вход в Telegram спрашивает код — его нужно один раз ввести в терминале."),
    Command("scenarios_parse", "Сценарии: разбор через Claude API", "Telegram",
            ("-m", "bot.scenarios", "parse"),
            (Arg("--limit", "Сколько постов", "int", default=20),
             Arg("--model", "Модель", "text", help="пусто — scenarios.model из настроек")),
            "План А, платно: нужен ANTHROPIC_API_KEY. План Б — скилл telegram-scenarios "
            "в Claude Code."),
    Command("scenarios_list", "Сценарии: активные ячейки", "Telegram",
            ("-m", "bot.scenarios", "list")),
    Command(
        "llm_parse", "Разбор постов Claude", "Telegram", ("-m", "bot.analyzer.llm_parser"),
        (Arg("--parse", "Разобрать новые посты", "flag"),
         Arg("--compare", "Сверить с ручной разметкой", "flag"),
         Arg("--limit", "Сколько постов", "int", default=20),
         Arg("--model", "Модель", "text")),
        "Ручная разметка (--label) только в терминале: она задаёт вопросы.",
    ),
    Command("channel_study", "Польза мнений канала", "Telegram",
            ("-m", "bot.research.channel_study"),
            (_symbol(), Arg("--model", "Модель", "text"))),
    Command(
        "live_bot", "Бот из config/live.yaml", "Бот", ("-m", "bot.runner.live"),
        (Arg("--mode", "Режим", "choice", ("paper", "demo"), default="paper"),
         Arg("--hours", "Часов", "float", default=24.0)),
        "Тот же бот, что в терминале. Реальный счёт отсюда не запускается.",
    ),
    Command("live_vs_backtest", "Бот против бэктеста", "Бот",
            ("-m", "bot.research.live_vs_backtest"),
            (Arg("--mode", "Режим", "choice", ("paper", "demo"), default="paper"),)),
    Command("live_vs_demo", "Реальный счёт против демо", "Бот",
            ("-m", "bot.research.live_vs_demo")),
)
BY_ID = {c.id: c for c in COMMANDS}


def resolve_choices(arg: Arg, symbols: list[str]) -> tuple[str, ...]:
    if arg.choices == ("@symbols",):
        return tuple(symbols)
    if arg.choices == ("@strategies",):
        return tuple(sorted(STRATEGIES))
    return arg.choices


def catalog(symbols: list[str]) -> list[dict[str, Any]]:
    out = []
    for c in COMMANDS:
        item = asdict(c)
        item["command"] = "python " + " ".join(c.target)
        item["args"] = [{**asdict(a), "choices": resolve_choices(a, symbols)} for a in c.args]
        out.append(item)
    return out


def build_argv(command: Command, values: dict[str, Any], symbols: list[str]) -> list[str]:
    """Аргументы командной строки из значений формы. Неверное значение — ValueError."""
    argv: list[str] = []
    for arg in command.args:
        value = values.get(arg.flag)
        empty = value is None or value == "" or value == [] or value is False
        if empty:
            if arg.required:
                raise ValueError(f"Не заполнено: {arg.label}")
            continue
        choices = resolve_choices(arg, symbols)
        if arg.kind == "flag":
            if value is not True:
                raise ValueError(f"{arg.label}: ожидается да/нет")
            argv.append(arg.flag)
        elif arg.kind in ("multi", "kv"):
            items = value if isinstance(value, list) else [value]
            for item in items:
                text = str(item).strip()
                if not text:
                    continue
                if arg.kind == "multi" and text not in choices:
                    raise ValueError(f"{arg.label}: недопустимое значение {text!r}")
                if arg.kind == "kv" and not KV_RE.match(text):
                    raise ValueError(f"{arg.label}: нужно ключ=значение, а не {text!r}")
                argv += [arg.flag, text]
        else:
            argv += [arg.flag, _single(arg, value, choices)]
    return argv


def _single(arg: Arg, value: Any, choices: tuple[str, ...]) -> str:
    text = str(value).strip()
    if arg.kind == "choice":
        if text not in choices:
            raise ValueError(f"{arg.label}: недопустимое значение {text!r}")
    elif arg.kind == "date":
        if not DATE_RE.match(text):
            raise ValueError(f"{arg.label}: дата в виде ГГГГ-ММ-ДД, а не {text!r}")
    elif arg.kind == "int":
        text = str(int(text))
    elif arg.kind == "float":
        text = repr(float(text))
    elif not TEXT_RE.match(text):
        raise ValueError(f"{arg.label}: недопустимый текст {text!r}")
    return text


@dataclass
class CommandRunner:
    jobs: JobRunner
    symbols: Callable[[], list[str]]
    python: str = sys.executable
    cwd: str = str(PROJECT_ROOT)
    _procs: dict[str, subprocess.Popen[str]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, command_id: str, values: dict[str, Any]) -> Job:
        command = BY_ID.get(command_id)
        if command is None:
            raise KeyError(command_id)
        argv = build_argv(command, values, self.symbols())
        running = [j for j in self.jobs.list(200) if j.kind == "command" and j.status == "running"]
        if len(running) >= MAX_RUNNING:
            raise RuntimeError(f"Уже идут {len(running)} команды — дождитесь окончания")
        full = [self.python, *command.target, *argv]
        shown = "python " + " ".join([*command.target, *argv])

        def work(job: Job) -> str:
            job.say(f"$ {shown}")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(
                full,
                cwd=self.cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            with self._lock:
                self._procs[job.job_id] = proc
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    job.say(line)
                code = proc.wait()
            finally:
                with self._lock:
                    self._procs.pop(job.job_id, None)
            if code != 0:
                raise RuntimeError(f"команда завершилась с кодом {code}")
            return "код 0"

        return self.jobs.start(
            "command", command.title, work, {"command": command_id, "argv": argv}, exclusive=False
        )

    def stop(self, job_id: str) -> bool:
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is None:
            return False
        proc.terminate()
        return True
