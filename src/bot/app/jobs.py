"""Фоновые задачи веб-интерфейса: бэктест, загрузка истории, команды терминала.

У каждой задачи есть журнал строк (что она делает сейчас). Он виден в интерфейсе и
сохраняется в logs/jobs/<дата>_<id>_<вид>.log — чтобы потом разобрать его глазами
или отдать ИИ на проверку.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_LOG_LINES = 5000

Say = Callable[[str], None]
# Работа задачи: получает саму задачу (для job.say), возвращает результат (например,
# id папки бэктеста) или None.
Work = Callable[["Job"], str | None]
# Бэктест: (запрос из формы, куда писать ход работы) -> id папки бэктеста
RunFn = Callable[[dict[str, Any], Say], str]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Job:
    job_id: str
    kind: str  # backtest | download | command | ...
    title: str
    status: str = "running"  # running | done | error
    result: str | None = None
    error: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    started: str = field(default_factory=_now)
    finished: str | None = None
    log: list[str] = field(default_factory=list)
    log_path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def say(self, line: str) -> None:
        text = str(line).rstrip()
        with self._lock:
            self.log.append(text)
            if len(self.log) > MAX_LOG_LINES:
                del self.log[: len(self.log) - MAX_LOG_LINES]
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(text + "\n")

    @property
    def backtest_id(self) -> str | None:
        return self.result if self.kind == "backtest" else None

    def summary(self, log_tail: int = 0) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "result": self.result,
            "backtest_id": self.backtest_id,
            "error": self.error,
            "started": self.started,
            "finished": self.finished,
        }
        if log_tail:
            with self._lock:
                data["log"] = self.log[-log_tail:]
        return data


class JobRunner:
    """exclusive=True — одновременно только одна задача этого вида (например, бэктест:
    два параллельных бэктеста тормозят друг друга и путают журнал)."""

    def __init__(self, log_dir: Path | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.log_dir = log_dir

    def busy(self, kind: str) -> bool:
        with self._lock:
            return any(j.status == "running" and j.kind == kind for j in self._jobs.values())

    def start(
        self,
        kind: str,
        title: str,
        work: Work,
        request: dict[str, Any] | None = None,
        exclusive: bool = True,
        wait: bool = False,
    ) -> Job:
        """Запустить задачу. wait=True — выполнить сразу (для тестов)."""
        with self._lock:
            if exclusive and any(
                j.status == "running" and j.kind == kind for j in self._jobs.values()
            ):
                raise RuntimeError(f"Уже идёт другая задача: {kind}")
            job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, title=title)
            job.request = dict(request or {})
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
                job.log_path = self.log_dir / f"{stamp}_{job.job_id}_{kind}.log"
            self._jobs[job.job_id] = job
        job.say(f"{job.started} старт: {title}")
        if wait:
            self._execute(job, work)
        else:
            threading.Thread(target=self._execute, args=(job, work), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.started, reverse=True)[:limit]

    def _execute(self, job: Job, work: Work) -> None:
        try:
            job.result = work(job)
            job.status = "done"
            job.say(f"{_now()} готово" + (f": {job.result}" if job.result else ""))
        except Exception as exc:  # ошибка показывается в интерфейсе, сервер продолжает работать
            job.error = str(exc) or type(exc).__name__
            job.status = "error"
            job.say(f"{_now()} ошибка: {job.error}")
        finally:
            job.finished = _now()
