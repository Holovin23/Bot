"""Запуск бэктестов из веб-интерфейса в фоне. Одновременно идёт один бэктест."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# (стратегия, символ, таймфрейм, начало, конец, параметры) -> id папки бэктеста
RunFn = Callable[[str, str, str, str, str, dict[str, Any]], str]


@dataclass
class Job:
    job_id: str
    status: str = "running"  # running | done | error
    backtest_id: str | None = None
    error: str | None = None
    request: dict[str, Any] = field(default_factory=dict)


class JobRunner:
    def __init__(self, run: RunFn) -> None:
        self._run = run
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def busy(self) -> bool:
        with self._lock:
            return any(j.status == "running" for j in self._jobs.values())

    def start(self, request: dict[str, Any], wait: bool = False) -> Job:
        """Запустить бэктест. wait=True — выполнить сразу (для тестов)."""
        with self._lock:
            if any(j.status == "running" for j in self._jobs.values()):
                raise RuntimeError("Уже идёт другой бэктест")
            job = Job(job_id=uuid.uuid4().hex[:12], request=dict(request))
            self._jobs[job.job_id] = job
        if wait:
            self._execute(job)
        else:
            threading.Thread(target=self._execute, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _execute(self, job: Job) -> None:
        r = job.request
        try:
            job.backtest_id = self._run(
                r["strategy"], r["symbol"], r["timeframe"], r["start"], r["end"], r["params"]
            )
            job.status = "done"
        except Exception as exc:  # ошибка показывается в интерфейсе, сервер продолжает работать
            job.error = str(exc)
            job.status = "error"
