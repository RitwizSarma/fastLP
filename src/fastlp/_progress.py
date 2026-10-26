"""Low-overhead stage reporting for long fits."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    completed: int | None
    total: int | None
    elapsed_seconds: float
    message: str


class ProgressReporter:
    def __init__(
        self,
        enabled: bool,
        callback: Callable[[ProgressEvent], None] | None = None,
        *,
        interval: float = 5.0,
    ) -> None:
        self.enabled = enabled
        self.callback = callback
        self.interval = interval
        self.started = perf_counter()
        self.stage_name = ""
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None

    def __enter__(self) -> "ProgressReporter":
        if self.enabled:
            self._thread = Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self.enabled:
            status = "failed" if exc_type is not None else "complete"
            self._write(f"fit {status} ({perf_counter() - self.started:.1f}s)")

    def _write(self, message: str) -> None:
        with self._lock:
            print(f"fastLP: {message}", file=sys.stderr, flush=True)

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            if self.stage_name:
                self._write(
                    f"{self.stage_name}... {perf_counter() - self.started:.1f}s elapsed"
                )

    def update(
        self,
        stage: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        message: str = "",
    ) -> None:
        if not self.enabled and self.callback is None:
            return
        self.stage_name = stage
        event = ProgressEvent(stage, completed, total, perf_counter() - self.started, message)
        if self.enabled:
            counter = f" {completed}/{total}" if completed is not None and total is not None else ""
            detail = f" — {message}" if message else ""
            self._write(f"{stage}{counter}{detail} ({event.elapsed_seconds:.1f}s)")
        if self.callback is not None:
            self.callback(event)
