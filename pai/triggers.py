"""感知層：觸發器。每個觸發器的 poll() 回傳 0..n 個標準 Event。"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Callable, Iterable, Optional

from .core import Event


class IntervalTrigger:
    """固定間隔觸發（心跳/輪詢型主動性）。"""

    def __init__(self, name: str, interval_sec: float,
                 payload_fn: Optional[Callable[[], dict]] = None):
        self.name = name
        self.interval = interval_sec
        self.payload_fn = payload_fn or (lambda: {})
        self._last = 0.0

    def poll(self) -> Iterable[Event]:
        now = time.time()
        if now - self._last >= self.interval:
            self._last = now
            yield Event(source=self.name, kind="schedule.tick", payload=self.payload_fn())


class ScheduleTrigger:
    """每日定時觸發（HH:MM）。"""

    def __init__(self, name: str, at: str, payload: Optional[dict] = None):
        self.name = name
        self.at = at  # "HH:MM"
        self.payload = payload or {}
        self._fired_date: Optional[str] = None

    def poll(self) -> Iterable[Event]:
        now = datetime.now()
        if now.strftime("%H:%M") == self.at and self._fired_date != now.strftime("%Y-%m-%d"):
            self._fired_date = now.strftime("%Y-%m-%d")
            yield Event(source=self.name, kind="schedule.daily",
                        payload={"at": self.at, **self.payload})


class FileWatchTrigger:
    """監看檔案/目錄變動（mtime 比對，零依賴）。"""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self._snapshot = self._scan()

    def _scan(self) -> dict[str, float]:
        snap: dict[str, float] = {}
        if os.path.isfile(self.path):
            snap[self.path] = os.path.getmtime(self.path)
        elif os.path.isdir(self.path):
            for root, _dirs, files in os.walk(self.path):
                for f in files:
                    p = os.path.join(root, f)
                    try:
                        snap[p] = os.path.getmtime(p)
                    except OSError:
                        pass
        return snap

    def poll(self) -> Iterable[Event]:
        current = self._scan()
        for p, mtime in current.items():
            old = self._snapshot.get(p)
            if old is None:
                yield Event(source=self.name, kind="file.created", payload={"path": p})
            elif mtime > old:
                yield Event(source=self.name, kind="file.changed", payload={"path": p})
        for p in self._snapshot:
            if p not in current:
                yield Event(source=self.name, kind="file.deleted", payload={"path": p})
        self._snapshot = current


class ThresholdTrigger:
    """數值監控觸發：metric_fn 超過/低於閾值時發事件（含遲滯避免抖動）。"""

    def __init__(self, name: str, metric_fn: Callable[[], float],
                 threshold: float, direction: str = "above",
                 check_interval: float = 1.0):
        self.name = name
        self.metric_fn = metric_fn
        self.threshold = threshold
        self.direction = direction      # "above" | "below"
        self.check_interval = check_interval
        self._last_check = 0.0
        self._in_breach = False

    def poll(self) -> Iterable[Event]:
        now = time.time()
        if now - self._last_check < self.check_interval:
            return
        self._last_check = now
        value = self.metric_fn()
        breach = value > self.threshold if self.direction == "above" else value < self.threshold
        if breach and not self._in_breach:
            self._in_breach = True
            yield Event(source=self.name, kind="metric.breach",
                        payload={"value": value, "threshold": self.threshold,
                                 "direction": self.direction})
        elif not breach and self._in_breach:
            self._in_breach = False
            yield Event(source=self.name, kind="metric.recovered",
                        payload={"value": value, "threshold": self.threshold})
