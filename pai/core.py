"""PAI 核心：事件模型、意圖模型、自主等級與主動行為迴圈。"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Optional

logger = logging.getLogger("pai")


class AutonomyLevel(IntEnum):
    """自主等級：決定 PAI 對一個意圖「可以主動到什麼程度」。

    OBSERVE: 只記錄，不打擾
    SUGGEST: 主動通知/建議，但不執行
    ASK:     請求使用者確認後執行
    ACT:     直接自動執行（高信心、低風險時）
    """
    OBSERVE = 0
    SUGGEST = 1
    ASK = 2
    ACT = 3


@dataclass
class Event:
    """感知層產生的標準事件格式。"""
    source: str                       # 觸發器名稱
    kind: str                         # 事件類型，如 "schedule.tick", "file.changed"
    payload: dict = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return {
            "id": self.id, "source": self.source, "kind": self.kind,
            "payload": self.payload, "ts": self.ts.isoformat(),
        }


def _event_summary(event: "Event") -> str:
    """把事件壓成一行文字，供記憶檢索使用。"""
    payload = " ".join(f"{k}={v}" for k, v in event.payload.items())
    return f"{event.kind} from {event.source}: {payload}".strip()


@dataclass
class Intent:
    """決策層輸出的標準意圖格式。"""
    action: str                       # 想執行的動作名稱
    params: dict = field(default_factory=dict)
    confidence: float = 0.5           # 0~1，決策信心
    urgency: float = 0.5              # 0~1，緊急程度
    rationale: str = ""               # 決策理由（可解釋性）
    requested_level: AutonomyLevel = AutonomyLevel.SUGGEST
    event_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "action": self.action, "params": self.params,
            "confidence": self.confidence, "urgency": self.urgency,
            "rationale": self.rationale,
            "requested_level": int(self.requested_level),
            "event_id": self.event_id,
        }


class PAIAgent:
    """主動式 AI 代理：把觸發器、決策腦、治理政策、行動與記憶組裝成迴圈。"""

    def __init__(self, name: str, brain, policy, memory, actions: dict[str, Any],
                 confirm_handler: Optional[Callable[[Intent], bool]] = None,
                 reflective=None):
        self.name = name
        self.brain = brain
        self.policy = policy
        self.memory = memory
        self.actions = actions            # {action_name: Action 實例}
        self.reflective = reflective      # ReflectiveMemory（記憶式即時學習，選用）
        self.triggers: list = []
        self.confirm_handler = confirm_handler or (lambda intent: False)
        self._stop = threading.Event()

    # ---- 組裝 ----
    def add_trigger(self, trigger) -> "PAIAgent":
        self.triggers.append(trigger)
        return self

    # ---- 主迴圈 ----
    def run(self, duration: Optional[float] = None, poll_interval: float = 0.5):
        """啟動主動行為迴圈。duration=None 表示持續運行。"""
        logger.info("[%s] PAI agent started (%d triggers)", self.name, len(self.triggers))
        start = time.time()
        try:
            while not self._stop.is_set():
                for trigger in self.triggers:
                    for event in trigger.poll():
                        self._handle_event(event)
                if duration is not None and time.time() - start >= duration:
                    break
                time.sleep(poll_interval)
        finally:
            logger.info("[%s] PAI agent stopped", self.name)

    def stop(self):
        self._stop.set()

    # ---- 單一事件的完整生命週期 ----
    def _handle_event(self, event: Event):
        self.memory.record_event(event)
        context = self.memory.build_context(event)

        # 記憶式即時學習：把相似情境的回饋教訓注入決策上下文
        if self.reflective is not None:
            lessons = self.reflective.lessons_for(_event_summary(event))
            if lessons:
                context["lessons_from_feedback"] = lessons

        intents = self.brain.decide(event, context)
        for intent in intents:
            intent.event_id = event.id
            self._dispatch(intent, event)

    def _dispatch(self, intent: Intent, event: Event):
        from .protocol import build_record

        context = self.memory.build_context(event)
        granted = self.policy.gate(intent, self.memory)
        cost = self.policy.last_interruption_cost
        self.memory.record_intent(intent, granted=int(granted))

        execution = {"actions_taken": [], "status": "not_executed"}
        feedback = "pending"

        if granted == AutonomyLevel.OBSERVE:
            logger.debug("OBSERVE only: %s (%s)", intent.action, intent.rationale)
            execution["status"] = "observed"

        elif granted == AutonomyLevel.SUGGEST:
            notifier = self.actions.get("__notify__")
            if notifier:
                notifier.execute(Intent(
                    action="__notify__",
                    params={"title": f"建議：{intent.action}",
                            "body": intent.rationale, "intent": intent.to_dict()},
                    confidence=intent.confidence, urgency=intent.urgency,
                ))
            self.memory.record_outcome(intent, status="suggested")
            execution["status"] = "suggested"

        else:  # ASK 或 ACT
            action = self.actions.get(intent.action)
            if action is None:
                logger.warning("No action registered for intent '%s'", intent.action)
                execution["status"] = "no_action_registered"
            else:
                approved = True
                if granted == AutonomyLevel.ASK:
                    approved = self.confirm_handler(intent)
                    if not approved:
                        self.memory.record_outcome(intent, status="declined")
                        self.memory.record_feedback(intent, positive=False)
                        execution["status"] = "declined"
                        feedback = "rejected"
                        self._learn(event, intent, "rejected")
                if approved:
                    try:
                        result = action.execute(intent)
                        self.memory.record_outcome(intent, status="executed", result=result)
                        execution["actions_taken"].append(
                            {"tool": intent.action, "status": "ok", "result": str(result)})
                        execution["status"] = "executed"
                        logger.info("Executed '%s' (confidence=%.2f): %s",
                                    intent.action, intent.confidence, intent.rationale)
                    except Exception as exc:  # noqa: BLE001
                        self.memory.record_outcome(intent, status="failed", result=str(exc))
                        execution["status"] = "failed"
                        logger.exception("Action '%s' failed", intent.action)

        # ACT 成功且使用者沒有否決 → 視為一次正向經驗
        if execution["status"] == "executed" and granted == AutonomyLevel.ACT:
            self._learn(event, intent, "accepted")

        # 產生並保存 PAI Protocol 標準紀錄
        record = build_record(event, context, intent, granted, cost,
                              execution=execution, user_feedback=feedback)
        self.memory.record_protocol(record)

    # ---- 記憶式學習：對外的回饋 API ----
    def record_user_feedback(self, event: Event, intent: Intent, feedback: str):
        """外部 UI 收到使用者回饋（accepted/rejected/modified/ignored）時呼叫。"""
        self._learn(event, intent, feedback)

    def _learn(self, event: Event, intent: Intent, feedback: str):
        if self.reflective is not None:
            self.reflective.add_experience(
                _event_summary(event), intent.action, intent.rationale, feedback)
