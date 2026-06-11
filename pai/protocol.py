"""PAI Protocol v1.1 — 主動式 AI 標準資料格式。

每一次 PAI 的主動行為生命週期，都會產生一份 6 層 JSON 紀錄：
1_perception   感知層：發生了什麼事（取代傳統 prompt）
2_context      脈絡層：使用者狀態與相關記憶
3_anticipation 預判層：意圖、緊急度、信心、干擾成本
4_execution    執行層：背景做了什麼
5_delivery     交付層：用什麼等級通知/打擾使用者
6_adaptation   學習層：使用者回饋與權重調整

交付等級（delivery_mode）：
  level_0_silent     無聲執行，只留日誌
  level_1_soft_nudge 微光提示/非阻斷通知
  level_2_approval   彈出草稿，請求人類授權
  level_3_interrupt  強制打斷（僅限極高緊急度）
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from .core import AutonomyLevel, Event, Intent

PAI_PROTOCOL_VERSION = "1.1"

# 自主等級 → 交付模式 對應
LEVEL_TO_DELIVERY = {
    AutonomyLevel.OBSERVE: "level_0_silent",
    AutonomyLevel.SUGGEST: "level_1_soft_nudge",
    AutonomyLevel.ASK: "level_2_approval",
    AutonomyLevel.ACT: "level_0_silent",   # 自動執行 = 無聲完成，事後可查
}


def build_record(
    event: Event,
    context: dict,
    intent: Intent,
    granted: AutonomyLevel,
    interruption_cost: float,
    execution: Optional[dict] = None,
    user_feedback: str = "pending",
) -> dict:
    """組裝一份符合 PAI Protocol 的完整紀錄。"""
    return {
        "pai_protocol_version": PAI_PROTOCOL_VERSION,
        "record_id": f"pai_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),

        "1_perception": {
            "trigger_source": event.source,
            "event_type": event.kind,
            "raw_data_summary": event.payload,
        },
        "2_context": {
            "user_current_state": context.get("user_state", "unknown"),
            "relevant_memory": context.get("recent_events", [])[:5],
            "action_history": context.get("action_history", []),
        },
        "3_anticipation": {
            "predicted_intent": intent.rationale,
            "action": intent.action,
            "params": intent.params,
            "urgency_score": intent.urgency,
            "confidence_score": intent.confidence,
            "interruption_cost": interruption_cost,
            "requested_level": int(intent.requested_level),
            "granted_level": int(granted),
        },
        "4_execution": execution or {"actions_taken": [], "status": "not_executed"},
        "5_delivery": {
            "delivery_mode": LEVEL_TO_DELIVERY[granted],
            "requires_human_approval": granted == AutonomyLevel.ASK,
        },
        "6_adaptation": {
            "user_feedback": user_feedback,   # accepted | rejected | modified | ignored | pending
            "learning_adjustment": None,
        },
    }


def to_json(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2)


# ---- .pai.json 檔案：單筆 PAI Protocol 行為紀錄（UTF-8 JSON）----
# 注意：`.pai` 附檔名保留給二進位打包容器（見 paifile.py）

def save_pai(record: dict, directory: str = ".") -> str:
    """把一筆 PAI Protocol 紀錄存成 <record_id>.pai.json 檔案，回傳路徑。"""
    import os
    path = os.path.join(directory, f"{record['record_id']}.pai.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(record))
    return path


def load_pai(path: str) -> dict:
    """讀取 .pai.json 檔案並做基本格式驗證。"""
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    required = {"pai_protocol_version", "record_id", "timestamp",
                "1_perception", "2_context", "3_anticipation",
                "4_execution", "5_delivery", "6_adaptation"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"Invalid .pai file, missing keys: {sorted(missing)}")
    return record
