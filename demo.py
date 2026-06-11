"""PAI 框架示範：一個會主動監控、建議、確認與自動行動的代理。

情境模擬：
1. ThresholdTrigger 監控一個模擬的「CPU 使用率」，飆高時 → 高信心+高緊急 → ASK（請求確認後清理）
2. IntervalTrigger 心跳檢查「待辦數量」，太多時 → SUGGEST（只通知不行動）
3. FileWatchTrigger 監看 ./watched 目錄，新檔案 → ACT（自動歸檔，低風險動作）

全程使用 RuleBrain（離線可跑）。設 ANTHROPIC_API_KEY 後可換 LLMBrain。
"""
import logging
import os
import random
import shutil

from pai import (
    AutonomyLevel, CallbackAction, ConsoleNotifier, Intent, IntervalTrigger,
    Memory, PAIAgent, ProactivityPolicy, Rule, RuleBrain, FileWatchTrigger,
    ThresholdTrigger,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

BASE = os.path.dirname(os.path.abspath(__file__))
WATCH_DIR = os.path.join(BASE, "watched")
ARCHIVE_DIR = os.path.join(BASE, "archive")
os.makedirs(WATCH_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# ---- 模擬資料來源 ----
_sim = {"cpu": 30.0, "todos": 3}

def fake_cpu() -> float:
    _sim["cpu"] = min(100.0, max(5.0, _sim["cpu"] + random.uniform(-10, 18)))
    return _sim["cpu"]

# ---- 規則腦 ----
def rule_cpu(event, context):
    if event.kind == "metric.breach" and event.source == "cpu-monitor":
        return Intent(
            action="cleanup",
            params={"target": "cpu", "value": event.payload["value"]},
            confidence=0.9, urgency=0.95,
            rationale=f"CPU 使用率 {event.payload['value']:.0f}% 超過 {event.payload['threshold']:.0f}%，建議清理背景程序",
            requested_level=AutonomyLevel.ASK,
        )

def rule_todos(event, context):
    if event.kind == "schedule.tick" and event.payload.get("todos", 0) > 5:
        return Intent(
            action="__notify__",
            params={"title": "待辦堆積", "body": f"目前有 {event.payload['todos']} 件待辦，建議安排處理"},
            confidence=0.7, urgency=0.4,
            rationale="待辦數量超過 5 件",
            requested_level=AutonomyLevel.SUGGEST,
        )

def rule_new_file(event, context):
    if event.kind == "file.created":
        return Intent(
            action="archive_file",
            params={"path": event.payload["path"]},
            confidence=0.95, urgency=0.3,
            rationale=f"偵測到新檔案 {os.path.basename(event.payload['path'])}，自動歸檔",
            requested_level=AutonomyLevel.ACT,
        )

brain = RuleBrain([
    Rule("cpu-breach", rule_cpu),
    Rule("todo-pileup", rule_todos),
    Rule("auto-archive", rule_new_file),
])

# ---- 動作 ----
def do_cleanup(intent: Intent):
    _sim["cpu"] = 25.0
    print(f"   🧹 已清理（{intent.params['target']}），CPU 回到 {_sim['cpu']:.0f}%")
    return {"cleaned": True}

def do_archive(intent: Intent):
    src = intent.params["path"]
    dst = os.path.join(ARCHIVE_DIR, os.path.basename(src))
    shutil.move(src, dst)
    print(f"   📦 已自動歸檔 → {dst}")
    return {"archived": dst}

actions = {
    "__notify__": ConsoleNotifier(),
    "cleanup": CallbackAction(do_cleanup),
    "archive_file": CallbackAction(do_archive),
}

# ---- 治理政策 ----
policy = ProactivityPolicy(
    min_confidence=0.4,
    act_confidence=0.85,
    default_max_level=AutonomyLevel.SUGGEST,
    action_max_levels={
        "cleanup": AutonomyLevel.ASK,        # 有副作用 → 最多問過再做
        "archive_file": AutonomyLevel.ACT,   # 低風險 → 允許全自動
        "__notify__": AutonomyLevel.SUGGEST,
    },
    quiet_hours=None,
    max_interruptions_per_hour=10,
    # 干擾成本估計：正式版可依「使用者是否在專注模式/開會」動態回傳 0~1
    interruption_cost_fn=lambda: 0.2,
)

# ---- 確認處理器（demo 自動同意；正式版接 UI/聊天確認）----
def confirm(intent: Intent) -> bool:
    print(f"\n❓ [PAI 請求確認] {intent.rationale}\n   → demo 自動同意")
    return True

# ---- 組裝代理 ----
agent = PAIAgent(
    name="demo-pai",
    brain=brain,
    policy=policy,
    memory=Memory(os.path.join(BASE, "pai_memory.db")),
    actions=actions,
    confirm_handler=confirm,
)
agent.add_trigger(ThresholdTrigger("cpu-monitor", fake_cpu, threshold=75, check_interval=1))
agent.add_trigger(IntervalTrigger("todo-check", interval_sec=4,
                                  payload_fn=lambda: {"todos": random.randint(2, 9)}))
agent.add_trigger(FileWatchTrigger("inbox-watch", WATCH_DIR))

if __name__ == "__main__":
    import json

    # 模擬外部放入一個新檔案
    with open(os.path.join(WATCH_DIR, "report.txt"), "w") as f:
        f.write("demo file")
    print("=== PAI demo：運行 15 秒 ===")
    agent.run(duration=15, poll_interval=0.5)

    print("\n=== 最新一筆 PAI Protocol 標準紀錄 ===")
    records = agent.memory.latest_protocol_records(1)
    if records:
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
    print("\n=== demo 結束（事件/意圖/Protocol 紀錄已寫入 pai_memory.db）===")
