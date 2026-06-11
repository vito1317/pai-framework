"""記憶式即時學習示範（零依賴，不需模型）。

模擬：使用者連續拒絕「半夜促銷信通知」這類建議，
之後遇到相似事件時，ReflectiveMemory 檢索出教訓並注入決策腦，
決策腦據此改判（這裡用一個會讀取 lessons 的簡單規則腦示意）。
"""
import os

from pai import HashingEmbedder, ReflectiveMemory
from pai.core import Event, Intent, AutonomyLevel, PAIAgent, _event_summary
from pai.brain import RuleBrain, Rule
from pai.policy import ProactivityPolicy
from pai.memory import Memory
from pai.actions import ConsoleNotifier, CallbackAction

DB = "learn_demo.db"
for ext in ("", "-wal", "-shm"):
    try: os.remove(DB + ext)
    except OSError: pass

reflective = ReflectiveMemory(DB, embedder=HashingEmbedder())

# 決策腦：對「促銷信」事件建議通知；但若記憶中已有負面教訓則自動退讓
def rule_promo(event, context):
    if event.kind != "promo_email":
        return None
    lessons = context.get("lessons_from_feedback", "")
    if "使用者拒絕" in lessons:
        # 學到教訓：這類建議使用者不喜歡 → 降為 OBSERVE，不打擾
        return Intent(action="__notify__",
                      params={"title": "促銷信", "body": event.payload.get("subject", "")},
                      confidence=0.5, urgency=0.2,
                      rationale="偵測到促銷信（但過往使用者拒絕此類通知，改為靜默觀察）",
                      requested_level=AutonomyLevel.OBSERVE)
    return Intent(action="__notify__",
                  params={"title": "促銷信", "body": event.payload.get("subject", "")},
                  confidence=0.7, urgency=0.5,
                  rationale="偵測到促銷信，主動通知使用者",
                  requested_level=AutonomyLevel.SUGGEST)

agent = PAIAgent(
    name="learn-demo",
    brain=RuleBrain([Rule("promo", rule_promo)]),
    policy=ProactivityPolicy(min_confidence=0.3, interruption_cost_fn=lambda: 0.1),
    memory=Memory(DB),
    actions={"__notify__": ConsoleNotifier()},
    confirm_handler=lambda i: False,
    reflective=reflective,
)

def fire(subject):
    e = Event(source="gmail", kind="promo_email", payload={"subject": subject})
    agent._handle_event(e)
    return e

print("=== 第 1~3 次：促銷信通知，使用者每次都拒絕（模擬回饋）===")
for subj in ["限時 5 折優惠！", "週年慶下殺", "黑色星期五特賣"]:
    e = fire(subj)
    # 模擬使用者回饋：拒絕這類通知
    agent.record_user_feedback(e, Intent(action="__notify__", rationale="促銷信通知"), "rejected")
    print(f"  → 已記錄負面回饋：{subj}")

print("\n=== 第 4 次：新的促銷信，PAI 應已從回饋學到要靜默 ===")
e = Event(source="gmail", kind="promo_email", payload={"subject": "年中慶買一送一"})
lessons = reflective.lessons_for(_event_summary(e))
print("檢索到的教訓：")
print("  " + lessons.replace("\n", "\n  "))
agent._handle_event(e)

recs = agent.memory.latest_protocol_records(1)
if recs:
    a = recs[0]["3_anticipation"]
    print(f"\n本次決策：{a['action']} | granted_level={recs[0]['5_delivery']['delivery_mode']}")
    print(f"理由：{a['predicted_intent']}")
    print("\n✅ 學習生效：從『主動通知(SUGGEST)』自我校準為『靜默觀察(level_0)』，全程未動模型權重")

for ext in ("", "-wal", "-shm"):
    try: os.remove(DB + ext)
    except OSError: pass
