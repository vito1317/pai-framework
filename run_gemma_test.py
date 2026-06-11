"""實跑測試：載入 gemma-guardian.pai，用內嵌的 Gemma 4 26B-A4B 做一次真實決策。"""
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from pai import load_runtime
from pai.core import Event

t0 = time.time()
agent = load_runtime(
    "gemma-guardian.pai",
    handlers={
        "ops.cleanup": lambda i: print("   🧹 cleanup 執行"),
        "ops.archive": lambda i: print("   📦 archive 執行"),
    },
    confirm_handler=lambda i: (print(f"❓ 確認: {i.rationale} → 同意"), True)[1],
    memory_path="gemma_test.db",
)
print(f"brain = {type(agent.brain).__name__}（載入耗時 {time.time()-t0:.1f}s，"
      f"權重抽取到快取後 llama.cpp 於首次決策時才載入）")

# 餵一個「高優先信件」事件，讓 Gemma 4 做真實的主動決策
event = Event(
    source="gmail_webhook",
    kind="high_priority_email_received",
    payload={
        "from": "AWS 專案客戶窗口",
        "summary": "明天上線的測試環境發現嚴重 Bug，語氣急迫，SLA 要求 1 小時內回覆",
    },
)
t1 = time.time()
agent._handle_event(event)
print(f"\n決策耗時 {time.time()-t1:.1f}s")

recs = agent.memory.latest_protocol_records(1)
if recs:
    import json
    print("\n=== Gemma 4 產生的 PAI Protocol 紀錄（3_anticipation）===")
    print(json.dumps(recs[0]["3_anticipation"], ensure_ascii=False, indent=2))
    print("delivery:", recs[0]["5_delivery"])
else:
    print("（模型判定不需主動行為，或意圖被 policy 降為 OBSERVE）")
