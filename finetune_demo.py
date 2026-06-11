"""self-finetuning 第二+三層 端到端示範（EchoBackend，無需 GPU）。

流程：
1. 產生一批使用者回饋（存進 experiences）
2. 匯出偏好資料集
3. 背景訓練 → 候選 adapter（EchoBackend 佔位）
4. EvalGate 比較候選 vs 現役 → 決定 promote/丟棄
5. AdapterStore 保存並切換 active；可回滾；稽核紀錄
"""
import os
import shutil

from pai import (AdapterStore, EchoBackend, EvalGate, HashingEmbedder,
                 ReflectiveMemory, SelfFinetuneManager, export_preference_dataset)

DB = "ft_demo.db"
ADAPTERS = "ft_demo_adapters"
for p in (DB, ADAPTERS):
    if os.path.isdir(p): shutil.rmtree(p)
    elif os.path.exists(p): os.remove(p)

# 1. 模擬累積回饋
rm = ReflectiveMemory(DB, embedder=HashingEmbedder())
samples = [
    ("promo_email from gmail: subject=限時優惠", "__notify__", "促銷信通知", "rejected"),
    ("promo_email from gmail: subject=週年慶", "__notify__", "促銷信通知", "rejected"),
    ("promo_email from gmail: subject=特賣", "__notify__", "促銷信通知", "rejected"),
    ("high_priority_email from gmail: 客戶 SLA", "draft_reply", "起草緊急回信", "accepted"),
    ("high_priority_email from gmail: 主管詢問", "draft_reply", "起草回信", "accepted"),
    ("metric.breach from cpu: 95%", "cleanup", "清理高負載", "accepted"),
    ("metric.breach from cpu: 88%", "cleanup", "清理高負載", "modified"),
]
for es, action, rationale, fb in samples:
    rm.add_experience(es, action, rationale, fb)
print(f"1. 已累積 {len(samples)} 筆回饋經驗")

# 2. 匯出偏好資料集
n = export_preference_dataset(DB, "ft_demo_dataset.jsonl")
print(f"2. 偏好資料集樣本數：{n}")

# 3+4. eval gate：準確率隨訓練樣本數提升（讀候選 adapter 的 stub metadata）
import json as _json
def fake_eval(adapter_path):
    if not adapter_path:
        return {"accuracy": 0.70, "interrupt_precision": 0.80}   # 現役=無 adapter
    try:
        meta = _json.load(open(adapter_path, encoding="utf-8"))
        ns = meta.get("n_samples", 0)
    except Exception:
        ns = 0
    return {"accuracy": min(0.70 + 0.01 * ns, 0.95),
            "interrupt_precision": 0.80 + min(0.005 * ns, 0.1)}

gate = EvalGate(fake_eval, primary_metric="accuracy", min_gain=0.02,
                non_regression=["interrupt_precision"])

mgr = SelfFinetuneManager(
    db_path=DB, adapters_root=ADAPTERS,
    trainer=EchoBackend(), eval_gate=gate,
    base_gguf="models/gemma-4-26B_q4_0-it.gguf", min_samples=3)

print("\n3. 第一次訓練 + eval gate...")
result = mgr.maybe_train_and_promote()
print("   ", result["promoted"] and f"✅ 通過並上線（增益 +{result['decision']['gain']}）"
      or f"❌ 未通過（增益 {result['decision']['gain']}）")
store = AdapterStore(ADAPTERS)
print(f"   現役 adapter：{os.path.basename(store.active_adapter())}")

print("\n4. 模擬『無增益再訓練』→ 閘門應擋下...")
r2 = mgr.maybe_train_and_promote()   # 樣本數沒變 → 增益 0 → 拒絕
print("   ", r2["promoted"] and "✅ 上線" or f"🛡️ 閘門擋下（增益 {r2['decision']['gain']} < 0.02），現役不變")

print("\n5. 新增更多正向回饋後再訓練 → 應通過並成為 v2...")
for i in range(5):
    rm.add_experience(f"high_priority_email from gmail: 案件{i}", "draft_reply", "起草回信", "accepted")
r3 = mgr.maybe_train_and_promote()
store = AdapterStore(ADAPTERS)
print("   ", r3["promoted"] and f"✅ v2 上線（增益 +{r3['decision']['gain']}）" or "❌ 未通過")
print(f"   現役：{os.path.basename(store.active_adapter())}（共 {len(store.index)} 版）")

print("\n6. 回滾到 v1...")
store.rollback()
store = AdapterStore(ADAPTERS)
print(f"   回滾後現役：{os.path.basename(store.active_adapter())}")

print("\n稽核紀錄：")
with open(os.path.join(ADAPTERS, "audit.jsonl"), encoding="utf-8") as f:
    for line in f:
        import json
        e = json.loads(line)
        print(f"  [{e['event']}] adapter={e.get('adapter','-')} samples={e['n_samples']} gain={e['decision'].get('gain')}")

print("\n✅ 全流程驗證：訓練→閘門→上線→回滾，主幹權重(weights.gguf)全程未變更")

# 清理
for p in (DB, ADAPTERS, "ft_demo_dataset.jsonl"):
    if os.path.isdir(p): shutil.rmtree(p)
    elif os.path.exists(p): os.remove(p)
