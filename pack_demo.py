"""示範：把一個主動式 agent 打包成 .pai 檔，再載入驗證。"""
import os

from pai import PaiReader, load_agent, pack_agent

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "ops-guardian.pai")

# 模擬一個嵌入的「模型權重」二進位段（實務上放 .gguf 檔路徑）
fake_weights = os.path.join(BASE, "_fake_weights.gguf")
with open(fake_weights, "wb") as f:
    f.write(os.urandom(256 * 1024))   # 256 KB 假權重

path = pack_agent(
    OUT,
    manifest={
        "name": "ops-guardian",
        "version": "1.0.0",
        "author": "vito1317 <service@vito1317.com>",
        "description": "監控系統指標並主動處理異常的 PAI agent",
        "framework_compat": ">=0.1.0",
    },
    policy={
        "min_confidence": 0.4, "act_confidence": 0.85,
        "default_max_level": 1,
        "action_max_levels": {"cleanup": 2, "archive_file": 3},
        "quiet_hours": [22, 8], "max_interruptions_per_hour": 6,
    },
    triggers=[
        {"type": "threshold", "name": "cpu-monitor",
         "params": {"metric": "cpu", "threshold": 75, "direction": "above"}},
        {"type": "interval", "name": "todo-check", "params": {"interval_sec": 300}},
        {"type": "filewatch", "name": "inbox-watch", "params": {"path": "~/inbox"}},
    ],
    rules=[
        {"when": {"kind": "metric.breach", "source": "cpu-monitor"},
         "intent": {"action": "cleanup", "confidence": 0.9, "urgency": 0.95,
                    "level": 2, "rationale": "CPU 超標，建議清理"}},
        {"when": {"kind": "file.created"},
         "intent": {"action": "archive_file", "confidence": 0.95, "urgency": 0.3,
                    "level": 3, "rationale": "新檔案自動歸檔"}},
    ],
    actions={
        "cleanup": {"type": "callback", "handler": "ops.cleanup"},
        "archive_file": {"type": "callback", "handler": "ops.archive"},
        "__notify__": {"type": "console"},
    },
    brain={"type": "llm", "model": "claude-sonnet-4-6",
           "fallback": "rules", "user_profile": "SRE 工程師"},
    weights_path=fake_weights,
)
os.remove(fake_weights)

print(f"✅ 已打包：{path}（{os.path.getsize(path):,} bytes）\n")

r = PaiReader(path)
info = r.info()
print("段表：")
for name, meta in info["sections"].items():
    print(f"  {name:16s} stored={meta['stored_bytes']:>9,} raw={meta['raw_bytes']:>9,}")

agent_def = load_agent(path)
print(f"\nmanifest: {agent_def['manifest']['name']} v{agent_def['manifest']['version']}")
print(f"rules: {len(agent_def['rules'])} 條, triggers: {len(agent_def['triggers'])} 個")
print(f"has_weights: {agent_def['has_weights']}")

# 校驗測試：竄改一個 byte 應該被 SHA-256 抓到
raw = bytearray(open(path, "rb").read())
raw[-1] ^= 0xFF
with open(path + ".tampered", "wb") as f:
    f.write(raw)
try:
    PaiReader(path + ".tampered").read_bytes("weights.gguf")
    print("\n❌ 竄改未被偵測（不應發生）")
except ValueError as e:
    print(f"\n✅ 竄改偵測成功：{e}")
os.remove(path + ".tampered")
