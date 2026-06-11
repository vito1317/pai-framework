"""把 Gemma 4 26B-A4B QAT q4_0 打包成可離線運行的 guardian .pai agent。

設計原則（自洽）：每個 action 都有對應的 trigger 會產生它需要的事件，
且 llama-server 不可用退回 RuleBrain 時，每條規則仍可獨立運作。

guardian 行為：
- cpu-monitor (threshold) → metric.breach → cleanup（清理高負載，最多 ASK）
- inbox-watch (filewatch) → file.created → archive_file（自動歸檔，可 ACT）
"""
import os

from pai import pack_agent

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "models", "gemma-4-26B_q4_0-it.gguf")
OUT = os.path.join(BASE, "gemma-guardian.pai")

path = pack_agent(
    OUT,
    manifest={
        "name": "gemma-guardian",
        "version": "1.1.0",
        "author": "vito1317 <service@vito1317.com>",
        "description": "以 Gemma 4 26B-A4B (QAT q4_0) 為本地決策腦的主動式 guardian agent："
                       "監控系統指標並主動處理異常、自動歸檔新檔案",
        "framework_compat": ">=0.1.0",
        "base_model": "google/gemma-4-26B-A4B-it-qat-q4_0",
        "model_license": "apache-2.0",
    },
    policy={
        "min_confidence": 0.4, "act_confidence": 0.85,
        "default_max_level": 1,
        "action_max_levels": {"cleanup": 2, "archive_file": 3, "__notify__": 1},
        "max_interruptions_per_hour": 6,
    },
    # 每個 action 都有對應 trigger 來源（無死角）
    triggers=[
        {"type": "threshold", "name": "cpu-monitor",
         "params": {"metric": "cpu", "threshold": 75, "direction": "above",
                    "check_interval": 2}},
        {"type": "filewatch", "name": "inbox-watch", "params": {"path": "./watched"}},
    ],
    # RuleBrain fallback：兩條規則涵蓋兩個 action
    rules=[
        {"when": {"kind": "metric.breach", "source": "cpu-monitor"},
         "intent": {"action": "cleanup", "confidence": 0.9, "urgency": 0.95,
                    "level": 2, "rationale": "CPU 使用率超過閾值，建議清理背景程序"}},
        {"when": {"kind": "file.created"},
         "intent": {"action": "archive_file", "confidence": 0.95, "urgency": 0.3,
                    "level": 3, "rationale": "偵測到新檔案，自動歸檔"}},
    ],
    actions={
        "cleanup": {"type": "callback", "handler": "ops.cleanup"},
        "archive_file": {"type": "callback", "handler": "ops.archive"},
        "__notify__": {"type": "console"},
    },
    brain={
        "type": "llm",
        "engine": "llama-server",
        # 刻意設小：決策 payload 很短，省記憶體（模型原生支援 256K）
        "n_ctx": 4096,
        "learning": True,
        "user_profile": "SRE/工程師，工作時間偏好最少打擾；高風險動作必須先確認",
    },
    weights_path=WEIGHTS,
)
print(f"✅ 已打包：{path}（{os.path.getsize(path):,} bytes）")
