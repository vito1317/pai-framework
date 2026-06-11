"""把 Gemma 4 26B-A4B QAT q4_0 打包成可離線運行的 .pai agent。"""
import os

from pai import pack_agent

BASE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE, "models", "gemma-4-26B_q4_0-it.gguf")
OUT = os.path.join(BASE, "gemma-guardian.pai")

path = pack_agent(
    OUT,
    manifest={
        "name": "gemma-guardian",
        "version": "1.0.0",
        "author": "vito1317 <service@vito1317.com>",
        "description": "以 Gemma 4 26B-A4B (QAT q4_0) 為本地決策腦的主動式 agent",
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
    triggers=[
        {"type": "interval", "name": "inbox-pulse", "params": {"interval_sec": 10}},
        {"type": "filewatch", "name": "inbox-watch", "params": {"path": "./watched"}},
    ],
    rules=[
        # LLM 失敗時的安全 fallback 規則
        {"when": {"kind": "file.created"},
         "intent": {"action": "archive_file", "confidence": 0.95, "urgency": 0.3,
                    "level": 3, "rationale": "新檔案自動歸檔（規則 fallback）"}},
    ],
    actions={
        "cleanup": {"type": "callback", "handler": "ops.cleanup"},
        "archive_file": {"type": "callback", "handler": "ops.archive"},
        "__notify__": {"type": "console"},
    },
    brain={
        "type": "llm",
        "n_ctx": 4096,
        "user_profile": "工程師，工作時間 9-18，偏好最少打擾；高風險動作必須先確認",
    },
    weights_path=WEIGHTS,
)
print(f"✅ 已打包：{path}（{os.path.getsize(path):,} bytes）")
