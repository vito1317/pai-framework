"""把 openbmb/MiniCPM-o-4_5 打包成全雙工 omni guardian .pai agent。

MiniCPM-o 4.5（9B，omni：影像+音訊輸入、文字+語音輸出、原生主動互動）。
本包不內嵌權重（transformers 路徑從 HF 載入），所以 .pai 很小，只帶
agent 設定；若要離線單檔發布，把 weights_path 指到 GGUF 即可（見下方註解）。

guardian 行為（全雙工場景）：
- omni-duplex (持續感知) → proactive_utterance → 由治理層決定是否打擾
- 仍保留 filewatch → archive_file 作為非語音的離散規則範例
"""
import os

from pai import pack_agent

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "minicpm-o-guardian.pai")

# 若要離線單檔（內嵌 GGUF），下載 openbmb/MiniCPM-o-4_5-gguf 後設定此路徑；
# 預設 None → transformers 路徑從 HF hub 載入 openbmb/MiniCPM-o-4_5
WEIGHTS = None  # e.g. os.path.join(BASE, "models", "MiniCPM-o-4_5-Q4_K_M.gguf")

kwargs = dict(
    manifest={
        "name": "minicpm-o-guardian",
        "version": "1.0.0",
        "author": "vito1317 <service@vito1317.com>",
        "description": "以 MiniCPM-o 4.5（全雙工 omni）為決策腦的主動式 agent："
                       "持續看＋聽現場，主動發起提醒/評論，並可語音回應",
        "framework_compat": ">=0.1.0",
        "base_model": "openbmb/MiniCPM-o-4_5",
        "model_license": "apache-2.0",
        "modality": "omni (video+audio in, text+speech out, full-duplex)",
    },
    policy={
        "min_confidence": 0.5, "act_confidence": 0.9,
        "default_max_level": 1,                 # 主動發話預設只到「建議」
        "action_max_levels": {"speak": 1, "remind": 2, "archive_file": 3, "__notify__": 1},
        "quiet_hours": [22, 8],
        "max_interruptions_per_hour": 8,
    },
    triggers=[
        # 全雙工持續感知由 DuplexOmniLoop 在執行期注入；這裡保留離散規則範例
        {"type": "filewatch", "name": "inbox-watch", "params": {"path": "./watched"}},
    ],
    rules=[
        # RuleBrain fallback：模型主動輸出時的保底處理
        {"when": {"kind": "proactive_utterance"},
         "intent": {"action": "speak", "confidence": 0.6, "urgency": 0.4,
                    "level": 1, "rationale": "模型主動發話，建議轉達使用者"}},
        {"when": {"kind": "file.created"},
         "intent": {"action": "archive_file", "confidence": 0.95, "urgency": 0.3,
                    "level": 3, "rationale": "偵測到新檔案，自動歸檔"}},
    ],
    actions={
        "speak": {"type": "console"},           # 正式版接 TTS 播放
        "remind": {"type": "console"},
        "archive_file": {"type": "callback", "handler": "ops.archive"},
        "__notify__": {"type": "console"},
    },
    brain={
        "type": "llm",
        "engine": "minicpm-o",
        "model": "openbmb/MiniCPM-o-4_5",
        "omni_engine": "transformers",          # 或 "llama-server"（需內嵌/指定 GGUF）
        "device": "cuda",                        # Apple Silicon 用 "mps"
        "init_vision": True, "init_audio": True, "init_tts": True,
        "learning": True,
        "user_profile": "偏好簡短、最少打擾；高風險動作先確認",
    },
)
if WEIGHTS:
    kwargs["weights_path"] = WEIGHTS

path = pack_agent(OUT, **kwargs)
print(f"✅ 已打包：{path}（{os.path.getsize(path):,} bytes）"
      + ("（含內嵌 GGUF）" if WEIGHTS else "（不含權重，transformers 從 HF 載入）"))
