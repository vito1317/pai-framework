"""把 MiniCPM-o 4.5 全套 omni 權重（LLM + vision + audio + tts + token2wav）
內嵌進單一 .pai，做成離線、可單檔發布的全雙工 omni agent。

權重段命名慣例（PAI 格式支援任意具名段）：
  weights.gguf              主 LLM（決策腦，Q4_K_M）
  weights/vision.gguf       視覺投影（mmproj）
  weights/audio.gguf        音訊編碼器
  weights/tts.gguf          TTS 主體
  weights/tts-projector.gguf
  weights/token2wav/*.gguf  token2wav 元件（encoder/flow/hifigan/...）
"""
import os

from pai.paifile import PaiWriter

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models", "minicpm-o")
OUT = os.path.join(BASE, "minicpm-o-omni-guardian.pai")

# (.pai 段名, 來源檔相對路徑)
WEIGHT_FILES = [
    ("weights.gguf",                 "MiniCPM-o-4_5-Q4_K_M.gguf"),
    ("weights/vision.gguf",          "vision/MiniCPM-o-4_5-vision-F16.gguf"),
    ("weights/audio.gguf",           "audio/MiniCPM-o-4_5-audio-F16.gguf"),
    ("weights/tts.gguf",             "tts/MiniCPM-o-4_5-tts-F16.gguf"),
    ("weights/tts-projector.gguf",   "tts/MiniCPM-o-4_5-projector-F16.gguf"),
    ("weights/token2wav/encoder.gguf",       "token2wav-gguf/encoder.gguf"),
    ("weights/token2wav/flow_extra.gguf",    "token2wav-gguf/flow_extra.gguf"),
    ("weights/token2wav/flow_matching.gguf", "token2wav-gguf/flow_matching.gguf"),
    ("weights/token2wav/hifigan2.gguf",      "token2wav-gguf/hifigan2.gguf"),
    ("weights/token2wav/prompt_cache.gguf",  "token2wav-gguf/prompt_cache.gguf"),
]

manifest = {
    "pai_format": 1, "kind": "proactive-agent",
    "name": "minicpm-o-omni-guardian", "version": "1.0.0",
    "author": "vito1317 <service@vito1317.com>",
    "description": "內嵌 MiniCPM-o 4.5 全套 omni 權重（LLM+vision+audio+tts+token2wav）"
                   "的離線全雙工主動式 agent",
    "base_model": "openbmb/MiniCPM-o-4_5-gguf", "model_license": "apache-2.0",
    "modality": "omni (video+audio in, text+speech out, full-duplex)",
    "weight_layout": {seg: src for seg, src in WEIGHT_FILES},
}
policy = {"min_confidence": 0.5, "act_confidence": 0.9, "default_max_level": 1,
          "action_max_levels": {"speak": 1, "archive_file": 3, "__notify__": 1},
          "quiet_hours": [22, 8], "max_interruptions_per_hour": 8}
triggers = [{"type": "filewatch", "name": "inbox-watch", "params": {"path": "./watched"}}]
rules = [
    {"when": {"kind": "proactive_utterance"},
     "intent": {"action": "speak", "confidence": 0.6, "urgency": 0.4, "level": 1,
                "rationale": "模型主動發話，建議轉達使用者"}},
    {"when": {"kind": "file.created"},
     "intent": {"action": "archive_file", "confidence": 0.95, "urgency": 0.3, "level": 3,
                "rationale": "偵測到新檔案，自動歸檔"}},
]
actions = {"speak": {"type": "console"},
           "archive_file": {"type": "callback", "handler": "ops.archive"},
           "__notify__": {"type": "console"}}
brain = {"type": "llm", "engine": "minicpm-o", "omni_engine": "llama-server",
         "model": "weights.gguf", "n_ctx": 4096, "learning": True,
         "user_profile": "偏好簡短、最少打擾；高風險動作先確認"}


def main():
    missing = [src for _seg, src in WEIGHT_FILES
               if not os.path.exists(os.path.join(MODELS, src))]
    if missing:
        raise SystemExit(f"缺少權重檔（請先下載）：{missing}")

    w = PaiWriter()
    w.add_json("manifest.json", manifest)
    w.add_json("policy.json", policy)
    w.add_json("triggers.json", triggers)
    w.add_json("rules.json", rules)
    w.add_json("actions.json", actions)
    w.add_json("brain.json", brain)
    for seg, src in WEIGHT_FILES:
        w.add_file(seg, os.path.join(MODELS, src), compress=False)  # 權重不壓縮，可 mmap
    w.write(OUT)
    print(f"✅ 已打包：{OUT}（{os.path.getsize(OUT):,} bytes，{len(WEIGHT_FILES)} 個權重段）")


if __name__ == "__main__":
    main()
