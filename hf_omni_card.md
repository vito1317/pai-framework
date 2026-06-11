---
license: apache-2.0
base_model: openbmb/MiniCPM-o-4_5-gguf
tags:
  - proactive-ai
  - pai
  - agent
  - omni
  - full-duplex
  - speech
  - gguf
---

# minicpm-o-omni-guardian.pai — 內嵌全套 omni 權重的離線全雙工 agent

單一 `.pai`（8.87 GB）= 一個完整、可離線運行的全雙工 omni 主動式 agent，
內嵌 **MiniCPM-o 4.5** 的全套 GGUF 權重。

## 內嵌權重段（10 個，皆 SHA-256 校驗、64-byte 對齊、可 mmap）

| .pai 段 | 來源 | 用途 |
|---|---|---|
| `weights.gguf` | MiniCPM-o-4_5-Q4_K_M (4.7G) | 主 LLM 決策腦 |
| `weights/vision.gguf` | vision-F16 (1.0G) | 視覺編碼 |
| `weights/audio.gguf` | audio-F16 (630M) | 音訊編碼 |
| `weights/tts.gguf` | tts-F16 (1.1G) | 語音合成主體 |
| `weights/tts-projector.gguf` | projector-F16 (14M) | TTS 投影 |
| `weights/token2wav/*` | encoder/flow/hifigan/prompt_cache (882M) | token→波形 |

## 使用

```bash
pip install paigent
brew install llama.cpp      # 全雙工建議搭配官方 llama.cpp-omni + WebRTC demo

python3 -m pai info minicpm-o-omni-guardian.pai
python3 -m pai run  minicpm-o-omni-guardian.pai
```

`brain.json` 設 `engine=minicpm-o, omni_engine=llama-server`：loader 會把 `weights.gguf`
抽到快取交給 llama-server 做決策；多模態元件供 omni 串流使用。

## Self-Finetuning（與框架共用）

記憶式即時學習（第1層）開箱即用；LoRA 熱插拔（第2層，`/lora-adapters`）+ EvalGate 閘門
（第3層）皆沿用框架的 `attach_self_finetuning`，主幹權重永不變更、可一鍵回滾。

## 相關連結

- 框架原始碼 / PyPI：[github.com/vito1317/pai-framework](https://github.com/vito1317/pai-framework) · `pip install paigent`
- 不內嵌權重的輕量版：[vito95311/minicpm-o-guardian-pai](https://huggingface.co/vito95311/minicpm-o-guardian-pai)
- Gemma 4 離線版：[vito95311/gemma-guardian-pai](https://huggingface.co/vito95311/gemma-guardian-pai)

## License

框架：MIT。權重：apache-2.0（OpenBMB MiniCPM-o 4.5）。

Author: vito1317 <service@vito1317.com>
