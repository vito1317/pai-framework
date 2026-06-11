---
license: apache-2.0
base_model: openbmb/MiniCPM-o-4_5
tags:
  - proactive-ai
  - pai
  - agent
  - omni
  - full-duplex
  - speech
---

# minicpm-o-guardian.pai — Full-Duplex Omni Proactive AI Agent

以 **openbmb/MiniCPM-o-4_5**（9B omni：影像+音訊輸入、文字+語音輸出、全雙工）為決策腦的
主動式 agent，打包成單一 `.pai` 檔。

> 本檔 **不內嵌權重**（約 2 KB，只含 agent 設定，引用基底模型）。`transformers` 路徑執行時
> 會從 Hugging Face 自動載入 `openbmb/MiniCPM-o-4_5`。若要離線單檔，可改打包內嵌 GGUF
> （`openbmb/MiniCPM-o-4_5-gguf`），見 `pack_minicpm_o.py` 的 `WEIGHTS`。
>
> （刻意不設 `base_model_relation`：這不是 adapter/finetune/quantized/merge 任一種，
> 而是「引用基底模型的 agent 設定」，避免被歸入 MiniCPM-o 的 adapters 列表造成誤導。）

## 為什麼用 MiniCPM-o

MiniCPM-o 4.5 能同時看、聽、說，且**原生支援主動互動**（依現場持續理解主動發起提醒/評論）
——天然契合 PAI 的「事件驅動 + 持續感知 + 受治理主動行為」範式。

## 內容（`.pai` 段）

| 段 | 內容 |
|---|---|
| `manifest.json` | omni agent 中繼資料 |
| `policy.json` | 主動性治理（安靜時段、自主等級上限、干擾度公式） |
| `triggers.json` / `rules.json` / `actions.json` | 宣告式感知/規則/動作 |
| `brain.json` | `engine: minicpm-o`、`omni_engine: transformers` |

## 使用

```bash
git clone https://github.com/vito1317/pai-framework && cd pai-framework
pip install "transformers==4.51.0" accelerate "torch>=2.3.0" torchaudio "minicpmo-utils[all]>=1.0.5"

# 下載本 .pai 後
python3 -m pai run minicpm-o-guardian.pai
```

`DuplexOmniLoop` 把模型的全雙工串流包成「持續感知 trigger」：模型自己決定何時主動發話，
PAI 治理層仍決定是否放行、用什麼等級打擾。微調走 `LlamaFactoryBackend`（LoRA），
self-finetuning 三層（記憶 / LoRA 熱插拔 / EvalGate）完全沿用。

## 硬體

- transformers 路徑：建議 CUDA GPU（≥16GB）或 Apple Silicon。device 自動偵測（cuda→mps→cpu），無需手動指定
- 全雙工即時串流建議搭配官方 `llama.cpp-omni` + WebRTC demo

## 相關連結

- 框架原始碼：[github.com/vito1317/pai-framework](https://github.com/vito1317/pai-framework)
- 內嵌權重的離線版本：[vito95311/gemma-guardian-pai](https://huggingface.co/vito95311/gemma-guardian-pai)（Gemma 4 26B-A4B）

## License

框架：MIT。模型：apache-2.0（OpenBMB MiniCPM-o 4.5）。

Author: vito1317 <service@vito1317.com>
