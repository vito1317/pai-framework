---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
tags:
  - proactive-ai
  - pai
  - agent
  - gguf
  - llama.cpp
---

# gemma-guardian.pai — Proactive AI Agent（內嵌 Gemma 4 26B-A4B）

單一 `.pai` 檔案 = 一個完整、可離線運行的主動式 AI agent。

`.pai` 是一個類 GGUF 的二進位容器格式，本檔打包了：

| 段 | 內容 |
|---|---|
| `manifest.json` | agent 中繼資料 |
| `policy.json` | 主動性治理（信心門檻、自主等級上限、干擾度公式） |
| `triggers.json` / `rules.json` / `actions.json` | 宣告式感知/規則/動作 |
| `brain.json` | 決策腦設定（llama-server 引擎，`n_ctx=4096`） |
| `weights.gguf` | **google/gemma-4-26B-A4B-it QAT q4_0**（14.4 GB，SHA-256 校驗） |

> **此包的 guardian 行為**：`cpu-monitor`（threshold）→ `cleanup`（清理高負載，最多 ASK）；
> `inbox-watch`（filewatch）→ `archive_file`（自動歸檔，可 ACT）。每個 action 都有對應 trigger，
> llama-server 不可用時退回 RuleBrain 仍可獨立運作。

## 使用方式

```bash
# 1. 取得框架
git clone https://github.com/vito1317/pai-framework
cd pai-framework

# 2. 安裝推理引擎
brew install llama.cpp   # 或任何提供 llama-server 的安裝方式

# 3. 下載本檔後直接運行
python3 -m pai info gemma-guardian.pai
python3 -m pai run  gemma-guardian.pai
```

框架會從 `.pai` 抽出權重、自動啟動本地 `llama-server`（Metal/CUDA），
由 Gemma 4 對事件做主動決策（urgency/confidence/干擾成本），經治理層核准後行動。
全程離線，資料不出機。

## 即時 Self-Finetuning（三層，主幹權重永不變更）

這個 agent 會從使用者回饋中持續自我校準，採三層遞進設計（詳見
[SELF_FINETUNING.md](https://github.com/vito1317/pai-framework/blob/main/SELF_FINETUNING.md)）：

| 層 | 機制 | 即時性 | 動權重 |
|---|---|---|---|
| 1 | **ReflectiveMemory** — 回饋存成經驗向量，相似情境檢索教訓注入決策 | 毫秒級 | 否 |
| 2 | **LoRA adapter 熱插拔** — 回饋→偏好資料集→`llama-finetune`→adapter→`--lora` 載入 | 分鐘/小時 | 只動 adapter |
| 3 | **EvalGate** — 候選 adapter 須在離線指標顯著勝出且無退化才上線；可一鍵回滾 | 天/週 | adapter（不上線就丟） |

主幹 `weights.gguf` 永遠不變，學習只新增/切換可回滾的小 adapter，並且任何權重變更
都必須先通過品質閘門——這避免了線上學習常見的「越學越壞」與災難性遺忘。

## 規格

- 格式：`.pai` Binary Format v1（[規格](https://github.com/vito1317/pai-framework/blob/main/FORMAT.md)）
- 基底模型：google/gemma-4-26B-A4B-it（QAT q4_0，apache-2.0）
- 權重 SHA-256：`4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- 建議硬體：≥24GB 統一記憶體（Apple Silicon）或同級 GPU
- 實測：Apple M4 32GB — 載入 ~8s（快取後）、單次 LLM 決策 ~35s
- `n_ctx=4096` 為刻意設定（決策 payload 很短，省記憶體；模型原生支援 256K）

> **不是每個事件都要等 35 秒**：採 rule-first 雙層觸發——`RuleBrain`（前線哨兵）即時、零成本
> 過濾大多數事件，只有規則判定值得深入時才喚醒 Gemma 4 做開放式判斷，那一次才約 35s。

## 相關連結

- 框架原始碼：[github.com/vito1317/pai-framework](https://github.com/vito1317/pai-framework) · `pip install paigent`
- 全雙工 omni（不含權重，輕量）：[vito95311/minicpm-o-guardian-pai](https://huggingface.co/vito95311/minicpm-o-guardian-pai)
- 全雙工 omni（內嵌全套權重，離線）：[vito95311/minicpm-o-omni-guardian-pai](https://huggingface.co/vito95311/minicpm-o-omni-guardian-pai)

## License

框架：MIT。內嵌權重：apache-2.0（Google Gemma 4）。

Author: vito1317 <service@vito1317.com>
