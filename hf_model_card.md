---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
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
| `brain.json` | 決策腦設定（llama-server 引擎） |
| `weights.gguf` | **google/gemma-4-26B-A4B-it QAT q4_0**（14.4 GB，SHA-256 校驗） |

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

## 規格

- 格式：`.pai` Binary Format v1（[規格](https://github.com/vito1317/pai-framework/blob/main/FORMAT.md)）
- 基底模型：google/gemma-4-26B-A4B-it（QAT q4_0，apache-2.0）
- 權重 SHA-256：`4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`
- 建議硬體：≥24GB 統一記憶體（Apple Silicon）或同級 GPU
- 實測：Apple M4 32GB — 載入 ~8s（快取後）、單次決策 ~35s

## License

框架：MIT。內嵌權重：apache-2.0（Google Gemma 4）。

Author: vito1317 <service@vito1317.com>
