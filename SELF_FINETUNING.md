# PAI Self-Finetuning 架構（即時自我學習）

讓 PAI agent 從使用者回饋中持續自我校準。採三層遞進設計，風險由低到高、即時性由高到低，全部以 `.pai` 為載體。

## 為什麼不直接「線上改權重」

直接對 26B 主幹做線上梯度更新在主動式 agent 場景是反模式：
1. **災難性遺忘**——單樣本 SGD 會破壞既有能力，且不可逆。
2. **污染風險**——壞樣本/對抗回饋直接寫進權重，無法回滾。
3. **成本**——全量微調需數百 GB 顯存；即使 QLoRA，單步也要數秒並佔用推理資源。

因此正確做法是把「學習」與「主幹權重」解耦：能即時學的用記憶，要改行為的用可插拔 adapter，且任何權重變更都必須通過離線品質閘門。

---

## 第一層：記憶/檢索式學習（已實作 ✅）

**即時性**：毫秒級　**風險**：零　**動權重**：否

每次 (事件 → 意圖 → 回饋) 存成經驗向量；下次相似事件檢索 top-k 經驗，把「過往教訓」注入決策腦 prompt。

- 模組：`pai/learning.py` 的 `ReflectiveMemory`
- embedding：`EmbeddingClient`（llama-server `/v1/embeddings` 語義檢索）或 `HashingEmbedder`（零依賴詞袋）
- 啟用：`brain.json` 設 `"learning": true`
- 回饋 API：`agent.record_user_feedback(event, intent, "accepted|rejected|modified|ignored")`
- 已驗證：使用者連續拒絕某類通知後，PAI 自動把該類行為從 SUGGEST 降為靜默，全程未動權重

這層解決了 80% 的「希望它記住我的偏好」需求，且立即生效、可解釋、可刪除單筆。

---

## 第二層：LoRA adapter 熱插拔（設計）

**即時性**：分鐘～小時級（背景）　**風險**：低（可回滾）　**動權重**：只動 adapter，主幹不變

當第一層累積足夠回饋（例如 ≥200 筆、且涵蓋多種情境），在背景把回饋轉成偏好資料集，跑一次 LoRA/QLoRA 微調，產生一個幾十 MB 的 adapter。

```
回饋資料庫 ──> build dataset (accepted=正例, rejected=負例)
           ──> QLoRA 訓練 (背景, 不阻塞推理)
           ──> adapter.safetensors
           ──> 存進 .pai 新段 adapters/<timestamp>.safetensors
           ──> llama-server --lora 熱載入（主幹權重不動）
```

`.pai` 格式擴充（新慣例段，向後相容）：

| 段 | 內容 |
|---|---|
| `adapters/active.safetensors` | 目前生效的 LoRA |
| `adapters/<ts>.safetensors` | 歷史 adapter（可一鍵回滾） |
| `training/feedback.jsonl` | 累積的偏好樣本 |
| `training/state.json` | 上次訓練時間、樣本數、版本 |

關鍵安全設計：主幹權重段 `weights.gguf` **永遠不變**，學習只新增/切換 adapter 段。要回滾就把 `active` 指回舊 adapter。

訓練引擎建議：`llama.cpp` 的 `llama-finetune`（純 C++、無重依賴、能直接吃 GGUF）或 PEFT+bitsandbytes（功能強、依賴重）。`brain.json` 以 `"finetune": {"engine": "...", "min_samples": 200, "schedule": "daily"}` 設定。

---

## 第三層：排程重訓 + Eval Gate（設計）

**即時性**：天/週級　**風險**：中（有閘門擋住）　**動權重**：adapter（不上線就丟棄）

新 adapter 不是訓完就用，**必須先通過離線品質閘門**才會切成 active：

```
新 adapter ──> 在固定 eval set 上跑（保留的歷史情境 + 黃金答案）
          ──> 比較 新 vs 現役 adapter 的指標（決策正確率、打擾精準度、拒絕率）
          ──> 新的「顯著勝出」才 promote 成 active；否則保留現役、丟棄新 adapter
          ──> 每次 promote 寫入稽核紀錄（誰、何時、用哪批資料、指標多少）
```

Eval set 從 PAI Protocol 紀錄自動累積（使用者明確 accepted/rejected 的情境就是天然標註）。閘門條件可設保守，例如「新 adapter 決策正確率 ≥ 現役 + 2%，且打擾精準度不下降」。

這層防止「越學越壞」——這正是線上學習最大的坑。

---

## 路線圖

- [x] 第一層 ReflectiveMemory（已實作並驗證）
- [ ] 第二層 `.pai` adapter 段 + `PaiWriter.add_adapter()` / loader `--lora` 熱載入
- [ ] 第二層 背景訓練 worker（llama-finetune 包裝）
- [ ] 第三層 EvalGate（eval set 累積 + 指標比較 + promote/rollback + 稽核）

---
Author: vito1317 <service@vito1317.com>
