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

## 實作狀態

全部三層已實作並通過端到端測試（`finetune_demo.py`，EchoBackend 無 GPU 驗證）：

- [x] 第一層 ReflectiveMemory（記憶式即時學習）
- [x] 第二層 訓練後端（`EchoBackend` 測試用 / `LlamaFinetuneBackend` 真實，包 `llama-finetune`）
- [x] 第二層 `export_preference_dataset`（回饋→偏好資料集）
- [x] 第二層 `AdapterStore`（側車 adapter 管理、active 指標、歷史、回滾）
- [x] 第二層 `LlamaServerBrain` 以 `--lora` 熱載入現役 adapter（主幹不動）
- [x] 第二層 `bake_adapter_into_pai`（發布時把現役 adapter 烘焙進 `.pai`）
- [x] 第三層 `EvalGate`（主指標增益門檻 + 非退化指標檢查）
- [x] 第三層 `SelfFinetuneManager`（訓練→閘門→promote/丟棄→稽核 jsonl）

實測驗證：第一次訓練 +0.07 增益通過上線；無增益再訓練被閘門擋下（保護「越學越壞」）；
新增回饋後 v2 +0.05 通過；可一鍵回滾到 v1；全程主幹權重 `weights.gguf` 未變更。

### 完整 1+2+3 一行掛上（模型無關，Gemma 與 MiniCPM-o 皆適用）

```python
from pai import load_runtime, attach_self_finetuning

agent = load_runtime("gemma-guardian.pai", handlers=..., metrics=...)
mgr = attach_self_finetuning(agent, eval_fn=my_eval_fn, min_samples=200)
mgr.run_periodic(interval_sec=86400)     # 每日：匯出回饋→訓練→閘門→上線（熱切換）
agent.run()
```

`attach_self_finetuning` 會依決策腦自動選後端與熱切換機制：

| 決策腦 | 訓練後端 | 上線熱切換 |
|---|---|---|
| `LlamaServerBrain`（Gemma / MiniCPM-o GGUF） | `LlamaFinetuneBackend`（llama-finetune，GGUF LoRA） | llama-server `/lora-adapters`，主幹不重載 |
| `MiniCPMoBrain`（transformers） | `LlamaFactoryBackend`（PEFT LoRA） | PEFT `set_adapter`，主幹不重載 |

四個環節對照你列的需求：
1. **資料來源**：`ActionFeedback`（核准/駁回）→ `experiences` 表 → `export_preference_dataset()`；
   `run_periodic()` 定期觸發。
2. **訓練**：`LlamaFinetuneBackend`（QAT q4_0 上跑 LoRA）/ `LlamaFactoryBackend`。主幹不變。
3. **上線**：`/lora-adapters` 或 PEFT `set_adapter` 執行期熱切換，**不重載 14.4GB 主幹**；
   `AdapterStore.rollback()` 一鍵摘掉回滾。
4. **閘門**：`EvalGate` 離線評測，沒退化才掛上。

> 注意：`/lora-adapters` 熱切換要求 adapter 在啟動時已用 `--lora` 預載（loader 會把
> AdapterStore 內所有 adapter 預載）；訓練出的**全新** adapter 若不在預載清單，需重啟一次
> 才能進入可熱切換集合——這是 llama.cpp 現況限制。

### 用真實 LoRA 訓練

```python
from pai import SelfFinetuneManager, EvalGate, LlamaFinetuneBackend
mgr = SelfFinetuneManager(
    db_path="pai_memory.db", adapters_root="pai_adapters",
    trainer=LlamaFinetuneBackend("llama-finetune"),   # 需 llama.cpp 編出 llama-finetune
    eval_gate=EvalGate(my_eval_fn, min_gain=0.02),
    base_gguf="models/gemma-4-26B_q4_0-it.gguf", min_samples=200)
mgr.maybe_train_and_promote()    # 可放進排程（每日/每週背景跑）
```

loader 會自動偵測 `pai_adapters/` 的現役 adapter 並在啟動 llama-server 時 `--lora` 掛上。

---
Author: vito1317 <service@vito1317.com>
