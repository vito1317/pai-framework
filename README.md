# PAI — Proactive AI Framework（主動式 AI 框架）

一個通用、零依賴（純 Python 標準庫）、可直接運行的主動式 AI 框架原型，並內建 **PAI Protocol v1.1** 標準資料格式。

核心理念：從「被動式 AI（你下指令它才做事）」轉為「事件驅動 + 持續感知 + 受治理的主動行為」。

## 架構

```
┌─ 感知層 Triggers ─────────────────────────────┐
│ IntervalTrigger / ScheduleTrigger             │
│ FileWatchTrigger / ThresholdTrigger           │  → Event（標準事件格式）
└───────────────────────────────────────────────┘
                    ↓
┌─ 決策層 Brain ────────────────────────────────┐
│ RuleBrain（規則，離線、零成本、前線哨兵）        │
│ LLMBrain（LLM 判斷，僅在需要時喚醒，失敗退回規則）│  → Intent（action/confidence/urgency/rationale）
└───────────────────────────────────────────────┘
                    ↓
┌─ 治理層 ProactivityPolicy ────────────────────┐
│ 1. 信心門檻  2. 動作風險上限  3. 回饋自動降級    │
│ 4. 安靜時段  5. 干擾度公式  6. 每小時打擾上限    │  → 授予的自主等級
└───────────────────────────────────────────────┘
                    ↓
┌─ 行動層 Actions ──────────────────────────────┐
│ OBSERVE 只記錄 │ SUGGEST 通知 │ ASK 確認 │ ACT 自動執行 │
└───────────────────────────────────────────────┘
                    ↓
┌─ 記憶層 Memory（SQLite）──────────────────────┐
│ 事件 / 意圖 / 結果 / 回饋 / PAI Protocol 紀錄   │ → 回饋迴圈調節未來主動性
└───────────────────────────────────────────────┘
```

## PAI 干擾度公式

```
urgency × confidence > interruption_cost  才允許打擾使用者
```

`interruption_cost_fn` 可依使用者當下狀態（專注模式、開會、深夜）動態回傳 0~1。

## 自主等級 × 交付模式

| AutonomyLevel | 行為 | 交付模式 |
|---|---|---|
| OBSERVE (0) | 只記錄，不打擾 | level_0_silent |
| SUGGEST (1) | 主動通知/建議 | level_1_soft_nudge |
| ASK (2) | 草稿+請求人類授權 | level_2_approval |
| ACT (3) | 高信心低風險時自動執行 | level_0_silent（事後可查） |

風險高的動作用 `action_max_levels` 鎖死上限（例：寄信最多 ASK，永不自動寄出）。

## PAI Protocol v1.1（標準資料格式）

每次主動行為產生一份 6 層 JSON 紀錄（存入 SQLite，可直接對外交換）：

```json
{
  "pai_protocol_version": "1.1",
  "record_id": "pai_20260611_a1b2c3",
  "timestamp": "2026-06-11T15:39:50Z",
  "1_perception":   { "trigger_source": "...", "event_type": "...", "raw_data_summary": {} },
  "2_context":      { "user_current_state": "...", "relevant_memory": [], "action_history": [] },
  "3_anticipation": { "predicted_intent": "...", "urgency_score": 0.95, "confidence_score": 0.88,
                      "interruption_cost": 0.9, "requested_level": 2, "granted_level": 2 },
  "4_execution":    { "actions_taken": [{ "tool": "...", "status": "ok" }], "status": "executed" },
  "5_delivery":     { "delivery_mode": "level_2_approval", "requires_human_approval": true },
  "6_adaptation":   { "user_feedback": "pending", "learning_adjustment": null }
}
```

## `.pai` 打包格式（類 GGUF 的二進位容器）

整個 agent 可打包成單一 `.pai` 檔案發布（規格見 [FORMAT.md](FORMAT.md)）：
magic `PAI\x01` + 段表（64-byte 對齊、每段 SHA-256 校驗、zlib 壓縮）+ 資料區，
可內嵌 `manifest / policy / triggers / rules / actions / brain / memory.db / weights.gguf`。

```python
from pai import pack_agent, load_agent
pack_agent("my-agent.pai", manifest={...}, policy={...},
           triggers=[...], rules=[...], actions={...},
           weights_path="model-q4.gguf")   # 選用：內嵌本地模型
agent_def = load_agent("my-agent.pai")
```

執行期的單筆行為紀錄則是文字 JSON，附檔名 `.pai.json`。

## 快速開始

```bash
python3 demo.py        # 主動行為迴圈示範（零依賴）
python3 pack_demo.py   # .pai 打包/載入/防竄改示範
```

demo 模擬三種主動行為：
1. CPU 飆高（ThresholdTrigger）→ 高緊急 → **ASK**（確認後清理）
2. 待辦堆積（IntervalTrigger）→ 中緊急 → **SUGGEST**（只通知）
3. 新檔案出現（FileWatchTrigger）→ 低風險 → **ACT**（自動歸檔）

## 一鍵執行 .pai

```bash
python3 -m pai info ops-guardian.pai    # 查看打包內容
python3 -m pai run  ops-guardian.pai    # 載入並運行（未注入的 handler 以 stub 示意）
```

程式內載入（注入真實 handler / metric）：

```python
from pai import load_runtime
agent = load_runtime("ops-guardian.pai",
    handlers={"ops.cleanup": my_cleanup_fn},
    metrics={"cpu": read_cpu_percent})
agent.run()
```

## 底層模型（三層選擇，模型無關）

| 決策腦 | 底層模型 | 用途 |
|---|---|---|
| `RuleBrain` | 無（純規則） | 前線哨兵、離線運行、LLM 的安全 fallback |
| `LLMBrain` | 雲端 API（預設 `claude-sonnet-4-6`，可換任何模型） | 開放式判斷 |
| `LocalLLMBrain` | `.pai` 內嵌的 GGUF 權重，經 llama.cpp 推理（`pip install llama-cpp-python`），建議 4B–8B instruct 量化模型 | 完全離線、資料不出機 |

`brain.json` 決定用哪個；本地權重存在時優先使用，載入失敗自動退回規則腦。

## 接上 LLM

```python
from pai import LLMBrain, RuleBrain

brain = LLMBrain(
    available_actions=["cleanup", "archive_file", "__notify__"],
    model="claude-sonnet-4-6",          # 需設 ANTHROPIC_API_KEY
    fallback=RuleBrain([...]),          # LLM 失敗時退回規則
    user_profile="工程師，上班時間 9-18，討厭被打擾",
)
```

成本控制採「雙層觸發」：RuleBrain 當前線哨兵過濾大多數事件，只有規則判定值得深入時才喚醒 LLM。

## 落地建議

1. **預設 Human-in-the-loop**：初期把所有有副作用的動作上限設為 ASK，跑出信任後再逐步開放 ACT。
2. **可撤銷性**：所有自動化動作應提供 Undo（demo 的歸檔動作即可逆）。
3. **回饋驅動**：使用者連續拒絕同類建議時，policy 會自動降級該動作（內建）。

---
Author: vito1317 <service@vito1317.com>
