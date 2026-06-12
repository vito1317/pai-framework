# PAI — Proactive AI Framework（主動式 AI 框架）

一個通用、零依賴（純 Python 標準庫）、可直接運行的主動式 AI 框架原型，並內建 **PAID Protocol v1.2**（Proactive Agent Infrastructure with Dynamic-finetuning） 標準資料格式。

核心理念：從「被動式 AI（你下指令它才做事）」轉為「事件驅動 + 持續感知 + 受治理的主動行為」。

## 安裝

[![PyPI](https://img.shields.io/pypi/v/paigent)](https://pypi.org/project/paigent/)

```bash
pip install paigent          # PyPI 發佈名為 paigent；import 仍是 `import pai`，CLI 仍是 `pai`
```

## 預打包的 .pai agent（Hugging Face）

| 模型頁 | 決策腦 | 說明 |
|---|---|---|
| [vito95311/gemma-guardian-pai](https://huggingface.co/vito95311/gemma-guardian-pai) | Gemma 4 26B-A4B（QAT q4_0，內嵌 14.4GB GGUF） | 離線 guardian：CPU 監控→清理、新檔案→自動歸檔 |
| [vito95311/minicpm-o-guardian-pai](https://huggingface.co/vito95311/minicpm-o-guardian-pai) | MiniCPM-o 4.5（全雙工 omni，不內嵌權重） | 即時看＋聽、原生主動互動、語音回應 |

## 架構

詳細架構圖：[docs/architecture.svg](docs/architecture.svg)（手繪精緻版）、
[docs/architecture.mermaid](docs/architecture.mermaid)（Mermaid 原始碼，可於 mermaid.live 預覽）。

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
│ 事件 / 意圖 / 結果 / 回饋 / PAID Protocol 紀錄   │ → 回饋迴圈調節未來主動性
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

## PAID Protocol v1.2（標準資料格式）

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

## 底層模型（模型無關，可插拔決策腦）

| 決策腦 | 底層模型 | 用途 |
|---|---|---|
| `RuleBrain` | 無（純規則） | 前線哨兵、離線運行、LLM 的安全 fallback |
| `LLMBrain` | 雲端 API（預設 `claude-sonnet-4-6`，可換任何模型） | 開放式判斷 |
| `LocalLLMBrain` | `.pai` 內嵌 GGUF，經 `llama-cpp-python` | 完全離線、資料不出機 |
| `LlamaServerBrain` | `.pai` 內嵌 GGUF，經 llama.cpp `llama-server`（支援 `--lora` 熱載入） | 離線、新架構支援最快、常駐免重載 |
| `MiniCPMoBrain` | **openbmb/MiniCPM-o-4_5**（全雙工 omni：影像+音訊輸入、文字+語音輸出） | 即時看＋聽現場、原生主動互動、語音回應 |

`brain.json` 的 `engine` 決定用哪個（`llama-server` / `llama-cpp-python` / `minicpm-o`）；
載入失敗一律自動退回 `RuleBrain`。

### 全雙工語音 / Omni 版本（MiniCPM-o 4.5）

MiniCPM-o 4.5（9B，SigLip2 + Whisper + CosyVoice2 + Qwen3-8B）能同時看、聽、說，
且**原生支援主動互動**——天然契合 PAI。打包：

```bash
python3 pack_minicpm_o.py          # 預設不內嵌權重，transformers 從 HF 載入
```

`DuplexOmniLoop` 把模型的全雙工串流（`streaming_prefill`/`streaming_generate`）包成
「持續感知 trigger」：模型自己決定何時主動發話，PAI 治理層仍負責是否放行、用什麼等級打擾。
微調走 `LlamaFactoryBackend`（包 LLaMA-Factory 做 LoRA），其餘 self-finetuning 編排（EvalGate /
promote / rollback / 稽核）完全沿用。詳見 `pack_minicpm_o.py` 與 `pai/omni_brain.py`。

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

## 接平台（哨兵 → webhook）

把 PAI 當「主動哨兵」掛到既有後端：PAI 負責持續感知＋治理判斷，真正的副作用交給你的平台 API。
用 `WebhookNotifier` 或 `CallbackAction` 把通過治理層的意圖 POST 到 `/webhooks/{node}`：

```python
from pai import load_runtime
from pai.actions import WebhookNotifier, CallbackAction
import urllib.request, json

def post_webhook(node):
    def _fn(intent):
        body = json.dumps({
            "node": node, "action": intent.action, "params": intent.params,
            "confidence": intent.confidence, "urgency": intent.urgency,
            "rationale": intent.rationale,
        }).encode()
        req = urllib.request.Request(
            f"https://your-platform.example/webhooks/{node}",
            data=body, headers={"content-type": "application/json",
                                 "authorization": "Bearer <token>"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return {"status": r.status}
    return CallbackAction(_fn)

agent = load_runtime("gemma-guardian.pai", handlers={
    "cleanup":      post_webhook("ops.cleanup"),
    "archive_file": post_webhook("files.archive"),
}, metrics={"cpu": read_cpu})
# 或直接用內建：actions={"__notify__": WebhookNotifier("https://.../webhooks/notify")}
agent.run()
```

治理層先過濾（信心/風險/安靜時段/干擾度/頻率），只有真正該動作的意圖才會打到你的平台——
等於替既有系統加一層「會自己判斷何時該出手」的主動神經。

## 落地建議

1. **預設 Human-in-the-loop**：初期把所有有副作用的動作上限設為 ASK，跑出信任後再逐步開放 ACT。
2. **可撤銷性**：所有自動化動作應提供 Undo（demo 的歸檔動作即可逆）。
3. **回饋驅動**：使用者連續拒絕同類建議時，policy 會自動降級該動作（內建）。

---
Author: vito1317 <service@vito1317.com>
