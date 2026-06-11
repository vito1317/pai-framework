# `.pai` Binary Format Specification v1

**PAI（Proactive AI）打包容器格式** — 類比 `.gguf` 之於模型權重，`.pai` 把一個完整的主動式 AI agent 打包成單一可發布、可校驗的二進位檔案。

## 設計目標

1. **單檔發布**：agent 的 manifest、治理政策、觸發器、規則、動作、記憶快照、（選用）嵌入模型權重，全部在一個檔案內。
2. **mmap 友善**：所有資料段 64-byte 對齊，大型權重段不壓縮，可直接記憶體映射讀取（與 GGUF 同理）。
3. **完整性校驗**：每段附 SHA-256，防損毀與竄改。
4. **向前相容**：版本欄位 + 具名段表，新版本可加段而不破壞舊讀取器。

## 二進位佈局（全部 little-endian）

```
┌───────────────────────────────────────────────┐
│ Header (32 bytes)                             │
│   magic       4B   = "PAI\x01"                │
│   version     u32  = 1                        │
│   n_sections  u32                             │
│   flags       u32  (bit0: 全檔預設 zlib 壓縮)  │
│   created_at  u64  (unix epoch 秒)            │
│   reserved    8B                              │
├───────────────────────────────────────────────┤
│ Section Table (n_sections 筆)                 │
│   name_len    u16                             │
│   name        UTF-8 bytes                     │
│   stype       u32  (0=json 1=bin 2=sqlite     │
│                     3=text 4=gguf)            │
│   compressed  u8   (0/1，每段獨立)             │
│   offset      u64  (檔案絕對位移，64B 對齊)    │
│   size        u64  (儲存大小=壓縮後)           │
│   raw_size    u64  (解壓後大小)                │
│   sha256      32B  (解壓後內容雜湊)            │
├───────────────────────────────────────────────┤
│ Data Area                                     │
│   各段資料，起點 64-byte 對齊，\x00 padding    │
└───────────────────────────────────────────────┘
```

## 標準段（section）

| 段名 | 必要 | 內容 |
|---|---|---|
| `manifest.json` | ✅ | agent 名稱、版本、作者、說明、相容框架版本 |
| `policy.json` | ✅ | ProactivityPolicy 參數（信心門檻、等級上限、安靜時段、干擾公式參數） |
| `triggers.json` | ✅ | 觸發器宣告：`[{"type": "threshold", "params": {...}}]` |
| `rules.json` | ✅ | 宣告式規則：事件條件 → 意圖（action/confidence/urgency/level） |
| `actions.json` | ✅ | 動作宣告：`{"notify": {"type": "webhook", "url": ...}}` |
| `brain.json` | 選用 | 決策腦設定（rule/llm、模型名、system prompt、fallback） |
| `memory.db` | 選用 | SQLite 記憶快照（出廠記憶/個人化狀態隨包攜帶） |
| `weights.gguf` | 選用 | 嵌入的本地模型權重（不壓縮，可 mmap） |

## 與相關格式的關係

| 格式 | 角色 |
|---|---|
| `.gguf` | 打包「模型權重」 |
| `.pai` | 打包「整個主動式 agent」（行為+治理+記憶，可內嵌 .gguf） |
| `.pai.json` | 單筆 PAI Protocol 行為紀錄（執行期事件，文字 JSON） |

## API

```python
from pai import pack_agent, load_agent, PaiReader

# 打包
pack_agent("my-agent.pai",
    manifest={"name": "ops-guardian", "version": "1.0.0", "author": "vito1317"},
    policy={...}, triggers=[...], rules=[...], actions={...},
    memory_db="pai_memory.db",          # 選用
    weights_path="qwen3-4b-q4.gguf",    # 選用
)

# 載入
agent_def = load_agent("my-agent.pai")
r = PaiReader("my-agent.pai")
print(r.info())                          # 段表、大小
data = r.read_bytes("weights.gguf")      # 自動 SHA-256 校驗
```

---
Author: vito1317 <service@vito1317.com>
