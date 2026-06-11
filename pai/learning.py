"""self-finetuning 第一層：ReflectiveMemory（記憶/檢索式即時學習）。

最即時、零風險、不動權重的「自我學習」：
- 每次主動行為的結果與使用者回饋存成「經驗」(experience)
- 下次遇到相似事件時，檢索最相關的成功/失敗經驗，注入決策腦的 prompt
- 使用者持續回饋 → 行為持續校準，無需重訓、可立即生效、可解釋、可刪除

embedding 來源可插拔：
- EmbeddingClient(server)：用 llama-server / OpenAI 相容 /v1/embeddings（語義檢索）
- HashingEmbedder：零依賴詞袋雜湊向量（離線 fallback，仍有不錯的詞彙重疊召回）
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


# ---------- embedding backends ----------

class HashingEmbedder:
    """零依賴詞袋雜湊向量（hashing trick）。免模型、離線可用。"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        # 同時切英數詞與 CJK 單字（中文無空格）
        return re.findall(r"[a-zA-Z0-9_]+|[一-鿿]", text.lower())

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = hash(tok) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class EmbeddingClient:
    """呼叫 OpenAI 相容 /v1/embeddings（如 llama-server --embeddings）。"""

    def __init__(self, base_url: str, model: str = "default", fallback: Optional[HashingEmbedder] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback = fallback or HashingEmbedder()

    def embed(self, text: str) -> list[float]:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/embeddings",
                data=json.dumps({"input": text, "model": self.model}).encode(),
                headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            v = data["data"][0]["embedding"]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            return [x / norm for x in v]
        except Exception:  # noqa: BLE001
            return self.fallback.embed(text)


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))   # 兩邊都已正規化


# ---------- experience store ----------

@dataclass
class Experience:
    event_summary: str
    action: str
    rationale: str
    feedback: str          # accepted | rejected | modified | ignored
    score: float           # +1 正向 / -1 負向 / 0 中性
    ts: str


class ReflectiveMemory:
    """經驗的儲存與相似檢索（SQLite，向量存 JSON）。"""

    def __init__(self, db_path: str = "pai_memory.db", embedder=None):
        self.embedder = embedder or HashingEmbedder()
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock, self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT, event_summary TEXT, action TEXT, rationale TEXT,
                    feedback TEXT, score REAL, embedding TEXT
                )""")

    _FEEDBACK_SCORE = {"accepted": 1.0, "modified": 0.3, "ignored": -0.5, "rejected": -1.0}

    def add_experience(self, event_summary: str, action: str, rationale: str,
                       feedback: str):
        score = self._FEEDBACK_SCORE.get(feedback, 0.0)
        emb = self.embedder.embed(f"{event_summary} :: {action}")
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO experiences (ts,event_summary,action,rationale,feedback,score,embedding)"
                " VALUES (?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), event_summary, action,
                 rationale, feedback, score, json.dumps(emb)))

    def retrieve(self, event_summary: str, k: int = 3, min_sim: float = 0.25) -> list[dict]:
        query = self.embedder.embed(event_summary)
        with self._lock:
            rows = self.conn.execute(
                "SELECT event_summary,action,rationale,feedback,score,embedding FROM experiences"
            ).fetchall()
        scored = []
        for es, action, rationale, feedback, score, emb in rows:
            sim = _cosine(query, json.loads(emb))
            if sim >= min_sim:
                scored.append((sim, {"event": es, "action": action,
                                     "rationale": rationale, "feedback": feedback,
                                     "score": score, "similarity": round(sim, 3)}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _s, d in scored[:k]]

    def lessons_for(self, event_summary: str, k: int = 3) -> str:
        """把相似經驗整理成可注入 prompt 的「過往教訓」文字。"""
        hits = self.retrieve(event_summary, k=k)
        if not hits:
            return ""
        lines = []
        for h in hits:
            verdict = {"accepted": "使用者接受", "rejected": "使用者拒絕",
                       "modified": "使用者修改後採用", "ignored": "使用者忽略"}.get(
                           h["feedback"], h["feedback"])
            lines.append(f"- 相似情境「{h['event']}」→ 曾建議 {h['action']}（{h['rationale']}），"
                         f"結果：{verdict}（相似度 {h['similarity']}）")
        return "過往相似情境的回饋教訓（請據此校準本次判斷）：\n" + "\n".join(lines)
