"""記憶層：事件、意圖、結果與使用者回饋的持久化（SQLite，零依賴）。

提供 build_context() 給決策腦：近期事件 + 同類動作的歷史成效 + 拒絕統計，
讓 PAI 的主動行為能隨回饋自我調節。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from .core import Event, Intent


class Memory:
    def __init__(self, db_path: str = "pai_memory.db"):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        with self._lock, self.conn:
            self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, source TEXT, kind TEXT,
                payload TEXT, ts TEXT
            );
            CREATE TABLE IF NOT EXISTS intents (
                ts TEXT, event_id TEXT, action TEXT, params TEXT,
                confidence REAL, urgency REAL, rationale TEXT,
                requested_level INTEGER, granted_level INTEGER
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                ts TEXT, action TEXT, status TEXT, result TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback (
                ts TEXT, action TEXT, positive INTEGER
            );
            CREATE TABLE IF NOT EXISTS protocol_records (
                record_id TEXT PRIMARY KEY, ts TEXT, record TEXT
            );
            """)

    def record_protocol(self, record: dict):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO protocol_records VALUES (?,?,?)",
                (record["record_id"], record["timestamp"],
                 json.dumps(record, ensure_ascii=False)))

    def latest_protocol_records(self, n: int = 5) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT record FROM protocol_records ORDER BY ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    # ---- 寫入 ----
    def record_event(self, event: Event):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO events VALUES (?,?,?,?,?)",
                (event.id, event.source, event.kind,
                 json.dumps(event.payload, ensure_ascii=False), event.ts.isoformat()))

    def record_intent(self, intent: Intent, granted: int):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO intents VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), intent.event_id,
                 intent.action, json.dumps(intent.params, ensure_ascii=False),
                 intent.confidence, intent.urgency, intent.rationale,
                 int(intent.requested_level), granted))

    def record_outcome(self, intent: Intent, status: str, result=None):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO outcomes VALUES (?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), intent.action, status,
                 json.dumps(result, ensure_ascii=False, default=str)))

    def record_feedback(self, intent: Intent, positive: bool):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO feedback VALUES (?,?,?)",
                (datetime.now(timezone.utc).isoformat(), intent.action, int(positive)))

    # ---- 查詢 ----
    def recent_declines(self, action: str, hours: int = 24) -> int:
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM outcomes WHERE action=? AND status='declined' AND ts>=?",
                (action, since)).fetchone()
        return row[0]

    def build_context(self, event: Event, recent_n: int = 10) -> dict:
        """組裝給決策腦的上下文。"""
        with self._lock:
            recent = self.conn.execute(
                "SELECT source, kind, payload, ts FROM events ORDER BY ts DESC LIMIT ?",
                (recent_n,)).fetchall()
            stats = self.conn.execute(
                "SELECT action, status, COUNT(*) FROM outcomes GROUP BY action, status"
            ).fetchall()
        return {
            "recent_events": [
                {"source": s, "kind": k, "payload": json.loads(p or "{}"), "ts": t}
                for s, k, p, t in recent
            ],
            "action_history": [
                {"action": a, "status": st, "count": c} for a, st, c in stats
            ],
            "now": datetime.now(timezone.utc).isoformat(),
        }
