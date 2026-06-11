"""行動層：可被 Intent 觸發的動作。"""
from __future__ import annotations

import json
from typing import Callable

from .core import Intent


class ConsoleNotifier:
    """把通知印到終端（開發/示範用）。"""

    def execute(self, intent: Intent):
        params = intent.params
        title = params.get("title", intent.action)
        body = params.get("body", intent.rationale)
        print(f"\n🔔 [PAI 通知] {title}\n   {body}")
        return {"notified": True}


class WebhookNotifier:
    """POST JSON 到任意 webhook（Slack/Discord/自建服務）。"""

    def __init__(self, url: str):
        self.url = url

    def execute(self, intent: Intent):
        import urllib.request
        data = json.dumps({
            "title": intent.params.get("title", intent.action),
            "body": intent.params.get("body", intent.rationale),
            "intent": intent.to_dict(),
        }, ensure_ascii=False).encode()
        req = urllib.request.Request(
            self.url, data=data, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"status": resp.status}


class CallbackAction:
    """把任意 Python 函式包成動作。"""

    def __init__(self, fn: Callable[[Intent], object]):
        self.fn = fn

    def execute(self, intent: Intent):
        return self.fn(intent)
