"""決策層：把 Event + 記憶上下文 轉成 0..n 個 Intent。

兩種腦：
- RuleBrain：純規則，零依賴、可離線運行，適合確定性場景與 LLM 的安全後備。
- LLMBrain：呼叫 LLM（預設 Anthropic API）做開放式判斷，輸出結構化 JSON 意圖。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

from ._jsonutil import extract_json_array, retry_call
from .core import AutonomyLevel, Event, Intent

logger = logging.getLogger("pai.brain")


def parse_intents(text: str, available_actions) -> list[Intent]:
    """共用：把 LLM 文字輸出穩健地解析成 Intent 清單。

    用括號配對掃描抽出第一個平衡的 JSON 陣列（容忍多陣列/夾帶文字/code fence），
    過濾未登記的 action，並對缺欄位給安全預設。所有決策腦共用此函式。
    """
    intents = []
    for it in extract_json_array(text):
        if not isinstance(it, dict) or it.get("action") not in available_actions:
            continue
        try:
            intents.append(Intent(
                action=it["action"],
                params=it.get("params", {}) or {},
                confidence=float(it.get("confidence", 0.5)),
                urgency=float(it.get("urgency", 0.5)),
                rationale=it.get("rationale", ""),
                requested_level=AutonomyLevel(int(it.get("requested_level", 1))),
            ))
        except (ValueError, TypeError):
            continue
    return intents


@dataclass
class Rule:
    """單條規則：match(event, context) → 可選 Intent。"""
    name: str
    match: Callable[[Event, dict], Optional[Intent]]


class RuleBrain:
    def __init__(self, rules: Optional[list[Rule]] = None):
        self.rules = rules or []

    def add_rule(self, rule: Rule) -> "RuleBrain":
        self.rules.append(rule)
        return self

    def decide(self, event: Event, context: dict) -> list[Intent]:
        intents = []
        for rule in self.rules:
            try:
                intent = rule.match(event, context)
                if intent is not None:
                    intents.append(intent)
            except Exception:  # noqa: BLE001
                logger.exception("Rule '%s' raised", rule.name)
        return intents


_LLM_SYSTEM_PROMPT = """\
You are the decision engine of a proactive AI agent.
Given an event and context, decide whether the agent should act proactively.

Respond ONLY with a JSON array (possibly empty) of intents:
[{"action": "<one of the available actions>",
  "params": {},
  "confidence": 0.0-1.0,
  "urgency": 0.0-1.0,
  "rationale": "<short reason, same language as the user>",
  "requested_level": 0|1|2|3}]

Levels: 0=observe only, 1=suggest/notify, 2=ask for confirmation, 3=act autonomously.
Be conservative: prefer lower levels unless confidence is high and risk is low.
Return [] when no proactive behavior is warranted (most events deserve []).
"""


class LLMBrain:
    """以 LLM 為決策核心。失敗時自動退回 fallback（通常是 RuleBrain）。"""

    def __init__(self, available_actions: list[str],
                 model: str = "claude-sonnet-4-6",
                 api_key: Optional[str] = None,
                 fallback: Optional[RuleBrain] = None,
                 user_profile: str = ""):
        self.available_actions = available_actions
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.fallback = fallback
        self.user_profile = user_profile

    def decide(self, event: Event, context: dict) -> list[Intent]:
        try:
            return self._decide_llm(event, context)
        except Exception:  # noqa: BLE001
            logger.exception("LLM decision failed; using fallback")
            if self.fallback:
                return self.fallback.decide(event, context)
            return []

    def _decide_llm(self, event: Event, context: dict) -> list[Intent]:
        import urllib.request

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        user_msg = json.dumps({
            "event": event.to_dict(),
            "context": context,
            "available_actions": self.available_actions,
            "user_profile": self.user_profile,
        }, ensure_ascii=False)

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({
                "model": self.model,
                "max_tokens": 1024,
                "system": _LLM_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_msg}],
            }).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )

        def _do():
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())

        # 暫時性網路錯誤先重試一次，救回 LLM 級判斷再談 fallback
        data = retry_call(_do, attempts=2)
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return self._parse(text)

    def _parse(self, text: str) -> list[Intent]:
        return parse_intents(text, self.available_actions)
