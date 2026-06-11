"""治理層：ProactivityPolicy 決定一個 Intent 最終被授予的自主等級。

主動式 AI 的核心風險是「過度打擾」與「越權行動」，因此所有意圖都必須過 policy gate：
1. 信心門檻：低信心一律降為 OBSERVE
2. 自主等級上限：依動作風險設定每個 action 的最高等級
3. 安靜時段：非緊急事項降為 OBSERVE（緊急可突破，但最多 SUGGEST）
4. 頻率限制：單位時間打擾次數上限，超過自動降級
5. 回饋調節：使用者常拒絕的動作自動變保守
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from .core import AutonomyLevel, Intent


class ProactivityPolicy:
    def __init__(
        self,
        min_confidence: float = 0.4,
        act_confidence: float = 0.85,
        default_max_level: AutonomyLevel = AutonomyLevel.SUGGEST,
        action_max_levels: Optional[dict[str, AutonomyLevel]] = None,
        quiet_hours: Optional[tuple[int, int]] = None,   # (22, 8) = 22:00–08:00
        max_interruptions_per_hour: int = 6,
        urgency_override: float = 0.9,                   # 緊急度 >= 此值可突破安靜時段
        decline_penalty_threshold: int = 3,              # 連續拒絕 N 次後該動作降級
        interruption_cost_fn=None,                       # () -> 0~1，估計當下打擾使用者的成本
    ):
        self.min_confidence = min_confidence
        self.act_confidence = act_confidence
        self.default_max_level = default_max_level
        self.action_max_levels = action_max_levels or {}
        self.quiet_hours = quiet_hours
        self.max_interruptions_per_hour = max_interruptions_per_hour
        self.urgency_override = urgency_override
        self.decline_penalty_threshold = decline_penalty_threshold
        self.interruption_cost_fn = interruption_cost_fn or (lambda: 0.3)
        self.last_interruption_cost: float = 0.0
        self._interruptions: list[float] = []

    def gate(self, intent: Intent, memory) -> AutonomyLevel:
        # 0. 估計當下打擾成本（PAI 干擾度公式的右側）
        self.last_interruption_cost = float(self.interruption_cost_fn())

        # 1. 信心門檻
        if intent.confidence < self.min_confidence:
            return AutonomyLevel.OBSERVE

        # 2. 等級上限 = min(意圖請求, 動作上限)
        cap = self.action_max_levels.get(intent.action, self.default_max_level)
        level = min(intent.requested_level, cap)

        # ACT 需要更高信心
        if level == AutonomyLevel.ACT and intent.confidence < self.act_confidence:
            level = AutonomyLevel.ASK

        # 3. 回饋調節：使用者最近頻繁拒絕此動作 → 降一級
        declines = memory.recent_declines(intent.action)
        if declines >= self.decline_penalty_threshold and level > AutonomyLevel.OBSERVE:
            level = AutonomyLevel(level - 1)

        # 4. 安靜時段
        if self._in_quiet_hours() and level > AutonomyLevel.OBSERVE:
            if intent.urgency >= self.urgency_override:
                level = min(level, AutonomyLevel.SUGGEST)
            else:
                return AutonomyLevel.OBSERVE

        # 5. PAI 干擾度公式：urgency × confidence 必須大於打擾成本，才允許打擾
        if level in (AutonomyLevel.SUGGEST, AutonomyLevel.ASK):
            if intent.urgency * intent.confidence <= self.last_interruption_cost:
                return AutonomyLevel.OBSERVE

        # 6. 頻率限制（SUGGEST/ASK 都算打擾）
        if level in (AutonomyLevel.SUGGEST, AutonomyLevel.ASK):
            now = time.time()
            self._interruptions = [t for t in self._interruptions if now - t < 3600]
            if len(self._interruptions) >= self.max_interruptions_per_hour:
                return AutonomyLevel.OBSERVE
            self._interruptions.append(now)

        return level

    def _in_quiet_hours(self) -> bool:
        if not self.quiet_hours:
            return False
        start, end = self.quiet_hours
        hour = datetime.now().hour
        return (start <= hour or hour < end) if start > end else (start <= hour < end)
