"""LocalLLMBrain — 用 .pai 內嵌的 GGUF 權重在本地推理的決策腦。

底層引擎：llama.cpp（經 llama-cpp-python 綁定）。
    pip install llama-cpp-python

模型無關：任何 GGUF 量化模型都可以（建議 4B~8B instruct 級，如
Qwen3-4B-Instruct Q4_K_M、Llama-3.x-8B Q4），打包進 .pai 的
weights.gguf 段即可離線運行。失敗時自動退回 fallback（RuleBrain）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from typing import Optional

from .brain import RuleBrain, _LLM_SYSTEM_PROMPT
from .core import AutonomyLevel, Event, Intent

logger = logging.getLogger("pai.local_brain")


class LocalLLMBrain:
    def __init__(self, available_actions: list[str],
                 weights_path: Optional[str] = None,    # 直接給 gguf 路徑
                 weights_bytes: Optional[bytes] = None, # 或給 .pai 段內容
                 n_ctx: int = 8192,
                 n_gpu_layers: int = -1,                # -1 = 全部丟 GPU/Metal
                 fallback: Optional[RuleBrain] = None,
                 user_profile: str = ""):
        self.available_actions = available_actions
        self.fallback = fallback
        self.user_profile = user_profile
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None
        self._disabled = False   # 載入失敗一次後永久走 fallback，避免重複重試
        self._weights_path = weights_path
        if weights_bytes is not None and weights_path is None:
            self._weights_path = self._materialize(weights_bytes)

    @staticmethod
    def _materialize(blob: bytes) -> str:
        """把 .pai 內嵌權重落地到快取（以內容雜湊命名，重複載入零成本）。"""
        digest = hashlib.sha256(blob).hexdigest()[:16]
        cache_dir = os.path.join(tempfile.gettempdir(), "pai_weights_cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{digest}.gguf")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(blob)
        return path

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        if not self._weights_path:
            raise RuntimeError("LocalLLMBrain 沒有可用的 GGUF 權重")
        from llama_cpp import Llama  # 延遲匯入，未安裝時走 fallback
        logger.info("Loading local GGUF model: %s", self._weights_path)
        self._llm = Llama(model_path=self._weights_path, n_ctx=self.n_ctx,
                          n_gpu_layers=self.n_gpu_layers, verbose=False)

    def decide(self, event: Event, context: dict) -> list[Intent]:
        if self._disabled:
            return self.fallback.decide(event, context) if self.fallback else []
        try:
            self._ensure_loaded()
            user_msg = json.dumps({
                "event": event.to_dict(), "context": context,
                "available_actions": self.available_actions,
                "user_profile": self.user_profile,
            }, ensure_ascii=False)
            out = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": _LLM_SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                max_tokens=1024, temperature=0.2,
            )
            text = out["choices"][0]["message"]["content"]
            return self._parse(text)
        except Exception as exc:  # noqa: BLE001
            if self._llm is None:
                self._disabled = True   # 模型載入失敗（如未裝 llama-cpp）→ 之後直接走 fallback
                logger.warning("Local LLM unavailable (%s); falling back to rules permanently", exc)
            else:
                logger.exception("Local LLM decision failed; using fallback")
            return self.fallback.decide(event, context) if self.fallback else []

    def _parse(self, text: str) -> list[Intent]:
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return []
        intents = []
        for it in json.loads(text[start:end + 1]):
            if it.get("action") not in self.available_actions:
                continue
            intents.append(Intent(
                action=it["action"], params=it.get("params", {}),
                confidence=float(it.get("confidence", 0.5)),
                urgency=float(it.get("urgency", 0.5)),
                rationale=it.get("rationale", ""),
                requested_level=AutonomyLevel(int(it.get("requested_level", 1))),
            ))
        return intents
