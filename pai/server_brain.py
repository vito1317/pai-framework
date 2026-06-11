"""LlamaServerBrain — 透過本地 llama-server（llama.cpp 官方 HTTP 服務）推理的決策腦。

優點：llama.cpp 主線更新最快（新架構如 gemma4 第一時間支援），
且權重以 mmap 載入、服務常駐，多次決策免重複載入。

    brew install llama.cpp

行為：
- 若指定 base_url 且可連線 → 直接使用現有 server
- 否則用 weights_path 自動啟動 llama-server 子行程（首次呼叫時），結束時可 close()
"""
from __future__ import annotations

import atexit
import json
import logging
import shutil
import subprocess
import time
import urllib.request
from typing import Optional

from .brain import RuleBrain, _LLM_SYSTEM_PROMPT
from .core import AutonomyLevel, Event, Intent

logger = logging.getLogger("pai.server_brain")


class LlamaServerBrain:
    def __init__(self, available_actions: list[str],
                 weights_path: Optional[str] = None,
                 base_url: Optional[str] = None,      # 既有 server，如 http://127.0.0.1:8080
                 port: int = 8089,
                 n_ctx: int = 4096,
                 server_bin: str = "llama-server",
                 fallback: Optional[RuleBrain] = None,
                 user_profile: str = "",
                 startup_timeout: float = 300.0):
        self.available_actions = available_actions
        self.weights_path = weights_path
        self.base_url = base_url
        self.port = port
        self.n_ctx = n_ctx
        self.server_bin = server_bin
        self.fallback = fallback
        self.user_profile = user_profile
        self.startup_timeout = startup_timeout
        self._proc: Optional[subprocess.Popen] = None
        self._disabled = False

    # ---- server 生命週期 ----
    def _healthy(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as resp:
                return resp.status == 200
        except Exception:  # noqa: BLE001
            return False

    def _ensure_server(self) -> str:
        if self.base_url and self._healthy(self.base_url):
            return self.base_url
        url = f"http://127.0.0.1:{self.port}"
        if self._healthy(url):
            return url
        if self._proc is not None:           # 已啟動但還沒 ready
            return self._wait_ready(url)

        if not self.weights_path:
            raise RuntimeError("沒有可用的 llama-server，也沒有權重可自行啟動")
        binpath = shutil.which(self.server_bin)
        if not binpath:
            raise RuntimeError(f"找不到 {self.server_bin}（brew install llama.cpp）")

        logger.info("Starting llama-server on :%d with %s", self.port, self.weights_path)
        self._proc = subprocess.Popen(
            [binpath, "-m", self.weights_path, "--port", str(self.port),
             "-c", str(self.n_ctx), "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)
        return self._wait_ready(url)

    def _wait_ready(self, url: str) -> str:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError(f"llama-server 異常退出（code={self._proc.returncode}）")
            if self._healthy(url):
                logger.info("llama-server ready at %s", url)
                return url
            time.sleep(2)
        raise RuntimeError("llama-server 啟動逾時")

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    # ---- 決策 ----
    def decide(self, event: Event, context: dict) -> list[Intent]:
        if self._disabled:
            return self.fallback.decide(event, context) if self.fallback else []
        try:
            url = self._ensure_server()
            return self._decide(url, event, context)
        except Exception as exc:  # noqa: BLE001
            if self._proc is None or (self._proc and self._proc.poll() is not None):
                self._disabled = True
                logger.warning("llama-server unavailable (%s); falling back to rules", exc)
            else:
                logger.exception("llama-server decision failed; using fallback")
            return self.fallback.decide(event, context) if self.fallback else []

    def _decide(self, url: str, event: Event, context: dict) -> list[Intent]:
        user_msg = json.dumps({
            "event": event.to_dict(), "context": context,
            "available_actions": self.available_actions,
            "user_profile": self.user_profile,
        }, ensure_ascii=False)
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "system", "content": _LLM_SYSTEM_PROMPT},
                             {"role": "user", "content": user_msg}],
                "max_tokens": 1024, "temperature": 0.2,
            }).encode(),
            headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
        text = data["choices"][0]["message"]["content"]
        return self._parse(text)

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
