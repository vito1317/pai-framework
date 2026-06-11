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
import os
import shutil
import subprocess
import time
import urllib.request
from typing import Optional

from ._jsonutil import retry_call
from .brain import RuleBrain, parse_intents, _LLM_SYSTEM_PROMPT
from .core import AutonomyLevel, Event, Intent

logger = logging.getLogger("pai.server_brain")


class LlamaServerBrain:
    def __init__(self, available_actions: list[str],
                 weights_path: Optional[str] = None,
                 base_url: Optional[str] = None,      # 既有 server，如 http://127.0.0.1:8080
                 port: int = 8089,
                 n_ctx: int = 4096,
                 server_bin: str = "llama-server",
                 lora_path: Optional[str] = None,
                 lora_paths: Optional[list] = None,   # 啟動時預載多個 adapter（供執行期熱切換）
                 fallback: Optional[RuleBrain] = None,
                 user_profile: str = "",
                 startup_timeout: float = 300.0):
        self.available_actions = available_actions
        self.weights_path = weights_path
        self.base_url = base_url
        self.port = port
        self.n_ctx = n_ctx
        self.server_bin = server_bin
        # 啟動時預載的 adapter 清單（第一個預設為現役 scale=1，其餘 scale=0）。
        # 執行期可用 /lora-adapters 在這些已載入的 adapter 間熱切換，主幹不重載。
        self.lora_paths = list(lora_paths) if lora_paths else ([lora_path] if lora_path else [])
        self.lora_path = self.lora_paths[0] if self.lora_paths else None
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

        cmd = [binpath, "-m", self.weights_path, "--port", str(self.port),
               "-c", str(self.n_ctx), "--no-webui"]
        # 預載所有候選 adapter（之後可在它們之間執行期熱切換，主幹只載入一次）
        for p in self.lora_paths:
            if p and os.path.exists(p):
                cmd += ["--lora", p]
                logger.info("Preloading LoRA adapter: %s", p)
        logger.info("Starting llama-server on :%d with %s", self.port, self.weights_path)
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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

    # ---- 執行期 LoRA 熱切換（不重載 14.4GB 主幹）----
    def list_lora_adapters(self) -> list:
        """GET /lora-adapters：回傳已載入的 adapter（含 id / path / scale）。"""
        url = self._ensure_server()
        with urllib.request.urlopen(f"{url}/lora-adapters", timeout=15) as resp:
            return json.loads(resp.read())

    def set_lora_scales(self, scales: list) -> dict:
        """POST /lora-adapters：設定各 adapter 的 scale，例如 [{"id":0,"scale":1.0},...]。"""
        url = self._ensure_server()
        req = urllib.request.Request(
            f"{url}/lora-adapters", data=json.dumps(scales).encode(),
            headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {"status": resp.status}

    def activate_lora(self, path: str, scale: float = 1.0) -> dict:
        """執行期熱切換到指定 adapter：把它的 scale 設為 scale、其餘設 0，主幹不重載。

        該 adapter 必須在啟動時已用 --lora 預載（在 self.lora_paths 內）。
        若是訓練出的全新 adapter 不在預載清單，需重啟才能掛入（llama.cpp 現況限制）。
        """
        adapters = self.list_lora_adapters()
        scales, matched = [], False
        for a in adapters:
            apath = a.get("path", "")
            on = os.path.abspath(apath) == os.path.abspath(path)
            matched = matched or on
            scales.append({"id": a.get("id"), "scale": scale if on else 0.0})
        if not matched:
            raise ValueError(f"adapter 未預載，無法執行期熱切換：{path}"
                             "（請放進 lora_paths 啟動時預載，或重啟 server）")
        self.set_lora_scales(scales)
        self.lora_path = path
        logger.info("Hot-swapped active LoRA → %s (no base reload)", path)
        return {"active": path}

    def deactivate_lora(self) -> dict:
        """把所有 adapter scale 設 0（回到純主幹），主幹不重載。"""
        adapters = self.list_lora_adapters()
        self.set_lora_scales([{"id": a.get("id"), "scale": 0.0} for a in adapters])
        self.lora_path = None
        return {"active": None}

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

        def _do():
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())

        data = retry_call(_do, attempts=2)   # 暫時性錯誤先重試再談 fallback
        text = data["choices"][0]["message"]["content"]
        return parse_intents(text, self.available_actions)
