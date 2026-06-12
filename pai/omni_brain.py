"""MiniCPMoBrain — 以 openbmb/MiniCPM-o-4_5 為決策腦的全雙工 omni 版本。

MiniCPM-o 4.5（9B，建構於 SigLip2 + Whisper-medium + CosyVoice2 + Qwen3-8B）能即時
同時處理影像/音訊輸入串流，並同步產生文字＋語音輸出，且**原生支援主動式互動**
（根據對現場的持續理解主動發起提醒/評論）——天然契合 PAI 的主動式範式。

本模組提供兩種接法：
1. MiniCPMoBrain（決策腦）：把 PAI 的「事件→意圖」交給 MiniCPM-o 判斷。
   - engine="transformers"：用官方 PyTorch 棧（trust_remote_code，支援多模態/語音輸入）
   - engine="llama-server"：用 openbmb/MiniCPM-o-4_5-gguf 經 llama.cpp（純文字決策，最省資源）
2. DuplexOmniLoop（全雙工迴圈）：包裝 streaming_prefill / streaming_generate，
   讓 MiniCPM-o 在背景持續看＋聽，主動產生「主動意圖」餵回 PAI 的治理層。

依賴（僅 transformers 路徑需要）：
    pip install "transformers==4.51.0" accelerate "torch>=2.3.0" torchaudio "minicpmo-utils[all]>=1.0.5"
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ._jsonutil import retry_call
from .brain import RuleBrain, parse_intents, _LLM_SYSTEM_PROMPT
from .core import AutonomyLevel, Event, Intent

logger = logging.getLogger("pai.omni_brain")


class MiniCPMoBrain:
    """MiniCPM-o 4.5 決策腦。預設用 transformers PyTorch 棧；失敗退回 fallback。"""

    def __init__(self, available_actions: list[str],
                 model_path: str = "openbmb/MiniCPM-o-4_5",
                 engine: str = "transformers",      # transformers | llama-server
                 base_url: Optional[str] = None,    # engine=llama-server 時的端點
                 device: Optional[str] = None,      # None=自動偵測 cuda→mps→cpu
                 init_audio: bool = True,
                 init_tts: bool = True,
                 init_vision: bool = True,
                 fallback: Optional[RuleBrain] = None,
                 user_profile: str = ""):
        self.available_actions = available_actions
        self.model_path = model_path
        self.engine = engine
        self.base_url = base_url
        self.device = device
        self.init_audio = init_audio
        self.init_tts = init_tts
        self.init_vision = init_vision
        self.fallback = fallback
        self.user_profile = user_profile
        self._model = None
        self._disabled = False

    @staticmethod
    def _auto_device() -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ---- 模型載入（transformers 路徑）----
    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel
        device = self.device or self._auto_device()
        self.device = device
        logger.info("Loading MiniCPM-o 4.5 (%s) on %s ...", self.model_path, device)
        model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
            init_vision=self.init_vision, init_audio=self.init_audio,
            init_tts=self.init_tts)
        model = model.eval().to(device)
        if self.init_tts:
            model.init_tts()
        self._model = model

    # ---- 執行期 LoRA 熱切換（self-finetuning 第二層；主幹不重載）----
    def activate_lora(self, adapter_path: str, adapter_name: str = "active") -> dict:
        """transformers/PEFT 路徑：載入並切到指定 LoRA adapter，主幹權重不動。

        llama-server 路徑請改用 LlamaServerBrain.activate_lora（/lora-adapters）。
        """
        if self.engine != "transformers":
            raise RuntimeError("activate_lora 僅用於 transformers 路徑；"
                               "llama-server 路徑請用 LlamaServerBrain.activate_lora")
        self._ensure_model()
        from peft import PeftModel
        # 已是 PeftModel 就加 adapter，否則包成 PeftModel
        if isinstance(self._model, PeftModel) or hasattr(self._model, "load_adapter"):
            try:
                self._model.load_adapter(adapter_path, adapter_name=adapter_name)
            except Exception:  # 同名已載入 → 直接切換
                pass
            self._model.set_adapter(adapter_name)
        else:
            self._model = PeftModel.from_pretrained(
                self._model, adapter_path, adapter_name=adapter_name)
        logger.info("MiniCPM-o transformers: activated LoRA %s (no base reload)", adapter_path)
        return {"active": adapter_path}

    def deactivate_lora(self) -> dict:
        """停用 adapter，回到純主幹（主幹不重載）。"""
        if self._model is not None and hasattr(self._model, "disable_adapter_layers"):
            self._model.disable_adapter_layers()
        return {"active": None}

    # ---- 決策 ----
    def decide(self, event: Event, context: dict) -> list[Intent]:
        if self._disabled:
            return self.fallback.decide(event, context) if self.fallback else []
        try:
            if self.engine == "llama-server":
                return self._decide_server(event, context)
            return self._decide_transformers(event, context)
        except Exception as exc:  # noqa: BLE001
            if self._model is None and self.engine == "transformers":
                self._disabled = True
                logger.warning("MiniCPM-o unavailable (%s); falling back to rules", exc)
            else:
                logger.exception("MiniCPM-o decision failed; using fallback")
            return self.fallback.decide(event, context) if self.fallback else []

    def _user_payload(self, event: Event, context: dict) -> str:
        return json.dumps({
            "event": event.to_dict(), "context": context,
            "available_actions": self.available_actions,
            "user_profile": self.user_profile,
        }, ensure_ascii=False)

    def _decide_transformers(self, event: Event, context: dict) -> list[Intent]:
        self._ensure_model()
        msgs = [{"role": "user", "content": [self._user_payload(event, context)]}]
        # 純文字決策：不產生語音，關閉 thinking 以求快
        text = self._model.chat(msgs=msgs, system_prompt=_LLM_SYSTEM_PROMPT,
                                generate_audio=False, max_new_tokens=1024,
                                temperature=0.2)
        if isinstance(text, tuple):   # (text, audio) 形式時取文字
            text = text[0]
        return parse_intents(text, self.available_actions)

    def _decide_server(self, event: Event, context: dict) -> list[Intent]:
        import urllib.request
        url = (self.base_url or "http://127.0.0.1:8089").rstrip("/")
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=json.dumps({
                "messages": [{"role": "system", "content": _LLM_SYSTEM_PROMPT},
                             {"role": "user", "content": self._user_payload(event, context)}],
                "max_tokens": 1024, "temperature": 0.2,
            }).encode(),
            headers={"content-type": "application/json"})

        def _do():
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())

        data = retry_call(_do, attempts=2)
        return parse_intents(data["choices"][0]["message"]["content"],
                             self.available_actions)


class DuplexOmniLoop:
    """全雙工迴圈：MiniCPM-o 持續看＋聽串流，主動產生意圖回灌 PAI 治理層。

    與離散觸發器並存——這是一個「持續感知 trigger」的高階形態：
    模型自己決定何時該主動發話/提醒，PAI policy 仍負責是否放行、用什麼等級打擾。
    """

    def __init__(self, brain: MiniCPMoBrain, agent, session_id: str = "duplex",
                 ref_audio_path: Optional[str] = None, language: str = "zh"):
        self.brain = brain
        self.agent = agent
        self.session_id = session_id
        self.ref_audio_path = ref_audio_path
        self.language = language

    def run(self, omni_stream):
        """omni_stream: 產生 (video_frame|audio_chunk) 的可迭代物（即時或錄影）。

        每當模型主動輸出文字，封裝成 proactive 事件交給 agent 的治理層處理。
        """
        self.brain._ensure_model()
        model = self.brain._model
        import librosa  # noqa: F401  （ref_audio 載入用）

        model.reset_session()
        if self.ref_audio_path:
            ref, _ = __import__("librosa").load(self.ref_audio_path, sr=16000, mono=True)
            model.init_token2wav_cache(ref)
            sys_msg = model.get_sys_prompt(ref_audio=ref, mode="omni", language=self.language)
        else:
            sys_msg = model.get_sys_prompt(mode="omni", language=self.language)
        model.streaming_prefill(session_id=self.session_id, msgs=[sys_msg])

        for chunk in omni_stream:
            model.streaming_prefill(
                session_id=self.session_id,
                msgs=[{"role": "user", "content": [chunk]}],
                omni_mode=True, is_last_chunk=False)
            # 讓模型決定是否主動發話（短輪詢）
            for text_chunk, finished in model.streaming_generate(
                    session_id=self.session_id, generate_audio=False,
                    use_tts_template=True, enable_thinking=False, do_sample=False):
                if text_chunk.strip():
                    ev = Event(source="omni-duplex", kind="proactive_utterance",
                               payload={"text": text_chunk})
                    # 交回 PAI：由治理層決定是否真的打擾使用者
                    self.agent._handle_event(ev)
                if finished:
                    break
