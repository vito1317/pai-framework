"""attach_self_finetuning — 一行把完整 self-finetuning（1+2+3）掛到任何 PAI agent。

模型無關：對 Gemma（llama-server）、MiniCPM-o（llama-server 或 transformers）皆適用。
自動依 agent 的決策腦選擇對應的訓練後端與熱切換機制：
  - LlamaServerBrain      → LlamaFinetuneBackend（GGUF LoRA）+ /lora-adapters 熱切換
  - MiniCPMoBrain(omni=transformers) → LlamaFactoryBackend（PEFT LoRA）+ PEFT 熱切換

回傳 SelfFinetuneManager；呼叫 .run_periodic(interval) 即定期自我微調並上線。
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from .finetune import (EvalGate, LlamaFactoryBackend, LlamaFinetuneBackend,
                       SelfFinetuneManager)


def attach_self_finetuning(
    agent,
    base_gguf: Optional[str] = None,
    base_model: str = "openbmb/MiniCPM-o-4_5",
    eval_fn: Optional[Callable] = None,
    min_samples: int = 200,
    min_gain: float = 0.02,
    adapters_root: Optional[str] = None,
    memory_path: str = "pai_memory.db",
) -> SelfFinetuneManager:
    brain = agent.brain
    bname = type(brain).__name__
    adapters_root = adapters_root or os.path.join(
        os.path.dirname(os.path.abspath(memory_path)), "pai_adapters")

    # 依決策腦挑訓練後端
    if bname == "LlamaServerBrain":
        trainer = LlamaFinetuneBackend()                  # GGUF LoRA
        base = base_gguf or getattr(brain, "weights_path", None)
    elif bname == "MiniCPMoBrain":
        if getattr(brain, "engine", "") == "transformers":
            trainer = LlamaFactoryBackend(base_model=getattr(brain, "model_path", base_model))
            base = None
        else:                                             # MiniCPM-o GGUF via llama-server
            trainer = LlamaFinetuneBackend()
            base = base_gguf or getattr(brain, "model_path", None)
    else:
        raise RuntimeError(f"{bname} 不支援 LoRA self-finetuning（僅記憶式學習）")

    gate = EvalGate(eval_fn, min_gain=min_gain,
                    non_regression=["interrupt_precision"]) if eval_fn else None
    return SelfFinetuneManager(
        db_path=memory_path, adapters_root=adapters_root, trainer=trainer,
        eval_gate=gate, base_gguf=base, min_samples=min_samples, brain=brain)
