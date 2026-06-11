"""self-finetuning 第二/三層：LoRA adapter 訓練、熱插拔、eval gate。

設計重點：
- 主幹權重（weights.gguf）永遠不動；學習只產生/切換小的 LoRA adapter。
- adapter 以「側車」AdapterStore 管理（小檔、可即時切換、保留歷史可回滾），
  不必每次重寫 14GB 的 .pai。發布時可用 bake_adapter_into_pai() 烘焙進 .pai。
- 任何 adapter 上線前必須通過 EvalGate（離線比較新舊，新的顯著勝出才 promote）。
- 訓練後端可插拔：LlamaFinetuneBackend（真實）/ EchoBackend（無 GPU 測管線）。

資料流：
  回饋(experiences) → 偏好資料集 jsonl → TrainerBackend → 候選 adapter
                   → EvalGate(候選 vs 現役) → promote 或 丟棄 → AdapterStore + 稽核
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Protocol

logger = logging.getLogger("pai.finetune")


# ---------- 偏好資料集 ----------

def export_preference_dataset(db_path: str, out_jsonl: str) -> int:
    """把 experiences 表轉成偏好訓練樣本（accepted=正例、rejected=負例）。

    回傳寫出的樣本數。樣本格式（chat 偏好對）：
      {"prompt": <情境>, "chosen": <好行為>, "rejected": <壞行為>}
    accepted → chosen；rejected → rejected。同情境若兩者都有則配成對比對。
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT event_summary, action, rationale, feedback, score "
            "FROM experiences").fetchall()
    except sqlite3.OperationalError:
        return 0
    finally:
        conn.close()

    by_event: dict[str, dict] = {}
    for es, action, rationale, feedback, score in rows:
        slot = by_event.setdefault(es, {"good": [], "bad": []})
        line = f"{action}: {rationale}"
        if feedback in ("accepted", "modified"):
            slot["good"].append(line)
        elif feedback in ("rejected", "ignored"):
            slot["bad"].append(line)

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for es, slot in by_event.items():
            chosen = slot["good"][0] if slot["good"] else "（保持靜默，不主動打擾）"
            rejected = slot["bad"][0] if slot["bad"] else None
            if rejected is None and not slot["good"]:
                continue
            f.write(json.dumps({
                "prompt": f"情境：{es}\n該如何主動回應？",
                "chosen": chosen,
                "rejected": rejected or "（過度主動地打擾使用者）",
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------- 訓練後端 ----------

class TrainerBackend(Protocol):
    def train(self, dataset_jsonl: str, base_gguf: Optional[str], out_adapter: str) -> str:
        ...


class EchoBackend:
    """無 GPU 的測試後端：產生一個帶 metadata 的佔位 adapter，用來驗證整條管線。"""

    def train(self, dataset_jsonl: str, base_gguf: Optional[str], out_adapter: str) -> str:
        with open(dataset_jsonl, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        meta = {"_pai_stub_adapter": True, "trained_at": time.time(),
                "n_samples": n, "base": os.path.basename(base_gguf or "")}
        with open(out_adapter, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        logger.info("EchoBackend produced stub adapter (%d samples) → %s", n, out_adapter)
        return out_adapter


class LlamaFactoryBackend:
    """真實後端：包 LLaMA-Factory 對 MiniCPM-o 4.5 做 LoRA 微調。

    MiniCPM-o 官方建議用 LLaMA-Factory 微調。產出的是 PEFT LoRA adapter
    （safetensors），主幹權重不動。需先 `pip install llamafactory` 並備妥 dataset 設定。
    """

    def __init__(self, base_model: str = "openbmb/MiniCPM-o-4_5",
                 template: str = "minicpm_o", extra_args: Optional[dict] = None):
        self.base_model = base_model
        self.template = template
        self.extra_args = extra_args or {}

    def train(self, dataset_jsonl: str, base_gguf: Optional[str], out_adapter: str) -> str:
        from llamafactory.train.tuner import run_exp  # 延遲匯入
        args = {
            "stage": "sft", "do_train": True, "model_name_or_path": self.base_model,
            "dataset_dir": os.path.dirname(dataset_jsonl) or ".",
            "dataset": os.path.splitext(os.path.basename(dataset_jsonl))[0],
            "template": self.template, "finetuning_type": "lora",
            "lora_target": "all", "output_dir": out_adapter,
            "overwrite_output_dir": True, "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8, "lr_scheduler_type": "cosine",
            "learning_rate": 1e-4, "num_train_epochs": 3.0, "bf16": True,
            **self.extra_args,
        }
        logger.info("Running LLaMA-Factory LoRA SFT on %s", self.base_model)
        run_exp(args)
        return out_adapter


class LlamaFinetuneBackend:
    """真實後端：包 llama.cpp 的 llama-finetune，產生 GGUF 格式 LoRA adapter。

    需要 llama.cpp 編譯出 llama-finetune（部分發行版為 finetune）。
    主幹 base_gguf 不會被修改，只輸出 adapter。
    """

    def __init__(self, bin_path: str = "llama-finetune", extra_args: Optional[list] = None):
        self.bin_path = bin_path
        self.extra_args = extra_args or []

    def train(self, dataset_jsonl: str, base_gguf: Optional[str], out_adapter: str) -> str:
        exe = shutil.which(self.bin_path)
        if not exe or not base_gguf:
            raise RuntimeError("llama-finetune 不可用或缺少 base_gguf")
        cmd = [exe, "-m", base_gguf, "--lora-out", out_adapter,
               "--train-data", dataset_jsonl, *self.extra_args]
        logger.info("Running llama-finetune: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return out_adapter


# ---------- adapter 側車儲存 ----------

@dataclass
class AdapterRecord:
    id: str
    path: str
    created_at: str
    n_samples: int
    metrics: dict
    active: bool


class AdapterStore:
    """管理 adapter 檔案與 active 指標（小檔，不動 14GB 主幹）。"""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.index_path = os.path.join(root, "index.json")
        self.index: list[dict] = self._load()

    def _load(self) -> list[dict]:
        if os.path.exists(self.index_path):
            with open(self.index_path, encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def add(self, src_adapter: str, n_samples: int, metrics: dict, active: bool = False) -> AdapterRecord:
        aid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_") + uuid.uuid4().hex[:6]
        ext = os.path.splitext(src_adapter)[1] or ".gguf"
        dst = os.path.join(self.root, f"adapter-{aid}{ext}")
        shutil.copyfile(src_adapter, dst)
        rec = AdapterRecord(aid, dst, datetime.now(timezone.utc).isoformat(),
                            n_samples, metrics, active)
        if active:
            for r in self.index:
                r["active"] = False
        self.index.append(asdict(rec))
        self._save()
        return rec

    def active_adapter(self) -> Optional[str]:
        for r in self.index:
            if r["active"]:
                return r["path"]
        return None

    def promote(self, adapter_id: str):
        for r in self.index:
            r["active"] = (r["id"] == adapter_id)
        self._save()

    def rollback(self):
        """切回上一個（時間序）adapter；若只剩一個則停用學習成果。"""
        active_idx = next((i for i, r in enumerate(self.index) if r["active"]), None)
        for r in self.index:
            r["active"] = False
        if active_idx is not None and active_idx > 0:
            self.index[active_idx - 1]["active"] = True
        self._save()


# ---------- eval gate（第三層） ----------

class EvalGate:
    """離線品質閘門：候選 adapter 必須顯著勝過現役才放行。

    eval_fn(adapter_path_or_None) -> 指標 dict（如 {"accuracy": .., "interrupt_precision": ..}）。
    預設用 PAI Protocol 累積的「使用者明確回饋」情境當 eval set（呼叫端注入 eval_fn）。
    """

    def __init__(self, eval_fn: Callable[[Optional[str]], dict],
                 primary_metric: str = "accuracy",
                 min_gain: float = 0.02,
                 non_regression: Optional[list[str]] = None):
        self.eval_fn = eval_fn
        self.primary_metric = primary_metric
        self.min_gain = min_gain
        self.non_regression = non_regression or []

    def judge(self, candidate_adapter: str, active_adapter: Optional[str]) -> dict:
        cand = self.eval_fn(candidate_adapter)
        base = self.eval_fn(active_adapter)
        gain = cand.get(self.primary_metric, 0) - base.get(self.primary_metric, 0)
        regressed = [m for m in self.non_regression
                     if cand.get(m, 0) < base.get(m, 0)]
        promote = gain >= self.min_gain and not regressed
        return {"promote": promote, "gain": round(gain, 4),
                "regressed": regressed, "candidate_metrics": cand,
                "active_metrics": base}


# ---------- 編排器 ----------

class SelfFinetuneManager:
    """串起：累積回饋 → 訓練 → eval gate → promote/丟棄 → 稽核。"""

    def __init__(self, db_path: str, adapters_root: str,
                 trainer: TrainerBackend, eval_gate: Optional[EvalGate] = None,
                 base_gguf: Optional[str] = None, min_samples: int = 50):
        self.db_path = db_path
        self.store = AdapterStore(adapters_root)
        self.trainer = trainer
        self.eval_gate = eval_gate
        self.base_gguf = base_gguf
        self.min_samples = min_samples
        self.audit_path = os.path.join(adapters_root, "audit.jsonl")

    def _audit(self, entry: dict):
        entry["ts"] = datetime.now(timezone.utc).isoformat()
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def maybe_train_and_promote(self, workdir: Optional[str] = None) -> dict:
        workdir = workdir or self.store.root
        ds = os.path.join(workdir, "feedback.jsonl")
        n = export_preference_dataset(self.db_path, ds)
        if n < self.min_samples:
            return {"trained": False, "reason": f"樣本不足 {n}/{self.min_samples}"}

        cand = os.path.join(workdir, "candidate-adapter.gguf")
        self.trainer.train(ds, self.base_gguf, cand)

        decision = {"promote": True, "gain": None}
        if self.eval_gate is not None:
            decision = self.eval_gate.judge(cand, self.store.active_adapter())

        if decision["promote"]:
            rec = self.store.add(cand, n_samples=n,
                                 metrics=decision.get("candidate_metrics", {}),
                                 active=True)
            self._audit({"event": "promote", "adapter": rec.id,
                         "n_samples": n, "decision": decision})
            return {"trained": True, "promoted": True, "adapter": rec.id,
                    "decision": decision}

        self._audit({"event": "reject", "n_samples": n, "decision": decision})
        return {"trained": True, "promoted": False, "decision": decision}
