"""PAI — Proactive AI Framework (主動式 AI 框架)

感知 (Triggers) → 決策 (Brain) → 治理 (Policy) → 行動 (Actions) → 記憶/回饋 (Memory)
"""
from .core import PAIAgent, Event, Intent, AutonomyLevel
from .triggers import IntervalTrigger, ScheduleTrigger, FileWatchTrigger, ThresholdTrigger
from .brain import RuleBrain, LLMBrain, Rule
from .policy import ProactivityPolicy
from .actions import ConsoleNotifier, WebhookNotifier, CallbackAction
from .memory import Memory
from .protocol import PAI_PROTOCOL_VERSION, build_record, to_json, save_pai, load_pai
from .paifile import PaiWriter, PaiReader, pack_agent, load_agent, bake_adapter_into_pai
from .finetune import (
    AdapterStore, EvalGate, SelfFinetuneManager,
    EchoBackend, LlamaFinetuneBackend, LlamaFactoryBackend, export_preference_dataset,
)
from .omni_brain import MiniCPMoBrain, DuplexOmniLoop
from .selfft import attach_self_finetuning
from .loader import load_runtime
from .learning import ReflectiveMemory, HashingEmbedder, EmbeddingClient

__version__ = "0.1.1"
__author__ = "vito1317 <service@vito1317.com>"

__all__ = [
    "PAIAgent", "Event", "Intent", "AutonomyLevel",
    "IntervalTrigger", "ScheduleTrigger", "FileWatchTrigger", "ThresholdTrigger",
    "RuleBrain", "LLMBrain", "Rule",
    "ProactivityPolicy",
    "ConsoleNotifier", "WebhookNotifier", "CallbackAction",
    "Memory",
    "PAI_PROTOCOL_VERSION", "build_record", "to_json", "save_pai", "load_pai",
    "PaiWriter", "PaiReader", "pack_agent", "load_agent", "load_runtime",
    "ReflectiveMemory", "HashingEmbedder", "EmbeddingClient",
    "bake_adapter_into_pai", "AdapterStore", "EvalGate", "SelfFinetuneManager",
    "EchoBackend", "LlamaFinetuneBackend", "LlamaFactoryBackend", "export_preference_dataset",
    "MiniCPMoBrain", "DuplexOmniLoop", "attach_self_finetuning",
]
