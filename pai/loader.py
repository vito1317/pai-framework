"""Loader — 把 .pai 檔案實例化成可運行的 PAIAgent。

宣告式定義（triggers.json / rules.json / actions.json / policy.json / brain.json）
→ 框架物件。需要本機能力的部分（動作 handler、threshold 的 metric 函式）
由呼叫端以 registry 注入，.pai 檔內只存名稱——確保檔案本身不含可執行碼（安全）。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .actions import CallbackAction, ConsoleNotifier, WebhookNotifier
from .brain import LLMBrain, Rule, RuleBrain
from .core import AutonomyLevel, Intent, PAIAgent
from .memory import Memory
from .paifile import PaiReader
from .policy import ProactivityPolicy
from .triggers import FileWatchTrigger, IntervalTrigger, ScheduleTrigger, ThresholdTrigger

logger = logging.getLogger("pai.loader")


def _make_rule(decl: dict) -> Rule:
    """宣告式規則 → Rule。when 條件支援 kind/source 精確比對與 payload 比較。"""
    when, spec = decl.get("when", {}), decl["intent"]

    def match(event, context):
        if "kind" in when and event.kind != when["kind"]:
            return None
        if "source" in when and event.source != when["source"]:
            return None
        for cond in when.get("payload", []):   # {"field": "todos", "op": ">", "value": 5}
            v = event.payload.get(cond["field"])
            if v is None:
                return None
            op, ref = cond.get("op", "=="), cond["value"]
            ok = {"==": v == ref, "!=": v != ref, ">": v > ref,
                  "<": v < ref, ">=": v >= ref, "<=": v <= ref}.get(op, False)
            if not ok:
                return None
        return Intent(
            action=spec["action"],
            params={**spec.get("params", {}), "event_payload": event.payload},
            confidence=float(spec.get("confidence", 0.6)),
            urgency=float(spec.get("urgency", 0.5)),
            rationale=spec.get("rationale", ""),
            requested_level=AutonomyLevel(int(spec.get("level", 1))),
        )

    return Rule(decl.get("name", spec["action"]), match)


def _make_trigger(decl: dict, metrics: dict[str, Callable[[], float]]):
    t, name, p = decl["type"], decl.get("name", decl["type"]), decl.get("params", {})
    if t == "interval":
        return IntervalTrigger(name, p["interval_sec"])
    if t == "schedule":
        return ScheduleTrigger(name, p["at"], p.get("payload"))
    if t == "filewatch":
        import os
        return FileWatchTrigger(name, os.path.expanduser(p["path"]))
    if t == "threshold":
        metric = metrics.get(p["metric"])
        if metric is None:
            raise ValueError(f"threshold 觸發器需要 metrics registry 提供 '{p['metric']}'")
        return ThresholdTrigger(name, metric, p["threshold"],
                                p.get("direction", "above"),
                                p.get("check_interval", 1.0))
    raise ValueError(f"未知觸發器類型：{t}")


def _make_action(decl: dict, handlers: dict[str, Callable]):
    t = decl["type"]
    if t == "console":
        return ConsoleNotifier()
    if t == "webhook":
        return WebhookNotifier(decl["url"])
    if t == "callback":
        fn = handlers.get(decl["handler"])
        if fn is None:
            raise ValueError(f"callback 動作需要 handlers registry 提供 '{decl['handler']}'")
        return CallbackAction(fn)
    raise ValueError(f"未知動作類型：{t}")


def load_runtime(
    path: str,
    handlers: Optional[dict[str, Callable]] = None,
    metrics: Optional[dict[str, Callable[[], float]]] = None,
    confirm_handler: Optional[Callable] = None,
    memory_path: str = "pai_memory.db",
    prefer_local_weights: bool = True,
) -> PAIAgent:
    """讀取 .pai 檔 → 可直接 run() 的 PAIAgent。"""
    handlers, metrics = handlers or {}, metrics or {}
    r = PaiReader(path)
    manifest = r.manifest
    logger.info("Loading agent '%s' v%s", manifest.get("name"), manifest.get("version"))

    # 政策
    pj = r.read_json("policy.json")
    policy = ProactivityPolicy(
        min_confidence=pj.get("min_confidence", 0.4),
        act_confidence=pj.get("act_confidence", 0.85),
        default_max_level=AutonomyLevel(pj.get("default_max_level", 1)),
        action_max_levels={k: AutonomyLevel(v)
                           for k, v in pj.get("action_max_levels", {}).items()},
        quiet_hours=tuple(pj["quiet_hours"]) if pj.get("quiet_hours") else None,
        max_interruptions_per_hour=pj.get("max_interruptions_per_hour", 6),
    )

    # 動作
    actions = {name: _make_action(decl, handlers)
               for name, decl in r.read_json("actions.json").items()}

    # 決策腦：規則永遠先建好（當 fallback）
    rule_brain = RuleBrain([_make_rule(d) for d in r.read_json("rules.json")])
    brain = rule_brain
    if "brain.json" in r.section_names:
        bj = r.read_json("brain.json")
        if bj.get("type") == "llm":
            avail = list(actions)
            if prefer_local_weights and "weights.gguf" in r.section_names:
                # 串流抽取到快取（不會把整個權重讀進 RAM）
                weights_path = r.extract_to_cache("weights.gguf")
                engine = bj.get("engine", "llama-server")
                if engine == "llama-server":
                    from .server_brain import LlamaServerBrain
                    brain = LlamaServerBrain(
                        avail, weights_path=weights_path,
                        base_url=bj.get("base_url"),
                        port=bj.get("port", 8089),
                        n_ctx=bj.get("n_ctx", 4096),
                        fallback=rule_brain,
                        user_profile=bj.get("user_profile", ""))
                else:  # engine == "llama-cpp-python"
                    from .local_brain import LocalLLMBrain
                    brain = LocalLLMBrain(
                        avail, weights_path=weights_path,
                        n_ctx=bj.get("n_ctx", 8192),
                        fallback=rule_brain, user_profile=bj.get("user_profile", ""))
            else:
                brain = LLMBrain(avail, model=bj.get("model", "claude-sonnet-4-6"),
                                 fallback=rule_brain,
                                 user_profile=bj.get("user_profile", ""))

    # 記憶式即時學習（self-finetuning 第一層）：brain.json 設 learning=true 即啟用
    reflective = None
    bj = r.read_json("brain.json") if "brain.json" in r.section_names else {}
    if bj.get("learning"):
        from .learning import EmbeddingClient, HashingEmbedder, ReflectiveMemory
        embedder = HashingEmbedder()
        # 若決策腦自帶本地 server，可共用它的 /v1/embeddings 做語義檢索
        base_url = getattr(brain, "base_url", None)
        if base_url:
            embedder = EmbeddingClient(base_url, fallback=HashingEmbedder())
        reflective = ReflectiveMemory(memory_path, embedder=embedder)

    agent = PAIAgent(
        name=manifest.get("name", "pai-agent"),
        brain=brain, policy=policy, memory=Memory(memory_path),
        actions=actions, confirm_handler=confirm_handler,
        reflective=reflective,
    )
    for decl in r.read_json("triggers.json"):
        try:
            agent.add_trigger(_make_trigger(decl, metrics))
        except ValueError as e:
            logger.warning("略過觸發器 %s：%s", decl.get("name"), e)
    return agent
