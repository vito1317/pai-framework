"""等 Gemma-4 GGUF 下載完成 → 打包成 .pai → 實跑本地推理。全程自動，結果寫 log。"""
import os
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
GGUF = os.path.join(BASE, "models", "gemma-4-26B_q4_0-it.gguf")
PAI = os.path.join(BASE, "gemma-guardian.pai")
EXPECTED = 14439361440


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_download():
    log("等待 GGUF 下載完成…")
    stable = 0
    last = -1
    while True:
        size = os.path.getsize(GGUF) if os.path.exists(GGUF) else 0
        if size >= EXPECTED:
            log(f"下載完成：{size:,} bytes")
            return
        stable = stable + 1 if size == last else 0
        last = size
        if stable > 60:  # 連續 10 分鐘沒成長 → 視為卡住
            log(f"下載疑似中斷，目前 {size:,}/{EXPECTED:,}")
            sys.exit(1)
        time.sleep(10)


def pack():
    from pai import pack_agent, PaiReader
    log("打包成 .pai（串流，記憶體占用恆定）…")
    pack_agent(
        PAI,
        manifest={"name": "gemma-guardian", "version": "1.0.0",
                  "author": "vito1317 <service@vito1317.com>",
                  "description": "內嵌 Gemma-4-26B-A4B-it (QAT q4_0) 的離線 PAI agent",
                  "model": "google/gemma-4-26B-A4B-it-qat-q4_0-gguf"},
        policy={"min_confidence": 0.4, "act_confidence": 0.85, "default_max_level": 1,
                "action_max_levels": {"cleanup": 2, "archive_file": 3}},
        triggers=[{"type": "threshold", "name": "cpu-monitor",
                   "params": {"metric": "cpu", "threshold": 75}}],
        rules=[{"when": {"kind": "metric.breach"},
                "intent": {"action": "cleanup", "confidence": 0.9, "urgency": 0.95,
                           "level": 2, "rationale": "CPU 超標"}}],
        actions={"cleanup": {"type": "callback", "handler": "ops.cleanup"},
                 "__notify__": {"type": "console"}},
        brain={"type": "llm", "model": "local", "n_ctx": 4096,
               "user_profile": "SRE 工程師，討厭被打擾"},
        weights_path=GGUF,
    )
    r = PaiReader(PAI)
    log(f"打包完成：{os.path.getsize(PAI):,} bytes，段：{r.section_names}")


def infer():
    from pai.paifile import PaiReader
    from pai.local_brain import LocalLLMBrain
    from pai.core import Event

    log("從 .pai 抽取權重並載入 Gemma-4（llama.cpp / Metal）…")
    r = PaiReader(PAI)
    wpath = r.extract_to_cache("weights.gguf")
    log(f"權重就緒（SHA-256 已校驗）：{wpath}")

    brain = LocalLLMBrain(available_actions=["cleanup", "__notify__"],
                          weights_path=wpath, n_ctx=4096, n_gpu_layers=-1)
    t0 = time.time()
    event = Event(source="cpu-monitor", kind="metric.breach",
                  payload={"value": 93, "threshold": 75, "direction": "above"})
    intents = brain.decide(event, {"user_state": "focus_mode_coding"})
    log(f"推理完成，耗時 {time.time()-t0:.1f}s")
    log(f"Gemma-4 產出 {len(intents)} 個意圖：")
    for it in intents:
        log(f"  → action={it.action} conf={it.confidence} urg={it.urgency} "
            f"level={int(it.requested_level)} | {it.rationale}")
    # 直接問一句驗證模型本體
    out = brain._llm.create_chat_completion(
        messages=[{"role": "user", "content": "用繁體中文一句話自我介紹你是哪個模型。"}],
        max_tokens=128, temperature=0.3)
    log("模型自述：" + out["choices"][0]["message"]["content"].strip())


if __name__ == "__main__":
    wait_download()
    pack()
    infer()
    log("✅ 全部完成")
