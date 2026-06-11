"""PAI CLI：
    python3 -m pai info  <agent.pai>          顯示 .pai 檔內容
    python3 -m pai run   <agent.pai> [秒數]   載入並運行 agent
"""
import json
import logging
import sys


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("info", "run"):
        print(__doc__)
        sys.exit(1)
    cmd, path = sys.argv[1], sys.argv[2]

    if cmd == "info":
        from .paifile import PaiReader, load_agent
        r = PaiReader(path)
        print(json.dumps({**load_agent(path)["manifest"], **r.info()},
                         ensure_ascii=False, indent=2))
        return

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    duration = float(sys.argv[3]) if len(sys.argv) > 3 else None
    from .loader import load_runtime

    def confirm(intent):
        ans = input(f"\n❓ [PAI 請求確認] {intent.rationale} (y/N): ").strip().lower()
        return ans == "y"

    # CLI 模式下未注入的 handler/metric 一律以 stub 代替（不執行真動作）
    class _StubHandlers(dict):
        def get(self, key, default=None):
            if key not in self:
                self[key] = lambda intent, _k=key: print(
                    f"   ⚙️ [stub] handler '{_k}' 未注入，僅示意執行：{intent.params}")
            return self[key]

    class _StubMetrics(dict):
        def get(self, key, default=None):
            if key not in self:
                self[key] = lambda: 0.0
            return self[key]

    agent = load_runtime(path, handlers=_StubHandlers(),
                         metrics=_StubMetrics(), confirm_handler=confirm)
    agent.run(duration=duration)


if __name__ == "__main__":
    main()
