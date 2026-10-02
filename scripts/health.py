#!/usr/bin/env python3
"""Pre-flight: is every piece of the stack actually answering?"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fred import config, cosmos, anchors, vss, agent, trace

print(config.describe()); print()
checks = {}
ok, detail = cosmos.health();              checks["cosmos3_reason"] = {"ok": ok, "detail": detail[:160]}
ok, detail = anchors.canary_health();      checks["canary_1b"] = {"ok": ok, "detail": detail[:160]}
checks["vss_backend"] = vss.health()
models = agent.list_models()
checks["wandb_inference"] = {"ok": bool(models), "model_count": len(models),
                             "picked": agent.pick_model() if models else None}
checks["weave"] = trace.init()
print(json.dumps(checks, indent=2))
sys.exit(0 if any(c.get("ok") for c in checks.values()) else 1)
