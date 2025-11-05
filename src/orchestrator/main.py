from __future__ import annotations
import json
import numpy as np
from typing import Dict, Any

from src.core import (
    HJepa, Tokenizer, GWT, CTLNN, TaskPolicy, Probe,
    MetaController, MemoryMesh, Safety, EventCameraSimulator
)

class Orchestrator:
    def __init__(self) -> None:
        self.sensor = EventCameraSimulator()
        self.h_jepa = HJepa()
        self.tokenizer = Tokenizer()
        self.gwt = GWT()
        self.ctlnn = CTLNN()
        self.policy = TaskPolicy()
        self.probe = Probe()
        self.meta = MetaController()
        self.mem = MemoryMesh()
        self.safety = Safety()
        self.last_action: int | None = None

    def run_once(self) -> Dict[str, Any]:
        events = self.sensor.get_events()
        x0 = np.stack([e["frame"] for e in events]).astype(np.float32)  # [N, H, W]
        enc = self.h_jepa.encode(x0)  # z:[N, D], errors:[N]
        z, errors = enc["z"], enc["errors"]
        tokens = self.tokenizer.encode(z, errors)  # [N]
        s_gwt = self.gwt.update(tokens)  # [D]
        h_t = self.ctlnn.step(z, s_gwt, self.last_action)  # [H]
        action = self.policy.select_action(h_t, s_gwt)
        metrics = self.probe.evaluate(h_t, errors)
        meta_params = self.meta.decide(metrics)
        self.mem.maybe_store(z, metrics)
        ok, violations = self.safety.evaluate(action, {"metrics": metrics})
        self.last_action = int(action["id"])
        return {
            "action": action, "metrics": metrics, "meta": meta_params,
            "safety_ok": ok, "violations": violations
        }

if __name__ == "__main__":
    out = Orchestrator().run_once()
    print(json.dumps(out, indent=2))