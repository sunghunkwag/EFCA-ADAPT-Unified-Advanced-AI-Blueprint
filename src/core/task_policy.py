from __future__ import annotations
import numpy as np
from typing import Dict, Any

class TaskPolicy:
    def __init__(self, n_actions: int = 4) -> None:
        self.n_actions = int(n_actions)

    def select_action(self, h_t: np.ndarray, s_gwt: np.ndarray) -> Dict[str, Any]:
        if h_t.ndim != 1 or s_gwt.ndim != 1:
            raise ValueError("h_t and s_gwt must be 1D")
        # Greedy stub on sign of sum
        score = float(h_t.sum() + s_gwt.sum())
        a_id = 0 if score >= 0 else 1
        return {"type": "discrete", "id": a_id, "score": score}