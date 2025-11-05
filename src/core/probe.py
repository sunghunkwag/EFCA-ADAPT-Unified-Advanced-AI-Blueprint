from __future__ import annotations
import numpy as np
from typing import Dict

class Probe:
    def evaluate(self, h_t: np.ndarray, errors: np.ndarray) -> Dict[str, float]:
        if h_t.ndim != 1 or errors.ndim != 1:
            raise ValueError("h_t and errors must be 1D")
        unc = float(np.clip(errors.mean(), 0.0, 1.0))
        comp = float(np.clip(1.0 - unc, 0.0, 1.0))
        eff = float(np.clip(h_t.std() / (abs(h_t.mean()) + 1e-6), 0.0, 1.0))
        return {"uncertainty": unc, "comprehension": comp, "efficiency": eff}