from __future__ import annotations
import numpy as np

class MemoryMesh:
    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = int(capacity)
        self.buf = []

    def maybe_store(self, z: np.ndarray, metrics: dict) -> None:
        if len(self.buf) < self.capacity:
            self.buf.append((float(z.sum()), float(metrics.get("uncertainty", 0.0))))