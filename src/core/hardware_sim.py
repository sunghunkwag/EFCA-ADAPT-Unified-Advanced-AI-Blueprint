from __future__ import annotations
import numpy as np
from typing import List, Dict

class EventCameraSimulator:
    def __init__(self, width: int = 128, height: int = 128, n: int = 8, seed: int = 7) -> None:
        self.width = int(width); self.height = int(height); self.n = int(n)
        self.rng = np.random.default_rng(seed)

    def get_events(self) -> List[Dict]:
        events = []
        for _ in range(self.n):
            frame = self.rng.integers(0, 2, size=(self.height, self.width), dtype=np.int8)
            ts = float(self.rng.random())
            events.append({"frame": frame, "ts": ts})
        return events