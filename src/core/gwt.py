from __future__ import annotations
import numpy as np

class GWT:
    def __init__(self, state_dim: int = 32) -> None:
        self.state_dim = int(state_dim)
        self.state = np.zeros((self.state_dim,), dtype=np.float32)

    def update(self, tokens: np.ndarray) -> np.ndarray:
        if tokens.ndim != 1:
            raise ValueError("tokens must be [N]")
        # Deterministic aggregation
        agg = np.array([tokens.mean(), tokens.std()], dtype=np.float32)
        # Project to state
        self.state[:2] = agg
        return self.state.copy()