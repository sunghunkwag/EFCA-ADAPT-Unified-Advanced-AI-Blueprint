from __future__ import annotations
import numpy as np

class CTLNN:
    def __init__(self, hidden_dim: int = 64, dt: float = 0.05, seed: int = 123) -> None:
        self.hidden_dim = int(hidden_dim)
        self.dt = float(dt)
        self.h = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def step(self, z: np.ndarray, s_gwt: np.ndarray, last_action: int | None) -> np.ndarray:
        if z.ndim != 2:
            raise ValueError("z must be [N, D]")
        if s_gwt.ndim != 1:
            raise ValueError("s_gwt must be [D]")
        # Simple ODE-like update: dh/dt = -h + U(z_mean, s_gwt)
        u = z.mean(axis=0).mean().astype(np.float32) if z.size else np.float32(0.0)
        s = s_gwt.mean().astype(np.float32)
        inp = float(u + s + (last_action or 0))
        self.h = (1.0 - self.dt) * self.h + self.dt * inp
        return self.h.copy()