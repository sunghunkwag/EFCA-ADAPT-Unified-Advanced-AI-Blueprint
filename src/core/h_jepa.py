from __future__ import annotations
import numpy as np
from typing import Dict, Any

class HJepa:
    def __init__(self, latent_dim: int = 64, seed: int = 42) -> None:
        self.latent_dim = int(latent_dim)
        self.rng = np.random.default_rng(seed)

    def encode(self, x0: np.ndarray) -> Dict[str, Any]:
        if not isinstance(x0, np.ndarray):
            raise TypeError("x0 must be a numpy array")
        if x0.ndim < 2:
            raise ValueError("x0 must be at least 2D [N, ...]")
        n = x0.shape[0]
        z = self.rng.standard_normal((n, self.latent_dim)).astype(np.float32)
        errors = np.abs(self.rng.standard_normal((n,), dtype=np.float32))
        return {"z": z, "errors": errors}