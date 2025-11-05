from __future__ import annotations
import numpy as np

class Tokenizer:
    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = int(vocab_size)

    def encode(self, z: np.ndarray, errors: np.ndarray) -> np.ndarray:
        if z.ndim != 2:
            raise ValueError("z must be [N, D]")
        if errors.ndim != 1 or errors.shape[0] != z.shape[0]:
            raise ValueError("errors shape must be [N] matching z")
        # Simple hashing-based tokenization
        h = (np.abs(z).sum(axis=1) + errors).astype(np.float32)
        tokens = (h / (h.max() + 1e-6) * (self.vocab_size - 1)).astype(np.int32)
        tokens = np.clip(tokens, 0, self.vocab_size - 1)
        return tokens