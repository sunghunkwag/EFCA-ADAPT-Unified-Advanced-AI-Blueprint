from __future__ import annotations
from typing import Dict

class MetaController:
    def decide(self, metrics: Dict[str, float]) -> Dict[str, float]:
        eps = 0.1 if metrics["uncertainty"] < 0.5 else 0.3
        return {"epsilon": eps, "lambda_k": 0.5}