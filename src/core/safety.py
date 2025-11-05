from __future__ import annotations
from typing import Tuple, List, Dict

class Safety:
    def evaluate(self, action: Dict, context: Dict) -> Tuple[bool, List[str]]:
        violations: List[str] = []
        if action.get("type") != "discrete":
            violations.append("non_discrete_action")
        ok = len(violations) == 0
        return ok, violations