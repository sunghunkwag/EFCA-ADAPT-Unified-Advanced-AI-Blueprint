from src.orchestrator import Orchestrator

def test_smoke():
    out = Orchestrator().run_once()
    assert "action" in out and "metrics" in out
    assert "safety_ok" in out and isinstance(out["safety_ok"], bool)