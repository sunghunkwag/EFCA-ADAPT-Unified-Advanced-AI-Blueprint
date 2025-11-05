from src.orchestrator import Orchestrator

def main() -> None:
    result = Orchestrator().run_once()
    print("Run summary:", result)

if __name__ == "__main__":
    main()