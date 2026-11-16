# tests/test_planner.py
import sys
import time
from pathlib import Path
import json

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.planner import create_plan

def test_planner(provider="gemini"):
    print("=" * 60)
    print(f"Testing Planner with provider: [{provider.upper()}]")
    print("=" * 60)
    
    start_time = time.time()
    goal = "Search weather in Tokyo and save to desktop"
    print(f"Goal: {goal}")
    
    try:
        plan = create_plan(goal, provider=provider)
        elapsed = time.time() - start_time
        print(f"\n[Success] Plan generated in {elapsed:.2f} seconds!")
        print("\nGenerated Plan:")
        print(json.dumps(plan, indent=2))
    except Exception as e:
        print(f"\n[Error] Planner failed with: {e}")

if __name__ == "__main__":
    # Test with OpenRouter first (fast & reliable)
    test_planner(provider="openrouter")
    
    print("\n" + "-" * 60 + "\n")
    
    # Test with Gemini
    test_planner(provider="gemini")
