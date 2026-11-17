"""
tests/test_prompt.py - Test suite for LangChain prompt modernization.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.prompt import (
    get_current_time_str,
    get_system_context_str,
    get_rendered_system_prompt,
    create_jarvis_prompt_template,
)


def test_prompt_generation():
    print("[Test] Testing dynamic prompt rendering...")

    # 1. Test dynamic timestamp
    time_str = get_current_time_str()
    assert len(time_str) > 0, "Current time string should not be empty"
    print(f"  Dynamic time: {time_str}")

    # 2. Test system context injection with dummy memory
    dummy_memory = {
        "identity": {"name": {"value": "Tony Stark"}},
        "preferences": {"favorite_food": {"value": "Cheeseburger"}}
    }
    context_str = get_system_context_str(dummy_memory)
    assert "Operating System" in context_str, "Should include Operating System"
    assert "Tony Stark" in context_str, "Should include memory name"
    print("  Context string contains OS and memory entries")

    # 3. Test rendered full system prompt
    full_prompt = get_rendered_system_prompt(dummy_memory)
    assert "JARVIS" in full_prompt, "System prompt should mention JARVIS persona"
    assert "Tony Stark" in full_prompt, "Rendered prompt should include user memory"
    print(f"  Full rendered prompt generated ({len(full_prompt)} chars)")

    # 4. Test LangChain ChatPromptTemplate formatting
    template = create_jarvis_prompt_template()
    formatted_messages = template.format_messages(
        system_context=context_str,
        input="What is my name?"
    )
    assert len(formatted_messages) == 2, "Should produce SystemMessage and HumanMessage"
    assert "Tony Stark" in formatted_messages[0].content
    assert formatted_messages[1].content == "What is my name?"
    print("  LangChain ChatPromptTemplate formatted successfully into message objects")

    print("[Test] All prompt tests passed successfully!")


if __name__ == "__main__":
    test_prompt_generation()