"""
core/prompt.py - Centralized Prompt Engineering with LangChain ChatPromptTemplate.

This module upgrades static text prompts to dynamic, context-aware LangChain prompt
templates. It injects dynamic runtime variables (system time, operating system,
and persistent memory facts) cleanly into the system prompt.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from config import settings
from memory.memory_manager import load_memory, format_memory_for_prompt

PROMPT_FILE = settings.core_dir / "prompt.txt"

DEFAULT_FALLBACK_PROMPT = (
    "You are JARVIS, a sharp and efficient AI assistant. Calm, direct, professional. "
    "Always use the correct tool. Never simulate or guess results. "
    "Keep responses concise (1-2 sentences max unless reporting data)."
)


def _load_base_instructions() -> str:
    """Loads the core persona guidelines from prompt.txt with fallback."""
    try:
        if PROMPT_FILE.exists():
            return PROMPT_FILE.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[Prompt] Could not read {PROMPT_FILE}: {e}")
    return DEFAULT_FALLBACK_PROMPT


def get_current_time_str() -> str:
    """Returns dynamic localized date and time string."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


def get_system_context_str(memory_dict: Optional[dict] = None) -> str:
    """
    Compiles dynamic runtime context including OS, timestamp, and active memory facts.
    """
    time_str = get_current_time_str()
    os_name = settings.os_system.title()

    if memory_dict is None:
        try:
            memory_dict = load_memory()
        except Exception:
            memory_dict = {}

    memory_str = format_memory_for_prompt(memory_dict)
    memory_section = f"\nUSER CONTEXT & MEMORY:\n{memory_str}\n" if memory_str else ""

    return (
        f"\nENVIRONMENT & SYSTEM CONTEXT:\n"
        f"- Operating System: {os_name}\n"
        f"- Current Date & Time: {time_str}\n"
        f"{memory_section}"
    )


def create_jarvis_prompt_template() -> ChatPromptTemplate:
    """
    Constructs a LangChain ChatPromptTemplate with dynamic runtime variables.

    Returns:
        ChatPromptTemplate suitable for LangChain agents and chains.
    """
    base_instructions = _load_base_instructions()

    template = (
        f"{base_instructions}\n\n"
        "{system_context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    return prompt


def get_rendered_system_prompt(memory_dict: Optional[dict] = None) -> str:
    """
    Generates the complete resolved system prompt string.
    Used for Gemini Live API and components requiring a raw system instruction string.
    """
    base_instructions = _load_base_instructions()
    system_context = get_system_context_str(memory_dict=memory_dict)
    return f"{base_instructions}\n\n{system_context}".strip()