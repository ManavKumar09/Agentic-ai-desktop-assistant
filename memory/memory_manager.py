import json
import re
from datetime import datetime
from threading import Lock
from pathlib import Path
import sys

from core.llm_factory import get_chat_model, extract_text
from langchain_core.messages import SystemMessage, HumanMessage

from pydantic import BaseModel, Field
from typing import Dict, Optional

class FactEntry(BaseModel):
    value: str = Field(description="The concise fact or preference value")

class ExtractedMemory(BaseModel):
    identity: Dict[str, FactEntry] = Field(
        default_factory=dict, 
        description="Personal facts like name, age, city, birthday, job, etc."
    )
    preferences: Dict[str, FactEntry] = Field(
        default_factory=dict, 
        description="Favorites or preferences: food, music, color, sports, etc."
    )
    projects: Dict[str, FactEntry] = Field(
        default_factory=dict, 
        description="Active projects, ongoing work, or goals in progress"
    )
    relationships: Dict[str, FactEntry] = Field(
        default_factory=dict, 
        description="People mentioned: friends, family, colleagues"
    )
    wishes: Dict[str, FactEntry] = Field(
        default_factory=dict, 
        description="Future plans, wishlist items, travel dreams"
    )
    notes: Dict[str, FactEntry] = Field(
        default_factory=dict, 
        description="Habits, routines, schedules, or other memorable notes"
    )

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR         = get_base_dir()
MEMORY_PATH      = BASE_DIR / "memory" / "long_term.json"
_lock            = Lock()
MAX_VALUE_LENGTH = 380
MEMORY_MAX_CHARS = 2200


def _empty_memory() -> dict:
    return {
        "identity":      {},
        "preferences":   {},
        "projects":      {},
        "relationships": {},
        "wishes":        {},
        "notes":         {}
    }


def load_memory() -> dict:
    if not MEMORY_PATH.exists():
        return _empty_memory()

    with _lock:
        try:
            data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                base = _empty_memory()
                for key in base:
                    if key not in data:
                        data[key] = {}
                return data
            return _empty_memory()
        except Exception as e:
            print(f"[Memory] ⚠️ Load error: {e}")
            return _empty_memory()


def _all_entries(memory: dict) -> list[tuple]:
    entries = []
    for cat, items in memory.items():
        if not isinstance(items, dict):
            continue
        for key, entry in items.items():
            if isinstance(entry, dict) and "value" in entry:
                entries.append((cat, key, entry))
    return entries


def _trim_to_limit(memory: dict) -> dict:
    serialized = json.dumps(memory, ensure_ascii=False)
    if len(serialized) <= MEMORY_MAX_CHARS:
        return memory

    entries = _all_entries(memory)
    entries.sort(key=lambda t: t[2].get("updated", "0000-00-00"))

    for cat, key, _ in entries:
        if len(json.dumps(memory, ensure_ascii=False)) <= MEMORY_MAX_CHARS:
            break
        del memory[cat][key]
        print(f"[Memory] 🗑️  Trimmed {cat}/{key} (limit: {MEMORY_MAX_CHARS} chars)")

    return memory


def save_memory(memory: dict) -> None:
    if not isinstance(memory, dict):
        return

    memory = _trim_to_limit(memory)

    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        MEMORY_PATH.write_text(
            json.dumps(memory, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )


def _truncate_value(val: str) -> str:
    if isinstance(val, str) and len(val) > MAX_VALUE_LENGTH:
        return val[:MAX_VALUE_LENGTH].rstrip() + "…"
    return val


def _recursive_update(target: dict, updates: dict) -> bool:
    changed = False
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        if isinstance(value, dict) and "value" not in value:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
                changed = True
            if _recursive_update(target[key], value):
                changed = True
        else:
            if isinstance(value, dict) and "value" in value:
                new_val = _truncate_value(str(value["value"]))
            else:
                new_val = _truncate_value(str(value))

            entry    = {"value": new_val, "updated": datetime.now().strftime("%Y-%m-%d")}
            existing = target.get(key, {})
            if not isinstance(existing, dict) or existing.get("value") != new_val:
                target[key] = entry
                changed = True

    return changed


def update_memory(memory_update: dict) -> dict:
    if not isinstance(memory_update, dict) or not memory_update:
        return load_memory()

    memory = load_memory()
    if _recursive_update(memory, memory_update):
        save_memory(memory)
        print(f"[Memory] 💾 Saved: {list(memory_update.keys())}")
    return memory


def should_extract_memory(user_text: str, jarvis_text: str, api_key: str = "") -> bool:
    try:
        

        combined = f"User: {user_text[:300]}\nJarvis: {jarvis_text[:1000]}"

        model = get_chat_model(provider="openrouter", temperature=0.0, max_tokens=5)
        messages = [
            SystemMessage(content="You are a memory relevance checker. Reply only YES or NO."),
            HumanMessage(content=(
                f"Does this conversation contain ANY of the following?\n"
                f"- Personal facts (name, age, city, job, birthday, nationality)\n"
                f"- Preferences or favorites (food, color, music, sport, game, film, book, etc.)\n"
                f"- Active projects or goals the user is working on\n"
                f"- People in the user's life (friends, family, partner, colleagues)\n"
                f"- Things the user wants to do or buy in the future\n"
                f"- Any other fact worth remembering long-term\n\n"
                f"Reply only YES or NO.\n\nConversation:\n{combined}"
            ))
        ]
        result = extract_text(model.invoke(messages)).strip()
        return "YES" in result.upper()

    except Exception as e:
        print(f"[Memory] ⚠️ Stage1 check failed: {e}")
        return False


def extract_memory(user_text: str, jarvis_text: str, api_key: str = "") -> dict:
    """
    Extracts structured memory facts using LangChain's with_structured_output.
    Eliminates regex stripping and manual json.loads parsing.
    """
    try:
        combined = f"User: {user_text[:600]}\nJarvis: {jarvis_text[:300]}"
        
        # 1. Initialize model via factory
        model = get_chat_model(provider="openrouter", temperature=0.1, max_tokens=1024)
        
        # 2. Bind the Pydantic schema for guaranteed structured output
        structured_llm = model.with_structured_output(ExtractedMemory)
        
        prompt = (
            "You are a memory extraction assistant. Extract ALL memorable personal facts, "
            "preferences, projects, relationships, and wishes from the conversation below.\n"
            "If nothing memorable is mentioned, return empty dictionaries.\n\n"
            f"Conversation:\n{combined}"
        )
        
        # 3. Invoke structured LLM directly
        result: ExtractedMemory = structured_llm.invoke(prompt)
        
        # 4. Convert Pydantic object to dict for saving (format matches existing JSON structure)
        data = result.model_dump()
        
        # Clean out empty categories
        return {cat: items for cat, items in data.items() if items}

    except Exception as e:
        if "429" not in str(e):
            print(f"[Memory] ⚠️ Structured extract failed: {e}")
        return {}


def format_memory_for_prompt(memory: dict | None) -> str:
    if not memory:
        return ""

    lines = []

    identity  = memory.get("identity", {})
    id_fields = ["name", "age", "birthday", "city", "job", "language", "school", "nationality"]
    for field in id_fields:
        entry = identity.get(field)
        if entry:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"{field.title()}: {val}")
    for key, entry in identity.items():
        if key in id_fields:
            continue
        val = entry.get("value") if isinstance(entry, dict) else entry
        if val:
            lines.append(f"{key.replace('_', ' ').title()}: {val}")

    prefs = memory.get("preferences", {})
    if prefs:
        lines.append("")
        lines.append("Preferences:")
        for key, entry in list(prefs.items())[:15]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    projects = memory.get("projects", {})
    if projects:
        lines.append("")
        lines.append("Active Projects / Goals:")
        for key, entry in list(projects.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    rels = memory.get("relationships", {})
    if rels:
        lines.append("")
        lines.append("People in their life:")
        for key, entry in list(rels.items())[:10]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    wishes = memory.get("wishes", {})
    if wishes:
        lines.append("")
        lines.append("Wishes / Plans / Wants:")
        for key, entry in list(wishes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key.replace('_', ' ').title()}: {val}")

    notes = memory.get("notes", {})
    if notes:
        lines.append("")
        lines.append("Other notes:")
        for key, entry in list(notes.items())[:8]:
            val = entry.get("value") if isinstance(entry, dict) else entry
            if val:
                lines.append(f"  - {key}: {val}")

    if not lines:
        return ""

    header = "[WHAT YOU KNOW ABOUT THIS PERSON — use naturally, never recite like a list]\n"
    result = header + "\n".join(lines)
    if len(result) > 2000:
        result = result[:1997] + "…"

    return result + "\n"


def remember(key: str, value: str, category: str = "notes") -> str:
    valid = {"identity", "preferences", "projects", "relationships", "wishes", "notes"}
    if category not in valid:
        category = "notes"
    update_memory({category: {key: {"value": value}}})
    return f"Remembered: {category}/{key} = {value}"


def forget(key: str, category: str = "notes") -> str:
    memory = load_memory()
    cat    = memory.get(category, {})
    if key in cat:
        del cat[key]
        memory[category] = cat
        save_memory(memory)
        return f"Forgotten: {category}/{key}"
    return f"Not found: {category}/{key}"

forget_memory = forget