"""
Memory Extractor

Strict extractor that stores only explicit, stable user facts/preferences.
"""

import json
import re
from typing import Optional

from backend.app.core.llm.groq_client import generate_response
from .memory_schema import Memory
from .memory_store import get_memory_store


NEVER_EXTRACT_VALUES = {
    "aria",
    "i understand",
    "i get it",
    "as an ai",
}

NEVER_EXTRACT_KEYS = {
    "assistant_name",
    "ai_name",
}

INVALID_NAME_WORDS = {
    "pursuing",
    "working",
    "studying",
    "trying",
    "doing",
    "going",
    "learning",
}


def _is_memory_supported_by_message(key: str, value: str, user_message: str) -> bool:
    msg = (user_message or "").lower()
    key_l = (key or "").strip().lower()
    value_l = (value or "").strip().lower()
    if not msg or not value_l:
        return False
    if key_l in NEVER_EXTRACT_KEYS:
        return False
    if value_l in NEVER_EXTRACT_VALUES:
        return False

    if value_l in msg:
        return True

    trusted_keys = {
        "name",
        "role",
        "occupation",
        "location",
        "anime_preference",
        "language_preference",
        "tool_preference",
        "preference",
    }
    if key_l in trusted_keys:
        tokens = [t for t in value_l.split() if len(t) > 2]
        if tokens and any(t in msg for t in tokens):
            return True

    return False


def _extract_patterns(user_message: str, user_id: str) -> list[Memory]:
    memories: list[Memory] = []
    message_lower = user_message.lower()

    # Keep "I'm" preamble flexible, but keep captured name strict.
    name_patterns = [
        r"i'?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"my\s+name\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"i\s+am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"call\s+me\s+([A-Z][a-z]+)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, user_message)
        if match:
            name = match.group(1).strip()
            if 1 < len(name) < 50 and name.lower() not in INVALID_NAME_WORDS:
                memories.append(Memory.create(user_id=user_id, type="fact", key="name", value=name, confidence=0.7))
                break

    # Food-specific preference capture.
    favorite_food_patterns = [
        r"my\s+favo(?:u)?rite\s+food\s+is\s+([^,.!?]+)",
        r"i\s+love\s+eating\s+([^,.!?]+)",
        r"i\s+like\s+eating\s+([^,.!?]+)",
    ]
    for pattern in favorite_food_patterns:
        match = re.search(pattern, message_lower)
        if match:
            food = match.group(1).strip()
            if 1 < len(food) < 80:
                memories.append(
                    Memory.create(
                        user_id=user_id,
                        type="preference",
                        key="favorite_food",
                        value=food,
                        confidence=0.7,
                    )
                )
                break

    preference_patterns = [
        r"i\s+prefer\s+([^,.!?]+)",
        r"i\s+like\s+([^,.!?]+)",
        r"my\s+favorite\s+([^,.!?]+)\s+is\s+([^,.!?]+)",
    ]
    for pattern in preference_patterns:
        match = re.search(pattern, message_lower)
        if not match:
            continue
        if len(match.groups()) == 2:
            key = match.group(1).strip()
            value = match.group(2).strip()
        else:
            key = "preference"
            value = match.group(1).strip()
        if value.startswith("eating "):
            # Already handled in favorite_food_patterns to avoid duplicate generic preference.
            continue
        if key in {"food", "favorite food", "favourite food"}:
            key = "favorite_food"
        if 2 < len(value) < 100:
            memories.append(Memory.create(user_id=user_id, type="preference", key=key, value=value, confidence=0.7))

    return memories


def extract_memory(user_message: str, user_id: str) -> Optional[Memory]:
    pattern_memories = _extract_patterns(user_message, user_id)
    stored_any = None
    if pattern_memories:
        for memory in pattern_memories:
            stored = get_memory_store().add_or_update_memory(memory)
            if stored:
                stored_any = stored
                print(f"Memory stored (pattern): {stored.key} = {stored.value}")
        # Pattern path is strict and explicit; avoid LLM extraction here.
        return stored_any

    prompt = """You are a skeptical auditor.

Source of Truth:
- ONLY extract facts explicitly stated by the USER.
- If the ASSISTANT suggested something (e.g., "Do you like coding?") and the user didn't say "Yes," you MUST NOT store it.
- If the user's message is ambiguous, do not guess. Return null.

Never extract:
- AI names (like "Aria")
- Polite filler ("I understand")
- Hypothetical examples used in the conversation

Output format:
- Return either null, or a JSON array.
- Each object must be: {"type":"fact|preference|constraint","key":"...","value":"..."}.
- Do not include anything else.

User message: """ + user_message

    try:
        response = generate_response(prompt)
        if not response:
            return None

        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        if response.lower() == "null" or not response:
            return None

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            return None

        payload = [data] if isinstance(data, dict) else data if isinstance(data, list) else None
        if payload is None:
            return None

        stored_any = None
        for item in payload:
            if not isinstance(item, dict):
                continue
            if not {"type", "key", "value"}.issubset(item.keys()):
                continue

            m_type = item.get("type")
            m_key = item.get("key")
            m_value = item.get("value")
            if m_type not in ["preference", "constraint", "fact"]:
                continue
            if not isinstance(m_key, str) or not isinstance(m_value, str):
                continue
            if m_key.strip().lower() in NEVER_EXTRACT_KEYS:
                continue
            if m_value.strip().lower() in NEVER_EXTRACT_VALUES:
                continue
            if not _is_memory_supported_by_message(m_key, m_value, user_message):
                continue

            memory = Memory.create(
                user_id=user_id,
                type=m_type,
                key=m_key.strip(),
                value=m_value.strip(),
                confidence=0.7,
            )
            stored = get_memory_store().add_or_update_memory(memory)
            if stored:
                stored_any = stored
                print(f"Memory stored: {stored.key} = {stored.value} (confidence: {stored.confidence:.2f})")

        return stored_any

    except RuntimeError as e:
        print(f"Memory extraction API error (non-fatal): {e}")
        return None
    except Exception as e:
        print(f"Memory extraction error (non-fatal): {type(e).__name__}: {e}")
        return None
