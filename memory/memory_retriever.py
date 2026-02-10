"""
Memory Retriever

Retrieves relevant memories for the current conversation context.
"""

import re
from .memory_store import get_memory_store

ESSENTIAL_KEYS = {
    "name",
    "role",
    "occupation",
    "location",
    "favorite_food",
    "anime_preference",
    "language_preference",
    "tool_preference",
}


def retrieve_memories(user_message: str, user_id: str) -> str:
    """
    Retrieve relevant memories for the current user message.
    
    Returns a short summary string suitable for system prompt.
    Uses all memories with confidence >= 0.3 (new memories start at 0.7).
    """
    store = get_memory_store()
    store.ensure_bootstrap_memories(user_id)
    # Get user memories with confidence >= 0.3 (includes new memories at 0.7)
    memories = store.get_memories_by_confidence(user_id=user_id, min_confidence=0.3)
    
    if not memories:
        return ""
    
    # Rank by relevance to current message, then confidence and recency.
    msg_tokens = set(re.findall(r"[a-z0-9]+", user_message.lower()))

    def score(memory):
        key_tokens = set(re.findall(r"[a-z0-9]+", memory.key.lower()))
        value_tokens = set(re.findall(r"[a-z0-9]+", memory.value.lower()))
        overlap = len(msg_tokens.intersection(key_tokens.union(value_tokens)))
        identity_bonus = 1 if memory.key.lower() in {"name", "role"} else 0
        food_bonus = 0
        if {"food", "favourite", "favorite"}.intersection(msg_tokens):
            if memory.key.lower() == "favorite_food":
                food_bonus = 3
            elif memory.key.lower() == "preference":
                food_bonus = -1
        return (overlap + identity_bonus + food_bonus, memory.confidence, memory.last_updated.timestamp())

    essentials = [m for m in memories if m.key.lower() in ESSENTIAL_KEYS]
    essentials = sorted(essentials, key=lambda m: (m.confidence, m.last_updated.timestamp()), reverse=True)
    essential_by_key = {}
    for m in essentials:
        essential_by_key.setdefault(m.key.lower(), m)
    essential_selected = list(essential_by_key.values())

    memories_sorted = sorted(memories, key=score, reverse=True)
    selected = []
    seen_ids = set()
    for m in essential_selected:
        if m.id not in seen_ids:
            selected.append(m)
            seen_ids.add(m.id)
    for m in memories_sorted:
        if len(selected) >= 8:
            break
        if m.id in seen_ids:
            continue
        selected.append(m)
        seen_ids.add(m.id)

    # Build concise summary string
    parts = []
    for memory in selected:
        parts.append(f"- {memory.key}: {memory.value}")
    
    context = "\n".join(parts)
    return context
