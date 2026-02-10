"""
Memory Store

Persistent multi-user memory storage with confidence management.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .memory_schema import Memory


class MemoryStore:
    """Persistent memory store (multi-user via user_id)."""

    def __init__(self):
        self._memories: dict[str, Memory] = {}
        self._bootstrapped_users: set[str] = set()
        self._storage_path = Path(__file__).resolve().parent / "memories.json"
        self._load()

    def _load(self):
        if not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
            dirty = False
            for item in data:
                if not isinstance(item, dict):
                    dirty = True
                    continue
                try:
                    memory = Memory.from_dict(item)
                    normalized = self._normalize_memory(memory)
                    if normalized is None:
                        dirty = True
                        continue
                    if normalized.key != memory.key or normalized.value != memory.value:
                        dirty = True
                    self._memories[normalized.id] = normalized
                except Exception:
                    dirty = True
                    continue
            if dirty:
                self._save()
        except Exception:
            pass

    def _save(self):
        try:
            payload = [m.to_dict() for m in self._memories.values()]
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def add_or_update_memory(self, memory: Memory) -> Optional[Memory]:
        """
        Add new memory or update existing one.

        Rules:
        - New memory starts at 0.7
        - Same key+value confirms (+0.1)
        - Same key + different value overwrites and resets to 0.7
        """
        memory = self._normalize_memory(memory)
        if memory is None:
            return None

        existing = self._find_by_key(memory.key, memory.user_id)
        if existing:
            if existing.value.lower() == memory.value.lower():
                existing.confidence = min(1.0, existing.confidence + 0.1)
            else:
                existing.value = memory.value
                existing.confidence = 0.7
            existing.last_updated = datetime.now()
            self._save()
            return existing

        self._memories[memory.id] = memory
        self._save()
        return memory

    def get_all_memories(self) -> List[Memory]:
        return list(self._memories.values())

    def get_user_memories(self, user_id: str) -> List[Memory]:
        return [m for m in self._memories.values() if m.user_id == user_id]

    def get_memories_by_confidence(self, user_id: str, min_confidence: float = 0.5) -> List[Memory]:
        return [
            m for m in self._memories.values()
            if m.user_id == user_id and m.confidence >= min_confidence
        ]

    def delete_memory(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._save()
            return True
        return False

    def apply_strong_feedback(self, user_id: str, user_message: str) -> int:
        """
        Apply strong sentiment updates to matching memories.
        - Strong criticism: confidence -= 0.7
        - Strong appreciation: confidence += 0.7
        Returns number of affected memories.
        """
        msg = (user_message or "").lower()
        if not msg:
            return 0

        negative_targets = self._extract_feedback_targets(
            msg,
            patterns=[
                r"\bi\s+(?:really\s+)?hate\s+([^,.!?]+)",
                r"\bi\s+strongly\s+dislike\s+([^,.!?]+)",
                r"\bi\s+absolutely\s+dislike\s+([^,.!?]+)",
                r"\bi\s+can't\s+stand\s+([^,.!?]+)",
            ],
        )
        positive_targets = self._extract_feedback_targets(
            msg,
            patterns=[
                r"\bi\s+(?:really\s+)?love\s+([^,.!?]+)",
                r"\bi\s+absolutely\s+love\s+([^,.!?]+)",
                r"\bi\s+strongly\s+like\s+([^,.!?]+)",
                r"\bi\s+adore\s+([^,.!?]+)",
            ],
        )

        adjusted = 0
        user_memories = self.get_user_memories(user_id)
        for memory in user_memories:
            mem_text = f"{memory.key} {memory.value}".lower()
            if any(t and (t in mem_text or mem_text in t) for t in negative_targets):
                memory.confidence = max(0.0, memory.confidence - 0.7)
                memory.last_updated = datetime.now()
                adjusted += 1
            elif any(t and (t in mem_text or mem_text in t) for t in positive_targets):
                memory.confidence = min(1.0, memory.confidence + 0.7)
                memory.last_updated = datetime.now()
                adjusted += 1

        if adjusted:
            self._save()
        return adjusted

    def _extract_feedback_targets(self, msg: str, patterns: List[str]) -> List[str]:
        targets: List[str] = []
        for pattern in patterns:
            for match in re.finditer(pattern, msg):
                target = (match.group(1) or "").strip().lower()
                target = re.sub(r"^(?:eating|watching|using)\s+", "", target).strip()
                if len(target) >= 2:
                    targets.append(target)
        return targets

    def _find_by_key(self, key: str, user_id: str) -> Optional[Memory]:
        for memory in self._memories.values():
            if memory.user_id == user_id and memory.key.lower() == key.lower():
                return memory
        return None

    def _normalize_memory(self, memory: Memory) -> Optional[Memory]:
        key = (memory.key or "").strip().lower()
        value = (memory.value or "").strip()
        value_l = value.lower()

        drop_keys = {
            "contextual_conversation_topic",
            "has_not_shared_favourite_food",
            "has_not_shared_favorite_food",
            "study_location",
        }
        drop_values = {
            "somewhere to be determined",
            "in year",
            "the city where my college is",
            "true",
            "false",
            "unknown",
            "n/a",
        }
        invalid_name_values = {
            "pursuing",
            "working",
            "studying",
            "trying",
            "doing",
            "going",
            "learning",
        }

        if not key or not value:
            return None
        if key in drop_keys:
            return None
        if value_l in drop_values:
            return None
        if key == "name" and value_l in invalid_name_values:
            return None

        # Migrate generic food preferences into specific key.
        food_terms = {"pizza", "burger", "biryani", "pasta", "noodles", "rice", "sushi"}
        if key == "preference":
            if re.search(r"\beat(ing)?\b", value_l) or any(t in value_l for t in food_terms):
                key = "favorite_food"
                value = re.sub(r"^\s*eating\s+", "", value, flags=re.IGNORECASE).strip()
                value_l = value.lower()
                if not value:
                    return None

        # Normalize interest aliases.
        if key in {"interest", "interests"} and "aot" in value_l:
            key = "anime_preference"

        memory.key = key
        memory.value = value
        return memory

    def ensure_bootstrap_memories(self, user_id: str):
        user_key = user_id.strip().lower()
        if user_key in self._bootstrapped_users:
            return

        # One-time migration path: if legacy memories had no user_id they were loaded as "guest".
        # If this user has no memories yet, copy guest memories so old profile data is not lost.
        user_memories = self.get_user_memories(user_key)
        guest_memories = self.get_user_memories("guest")
        if not user_memories and guest_memories and user_key != "guest":
            for gm in guest_memories:
                self.add_or_update_memory(
                    Memory.create(
                        user_id=user_key,
                        type=gm.type,
                        key=gm.key,
                        value=gm.value,
                        confidence=gm.confidence,
                    )
                )

        if user_key == "aaditya":
            self.add_or_update_memory(
                Memory.create(
                    user_id="aaditya",
                    type="fact",
                    key="name",
                    value="Aaditya",
                    confidence=0.9,
                )
            )
            self.add_or_update_memory(
                Memory.create(
                    user_id="aaditya",
                    type="preference",
                    key="anime_preference",
                    value="AOT",
                    confidence=0.9,
                )
            )

        self._bootstrapped_users.add(user_key)

    def clear(self):
        self._memories.clear()
        self._save()


_memory_store = MemoryStore()


def get_memory_store() -> MemoryStore:
    return _memory_store
