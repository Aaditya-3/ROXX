"""
Persistent JSON store for structured brain memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from backend.app.brain.models import StructuredEvent, StructuredPreference, UserMemoryState, utc_iso_now


class BrainMemoryStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: dict[str, UserMemoryState] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return
            for user_id, raw in payload.items():
                if isinstance(raw, dict):
                    state = UserMemoryState.from_dict(raw)
                    state.user_id = str(user_id)
                    self._records[state.user_id] = state
        except Exception:
            return

    def _save(self):
        data = {uid: state.to_dict() for uid, state in self._records.items()}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_user_state(self, user_id: str) -> UserMemoryState:
        uid = (user_id or "").strip() or "guest"
        with self._lock:
            if uid not in self._records:
                self._records[uid] = UserMemoryState(user_id=uid)
                self._save()
            return UserMemoryState.from_dict(self._records[uid].to_dict())

    def _update_state(self, state: UserMemoryState):
        state.updated_at = utc_iso_now()
        self._records[state.user_id] = state
        self._save()

    def upsert_event(self, user_id: str, event: StructuredEvent):
        uid = (user_id or "").strip() or "guest"
        with self._lock:
            state = self._records.get(uid) or UserMemoryState(user_id=uid)
            replaced = False
            event_name = event.name.lower().strip()
            for idx, existing in enumerate(state.events):
                if existing.name.lower().strip() == event_name:
                    state.events[idx] = event
                    replaced = True
                    break
            if not replaced:
                state.events.append(event)
            self._update_state(state)

    def upsert_preference(self, user_id: str, pref: StructuredPreference):
        uid = (user_id or "").strip() or "guest"
        with self._lock:
            state = self._records.get(uid) or UserMemoryState(user_id=uid)
            replaced = False
            for idx, existing in enumerate(state.profile.preferences):
                if existing.key == pref.key and existing.category == pref.category:
                    state.profile.preferences[idx] = pref
                    replaced = True
                    break
            if not replaced:
                state.profile.preferences.append(pref)

            value_text = str(pref.value).strip().lower()
            if pref.key.startswith("like") and value_text and value_text not in {"true", "false"}:
                if value_text not in [x.lower() for x in state.profile.likes]:
                    state.profile.likes.append(str(pref.value))
            if pref.key.startswith("dislike") and value_text:
                if value_text not in [x.lower() for x in state.profile.dislikes]:
                    state.profile.dislikes.append(str(pref.value))

            self._update_state(state)

    def set_profile_list(self, user_id: str, key: str, value: str):
        uid = (user_id or "").strip() or "guest"
        clean = str(value or "").strip()
        if not clean:
            return
        with self._lock:
            state = self._records.get(uid) or UserMemoryState(user_id=uid)
            target: list[str] = []
            if key == "likes":
                target = state.profile.likes
            elif key == "dislikes":
                target = state.profile.dislikes
            elif key == "traits":
                target = state.profile.traits
            if target and clean.lower() in [x.lower() for x in target]:
                self._update_state(state)
                return
            target.append(clean)
            self._update_state(state)

    def export_user_context(self, user_id: str) -> dict[str, Any]:
        state = self.get_user_state(user_id)
        return state.to_dict()


_brain_store: BrainMemoryStore | None = None


def get_brain_store(path: Path) -> BrainMemoryStore:
    global _brain_store
    if _brain_store is None:
        _brain_store = BrainMemoryStore(path=path)
    return _brain_store
