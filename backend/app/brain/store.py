"""
Relational store for structured brain memory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from sqlalchemy import func

from backend.app.brain.models import StructuredEvent, StructuredPreference, UserMemoryState, utc_iso_now
from backend.app.core.db.relational import DBUserEvent, DBUserPreference, get_relational_session, init_relational_db


def _utc_now():
    return datetime.now(timezone.utc)


def _serialize_pref_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return str(value)


def _deserialize_pref_value(raw: str) -> Any:
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return raw


class BrainMemoryStore:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = Lock()
        init_relational_db()
        self._migrate_legacy_json()

    def _migrate_legacy_json(self):
        if self.path is None or not self.path.exists():
            return
        try:
            with get_relational_session() as db:
                has_existing = bool(db.query(DBUserPreference.id).first() or db.query(DBUserEvent.id).first())
            if has_existing:
                return
        except Exception:
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return
            for user_id, raw in payload.items():
                if not isinstance(raw, dict):
                    continue
                state = UserMemoryState.from_dict(raw)
                uid = str(user_id or state.user_id or "").strip() or "guest"
                for event in state.events:
                    self.upsert_event(uid, event)
                for pref in state.profile.preferences:
                    self.upsert_preference(uid, pref)
                for value in state.profile.likes:
                    self.set_profile_list(uid, "likes", value)
                for value in state.profile.dislikes:
                    self.set_profile_list(uid, "dislikes", value)
                for value in state.profile.traits:
                    self.set_profile_list(uid, "traits", value)
        except Exception:
            return

    def get_user_state(self, user_id: str) -> UserMemoryState:
        uid = (user_id or "").strip() or "guest"
        with self._lock, get_relational_session() as db:
            pref_rows = db.query(DBUserPreference).filter(DBUserPreference.user_id == uid).all()
            event_rows = db.query(DBUserEvent).filter(DBUserEvent.user_id == uid).all()

        preferences: list[StructuredPreference] = []
        likes: list[str] = []
        dislikes: list[str] = []
        traits: list[str] = []

        for row in pref_rows:
            category = str(row.category or "general")
            key = str(row.pref_key or "").strip()
            value = _deserialize_pref_value(str(row.pref_value or ""))
            pref = StructuredPreference(
                category=category,
                key=key,
                value=value,
                source_message=str(row.source_message or ""),
                updated_at=(row.updated_at or _utc_now()).isoformat(),
            )
            preferences.append(pref)

            if category == "profile":
                value_text = str(value).strip()
                if key == "likes" and value_text and value_text.lower() not in {x.lower() for x in likes}:
                    likes.append(value_text)
                elif key == "dislikes" and value_text and value_text.lower() not in {x.lower() for x in dislikes}:
                    dislikes.append(value_text)
                elif key == "traits" and value_text and value_text.lower() not in {x.lower() for x in traits}:
                    traits.append(value_text)

        events = [
            StructuredEvent(
                name=str(ev.name or "").strip(),
                date=str(ev.event_date.isoformat() if ev.event_date else ""),
                source_message=str(ev.source_message or ""),
                updated_at=(ev.updated_at or _utc_now()).isoformat(),
            )
            for ev in event_rows
        ]

        state = UserMemoryState(user_id=uid)
        state.profile.preferences = preferences
        state.profile.likes = likes
        state.profile.dislikes = dislikes
        state.profile.traits = traits
        state.events = events
        state.updated_at = utc_iso_now()
        return state

    def upsert_event(self, user_id: str, event: StructuredEvent):
        uid = (user_id or "").strip() or "guest"
        name = str(event.name or "").strip()
        if not name or not event.date:
            return
        try:
            event_date = datetime.fromisoformat(event.date).date()
        except Exception:
            return

        with self._lock, get_relational_session() as db:
            existing = (
                db.query(DBUserEvent)
                .filter(DBUserEvent.user_id == uid, func.lower(DBUserEvent.name) == name.lower())
                .first()
            )
            if existing:
                existing.event_date = event_date
                existing.source_message = event.source_message
                existing.updated_at = _utc_now()
                return
            db.add(
                DBUserEvent(
                    user_id=uid,
                    name=name,
                    event_date=event_date,
                    source_message=event.source_message,
                    created_at=_utc_now(),
                    updated_at=_utc_now(),
                )
            )

    def upsert_preference(self, user_id: str, pref: StructuredPreference):
        uid = (user_id or "").strip() or "guest"
        key = str(pref.key or "").strip()
        if not key:
            return
        category = str(pref.category or "general").strip() or "general"
        value = _serialize_pref_value(pref.value)

        with self._lock, get_relational_session() as db:
            existing = (
                db.query(DBUserPreference)
                .filter(
                    DBUserPreference.user_id == uid,
                    DBUserPreference.category == category,
                    DBUserPreference.pref_key == key,
                )
                .first()
            )
            if existing:
                existing.pref_value = value
                existing.source_message = pref.source_message
                existing.updated_at = _utc_now()
                return
            db.add(
                DBUserPreference(
                    user_id=uid,
                    category=category,
                    pref_key=key,
                    pref_value=value,
                    source_message=pref.source_message,
                    created_at=_utc_now(),
                    updated_at=_utc_now(),
                )
            )

    def set_profile_list(self, user_id: str, key: str, value: str):
        uid = (user_id or "").strip() or "guest"
        list_key = str(key or "").strip().lower()
        clean = str(value or "").strip()
        if list_key not in {"likes", "dislikes", "traits"} or not clean:
            return

        with self._lock, get_relational_session() as db:
            existing = (
                db.query(DBUserPreference)
                .filter(
                    DBUserPreference.user_id == uid,
                    DBUserPreference.category == "profile",
                    DBUserPreference.pref_key == list_key,
                    func.lower(DBUserPreference.pref_value) == clean.lower(),
                )
                .first()
            )
            if existing:
                existing.updated_at = _utc_now()
                return
            db.add(
                DBUserPreference(
                    user_id=uid,
                    category="profile",
                    pref_key=list_key,
                    pref_value=clean,
                    source_message=f"profile:{list_key}",
                    created_at=_utc_now(),
                    updated_at=_utc_now(),
                )
            )

    def export_user_context(self, user_id: str) -> dict[str, Any]:
        state = self.get_user_state(user_id)
        return state.to_dict()


_brain_store: BrainMemoryStore | None = None


def get_brain_store(path: Path) -> BrainMemoryStore:
    global _brain_store
    if _brain_store is None:
        _brain_store = BrainMemoryStore(path=path)
    return _brain_store
