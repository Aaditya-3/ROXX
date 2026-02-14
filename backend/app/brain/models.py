"""
Structured memory and intent models for the brain layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Intent(str, Enum):
    GREETING = "greeting"
    MEMORY_UPDATE = "memory_update"
    MEMORY_QUERY = "memory_query"
    GUESS_REQUEST = "guess_request"
    FACTUAL_QUESTION = "factual_question"
    OTHER = "other"


@dataclass
class StructuredEvent:
    type: str = "event"
    name: str = ""
    date: str = ""
    source_message: str = ""
    updated_at: str = field(default_factory=utc_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "event",
            "name": self.name,
            "date": self.date,
            "source_message": self.source_message,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredEvent":
        return cls(
            type="event",
            name=str(data.get("name") or "").strip(),
            date=str(data.get("date") or "").strip(),
            source_message=str(data.get("source_message") or ""),
            updated_at=str(data.get("updated_at") or utc_iso_now()),
        )


@dataclass
class StructuredPreference:
    type: str = "preference"
    category: str = "general"
    key: str = ""
    value: Any = None
    source_message: str = ""
    updated_at: str = field(default_factory=utc_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "preference",
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "source_message": self.source_message,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredPreference":
        return cls(
            type="preference",
            category=str(data.get("category") or "general"),
            key=str(data.get("key") or "").strip(),
            value=data.get("value"),
            source_message=str(data.get("source_message") or ""),
            updated_at=str(data.get("updated_at") or utc_iso_now()),
        )


@dataclass
class UserProfile:
    preferences: list[StructuredPreference] = field(default_factory=list)
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    traits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferences": [p.to_dict() for p in self.preferences],
            "likes": list(self.likes),
            "dislikes": list(self.dislikes),
            "traits": list(self.traits),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        prefs = [StructuredPreference.from_dict(x) for x in (data.get("preferences") or []) if isinstance(x, dict)]
        likes = [str(x) for x in (data.get("likes") or []) if str(x).strip()]
        dislikes = [str(x) for x in (data.get("dislikes") or []) if str(x).strip()]
        traits = [str(x) for x in (data.get("traits") or []) if str(x).strip()]
        return cls(preferences=prefs, likes=likes, dislikes=dislikes, traits=traits)


@dataclass
class UserMemoryState:
    user_id: str
    profile: UserProfile = field(default_factory=UserProfile)
    events: list[StructuredEvent] = field(default_factory=list)
    updated_at: str = field(default_factory=utc_iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "profile": self.profile.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserMemoryState":
        profile = UserProfile.from_dict(dict(data.get("profile") or {}))
        events = [StructuredEvent.from_dict(x) for x in (data.get("events") or []) if isinstance(x, dict)]
        return cls(
            user_id=str(data.get("user_id") or "guest"),
            profile=profile,
            events=events,
            updated_at=str(data.get("updated_at") or utc_iso_now()),
        )


@dataclass
class BrainDecision:
    intent: Intent
    response_style: str
    deterministic_hints: list[str] = field(default_factory=list)
    direct_response: str | None = None
