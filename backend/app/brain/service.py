"""
Brain layer:
- intent classification
- structured memory extraction/storage
- temporal normalization
- logic-first query handling
- response style control
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from backend.app.brain.intent import classify_intent
from backend.app.brain.models import BrainDecision, Intent, StructuredEvent, StructuredPreference
from backend.app.brain.store import BrainMemoryStore, get_brain_store
from backend.app.brain.temporal import parse_date_from_text


SHOW_MAIN_CHARACTER: dict[str, str] = {
    "naruto": "Naruto Uzumaki",
    "one piece": "Monkey D. Luffy",
    "bleach": "Ichigo Kurosaki",
    "dragon ball": "Goku",
    "attack on titan": "Eren Yeager",
    "death note": "Light Yagami",
    "demon slayer": "Tanjiro Kamado",
    "jujutsu kaisen": "Yuji Itadori",
}

EVENT_KEYWORDS = (
    "test",
    "exam",
    "interview",
    "meeting",
    "deadline",
    "class",
    "appointment",
    "birthday",
    "presentation",
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 3}


def _style_for_intent(intent: Intent) -> str:
    if intent == Intent.GREETING:
        return "warm_conversational"
    if intent == Intent.FACTUAL_QUESTION:
        return "concise_direct"
    if intent == Intent.GUESS_REQUEST:
        return "short_guess"
    if intent == Intent.MEMORY_QUERY:
        return "memory_query_concise"
    if intent == Intent.MEMORY_UPDATE:
        return "confirm_brief"
    return "balanced_direct"


@dataclass
class ExtractedMemory:
    event: StructuredEvent | None = None
    preferences: list[StructuredPreference] = None
    likes: list[str] = None
    dislikes: list[str] = None
    traits: list[str] = None

    def __post_init__(self):
        self.preferences = list(self.preferences or [])
        self.likes = list(self.likes or [])
        self.dislikes = list(self.dislikes or [])
        self.traits = list(self.traits or [])

    @property
    def has_updates(self) -> bool:
        return bool(self.event or self.preferences or self.likes or self.dislikes or self.traits)


class BrainService:
    def __init__(self, store: BrainMemoryStore):
        self.store = store

    def _extract_event(self, message: str, today: date) -> StructuredEvent | None:
        text = (message or "").strip()
        lower = text.lower()
        date_iso = parse_date_from_text(text, today=today)
        if not date_iso:
            return None

        name = ""
        for keyword in EVENT_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", lower):
                name = keyword
                break

        if not name:
            m = re.search(r"\bmy\s+([a-z][a-z0-9\s]{1,32})\s+(?:is|was|will be|on|at)\b", lower)
            if m:
                candidate = m.group(1).strip()
                if candidate and len(candidate) <= 32:
                    name = candidate
        if not name:
            return None

        return StructuredEvent(name=name, date=date_iso, source_message=text)

    def _extract_preferences(self, message: str) -> ExtractedMemory:
        text = (message or "").strip()
        lower = text.lower()
        out = ExtractedMemory()

        # Main-character preference.
        if ("main character" in lower or re.search(r"\bmc\b", lower)) and "anime" in lower:
            out.preferences.append(
                StructuredPreference(
                    category="anime",
                    key="likes_main_character",
                    value=True,
                    source_message=text,
                )
            )

        like_match = re.search(r"\bi\s+(?:like|love|prefer)\s+(.+?)(?:[.!?]|$)", lower)
        if like_match:
            raw = like_match.group(1).strip()
            if raw:
                out.likes.append(raw)
                out.preferences.append(
                    StructuredPreference(
                        category="general",
                        key="like_item",
                        value=raw,
                        source_message=text,
                    )
                )

        dislike_match = re.search(r"\bi\s+(?:dislike|hate|don't like)\s+(.+?)(?:[.!?]|$)", lower)
        if dislike_match:
            raw = dislike_match.group(1).strip()
            if raw:
                out.dislikes.append(raw)
                out.preferences.append(
                    StructuredPreference(
                        category="general",
                        key="dislike_item",
                        value=raw,
                        source_message=text,
                    )
                )

        trait_match = re.search(r"\bi\s+(?:am|\'m)\s+(.+?)(?:[.!?]|$)", lower)
        if trait_match:
            trait = trait_match.group(1).strip()
            if trait and len(trait) <= 50 and "on " not in trait:
                out.traits.append(trait)

        return out

    def _extract_structured_memory(self, message: str, today: date) -> ExtractedMemory:
        pref = self._extract_preferences(message)
        event = self._extract_event(message, today=today)
        pref.event = event
        return pref

    def _upsert_memory(self, user_id: str, memory: ExtractedMemory):
        if memory.event:
            self.store.upsert_event(user_id, memory.event)
        for pref in memory.preferences:
            self.store.upsert_preference(user_id, pref)
        for item in memory.likes:
            self.store.set_profile_list(user_id, "likes", item)
        for item in memory.dislikes:
            self.store.set_profile_list(user_id, "dislikes", item)
        for item in memory.traits:
            self.store.set_profile_list(user_id, "traits", item)

    def _event_status(self, event_date: str, today: date) -> str:
        try:
            d = datetime.fromisoformat(event_date).date()
        except Exception:
            return "unknown"
        if d < today:
            return "past"
        if d > today:
            return "future"
        return "today"

    def _find_event_for_query(self, query: str, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        lower = (query or "").lower()
        event_name = ""
        m = re.search(r"\bwhen\s+(?:is|was)\s+my\s+([a-z0-9\s]{2,40})\??$", lower)
        if m:
            event_name = m.group(1).strip()
        if not event_name:
            for k in EVENT_KEYWORDS:
                if re.search(rf"\b{k}\b", lower):
                    event_name = k
                    break
        if not event_name:
            return None

        query_tokens = _tokens(event_name)
        best = None
        best_score = -1
        for ev in events:
            name = str(ev.get("name") or "")
            ev_tokens = _tokens(name)
            score = len(query_tokens.intersection(ev_tokens))
            if score > best_score:
                best = ev
                best_score = score
        if query_tokens and best_score <= 0:
            return None
        return best

    def _has_pref_likes_main_character(self, preferences: list[dict[str, Any]]) -> bool:
        for pref in preferences:
            key = str(pref.get("key") or "").strip().lower()
            val = pref.get("value")
            if key in {"likes_main_character", "likes_mc"} and bool(val):
                return True
        return False

    def _show_from_query(self, query: str) -> str:
        lower = (query or "").lower()
        for show in SHOW_MAIN_CHARACTER:
            if show in lower:
                return show
        return ""

    def _relevant_preferences(self, query: str, preferences: list[dict[str, Any]], include_main_char_pref: bool) -> list[dict[str, Any]]:
        q = _tokens(query)
        out: list[dict[str, Any]] = []
        for pref in preferences:
            key = str(pref.get("key") or "")
            category = str(pref.get("category") or "")
            value = str(pref.get("value") or "")
            if include_main_char_pref and key == "likes_main_character":
                out.append(pref)
                continue
            combo = f"{key} {category} {value}"
            if q.intersection(_tokens(combo)):
                out.append(pref)
        return out[:8]

    def _relevant_events(self, query: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q = _tokens(query)
        out: list[dict[str, Any]] = []
        for ev in events:
            name = str(ev.get("name") or "")
            if q.intersection(_tokens(name)):
                out.append(ev)
        return out[:6]

    def _short_term_context(self, chat_session: Any, limit: int = 10) -> str:
        if chat_session is None:
            return ""
        msgs = list(getattr(chat_session, "messages", []) or [])
        if not msgs:
            return ""
        rows = []
        for msg in msgs[-limit:]:
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            label = "User" if role == "user" else "Assistant"
            rows.append(f"{label}: {str(msg.get('content') or '').strip()}")
        return "\n".join(rows)

    def _memory_update_ack(self, memory: ExtractedMemory) -> str:
        lines: list[str] = []
        if memory.event:
            lines.append(f"Got it. I'll remember your {memory.event.name} is on {memory.event.date}.")
        if memory.preferences:
            if any(p.key == "likes_main_character" and bool(p.value) for p in memory.preferences):
                lines.append("Noted. You prefer main characters in anime.")
            elif memory.likes:
                lines.append("Got it. I'll remember your preference.")
            elif memory.dislikes:
                lines.append("Understood. I'll remember what you dislike.")
        if memory.traits:
            lines.append("Noted.")
        if not lines:
            return "Noted."
        return " ".join(lines[:2])

    def process(self, user_id: str, message: str, chat_session: Any | None = None) -> BrainDecision:
        today = datetime.now().astimezone().date()
        intent = classify_intent(message)
        style = _style_for_intent(intent)
        hints: list[str] = [f"TODAY_DATE: {today.isoformat()}", f"INTENT: {intent.value}"]

        state = self.store.get_user_state(user_id).to_dict()
        profile = dict(state.get("profile") or {})
        preferences = [dict(x) for x in (profile.get("preferences") or []) if isinstance(x, dict)]
        events = [dict(x) for x in (state.get("events") or []) if isinstance(x, dict)]

        # 1) Memory update path.
        if intent == Intent.MEMORY_UPDATE:
            extracted = self._extract_structured_memory(message, today=today)
            if extracted.has_updates:
                self._upsert_memory(user_id, extracted)
                return BrainDecision(
                    intent=intent,
                    response_style=style,
                    deterministic_hints=hints,
                    direct_response=self._memory_update_ack(extracted),
                )

        # Refresh state after potential update.
        state = self.store.get_user_state(user_id).to_dict()
        profile = dict(state.get("profile") or {})
        preferences = [dict(x) for x in (profile.get("preferences") or []) if isinstance(x, dict)]
        events = [dict(x) for x in (state.get("events") or []) if isinstance(x, dict)]

        # 2) Memory query logic path.
        if intent == Intent.MEMORY_QUERY:
            event = self._find_event_for_query(message, events)
            if event:
                ev_name = str(event.get("name") or "event")
                ev_date = str(event.get("date") or "")
                status = self._event_status(ev_date, today=today)
                hints.append(f"DERIVED_KNOWLEDGE: event={ev_name}, date={ev_date}, status={status}")
                if status == "past":
                    return BrainDecision(intent=intent, response_style=style, deterministic_hints=hints, direct_response=f"Your {ev_name} was on {ev_date}.")
                if status == "future":
                    return BrainDecision(intent=intent, response_style=style, deterministic_hints=hints, direct_response=f"Your {ev_name} is on {ev_date}.")
                if status == "today":
                    return BrainDecision(intent=intent, response_style=style, deterministic_hints=hints, direct_response=f"Your {ev_name} is today ({ev_date}).")

            if re.search(r"\bwhen\s+(?:is|was)\s+my\b", (message or "").lower()):
                return BrainDecision(
                    intent=intent,
                    response_style=style,
                    deterministic_hints=hints,
                    direct_response="I don't have any record of that.",
                )

            if re.search(r"\bwhat\s+is\s+my\b", (message or "").lower()) and not preferences:
                return BrainDecision(
                    intent=intent,
                    response_style=style,
                    deterministic_hints=hints,
                    direct_response="I don't have that information yet.",
                )

        # 3) Guessing logic path.
        if intent == Intent.GUESS_REQUEST:
            show = self._show_from_query(message)
            if show and re.search(r"\bcharacter\b", (message or "").lower()):
                main_char = SHOW_MAIN_CHARACTER.get(show, "")
                if main_char:
                    likes_mc = self._has_pref_likes_main_character(preferences)
                    if likes_mc:
                        hints.append(f"DERIVED_KNOWLEDGE: user_prefers_main_character=true, {show}_mc={main_char}")
                        return BrainDecision(
                            intent=intent,
                            response_style=style,
                            deterministic_hints=hints,
                            direct_response=f"My guess is {main_char}.",
                        )
                    return BrainDecision(
                        intent=intent,
                        response_style=style,
                        deterministic_hints=hints,
                        direct_response=f"I'm guessing: {main_char}.",
                    )

        # 4) Relevance-filtered structured context for LLM.
        include_main_char_pref = intent == Intent.GUESS_REQUEST
        rel_prefs = self._relevant_preferences(message, preferences, include_main_char_pref=include_main_char_pref)
        rel_events = self._relevant_events(message, events)
        if rel_prefs:
            hints.append(f"USER_PROFILE_PREFERENCES: {json.dumps(rel_prefs, ensure_ascii=True)}")
        if rel_events:
            hints.append(f"USER_EVENTS: {json.dumps(rel_events, ensure_ascii=True)}")

        short_term = self._short_term_context(chat_session, limit=10)
        if short_term:
            hints.append(f"SHORT_TERM_MEMORY:\n{short_term}")

        return BrainDecision(intent=intent, response_style=style, deterministic_hints=hints)


_brain_service: BrainService | None = None


def get_brain_service(base_dir: Path) -> BrainService:
    global _brain_service
    if _brain_service is None:
        # Legacy JSON path is used only for one-time migration into relational tables.
        path = base_dir / "memory" / "structured_memory.json"
        _brain_service = BrainService(store=get_brain_store(path=path))
    return _brain_service
