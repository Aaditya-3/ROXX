"""
FastAPI Backend

Main API server for the chat application.
Uses Groq as the only LLM provider.
"""

# CRITICAL: Load .env FIRST, before any other imports that use env vars.
from pathlib import Path
from dotenv import load_dotenv

# Load from project root by path (works regardless of current working directory)
_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / ".env"
load_dotenv(dotenv_path=_env_file)
if _env_file.exists():
    print(f"Loaded .env from: {_env_file}")
else:
    load_dotenv()  # fallback: current directory

import os
import json
import re
import time
print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))

from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid

from memory.memory_extractor import extract_memory
from memory.memory_retriever import retrieve_memories
from memory.memory_store import get_memory_store
from backend.app.core.tools.realtime_info import should_fetch_realtime, get_realtime_context
from backend.app.core.db.mongo import get_db
from backend.app.core.db.relational import (
    DBChatMessage,
    DBChatSession,
    DBUsageLog,
    DBUser,
    DBUserSetting,
    get_relational_session,
    init_relational_db,
)
from backend.app.core.config import get_settings
from backend.app.core.middleware import setup_middleware, validate_message_payload
from backend.app.core.tools.preference_reasoning import (
    parse_preference_query,
    classify_memory_for_query,
)
from backend.app.orchestrator.factory import build_chat_orchestrator
from backend.app.orchestrator.stream_handler import OrchestratorStreamHandler
from backend.app.orchestrator.types import OrchestratorInput
from backend.app.observability.logging import setup_logging, log_event
from backend.app.observability.metrics import metrics
from backend.app.services.semantic_memory_service import get_semantic_memory_service
from backend.app.services.token_usage import estimate_tokens, estimate_cost_usd
from backend.app.services.tool_router import run_agent_turn
from backend.app.tasks.memory_tasks import enqueue_ingest_message, run_compression, run_decay, run_reembedding
from backend.app.security.jwt_rotation import issue_access_token, issue_refresh_token, rotate_tokens
from backend.app.brain.service import get_brain_service

project_root = _project_root
settings = get_settings()
init_relational_db()
setup_logging()
brain_service = get_brain_service(project_root)
api_key_loaded = bool(os.getenv("GROQ_API_KEY"))
realtime_web_enabled = os.getenv("ENABLE_REALTIME_WEB", "true").strip().lower() in {"1", "true", "yes", "on"}
mongo_enabled = os.getenv("ENABLE_MONGO", "false").strip().lower() in {"1", "true", "yes", "on"}
SEMANTIC_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "but", "with", "about",
    "what", "which", "who", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "my", "me", "i", "your", "you", "it", "that", "this", "there",
    "tell", "show", "list", "please", "have",
}
SCHEDULE_MARKERS = {
    "schedule", "class", "classes", "test", "exam", "deadline", "event",
    "upcoming", "tomorrow", "today", "important", "date", "day",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
if not api_key_loaded:
    print("WARNING: GROQ_API_KEY not found. Create .env in project root and set GROQ_API_KEY=...")


def sanitize_response(user_message: str, reply: str) -> str:
    """
    Post-process the model reply to remove common chatty greetings / fluff.
    Keeps things concise and focused on the user's input.
    """
    if not isinstance(reply, str):
        return reply

    text = reply.strip()

    # If the user didn't ask a question and is just stating facts,
    # aggressively strip leading greetings and small talk.
    user_is_question = "?" in user_message
    if not user_is_question:
        lower = text.lower()
        greeting_prefixes = [
            "hi,", "hi ", "hello,", "hello ",
            "hey,", "hey ", "greetings,", "greetings ",
            "sure,", "of course,", "i can see why", "i understand"
        ]
        for prefix in greeting_prefixes:
            if lower.startswith(prefix):
                # Remove the prefix from the original text, not just lower
                text = text[len(prefix):].lstrip()
                lower = text.lower()

    # Remove common meta-reasoning lead-ins that expose internal logic.
    meta_prefix_patterns = [
        r"^(based on|according to|from)\s+(the\s+)?(memory|context|rules|instructions|assumptions?)\b[\s,:-]*",
        r"^i\s+(assume|inferred?|reasoned?|used|considered)\b[\s,:-]*",
        r"^internally\b[\s,:-]*",
    ]
    original = text
    for pattern in meta_prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = text.strip()
    if not text:
        text = original.strip()

    return text


def _normalize_text_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _semantic_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    return {t for t in tokens if len(t) > 2 and t not in SEMANTIC_STOPWORDS}


def _is_schedule_intent_message(message: str) -> bool:
    msg_l = (message or "").lower().strip()
    if not msg_l:
        return False
    msg_tokens = set(re.findall(r"[a-z0-9]+", msg_l))
    if _extract_day_reference(msg_l):
        return True
    if SCHEDULE_MARKERS.intersection(msg_tokens):
        return True
    if re.search(r"\bwhen\s+is\b", msg_l) and {"class", "test", "exam", "event", "schedule"}.intersection(msg_tokens):
        return True
    return False


def _last_assistant_reply(chat_session: "ChatSession") -> str:
    for msg in reversed(chat_session.messages):
        if str(msg.get("role", "")).lower() == "assistant":
            content = str(msg.get("content", "")).strip()
            if content:
                return content
    return ""


def _is_repetitive_reply(reply: str, chat_session: "ChatSession") -> bool:
    cand = _normalize_text_for_compare(reply)
    if not cand:
        return False
    recent_assistant = []
    for msg in reversed(chat_session.messages):
        if str(msg.get("role", "")).lower() != "assistant":
            continue
        content = _normalize_text_for_compare(str(msg.get("content", "")))
        if content:
            recent_assistant.append(content)
        if len(recent_assistant) >= 3:
            break
    for prev in recent_assistant:
        if cand == prev:
            return True
        if len(cand) > 40 and SequenceMatcher(None, cand, prev).ratio() >= 0.95:
            return True
    return False


def _is_irrelevant_reply(user_message: str, reply: str) -> bool:
    user_sem = _semantic_tokens(user_message)
    reply_sem = _semantic_tokens(reply)
    if not reply_sem:
        return False

    # Reject schedule-like answers for non-schedule questions.
    if not _is_schedule_intent_message(user_message):
        schedule_overlap = len(reply_sem.intersection(SCHEDULE_MARKERS))
        if schedule_overlap >= 2:
            return True

    if user_sem:
        overlap = len(user_sem.intersection(reply_sem))
        if overlap == 0 and len(reply.split()) >= 10:
            return True

    return False


from backend.app.core.auth.simple_auth import router as simple_auth_router
from backend.app.api.platform import router as platform_router


app = FastAPI(title="AI Chat with Memory")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
setup_middleware(app)

frontend_path = project_root / "frontend"
if frontend_path.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_path)), name="frontend")

# Mount auth router
app.include_router(simple_auth_router)
app.include_router(platform_router)


@app.post("/auth/token/issue")
async def issue_tokens(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    return {
        "access_token": issue_access_token(user_id),
        "refresh_token": issue_refresh_token(user_id),
    }


@app.post("/auth/token/rotate")
async def rotate_user_tokens(payload: dict):
    user_id = str((payload or {}).get("user_id") or "").strip()
    refresh_token = str((payload or {}).get("refresh_token") or "").strip()
    if not user_id or not refresh_token:
        raise HTTPException(status_code=400, detail="user_id and refresh_token are required")
    try:
        access, refresh = rotate_tokens(user_id, refresh_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    return {"access_token": access, "refresh_token": refresh}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "")
    classification = type(exc).__name__
    log_event(
        "unhandled_exception",
        request_id=request_id,
        path=request.url.path,
        error_class=classification,
        error=str(exc),
    )
    return JSONResponse(status_code=500, content={"error": f"Internal error: {classification}"})


def _parse_iso_datetime(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


class ChatSession:
    def __init__(self, user_id: str):
        now = datetime.now(timezone.utc)
        self.id: str = str(uuid.uuid4())
        self.user_id: str = user_id
        self.title: str = "New Chat"
        self.created_at: datetime = now
        self.updated_at: datetime = now
        self.messages: List[dict] = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatSession":
        session = cls(user_id=str(data.get("user_id") or "guest"))
        session.id = str(data.get("id") or str(uuid.uuid4()))
        session.title = str(data.get("title") or "New Chat")
        session.created_at = _parse_iso_datetime(data.get("created_at"))
        session.updated_at = _parse_iso_datetime(data.get("updated_at"))
        session.messages = [x for x in (data.get("messages") or []) if isinstance(x, dict)]
        return session


chat_sessions: dict[str, ChatSession] = {}
current_chat_id: Optional[str] = None
chat_storage_path = project_root / "memory" / "chat_sessions.json"
MAX_USER_CHAT_TOKENS = int(os.getenv("MAX_USER_CHAT_TOKENS", "130000"))
CHAT_HISTORY_MAX_TOKENS = int(os.getenv("CHAT_HISTORY_MAX_TOKENS", "3200"))
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "40"))
LONG_TERM_PROFILE_MAX_ITEMS = int(os.getenv("LONG_TERM_PROFILE_MAX_ITEMS", "40"))
SESSION_TEMP_FACTS_MAX_ITEMS = int(os.getenv("SESSION_TEMP_FACTS_MAX_ITEMS", "14"))

if mongo_enabled:
    try:
        _db = get_db()
        chat_collection = _db["chat_sessions"]
    except Exception:
        chat_collection = None
else:
    chat_collection = None


def _disable_chat_mongo(reason: str):
    global chat_collection
    if chat_collection is not None:
        print(f"INFO: MongoDB chat storage disabled for this run: {reason}")
    chat_collection = None


def _synthetic_username_for_user_id(user_id: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "_", (user_id or "").lower()).strip("_")
    if not compact:
        compact = "user"
    return f"u_{compact}"[:120]


def _ensure_user_row(user_id: str, db=None):
    uid = (user_id or "").strip()
    if not uid:
        return
    if db is None:
        with get_relational_session() as session:
            _ensure_user_row(uid, db=session)
        return

    existing = db.query(DBUser).filter(DBUser.id == uid).first()
    if existing:
        return

    candidate = _synthetic_username_for_user_id(uid)
    unique_username = candidate
    idx = 1
    while db.query(DBUser.id).filter(DBUser.username == unique_username).first():
        suffix = f"_{idx}"
        unique_username = f"{candidate[: max(1, 120 - len(suffix))]}{suffix}"
        idx += 1

    db.add(
        DBUser(
            id=uid,
            username=unique_username,
            email=None,
            password_hash="external_auth",
            name=uid,
            plan_type="free",
            is_active=True,
        )
    )


def _estimate_tokens(text: str) -> int:
    # Lightweight approximation for memory budgeting.
    return max(1, len((text or "").strip()) // 4)


def _chat_token_count(session: ChatSession) -> int:
    total = 0
    for message in session.messages:
        total += _estimate_tokens(str(message.get("content", "")))
    return total


def _persist_chat_session(session: ChatSession):
    with get_relational_session() as db:
        _ensure_user_row(session.user_id, db=db)
        row = db.query(DBChatSession).filter(DBChatSession.id == session.id).first()
        if row is None:
            row = DBChatSession(
                id=session.id,
                user_id=session.user_id,
                title=session.title or "New Chat",
                created_at=session.created_at,
                updated_at=session.updated_at,
            )
            db.add(row)
        else:
            row.user_id = session.user_id
            row.title = session.title or "New Chat"
            row.created_at = session.created_at
            row.updated_at = session.updated_at

        db.query(DBChatMessage).filter(DBChatMessage.session_id == session.id).delete(synchronize_session=False)
        for item in session.messages:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            ts = _parse_iso_datetime(item.get("timestamp"))
            db.add(
                DBChatMessage(
                    session_id=session.id,
                    user_id=session.user_id,
                    role=role,
                    content=content,
                    timestamp=ts,
                    token_count=_estimate_tokens(content),
                )
            )


def _delete_chat_session_storage(chat_id: str):
    with get_relational_session() as db:
        db.query(DBChatMessage).filter(DBChatMessage.session_id == chat_id).delete(synchronize_session=False)
        db.query(DBChatSession).filter(DBChatSession.id == chat_id).delete(synchronize_session=False)


def _load_chat_sessions_from_relational() -> int:
    chat_sessions.clear()
    with get_relational_session() as db:
        session_rows = db.query(DBChatSession).order_by(DBChatSession.updated_at.desc()).all()
        message_rows = db.query(DBChatMessage).order_by(DBChatMessage.timestamp.asc()).all()

    grouped_messages: dict[str, list[dict[str, Any]]] = {}
    for msg in message_rows:
        grouped_messages.setdefault(msg.session_id, []).append(
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": (msg.timestamp or datetime.now(timezone.utc)).isoformat(),
            }
        )

    for row in session_rows:
        session = ChatSession(user_id=row.user_id)
        session.id = row.id
        session.title = row.title or "New Chat"
        session.created_at = row.created_at or datetime.now(timezone.utc)
        session.updated_at = row.updated_at or session.created_at
        session.messages = grouped_messages.get(row.id, [])
        chat_sessions[session.id] = session

    return len(session_rows)


def _load_legacy_chat_sessions_from_mongo() -> list[ChatSession]:
    if chat_collection is None:
        return []
    try:
        data = list(chat_collection.find({}, {"_id": 0}))
    except Exception as exc:
        _disable_chat_mongo(str(exc))
        return []
    out: list[ChatSession] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ChatSession.from_dict(item))
        except Exception:
            continue
    return out


def _load_legacy_chat_sessions_from_json() -> list[ChatSession]:
    if not chat_storage_path.exists():
        return []
    try:
        with open(chat_storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: list[ChatSession] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            out.append(ChatSession.from_dict(item))
        except Exception:
            continue
    return out


def _trim_chat_to_budget(session: ChatSession, token_budget: int):
    while session.messages and _chat_token_count(session) > token_budget:
        session.messages.pop(0)


def _build_long_term_profile(user_id: str, query_message: str = "", max_items: int = LONG_TERM_PROFILE_MAX_ITEMS) -> str:
    """
    Build a compact long-term profile snapshot from persisted user memories.
    """
    store = get_memory_store()
    memories = [m for m in store.get_user_memories(user_id) if m.confidence >= 0.5]
    if not memories:
        return ""

    def _sort_key(memory):
        return (memory.confidence, memory.last_updated.timestamp())

    def _is_critical(memory) -> bool:
        md = memory.metadata or {}
        return str(md.get("domain", "")).lower() == "critical_context"

    def _is_profile(memory) -> bool:
        md = memory.metadata or {}
        return str(md.get("scope", "")).upper() == "PROFILE"

    def _is_global_rule(memory) -> bool:
        md = memory.metadata or {}
        scope = str(md.get("scope", "")).upper()
        rule_type = str(md.get("rule_type", "")).lower()
        key_l = str(memory.key or "").lower()
        return (
            (scope == "GLOBAL_SCOPE" and rule_type in {"universal", "generic"})
            or key_l.startswith("custom.preference_rule.")
            or key_l == "entertainment.global_character_rule"
        )

    critical = sorted([m for m in memories if _is_critical(m)], key=_sort_key, reverse=True)
    profile = sorted([m for m in memories if _is_profile(m)], key=_sort_key, reverse=True)
    global_rules = sorted([m for m in memories if _is_global_rule(m)], key=_sort_key, reverse=True)
    recent = sorted(memories, key=_sort_key, reverse=True)

    query_tokens = _semantic_tokens(query_message)

    def _memory_tokens(memory) -> set[str]:
        md = memory.metadata or {}
        raw = " ".join(
            [
                str(memory.key or ""),
                str(memory.value or ""),
                str(md.get("domain", "") or ""),
                str(md.get("category", "") or ""),
                str(md.get("subject", "") or ""),
            ]
        ).lower()
        return set(re.findall(r"[a-z0-9]+", raw))

    def _is_query_relevant(memory) -> bool:
        if _is_critical(memory) or _is_profile(memory):
            return True
        if not query_tokens:
            return True
        overlap = len(query_tokens.intersection(_memory_tokens(memory)))
        if overlap == 0:
            return False
        if _is_global_rule(memory) and overlap < 2:
            return False
        return True

    critical = [m for m in critical if _is_query_relevant(m)]
    profile = [m for m in profile if _is_query_relevant(m)]
    global_rules = [m for m in global_rules if _is_query_relevant(m)]
    recent = [m for m in recent if _is_query_relevant(m)]

    selected = []
    selected_ids = set()

    def _add_bucket(items, limit: int):
        for item in items:
            if len(selected) >= max_items or limit <= 0:
                break
            if item.id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.id)
            limit -= 1

    _add_bucket(critical, 12)
    _add_bucket(profile, 10)
    _add_bucket(global_rules, 10)
    _add_bucket(recent, max_items)

    lines = []
    for m in selected:
        md = m.metadata or {}
        scope = md.get("scope")
        rule_type = md.get("rule_type")
        subject = md.get("subject")
        importance = md.get("importance")
        suffix_parts = []
        if scope:
            suffix_parts.append(f"scope={scope}")
        if rule_type:
            suffix_parts.append(f"rule_type={rule_type}")
        if subject:
            suffix_parts.append(f"subject={subject}")
        if importance:
            suffix_parts.append(f"importance={importance}")
        suffix = f" [{'; '.join(suffix_parts)}]" if suffix_parts else ""
        lines.append(f"- {m.key}: {m.value}{suffix}")
    return "\n".join(lines)


def _build_chat_history_context(chat_session: ChatSession, max_messages: int = CHAT_HISTORY_MAX_MESSAGES) -> str:
    """
    Build recent in-session chat history context for continuity/coreference.
    """
    if not chat_session.messages:
        return ""

    budgeted_lines = []
    used_tokens = 0
    included = 0
    for msg in reversed(chat_session.messages):
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = "User" if role == "user" else "Assistant"
        line = f"{label}: {content}"
        line_tokens = _estimate_tokens(line)
        if budgeted_lines and used_tokens + line_tokens > CHAT_HISTORY_MAX_TOKENS:
            break
        budgeted_lines.append(line)
        used_tokens += line_tokens
        included += 1
        if included >= max_messages:
            break

    budgeted_lines.reverse()
    return "\n".join(budgeted_lines)


def _build_session_temp_facts(chat_session: ChatSession, max_items: int = SESSION_TEMP_FACTS_MAX_ITEMS) -> str:
    """
    Build temporary chat-scoped user facts from this session only.
    Captures recent declarative user statements to strengthen follow-up continuity
    without polluting long-term memory.
    """
    if not chat_session.messages:
        return ""

    question_starters = {
        "what", "which", "who", "when", "where", "why", "how",
        "do", "does", "did", "can", "could", "would", "should",
        "is", "are", "am", "will",
    }
    seen = set()
    items = []
    for msg in reversed(chat_session.messages):
        if str(msg.get("role", "")).lower() != "user":
            continue
        content = re.sub(r"\s+", " ", str(msg.get("content", "")).strip())
        if not content:
            continue
        lower = content.lower()
        first = re.findall(r"[a-z]+", lower)
        first_token = first[0] if first else ""
        is_question_like = ("?" in content) or (first_token in question_starters)
        if is_question_like:
            continue
        norm = lower.strip(" .!?")
        if len(norm) < 4 or norm in seen:
            continue
        seen.add(norm)
        items.append(content)
        if len(items) >= max_items:
            break

    if not items:
        return ""

    items.reverse()
    return "\n".join(f"- {item}" for item in items)


def _latest_entity_from_chat_history(chat_session: ChatSession, max_messages: int = 14) -> str:
    """
    Extract the most recent explicit entity mention like 'in/from/of <entity>'
    from recent user messages.
    """
    if not chat_session.messages:
        return ""
    recent = list(reversed(chat_session.messages[-max_messages:]))
    for msg in recent:
        if str(msg.get("role", "")).lower() != "user":
            continue
        text = str(msg.get("content", "")).lower()
        m = re.search(r"\b(?:in|from|of)\s+([a-z0-9' \-]{2,40})\b", text)
        if not m:
            continue
        entity = re.sub(r"[^a-z0-9]+", "_", m.group(1)).strip("_")
        if entity:
            return entity
    return ""


def _latest_assistant_question(chat_session: ChatSession, max_messages: int = 16) -> str:
    if not chat_session.messages:
        return ""
    recent = list(reversed(chat_session.messages[-max_messages:]))
    fallback = ""
    for msg in recent:
        if str(msg.get("role", "")).lower() != "assistant":
            continue
        text = str(msg.get("content", "")).strip()
        if text and not fallback:
            fallback = text
        if "?" in text:
            return text
    return fallback


def _resolve_short_reply_context(user_message: str, chat_session: ChatSession) -> str:
    """
    Expand short confirmations like 'yeah' using the last assistant question,
    so downstream logic can preserve context.
    """
    msg = (user_message or "").strip()
    msg_l = msg.lower()
    if not msg:
        return msg

    affirmative = {"yes", "yeah", "yep", "yup", "right", "correct", "true", "ok", "okay"}
    negative = {"no", "nope", "nah", "false"}
    if msg_l not in affirmative and msg_l not in negative:
        return msg

    question = _latest_assistant_question(chat_session)
    if not question:
        return msg

    question_text = re.sub(r"\s+", " ", question).strip().rstrip(" ?!.")
    if not question_text:
        return msg

    if msg_l in affirmative:
        return f"I confirm this: {question_text}."
    return f"I reject this: {question_text}."


def _is_short_acknowledgement(user_message: str) -> bool:
    msg = (user_message or "").strip().lower()
    if not msg:
        return False
    tokens = re.findall(r"[a-z]+", msg)
    if not tokens or len(tokens) > 3:
        return False
    ack_tokens = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "right", "correct", "no", "nope", "nah"}
    return all(token in ack_tokens for token in tokens)


def _augment_query_for_continuity(user_message: str, chat_session: ChatSession) -> str:
    """
    If current query is implicit ('that show', 'it', 'the one I mentioned'),
    attach the last explicit entity from this chat so deterministic reasoning
    can connect turns consistently.
    """
    msg = (user_message or "").strip()
    msg_l = msg.lower()
    if not msg:
        return msg

    has_entity_now = bool(re.search(r"\b(?:in|from|of)\s+[a-z0-9' \-]{2,40}\b", msg_l))
    if has_entity_now:
        return msg

    implicit_markers = {
        "that show",
        "that anime",
        "that series",
        "the character i mentioned",
        "the one i mentioned",
        "that one",
        "it",
    }
    has_implicit_marker = any(marker in msg_l for marker in implicit_markers)
    # Also treat incomplete preference follow-ups as implicit references.
    asks_fav_without_entity = (
        ("fav" in msg_l or "favorite" in msg_l or "favourite" in msg_l)
        and ("char" in msg_l or "character" in msg_l)
    )
    if not has_implicit_marker and not asks_fav_without_entity:
        return msg

    entity = _latest_entity_from_chat_history(chat_session)
    if not entity:
        return msg

    entity_human = entity.replace("_", " ")
    return f"{msg} in {entity_human}"


def _extract_day_reference(msg_l: str) -> str:
    m = re.search(r"\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", msg_l)
    return (m.group(1) if m else "").lower()


def _resolve_day_to_date(day_label: str, now: datetime) -> str:
    day_label = (day_label or "").lower().strip()
    if not day_label:
        return ""

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    if day_label == "today":
        dt = now
    elif day_label == "tomorrow":
        dt = now + timedelta(days=1)
    elif day_label in weekdays:
        target = weekdays[day_label]
        delta = (target - now.weekday()) % 7
        if delta == 0:
            delta = 7
        dt = now + timedelta(days=delta)
    else:
        return day_label
    return dt.strftime("%A, %B %d, %Y")


def _same_day_reference(stored_day: str, query_day: str, now: datetime) -> bool:
    s = (stored_day or "").lower().strip()
    q = (query_day or "").lower().strip()
    if not s or not q:
        return False
    if s == q:
        return True
    s_date = _resolve_day_to_date(s, now)
    q_date = _resolve_day_to_date(q, now)
    return bool(s_date and q_date and s_date == q_date)


def handle_schedule_query(user_message: str, user_id: str) -> Optional[str]:
    """
    Deterministic upcoming-event resolver driven by stored lifestyle.upcoming_event.*
    memories. Does not require '?' to interpret a query.
    """
    msg_l = (user_message or "").lower().strip()
    if not msg_l:
        return None
    if not _is_schedule_intent_message(msg_l):
        return None

    store = get_memory_store()
    memories = [
        m
        for m in store.get_user_memories(user_id)
        if m.confidence >= 0.5 and m.key.startswith("lifestyle.upcoming_event.")
    ]
    if not memories:
        return None

    # Latest value per event label.
    latest: dict[str, str] = {}
    memories_sorted = sorted(memories, key=lambda m: (m.confidence, m.last_updated.timestamp()), reverse=True)
    for m in memories_sorted:
        md = m.metadata or {}
        event_label = str(md.get("event") or "").strip().lower()
        if not event_label:
            event_label = m.key.replace("lifestyle.upcoming_event.", "").replace("_", " ").strip().lower()
        if not event_label:
            continue
        if event_label not in latest:
            latest[event_label] = (m.value or "").strip().lower()

    if not latest:
        return None

    now = datetime.now().astimezone()
    day_query = _extract_day_reference(msg_l)
    asks_when = "when" in msg_l

    msg_tokens = set(re.findall(r"[a-z0-9]+", msg_l))

    def _event_match_score(event_label: str) -> int:
        tokens = set(re.findall(r"[a-z0-9]+", event_label))
        return len(tokens.intersection(msg_tokens))

    best_event = ""
    best_score = 0
    for event_label in latest.keys():
        s = _event_match_score(event_label)
        if s > best_score:
            best_score = s
            best_event = event_label

    # If a concrete event is asked, answer it directly.
    if best_event and best_score > 0:
        day = latest[best_event]
        pretty = _resolve_day_to_date(day, now)
        if asks_when or not day_query:
            return f"Your important {best_event} is on {pretty}."
        if _same_day_reference(day, day_query, now):
            return f"Yes, your important {best_event} is on {pretty}."
        return f"Your important {best_event} is on {pretty}, not {day_query}."

    if day_query:
        hits = []
        for event_label, d in latest.items():
            if _same_day_reference(d, day_query, now):
                hits.append((event_label, d))
        if hits:
            labels = [f"{event_label} on {_resolve_day_to_date(day, now)}" for event_label, day in hits]
            return "You have " + " and ".join(labels) + "."

    # If user asks generally about schedule/important items without naming event.
    generic_intent = bool(
        day_query
        or asks_when
        or {"important", "schedule"}.intersection(msg_tokens)
        or {"class", "classes", "test", "exam", "event", "upcoming", "deadline"}.intersection(msg_tokens)
    )
    if generic_intent and latest:
        event_label, day = next(iter(latest.items()))
        return f"Your important {event_label} is on {_resolve_day_to_date(day, now)}."

    return None


def enforce_user_chat_token_budget(user_id: str):
    if MAX_USER_CHAT_TOKENS <= 0:
        return
    user_sessions = [s for s in chat_sessions.values() if s.user_id == user_id]
    if not user_sessions:
        return

    # Oldest chats are removed first.
    user_sessions.sort(key=lambda s: s.created_at)
    total_tokens = sum(_chat_token_count(s) for s in user_sessions)

    while len(user_sessions) > 1 and total_tokens > MAX_USER_CHAT_TOKENS:
        oldest = user_sessions.pop(0)
        total_tokens -= _chat_token_count(oldest)
        chat_sessions.pop(oldest.id, None)
        _delete_chat_session_storage(oldest.id)

    # If one chat is still above budget, trim oldest messages within it.
    if user_sessions and total_tokens > MAX_USER_CHAT_TOKENS:
        remaining = user_sessions[0]
        _trim_chat_to_budget(remaining, MAX_USER_CHAT_TOKENS)
        remaining.updated_at = datetime.now(timezone.utc)
        _persist_chat_session(remaining)


def load_chat_sessions():
    loaded = _load_chat_sessions_from_relational()
    if loaded > 0:
        return

    legacy = _load_legacy_chat_sessions_from_mongo()
    if not legacy:
        legacy = _load_legacy_chat_sessions_from_json()
    if not legacy:
        return

    for session in legacy:
        chat_sessions[session.id] = session
        _persist_chat_session(session)


def save_chat_sessions():
    for session in chat_sessions.values():
        _persist_chat_session(session)


def _load_chat_session_from_relational(chat_id: str) -> ChatSession | None:
    cid = str(chat_id or "").strip()
    if not cid:
        return None
    with get_relational_session() as db:
        row = db.query(DBChatSession).filter(DBChatSession.id == cid).first()
        if row is None:
            return None
        messages = (
            db.query(DBChatMessage)
            .filter(DBChatMessage.session_id == cid)
            .order_by(DBChatMessage.timestamp.asc())
            .all()
        )
    session = ChatSession(user_id=row.user_id)
    session.id = row.id
    session.title = row.title or "New Chat"
    session.created_at = row.created_at or datetime.now(timezone.utc)
    session.updated_at = row.updated_at or session.created_at
    session.messages = [
        {
            "role": m.role,
            "content": m.content,
            "timestamp": (m.timestamp or datetime.now(timezone.utc)).isoformat(),
        }
        for m in messages
        if m.role in {"user", "assistant"} and str(m.content or "").strip()
    ]
    return session


def _record_usage_log(user_id: str, session_id: str | None, usage: dict[str, Any], model_used: str | None = None):
    uid = (user_id or "").strip()
    if not uid:
        return
    try:
        input_tokens = int(usage.get("input_tokens_est") or 0)
        output_tokens = int(usage.get("output_tokens_est") or 0)
        cost_usd = float(usage.get("cost_est_usd") or 0.0)
        model_name = str(model_used or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

        with get_relational_session() as db:
            _ensure_user_row(uid, db=db)
            db.add(
                DBUsageLog(
                    user_id=uid,
                    session_id=(session_id or None),
                    model_used=model_name,
                    tokens_input=input_tokens,
                    tokens_output=output_tokens,
                    cost_usd=cost_usd,
                    created_at=datetime.now(timezone.utc),
                )
            )
    except Exception as exc:
        log_event("usage_log_write_failed", user_id=uid, error=str(exc))


def _get_or_create_user_settings(user_id: str, db) -> DBUserSetting:
    uid = (user_id or "").strip()
    row = db.query(DBUserSetting).filter(DBUserSetting.user_id == uid).first()
    if row is not None:
        return row
    _ensure_user_row(uid, db=db)
    row = DBUserSetting(
        user_id=uid,
        tone_preference="balanced",
        memory_enabled=True,
        reasoning_mode="standard",
        updated_at=datetime.now(timezone.utc),
    )
    db.add(row)
    return row


load_chat_sessions()


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    use_tools: bool = False
    scope: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    chat_id: str
    usage: Optional[dict[str, Any]] = None
    semantic_memories: Optional[list[dict[str, Any]]] = None


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class AgentChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None


class AgentChatResponse(BaseModel):
    reply: str
    tool_events: list[dict[str, Any]]
    usage: dict[str, Any]


class UserSettingsUpdate(BaseModel):
    tone_preference: Optional[str] = None
    memory_enabled: Optional[bool] = None
    reasoning_mode: Optional[str] = None


def _deterministic_memory_context(query: str, user_id: str) -> str:
    try:
        return retrieve_memories(query, user_id)
    except Exception:
        return ""


def _semantic_memory_context(user_id: str, query: str) -> tuple[list[dict[str, Any]], str]:
    service = get_semantic_memory_service()
    return service.retrieve_context(
        user_id=user_id,
        query=query,
        top_k=settings.semantic_top_k,
        scopes=["user", "project", "conversation", "global"],
    )


def _semantic_memory_node(memory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "memory_id": memory.id,
        "user_id": memory.user_id,
        "content": memory.content,
        "embedding": memory.embedding,
        "memory_type": memory.memory_type,
        "type": memory.memory_type,
        "importance_score": memory.importance_score,
        "reinforcement_count": memory.reinforcement_count,
        "created_at": memory.created_at.isoformat(),
        "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
        "decay_factor": memory.decay_factor,
        "scope": memory.scope,
        "source_message_id": memory.source_message_id,
        "metadata": memory.metadata or {},
        "is_active": memory.is_active,
        "is_archived": memory.is_archived,
        "archived_at": memory.archived_at.isoformat() if memory.archived_at else None,
    }


orchestrator = build_chat_orchestrator(
    deterministic_memory_fn=_deterministic_memory_context,
    semantic_retrieve_fn=_semantic_memory_context,
    recency_buffer_fn=_build_chat_history_context,
    realtime_fn=lambda msg: get_realtime_context(msg) or "",
    should_realtime_fn=lambda msg: bool(realtime_web_enabled and should_fetch_realtime(msg)),
    tool_agent_fn=run_agent_turn,
    sanitize_reply_fn=sanitize_response,
)
stream_handler = OrchestratorStreamHandler()


def detect_category_query(user_message: str) -> Optional[str]:
    msg = (user_message or "").lower()
    if not msg:
        return None
    is_query = "?" in msg or any(w in msg for w in ["what", "which", "show", "list", "tell me"])
    if not is_query:
        return None

    asks_top3 = ("top 3" in msg or "top3" in msg) and any(
        marker in msg for marker in ["category", "categories", "each", "every", "per"]
    )

    if any(p in msg for p in ["my preferences", "my preference", "preferences do i have", "all preferences"]):
        return "preference_by_category" if asks_top3 else "preference"
    if any(p in msg for p in ["my facts", "facts about me", "all facts", "my profile facts"]):
        return "fact_by_category" if asks_top3 else "fact"
    if any(p in msg for p in ["my constraints", "all constraints", "constraints i have"]):
        return "constraint_by_category" if asks_top3 else "constraint"
    if asks_top3:
        return "all_by_category"
    return None


def build_category_top3_reply(user_id: str, category: str) -> str:
    store = get_memory_store()
    type_filter = category
    by_category = False
    if category.endswith("_by_category"):
        by_category = True
        type_filter = category.replace("_by_category", "")
    elif category in {"preference", "fact", "constraint"}:
        # Default to per-category view for category queries.
        by_category = True

    memories = store.get_user_memories(user_id)
    if type_filter == "all":
        items = [m for m in memories if m.confidence >= 0.5]
    else:
        items = [m for m in memories if m.type == type_filter and m.confidence >= 0.5]

    if not items:
        label = "entries" if type_filter == "all" else f"{type_filter}s"
        return f"I do not have any stored {label} for you yet."

    if not by_category:
        top = sorted(items, key=lambda m: (m.confidence, m.last_updated.timestamp()), reverse=True)[:3]
        label = {"preference": "preferences", "fact": "facts", "constraint": "constraints", "all": "entries"}.get(type_filter, "entries")
        lines = [f"Top 3 {label} in your profile:"]
        for idx, m in enumerate(top, start=1):
            lines.append(f"{idx}. {m.key}: {m.value} (confidence {m.confidence:.2f})")
        return "\n".join(lines)

    grouped: dict[str, list] = {}
    for m in items:
        md = m.metadata or {}
        domain = str(md.get("domain") or "").strip().lower()
        cat = str(md.get("category") or "").strip().lower()
        group_key = f"{domain}.{cat}" if domain and cat else m.key
        grouped.setdefault(group_key, []).append(m)

    def _top_values(group_items: list) -> list:
        sorted_items = sorted(group_items, key=lambda x: (x.confidence, x.last_updated.timestamp()), reverse=True)
        seen_values = set()
        top_items = []
        for item in sorted_items:
            v = (item.value or "").strip().lower()
            if not v or v in seen_values:
                continue
            seen_values.add(v)
            top_items.append(item)
            if len(top_items) >= 3:
                break
        return top_items

    group_rows = []
    for group_key, group_items in grouped.items():
        top_items = _top_values(group_items)
        if not top_items:
            continue
        latest_ts = max(i.last_updated.timestamp() for i in top_items)
        value_text = ", ".join(f"{i.value} ({i.confidence:.2f})" for i in top_items)
        group_rows.append((latest_ts, f"{group_key}: {value_text}"))

    if not group_rows:
        return "I do not have enough categorized information yet."

    group_rows.sort(key=lambda x: x[0], reverse=True)
    label = {"preference": "preferences", "fact": "facts", "constraint": "constraints", "all": "memory entries"}.get(type_filter, "entries")
    lines = [f"Top 3 values per category for your {label}:"]
    for idx, (_, text) in enumerate(group_rows, start=1):
        lines.append(f"{idx}. {text}")
    return "\n".join(lines)


def handle_preference_query(user_message: str, user_id: str) -> Optional[str]:
    """
    Hybrid resolver for "my favorite <subject> [in/from/of <entity>]".
    1) Prefer scoped DB facts.
    2) Use global rules for inference when needed.
    3) If evidence is weak, return None so the main LLM flow can answer naturally.
    """
    subject_key, subject_label, entity_key = parse_preference_query(user_message)
    if not subject_key:
        return None

    store = get_memory_store()
    user_memories = [m for m in store.get_user_memories(user_id) if m.confidence >= 0.5]

    specific_values: list[str] = []
    global_values: list[str] = []
    for memory in user_memories:
        value = (memory.value or "").strip()
        if not value:
            continue

        is_specific, is_global = classify_memory_for_query(memory, subject_key, entity_key)
        if is_specific:
            specific_values.append(value)
        if is_global:
            global_values.append(value)

    seen = set()
    specific_values = [v for v in specific_values if not (v.lower() in seen or seen.add(v.lower()))]
    seen = set()
    global_values = [v for v in global_values if not (v.lower() in seen or seen.add(v.lower()))]

    subject_title = subject_label.strip().title()
    entity_title = entity_key.replace("_", " ").title() if entity_key else ""

    # Strongest path: exact stored value.
    if specific_values and entity_title:
        lead = f"For {subject_title} in {entity_title}, you prefer {specific_values[0]}."
        if global_values:
            return f"{lead} Broadly, your pattern is {global_values[0]}."
        return lead
    if specific_values:
        return f"Your {subject_title} preference is {specific_values[0]}."

    # Keep broad profile answer if user asked generally (no entity requested).
    if global_values and not entity_title:
        return f"Your broader {subject_title} preference rule is {global_values[0]}."

    # Weak evidence: no direct stored match for this preference query.
    return None


def handle_builtin_realtime_query(user_message: str) -> Optional[str]:
    """
    Deterministic realtime answers for simple date/time queries.
    This bypasses LLM uncertainty and guarantees correct current time reporting.
    """
    msg = (user_message or "").strip().lower()
    if not msg:
        return None

    asks_date = any(
        p in msg
        for p in [
            "current date",
            "today date",
            "what is the date",
            "what's the date",
            "today's date",
            "todays date",
            "date today",
            "data today",  # common typo for "date today"
        ]
    )
    asks_time = any(p in msg for p in ["current time", "what is the time", "what's the time", "time now", "right now time", "time today"])
    asks_day = any(p in msg for p in ["what day is it", "which day is today", "day today", "today is what day"])
    if not asks_date and "today" in msg and ("date" in msg or "data" in msg):
        asks_date = True
    if not (asks_date or asks_time or asks_day):
        return None

    now = datetime.now().astimezone()
    date_part = now.strftime("%A, %B %d, %Y")
    time_part = now.strftime("%I:%M:%S %p %Z")

    if asks_date and asks_time:
        return f"Right now it is {date_part}, {time_part}."
    if asks_date:
        return f"Today is {date_part}."
    if asks_time:
        return f"The current time is {time_part}."
    return f"Today is {date_part}."


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    """
    Main chat endpoint.
    1. Receive user message
    2. Build layered context (deterministic + semantic + recency + hints)
    3. Rank context and assemble prompt
    4. Invoke LLM through orchestrator abstraction
    5. Persist chat + schedule background memory hooks
    6. Return response + usage metadata
    """
    try:
        user_id = (x_user_id or "").strip()
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "X-User-ID header is required"})

        user_message = request.message.strip()
        validate_message_payload(user_message)

        store = get_memory_store()
        adjusted_count = store.apply_strong_feedback(user_id, user_message)
        if adjusted_count:
            print(f"Applied strong sentiment feedback to {adjusted_count} memory item(s)")

        chat_id = request.chat_id
        chat_session = None
        if chat_id:
            chat_session = chat_sessions.get(chat_id)
            if chat_session is None:
                chat_session = _load_chat_session_from_relational(chat_id)
                if chat_session is not None:
                    chat_sessions[chat_session.id] = chat_session

        if chat_session is None:
            chat_session = ChatSession(user_id=user_id)
            words = user_message.split()[:5]
            chat_session.title = " ".join(words) + ("..." if len(user_message.split()) > 5 else "")
            chat_sessions[chat_session.id] = chat_session
            chat_id = chat_session.id
            _persist_chat_session(chat_session)
        else:
            if chat_session.user_id != user_id:
                return JSONResponse(status_code=404, content={"error": "Chat not found"})
            chat_session.updated_at = datetime.now(timezone.utc)

        # Resolve short acknowledgements after chat context is known.
        resolved_user_message = _resolve_short_reply_context(user_message, chat_session)
        continuity_message = _augment_query_for_continuity(resolved_user_message, chat_session)
        is_ack_reply = _is_short_acknowledgement(user_message)
        brain_decision = brain_service.process(
            user_id=user_id,
            message=continuity_message,
            chat_session=chat_session,
        )

        # --------------------------------------------------------------
        # A. Handle explicit category-profile queries (top 3)
        # --------------------------------------------------------------
        category_query = None if (is_ack_reply or brain_decision.direct_response) else detect_category_query(continuity_message)
        category_hint = build_category_top3_reply(user_id, category_query) if category_query else None

        # --------------------------------------------------------------
        # A2. Deterministic realtime answer for date/time style queries
        # --------------------------------------------------------------
        builtin_realtime_hint = None if (is_ack_reply or brain_decision.direct_response) else handle_builtin_realtime_query(continuity_message)

        # --------------------------------------------------------------
        # A2b. Deterministic schedule answer (tests/classes + day)
        # --------------------------------------------------------------
        schedule_hint = None if (is_ack_reply or brain_decision.direct_response) else handle_schedule_query(continuity_message, user_id)

        # --------------------------------------------------------------
        # A3. Deterministic preference reasoning (domain-agnostic)
        # --------------------------------------------------------------
        preference_hint = None if (is_ack_reply or brain_decision.direct_response) else handle_preference_query(continuity_message, user_id)

        deterministic_hints = [
            h
            for h in (
                list(brain_decision.deterministic_hints)
                + [category_hint, builtin_realtime_hint, schedule_hint, preference_hint]
            )
            if h
        ]

        if brain_decision.direct_response:
            reply = sanitize_response(continuity_message, brain_decision.direct_response)
            input_tokens = estimate_tokens(continuity_message)
            output_tokens = estimate_tokens(reply)
            usage = {
                "input_tokens_est": input_tokens,
                "output_tokens_est": output_tokens,
                "cost_est_usd": estimate_cost_usd(input_tokens, output_tokens),
                "llm_latency_ms": 0.0,
                "path": "brain_direct",
                "intent": brain_decision.intent.value,
            }
            semantic_rows = []
            tool_events = []
            metrics.inc("llm_tokens_input_total", input_tokens)
            metrics.inc("llm_tokens_output_total", output_tokens)
            metrics.inc("llm_cost_usd_total", usage["cost_est_usd"])
            usage["tool_events"] = tool_events
        else:
            orchestrator_payload = OrchestratorInput(
                user_id=user_id,
                chat_id=chat_id,
                user_message=user_message,
                continuity_message=continuity_message,
                use_tools=bool(request.use_tools and settings.enable_tools),
                scope=request.scope,
                response_style=brain_decision.response_style,
            )
            pipeline_result = orchestrator.run(
                payload=orchestrator_payload,
                chat_session=chat_session,
                deterministic_hints=deterministic_hints,
            )
            reply = pipeline_result.reply
            usage = pipeline_result.usage
            semantic_rows = pipeline_result.semantic_rows
            tool_events = pipeline_result.tool_events
            usage["tool_events"] = tool_events
            usage["intent"] = brain_decision.intent.value

        # --------------------------------------------------------------
        # E. AFTER response is generated, extract any new memory
        #    so the LLM doesn't see what it just created on this turn.
        # --------------------------------------------------------------
        try:
            memory_extracted = extract_memory(continuity_message, user_id)
            if memory_extracted:
                print("=== NEW MEMORY EXTRACTED AFTER RESPONSE ===")
                print(f"{memory_extracted.key}: {memory_extracted.value} (conf={memory_extracted.confidence:.2f})")
                print("=== END NEW MEMORY ===")
            else:
                print("INFO: No memory extracted from this message")
        except Exception as e:
            print(f"WARNING: Memory extraction error (non-fatal): {e}")

        if settings.enable_background_tasks and settings.enable_semantic_memory:
            background_tasks.add_task(
                enqueue_ingest_message,
                user_id,
                continuity_message,
                "",
                request.scope,
            )

        chat_session.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        chat_session.messages.append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        _persist_chat_session(chat_session)
        enforce_user_chat_token_budget(user_id)
        _record_usage_log(
            user_id=user_id,
            session_id=chat_id,
            usage=usage,
            model_used=str(usage.get("model") or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
        )

        log_event(
            "chat_completed",
            request_id=getattr(http_request.state, "request_id", ""),
            user_id=user_id,
            chat_id=chat_id,
            semantic_memories=len(semantic_rows),
            input_tokens=usage.get("input_tokens_est", 0),
            output_tokens=usage.get("output_tokens_est", 0),
            tool_events=len(tool_events),
        )
        return ChatResponse(reply=reply, chat_id=chat_id, usage=usage, semantic_memories=semantic_rows)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in /chat: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal error: {str(e)}"})


@app.get("/")
async def root():
    index_path = project_root / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(
            str(index_path),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    return {
        "message": "AI Chat with Memory API",
        "status": "running",
        "endpoints": {"chat": "POST /chat", "health": "GET /health"},
        "api_key_loaded": api_key_loaded
    }


@app.get("/script.js")
async def serve_script():
    script_path = project_root / "frontend" / "script.js"
    if script_path.exists():
        return FileResponse(
            str(script_path),
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    raise HTTPException(status_code=404, detail="script.js not found")


@app.get("/app.jsx")
async def serve_app():
    app_path = project_root / "frontend" / "app.jsx"
    if app_path.exists():
        return FileResponse(
            str(app_path),
            media_type="application/javascript",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    raise HTTPException(status_code=404, detail="app.jsx not found")


@app.get("/chats", response_model=List[ChatSessionResponse])
async def get_chats(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    with get_relational_session() as db:
        rows = (
            db.query(DBChatSession)
            .filter(DBChatSession.user_id == user_id)
            .order_by(DBChatSession.updated_at.desc())
            .all()
        )
        sessions = []
        for row in rows:
            message_count = (
                db.query(DBChatMessage)
                .filter(DBChatMessage.session_id == row.id)
                .count()
            )
            sessions.append(
                ChatSessionResponse(
                    id=row.id,
                    title=row.title or "New Chat",
                    created_at=(row.created_at or datetime.now(timezone.utc)).isoformat(),
                    updated_at=(row.updated_at or datetime.now(timezone.utc)).isoformat(),
                    message_count=message_count,
                )
            )
    return sessions


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str, x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    session = chat_sessions.get(chat_id) or _load_chat_session_from_relational(chat_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat_sessions[session.id] = session
    return {
        "id": session.id,
        "user_id": session.user_id,
        "title": session.title,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "messages": session.messages
    }


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    session = chat_sessions.get(chat_id) or _load_chat_session_from_relational(chat_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat_sessions.pop(chat_id, None)
    _delete_chat_session_storage(chat_id)
    return {"status": "deleted", "chat_id": chat_id}


@app.post("/chats/new")
async def create_new_chat(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    chat_session = ChatSession(user_id=user_id)
    chat_sessions[chat_session.id] = chat_session
    _persist_chat_session(chat_session)
    enforce_user_chat_token_budget(user_id)
    return ChatSessionResponse(
        id=chat_session.id,
        title=chat_session.title,
        created_at=chat_session.created_at.isoformat(),
        updated_at=chat_session.updated_at.isoformat(),
        message_count=0
    )


@app.get("/settings")
async def get_user_settings(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    with get_relational_session() as db:
        row = _get_or_create_user_settings(user_id=user_id, db=db)
        return {
            "user_id": user_id,
            "tone_preference": row.tone_preference,
            "memory_enabled": bool(row.memory_enabled),
            "reasoning_mode": row.reasoning_mode,
            "updated_at": (row.updated_at or datetime.now(timezone.utc)).isoformat(),
        }


@app.post("/settings")
async def update_user_settings(
    payload: UserSettingsUpdate,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    with get_relational_session() as db:
        row = _get_or_create_user_settings(user_id=user_id, db=db)
        if payload.tone_preference is not None:
            row.tone_preference = str(payload.tone_preference).strip() or "balanced"
        if payload.memory_enabled is not None:
            row.memory_enabled = bool(payload.memory_enabled)
        if payload.reasoning_mode is not None:
            row.reasoning_mode = str(payload.reasoning_mode).strip() or "standard"
        row.updated_at = datetime.now(timezone.utc)
        return {
            "user_id": user_id,
            "tone_preference": row.tone_preference,
            "memory_enabled": bool(row.memory_enabled),
            "reasoning_mode": row.reasoning_mode,
            "updated_at": row.updated_at.isoformat(),
        }


@app.get("/memories")
async def get_memories(x_user_id: str = Header(..., alias="X-User-ID")):
    from memory.memory_store import get_memory_store
    store = get_memory_store()
    user_id = (x_user_id or "").strip()
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "X-User-ID header is required"})
    memories = store.get_user_memories(user_id)
    return {
        "user_id": user_id,
        "total": len(memories),
        "memories": [
            {
                "id": m.id,
                "user_id": m.user_id,
                "type": m.type,
                "key": m.key,
                "value": m.value,
                "confidence": m.confidence,
                "created_at": m.created_at.isoformat(),
                "last_updated": m.last_updated.isoformat()
            }
            for m in memories
        ]
    }


@app.get("/memories/structured")
async def get_structured_memories(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "X-User-ID header is required"})
    return brain_service.store.export_user_context(user_id)


@app.get("/memories/semantic")
async def get_semantic_memories(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "X-User-ID header is required"})
    service = get_semantic_memory_service()
    memories = service.list_user_memories(user_id)
    return {
        "user_id": user_id,
        "total": len(memories),
        "memories": [_semantic_memory_node(m) for m in memories],
    }


@app.delete("/memories/semantic/{memory_id}")
async def delete_semantic_memory(memory_id: str, x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "X-User-ID header is required"})
    service = get_semantic_memory_service()
    deleted = service.delete_user_memory(user_id=user_id, memory_id=memory_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"error": "Semantic memory not found"})
    return {"status": "deleted", "memory_id": memory_id}


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    raw_request: Request,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    if not settings.enable_streaming:
        return JSONResponse(status_code=400, content={"error": "Streaming is disabled"})
    request_id = getattr(raw_request.state, "request_id", "")
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    try:
        result = await chat(request=request, background_tasks=background_tasks, http_request=raw_request, x_user_id=x_user_id)
        if isinstance(result, JSONResponse):
            return result
        tool_events = result.usage.get("tool_events", []) if isinstance(result.usage, dict) else []
        return StreamingResponse(
            stream_handler.stream(
                text=result.reply,
                request_id=request_id,
                chat_id=result.chat_id,
                usage=result.usage or {},
                tool_events=tool_events,
                is_disconnected=raw_request.is_disconnected,
            ),
            media_type="text/event-stream",
            headers=headers,
        )
    except Exception as exc:
        return StreamingResponse(
            stream_handler.stream_error(str(exc), request_id=request_id),
            media_type="text/event-stream",
            headers=headers,
        )


@app.post("/chat/agent", response_model=AgentChatResponse)
async def chat_agent(
    request: AgentChatRequest,
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    if not settings.enable_tools:
        raise HTTPException(status_code=400, detail="Tool execution is disabled")
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    message = (request.message or "").strip()
    validate_message_payload(message)
    started = time.perf_counter()
    result = run_agent_turn(message)
    reply = sanitize_response(message, result.get("reply", ""))
    input_tokens = estimate_tokens(message)
    output_tokens = estimate_tokens(reply)
    metrics.inc("llm_tokens_input_total", input_tokens)
    metrics.inc("llm_tokens_output_total", output_tokens)
    usage = {
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "cost_est_usd": estimate_cost_usd(input_tokens, output_tokens),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    metrics.observe("llm_latency_seconds", usage["latency_ms"] / 1000.0)
    metrics.inc("llm_cost_usd_total", usage["cost_est_usd"])
    if settings.enable_background_tasks and settings.enable_semantic_memory:
        background_tasks.add_task(enqueue_ingest_message, user_id, message, "", "user")
    _record_usage_log(
        user_id=user_id,
        session_id=request.chat_id,
        usage=usage,
        model_used=str(usage.get("model") or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")),
    )
    return AgentChatResponse(reply=reply, tool_events=result.get("tool_events") or [], usage=usage)


@app.get("/tools")
async def list_tools():
    from backend.app.tools.registry import tool_registry

    return {"tools": tool_registry.list_tools()}


@app.post("/admin/semantic/decay")
async def run_semantic_decay(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    return run_decay(user_id)


@app.post("/admin/semantic/compress")
async def run_semantic_compression(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    return run_compression(user_id)


@app.post("/admin/semantic/reembed")
async def run_semantic_reembedding(
    payload: dict,
    x_user_id: str = Header(..., alias="X-User-ID"),
):
    user_id = str((payload or {}).get("user_id") or (x_user_id or "")).strip()
    reason = str((payload or {}).get("reason") or "model_update").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    return run_reembedding(user_id=user_id, reason=reason)


@app.get("/metrics")
async def get_metrics():
    payload, content_type = metrics.export()
    return PlainTextResponse(content=payload, media_type=content_type)


@app.get("/health")
async def health():
    vector_backend = "disabled"
    if settings.enable_semantic_memory:
        try:
            vector_backend = get_semantic_memory_service().store.__class__.__name__
        except Exception as exc:
            vector_backend = f"error:{type(exc).__name__}"
    return {
        "status": "ok",
        "api_key_loaded": api_key_loaded,
        "truth_db": "relational_sqlalchemy",
        "vector_store_backend": vector_backend,
        "semantic_memory_enabled": settings.enable_semantic_memory,
        "streaming_enabled": settings.enable_streaming,
        "tools_enabled": settings.enable_tools,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
