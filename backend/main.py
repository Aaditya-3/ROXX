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
print("GROQ_API_KEY loaded:", bool(os.getenv("GROQ_API_KEY")))

from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid

from memory.memory_extractor import extract_memory
from memory.memory_retriever import retrieve_memories
from memory.memory_store import get_memory_store
from backend.app.core.llm.groq_client import generate_response

project_root = _project_root
api_key_loaded = bool(os.getenv("GROQ_API_KEY"))
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
            "sure,", "of course,", "i can see why", "i understand", "eh?"
        ]
        for prefix in greeting_prefixes:
            if lower.startswith(prefix):
                # Remove the prefix from the original text, not just lower
                text = text[len(prefix):].lstrip()
                lower = text.lower()

    return text


from backend.app.core.auth.simple_auth import router as simple_auth_router


app = FastAPI(title="AI Chat with Memory")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = project_root / "frontend"
if frontend_path.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_path)), name="frontend")

# Mount auth router
app.include_router(simple_auth_router)


class ChatSession:
    def __init__(self, user_id: str):
        self.id: str = str(uuid.uuid4())
        self.user_id: str = user_id
        self.title: str = "New Chat"
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()
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
        session = cls(user_id=data.get("user_id", "guest"))
        session.id = data["id"]
        session.title = data.get("title", "New Chat")
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session.messages = data.get("messages", [])
        return session


chat_sessions: dict[str, ChatSession] = {}
current_chat_id: Optional[str] = None
chat_storage_path = project_root / "memory" / "chat_sessions.json"


def load_chat_sessions():
    if not chat_storage_path.exists():
        return
    try:
        with open(chat_storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                session = ChatSession.from_dict(item)
                chat_sessions[session.id] = session
    except Exception as e:
        print(f"Warning: failed to load chat sessions: {e}")


def save_chat_sessions():
    try:
        chat_storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [session.to_dict() for session in chat_sessions.values()]
        with open(chat_storage_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save chat sessions: {e}")


load_chat_sessions()


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    chat_id: str


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


def detect_category_query(user_message: str) -> Optional[str]:
    msg = (user_message or "").lower()
    if not msg:
        return None
    is_query = "?" in msg or any(w in msg for w in ["what", "which", "show", "list", "tell me"])
    if not is_query:
        return None

    if any(p in msg for p in ["my preferences", "my preference", "preferences do i have", "all preferences"]):
        return "preference"
    if any(p in msg for p in ["my facts", "facts about me", "all facts", "my profile facts"]):
        return "fact"
    if any(p in msg for p in ["my constraints", "all constraints", "constraints i have"]):
        return "constraint"
    return None


def build_category_top3_reply(user_id: str, category: str) -> str:
    store = get_memory_store()
    store.ensure_bootstrap_memories(user_id)
    items = [m for m in store.get_user_memories(user_id) if m.type == category]
    items = sorted(items, key=lambda m: (m.confidence, m.last_updated.timestamp()), reverse=True)[:3]
    if not items:
        return f"I do not have any stored {category}s for you yet."

    label = {"preference": "preferences", "fact": "facts", "constraint": "constraints"}[category]
    lines = [f"Top 3 {label} in your profile:"]
    for idx, m in enumerate(items, start=1):
        lines.append(f"{idx}. {m.key}: {m.value} (confidence {m.confidence:.2f})")
    return "\n".join(lines)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, x_user_id: str = Header(..., alias="X-User-ID")):
    """
    Main chat endpoint.
    1. Receive user message
    2. Extract memory if present
    3. Retrieve relevant memories
    4. Build single combined prompt (memory + user message)
    5. Call Groq generate_response(prompt)
    6. Store messages and return response
    """
    try:
        user_id = (x_user_id or "").strip()
        if not user_id:
            return JSONResponse(status_code=400, content={"error": "X-User-ID header is required"})

        user_message = request.message.strip()
        if not user_message:
            return JSONResponse(status_code=400, content={"error": "Message cannot be empty"})

        store = get_memory_store()
        store.ensure_bootstrap_memories(user_id)
        adjusted_count = store.apply_strong_feedback(user_id, user_message)
        if adjusted_count:
            print(f"Applied strong sentiment feedback to {adjusted_count} memory item(s)")

        chat_id = request.chat_id
        if not chat_id or chat_id not in chat_sessions:
            chat_session = ChatSession(user_id=user_id)
            words = user_message.split()[:5]
            chat_session.title = " ".join(words) + ("..." if len(user_message.split()) > 5 else "")
            chat_sessions[chat_session.id] = chat_session
            chat_id = chat_session.id
            save_chat_sessions()
        else:
            chat_session = chat_sessions[chat_id]
            if chat_session.user_id != user_id:
                return JSONResponse(status_code=404, content={"error": "Chat not found"})
            chat_session.updated_at = datetime.now()

        # --------------------------------------------------------------
        # A. Handle explicit category-profile queries (top 3)
        # --------------------------------------------------------------
        category_query = detect_category_query(user_message)
        if category_query:
            reply = build_category_top3_reply(user_id, category_query)
            chat_session.messages.append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            chat_session.messages.append({
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.now().isoformat()
            })
            save_chat_sessions()
            return ChatResponse(reply=reply, chat_id=chat_id)

        # --------------------------------------------------------------
        # B. Retrieve existing memories FIRST (no new extraction yet)
        # --------------------------------------------------------------
        memory_context = ""
        try:
            memory_context = retrieve_memories(user_message, user_id)
            if memory_context:
                print("=== MEMORY CONTEXT USED FOR THIS TURN ===")
                print(memory_context)
                print("=== END MEMORY CONTEXT ===")
            else:
                print("ℹ No memories to include in context")
        except Exception as e:
            print(f"⚠ Memory retrieval error (non-fatal): {e}")
            memory_context = ""

        # --------------------------------------------------------------
        # C. Build ONE combined prompt and call Groq
        #    (LLM only sees existing memories, not newly extracted ones)
        # --------------------------------------------------------------
        system_instructions = (
            "You are a grounded assistant. Refer ONLY to the provided [USER PROFILE] for personal details. "
            "If a detail is not in the profile, do not invent it. If you are unsure of the user's name or "
            "preferences, ask rather than assuming.\n"
            f"You are talking to {user_id}. Keep a warm, precise tone.\n"
            "Keep responses concise: 2-5 sentences unless the user asks for depth.\n"
        )

        full_prompt = f"{system_instructions}\nUser: {user_message}"
        if memory_context:
            full_prompt = f"""{system_instructions}
[USER PROFILE]
{memory_context}

User message:
{user_message}"""

        try:
            reply = generate_response(full_prompt)
            reply = sanitize_response(user_message, reply)
        except RuntimeError as e:
            return JSONResponse(status_code=500, content={"error": f"AI service error: {str(e)}"})

        # --------------------------------------------------------------
        # D. AFTER response is generated, extract any new memory
        #    so the LLM doesn't see what it just created on this turn.
        # --------------------------------------------------------------
        try:
            memory_extracted = extract_memory(user_message, user_id)
            if memory_extracted:
                print("=== NEW MEMORY EXTRACTED AFTER RESPONSE ===")
                print(f"{memory_extracted.key}: {memory_extracted.value} (conf={memory_extracted.confidence:.2f})")
                print("=== END NEW MEMORY ===")
            else:
                print("ℹ No memory extracted from this message")
        except Exception as e:
            print(f"⚠ Memory extraction error (non-fatal): {e}")

        chat_session.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        chat_session.messages.append({
            "role": "assistant",
            "content": reply,
            "timestamp": datetime.now().isoformat()
        })
        save_chat_sessions()

        return ChatResponse(reply=reply, chat_id=chat_id)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in /chat: {e}")
        return JSONResponse(status_code=500, content={"error": f"Internal error: {str(e)}"})


@app.get("/")
async def root():
    index_path = project_root / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
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
        return FileResponse(str(script_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found")


@app.get("/app.jsx")
async def serve_app():
    app_path = project_root / "frontend" / "app.jsx"
    if app_path.exists():
        return FileResponse(str(app_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.jsx not found")


@app.get("/chats", response_model=List[ChatSessionResponse])
async def get_chats(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    sessions = []
    for chat_id, session in sorted(chat_sessions.items(), key=lambda x: x[1].updated_at, reverse=True):
        if session.user_id != user_id:
            continue
        sessions.append(ChatSessionResponse(
            id=session.id,
            title=session.title,
            created_at=session.created_at.isoformat(),
            updated_at=session.updated_at.isoformat(),
            message_count=len(session.messages)
        ))
    return sessions


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: str, x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat not found")
    session = chat_sessions[chat_id]
    if session.user_id != user_id:
        raise HTTPException(status_code=404, detail="Chat not found")
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
    if chat_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Chat not found")
    if chat_sessions[chat_id].user_id != user_id:
        raise HTTPException(status_code=404, detail="Chat not found")
    del chat_sessions[chat_id]
    save_chat_sessions()
    return {"status": "deleted", "chat_id": chat_id}


@app.post("/chats/new")
async def create_new_chat(x_user_id: str = Header(..., alias="X-User-ID")):
    user_id = (x_user_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="X-User-ID header is required")
    chat_session = ChatSession(user_id=user_id)
    chat_sessions[chat_session.id] = chat_session
    save_chat_sessions()
    return ChatSessionResponse(
        id=chat_session.id,
        title=chat_session.title,
        created_at=chat_session.created_at.isoformat(),
        updated_at=chat_session.updated_at.isoformat(),
        message_count=0
    )


@app.get("/memories")
async def get_memories(x_user_id: str = Header(..., alias="X-User-ID")):
    from memory.memory_store import get_memory_store
    store = get_memory_store()
    user_id = (x_user_id or "").strip()
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "X-User-ID header is required"})
    store.ensure_bootstrap_memories(user_id)
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


@app.get("/health")
async def health():
    return {"status": "ok", "api_key_loaded": api_key_loaded}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
