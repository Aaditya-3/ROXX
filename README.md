# AI Chat System with Long-Term Memory

A complete chat application powered by Groq with persistent memory capabilities. The system remembers user preferences, constraints, and facts across conversations.

## Architecture

```
Frontend (React 18 + Tailwind CSS + Babel JSX)
        ->
FastAPI backend (Uvicorn)
        ->
Chat Orchestrator
  - deterministic memory retrieval
  - semantic memory retrieval (vector search)
  - optional tool routing
  - optional realtime context fetch
        ->
Groq LLM (`llama-3.1-8b-instant` by default)
        ->
Streaming + response persistence
        ->
Background semantic memory ingestion + decay/compression hooks
```

## Tech Stack (Current)

- Backend: Python, FastAPI, Uvicorn, Pydantic v2, python-dotenv, SQLAlchemy
- LLM: Groq Python SDK (`groq`) with model from `GROQ_MODEL` (default `llama-3.1-8b-instant`)
- Data: relational truth DB for users/sessions/messages/usage/settings (`DATABASE_URL`), semantic vector store in Qdrant (`QDRANT_URL` + `QDRANT_API_KEY`), optional local vector fallback
- Auth/Security: username/password auth (PBKDF2-HMAC with `hashlib` + `hmac`), JWT via `python-jose`, CORS middleware, request-id + security headers + prompt guard + rate limiting
- Realtime/HTTP: `httpx` for realtime web lookups and Google token verification
- Observability: structured app logs + Prometheus-style metrics endpoint (`/metrics`)
- Background tasks: FastAPI `BackgroundTasks`, optional Celery/Redis worker scaffolding
- Frontend: React 18 + ReactDOM 18 (CDN), Tailwind CSS (CDN), Babel Standalone for in-browser JSX transpile
- Serving: FastAPI serves `frontend/index.html`, `frontend/app.jsx`, and static assets
- External APIs: Groq API, Frankfurter currency API, DuckDuckGo Instant Answer API, Google `tokeninfo` endpoint (optional)

## Project Structure

```
project_root/
|-- backend/
|   |-- main.py                         # FastAPI app entrypoint
|   `-- app/
|       |-- core/                       # config + middleware + auth/db/llm/tool primitives
|       |-- observability/              # logging + metrics
|       |-- embeddings/                 # embedding provider abstraction
|       |-- vector_store/               # Qdrant/local vector repository
|       |-- memory/                     # semantic memory models
|       |-- services/                   # semantic retrieval, streaming, tools
|       |-- tools/                      # calculator, currency, web search
|       `-- tasks/                      # background memory tasks + Celery wiring
|-- memory/
|   |-- memory_schema.py
|   |-- memory_store.py
|   |-- memory_extractor.py
|   |-- memory_retriever.py
|   |-- memories.json                   # default memory persistence
|   |-- app.db                          # relational truth DB (SQLite default)
|   `-- semantic_memories.json          # semantic vector fallback storage
|-- frontend/
|   |-- index.html
|   |-- app.jsx
|   `-- script.js
|-- frontend-vite/                     # production frontend build scaffold
|   |-- package.json
|   |-- vite.config.js
|   `-- src/
|-- requirements.txt
|-- QUICKSTART.md
`-- README.md
```

## Production Features Implemented

- Semantic memory ingestion with embeddings and vector retrieval
- Vector backend abstraction: Qdrant (if configured) or local JSON fallback
- Ranked memory retrieval with similarity + importance + recency weighting
- Background semantic ingestion hooks via FastAPI `BackgroundTasks`
- Memory lifecycle utilities: decay and compression endpoints
- SSE streaming endpoint for token-by-token frontend rendering
- Tool registry and agentic chat endpoint (`/chat/agent`) with structured tool execution
- Observability improvements: structured request logs + `/metrics` endpoint
- Security hardening baseline: rate limiting, request-id propagation, security headers, prompt-injection guard
- Frontend memory inspector panel with semantic memory delete action

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

**CRITICAL:** You must create a `.env` file before starting the backend.

1. Copy `.env.example` to `.env`:
   ```bash
   # Windows
   copy .env.example .env
   
   # Linux/Mac
   cp .env.example .env
   ```

2. Add your Groq API key in `.env`:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```

   Get your API key from: https://console.groq.com

3. **Verify the `.env` file exists** in the project root (same directory as `requirements.txt`)

### 3. Run Backend

**Backend MUST be started from project root.** The `.env` file is loaded from the current working directory at startup.

From the project root directory (where `requirements.txt` and `.env` are located):

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Or with auto-reload:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will start on `http://localhost:8000`

**Verify Setup:**
When the backend starts, you must see:
```
GROQ_API_KEY loaded: True
```
If you see `GROQ_API_KEY loaded: False`, the `.env` file was not found (wrong directory or missing file).

**If you see `GROQ_API_KEY loaded: False`:**
1. **Create the `.env` file** (if you haven't already):
   ```bash
   python setup_env.py
   # OR manually: copy .env.example .env
   ```
2. **Restart the backend** - The `.env` file is only loaded at startup
3. Verify the `.env` file exists in the project root (same folder as `requirements.txt`)
4. Check the file contains `GROQ_API_KEY=...` (no spaces around `=`)
5. Test environment loading: `python test_env.py`

### 4. Open Frontend

The backend now serves the frontend automatically! Simply open your browser and go to:

```
http://localhost:8000
```

The frontend will be served from the root URL, and all API calls will work automatically.

**Note:** If you prefer to serve the frontend separately, you can still open `frontend/index.html` directly in your browser, but you'll need to update the `API_URL` in `script.js` back to `http://localhost:8000/chat`.

### 5. Optional Production Frontend (Vite)

```bash
cd frontend-vite
npm install
npm run dev
```

Vite dev server starts on `http://localhost:5173` and proxies API calls to `http://localhost:8000`.

## Key Environment Variables

```env
# truth database
DATABASE_URL=sqlite:///memory/app.db

# semantic memory
ENABLE_SEMANTIC_MEMORY=true
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=bge-small-en-v1.5
EMBEDDING_DIMS=384
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=mnemos_semantic_memory
REQUIRE_QDRANT=false

# streaming/tools/background
ENABLE_STREAMING=true
ENABLE_TOOLS=true
ENABLE_BACKGROUND_TASKS=true

# security/guardrails
RATE_LIMIT_RPM=60
ENABLE_PROMPT_GUARD=true
MAX_PROMPT_CHARS=6000

# observability
LOG_LEVEL=INFO
LLM_COST_INPUT_PER_1K=0
LLM_COST_OUTPUT_PER_1K=0
```

## Memory System

### What Gets Stored

The system extracts and stores **long-term, reusable information**:

- [x] **Preferences**: Language, style, format preferences
- [x] **Constraints**: Time limits, budget, availability
- [x] **Stable Facts**: Name, location, role, skills

### What Doesn't Get Stored

- [ ] Greetings (hello, hi, thanks)
- [ ] Emotions (happy, sad, excited)
- [ ] One-off messages
- [ ] Temporary states

### Memory Confidence

- **Initial confidence**: 0.7
- **Confirmation** (same info repeated): +0.1
- **Contradiction** (different value): -0.3
- **Deletion**: Memories below 0.3 confidence are removed

## Example Demo Conversation

```
User: Hi, I'm John and I prefer Python for coding.

Assistant: Hello John! I'll remember that you prefer Python for coding.

User: I'm working on a web project with a tight deadline.

Assistant: I understand you're working on a web project with a tight deadline. 
            I'll keep that in mind. Since you prefer Python, are you using 
            frameworks like Flask or FastAPI?

User: What's my preferred programming language?

Assistant: Your preferred programming language is Python.
```

## Security

- [x] API keys stored in `.env` (never committed)
- [x] No API keys in frontend code
- [x] Backend handles all API calls
- [x] CORS enabled for development

## API Endpoints

### POST `/chat`

Send a message to the AI.

**Request:**
```json
{
  "message": "Your message here",
  "chat_id": "optional_chat_id",
  "use_tools": false,
  "scope": "optional_scope"
}
```

**Response:**
```json
{
  "reply": "AI response here",
  "chat_id": "chat-id",
  "usage": {
    "input_tokens_est": 120,
    "output_tokens_est": 60,
    "cost_est_usd": 0.0
  },
  "semantic_memories": []
}
```

### POST `/chat/stream`

Returns `text/event-stream` with events:

- `start` (contains `chat_id`, `usage`)
- `token` (partial text chunks)
- `done`

### POST `/chat/agent`

Tool-enabled route for structured tool usage (`calculator`, `currency_convert`, `web_search`).

### GET `/tools`

List tool schemas exposed by the tool registry.

### GET `/settings` and POST `/settings`

Read/update persisted user settings from relational DB (`tone_preference`, `memory_enabled`, `reasoning_mode`).

### GET `/memories/semantic`

List semantic memory records for the current `X-User-ID`.

### DELETE `/memories/semantic/{memory_id}`

Delete one semantic memory record for the current `X-User-ID`.

### POST `/admin/semantic/decay`

Apply time-based importance decay for one user.

### POST `/admin/semantic/compress`

Run semantic memory compression for one user.

### GET `/metrics`

Prometheus-style metrics for request volume, latency, embedding time, retrieval time, and token estimates.

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "api_key_loaded": true,
  "truth_db": "relational_sqlalchemy",
  "vector_store_backend": "QdrantVectorStoreRepository",
  "semantic_memory_enabled": true,
  "streaming_enabled": true,
  "tools_enabled": true
}
```

## Notes

- Users/chat history/usage/settings persistence defaults to relational SQL (`DATABASE_URL`, SQLite at `memory/app.db` by default)
- Semantic memory fallback persistence is `memory/semantic_memories.json`
- Optional strict Qdrant mode can be enabled with `REQUIRE_QDRANT=true`
- Backend never crashes - all errors returned as JSON
- Uses Groq `llama-3.1-8b-instant` model for fast responses

## Design Decisions and Tradeoffs

- Qdrant is the preferred semantic backend, but the app keeps a JSON fallback so local dev works without infrastructure.
- Streaming currently uses SSE with chunked tokens emitted from the generated response path.
- Background processing is enabled with FastAPI `BackgroundTasks`; Celery/Redis wiring is included for production worker migration.
- Semantic ranking prioritizes practical relevance: similarity, importance score, and recency weighting.
- Security controls are lightweight by default (rate limit + prompt guard + headers) and can be tightened in gateway/WAF layers.

## Deployment Guide

1. Run API server (`uvicorn backend.main:app`) behind a reverse proxy.
2. Configure environment variables for Groq, semantic memory, and security knobs.
3. Provision Redis and Qdrant for production throughput.
4. Optionally run Celery worker/beat for scheduled decay/compression.
5. Expose `/metrics` to Prometheus and monitor latency + token usage.

## Scaling Guide

- Increase API replicas horizontally behind load balancing.
- Use Postgres/MySQL via `DATABASE_URL` for durable distributed truth state and Qdrant for semantic memory.
- Offload embedding/decay/compression work to Celery workers.
- Apply stricter rate limits and per-user quotas under higher concurrency.

## Refined Orchestrator Diagram

```text
User Input
-> Context Builder
   - deterministic memory
   - semantic vector retrieval
   - recency buffer
   - tool hints
-> Context Ranking Layer
-> Prompt Assembly Layer
-> LLM Invocation (timeout + retry policy)
-> Stream Handler (SSE: token/tool_call/done/error)
-> Persistence Layer (chat + memory updates)
-> Background Hooks (ingest/decay/compress/re-embed)
```

## Semantic Memory Node Schema

```json
{
  "memory_id": "uuid",
  "user_id": "string",
  "content": "string",
  "embedding": [0.0],
  "type": "fact|preference|emotional|goal|project|transient",
  "importance_score": 0.0,
  "reinforcement_count": 0,
  "created_at": "timestamp",
  "last_accessed": "timestamp|null",
  "decay_factor": 0.0,
  "scope": "global|user|conversation|project",
  "source_message_id": "string",
  "metadata": {}
}
```

## Internal Service Boundaries

- `app/orchestrator`: context builder, ranker, prompt assembler, pipeline, stream handler
- `app/memory`: memory models + semantic memory service + maintenance lifecycle
- `app/embeddings`: embedding provider abstraction and client adapters
- `app/vectorstore`: vector backend adapters (Qdrant + local fallback)
- `app/tools`: tool registry, schema validation, sandbox checks, executors
- `app/llm`: LLM client adapter with retry/timeout isolation
- `app/security`: replay protection, token rotation strategy
- `app/observability`: structured logs, metrics, optional tracing
- `app/tasks`: async ingestion/decay/compress/re-embedding hooks + Celery wiring

## Ranking and Decay Configuration

- Retrieval score:
  - `final_score = similarity*W1 + importance*W2 + recency*W3`
  - env vars: `MEM_RANK_WEIGHT_SIMILARITY`, `MEM_RANK_WEIGHT_IMPORTANCE`, `MEM_RANK_WEIGHT_RECENCY`
- Decay:
  - periodic multiplicative decay: `importance_score *= decay_factor`
  - archive threshold: `MEMORY_ARCHIVE_THRESHOLD`
  - delete threshold: `MEMORY_DELETE_THRESHOLD`

## Suggested Libraries

- API/runtime: FastAPI, Uvicorn, Pydantic v2
- Queue/workers: Celery + Redis
- Vector DB: Qdrant client
- Metrics: prometheus-client
- Tracing (optional): opentelemetry
- Embeddings: OpenAI or sentence-transformers local models

## Risks and Tradeoffs

- Local fallback mode improves DX but is not suitable for multi-node consistency.
- In-memory replay/rate-limit state is lightweight but should move to Redis for distributed deployments.
- LLM-based compression improves semantic quality but adds cost and background latency.
- Tool routing via LLM is flexible, but deterministic allowlists and strict schema validation remain mandatory.

## Troubleshooting

### GROQ_API_KEY loaded: False

**This is the most common issue. Fix it by:**

1. **Verify `.env` file exists:**
   ```bash
   # From project root, check if .env exists
   dir .env        # Windows
   ls -la .env     # Linux/Mac
   ```

2. **Check `.env` file format:**
   - Must be in project root (same folder as `requirements.txt`)
   - Must contain exactly: `GROQ_API_KEY=your_key_here`
   - No spaces around `=`
   - No quotes around the key
   - No trailing spaces

3. **Verify you're running from project root:**
   ```bash
   # Make sure you're in the directory with requirements.txt
   pwd              # Linux/Mac
   cd               # Windows (shows current directory)
   ```

4. **Restart the backend** after creating/editing `.env`

### Backend won't start:
- Check that `GROQ_API_KEY` is set in `.env`
- Verify Python dependencies are installed: `pip install -r requirements.txt`
- Check port 8000 is not in use
- Ensure you're running from project root directory

### Frontend can't connect:
- Ensure backend is running on `http://localhost:8000`
- Check browser console (F12) for CORS errors
- Update `API_URL` in `script.js` if backend is on different port
- Try accessing `http://localhost:8000/health` directly in browser

### No memory being stored:
- Memory extraction is automatic and optional
- Try explicit statements like "I prefer X" or "My name is Y"
- Check backend logs for extraction errors (non-fatal errors are logged but don't crash)
- Memory requires confidence > 0.5 to be retrieved

### Groq API errors:
- Verify your API key is valid
- Check your internet connection
- Ensure you have API quota remaining
- Check backend logs for specific error messages
