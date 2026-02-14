# Quick Start Guide

## Get Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Create `.env`
```bash
# Option A
python setup_env.py

# Option B
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac
```

Set at minimum:
```env
GROQ_API_KEY=your_key_here
```

### 3. Start Backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open App
Go to:
```text
http://localhost:8000
```

Optional production frontend scaffold:
```bash
cd frontend-vite
npm install
npm run dev
```

## Semantic Memory + Streaming Defaults

```env
ENABLE_SEMANTIC_MEMORY=true
ENABLE_STREAMING=true
ENABLE_TOOLS=true
```

Optional vector backend:
```env
DATABASE_URL=sqlite:///memory/app.db
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=mnemos_semantic_memory
REQUIRE_QDRANT=false
```

## Useful Endpoints

- `POST /chat`
- `POST /chat/stream`
- `POST /chat/agent`
- `GET /tools`
- `GET /settings`
- `POST /settings`
- `GET /memories/semantic`
- `DELETE /memories/semantic/{memory_id}`
- `POST /admin/semantic/decay`
- `POST /admin/semantic/compress`
- `POST /admin/semantic/reembed`
- `GET /metrics`
- `GET /health`
- `POST /auth/token/issue`
- `POST /auth/token/rotate`

## Verification Checklist

- [ ] Backend starts cleanly and `GROQ_API_KEY loaded: True` is shown.
- [ ] `POST /chat` returns `usage` and `chat_id`.
- [ ] `POST /chat/stream` emits `start`, `token`, and `done` events.
- [ ] `GET /memories/semantic` returns semantic memory rows after chatting.
- [ ] `GET /metrics` exposes request/latency/token metrics.
