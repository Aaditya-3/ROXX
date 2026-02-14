"""
Platform-level API endpoints for architecture and runtime inspection.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.config.runtime import get_runtime_config
from backend.app.core.config import get_settings


router = APIRouter(prefix="/api/v1/platform", tags=["platform"])


@router.get("/architecture")
async def architecture():
    diagram = """User Input
-> Intent Classifier
-> Brain Layer
   - structured memory extraction/update
   - temporal normalization (relative -> absolute date)
   - logic reasoning layer (derived event status)
   - response style control
-> Context Builder
   - deterministic memory
   - semantic retrieval
   - recency buffer
   - tool hints
-> Context Ranking Layer
-> Prompt Assembly
-> LLM Invocation
-> Stream Handler
-> Persistence Layer
-> Background Memory Hooks"""
    return {"diagram": diagram}


@router.get("/config")
async def platform_config():
    cfg = get_runtime_config()
    settings = get_settings()
    return {
        "semantic_enabled": settings.enable_semantic_memory,
        "streaming_enabled": settings.enable_streaming,
        "tools_enabled": settings.enable_tools,
        "brain_layer_enabled": True,
        "structured_memory_store": "relational_db:user_preferences,user_events",
        "truth_db": "DATABASE_URL",
        "vector_db": f"qdrant:{settings.qdrant_collection}",
        "require_qdrant": settings.require_qdrant,
        "ranking_weights": {
            "similarity": cfg.ranking_weights.similarity,
            "importance": cfg.ranking_weights.importance,
            "recency": cfg.ranking_weights.recency,
        },
        "llm_timeout_seconds": cfg.llm_timeout_seconds,
        "max_tool_calls": cfg.max_tool_calls,
    }
