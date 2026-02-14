"""
Central configuration for production-grade runtime features.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_env: str
    app_name: str
    debug: bool
    base_dir: Path
    memory_dir: Path
    semantic_store_path: Path
    enable_semantic_memory: bool
    enable_streaming: bool
    enable_tools: bool
    enable_background_tasks: bool
    enable_prompt_injection_guard: bool
    max_prompt_chars: int
    max_requests_per_minute: int
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    require_qdrant: bool
    embedding_provider: str
    embedding_model: str
    embedding_dims: int
    semantic_top_k: int
    semantic_token_budget: int
    importance_decay_per_day: float
    importance_drop_threshold: float
    semantic_compression_age_days: int
    semantic_compression_min_cluster: int
    llm_cost_input_per_1k: float
    llm_cost_output_per_1k: float
    cors_allow_origins: list[str]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[3]
    memory_dir = base_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
    if raw_origins.strip() == "*":
        cors_allow_origins = ["*"]
    else:
        cors_allow_origins = [x.strip() for x in raw_origins.split(",") if x.strip()]

    return Settings(
        app_env=os.getenv("APP_ENV", "dev").strip().lower(),
        app_name=os.getenv("APP_NAME", "Mnemos"),
        debug=_env_bool("DEBUG", False),
        base_dir=base_dir,
        memory_dir=memory_dir,
        semantic_store_path=memory_dir / "semantic_memories.json",
        enable_semantic_memory=_env_bool("ENABLE_SEMANTIC_MEMORY", True),
        enable_streaming=_env_bool("ENABLE_STREAMING", True),
        enable_tools=_env_bool("ENABLE_TOOLS", True),
        enable_background_tasks=_env_bool("ENABLE_BACKGROUND_TASKS", True),
        enable_prompt_injection_guard=_env_bool("ENABLE_PROMPT_GUARD", True),
        max_prompt_chars=_env_int("MAX_PROMPT_CHARS", 6000),
        max_requests_per_minute=_env_int("RATE_LIMIT_RPM", 60),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip(),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "mnemos_semantic_memory").strip() or "mnemos_semantic_memory",
        require_qdrant=_env_bool("REQUIRE_QDRANT", False),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").strip().lower(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "bge-small-en-v1.5").strip(),
        embedding_dims=_env_int("EMBEDDING_DIMS", 384),
        semantic_top_k=_env_int("SEMANTIC_TOP_K", 12),
        semantic_token_budget=_env_int("SEMANTIC_TOKEN_BUDGET", 900),
        importance_decay_per_day=_env_float("IMPORTANCE_DECAY_PER_DAY", 0.985),
        importance_drop_threshold=_env_float("IMPORTANCE_DROP_THRESHOLD", 0.18),
        semantic_compression_age_days=_env_int("SEMANTIC_COMPRESSION_AGE_DAYS", 14),
        semantic_compression_min_cluster=_env_int("SEMANTIC_COMPRESSION_MIN_CLUSTER", 3),
        llm_cost_input_per_1k=_env_float("LLM_COST_INPUT_PER_1K", 0.0),
        llm_cost_output_per_1k=_env_float("LLM_COST_OUTPUT_PER_1K", 0.0),
        cors_allow_origins=cors_allow_origins,
    )
