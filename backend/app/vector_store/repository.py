"""
Vector storage abstraction.

Uses Qdrant when available; otherwise falls back to local JSON persistence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

from backend.app.core.config import get_settings
from backend.app.embeddings.provider import cosine_similarity
from backend.app.memory.models import SemanticMemory
from backend.app.observability.logging import log_event
from backend.app.vector_store.qdrant_client import get_qdrant_client


@dataclass
class VectorHit:
    memory: SemanticMemory
    similarity: float


class VectorStoreRepository:
    def upsert(self, memory: SemanticMemory):
        raise NotImplementedError

    def search(
        self,
        vector: list[float],
        user_id: str,
        top_k: int,
        scopes: Optional[list[str]] = None,
    ) -> list[VectorHit]:
        raise NotImplementedError

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        raise NotImplementedError

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        raise NotImplementedError


class LocalVectorStoreRepository(VectorStoreRepository):
    def __init__(self):
        settings = get_settings()
        self.path: Path = settings.semantic_store_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: dict[str, SemanticMemory] = {}
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return
            for item in data:
                if not isinstance(item, dict):
                    continue
                mem = SemanticMemory.from_dict(item)
                self._records[mem.id] = mem
        except Exception as exc:
            log_event("semantic_store_load_failed", error=str(exc), path=str(self.path))

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            payload = [m.to_dict() for m in self._records.values()]
            json.dump(payload, f, indent=2)

    def upsert(self, memory: SemanticMemory):
        with self._lock:
            self._records[memory.id] = memory
            self._save()

    def search(
        self,
        vector: list[float],
        user_id: str,
        top_k: int,
        scopes: Optional[list[str]] = None,
    ) -> list[VectorHit]:
        scopes_set = set(scopes or [])
        hits: list[VectorHit] = []
        with self._lock:
            for memory in self._records.values():
                if not memory.is_active:
                    continue
                if memory.user_id != user_id:
                    continue
                if scopes_set and memory.scope not in scopes_set:
                    continue
                if not memory.embedding:
                    continue
                similarity = cosine_similarity(vector, memory.embedding)
                hits.append(VectorHit(memory=memory, similarity=similarity))
        hits.sort(key=lambda x: x.similarity, reverse=True)
        return hits[:top_k]

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        with self._lock:
            rows = [m for m in self._records.values() if m.user_id == user_id]
        rows.sort(key=lambda m: m.updated_at, reverse=True)
        return rows

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        with self._lock:
            memory = self._records.get(memory_id)
            if not memory or memory.user_id != user_id:
                return False
            del self._records[memory_id]
            self._save()
            return True


class QdrantVectorStoreRepository(VectorStoreRepository):
    def __init__(self):
        self.settings = get_settings()
        self.client = get_qdrant_client()
        self.collection = self.settings.qdrant_collection
        if self.client is not None:
            self._ensure_collection()

    def _ensure_collection(self):
        try:
            from qdrant_client.models import Distance, VectorParams  # type: ignore

            existing = self.client.get_collections().collections
            names = {c.name for c in existing}
            if self.collection not in names:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.settings.embedding_dims, distance=Distance.COSINE),
                )
        except Exception as exc:
            log_event("qdrant_collection_init_failed", error=str(exc), collection=self.collection)

    def _to_point(self, memory: SemanticMemory):
        from qdrant_client.models import PointStruct  # type: ignore

        payload = {
            "memory_id": memory.id,
            "user_id": memory.user_id,
            "scope": memory.scope,
            "memory_type": memory.memory_type,
            "importance_score": memory.importance_score,
            "reinforcement_count": memory.reinforcement_count,
            "decay_factor": memory.decay_factor,
            "tags": memory.tags,
            "source_message_id": memory.source_message_id,
            "created_at": memory.created_at.isoformat(),
            "updated_at": memory.updated_at.isoformat(),
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "content": memory.content,
            "embedding_model": memory.embedding_model,
            "embedding_provider": memory.embedding_provider,
            "is_active": memory.is_active,
            "is_archived": memory.is_archived,
            "archived_at": memory.archived_at.isoformat() if memory.archived_at else None,
            "metadata": memory.metadata,
        }
        return PointStruct(id=memory.id, vector=memory.embedding, payload=payload)

    def upsert(self, memory: SemanticMemory):
        if self.client is None:
            raise RuntimeError("Qdrant client unavailable")
        point = self._to_point(memory)
        self.client.upsert(collection_name=self.collection, points=[point], wait=False)

    def search(
        self,
        vector: list[float],
        user_id: str,
        top_k: int,
        scopes: Optional[list[str]] = None,
    ) -> list[VectorHit]:
        if self.client is None:
            return []
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue  # type: ignore

            conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            if scopes:
                conditions.append(FieldCondition(key="scope", match=MatchAny(any=scopes)))
            conditions.append(FieldCondition(key="is_active", match=MatchValue(value=True)))

            points = self.client.search(
                collection_name=self.collection,
                query_vector=vector,
                query_filter=Filter(must=conditions),
                limit=top_k,
                with_vectors=True,
            )
            hits: list[VectorHit] = []
            for p in points:
                payload = p.payload or {}
                stored_vector = list(getattr(p, "vector", []) or [])
                memory = SemanticMemory.from_dict(
                    {
                        "id": payload.get("memory_id") or str(p.id),
                        "user_id": payload.get("user_id"),
                        "content": payload.get("content"),
                        "memory_type": payload.get("memory_type"),
                        "scope": payload.get("scope"),
                        "importance_score": payload.get("importance_score", 0.5),
                        "reinforcement_count": payload.get("reinforcement_count", 0),
                        "decay_factor": payload.get("decay_factor", 0.01),
                        "tags": payload.get("tags", []),
                        "source_message_id": payload.get("source_message_id", ""),
                        "embedding": stored_vector,
                        "embedding_model": payload.get("embedding_model", ""),
                        "embedding_provider": payload.get("embedding_provider", ""),
                        "created_at": payload.get("created_at"),
                        "updated_at": payload.get("updated_at"),
                        "last_accessed": payload.get("last_accessed"),
                        "is_active": payload.get("is_active", True),
                        "is_archived": payload.get("is_archived", False),
                        "archived_at": payload.get("archived_at"),
                        "metadata": payload.get("metadata", {}),
                    }
                )
                hits.append(VectorHit(memory=memory, similarity=float(getattr(p, "score", 0.0))))
            return hits
        except Exception as exc:
            log_event("qdrant_search_failed", error=str(exc))
            return []

    def list_user_memories(self, user_id: str) -> list[SemanticMemory]:
        if self.client is None:
            return []
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

            points, _ = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    ]
                ),
                limit=500,
            )
            rows: list[SemanticMemory] = []
            for p in points:
                payload = p.payload or {}
                rows.append(
                    SemanticMemory.from_dict(
                        {
                            "id": payload.get("memory_id") or str(p.id),
                            "user_id": payload.get("user_id"),
                            "content": payload.get("content"),
                            "memory_type": payload.get("memory_type"),
                            "scope": payload.get("scope"),
                            "importance_score": payload.get("importance_score", 0.5),
                            "reinforcement_count": payload.get("reinforcement_count", 0),
                            "decay_factor": payload.get("decay_factor", 0.01),
                            "tags": payload.get("tags", []),
                            "source_message_id": payload.get("source_message_id", ""),
                            "embedding_model": payload.get("embedding_model", ""),
                            "embedding_provider": payload.get("embedding_provider", ""),
                            "created_at": payload.get("created_at"),
                            "updated_at": payload.get("updated_at"),
                            "last_accessed": payload.get("last_accessed"),
                            "is_active": payload.get("is_active", True),
                            "is_archived": payload.get("is_archived", False),
                            "archived_at": payload.get("archived_at"),
                            "metadata": payload.get("metadata", {}),
                        }
                    )
                )
            rows.sort(key=lambda m: m.updated_at, reverse=True)
            return rows
        except Exception as exc:
            log_event("qdrant_scroll_failed", error=str(exc))
            return []

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        if self.client is None:
            return False
        try:
            # Enforce user isolation before delete.
            points = self.client.retrieve(
                collection_name=self.collection,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return False
            payload = getattr(points[0], "payload", {}) or {}
            if str(payload.get("user_id") or "").strip() != str(user_id).strip():
                log_event("qdrant_delete_blocked_user_mismatch", memory_id=memory_id, requested_user=user_id)
                return False
            self.client.delete(collection_name=self.collection, points_selector=[memory_id], wait=True)
            return True
        except Exception as exc:
            log_event("qdrant_delete_failed", error=str(exc), memory_id=memory_id)
            return False


def iter_memory_tokens(text: str) -> Iterable[str]:
    for token in (text or "").lower().replace("\n", " ").split():
        t = token.strip(".,!?;:\"'()[]{}")
        if len(t) >= 3:
            yield t


_repo_cache: VectorStoreRepository | None = None


def get_vector_store() -> VectorStoreRepository:
    global _repo_cache
    if _repo_cache is not None:
        return _repo_cache

    repo: VectorStoreRepository
    settings = get_settings()
    qdrant_repo = QdrantVectorStoreRepository()
    if qdrant_repo.client is not None:
        repo = qdrant_repo
        log_event("vector_store_ready", backend="qdrant", collection=qdrant_repo.collection)
    elif settings.require_qdrant:
        raise RuntimeError("Qdrant is required but unavailable. Check QDRANT_URL and QDRANT_API_KEY.")
    else:
        repo = LocalVectorStoreRepository()
        log_event("vector_store_ready", backend="local_json", path=str(settings.semantic_store_path))

    _repo_cache = repo
    return _repo_cache
