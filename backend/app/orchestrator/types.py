"""
Data structures for orchestrator pipeline layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class OrchestratorInput:
    user_id: str
    chat_id: str
    user_message: str
    continuity_message: str
    use_tools: bool = False
    scope: Optional[str] = None
    response_style: str = ""


@dataclass
class ContextBundle:
    deterministic_memory_context: str = ""
    semantic_memory_context: str = ""
    semantic_rows: list[dict[str, Any]] = field(default_factory=list)
    recency_buffer: str = ""
    tool_hints: str = ""
    realtime_context: str = ""
    deterministic_hints: list[str] = field(default_factory=list)

    @property
    def merged_memory_context(self) -> str:
        parts = [self.deterministic_memory_context.strip(), self.semantic_memory_context.strip()]
        return "\n".join([p for p in parts if p])


@dataclass
class PipelineResult:
    reply: str
    usage: dict[str, Any]
    semantic_rows: list[dict[str, Any]]
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    prompt_used: str = ""


@dataclass
class OrchestratorDependencies:
    deterministic_memory_fn: Callable[[str, str], str]
    semantic_retrieve_fn: Callable[[str, str], tuple[list[dict[str, Any]], str]]
    recency_buffer_fn: Callable[[Any], str]
    llm_complete_fn: Callable[[str, float], str]
    realtime_fn: Callable[[str], str]
    should_realtime_fn: Callable[[str], bool]
    tool_agent_fn: Callable[[str], dict[str, Any]]
    sanitize_reply_fn: Callable[[str, str], str]
