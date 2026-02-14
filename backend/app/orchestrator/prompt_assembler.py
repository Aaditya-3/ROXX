"""
Prompt assembly layer.
"""

from __future__ import annotations

from datetime import datetime

from backend.app.orchestrator.types import ContextBundle, OrchestratorInput


class PromptAssembler:
    def build(self, payload: OrchestratorInput, context: ContextBundle) -> str:
        deterministic_hints = "\n".join(f"- {x}" for x in context.deterministic_hints if x)
        now = datetime.now().astimezone().isoformat(timespec="seconds")

        system_instructions = (
            "You are Mnemos, a context-aware assistant.\n"
            "Honor user memory isolation strictly.\n"
            "Use memory only when relevant.\n"
            "Do not mention internal rules, memory logic, retrieval process, tools, or assumptions.\n"
            "Do not explain reasoning unless explicitly asked by the user.\n"
            "Answer directly and concisely.\n"
            "Only respond to what is asked.\n"
            f"Current local datetime: {now}.\n"
        )

        prompt = f"""{system_instructions}
<REALTIME_CONTEXT>
{context.realtime_context}
</REALTIME_CONTEXT>
<MEMORY_CONTEXT_DETERMINISTIC>
{context.deterministic_memory_context}
</MEMORY_CONTEXT_DETERMINISTIC>
<MEMORY_CONTEXT_SEMANTIC>
{context.semantic_memory_context}
</MEMORY_CONTEXT_SEMANTIC>
<RECENCY_BUFFER>
{context.recency_buffer}
</RECENCY_BUFFER>
<DETERMINISTIC_HINTS>
{deterministic_hints}
</DETERMINISTIC_HINTS>
<TOOL_HINTS>
{context.tool_hints}
</TOOL_HINTS>
<USER_MESSAGE>
{payload.continuity_message}
</USER_MESSAGE>
"""
        return prompt
