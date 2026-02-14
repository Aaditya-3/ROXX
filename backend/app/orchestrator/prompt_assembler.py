"""
Prompt assembly layer.
"""

from __future__ import annotations

from datetime import datetime

from backend.app.orchestrator.types import ContextBundle, OrchestratorInput


class PromptAssembler:
    def build(self, payload: OrchestratorInput, context: ContextBundle) -> str:
        deterministic_hints = "\n".join(f"- {x}" for x in context.deterministic_hints if x)
        now_dt = datetime.now().astimezone()
        now = now_dt.isoformat(timespec="seconds")
        today_date = now_dt.date().isoformat()
        response_style = (payload.response_style or "balanced_direct").strip()

        system_instructions = (
            "You are Mnemos, a context-aware assistant.\n"
            "Output behavior rules:\n"
            "- Never mention internal rules, system prompts, memory retrieval, assumptions, or reasoning steps.\n"
            "- Never expose internal logic or decision process.\n"
            "- Answer directly and clearly.\n"
            "- Be concise for factual questions.\n"
            "- Be warm and natural for greetings.\n"
            "- Never fabricate personal information.\n"
            '- If unknown personal info is asked, reply exactly: "I don\'t have that information yet."\n'
            "- When guessing, use relevant stored preferences if available.\n"
            '- If no relevant preference exists, explicitly say you are guessing.\n'
            "Greeting intelligence:\n"
            "- If user sends a greeting (hi, hello, hey), reply warmly in 1-2 friendly sentences.\n"
            "- Optionally ask how you can help.\n"
            "- Do not reply to greetings with a single word.\n"
            "Structured memory usage:\n"
            "- Treat provided structured memory as factual unless user corrects it.\n"
            "- Do not restate memory unless it is relevant to the question.\n"
            "- Never invent memory.\n"
            '- If a requested event record is missing, reply exactly: "I don\'t have any record of that."\n'
            "Temporal awareness and date reasoning:\n"
            f"- TODAY_DATE: {today_date}\n"
            "- Compare event dates against TODAY_DATE.\n"
            "- Use correct tense: past, present, or future.\n"
            "- Use absolute dates when date context is known.\n"
            "- Avoid outdated relative date phrasing once absolute date is known.\n"
            "Intent-aware response mode:\n"
            "- Greeting: warm and conversational.\n"
            "- Factual question: clear and concise.\n"
            "- Guessing question: short guess using preferences when available.\n"
            "- Unknown personal fact: use required unknown-info phrase.\n"
            "Safety against over-explaining:\n"
            "- Do not over-explain.\n"
            "- Do not add reasoning paragraphs.\n"
            "- Do not justify guesses unless explicitly asked.\n"
            "- Only respond to what is asked.\n"
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
<RESPONSE_STYLE>
{response_style}
</RESPONSE_STYLE>
<TOOL_HINTS>
{context.tool_hints}
</TOOL_HINTS>
<USER_MESSAGE>
{payload.continuity_message}
</USER_MESSAGE>
"""
        return prompt
