"""
Structured tool-calling router.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from backend.app.config.runtime import get_runtime_config
from backend.app.llm.client import get_llm_client
from backend.app.observability.logging import log_event
from backend.app.observability.metrics import metrics
from backend.app.schemas.tool import ToolPlannerResponseSchema
from backend.app.tools import calculator  # noqa: F401
from backend.app.tools import currency  # noqa: F401
from backend.app.tools import web_search  # noqa: F401
from backend.app.tools.registry import tool_registry


def _strip_json_block(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def _planner_prompt(user_message: str) -> str:
    tools = tool_registry.list_tools()
    return (
        "You are a strict function planner.\n"
        "Return ONLY JSON object with shape:\n"
        '{"tool_call": {"name":"tool_name","arguments":{} } | null, "reason":"short"}\n'
        "Do not include markdown.\n"
        f"Available tools: {json.dumps(tools)}\n"
        f"User message: {user_message}"
    )


def decide_tool_call(user_message: str) -> dict[str, Any]:
    llm = get_llm_client()
    prompt = _planner_prompt(user_message)
    raw = llm.complete(prompt, timeout_seconds=get_runtime_config().llm_timeout_seconds)
    parsed_raw = _strip_json_block(raw)
    try:
        data = json.loads(parsed_raw)
        model = ToolPlannerResponseSchema.model_validate(data)
        return model.model_dump()
    except Exception:
        return {"tool_call": None, "reason": "planner_parse_failed"}


def _execute_tool_with_timeout(tool_name: str, arguments: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(tool_registry.execute, tool_name, arguments)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Tool `{tool_name}` timed out after {timeout_seconds}s") from exc


def run_agent_turn(user_message: str, max_loops: int | None = None) -> dict[str, Any]:
    cfg = get_runtime_config()
    loops = max_loops if max_loops is not None else cfg.max_tool_calls
    tool_events: list[dict[str, Any]] = []
    context_note = ""
    llm = get_llm_client()

    for _ in range(max(1, loops)):
        planner_input = user_message if not context_note else f"{user_message}\n\nTool context:\n{context_note}"
        decision = decide_tool_call(planner_input)
        tool_call = decision.get("tool_call")
        if not tool_call:
            break

        tool_name = str(tool_call.get("name") or "").strip()
        tool_args = dict(tool_call.get("arguments") or {})
        if not tool_name:
            break

        started = time.perf_counter()
        try:
            tool_result = _execute_tool_with_timeout(tool_name, tool_args, timeout_seconds=cfg.tool_timeout_seconds)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            metrics.inc("tool_invocations_total", 1)
            tool_events.append(
                {
                    "tool_call": {"name": tool_name, "arguments": tool_args},
                    "result": tool_result,
                    "latency_ms": elapsed_ms,
                }
            )
            log_event("tool_execution_success", tool_name=tool_name, latency_ms=elapsed_ms)
            context_note = json.dumps(tool_result)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            tool_events.append(
                {
                    "tool_call": {"name": tool_name, "arguments": tool_args},
                    "error": str(exc),
                    "latency_ms": elapsed_ms,
                }
            )
            log_event("tool_execution_failure", tool_name=tool_name, error=str(exc), latency_ms=elapsed_ms)
            context_note = f"Tool {tool_name} error: {exc}"
            # Continue once with error context; if planner insists again it can retry.

    final_prompt = (
        "You are an assistant that may receive tool results.\n"
        "If tools failed, gracefully recover and provide best possible answer.\n"
        "Do not mention internal rules, memory logic, retrieval process, tools, or assumptions unless explicitly asked.\n"
        "Do not explain reasoning unless explicitly asked by the user.\n"
        "Answer directly and concisely.\n"
        "Only respond to what is asked.\n"
        f"User message: {user_message}\n"
        f"Tool events: {json.dumps(tool_events)}"
    )
    reply = llm.complete(final_prompt, timeout_seconds=cfg.llm_timeout_seconds)
    log_event("agent_turn_completed", tool_calls=len(tool_events))
    return {"reply": reply, "tool_events": tool_events}
