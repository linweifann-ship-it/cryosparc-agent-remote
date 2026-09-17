"""OpenAI-compatible model tool-calling loop for read-only KB evidence."""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from kb_bridge import call_kb_tool, get_decision_context


KB_TOOL_NAMES = {
    "kb_search_cryoem_kb", "kb_get_dataset_summary", "kb_get_workflow", "kb_get_maps",
    "kb_get_failures", "kb_get_images", "kb_find_similar_cases", "kb_get_next_steps",
    "kb_get_job_doc", "kb_get_manual_annotations", "kb_get_decision_context",
}


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


def build_kb_tool_definitions() -> list[dict[str, Any]]:
    """Schemas exposed to the model; no CryoSPARC mutating tools are exposed."""
    text = {"type": "string"}
    top_k = {"type": "integer", "minimum": 1, "maximum": 20}
    return [
        _schema("kb_search_cryoem_kb", "Search cryo-EM workflow, documentation, and failure evidence.", {"query": text, "top_k": top_k, "kb_types": {"type": "array", "items": text}}, ["query"]),
        _schema("kb_get_dataset_summary", "Get historical summary for a known dataset ID.", {"dataset_id": text}, ["dataset_id"]),
        _schema("kb_get_workflow", "Get the historical workflow graph for a known dataset.", {"dataset_id": text}, ["dataset_id"]),
        _schema("kb_get_maps", "Get map and resolution records.", {"dataset_id": text, "multi_map": {"type": ["boolean", "string", "null"]}, "top_k": top_k}),
        _schema("kb_get_failures", "Retrieve historical failure patterns and mitigations.", {"failure_class": text, "job_type": text, "dataset_id": text, "top_k": top_k}),
        _schema("kb_get_images", "Retrieve historical image inventory and annotations.", {"dataset_id": text, "job_id": text, "image_type": text, "top_k": top_k}),
        _schema("kb_find_similar_cases", "Find non-excluded historical cases similar to the current dataset.", {"input_type": text, "molecule_type": text, "multi_map": {"type": ["boolean", "string", "null"]}, "top_k": top_k}),
        _schema("kb_get_next_steps", "Find observed next actions after a job type.", {"job_type": text, "top_k": top_k}, ["job_type"]),
        _schema("kb_get_job_doc", "Retrieve official CryoSPARC documentation for a job type.", {"job_type": text, "query": text, "top_k": top_k}),
        _schema("kb_get_manual_annotations", "Retrieve expert annotations, if available.", {"annotation_type": text, "dataset_id": text, "job_type": text, "top_k": top_k}),
        _schema("kb_get_decision_context", "Build a compact evidence bundle from the KB for the current decision.", {"dataset_info": {"type": "object"}, "current_state": {"type": "object"}, "candidate_actions": {"type": "array", "items": {"type": "object"}}, "top_k": top_k}),
    ]


def execute_kb_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "kb_get_decision_context":
        return get_decision_context(**arguments)
    if not tool_name.startswith("kb_"):
        raise ValueError(f"Tool is outside the KB allowlist: {tool_name}")
    return call_kb_tool(tool_name[3:], arguments)


async def run_kb_tool_call_loop(
    messages: list[dict[str, Any]],
    model_call: Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[dict[str, Any]]],
    tool_executor: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
    max_tool_calls: int = 4,
) -> dict[str, Any]:
    """Let the model request KB evidence, then continue until final JSON text."""
    working = list(messages)
    trace: list[dict[str, Any]] = []
    total_calls = 0
    tools = build_kb_tool_definitions()
    while True:
        result = await model_call(working, tools)
        tool_calls = result.get("tool_calls") or []
        if not tool_calls:
            result["tool_trace"] = trace
            result["messages_after_tools"] = working
            return result
        if total_calls + len(tool_calls) > max_tool_calls:
            # Preserve collected evidence, then force a final decision pass.
            working.append({
                "role": "user",
                "content": (
                    f"The KB call limit ({max_tool_calls}) has been reached. "
                    "Do not call any more tools. Based on the current state, "
                    "candidate actions, and KB evidence already provided, "
                    "return the final decision JSON now."
                ),
            })
            result = await model_call(working, [])
            if result.get("tool_calls"):
                raise RuntimeError("Model requested KB tools after the KB limit was reached.")
            result["tool_trace"] = trace
            result["messages_after_tools"] = working
            result["kb_limit_reached"] = True
            return result
        assistant_message = result.get("assistant_message") or {
            "role": "assistant", "content": result.get("raw_text") or None,
            "tool_calls": tool_calls,
        }
        working.append(assistant_message)
        for call in tool_calls:
            arguments: dict[str, Any] = {}
            function = call.get("function") or {}
            name = function.get("name")
            if name not in KB_TOOL_NAMES:
                raise RuntimeError(f"Model requested non-KB tool: {name}")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                tool_result = await tool_executor(name, arguments)
            except Exception as exc:
                tool_result = {"success": False, "error_type": type(exc).__name__, "error": str(exc)}
            trace.append({"tool_call_id": call.get("id"), "tool_name": name, "arguments": arguments, "result": tool_result})
            working.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "name": name,
                "content": json.dumps(tool_result, ensure_ascii=False, default=str),
            })
            total_calls += 1
