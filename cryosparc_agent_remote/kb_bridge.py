"""Read-only bridge to the external cryo-EM knowledge-base implementation.

The KB is intentionally loaded from its existing checkout instead of copying
or modifying its SQLite database.  All functions in this module are read-only
and return JSON-serializable dictionaries suitable for MCP tool responses.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Callable


DEFAULT_KB_ROOT = "/hdd1/huangjianhua/agent/data0/kb_build_v2"
_MODULE = None


def kb_root() -> Path:
    return Path(os.getenv("CRYOAGENT_KB_ROOT", DEFAULT_KB_ROOT)).expanduser()


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    path = kb_root() / "mcp" / "tools" / "kb_tools.py"
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge-base tools not found: {path}")
    spec = importlib.util.spec_from_file_location("cryoagent_external_kb_tools", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load knowledge-base tools from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE = module
    return module


def normalize_dataset_id(dataset_id: str | None = None, empiar_id: str | None = None) -> str | None:
    """Normalize runtime EMPIAR labels to the KB's numeric dataset IDs."""
    value = dataset_id or empiar_id
    if not value:
        return None
    value = str(value).strip()
    lower = value.lower()
    if lower.startswith("empiar-"):
        value = value[7:]
    elif lower.startswith("empiar"):
        value = value[6:].lstrip("-_")
    return value or None


def call_kb_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call one of the external KB's read-only functions."""
    module = _load_module()
    allowed = {
        "search_cryoem_kb", "get_dataset_summary", "get_workflow", "get_maps",
        "get_failures", "get_images", "find_similar_cases", "get_next_steps",
        "get_job_doc", "get_manual_annotations",
    }
    if tool_name not in allowed:
        raise ValueError(f"Unsupported or non-read-only KB tool: {tool_name}")
    function: Callable[..., dict[str, Any]] = getattr(module, tool_name)
    return function(**(arguments or {}))


def get_decision_context(
    dataset_info: dict[str, Any] | None = None,
    current_state: dict[str, Any] | None = None,
    candidate_actions: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Return a compact evidence bundle for a workflow decision.

    This is a convenience tool, not an automatic decision rule.  The model
    can call it autonomously, then still has to choose from live candidates.
    """
    dataset_info = dataset_info or {}
    current_state = current_state or {}
    dataset_id = normalize_dataset_id(
        dataset_info.get("dataset_id"), dataset_info.get("empiar_id")
    )
    evidence: dict[str, Any] = {
        "dataset_id": dataset_id,
        "source": "cryoem_agent_kb_v2",
        "exact_dataset_match": False,
    }
    if dataset_id:
        summary = call_kb_tool("get_dataset_summary", {"dataset_id": dataset_id})
        evidence["dataset_summary"] = summary
        evidence["exact_dataset_match"] = bool(summary.get("workflow_case"))

    similar_args = {
        "input_type": dataset_info.get("input_type"),
        "molecule_type": dataset_info.get("macromolecules_type") or dataset_info.get("molecule_type"),
        "multi_map": dataset_info.get("num_of_maps") not in (None, 0, 1, "0", "1"),
        "top_k": top_k,
    }
    evidence["similar_cases"] = call_kb_tool("find_similar_cases", similar_args)

    last_action = current_state.get("last_action") or current_state.get("job_type")
    if last_action:
        evidence["observed_next_steps"] = call_kb_tool(
            "get_next_steps", {"job_type": last_action, "top_k": top_k}
        )
    job_types = []
    for candidate in candidate_actions or []:
        job_type = candidate.get("job_type") or candidate.get("action")
        if job_type and job_type not in job_types:
            job_types.append(job_type)
    evidence["candidate_job_docs"] = [
        call_kb_tool("get_job_doc", {"job_type": job_type, "top_k": 2})
        for job_type in job_types[:8]
    ]
    if current_state.get("last_node_status") in {"failed", "error", "killed"} or current_state.get("has_error"):
        evidence["failures"] = call_kb_tool(
            "get_failures",
            {"job_type": last_action, "dataset_id": dataset_id, "top_k": top_k},
        )
    return evidence
