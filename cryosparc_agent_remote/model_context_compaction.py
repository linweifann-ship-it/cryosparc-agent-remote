"""Compact MCP replies before they are placed in a model conversation.

The registry and executor retain complete parameter and scheduler diagnostics
for validation and audit. Those objects are not suitable as repeated model
tool outputs: a scheduler snapshot can contain an entire cluster queue and a
generic candidate can contain every parameter of an unrelated job type. This
module changes presentation only; execution still uses the original objects.
"""
from __future__ import annotations

from typing import Any


MAX_INLINE_DEFAULT_PARAMETERS = 12


def compact_candidate_action(candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep every candidate while replacing verbose schemas with an index."""
    template = candidate.get("parameter_template") or {}
    required = sorted(
        name for name, spec in template.items()
        if isinstance(spec, dict) and spec.get("required")
    )
    defaults = candidate.get("default_parameters") or {}
    parameter_interface: dict[str, Any] = {
        "parameter_names": sorted(template),
        "required_parameters": required,
        "details_available_via": "get_job_doc",
    }
    # Fixed import facts remain visible. Large generic defaults can be queried
    # on demand rather than consuming every later model turn.
    if len(defaults) <= MAX_INLINE_DEFAULT_PARAMETERS:
        parameter_interface["default_parameters"] = defaults
    elif defaults:
        parameter_interface["default_parameters_omitted"] = len(defaults)

    result = {
        key: candidate.get(key)
        for key in (
            "action_id", "action_type", "workflow_node_id", "reference_job_uid",
            "reference_status", "job_type", "description", "execution_mode",
            "available", "blocked_by", "missing_required_inputs", "workflow_stage",
            "workflow_policy_recommendation",
        )
    }
    result["required_inputs"] = compact_required_inputs(candidate.get("required_inputs") or {})
    result["parameter_interface"] = parameter_interface
    return result


def compact_required_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Preserve connection identity without repeated source metadata."""
    compacted: dict[str, Any] = {}
    for name, sources in inputs.items():
        if not isinstance(sources, list):
            compacted[name] = sources
            continue
        compacted[name] = [
            {
                key: source.get(key)
                for key in (
                    "source_workflow_node_id", "source_logical_node_id",
                    "source_job_uid", "source_output", "result_names",
                    "source_job_type",
                )
                if source.get(key) is not None
            }
            if isinstance(source, dict) else source
            for source in sources
        ]
    return compacted


def compact_candidate_actions(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [compact_candidate_action(candidate) for candidate in candidates]


def compact_candidate_context_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy whose model-visible candidate lists are bounded."""
    result = dict(payload)
    for key in ("candidate_actions", "blocked_actions", "next_candidate_actions"):
        if isinstance(result.get(key), list):
            result[key] = compact_candidate_actions(result[key])
    return result


def compact_execution_response(response: dict[str, Any]) -> dict[str, Any]:
    """Expose created Job identities, not duplicated executor internals."""
    execution = response.get("execution_result") or {}
    return {
        "success": response.get("success"),
        "dry_run": response.get("dry_run"),
        "execution_mode": response.get("execution_mode"),
        "candidate_context": response.get("candidate_context"),
        "internal_decision": response.get("internal_decision"),
        "execution_result": {
            key: execution.get(key)
            for key in ("success", "dry_run", "execution_mode", "decision_type", "message")
            if key in execution
        } | {
            "execution_results": [
                compact_execution_result(result)
                for result in execution.get("execution_results", [])
            ],
            "issues": execution.get("issues") or [],
            "warnings": execution.get("warnings") or [],
        },
        "issues": response.get("issues") or [],
        "warnings": response.get("warnings") or [],
    }


def compact_execution_result(result: dict[str, Any]) -> dict[str, Any]:
    """Retain all fields required for runner Job waiting and error diagnosis."""
    compacted = {
        key: result.get(key)
        for key in (
            "success", "dry_run", "status", "project_uid", "workspace_uid",
            "job_uid", "job_type", "queued", "logical_workflow_step",
            "approval_required", "approval_reasons", "human_action_required",
            "error", "error_type", "issues",
        ) if key in result
    }
    logical = result.get("logical_job")
    if isinstance(logical, dict):
        compacted["logical_job"] = {
            key: logical.get(key)
            for key in (
                "logical_job_id", "logical_job_uid", "physical_job_ids",
                "winner_job_uid", "race_statuses", "cancellations",
            ) if key in logical
        }
    diagnostics = result.get("diagnostics")
    if isinstance(diagnostics, dict):
        compacted["diagnostics"] = {
            key: diagnostics.get(key)
            for key in ("execution_phase", "http_response") if key in diagnostics
        }
    return compacted
