# Builds model-facing CryoSPARC job result packages after execution finishes.
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Any

from workflow_state import ACTIVE_STATUSES, extract_workflow_state, find_node
from quality_policy import assess_node


TERMINAL_STATUSES = {"completed", "failed", "killed"}
SUCCESS_STATUSES = {"completed"}
DEFAULT_MAX_RUNTIME_HOURS = 12
DEFAULT_NO_PROGRESS_TIMEOUT_HOURS = 2


def get_job_result_package(
    project_uid: str,
    workspace_uid: str,
    job_uid: str,
    include_next_candidates: bool = True,
) -> dict[str, Any]:
    """Return an internal status or model-facing result package for one job."""
    workflow_state = _read_workflow_state_with_retry(project_uid, workspace_uid)
    node = find_node(workflow_state, job_uid)
    if node is None:
        return {
            "success": False,
            "ready_for_model": False,
            "internal_only": True,
            "message_type": "mcp_internal_job_status",
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_uid": job_uid,
            "issues": [
                {
                    "severity": "error",
                    "code": "job_not_found",
                    "message": f"Job {job_uid!r} was not found in the workspace.",
                    "path": "job_uid",
                }
            ],
        }

    if node["status"] in TERMINAL_STATUSES:
        return build_model_result_package(
            workflow_state,
            node,
            include_next_candidates=include_next_candidates,
        )

    return build_internal_status_package(workflow_state, node)


def _read_workflow_state_with_retry(project_uid: str, workspace_uid: str) -> dict[str, Any]:
    """Retry transient CryoSPARC reads before exposing an internal failure."""
    attempts = max(1, int(os.getenv("CRYOAGENT_CRYOSPARC_MAX_ATTEMPTS", "3")))
    backoff = max(0.0, float(os.getenv("CRYOAGENT_CRYOSPARC_RETRY_BACKOFF_SECONDS", "1")))
    for attempt in range(attempts):
        try:
            return extract_workflow_state(project_uid, workspace_uid)
        except Exception:
            if attempt + 1 >= attempts:
                raise
            sleep(min(backoff * (2 ** attempt), 16.0))
    raise RuntimeError("CryoSPARC workflow read produced no response")


def wait_for_job_result_package(
    project_uid: str,
    workspace_uid: str,
    job_uid: str,
    timeout_seconds: int = 0,
    poll_interval_seconds: int = 30,
    include_next_candidates: bool = True,
) -> dict[str, Any]:
    """Poll CryoSPARC until a job reaches a terminal state or the timeout expires."""
    deadline = monotonic() + max(timeout_seconds, 0)
    poll_count = 0

    while True:
        poll_count += 1
        package = get_job_result_package(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            job_uid=job_uid,
            include_next_candidates=include_next_candidates,
        )
        package["poll_count"] = poll_count
        package["timeout_seconds"] = timeout_seconds

        if package.get("ready_for_model") or timeout_seconds <= 0:
            return package
        if monotonic() >= deadline:
            package["timed_out"] = True
            return package

        sleep(max(poll_interval_seconds, 1))


def build_internal_status_package(
    workflow_state: dict[str, Any],
    node: dict[str, Any],
) -> dict[str, Any]:
    """Summarize queue/running state for MCP bookkeeping, not model input."""
    monitoring = build_active_monitoring(node)
    human_action = build_human_action(node)
    status_group = (
        "human_action_required"
        if human_action
        else
        "attention_required"
        if monitoring["attention_required"]
        else "active" if node["status"] in ACTIVE_STATUSES else "unknown"
    )
    return {
        "success": True,
        "ready_for_model": False,
        "internal_only": True,
        "message_type": "mcp_internal_job_status",
        "project_uid": workflow_state["project_uid"],
        "workspace_uid": workflow_state["workspace_uid"],
        "job_uid": node["cryosparc_job_uid"],
        "workflow_node_id": node["workflow_node_id"],
        "job_type": node["job_type"],
        "status": node["status"],
        "status_group": status_group,
        "updated_at": node["updated_at"],
        "outputs": summarize_outputs(node),
        "monitoring": monitoring,
        "human_action_required": bool(human_action),
        "human_action": human_action,
        "message": (
            human_action["instruction"]
            if human_action
            else
            "Job needs human attention; keep this status inside MCP and do not "
            "ask the model for the next decision yet."
            if monitoring["attention_required"]
            else (
                "Job is not finished; keep this status inside MCP and do not ask "
                "the model for the next decision yet."
            )
        ),
    }


def build_human_action(node: dict[str, Any]) -> dict[str, Any] | None:
    """Describe required human UI work for interactive jobs."""
    if node["job_type"] != "select_2D":
        return None
    return {
        "action_type": "cryosparc_interactive_selection",
        "job_uid": node["cryosparc_job_uid"],
        "job_type": node["job_type"],
        "status": node["status"],
        "instruction": (
            f"Open CryoSPARC job {node['cryosparc_job_uid']} in the UI, "
            "select good 2D classes in the Interactive tab, then finish the job."
        ),
    }


def build_model_result_package(
    workflow_state: dict[str, Any],
    node: dict[str, Any],
    include_next_candidates: bool,
) -> dict[str, Any]:
    """Build the result JSON that can be sent to the model for the next decision."""
    package = {
        "success": node["status"] in SUCCESS_STATUSES,
        "ready_for_model": True,
        "internal_only": False,
        "schema_version": "1.0",
        "message_type": "mcp_job_result",
        "task": "Review the finished CryoSPARC job and choose the next workflow action.",
        "project_uid": workflow_state["project_uid"],
        "workspace_uid": workflow_state["workspace_uid"],
        "job_uid": node["cryosparc_job_uid"],
        "workflow_node_id": node["workflow_node_id"],
        "job_type": node["job_type"],
        "title": node["title"],
        "status": node["status"],
        "updated_at": node["updated_at"],
        "timestamps": node.get("timestamps") or {},
        "runtime": node.get("runtime") or {},
        "run_errors": node.get("run_errors") or {},
        "has_error": node["has_error"],
        "has_warning": node["has_warning"],
        "inputs": node["inputs"],
        "outputs": summarize_outputs(node),
        "metrics": build_basic_metrics(node),
        "quality_assessment": assess_node(node),
        "workflow_context": {
            "workflow_status": workflow_state["workflow_status"],
            "running_nodes": workflow_state["running_nodes"],
            "failed_nodes": workflow_state["failed_nodes"],
            "terminal_nodes": workflow_state["terminal_nodes"],
            "current_node": {
                "workflow_node_id": node["workflow_node_id"],
                "job_type": node["job_type"],
                "status": node["status"],
            },
            # The old package exposed only the selected terminal node and a
            # few workflow counters.  Keep those stable, but also provide the
            # complete bounded DAG so downstream research agents can use the
            # parameters, inputs, outputs, metrics, errors, and dependencies
            # from every task in the workflow.
            "workflow_index": build_workflow_index(workflow_state),
        },
        # Kept for the local query service; real agents receive only the
        # workflow_index until triage explicitly requests node details.
        "workflow_detail_store": build_workflow_snapshot(workflow_state),
        "next_candidate_actions": [],
        "blocked_actions": [],
        "output_contract": {
            "return_json_only": True,
            "schema_version": "1.0",
            "allowed_decision_type": ["forward", "rollback", "branch", "stop"],
            "decision_rule": (
                "Choose one allowed decision_type. For forward or branch, provide "
                "the CryoSPARC job_type/action and explicit connections when needed; "
                "next_candidate_actions is optional context, not a hard constraint."
            ),
            "rollback_modes": [
                "mark_only",
                "rerun_from_target",
                "branch_from_target",
                "manual_review",
            ],
        },
    }

    if include_next_candidates and node["status"] == "completed":
        candidate_context = get_next_candidate_context(
            workflow_state["project_uid"],
            workflow_state["workspace_uid"],
            node["workflow_node_id"],
        )
        package["next_candidate_actions"] = candidate_context["candidate_actions"]
        package["blocked_actions"] = candidate_context["blocked_actions"]
        package["decision_hint"] = candidate_context["decision_hint"]

    return package


def summarize_outputs(node: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Keep output summaries compact enough for model context."""
    return {
        output_name: {
            "type": output["type"],
            "available": output["available"],
            "num_items": output["num_items"],
            "result_names": output["result_names"],
            "summary_keys": output["summary_keys"],
            "latest_summary_stat_keys": output["latest_summary_stat_keys"],
        }
        for output_name, output in node["outputs"].items()
    }


def build_workflow_snapshot(workflow_state: dict[str, Any]) -> dict[str, Any]:
    """Return rich workflow evidence with bounded per-output statistics."""
    nodes = []
    for node in workflow_state.get("nodes", []):
        compact = dict(node)
        outputs = {}
        for output_name, output in (node.get("outputs") or {}).items():
            item = dict(output)
            item["summary_values"] = _bound_json_value(item.get("summary_values"), 12000)
            item["latest_summary_stat_values"] = _bound_json_value(
                item.get("latest_summary_stat_values"), 12000
            )
            outputs[output_name] = item
        compact["outputs"] = outputs
        nodes.append(compact)
    return {
        "schema_version": workflow_state.get("schema_version"),
        "generated_at": workflow_state.get("generated_at"),
        "project_uid": workflow_state.get("project_uid"),
        "workspace_uid": workflow_state.get("workspace_uid"),
        "workflow_status": workflow_state.get("workflow_status"),
        "node_count": len(nodes),
        "nodes": nodes,
        "edges": workflow_state.get("edges", []),
        "root_nodes": workflow_state.get("root_nodes", []),
        "terminal_nodes": workflow_state.get("terminal_nodes", []),
        "running_nodes": workflow_state.get("running_nodes", []),
        "failed_nodes": workflow_state.get("failed_nodes", []),
        "node_mapping": workflow_state.get("node_mapping", {}),
    }


def build_workflow_index(workflow_state: dict[str, Any]) -> dict[str, Any]:
    """Return the small DAG view shown to the Supervisor on first pass."""
    nodes = []
    for node in workflow_state.get("nodes", []):
        nodes.append({
            "job_uid": node.get("cryosparc_job_uid"),
            "logical_node_id": node.get("logical_node_id"),
            "job_type": node.get("job_type"),
            "title": node.get("title"),
            "status": node.get("status"),
            "parent_job_uids": node.get("parent_job_uids", []),
            "child_job_uids": node.get("child_job_uids", []),
        })
    return {
        "schema_version": workflow_state.get("schema_version"),
        "project_uid": workflow_state.get("project_uid"),
        "workspace_uid": workflow_state.get("workspace_uid"),
        "workflow_status": workflow_state.get("workflow_status"),
        "node_count": len(nodes),
        "nodes": nodes,
        "edges": workflow_state.get("edges", []),
        "failed_nodes": workflow_state.get("failed_nodes", []),
        "running_nodes": workflow_state.get("running_nodes", []),
    }


def _bound_json_value(value: Any, max_chars: int) -> Any:
    """Preserve structured statistics while preventing one output from flooding prompts."""
    if value is None:
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)[:max_chars]
    if len(encoded) <= max_chars:
        return value
    if isinstance(value, dict):
        result = {}
        used = 2
        for key, item in value.items():
            piece = json.dumps({str(key): item}, ensure_ascii=False, default=str)
            if used + len(piece) > max_chars:
                break
            result[str(key)] = item
            used += len(piece)
        result["_truncated"] = True
        return result
    if isinstance(value, list):
        result = []
        used = 2
        for item in value:
            piece = json.dumps(item, ensure_ascii=False, default=str)
            if used + len(piece) > max_chars:
                break
            result.append(item)
            used += len(piece)
        return {"items": result, "_truncated": True}
    return str(value)[:max_chars] + "..."


def build_basic_metrics(node: dict[str, Any]) -> dict[str, Any]:
    """Expose generic metrics available from normalized CryoSPARC outputs."""
    return {
        "completed": node["status"] == "completed",
        "failed": node["status"] in {"failed", "killed"},
        "has_error": node["has_error"],
        "has_warning": node["has_warning"],
        "num_items_by_output": {
            output_name: output["num_items"]
            for output_name, output in node["outputs"].items()
        },
        "available_outputs": [
            output_name
            for output_name, output in node["outputs"].items()
            if output["available"]
        ],
    }


def build_active_monitoring(node: dict[str, Any]) -> dict[str, Any]:
    """Flag active jobs that run too long or show no registered output progress."""
    runtime_hours = active_runtime_hours(node)
    output_item_count = sum(
        output["num_items"]
        for output in node["outputs"].values()
    )
    flags = []
    if runtime_hours is not None and runtime_hours > DEFAULT_MAX_RUNTIME_HOURS:
        flags.append("max_runtime_exceeded")
    if (
        runtime_hours is not None
        and runtime_hours > DEFAULT_NO_PROGRESS_TIMEOUT_HOURS
        and output_item_count == 0
    ):
        flags.append("no_registered_output_progress")
    return {
        "max_runtime_hours": DEFAULT_MAX_RUNTIME_HOURS,
        "no_progress_timeout_hours": DEFAULT_NO_PROGRESS_TIMEOUT_HOURS,
        "runtime_hours": runtime_hours,
        "output_item_count": output_item_count,
        "attention_required": bool(flags),
        "flags": flags,
    }


def active_runtime_hours(node: dict[str, Any]) -> float | None:
    """Estimate runtime hours from the best available active-job timestamp."""
    timestamps = node.get("timestamps") or {}
    start_value = (
        timestamps.get("running_at")
        or timestamps.get("started_at")
        or timestamps.get("launched_at")
        or timestamps.get("queued_at")
    )
    start = parse_timestamp(start_value)
    if start is None:
        return None
    now = datetime.now(timezone.utc)
    return round((now - start).total_seconds() / 3600, 3)


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse ISO timestamps emitted by workflow_state."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def get_next_candidate_context(
    project_uid: str,
    workspace_uid: str,
    current_node_id: str,
) -> dict[str, Any]:
    """Import lazily to avoid a module cycle with action_registry."""
    from action_registry import get_candidate_actions

    return get_candidate_actions(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_node_id=current_node_id,
    )
