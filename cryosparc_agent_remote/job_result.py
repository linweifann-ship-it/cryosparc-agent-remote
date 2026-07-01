# Builds model-facing CryoSPARC job result packages after execution finishes.
from datetime import datetime, timezone
from time import monotonic, sleep
from typing import Any

from workflow_state import ACTIVE_STATUSES, extract_workflow_state, find_node


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
    workflow_state = extract_workflow_state(project_uid, workspace_uid)
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
    status_group = (
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
        "message": (
            "Job needs human attention; keep this status inside MCP and do not "
            "ask the model for the next decision yet."
            if monitoring["attention_required"]
            else (
                "Job is not finished; keep this status inside MCP and do not ask "
                "the model for the next decision yet."
            )
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
        "has_error": node["has_error"],
        "has_warning": node["has_warning"],
        "inputs": node["inputs"],
        "outputs": summarize_outputs(node),
        "metrics": build_basic_metrics(node),
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
        },
        "next_candidate_actions": [],
        "blocked_actions": [],
        "output_contract": {
            "return_json_only": True,
            "schema_version": "1.0",
            "allowed_decision_type": ["forward", "rollback", "branch", "stop"],
            "selected_actions_rule": (
                "Every selected action_id must come from next_candidate_actions."
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
