# Plans and eventually executes CryoSPARC jobs through a generic job API adapter.
from typing import Any

from cryosparc_client import cryosparc_client
from job_specs import get_job_spec


def plan_job_action(
    action: dict[str, Any],
    candidate_action: dict[str, Any] | None = None,
    lane: str | None = None,
) -> dict[str, Any]:
    """Build one dry-run job plan from a validated action and candidate context."""
    job_type = action["job_type"]
    spec = get_job_spec(job_type)
    connections = build_explicit_connections(action.get("connections"))
    if connections is None:
        connections = build_connections(
            (candidate_action or {}).get("required_inputs", {})
        )
    queue = build_queue_plan(spec, lane)
    approval_reasons = approval_reasons_for(action, spec)

    return {
        "plan_step": action["plan_step"],
        "action_id": action["action_id"],
        "action_type": action["action_type"],
        "workflow_node_id": action["workflow_node_id"],
        "job_type": job_type,
        "job_category": spec["category"],
        "execution_mode": action["execution_mode"],
        "mcp_tool_name": action.get("mcp_tool_name"),
        "connections": connections,
        "resolved_parameters": action["resolved_parameters"],
        "queue": queue,
        "requires_gpu": spec["requires_gpu"],
        "interactive": spec["interactive"],
        "approval_required": bool(
            action.get("approval_required")
            or spec["requires_approval"]
            or approval_reasons
        ),
        "approval_reasons": sorted(
            set(action.get("approval_reasons", []) + approval_reasons)
        ),
        "rollback_target": action.get("rollback_target"),
        "status": "planned",
    }


def build_connections(
    required_inputs: dict[str, list[dict[str, Any]]],
) -> dict[str, tuple[str, str] | list[tuple[str, str]]]:
    """Convert registry input metadata into cryosparc-tools connections."""
    connections: dict[str, tuple[str, str] | list[tuple[str, str]]] = {}
    for input_name, input_connections in required_inputs.items():
        values = [
            (
                connection["source_job_uid"],
                connection["source_output"],
            )
            for connection in input_connections
        ]
        if not values:
            continue
        connections[input_name] = values[0] if len(values) == 1 else values
    return connections


def build_explicit_connections(
    raw_connections: dict[str, Any] | None,
) -> dict[str, tuple[str, str] | list[tuple[str, str]]] | None:
    """Normalize model-supplied CryoSPARC connections, if present."""
    if raw_connections is None:
        return None
    connections: dict[str, tuple[str, str] | list[tuple[str, str]]] = {}
    for input_name, raw_value in raw_connections.items():
        values = normalize_connection_values(raw_value)
        if not values:
            continue
        connections[input_name] = values[0] if len(values) == 1 else values
    return connections


def normalize_connection_values(raw_value: Any) -> list[tuple[str, str]]:
    """Accept compact or object-shaped model connection values."""
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    normalized = []
    for value in values:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            normalized.append((str(value[0]), str(value[1])))
            continue
        if not isinstance(value, dict):
            continue
        source_job = (
            value.get("source_job_uid")
            or value.get("source_job")
            or value.get("job_uid")
        )
        source_output = value.get("source_output") or value.get("output")
        if source_job and source_output:
            normalized.append((str(source_job), str(source_output)))
    return normalized


def build_queue_plan(spec: dict[str, Any], lane: str | None) -> dict[str, Any]:
    """Prepare queue settings without submitting anything to CryoSPARC."""
    selected_lane = lane or spec.get("default_lane")
    return {
        "lane": selected_lane,
        "hostname": None,
        "gpus": [],
        "cluster_vars": {},
        "will_queue": not spec["interactive"],
    }


def approval_reasons_for(
    action: dict[str, Any],
    spec: dict[str, Any],
) -> list[str]:
    """Explain why a planned action needs human approval."""
    reasons: list[str] = []
    if spec["requires_approval"]:
        reasons.append("job_spec_requires_approval")
    if spec["interactive"]:
        reasons.append("interactive_job")
    if action["action_type"] == "branch":
        reasons.append("branch_decision")
    gpu_count = action["resolved_parameters"].get("compute_num_gpus")
    if (
        spec["requires_gpu"]
        and isinstance(gpu_count, int)
        and not isinstance(gpu_count, bool)
        and gpu_count > spec["max_auto_gpus"]
    ):
        reasons.append("high_gpu_count")
    return reasons


def execute_job_action(
    project_uid: str,
    workspace_uid: str,
    planned_action: dict[str, Any],
    dry_run: bool = True,
) -> dict[str, Any]:
    """
    Execute one planned job action.

    Dry-run mode is the default and only returns the plan. Live mode is kept
    behind approval gates and should be enabled only after policy is added.
    """
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "status": "planned",
            "planned_action": planned_action,
            "message": "Dry run only; no CryoSPARC job was created or queued.",
        }

    if planned_action["approval_required"]:
        return {
            "success": False,
            "dry_run": False,
            "status": "approval_required",
            "planned_action": planned_action,
            "issues": [
                {
                    "severity": "error",
                    "code": "approval_required",
                    "message": "Live execution requires human approval.",
                    "path": None,
                }
            ],
        }

    try:
        cs = cryosparc_client()
        workspace = cs.find_workspace(project_uid, workspace_uid)
        job = workspace.create_job(
            planned_action["job_type"],
            connections=planned_action["connections"],
            params=planned_action["resolved_parameters"],
            title=f"Agent {planned_action['action_id']}",
            desc="Created by cryosparc_agent execute_model_decision.",
        )
        queue = planned_action["queue"]
        queued = False
        if queue["will_queue"]:
            if queue["lane"]:
                job.queue(
                    lane=queue["lane"],
                    hostname=queue["hostname"],
                    gpus=queue["gpus"],
                    cluster_vars=queue["cluster_vars"],
                )
            else:
                job.queue()
            queued = True

        return {
            "success": True,
            "dry_run": False,
            "status": "queued" if queued else job.status,
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_uid": job.uid,
            "job_type": planned_action["job_type"],
            "queued": queued,
            "planned_action": planned_action,
        }
    except Exception as exc:
        return {
            "success": False,
            "dry_run": False,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "planned_action": planned_action,
        }
