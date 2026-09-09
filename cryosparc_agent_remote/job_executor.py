# Plans and eventually executes CryoSPARC jobs through a generic job API adapter.
import os
from time import monotonic, sleep
from typing import Any

from cryosparc_client import cryosparc_client
from job_specs import get_job_spec
from resource_scheduler import (
    apply_resource_overrides,
    build_scheduling_plan,
    make_logical_job_record,
    probe_cluster_resources,
    register_submission,
)


def plan_job_action(
    action: dict[str, Any],
    candidate_action: dict[str, Any] | None = None,
    lane: str | None = None,
) -> dict[str, Any]:
    """Build one dry-run job plan from a validated action and candidate context."""
    job_type = action["job_type"]
    spec = get_job_spec(job_type)
    metadata = (candidate_action or {}).get("job_spec_metadata") or {}
    if metadata:
        spec.update({
            key: value
            for key, value in metadata.items()
            if value is not None
        })
    connections = build_explicit_connections(action.get("connections"))
    if connections is None:
        connections = build_connections(
            (candidate_action or {}).get("required_inputs", {})
        )
    scheduling = build_scheduling_plan(
        job_type=job_type,
        spec=spec,
        params=action["resolved_parameters"],
        requested_lane=lane,
    )
    queue = scheduling["queue"]
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
        "resource_scheduling": scheduling,
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
    configured_lane = os.getenv("CRYOAGENT_GPU_LANE")
    selected_lane = (
        lane
        or (configured_lane if spec.get("requires_gpu") else None)
        or spec.get("default_lane")
    )
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
    allow_approval_required_create: bool = False,
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

    if planned_action["approval_required"] and not allow_approval_required_create:
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

    scheduling = refresh_scheduling_plan(planned_action)
    planned_action = {
        **planned_action,
        "queue": scheduling["queue"],
        "resource_scheduling": scheduling,
    }
    params = apply_resource_overrides(
        planned_action["resolved_parameters"],
        scheduling.get("parameter_overrides") or {},
    )
    logical_record = make_logical_job_record(planned_action)
    cryosparc_payload = build_cryosparc_payload(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        planned_action=planned_action,
        params=params,
    )
    if should_race_submit(planned_action):
        return execute_race_job_action(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            planned_action=planned_action,
            params=params,
            logical_record=logical_record,
        )
    job = None
    execution_phase = "create_job"
    try:
        cs = cryosparc_client()
        workspace = cs.find_workspace(project_uid, workspace_uid)
        job = workspace.create_job(
            cryosparc_payload["create_job"]["job_type"],
            connections=cryosparc_payload["create_job"]["connections"],
            params=cryosparc_payload["create_job"]["params"],
            title=cryosparc_payload["create_job"]["title"],
            desc=cryosparc_payload["create_job"]["desc"],
        )
        queue = planned_action["queue"]
        queued = False
        if queue["will_queue"]:
            execution_phase = "queue_job"
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
        register_submission(
            logical_record,
            job.uid,
            scheduling["resource_config"],
            scheduling["snapshot"],
            scheduling["reason"],
        )

        return {
            "success": True,
            "dry_run": False,
            "status": "queued" if queued else job.status,
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_uid": job.uid,
            "job_type": planned_action["job_type"],
            "queued": queued,
            "approval_required": planned_action["approval_required"],
            "approval_reasons": planned_action.get("approval_reasons", []),
            "approval_required_bypassed_for_creation": bool(
                planned_action["approval_required"] and allow_approval_required_create
            ),
            "human_action_required": bool(planned_action["approval_required"]),
            "planned_action": planned_action,
            "diagnostics": {
                "execution_phase": "completed",
                "cryosparc_payload": cryosparc_payload,
                "http_response": None,
            },
            "resource_scheduling": {
                "logical_job": logical_record.__dict__,
                "effective_parameters": params,
            },
        }
    except Exception as exc:
        http_response = extract_http_response(exc)
        return {
            "success": False,
            "dry_run": False,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_uid": getattr(job, "uid", None),
            "job_type": planned_action["job_type"],
            "queued": False,
            "planned_action": planned_action,
            "diagnostics": {
                "execution_phase": execution_phase,
                "cryosparc_payload": cryosparc_payload,
                "http_response": http_response,
            },
            "issues": [
                {
                    "severity": "error",
                    "code": "cryosparc_execution_error",
                    "message": str(exc),
                    "path": "cryoSPARC",
                }
            ],
            "resource_scheduling": {
                "logical_job": logical_record.__dict__,
                "effective_parameters": params,
            },
        }


def should_race_submit(planned_action: dict[str, Any]) -> bool:
    scheduling = planned_action.get("resource_scheduling") or {}
    policy = scheduling.get("policy") or {}
    queue = planned_action.get("queue") or {}
    race_lanes = scheduling.get("race_lanes") or []
    return bool(policy.get("race_mode") and queue.get("will_queue") and len(race_lanes) > 1)


def execute_race_job_action(
    project_uid: str,
    workspace_uid: str,
    planned_action: dict[str, Any],
    params: dict[str, Any],
    logical_record: Any,
) -> dict[str, Any]:
    """Submit one logical GPU step redundantly to multiple physical lanes."""
    scheduling = planned_action["resource_scheduling"]
    physical_jobs = []
    execution_phase = "create_race_jobs"
    try:
        cs = cryosparc_client()
        workspace = cs.find_workspace(project_uid, workspace_uid)
        for index, lane in enumerate(scheduling["race_lanes"], 1):
            lane_scheduling = refresh_scheduling_plan({
                **planned_action,
                "queue": {**planned_action["queue"], "lane": lane},
                "resource_scheduling": scheduling,
            })
            lane_queue = lane_scheduling["queue"]
            lane_action = {
                **planned_action,
                "queue": lane_queue,
                "resource_scheduling": lane_scheduling,
            }
            payload = build_cryosparc_payload(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                planned_action=lane_action,
                params=params,
            )
            job = workspace.create_job(
                payload["create_job"]["job_type"],
                connections=payload["create_job"]["connections"],
                params=payload["create_job"]["params"],
                title=f"{payload['create_job']['title']} physical_{index}",
                desc=payload["create_job"]["desc"],
            )
            execution_phase = f"queue_race_job_{index}"
            job.queue(
                lane=lane_queue["lane"],
                hostname=lane_queue["hostname"],
                gpus=lane_queue["gpus"],
                cluster_vars=lane_queue["cluster_vars"],
            )
            register_submission(
                logical_record,
                job.uid,
                lane_scheduling["resource_config"],
                lane_scheduling["snapshot"],
                lane_scheduling["reason"],
            )
            physical_jobs.append({
                "job_uid": job.uid,
                "job_type": planned_action["job_type"],
                "status": "queued",
                "queued": True,
                "lane": lane_queue["lane"],
                "resource_scheduling": lane_scheduling,
            })
            early_race_result = reconcile_race_jobs_once(
                workspace,
                [physical_job["job_uid"] for physical_job in physical_jobs],
            )
            if early_race_result.get("winner_job_uid"):
                physical_jobs[-1]["status"] = (
                    early_race_result.get("statuses", {}).get(job.uid)
                    or physical_jobs[-1]["status"]
                )
        race_result = wait_for_race_winner_and_kill_losers(
            workspace,
            [job["job_uid"] for job in physical_jobs],
            poll_interval_seconds=int(
                (scheduling.get("policy") or {}).get(
                    "race_poll_interval_seconds",
                    20,
                )
            ),
            timeout_seconds=int(
                (scheduling.get("policy") or {}).get(
                    "queue_start_timeout_seconds",
                    600,
                )
            ),
        )
        logical_job = {
            "logical_job_id": logical_record.logical_job_id,
            "logical_job_uid": race_result.get("winner_job_uid")
            or (physical_jobs[0]["job_uid"] if physical_jobs else None),
            "physical_job_ids": [job["job_uid"] for job in physical_jobs],
            "physical_jobs": physical_jobs,
            "winner_job_uid": race_result.get("winner_job_uid"),
            "cancellations": race_result.get("cancellations", []),
            "race_statuses": race_result.get("statuses", {}),
        }
        return {
            "success": True,
            "dry_run": False,
            "status": "queued",
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_uid": logical_job["logical_job_uid"],
            "job_type": planned_action["job_type"],
            "queued": True,
            "logical_workflow_step": True,
            "physical_execution_redundancy": "race",
            "logical_job": logical_job,
            "approval_required": planned_action["approval_required"],
            "approval_reasons": planned_action.get("approval_reasons", []),
            "approval_required_bypassed_for_creation": False,
            "human_action_required": bool(planned_action["approval_required"]),
            "planned_action": planned_action,
            "diagnostics": {
                "execution_phase": "completed",
                "physical_jobs": physical_jobs,
                "race_reconciliation": race_result,
                "http_response": None,
            },
            "resource_scheduling": {
                "logical_job": logical_record.__dict__,
                "effective_parameters": params,
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "dry_run": False,
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "project_uid": project_uid,
            "workspace_uid": workspace_uid,
            "job_uid": physical_jobs[0]["job_uid"] if physical_jobs else None,
            "job_type": planned_action["job_type"],
            "queued": False,
            "logical_workflow_step": True,
            "physical_execution_redundancy": "race",
            "physical_jobs": physical_jobs,
            "planned_action": planned_action,
            "diagnostics": {
                "execution_phase": execution_phase,
                "physical_jobs": physical_jobs,
                "http_response": extract_http_response(exc),
            },
            "issues": [
                {
                    "severity": "error",
                    "code": "cryosparc_race_execution_error",
                    "message": str(exc),
                    "path": "cryoSPARC",
                }
            ],
            "resource_scheduling": {
                "logical_job": logical_record.__dict__,
                "effective_parameters": params,
            },
        }


def wait_for_race_winner_and_kill_losers(
    workspace: Any,
    job_uids: list[str],
    poll_interval_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = monotonic() + max(timeout_seconds, 0)
    poll_count = 0
    while True:
        poll_count += 1
        result = reconcile_race_jobs_once(workspace, job_uids)
        if result.get("winner_job_uid"):
            result["poll_count"] = poll_count
            return result
        if timeout_seconds <= 0 or monotonic() >= deadline:
            jobs_by_uid = {job.uid: job for job in workspace.find_jobs()}
            statuses = get_race_statuses(jobs_by_uid, job_uids)
            return {
                "winner_job_uid": None,
                "statuses": statuses,
                "cancellations": [],
                "poll_count": poll_count,
                "timed_out": True,
            }
        sleep(max(poll_interval_seconds, 1))


def reconcile_race_jobs_once(
    workspace: Any,
    job_uids: list[str],
) -> dict[str, Any]:
    jobs_by_uid = {job.uid: job for job in workspace.find_jobs()}
    statuses = get_race_statuses(jobs_by_uid, job_uids)
    winner = select_race_winner(statuses)
    cancellations = (
        kill_non_running_race_losers(jobs_by_uid, winner, statuses)
        if winner
        else []
    )
    return {
        "winner_job_uid": winner,
        "statuses": statuses,
        "cancellations": cancellations,
    }


def get_race_statuses(
    jobs_by_uid: dict[str, Any],
    job_uids: list[str],
) -> dict[str, str]:
    return {
        job_uid: str(getattr(jobs_by_uid.get(job_uid), "status", "not_found"))
        for job_uid in job_uids
    }


def select_race_winner(statuses: dict[str, str]) -> str | None:
    normalized = {
        job_uid: str(status or "").lower()
        for job_uid, status in statuses.items()
    }
    for target_status in ("completed", "running", "started"):
        for job_uid, status in normalized.items():
            if status == target_status:
                return job_uid
    return None


def kill_non_running_race_losers(
    jobs_by_uid: dict[str, Any],
    winner_job_uid: str,
    statuses: dict[str, str],
) -> list[dict[str, Any]]:
    keep_statuses = {"completed", "running", "started", "failed", "killed"}
    results = []
    for job_uid, raw_status in statuses.items():
        status = str(raw_status or "").lower()
        if job_uid == winner_job_uid or status in keep_statuses:
            continue
        job = jobs_by_uid.get(job_uid)
        if job is None:
            results.append({
                "job_uid": job_uid,
                "previous_status": status,
                "action": "kill",
                "success": False,
                "error": "job_not_found",
            })
            continue
        results.append(cancel_physical_race_job(job, job_uid, status))
    return results


def cancel_physical_race_job(
    job: Any,
    job_uid: str,
    status: str,
) -> dict[str, Any]:
    errors = []
    for method_name in ("kill", "cancel"):
        method = getattr(job, method_name, None)
        if method is None:
            errors.append({
                "method": method_name,
                "error_type": "AttributeError",
                "error": f"job has no {method_name} method",
            })
            continue
        try:
            method()
            return {
                "job_uid": job_uid,
                "previous_status": status,
                "action": "kill",
                "method": method_name,
                "success": True,
            }
        except Exception as exc:
            errors.append({
                "method": method_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    return {
        "job_uid": job_uid,
        "previous_status": status,
        "action": "kill",
        "success": False,
        "attempts": errors,
    }


def refresh_scheduling_plan(planned_action: dict[str, Any]) -> dict[str, Any]:
    """Refresh resource state immediately before live submission."""
    existing = planned_action.get("resource_scheduling") or {}
    queue = planned_action.get("queue") or {}
    spec = get_job_spec(planned_action["job_type"])
    return build_scheduling_plan(
        job_type=planned_action["job_type"],
        spec=spec,
        params=planned_action["resolved_parameters"],
        requested_lane=queue.get("lane") or existing.get("selected_lane"),
        resource_snapshot=probe_cluster_resources(),
    )


def build_cryosparc_payload(
    project_uid: str,
    workspace_uid: str,
    planned_action: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record the cryosparc-tools operation payload before execution."""
    queue = planned_action["queue"]
    return {
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "create_job": {
            "job_type": planned_action["job_type"],
            "connections": planned_action["connections"],
            "params": params if params is not None else planned_action["resolved_parameters"],
            "title": f"Agent {planned_action['action_id']}",
            "desc": "Created by cryosparc_agent execute_model_decision.",
        },
        "queue": {
            "will_queue": queue["will_queue"],
            "lane": queue["lane"],
            "hostname": queue["hostname"],
            "gpus": queue["gpus"],
            "cluster_vars": queue["cluster_vars"],
        },
    }


def extract_http_response(exc: Exception) -> dict[str, Any] | None:
    """Best-effort extraction for httpx/cryosparc-tools HTTP errors."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    body = None
    try:
        body = response.text
    except Exception:
        try:
            body = response.content.decode("utf-8", errors="replace")
        except Exception:
            body = None
    parsed_body: Any = None
    if body:
        try:
            import json

            parsed_body = json.loads(body)
        except Exception:
            parsed_body = None
    request = getattr(response, "request", None)
    return {
        "status_code": getattr(response, "status_code", None),
        "reason_phrase": getattr(response, "reason_phrase", None),
        "url": str(getattr(request, "url", "")) if request is not None else None,
        "method": getattr(request, "method", None) if request is not None else None,
        "body": body,
        "json": parsed_body,
    }
