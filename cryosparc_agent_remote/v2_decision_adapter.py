# Adapts V2 model decisions into the existing internal execution contract.
from typing import Any

from action_registry import execute_model_decision_payload, get_candidate_actions
from schemas import parse_external_model_decision


def adapt_v2_decision_to_internal(
    v2_decision: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
    current_node_id: str | None = None,
) -> dict[str, Any]:
    """Convert a strict v3.0 minimal_v3 decision into internal schema v1."""
    external_decision, schema_issues = parse_external_model_decision(v2_decision)
    if external_decision is None:
        return {
            "success": False,
            "issues": [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in schema_issues
            ],
        }

    decision_type = v2_decision.get("decision_type")
    if decision_type == "stop":
        return adapt_non_action_decision(v2_decision)

    requested_actions = normalize_requested_actions(v2_decision)

    selected_actions = []
    issues = []
    for index, requested in enumerate(requested_actions):
        candidate = match_candidate(requested, candidate_actions)
        if candidate is not None:
            selected_actions.append(
                {
                    "action_id": candidate["action_id"],
                    "action_type": resolve_action_type(decision_type, candidate),
                    "workflow_node_id": candidate["workflow_node_id"],
                    "job_type": candidate["job_type"],
                    "parameters": requested.get("parameters") or {},
                }
            )
            continue
        selected_actions.append(
            build_generic_selected_action(
                requested,
                current_node_id,
                index,
                decision_type=decision_type,
            )
        )

    if issues:
        return {"success": False, "issues": issues}

    internal_decision_type = resolve_internal_decision_type(
        decision_type,
        selected_actions,
    )
    return {
        "success": True,
        "internal_decision": {
            "schema_version": "1.0",
            "decision_type": internal_decision_type,
            "selected_actions": selected_actions,
            "rollback_target": None,
            "branch_plan": build_branch_plan_if_needed(
                internal_decision_type,
                v2_decision,
            ),
            "reason": "External v3.0 minimal model decision.",
            "confidence": 0.0,
            "risk_flags": [],
            "evidence": [],
        },
    }


def execute_v2_model_decision_payload(
    v2_decision: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
    current_node_id: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Adapt a V2 decision and execute it through the existing internal chain."""
    candidate_context = get_candidate_actions(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_node_id=current_node_id,
    )
    adapter_result = adapt_v2_decision_to_internal(
        v2_decision,
        candidate_context["candidate_actions"],
        current_node_id=candidate_context["current_node_id"],
    )
    if not adapter_result["success"]:
        return {
            "success": False,
            "dry_run": dry_run,
            "execution_mode": "v2_adapter_failed",
            "candidate_context": summarize_candidate_context(candidate_context),
            "internal_decision": None,
            "execution_result": None,
            "issues": adapter_result["issues"],
        }

    execution_result = execute_model_decision_payload(
        adapter_result["internal_decision"],
        candidate_actions=candidate_context["candidate_actions"],
        dry_run=dry_run,
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        allow_internal_schema=True,
    )
    return {
        "success": execution_result["success"],
        "dry_run": dry_run,
        "execution_mode": "v2_adapter",
        "candidate_context": summarize_candidate_context(candidate_context),
        "internal_decision": adapter_result["internal_decision"],
        "execution_result": execution_result,
        "issues": execution_result.get("issues", []),
        "warnings": execution_result.get("warnings", []),
    }


def resolve_internal_decision_type(
    v2_decision_type: str,
    selected_actions: list[dict[str, Any]],
) -> str:
    """Map duplicate-child V2 forward choices to internal branch actions."""
    action_types = {action["action_type"] for action in selected_actions}
    if v2_decision_type == "forward" and action_types == {"branch"}:
        return "branch"
    return v2_decision_type


def build_branch_plan_if_needed(
    internal_decision_type: str,
    v2_decision: dict[str, Any],
) -> dict[str, Any] | None:
    """Provide a minimal branch_plan when MCP resolves duplicate references."""
    if internal_decision_type != "branch":
        return None
    return v2_decision.get("branch_plan") or {
        "branch_type": "single_candidate_from_duplicate_reference",
        "max_parallel_branches": 1,
        "notes": "MCP selected one matching internal candidate for a V2 forward decision.",
    }


def adapt_non_action_decision(v2_decision: dict[str, Any]) -> dict[str, Any]:
    """Convert v3 stop decisions into the internal schema."""
    decision_type = v2_decision.get("decision_type")
    return {
        "success": True,
        "internal_decision": {
            "schema_version": "1.0",
            "decision_type": decision_type,
            "selected_actions": [],
            "rollback_target": None,
            "branch_plan": None,
            "reason": f"External v3.0 minimal {decision_type} decision.",
            "confidence": 0.0,
            "risk_flags": [],
            "evidence": [],
        },
    }


def normalize_requested_actions(v2_decision: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the strict v3.0 selected_actions shape."""
    selected = v2_decision.get("selected_actions")
    if isinstance(selected, list):
        return [
            {
                "action": action.get("job_type"),
                "job_type": action.get("job_type"),
                "parameters": action.get("parameters") or {},
            }
            for action in selected
            if isinstance(action, dict)
        ]
    return []


def build_generic_selected_action(
    requested: dict[str, Any],
    current_node_id: str | None,
    index: int,
    decision_type: str,
) -> dict[str, Any]:
    """Build an internal action when no candidate matched the model request."""
    job_type = requested.get("job_type") or requested.get("action")
    return {
        "action_id": f"generic_{index}_{job_type}",
        "action_type": "branch" if decision_type == "branch" else "forward",
        "workflow_node_id": f"{current_node_id or 'current'}:{job_type}",
        "job_type": job_type,
        "parameters": requested.get("parameters") or {},
    }


def match_candidate(
    requested: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the internal candidate that best matches the model job_type."""
    job_type = requested.get("job_type") or requested.get("action")
    matches = [
        candidate
        for candidate in candidate_actions
        if not job_type or candidate["job_type"] == job_type
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return select_preferred_candidate(matches)
    return None


def resolve_action_type(decision_type: str, candidate: dict[str, Any]) -> str:
    if decision_type == "branch":
        return "branch"
    return candidate["action_type"]


def select_preferred_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Break V2 job-type ties using completed, earliest reference jobs first."""
    return sorted(
        candidates,
        key=lambda candidate: (
            status_rank(candidate.get("reference_status")),
            job_uid_number(candidate.get("reference_job_uid") or ""),
            candidate["action_id"],
        ),
    )[0]


def status_rank(status: str | None) -> int:
    """Prefer validated completed references over active or failed duplicates."""
    if status == "completed":
        return 0
    if status in {"queued", "launched", "started", "running"}:
        return 1
    if status in {"failed", "killed"}:
        return 3
    return 2


def job_uid_number(job_uid: str) -> int:
    """Extract the numeric part of CryoSPARC job IDs for stable sorting."""
    if job_uid.startswith("J") and job_uid[1:].isdigit():
        return int(job_uid[1:])
    return 10**12


def summarize_candidate_context(candidate_context: dict[str, Any]) -> dict[str, Any]:
    """Return safe internal matching context without exposing full connections."""
    return {
        "project_uid": candidate_context["project_uid"],
        "workspace_uid": candidate_context["workspace_uid"],
        "current_node_id": candidate_context["current_node_id"],
        "candidate_count": len(candidate_context["candidate_actions"]),
        "candidate_summaries": [
            {
                "action_id": candidate["action_id"],
                "action_type": candidate["action_type"],
                "workflow_node_id": candidate["workflow_node_id"],
                "reference_status": candidate.get("reference_status"),
                "job_type": candidate["job_type"],
            }
            for candidate in candidate_context["candidate_actions"]
        ],
        "blocked_count": len(candidate_context["blocked_actions"]),
        "decision_hint": candidate_context["decision_hint"],
    }


def issue(code: str, message: str, path: str | None = None) -> dict[str, Any]:
    """Build a normalized adapter issue."""
    return {
        "severity": "error",
        "code": code,
        "message": message,
        "path": path,
    }
