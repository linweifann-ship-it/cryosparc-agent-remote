# Adapts V2 model decisions into the existing internal execution contract.
from typing import Any

from action_registry import execute_model_decision_payload, get_candidate_actions


def adapt_v2_decision_to_internal(
    v2_decision: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert a V2 model decision into the internal schema v1 decision."""
    decision_type = v2_decision.get("decision_type")
    if decision_type in {"stop", "rollback"}:
        return adapt_non_action_decision(v2_decision)
    if decision_type not in {"forward", "branch"}:
        return {
            "success": False,
            "issues": [
                issue(
                    "unsupported_decision_type",
                    "decision_type must be forward, branch, rollback, or stop.",
                    "decision_type",
                )
            ],
        }

    requested_actions = normalize_requested_actions(v2_decision)
    if not requested_actions:
        return {
            "success": False,
            "issues": [
                issue(
                    "missing_action",
                    "forward/branch V2 decisions require action or selected_actions.",
                    "action",
                )
            ],
        }

    selected_actions = []
    issues = []
    for index, requested in enumerate(requested_actions):
        candidate = match_candidate(requested, candidate_actions)
        if candidate is None:
            issues.append(
                issue(
                    "candidate_not_found",
                    (
                        "No internal candidate action matches the V2 request "
                        f"{requested!r}."
                    ),
                    f"selected_actions.{index}",
                )
            )
            continue
        selected_actions.append(
            {
                "action_id": candidate["action_id"],
                "action_type": candidate["action_type"],
                "workflow_node_id": candidate["workflow_node_id"],
                "job_type": candidate["job_type"],
                "parameters": requested.get("parameters") or {},
            }
        )

    if issues:
        return {"success": False, "issues": issues}

    return {
        "success": True,
        "internal_decision": {
            "schema_version": "1.0",
            "decision_type": decision_type,
            "selected_actions": selected_actions,
            "rollback_target": None,
            "branch_plan": v2_decision.get("branch_plan"),
            "reason": v2_decision.get("reason") or "V2 model decision.",
            "confidence": v2_decision.get("confidence", 0.0),
            "risk_flags": v2_decision.get("risk_flags") or [],
            "evidence": v2_decision.get("evidence") or [],
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


def adapt_non_action_decision(v2_decision: dict[str, Any]) -> dict[str, Any]:
    """Convert V2 stop/rollback decisions into the internal schema."""
    decision_type = v2_decision.get("decision_type")
    rollback_target = v2_decision.get("rollback_target")
    return {
        "success": True,
        "internal_decision": {
            "schema_version": "1.0",
            "decision_type": decision_type,
            "selected_actions": [],
            "rollback_target": rollback_target,
            "branch_plan": None,
            "reason": v2_decision.get("reason") or f"V2 {decision_type} decision.",
            "confidence": v2_decision.get("confidence", 0.0),
            "risk_flags": v2_decision.get("risk_flags") or [],
            "evidence": v2_decision.get("evidence") or [],
        },
    }


def normalize_requested_actions(v2_decision: dict[str, Any]) -> list[dict[str, Any]]:
    """Support both compact single-action and explicit selected_actions shapes."""
    selected = v2_decision.get("selected_actions")
    if isinstance(selected, list):
        return [
            action
            for action in selected
            if isinstance(action, dict)
        ]

    action = v2_decision.get("action") or v2_decision.get("job_type")
    if not action:
        return []
    return [
        {
            "action": action,
            "job_type": v2_decision.get("job_type") or action,
            "parameters": v2_decision.get("parameters") or {},
            "action_id": v2_decision.get("action_id"),
            "workflow_node_id": v2_decision.get("workflow_node_id"),
        }
    ]


def match_candidate(
    requested: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the internal candidate that best matches the V2 model request."""
    action_id = requested.get("action_id")
    if action_id:
        return next(
            (
                candidate
                for candidate in candidate_actions
                if candidate["action_id"] == action_id
            ),
            None,
        )

    workflow_node_id = requested.get("workflow_node_id")
    job_type = requested.get("job_type") or requested.get("action")
    matches = [
        candidate
        for candidate in candidate_actions
        if (
            (not job_type or candidate["job_type"] == job_type)
            and (
                not workflow_node_id
                or candidate["workflow_node_id"] == workflow_node_id
            )
        )
    ]
    if len(matches) == 1:
        return matches[0]
    return None


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
