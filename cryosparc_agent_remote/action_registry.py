# Builds candidate CryoSPARC actions, validates model decisions, and plans safe execution.
from typing import Any

from schemas import (
    Action,
    ModelDecision,
    ResolvedAction,
    ValidationIssue,
    ValidationResult,
    parse_model_decision,
)
from job_executor import execute_job_action, plan_job_action
from job_specs import get_parameter_template
from workflow_state import content_hash, extract_workflow_state, find_node


REGISTRY_VERSION = "workflow_v1"

def get_candidate_actions(
    project_uid: str,
    workspace_uid: str,
    current_node_id: str | None = None,
) -> dict[str, Any]:
    """Read the current workflow and return selectable next actions."""
    workflow_state = extract_workflow_state(project_uid, workspace_uid)
    current_node = (
        find_node(workflow_state, current_node_id)
        if current_node_id
        else None
    )
    canonical_current_node_id = (
        current_node["workflow_node_id"]
        if current_node
        else current_node_id
    )
    candidate_actions, blocked_actions = generate_candidate_actions(
        workflow_state,
        canonical_current_node_id,
    )
    return {
        "schema_version": "1.0",
        "registry_version": REGISTRY_VERSION,
        "generated_at": workflow_state["generated_at"],
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "current_node_id": canonical_current_node_id,
        "requested_current_node_id": current_node_id,
        "workflow_status": workflow_state["workflow_status"],
        "candidate_actions": candidate_actions,
        "blocked_actions": blocked_actions,
        "decision_hint": "stop" if not candidate_actions else None,
    }


def generate_candidate_actions(
    workflow_state: dict[str, Any],
    current_node_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn the selected node's child jobs into forward or branch candidates."""
    if not current_node_id:
        return [], []

    current_node = find_node(workflow_state, current_node_id)
    if current_node is None:
        return [], [
            {
                "action_id": None,
                "available": False,
                "blocked_by": [f"Unknown current_node_id: {current_node_id}"],
            }
        ]

    child_uids = current_node["child_job_uids"]
    child_nodes = [
        find_node(workflow_state, child_uid)
        for child_uid in child_uids
    ]
    child_nodes = [
        node
        for node in child_nodes
        if node is not None and node["status"] not in {"failed", "killed"}
    ]

    is_branch = len(child_nodes) > 1
    candidates: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for child in child_nodes:
        action_type = "branch" if is_branch else "forward"
        action = build_candidate_action(
            workflow_state,
            child,
            action_type,
        )
        if action["available"]:
            candidates.append(action)
        else:
            blocked.append(action)

    candidates.sort(key=lambda action: action["action_id"])
    blocked.sort(key=lambda action: action["action_id"])
    return candidates, blocked


def build_candidate_action(
    workflow_state: dict[str, Any],
    target_node: dict[str, Any],
    action_type: str,
) -> dict[str, Any]:
    """Build one candidate action and record why it is unavailable, if blocked."""
    required_inputs: dict[str, list[dict[str, Any]]] = {}
    blocked_by: list[str] = []

    for input_name, connections in target_node["inputs"].items():
        required_inputs[input_name] = connections
        for connection in connections:
            source_node = find_node(
                workflow_state,
                connection["source_job_uid"],
            )
            if source_node is None:
                blocked_by.append(
                    f"Missing source job {connection['source_job_uid']}"
                )
                continue
            if source_node["status"] != "completed":
                blocked_by.append(
                    f"Source job {source_node['cryosparc_job_uid']} is "
                    f"{source_node['status']}, not completed"
                )
            source_output = source_node["outputs"].get(
                connection["source_output"]
            )
            if not source_output or not source_output["available"]:
                blocked_by.append(
                    f"Output {connection['source_job_uid']}."
                    f"{connection['source_output']} is unavailable"
                )

    parameter_template = build_parameter_template(target_node)

    return {
        "action_id": f"{action_type}_{target_node['cryosparc_job_uid']}",
        "action_type": action_type,
        "workflow_node_id": target_node["workflow_node_id"],
        "reference_job_uid": target_node["cryosparc_job_uid"],
        "job_type": target_node["job_type"],
        "description": (
            f"Reproduce the validated P2-W3 workflow node "
            f"{target_node['cryosparc_job_uid']}."
        ),
        "execution_mode": "dry_run_only",
        "available": not blocked_by,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "parameter_template": parameter_template,
        "default_parameters": target_node["key_parameters"],
    }


def build_parameter_template(
    target_node: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Combine static parameter rules with defaults copied from the reference job."""
    template = get_parameter_template(target_node["job_type"])
    for name, value in target_node["key_parameters"].items():
        template.setdefault(name, {"type": infer_json_type(value)})
        template[name]["default"] = value
    return template


def validate_model_decision_payload(
    payload: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate a model decision without creating or enqueueing CryoSPARC jobs."""
    decision, schema_issues = parse_model_decision(payload)
    if decision is None:
        result = ValidationResult(
            success=False,
            valid_schema=False,
            valid_actions=False,
            issues=schema_issues,
        )
        return result.model_dump()

    result = validate_decision_against_registry(
        decision,
        candidate_actions=candidate_actions,
    )

    return result.model_dump()


def execute_model_decision_payload(
    payload: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
    dry_run: bool = True,
    project_uid: str | None = None,
    workspace_uid: str | None = None,
) -> dict[str, Any]:
    """
    Convert a validated decision into an execution plan.

    The current implementation is intentionally dry-run first. It verifies the
    decision and returns the jobs that would be executed, but it does not create
    or queue CryoSPARC jobs.
    """
    validation = validate_model_decision_payload(
        payload,
        candidate_actions=candidate_actions,
    )
    if not validation["success"]:
        return {
            "success": False,
            "dry_run": dry_run,
            "execution_mode": "validation_failed",
            "decision_type": validation.get("decision_type"),
            "validation": validation,
            "execution_plan": None,
            "execution_results": [],
            "issues": validation["issues"],
            "warnings": validation["warnings"],
        }

    execution_plan = build_execution_plan(payload, validation, candidate_actions)
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "execution_mode": "dry_run",
            "decision_type": validation["decision_type"],
            "validation": validation,
            "execution_plan": execution_plan,
            "execution_results": [],
            "issues": [],
            "warnings": validation["warnings"],
            "message": "Dry run only; no CryoSPARC jobs were created or queued.",
        }

    if not project_uid or not workspace_uid:
        return {
            "success": False,
            "dry_run": False,
            "execution_mode": "missing_execution_context",
            "decision_type": validation["decision_type"],
            "validation": validation,
            "execution_plan": execution_plan,
            "execution_results": [],
            "issues": [
                {
                    "severity": "error",
                    "code": "missing_execution_context",
                    "message": "project_uid and workspace_uid are required for live execution.",
                    "path": None,
                }
            ],
            "warnings": validation["warnings"],
        }

    execution_results = [
        execute_job_action(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            planned_action=action,
            dry_run=False,
        )
        for action in execution_plan["actions"]
    ]
    return {
        "success": all(result["success"] for result in execution_results),
        "dry_run": False,
        "execution_mode": "live_execution",
        "model_visible": False,
        "next_model_input": None,
        "decision_type": validation["decision_type"],
        "validation": validation,
        "execution_plan": execution_plan,
        "execution_results": execution_results,
        "issues": [
            issue
            for result in execution_results
            for issue in result.get("issues", [])
        ],
        "warnings": validation["warnings"],
        "message": (
            "Live execution status is MCP-internal. Wait for the created job to "
            "complete, then call get_job_result_package before asking the model "
            "for the next decision."
        ),
    }


def build_execution_plan(
    payload: dict[str, Any],
    validation: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize validated decisions into a future execution-ready plan."""
    decision_type = validation["decision_type"]
    candidates_by_id = {
        action["action_id"]: action
        for action in candidate_actions or []
    }

    if decision_type in {"forward", "branch"}:
        actions = []
        for idx, action in enumerate(validation["resolved_actions"]):
            base_action = {
                "plan_step": idx + 1,
                "action_id": action["action_id"],
                "action_type": action["action_type"],
                "workflow_node_id": action["workflow_node_id"],
                "job_type": action["job_type"],
                "execution_mode": action["execution_mode"],
                "mcp_tool_name": action["mcp_tool_name"],
                "approval_required": False,
                "approval_reasons": [],
                "resolved_parameters": action["resolved_parameters"],
                "rollback_target": None,
                "status": "planned",
            }
            actions.append(
                plan_job_action(
                    base_action,
                    candidate_action=candidates_by_id.get(action["action_id"]),
                )
            )
        return make_execution_plan(payload, validation, actions)

    if decision_type == "rollback":
        rollback_target = payload.get("rollback_target")
        actions = [
            {
                "plan_step": 1,
                "action_id": None,
                "action_type": "rollback",
                "workflow_node_id": (
                    rollback_target or {}
                ).get("workflow_node_id"),
                "job_type": (rollback_target or {}).get("job_type"),
                "execution_mode": "dry_run_only",
                "mcp_tool_name": None,
                "approval_required": True,
                "approval_reasons": ["rollback_decision"],
                "resolved_parameters": {},
                "rollback_target": payload.get("rollback_target"),
                "status": "planned",
            }
        ]
        return make_execution_plan(payload, validation, actions)

    if decision_type == "stop":
        actions = [
            {
                "plan_step": 1,
                "action_id": None,
                "action_type": "stop",
                "workflow_node_id": None,
                "job_type": None,
                "execution_mode": "dry_run_only",
                "mcp_tool_name": None,
                "approval_required": False,
                "approval_reasons": [],
                "resolved_parameters": {},
                "rollback_target": None,
                "status": "planned",
            }
        ]
        return make_execution_plan(payload, validation, actions)

    return make_execution_plan(payload, validation, [])


def make_execution_plan(
    payload: dict[str, Any],
    validation: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the stable top-level ExecutionPlan object."""
    approval_reasons = sorted(
        {
            reason
            for action in actions
            for reason in action["approval_reasons"]
        }
    )
    plan_payload = {
        "decision_type": validation["decision_type"],
        "actions": actions,
    }
    return {
        "plan_id": content_hash("plan", plan_payload),
        "plan_version": "1.0",
        "status": "planned",
        "dry_run_only": True,
        "decision_type": validation["decision_type"],
        "action_count": len(actions),
        "approval_required": any(
            action["approval_required"]
            for action in actions
        ),
        "approval_reasons": approval_reasons,
        "actions": actions,
        "execution_results": [],
    }


def validate_decision_against_registry(
    decision: ModelDecision,
    candidate_actions: list[dict[str, Any]] | None = None,
) -> ValidationResult:
    """Check that selected actions are present in the candidate registry."""
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    resolved_actions: list[ResolvedAction] = []

    if decision.decision_type in {"rollback", "stop"}:
        return ValidationResult(
            success=True,
            valid_schema=True,
            valid_actions=True,
            decision_type=decision.decision_type,
            dry_run=True,
        )

    candidates_by_id = {
        action["action_id"]: action
        for action in candidate_actions or []
    }

    for idx, action in enumerate(decision.selected_actions):
        action_issues, action_warnings, resolved_action = (
            validate_action_against_candidates(
                action,
                candidates_by_id,
                idx,
            )
        )
        issues.extend(action_issues)
        warnings.extend(action_warnings)
        if resolved_action:
            resolved_actions.append(resolved_action)

    return ValidationResult(
        success=not issues,
        valid_schema=True,
        valid_actions=not issues,
        decision_type=decision.decision_type,
        dry_run=True,
        issues=issues,
        warnings=warnings,
        resolved_actions=resolved_actions,
    )


def validate_action_against_candidates(
    action: Action,
    candidates_by_id: dict[str, dict[str, Any]],
    index: int,
) -> tuple[list[ValidationIssue], list[ValidationIssue], ResolvedAction | None]:
    """Validate one selected action and resolve its final parameters."""
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    path = f"selected_actions.{index}"
    candidate = candidates_by_id.get(action.action_id)

    if candidate is None:
        issues.append(
            ValidationIssue(
                code="unknown_action_id",
                message=(
                    f"action_id {action.action_id!r} is not in the supplied "
                    "candidate_actions"
                ),
                path=f"{path}.action_id",
            )
        )
        return issues, warnings, None

    for field_name in ("action_type", "workflow_node_id", "job_type"):
        expected = candidate[field_name]
        actual = getattr(action, field_name)
        if actual != expected:
            issues.append(
                ValidationIssue(
                    code=f"{field_name}_mismatch",
                    message=(
                        f"{field_name} must be {expected!r} for "
                        f"action_id {action.action_id!r}"
                    ),
                    path=f"{path}.{field_name}",
                )
            )

    parameter_issues, resolved_parameters = validate_parameters(
        action.parameters,
        candidate["parameter_template"],
        path=f"{path}.parameters",
    )
    issues.extend(parameter_issues)

    warnings.append(
        ValidationIssue(
            severity="warning",
            code="dry_run_only_action",
            message=f"job_type {action.job_type!r} is registered for validation only",
            path=path,
        )
    )

    if issues:
        return issues, warnings, None

    return (
        issues,
        warnings,
        ResolvedAction(
            action_id=action.action_id,
            action_type=action.action_type,
            workflow_node_id=action.workflow_node_id,
            job_type=action.job_type,
            execution_mode=candidate["execution_mode"],
            resolved_parameters=resolved_parameters,
            mcp_tool_name=None,
        ),
    )


def validate_parameters(
    parameters: dict[str, Any],
    template: dict[str, dict[str, Any]],
    path: str,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Merge defaults with model-supplied parameters and validate the result."""
    issues: list[ValidationIssue] = []
    resolved = {
        name: spec["default"]
        for name, spec in template.items()
        if "default" in spec
    }
    resolved.update(parameters)

    for name, spec in template.items():
        if spec.get("required") and name not in resolved:
            issues.append(
                ValidationIssue(
                    code="missing_required_parameter",
                    message=f"Missing required parameter {name!r}",
                    path=f"{path}.{name}",
                )
            )

    for name, value in parameters.items():
        spec = template.get(name)
        if spec is None:
            issues.append(
                ValidationIssue(
                    code="unknown_parameter",
                    message=f"Parameter {name!r} is not allowed for this action",
                    path=f"{path}.{name}",
                )
            )
            continue

        issues.extend(
            validate_parameter_value(
                name,
                value,
                spec,
                f"{path}.{name}",
            )
        )

    return issues, resolved


def validate_parameter_value(
    name: str,
    value: Any,
    spec: dict[str, Any],
    path: str,
) -> list[ValidationIssue]:
    """Validate one parameter's type and simple numeric constraints."""
    issues: list[ValidationIssue] = []
    expected_type = spec.get("type")

    if expected_type and not matches_type(value, expected_type):
        return [
            ValidationIssue(
                code="parameter_type_mismatch",
                message=f"Parameter {name!r} must be {expected_type}",
                path=path,
            )
        ]

    if "enum" in spec and value not in spec["enum"]:
        issues.append(
            ValidationIssue(
                code="parameter_enum_mismatch",
                message=f"Parameter {name!r} must be one of {spec['enum']}",
                path=path,
            )
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and value < minimum:
            issues.append(
                ValidationIssue(
                    code="parameter_below_minimum",
                    message=f"Parameter {name!r} must be >= {minimum}",
                    path=path,
                )
            )
        if maximum is not None and value > maximum:
            issues.append(
                ValidationIssue(
                    code="parameter_above_maximum",
                    message=f"Parameter {name!r} must be <= {maximum}",
                    path=path,
                )
            )

    return issues


def matches_type(value: Any, expected_type: str) -> bool:
    """Map JSON schema-style type names to Python runtime checks."""
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def infer_json_type(value: Any) -> str:
    """Infer a compact JSON type name from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"
