"""Structured, bounded recovery guidance for required scientific parameters."""

from typing import Any


MAX_HEURISTIC_ATTEMPTS = 2


def missing_required_parameters(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return Registry-required parameters that lack a resolved default."""
    missing: list[dict[str, Any]] = []
    for action in actions:
        defaults = action.get("default_parameters") or {}
        for name, spec in (action.get("parameter_template") or {}).items():
            if not spec.get("required") or defaults.get(name) is not None:
                continue
            missing.append(
                {
                    "action_id": action.get("action_id"),
                    "job_type": action.get("job_type"),
                    "parameter": name,
                    "requirement_status": "required_missing",
                    "description": spec.get("description"),
                    "type": spec.get("type"),
                    "minimum": spec.get("minimum"),
                    "maximum": spec.get("maximum"),
                    "enum": spec.get("enum"),
                    "default": spec.get("default"),
                }
            )
    return missing


def recovery_feedback(
    actions: list[dict[str, Any]],
    dataset_info: dict[str, Any],
    attempts: dict[str, int],
) -> dict[str, Any] | None:
    """Build model-visible feedback without synthesizing scientific facts."""
    missing = missing_required_parameters(actions)
    if not missing:
        return None
    for item in missing:
        key = f"{item['action_id']}:{item['parameter']}"
        attempted = attempts.get(key, 0)
        item["heuristic_attempts"] = attempted
        item["heuristic_attempts_remaining"] = max(0, MAX_HEURISTIC_ATTEMPTS - attempted)
    return {
        "missing_required_parameters": missing,
        "dataset_evidence": dataset_info,
        "policy": {
            "observed_facts_must_not_be_invented": True,
            "heuristic_estimates_allowed": True,
            "heuristic_evidence_label_required": True,
            "request_input_only_when_no_safe_estimate": True,
            "max_attempts_per_action_parameter": MAX_HEURISTIC_ATTEMPTS,
        },
    }


def attempt_keys_for_decision(
    decision: dict[str, Any], feedback: dict[str, Any] | None
) -> list[str]:
    """Select missing-parameter retry keys for the Model's chosen action."""
    action = decision.get("action") or decision.get("job_type")
    if not feedback:
        return []
    return [
        f"{item['action_id']}:{item['parameter']}"
        for item in feedback.get("missing_required_parameters", [])
        if not action or item.get("job_type") == action
    ]


def consume_request_input_retry(
    decision: dict[str, Any],
    feedback: dict[str, Any] | None,
    attempts: dict[str, int],
) -> bool:
    """Consume one bounded retry for a missing-parameter request_input decision."""
    keys = attempt_keys_for_decision(decision, feedback)
    if not any(attempts.get(key, 0) < MAX_HEURISTIC_ATTEMPTS for key in keys):
        return False
    for key in keys:
        if attempts.get(key, 0) < MAX_HEURISTIC_ATTEMPTS:
            attempts[key] = attempts.get(key, 0) + 1
    return True
