"""Structured, bounded recovery guidance for required scientific parameters."""

from typing import Any


MAX_HEURISTIC_ATTEMPTS = 2


def missing_required_parameters(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return actionable scientific Registry requirements without defaults."""
    missing: list[dict[str, Any]] = []
    for action in actions:
        if not action.get("available", True) or action.get("blocked_by"):
            continue
        defaults = action.get("default_parameters") or {}
        for name, spec in (action.get("parameter_template") or {}).items():
            if (
                not spec.get("required")
                or defaults.get(name) is not None
                or not is_heuristic_scientific_parameter(spec)
            ):
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


def is_heuristic_scientific_parameter(spec: dict[str, Any]) -> bool:
    """Avoid asking Model to guess paths, credentials, or executable locations."""
    if spec.get("heuristic_allowed") is True:
        return True
    return spec.get("type") in {"number", "integer"}


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


def inject_model_parameter_recovery_guidance(
    model_input: dict[str, Any],
    candidate_context: dict[str, Any],
    dataset_info: dict[str, Any],
    attempts: dict[str, int],
) -> dict[str, Any] | None:
    """Attach first-decision guidance to the exact Model context contract."""
    feedback = recovery_feedback(
        candidate_context.get("candidate_actions") or [], dataset_info, attempts
    )
    if not feedback:
        return None
    guidance = {
        "guidance_type": "missing_required_scientific_parameter_recovery",
        "parameters": feedback["missing_required_parameters"],
        "dataset_evidence": feedback["dataset_evidence"],
        "policy": feedback["policy"],
        "model_instruction": (
            "For these required scientific parameters, you may provide a conservative "
            "heuristic estimate from domain knowledge and current evidence. Explicitly "
            "label it estimated or assumed in reason/evidence; do not invent observed facts. "
            "Use request_input only if no reasonable safe estimate is possible."
        ),
    }
    model_input["parameter_recovery_guidance"] = guidance
    failure_context = model_input.get("failure_context")
    if failure_context is None:
        failure_context = {"has_failure": False}
        model_input["failure_context"] = failure_context
    failure_context["parameter_recovery_guidance"] = guidance
    candidate_context["parameter_recovery_guidance"] = guidance
    return feedback


def candidate_actions_for_model_recovery(
    model_input: dict[str, Any], candidate_context: dict[str, Any]
) -> list[dict[str, Any]]:
    """Prefer candidates already selected for this Model decision context."""
    return model_input.get("candidate_actions") or candidate_context.get("candidate_actions") or []


def current_node_from_model_context(model_input: dict[str, Any]) -> str | None:
    """Recover the authoritative terminal node when a harness is restarted."""
    context = model_input.get("candidate_context") or {}
    return context.get("current_node_id") or model_input.get("current_job_uid")


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
