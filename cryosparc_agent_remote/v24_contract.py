# Shared constants and strict helpers for Schema V24 benchmark runs.
import json
from typing import Any

from schemas import parse_external_model_decision


INPUT_SCHEMA_VERSION = "2.1"
OUTPUT_SCHEMA_VERSION = "3.0"
SCHEMA_LABEL = "V24"

FORBIDDEN_OUTPUT_FIELDS = {
    "decision",
    "next_action",
    "action",
    "reason",
    "explanation",
    "confidence",
    "evidence",
    "rollback_target",
    "branch_plan",
    "workflow_node_id",
    "connections",
}

ERROR_CATEGORIES = {
    "model_invalid_json",
    "model_schema_violation",
    "model_invalid_action",
    "model_invalid_parameter",
    "model_repeated_action",
    "model_premature_stop",
    "connection_resolution_failed",
    "mcp_validation_failed",
    "mcp_execution_failed",
    "cryosparc_job_failed",
    "approval_required",
    "api_failure",
    "cluster_failure",
    "context_isolation_failed",
    "unknown",
}


V24_OUTPUT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "decision_type", "selected_actions"],
    "properties": {
        "schema_version": {"const": OUTPUT_SCHEMA_VERSION},
        "decision_type": {"enum": ["forward", "branch", "stop"]},
        "selected_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["job_type", "parameters"],
                "properties": {
                    "job_type": {"type": "string"},
                    "parameters": {"type": "object"},
                },
            },
        },
    },
}


def strict_parse_model_output(raw_text: str) -> dict[str, Any]:
    """Parse only a complete JSON object; never extract or repair model text."""
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError("Model output must be one JSON object.")
    return parsed


def validate_v24_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Validate strict minimal_v3 output and return normalized issues."""
    parsed, issues = parse_external_model_decision(decision)
    if parsed is None:
        return {
            "success": False,
            "valid_schema": False,
            "decision_type": decision.get("decision_type"),
            "issues": [
                issue.model_dump() if hasattr(issue, "model_dump") else issue
                for issue in issues
            ],
        }
    return {
        "success": True,
        "valid_schema": True,
        "decision_type": parsed.decision_type,
        "issues": [],
    }


def classify_validation_error(validation: dict[str, Any]) -> str:
    """Map schema/parameter validation issues to benchmark error categories."""
    issue_text = json.dumps(validation.get("issues", []), ensure_ascii=False)
    if "job_type" in issue_text or "unsupported_job_type" in issue_text:
        return "model_invalid_action"
    if "parameters" in issue_text or "parameter" in issue_text:
        return "model_invalid_parameter"
    return "model_schema_violation"

