# Runs a minimal MCP-model dry-run and live CryoSPARC job creation smoke test.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_registry import execute_model_decision_payload


PROJECT_UID = "P1"
WORKSPACE_UID = "W1"


def build_candidate_actions() -> list[dict]:
    """Return the MCP server candidate menu used for the smoke test."""
    return [
        {
            "action_id": "forward_agent_import_movies",
            "action_type": "forward",
            "workflow_node_id": "J_AGENT_IMPORT_TEMPLATE",
            "reference_job_uid": "J_AGENT_IMPORT_TEMPLATE",
            "job_type": "import_movies",
            "description": (
                "Agent smoke test: create an Import Movies job in building status."
            ),
            "execution_mode": "dry_run_only",
            "available": True,
            "blocked_by": [],
            "required_inputs": {},
            "parameter_template": {
                "blob_paths": {"type": "string", "required": True},
                "psize_A": {"type": "number", "minimum": 0},
                "accel_kv": {"type": "number", "minimum": 0},
            },
            "default_parameters": {},
        }
    ]


def build_model_decision() -> dict:
    """Return the simulated model decision JSON."""
    return {
        "schema_version": "1.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "action_id": "forward_agent_import_movies",
                "action_type": "forward",
                "workflow_node_id": "J_AGENT_IMPORT_TEMPLATE",
                "job_type": "import_movies",
                "parameters": {
                    "blob_paths": (
                        "/tmp/cryosparc_agent_smoke_test/no_real_movies_*.tif"
                    ),
                    "psize_A": 1.0,
                    "accel_kv": 300,
                },
            }
        ],
        "rollback_target": None,
        "branch_plan": None,
        "reason": (
            "Smoke test for MCP-model handoff and real CryoSPARC job creation."
        ),
        "confidence": 0.99,
        "risk_flags": ["smoke_test_fake_input_paths"],
        "evidence": ["Candidate action is available and has no upstream inputs."],
    }


def main() -> None:
    candidate_actions = build_candidate_actions()
    model_decision = build_model_decision()
    mcp_to_model_package = {
        "schema_version": "1.0",
        "task": "Choose the next CryoSPARC workflow action.",
        "project_uid": PROJECT_UID,
        "workspace_uid": WORKSPACE_UID,
        "current_node_id": "J_AGENT_IMPORT_TEMPLATE",
        "workflow_context": {
            "workflow_status": "smoke_test",
            "decision_hint": None,
        },
        "candidate_actions": candidate_actions,
        "output_contract": {
            "return_json_only": True,
            "schema_version": "1.0",
            "allowed_decision_type": ["forward", "rollback", "branch", "stop"],
            "selected_actions_rule": (
                "Every selected action_id must come from candidate_actions."
            ),
        },
    }

    dry_run_result = execute_model_decision_payload(
        model_decision,
        candidate_actions=candidate_actions,
        dry_run=True,
        project_uid=PROJECT_UID,
        workspace_uid=WORKSPACE_UID,
    )
    live_result = execute_model_decision_payload(
        model_decision,
        candidate_actions=candidate_actions,
        dry_run=False,
        project_uid=PROJECT_UID,
        workspace_uid=WORKSPACE_UID,
    )

    print(
        json.dumps(
            {
                "project_uid": PROJECT_UID,
                "workspace_uid": WORKSPACE_UID,
                "mcp_to_model_package": mcp_to_model_package,
                "model_decision": model_decision,
                "dry_run_result": dry_run_result,
                "live_result": live_result,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
