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
        "schema_version": "3.0",
        "decision_type": "forward",
        "selected_actions": [
            {
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
    }


def main() -> None:
    candidate_actions = build_candidate_actions()
    model_decision = build_model_decision()
    mcp_to_model_package = {
        "schema_version": "2.1",
        "task_type": "workflow_decision",
        "task": "Choose the next CryoSPARC workflow action.",
        "project_uid": PROJECT_UID,
        "workspace_uid": WORKSPACE_UID,
        "dataset_context": {
            "dataset_metadata": {
                "known_workflow_steps": None,
            },
            "dataset_parameter_facts": {},
            "dataset_parameter_facts_by_job_type": {},
        },
        "current_state": {
            "last_node_id": None,
            "last_action": None,
            "last_node_status": "not_started",
            "last_node_info": {},
            "state_features": {},
            "recent_job_history": [],
        },
        "output_contract": {
            "return_json_only": True,
            "schema_version": "3.0",
            "allowed_decision_type": ["forward", "branch", "stop"],
            "selected_action_fields": ["job_type", "parameters"],
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
