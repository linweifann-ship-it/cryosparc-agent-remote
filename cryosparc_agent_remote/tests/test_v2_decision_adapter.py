# Fixed tests for adapting V2 model decisions into internal execution plans.
import unittest
from unittest.mock import patch

from v2_decision_adapter import (
    adapt_v2_decision_to_internal,
    execute_v2_model_decision_payload,
)


def candidate_actions():
    """Return one internal candidate produced after J8 completes."""
    return [
        {
            "action_id": "forward_J9",
            "action_type": "forward",
            "workflow_node_id": "J9",
            "reference_job_uid": "J9",
            "job_type": "class_2D_new",
            "description": "Run 2D classification.",
            "execution_mode": "dry_run_only",
            "available": True,
            "blocked_by": [],
            "required_inputs": {},
            "parameter_template": {
                "compute_num_gpus": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 4,
                },
                "class2D_K": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 50,
                },
            },
            "default_parameters": {
                "compute_num_gpus": 4,
                "class2D_K": 50,
            },
        }
    ]


def candidate_context():
    """Return the internal candidate registry context."""
    return {
        "project_uid": "P2",
        "workspace_uid": "W3",
        "current_node_id": "J8",
        "candidate_actions": candidate_actions(),
        "blocked_actions": [],
        "decision_hint": None,
    }


def v2_forward_decision():
    """Return a compact V2 decision that does not expose action_id."""
    return {
        "schema_version": "2.0",
        "decision_type": "forward",
        "action": "class_2D_new",
        "parameters": {
            "compute_num_gpus": 4,
            "class2D_K": 50,
        },
        "reason": "Extracted particles are ready for 2D classification.",
        "confidence": 0.9,
        "risk_flags": [],
        "evidence": ["J8 completed with particles output."],
    }


class V2DecisionAdapterTests(unittest.TestCase):
    def test_v2_forward_maps_to_internal_candidate(self):
        result = adapt_v2_decision_to_internal(
            v2_forward_decision(),
            candidate_actions(),
        )

        self.assertTrue(result["success"])
        action = result["internal_decision"]["selected_actions"][0]
        self.assertEqual(action["action_id"], "forward_J9")
        self.assertEqual(action["job_type"], "class_2D_new")

    def test_unknown_v2_action_fails_before_execution(self):
        decision = v2_forward_decision()
        decision["action"] = "unknown_job_type"
        decision["job_type"] = "unknown_job_type"

        result = adapt_v2_decision_to_internal(decision, candidate_actions())

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "candidate_not_found")

    def test_v2_execute_dry_run_reuses_internal_executor(self):
        with patch(
            "v2_decision_adapter.get_candidate_actions",
            return_value=candidate_context(),
        ):
            result = execute_v2_model_decision_payload(
                v2_forward_decision(),
                project_uid="P2",
                workspace_uid="W3",
                current_node_id="J8",
                dry_run=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_mode"], "v2_adapter")
        self.assertEqual(
            result["internal_decision"]["selected_actions"][0]["action_id"],
            "forward_J9",
        )
        self.assertEqual(
            result["execution_result"]["execution_plan"]["actions"][0]["job_type"],
            "class_2D_new",
        )


if __name__ == "__main__":
    unittest.main()
