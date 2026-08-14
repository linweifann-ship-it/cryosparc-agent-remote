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
            "reference_status": "completed",
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


def duplicate_class2d_candidates():
    """Return duplicate same-type candidates produced by repeated live tests."""
    actions = []
    for action_id, job_uid, status in [
        ("branch_J31", "J31", "running"),
        ("branch_J33", "J33", "completed"),
        ("branch_J9", "J9", "completed"),
    ]:
        action = candidate_actions()[0].copy()
        action.update(
            {
                "action_id": action_id,
                "action_type": "branch",
                "workflow_node_id": job_uid,
                "reference_job_uid": job_uid,
                "reference_status": status,
            }
        )
        actions.append(action)
    return actions


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
    """Return a strict v3 minimal decision."""
    return {
        "schema_version": "3.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "job_type": "class_2D_new",
                "parameters": {
                    "compute_num_gpus": 4,
                    "class2D_K": 50,
                },
            }
        ],
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

    def test_unknown_v2_action_becomes_generic_plan(self):
        decision = v2_forward_decision()
        decision["selected_actions"][0]["job_type"] = "unknown_future_job"

        result = adapt_v2_decision_to_internal(decision, candidate_actions())

        self.assertTrue(result["success"])
        action = result["internal_decision"]["selected_actions"][0]
        self.assertEqual(action["action_id"], "generic_0_unknown_future_job")
        self.assertEqual(action["job_type"], "unknown_future_job")
        self.assertNotIn("connections", action)

    def test_duplicate_same_type_candidates_prefer_original_completed_job(self):
        result = adapt_v2_decision_to_internal(
            v2_forward_decision(),
            duplicate_class2d_candidates(),
        )

        self.assertTrue(result["success"])
        action = result["internal_decision"]["selected_actions"][0]
        self.assertEqual(action["action_id"], "branch_J9")
        self.assertEqual(action["workflow_node_id"], "J9")
        self.assertEqual(result["internal_decision"]["decision_type"], "branch")
        self.assertEqual(
            result["internal_decision"]["branch_plan"]["max_parallel_branches"],
            1,
        )

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

    def test_v3_rejects_forbidden_reason_field(self):
        decision = v2_forward_decision()
        decision["reason"] = "not allowed"

        result = adapt_v2_decision_to_internal(decision, candidate_actions())

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "schema_validation_error")

    def test_v3_rejects_forbidden_connections_field(self):
        decision = v2_forward_decision()
        decision["selected_actions"][0]["connections"] = {
            "particles": {
                "source_job_uid": "J37",
                "source_output": "particles_selected",
            }
        }

        result = adapt_v2_decision_to_internal(decision, candidate_actions())

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "schema_validation_error")
        self.assertEqual(result["issues"][0]["path"], "selected_actions.0.connections")

    def test_v3_stop_requires_empty_selected_actions(self):
        result = adapt_v2_decision_to_internal(
            {
                "schema_version": "3.0",
                "decision_type": "stop",
                "selected_actions": [],
            },
            candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["internal_decision"]["decision_type"], "stop")


if __name__ == "__main__":
    unittest.main()
