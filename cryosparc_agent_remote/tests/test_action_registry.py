# Fixed dry-run tests for model decision validation and planning.
import unittest

from action_registry import (
    execute_model_decision_payload,
    validate_model_decision_payload,
)


def candidate_actions():
    return [
        {
            "action_id": "forward_J8",
            "action_type": "forward",
            "workflow_node_id": "J8",
            "reference_job_uid": "J8",
            "job_type": "extract_micrographs_multi",
            "description": "Dry-run candidate for fixed tests.",
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
                "box_size_pix": {
                    "type": "integer",
                    "minimum": 32,
                    "default": 400,
                },
            },
            "default_parameters": {
                "compute_num_gpus": 4,
                "box_size_pix": 400,
            },
        }
    ]


def forward_decision(**overrides):
    decision = {
        "schema_version": "1.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "action_id": "forward_J8",
                "action_type": "forward",
                "workflow_node_id": "J8",
                "job_type": "extract_micrographs_multi",
                "parameters": {
                    "compute_num_gpus": 4,
                    "box_size_pix": 400,
                },
            }
        ],
        "rollback_target": None,
        "branch_plan": None,
        "reason": "Particles are ready for extraction.",
        "confidence": 0.94,
        "risk_flags": [],
        "evidence": ["Selected particles are available."],
    }
    decision.update(overrides)
    return decision


class ActionRegistryFixedTests(unittest.TestCase):
    def test_valid_forward_returns_planned_action(self):
        result = execute_model_decision_payload(
            forward_decision(),
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_mode"], "dry_run")
        self.assertEqual(result["execution_plan"]["plan_version"], "1.0")
        self.assertEqual(result["execution_plan"]["status"], "planned")
        self.assertEqual(result["execution_plan"]["action_count"], 1)
        self.assertFalse(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_id"],
            "forward_J8",
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["status"],
            "planned",
        )

    def test_parameter_above_maximum_is_invalid(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 16

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "parameter_above_maximum")

        execute_result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )
        self.assertFalse(execute_result["success"])
        self.assertIsNone(execute_result["execution_plan"])

    def test_high_gpu_count_requires_approval(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 8

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        action = result["execution_plan"]["actions"][0]
        self.assertTrue(result["success"])
        self.assertTrue(result["execution_plan"]["approval_required"])
        self.assertTrue(action["approval_required"])
        self.assertIn("high_gpu_count", action["approval_reasons"])
        self.assertIn("high_gpu_count", result["execution_plan"]["approval_reasons"])

    def test_live_execution_requires_project_and_workspace(self):
        result = execute_model_decision_payload(
            forward_decision(),
            candidate_actions=candidate_actions(),
            dry_run=False,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["execution_mode"], "missing_execution_context")
        self.assertEqual(result["issues"][0]["code"], "missing_execution_context")

    def test_high_gpu_live_execution_is_blocked_by_approval(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 8

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
            dry_run=False,
            project_uid="P1",
            workspace_uid="W1",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["execution_mode"], "live_execution")
        self.assertEqual(
            result["execution_results"][0]["status"],
            "approval_required",
        )

    def test_unknown_action_id_is_invalid(self):
        decision = forward_decision()
        decision["selected_actions"][0]["action_id"] = "forward_J999"

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "unknown_action_id")

    def test_stop_decision_is_valid_and_planned(self):
        decision = forward_decision(
            decision_type="stop",
            selected_actions=[],
            reason="No safe next action is available.",
            confidence=0.8,
        )

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_plan"]["decision_type"], "stop")
        self.assertFalse(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_type"],
            "stop",
        )

    def test_rollback_decision_is_valid_and_planned(self):
        rollback_target = {
            "workflow_node_id": "J6",
            "job_type": "template_picker_gpu",
            "reason_code": "poor_particle_picking",
            "rollback_mode": "rerun_from_target",
        }
        decision = forward_decision(
            decision_type="rollback",
            selected_actions=[],
            rollback_target=rollback_target,
            reason="Picking should be adjusted before extraction.",
            confidence=0.83,
            risk_flags=["particle_quality_uncertain"],
        )

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["approval_reasons"],
            ["rollback_decision"],
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_type"],
            "rollback",
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["rollback_target"],
            rollback_target,
        )


if __name__ == "__main__":
    unittest.main()
