# Fixed dry-run tests for model decision validation and planning.
import unittest

from action_registry import (
    execute_model_decision_payload,
    validate_model_decision_payload,
)


STATE_ID = "state_test"
CANDIDATE_SET_ID = "candidates_test"


def candidate_actions():
    return [
        {
            "action_id": "forward_J8",
            "action_type": "forward",
            "workflow_node_id": "node_extract_micrographs_multi_001",
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
        "state_snapshot_id": STATE_ID,
        "candidate_set_id": CANDIDATE_SET_ID,
        "decision_type": "forward",
        "selected_actions": [
            {
                "action_id": "forward_J8",
                "action_type": "forward",
                "workflow_node_id": "node_extract_micrographs_multi_001",
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
            expected_state_snapshot_id=STATE_ID,
            expected_candidate_set_id=CANDIDATE_SET_ID,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_mode"], "dry_run")
        self.assertEqual(result["execution_plan"]["plan_version"], "1.0")
        self.assertEqual(result["execution_plan"]["status"], "planned")
        self.assertEqual(result["execution_plan"]["action_count"], 1)
        self.assertFalse(result["execution_plan"]["approval_required"])
        self.assertEqual(result["planned_actions"][0]["action_id"], "forward_J8")
        self.assertEqual(result["planned_actions"][0]["status"], "planned")

    def test_parameter_above_maximum_is_invalid(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 16

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
            expected_state_snapshot_id=STATE_ID,
            expected_candidate_set_id=CANDIDATE_SET_ID,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "parameter_above_maximum")

        execute_result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
            expected_state_snapshot_id=STATE_ID,
            expected_candidate_set_id=CANDIDATE_SET_ID,
        )
        self.assertFalse(execute_result["success"])
        self.assertIsNone(execute_result["execution_plan"])

    def test_unknown_action_id_is_invalid(self):
        decision = forward_decision()
        decision["selected_actions"][0]["action_id"] = "forward_J999"

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
            expected_state_snapshot_id=STATE_ID,
            expected_candidate_set_id=CANDIDATE_SET_ID,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "unknown_action_id")

    def test_stale_state_snapshot_is_invalid(self):
        decision = forward_decision(state_snapshot_id="state_old")

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
            expected_state_snapshot_id=STATE_ID,
            expected_candidate_set_id=CANDIDATE_SET_ID,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "stale_workflow_state")

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
            expected_state_snapshot_id=STATE_ID,
            expected_candidate_set_id=CANDIDATE_SET_ID,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_plan"]["decision_type"], "stop")
        self.assertFalse(result["execution_plan"]["approval_required"])
        self.assertEqual(result["planned_actions"][0]["action_type"], "stop")

    def test_rollback_decision_is_valid_and_planned(self):
        rollback_target = {
            "workflow_node_id": "node_template_picker_gpu_001",
            "job_type": "template_picker_gpu",
            "reason_code": "poor_particle_picking",
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
            expected_state_snapshot_id=STATE_ID,
            expected_candidate_set_id=CANDIDATE_SET_ID,
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["approval_reasons"],
            ["rollback_decision"],
        )
        self.assertEqual(result["planned_actions"][0]["action_type"], "rollback")
        self.assertEqual(
            result["planned_actions"][0]["rollback_target"],
            rollback_target,
        )


if __name__ == "__main__":
    unittest.main()
