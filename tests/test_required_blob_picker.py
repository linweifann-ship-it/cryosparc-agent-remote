import unittest

from action_registry import validate_model_decision_payload


class RequiredBlobPickerTests(unittest.TestCase):
    def test_missing_diameter_fails_before_create(self):
        candidate = {
            "action_id": "forward_J2_blob_picker_gpu", "action_type": "forward",
            "workflow_node_id": "J2:blob_picker_gpu", "job_type": "blob_picker_gpu",
            "execution_mode": "create_job", "available": True, "blocked_by": [],
            "required_inputs": {}, "default_parameters": {},
            "parameter_template": {"diameter": {"type": "number", "minimum": 0, "required": True}},
        }
        payload = {"schema_version": "1.0", "decision_type": "forward", "reason": "test", "confidence": 0.9, "risk_flags": [], "evidence": [], "selected_actions": [{
            "action_id": candidate["action_id"], "action_type": "forward",
            "workflow_node_id": candidate["workflow_node_id"], "job_type": "blob_picker_gpu",
            "parameters": {}, "connections": {},
        }]}
        result = validate_model_decision_payload(payload, candidate_actions=[candidate])
        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "missing_required_parameter")
