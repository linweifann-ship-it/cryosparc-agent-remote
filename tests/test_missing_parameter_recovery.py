import unittest

from action_registry import validate_model_decision_payload
from missing_parameter_recovery import (
    attempt_keys_for_decision,
    consume_request_input_retry,
    recovery_feedback,
)


def candidate(job_type: str, parameter: str, spec: dict) -> dict:
    return {
        "action_id": f"registry_{job_type}",
        "action_type": "forward",
        "workflow_node_id": f"J10:{job_type}",
        "job_type": job_type,
        "execution_mode": "create_job",
        "available": True,
        "blocked_by": [],
        "required_inputs": {},
        "default_parameters": {},
        "parameter_template": {parameter: {"required": True, **spec}},
    }


def decision(action: dict, parameters: dict, *, reason: str, evidence: list[str]) -> dict:
    return {
        "schema_version": "1.0",
        "decision_type": "forward",
        "reason": reason,
        "confidence": 0.9,
        "risk_flags": [],
        "evidence": evidence,
        "selected_actions": [{
            "action_id": action["action_id"],
            "action_type": "forward",
            "workflow_node_id": action["workflow_node_id"],
            "job_type": action["job_type"],
            "parameters": parameters,
            "connections": {},
        }],
    }


class MissingParameterRecoveryTests(unittest.TestCase):
    def test_blob_picker_missing_diameter_is_structured_for_model(self):
        action = candidate("blob_picker_gpu", "diameter", {"type": "number", "minimum": 0})
        feedback = recovery_feedback([action], {"pixel_size_A": 0.6575}, {})
        item = feedback["missing_required_parameters"][0]
        self.assertEqual(item["job_type"], "blob_picker_gpu")
        self.assertEqual(item["parameter"], "diameter")
        self.assertEqual(item["requirement_status"], "required_missing")
        self.assertTrue(feedback["policy"]["observed_facts_must_not_be_invented"])
        self.assertTrue(feedback["policy"]["heuristic_estimates_allowed"])

    def test_labeled_heuristic_revalidates_before_any_create(self):
        action = candidate("blob_picker_gpu", "diameter", {"type": "number", "minimum": 0})
        result = validate_model_decision_payload(
            decision(
                action,
                {"diameter": 150},
                reason="estimated particle diameter from domain knowledge; not an observed fact",
                evidence=["assumed 150 A heuristic pending downstream pick QC"],
            ),
            candidate_actions=[action],
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["resolved_actions"][0]["resolved_parameters"]["diameter"], 150)

    def test_other_required_scientific_parameter_uses_same_path(self):
        action = candidate("extract_micrographs_multi", "box_size_pix", {"type": "integer", "minimum": 32})
        feedback = recovery_feedback([action], {}, {})
        self.assertEqual(feedback["missing_required_parameters"][0]["parameter"], "box_size_pix")

    def test_invalid_or_missing_parameter_remains_precreate_validation_failure(self):
        action = candidate("blob_picker_gpu", "diameter", {"type": "number", "minimum": 0})
        missing = validate_model_decision_payload(
            decision(action, {}, reason="test", evidence=[]), candidate_actions=[action]
        )
        invalid = validate_model_decision_payload(
            decision(action, {"diameter": -1}, reason="test", evidence=[]), candidate_actions=[action]
        )
        self.assertFalse(missing["success"])
        self.assertEqual(missing["issues"][0]["code"], "missing_required_parameter")
        self.assertFalse(invalid["success"])

    def test_attempts_are_bounded_and_request_input_is_permitted_afterwards(self):
        action = candidate("x", "p", {"type": "number"})
        feedback = recovery_feedback([action], {}, {})
        attempts = {}
        request_input = {"decision_type": "request_input", "action": "x"}
        self.assertEqual(attempt_keys_for_decision(request_input, feedback), ["registry_x:p"])
        self.assertTrue(consume_request_input_retry(request_input, feedback, attempts))
        self.assertEqual(attempts, {"registry_x:p": 1})
        self.assertTrue(consume_request_input_retry(request_input, feedback, attempts))
        self.assertEqual(attempts, {"registry_x:p": 2})
        self.assertFalse(consume_request_input_retry(request_input, feedback, attempts))
        self.assertEqual(attempts, {"registry_x:p": 2})


if __name__ == "__main__":
    unittest.main()
