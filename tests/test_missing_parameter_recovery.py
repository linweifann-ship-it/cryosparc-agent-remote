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
        action = candidate("blob_picker_gpu", "diameter", {
            "type": "number", "minimum": 0, "title": "Minimum particle diameter (A)",
            "description": "Min Particle diameter (A)", "unit": "A",
            "unit_source": "registry_ui_contract",
        })
        feedback = recovery_feedback([action], {"pixel_size_A": 0.6575}, {})
        item = feedback["missing_required_parameters"][0]
        self.assertEqual(item["job_type"], "blob_picker_gpu")
        self.assertEqual(item["parameter"], "diameter")
        self.assertEqual(item["requirement_status"], "required_missing")
        self.assertTrue(feedback["policy"]["observed_facts_must_not_be_invented"])
        self.assertTrue(feedback["policy"]["heuristic_estimates_allowed"])
        self.assertEqual(item["unit"], "A")
        self.assertEqual(item["unit_source"], "registry_ui_contract")
        self.assertTrue(feedback["policy"]["preserve_optional_registry_defaults_without_evidence"])

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

    def test_physical_parameter_claimed_as_pixels_is_rejected(self):
        action = candidate("blob_picker_gpu", "diameter", {"type": "number", "unit": "A"})
        result = validate_model_decision_payload(
            decision(
                action,
                {"diameter": 300},
                reason="estimated particle diameter from pixel size",
                evidence=["diameter of 300 pixels based on 0.6575 A/pixel"],
            ),
            candidate_actions=[action],
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "parameter_unit_mismatch")

    def test_physical_parameter_inferred_from_pixel_scale_only_is_rejected(self):
        action = candidate("blob_picker_gpu", "diameter", {"type": "number", "unit": "A"})
        result = validate_model_decision_payload(
            decision(
                action,
                {"diameter": 300},
                reason="estimated particle diameter from pixel size",
                evidence=["diameter=300 A based on available 0.6575 A/pixel dataset scale"],
            ),
            candidate_actions=[action],
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "parameter_pixel_scale_inference")

    def test_unspecified_optional_tuning_stays_at_registry_default(self):
        action = candidate("blob_picker_gpu", "diameter", {"type": "number", "unit": "A"})
        action["parameter_template"]["min_distance"] = {
            "type": "number", "default": 1.0, "unit": "particle_diameters",
        }
        action["default_parameters"] = {"min_distance": 1.0}
        result = validate_model_decision_payload(
            decision(
                action,
                {"diameter": 150},
                reason="estimated 150 A particle diameter from domain knowledge",
                evidence=["assumed 150 A; downstream pick QC is required"],
            ),
            candidate_actions=[action],
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["resolved_actions"][0]["resolved_parameters"]["min_distance"], 1.0)

    def test_unexplained_optional_override_during_recovery_is_rejected(self):
        action = candidate("blob_picker_gpu", "diameter", {"type": "number", "unit": "A"})
        action["parameter_template"]["min_distance"] = {
            "type": "number", "default": 1.0, "unit": "particle_diameters",
            "title": "Min. separation dist (diameters)",
        }
        action["default_parameters"] = {"min_distance": 1.0}
        result = validate_model_decision_payload(
            decision(
                action,
                {"diameter": 150, "min_distance": 150},
                reason="estimated 150 A particle diameter from domain knowledge",
                evidence=["assumed 150 A; downstream pick QC is required"],
            ),
            candidate_actions=[action],
        )
        self.assertFalse(result["success"])
        self.assertEqual(
            result["issues"][0]["code"], "unjustified_optional_parameter_override"
        )

    def test_explicitly_justified_optional_override_remains_model_owned(self):
        action = candidate("blob_picker_gpu", "diameter", {"type": "number", "unit": "A"})
        action["parameter_template"]["min_distance"] = {
            "type": "number", "default": 1.0, "unit": "particle_diameters",
            "title": "Min. separation dist (diameters)",
        }
        action["default_parameters"] = {"min_distance": 1.0}
        result = validate_model_decision_payload(
            decision(
                action,
                {"diameter": 150, "min_distance": 0.8},
                reason="estimated 150 A diameter; min_distance=0.8 is a labelled assumption",
                evidence=[
                    "min_distance=0.8 is an assumed particle-diameter separation; "
                    "risk of closer picks requires downstream pick QC"
                ],
            ),
            candidate_actions=[action],
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["resolved_actions"][0]["resolved_parameters"]["min_distance"], 0.8)

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
