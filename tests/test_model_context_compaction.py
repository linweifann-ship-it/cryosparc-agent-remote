import unittest

from model_context_compaction import (
    compact_candidate_action,
    compact_candidate_context_payload,
    compact_execution_response,
)


class ModelContextCompactionTests(unittest.TestCase):
    def test_candidate_keeps_identity_connections_and_parameter_index(self):
        candidate = {
            "action_id": "registry_J1_large_job", "action_type": "forward",
            "workflow_node_id": "J1:large_job", "job_type": "large_job",
            "available": True,
            "required_inputs": {"micrographs": [{"source_job_uid": "J1", "source_output": "exposures"}]},
            "parameter_template": {"diameter": {"required": True}, "mode": {"enum": ["a", "b"]}},
            "default_parameters": {str(index): index for index in range(20)},
        }
        compact = compact_candidate_action(candidate)
        self.assertEqual(compact["job_type"], "large_job")
        self.assertEqual(compact["required_inputs"]["micrographs"][0]["source_job_uid"], "J1")
        self.assertEqual(compact["parameter_interface"]["parameter_names"], ["diameter", "mode"])
        self.assertEqual(compact["parameter_interface"]["required_parameters"], ["diameter"])
        self.assertEqual(compact["parameter_interface"]["default_parameters_omitted"], 20)

    def test_small_import_defaults_are_preserved(self):
        payload = {"candidate_actions": [{
            "action_id": "initial_import_micrographs", "job_type": "import_micrographs",
            "parameter_template": {"cs_mm": {}, "psize_A": {}},
            "default_parameters": {"cs_mm": 2.7, "psize_A": 0.6575},
        }]}
        compact = compact_candidate_context_payload(payload)
        self.assertEqual(compact["candidate_actions"][0]["parameter_interface"]["default_parameters"], {"cs_mm": 2.7, "psize_A": 0.6575})

    def test_execution_keeps_logical_race_job_for_runner_waiting(self):
        response = {
            "success": True, "dry_run": False, "execution_mode": "v2_adapter",
            "candidate_context": {"candidate_count": 1}, "internal_decision": {"decision_type": "forward"},
            "execution_result": {"success": True, "execution_results": [{
                "success": True, "status": "queued", "project_uid": "P2", "workspace_uid": "W59",
                "job_uid": "J1", "job_type": "patch_ctf_estimation_multi", "queued": True,
                "logical_workflow_step": True,
                "logical_job": {"logical_job_id": "logical-1", "logical_job_uid": "J1", "physical_job_ids": ["J1", "J2"], "winner_job_uid": "J1", "large": "omit"},
                "planned_action": {"large": "omit"},
            }]},
        }
        result = compact_execution_response(response)["execution_result"]["execution_results"][0]
        self.assertEqual(result["job_uid"], "J1")
        self.assertEqual(result["logical_job"]["physical_job_ids"], ["J1", "J2"])
        self.assertNotIn("planned_action", result)
        self.assertNotIn("large", result["logical_job"])


if __name__ == "__main__":
    unittest.main()
