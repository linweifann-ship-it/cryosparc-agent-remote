import unittest

from quality_policy import assess_node, build_retry_candidate


class QualityPolicyTests(unittest.TestCase):
    def test_completed_without_numeric_resolution_is_conservative(self):
        result = assess_node({
            "status": "completed", "has_error": False, "has_warning": False,
            "outputs": {"volume": {"available": True, "num_items": 1,
                                      "summary_keys": ["fsc_curve"],
                                      "summary_values": {"fsc_curve": 0.143},
                                      "latest_summary_stat_keys": [],
                                      "latest_summary_stat_values": {}}},
            "run_errors": {},
        })
        self.assertEqual(result["quality_status"], "completed_observed")
        self.assertTrue(result["resolution_evidence"]["available"])
        self.assertEqual(result["resolution_evidence"]["values"]["fsc_curve"], 0.143)
        self.assertEqual(result["evidence_level"], "observed_state_only")

    def test_failed_job_gets_approval_gated_retry(self):
        node = {"status": "failed", "has_error": True, "job_type": "patch_ctf_estimation_multi",
                "workflow_node_id": "J8", "cryosparc_job_uid": "J8", "inputs": {},
                "key_parameters": {"compute_num_gpus": 1}}
        result = assess_node(dict(node, outputs={}, run_errors={"errors_run": ["failure"]}))
        self.assertEqual(result["quality_status"], "failed")
        retry = build_retry_candidate(node)
        self.assertTrue(retry["approval_required"])
        self.assertEqual(retry["recovery_policy"], "retry_failed_job")


if __name__ == "__main__":
    unittest.main()
