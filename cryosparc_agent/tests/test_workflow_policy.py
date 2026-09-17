import unittest

from workflow_policy import (
    annotate_candidates,
    build_decision_guidance,
    infer_current_stage,
    job_stage,
)


class WorkflowPolicyTests(unittest.TestCase):
    def test_import_micrographs_leads_to_ctf_or_motion(self):
        state = infer_current_stage([
            {"cryosparc_job_uid": "J6", "workflow_node_id": "J6",
             "job_type": "import_micrographs", "status": "completed"}
        ])
        self.assertEqual(state["current_stage"], "import")
        self.assertEqual(state["next_stages"], ["motion_correction", "ctf_estimation"])

    def test_requested_node_overrides_newer_workspace_job(self):
        state = infer_current_stage([
            {"cryosparc_job_uid": "J6", "workflow_node_id": "J6",
             "job_type": "import_micrographs", "status": "completed"},
            {"cryosparc_job_uid": "J8", "workflow_node_id": "J8",
             "job_type": "patch_ctf_estimation_multi", "status": "launched"},
        ], current_node_id="J6")
        self.assertEqual(state["current_stage"], "import")

    def test_refinement_leads_to_validation(self):
        self.assertEqual(job_stage("homo_refine_new"), "refinement_3d")
        state = infer_current_stage([
            {"cryosparc_job_uid": "J20", "workflow_node_id": "J20",
             "job_type": "homo_refine_new", "status": "completed"}
        ])
        self.assertIn("validation", state["next_stages"])

    def test_target_resolution_requires_refinement(self):
        guidance = build_decision_guidance(
            {"target_resolution_A": 2.8},
            {"resolution_evidence": {"values": {"fsc_resolution": 3.58}}},
            {"job_type": "validation"},
            [{"job_type": "nonuniform_refine_new", "available": True}],
        )
        self.assertEqual(guidance["target_status"], "unmet")
        self.assertTrue(guidance["must_continue_for_target"])

    def test_inspect_picks_is_preferred_after_blob_picker(self):
        guidance = build_decision_guidance(
            {},
            {},
            {"job_type": "blob_picker_gpu"},
            [{"job_type": "inspect_picks_v2", "available": True}],
        )
        self.assertEqual(
            guidance["inspect_picks"]["priority"],
            "mandatory",
        )
        self.assertIn("MUST choose inspect_picks_v2", guidance["inspect_picks"]["model_instruction"])

    def test_policy_is_advisory_and_does_not_filter(self):
        candidates = [{"job_type": "patch_ctf_estimation_multi"},
                      {"job_type": "blob_picker_gpu"}]
        result = annotate_candidates(candidates, "import")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["workflow_stage"], "ctf_estimation")
        self.assertEqual(result[0]["workflow_policy_recommendation"], "preferred_next_stage")


if __name__ == "__main__":
    unittest.main()
