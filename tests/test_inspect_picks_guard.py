import unittest

from cryosparc_mcp_server import apply_inspect_parameter_guard


class InspectPicksParameterGuardTests(unittest.TestCase):
    def setUp(self):
        self.validation = {"success": True, "valid_actions": True, "issues": []}

    def guard(self, parameters):
        return apply_inspect_parameter_guard(
            {"job_type": "inspect_picks_v2", "parameters": parameters},
            self.validation,
        )

    def test_hidden_keep_threshold_does_not_claim_automatic_execution(self):
        result = self.guard({"keep_threshold": 0.3, "do_auto_cluster": False})
        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][-1]["code"], "inspect_parameters_required")

    def test_direct_score_threshold_is_automatically_executable(self):
        self.assertTrue(self.guard({"ncc_score_thresh": 0.2})["success"])

    def test_auto_cluster_is_automatically_executable(self):
        self.assertTrue(self.guard({"do_auto_cluster": True})["success"])


if __name__ == "__main__":
    unittest.main()
