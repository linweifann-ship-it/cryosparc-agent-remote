import unittest
import importlib
import os
from unittest.mock import patch

from cryosparc_agent_remote.cryosift_adapter import evaluate_2d_classes_with_cryosift
from cryosparc_agent_remote.openai_agents_runner import CLOSED_LOOP_MCP_TOOLS, STATIC_MCP_PROTOCOL
from cryosparc_agent_remote.vision_inputs import vision_cache_dir


class EvidenceToolIntegrationTests(unittest.TestCase):
    def test_agents_runner_exposes_evidence_tools_without_removing_execution_tools(self):
        expected_tools = {
            "get_workflow_decision_context",
            "get_candidate_action_details",
            "kb_get_decision_context",
            "kb_search_cryoem_kb",
            "get_class_average_visual_context",
            "get_micrograph_visual_context",
            "get_pick_inspection_visual_context",
            "evaluate_2d_classes_with_cryosift",
            "validate_v2_model_decision",
            "execute_v2_model_decision",
            "wait_for_job_result_package",
        }

        self.assertTrue(expected_tools.issubset(set(CLOSED_LOOP_MCP_TOOLS)))
        self.assertIn("get_micrograph_visual_context", STATIC_MCP_PROTOCOL["required_mcp_sequence"][1])

    def test_cryosift_not_configured_returns_structured_status(self):
        with patch.dict(
            "os.environ",
            {
                "CRYOAGENT_CRYOSIFT_RUNNER": "/definitely/missing/cryosift_runner.py",
                "CRYOAGENT_CRYOSIFT_WEIGHTS": "/definitely/missing/final_model.pth",
            },
        ):
            result = evaluate_2d_classes_with_cryosift("P2", "J99")

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "not_configured")
        self.assertEqual(result["tool"], "cryosift")
        self.assertEqual(result["project_uid"], "P2")
        self.assertEqual(result["job_uid"], "J99")
        self.assertIn("missing_paths", result)

    def test_visual_cache_uses_writable_temp_default_and_allows_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(str(vision_cache_dir()), "/tmp/cryoagent/vision_inputs")
        with patch.dict(os.environ, {"CRYOAGENT_VISION_CACHE_DIR": "/var/tmp/custom-vision"}, clear=True):
            self.assertEqual(str(vision_cache_dir()), "/var/tmp/custom-vision")

    def test_mcp_server_defines_evidence_tool_entrypoints(self):
        try:
            mcp_server = importlib.import_module("cryosparc_agent_remote.cryosparc_mcp_server")
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("mcp"):
                self.skipTest("mcp package is not installed in this test environment")
            raise

        for name in (
            "kb_get_decision_context",
            "kb_search_cryoem_kb",
            "get_class_average_visual_context",
            "get_micrograph_visual_context",
            "get_pick_inspection_visual_context",
            "evaluate_2d_classes_with_cryosift",
        ):
            self.assertTrue(callable(getattr(mcp_server, name)))


if __name__ == "__main__":
    unittest.main()
