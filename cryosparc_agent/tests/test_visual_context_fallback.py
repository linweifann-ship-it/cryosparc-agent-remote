"""Regression tests for unavailable visual-overlay evidence."""
import unittest

from autonomous_mcp_closed_loop import build_autonomous_prompt
from model_direct_runner import build_workflow_decision_prompt


class VisualContextFallbackTests(unittest.TestCase):
    def test_autonomous_prompt_skips_missing_contact_sheet(self):
        messages = build_autonomous_prompt(
            model_input={"current_state": {"last_node_id": "J12"}},
            candidate_context={"candidate_actions": []},
            round_index=1,
            visual_context={
                "success": False,
                "kind": "pick_overlay",
                "error": "No readable micrographs were found.",
            },
            kb_tool_policy="disabled",
        )

        content = messages[-1]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("unavailable", content[0]["text"])
        self.assertNotIn("image_url", str(content))

    def test_direct_prompt_skips_missing_contact_sheet(self):
        messages = build_workflow_decision_prompt(
            {"current_state": {"last_node_id": "J12"}},
            visual_context={"success": False, "error": "overlay generation failed"},
        )

        content = messages[-1]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("unavailable", content[0]["text"])

    def test_autonomous_prompt_attaches_valid_contact_sheet(self):
        messages = build_autonomous_prompt(
            model_input={"current_state": {"last_node_id": "J12"}},
            candidate_context={"candidate_actions": []},
            round_index=1,
            visual_context={
                "kind": "pick_overlay",
                "contact_sheet": {"data_url": "data:image/png;base64,AA=="},
            },
            kb_tool_policy="disabled",
        )

        content = messages[-1]["content"]
        self.assertEqual(content[-1]["type"], "image_url")
        self.assertEqual(content[-1]["image_url"]["url"], "data:image/png;base64,AA==")

    def test_pick_prompt_attaches_statistics_then_three_visual_artifacts(self):
        dashboard_url = "data:image/png;base64,DASHBOARD"
        sheet_url = "data:image/png;base64,SHEET"
        high_power_url = "data:image/png;base64,HIGHPOWER"
        messages = build_autonomous_prompt(
            model_input={"current_state": {"last_node_id": "J46"}},
            candidate_context={"candidate_actions": []},
            round_index=2,
            visual_context={
                "kind": "pick_inspection",
                "structured_pick_statistics": {
                    "particle_count": 217150,
                    "candidate_upper_thresholds": [],
                },
                "exposure_plot": {"micrograph_count": 196},
                "power_histogram": {"finite_pair_count": 217150},
                "pick_qc_dashboard": {"data_url": dashboard_url},
                "contact_sheet": {"data_url": sheet_url},
                "high_power_targeted_inspection": {"data_url": high_power_url},
            },
            kb_tool_policy="disabled",
        )

        content = messages[-1]["content"]
        self.assertEqual(
            [item["type"] for item in content],
            ["text", "image_url", "image_url", "image_url"],
        )
        self.assertEqual(content[1]["image_url"]["url"], dashboard_url)
        self.assertEqual(content[2]["image_url"]["url"], sheet_url)
        self.assertEqual(content[3]["image_url"]["url"], high_power_url)
        self.assertIn("Structured pick statistics", content[0]["text"])
        self.assertIn("structured_pick_statistics", content[0]["text"])
        self.assertNotIn(dashboard_url, content[0]["text"])

    def test_pick_prompt_keeps_available_contact_sheet_when_dashboard_missing(self):
        sheet_url = "data:image/png;base64,SHEET"
        messages = build_autonomous_prompt(
            model_input={"current_state": {"last_node_id": "J46"}},
            candidate_context={"candidate_actions": []},
            round_index=2,
            visual_context={
                "kind": "pick_inspection",
                "pick_qc_dashboard": {},
                "contact_sheet": {"data_url": sheet_url},
            },
            kb_tool_policy="disabled",
        )

        content = messages[-1]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[-1]["image_url"]["url"], sheet_url)
        self.assertIn('"pick_qc_dashboard": "unavailable"', content[0]["text"])


if __name__ == "__main__":
    unittest.main()
