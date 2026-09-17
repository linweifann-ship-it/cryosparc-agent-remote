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


if __name__ == "__main__":
    unittest.main()
