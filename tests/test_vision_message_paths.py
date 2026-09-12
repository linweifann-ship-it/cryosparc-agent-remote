import unittest

from autonomous_mcp_closed_loop import build_autonomous_prompt


class VisionMessagePathTests(unittest.TestCase):
    def test_inspect_picks_visual_context_is_multimodal(self):
        messages = build_autonomous_prompt(
            {"current_state": {"last_node_id": "J10"}},
            {"candidate_actions": []},
            0,
            visual_context={
                "kind": "pick_inspection",
                "contact_sheet": {"data_url": "data:image/png;base64,PICKS"},
            },
            kb_tool_policy="disabled",
        )
        content = messages[-1]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,PICKS")
        self.assertIn("Inspect Picks", content[0]["text"])

    def test_class2d_select2d_visual_context_is_multimodal(self):
        messages = build_autonomous_prompt(
            {"current_state": {"last_node_id": "J20"}},
            {"candidate_actions": []},
            0,
            visual_context={
                "kind": "class_average",
                "contact_sheet": {"data_url": "data:image/png;base64,CLASSES"},
            },
            kb_tool_policy="disabled",
        )
        content = messages[-1]["content"]
        self.assertEqual(content[1]["type"], "image_url")
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,CLASSES")
        self.assertIn("Select 2D", content[0]["text"])

    def test_missing_visual_data_does_not_add_fake_image_evidence(self):
        messages = build_autonomous_prompt(
            {"current_state": {"last_node_id": "J20"}}, {"candidate_actions": []}, 0,
            visual_context={"kind": "class_average", "contact_sheet": {}}, kb_tool_policy="disabled",
        )
        self.assertFalse(any(isinstance(message.get("content"), list) for message in messages))


if __name__ == "__main__":
    unittest.main()
