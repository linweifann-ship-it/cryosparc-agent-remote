import unittest
from unittest import mock

import cryosparc_mcp_server


class McpVisionRegistrationTests(unittest.TestCase):
    def test_vision_tools_are_registered_on_the_runtime_server(self):
        tools = cryosparc_mcp_server.mcp._tool_manager._tools
        self.assertIn("get_pick_inspection_visual_context", tools)
        self.assertIn("get_class_average_visual_context", tools)

    def test_registered_vision_tools_dispatch_to_their_implementations(self):
        tools = cryosparc_mcp_server.mcp._tool_manager._tools
        pick_payload = {"contact_sheet": {"data_url": "data:image/png;base64,PICKS"}}
        class_payload = {"contact_sheet": {"data_url": "data:image/png;base64,CLASSES"}}
        with mock.patch.object(
            cryosparc_mcp_server,
            "build_pick_inspection_visual_context",
            return_value=pick_payload,
        ) as pick_builder, mock.patch.object(
            cryosparc_mcp_server,
            "build_class_average_visual_context",
            return_value=class_payload,
        ) as class_builder:
            self.assertEqual(
                tools["get_pick_inspection_visual_context"].fn("P2", "J314"),
                pick_payload,
            )
            self.assertEqual(
                tools["get_class_average_visual_context"].fn("P2", "J315"),
                class_payload,
            )
        pick_builder.assert_called_once()
        class_builder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
