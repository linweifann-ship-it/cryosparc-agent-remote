import os
import tempfile
import unittest
from unittest import mock

from autonomous_mcp_closed_loop import build_autonomous_prompt
from vision_inputs import (
    build_class_average_visual_context,
    build_pick_inspection_visual_context,
)


class _FakeProject:
    def __init__(self, job):
        self.job = job

    def find_job(self, job_uid):
        return self.job

    def download_mrc(self, source_path):
        return None, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]


class _FakeClient:
    def __init__(self, job):
        self.project = _FakeProject(job)

    def find_project(self, project_uid):
        return self.project


class _FakeClassJob:
    def load_output(self, name):
        assert name == "class_averages"
        return {
            "blob/path": ["classes.mrc", "classes.mrc"],
            "blob/idx": [0, 1],
            "blob/res_A": [3.2, 3.8],
        }

    def download_mrc(self, filename):
        return None, [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
        ]


class _FakePickJob:
    def load_output(self, name):
        if name == "micrographs":
            return {
                "micrograph_blob/path": ["micrograph_1.mrc"],
                "uid": [10],
                "micrograph_blob/is_background_subtracted": [False],
            }
        assert name == "particles"
        return {
            "location/micrograph_uid": [10, 10],
            "location/center_x_frac": [0.25, 0.75],
            "location/center_y_frac": [0.25, 0.75],
            "pick_stats/ncc_score": [0.2, 0.8],
            "pick_stats/power": [1.0, 2.0],
        }


class VisionMessagePathTests(unittest.TestCase):
    def test_class_average_context_encodes_class_ids_as_data_url(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakeClassJob())):
                with mock.patch.dict(os.environ, {"CRYOAGENT_VISION_CACHE_DIR": cache_dir}):
                    context = build_class_average_visual_context("P1", "J1", max_classes=2, tile_size=8)
        self.assertEqual(context["class_ids"], [0, 1])
        self.assertEqual(context["contact_sheet"]["label_format"], "class_id=<integer>")
        self.assertTrue(context["contact_sheet"]["data_url"].startswith("data:image/png;base64,"))

    def test_pick_context_builds_overlay_data_url(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with mock.patch("vision_inputs.cryosparc_client", return_value=_FakeClient(_FakePickJob())):
                with mock.patch.dict(os.environ, {"CRYOAGENT_VISION_CACHE_DIR": cache_dir}):
                    context = build_pick_inspection_visual_context("P1", "J2", max_micrographs=1, tile_size=16)
        self.assertEqual(context["overlay"]["marker"], "red circles")
        self.assertEqual(context["panels"][0]["overlay_count"], 2)
        self.assertTrue(context["contact_sheet"]["data_url"].startswith("data:image/png;base64,"))

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

    def test_cryosift_observation_is_preserved_in_model_text_context(self):
        observation = {"success": False, "status": "not_configured", "tool": "cryosift"}
        messages = build_autonomous_prompt(
            {"tool_evidence": {"cryosift": observation}}, {"candidate_actions": []}, 0,
            kb_tool_policy="disabled",
        )
        self.assertIn("cryosift", messages[-1]["content"])
        self.assertIn("not_configured", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
