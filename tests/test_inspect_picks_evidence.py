import unittest
from inspect_picks_evidence import build_inspect_picks_observation


class InspectPicksEvidenceTests(unittest.TestCase):
    def test_completed_inspect_picks_preserves_thresholds_scores_and_overlay_reference(self):
        result = {"job_uid": "J5", "outputs": {"particles": {"num_items": 90}}}
        visual = {"source": {"job_uid": "J4"}, "input_particle_count": 100, "image_count": 8, "score_statistics": {"ncc_score": {"p05": 0.05}, "power": {"p05": 455}}, "contact_sheet": {"local_path": "/tmp/J4.png", "mime_type": "image/png"}}
        observation = build_inspect_picks_observation(result, {"ncc_score_thresh": 0.05, "lpower_thresh_min": 455}, visual)
        self.assertEqual(observation["applied_thresholds"], {"ncc_score_thresh": 0.05, "lpower_thresh_min": 455})
        self.assertEqual(observation["selected_particle_count"], 90)
        self.assertEqual(observation["representative_overlay"]["local_path"], "/tmp/J4.png")
