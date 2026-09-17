"""Offline tests for Inspect Picks Exposure Plot and Power Histogram evidence."""
import io
import unittest

import numpy as np
from PIL import Image

from vision_inputs import build_pick_qc_dashboard


class PickQcDashboardTests(unittest.TestCase):
    def test_dashboard_contains_observed_distribution_metadata(self):
        dashboard, metadata = build_pick_qc_dashboard(
            pick_counts=np.asarray([8, 12, 19, 11, 23]),
            ncc_scores=np.asarray([0.03, 0.12, 0.28, np.nan, 0.42]),
            power_scores=np.asarray([210.0, 430.0, 860.0, 500.0, np.inf]),
            width=480,
            height=640,
            bins=12,
        )

        self.assertEqual(dashboard.size, (480, 640))
        exposure = metadata["exposure_plot"]
        histogram = metadata["power_histogram"]
        self.assertEqual(exposure["micrograph_count"], 5)
        self.assertEqual(exposure["pick_count_statistics"]["p50"], 12.0)
        self.assertEqual(histogram["finite_pair_count"], 3)
        self.assertEqual(histogram["nonfinite_pair_count"], 2)
        self.assertEqual(histogram["threshold_recommendation"], None)
        self.assertEqual(histogram["bin_count"], [12, 12])

        encoded = io.BytesIO()
        dashboard.save(encoded, format="PNG")
        self.assertEqual(Image.open(io.BytesIO(encoded.getvalue())).format, "PNG")

    def test_dashboard_handles_no_finite_score_pairs(self):
        _dashboard, metadata = build_pick_qc_dashboard(
            pick_counts=np.asarray([0, 0]),
            ncc_scores=np.asarray([np.nan, np.inf]),
            power_scores=np.asarray([np.inf, np.nan]),
        )

        histogram = metadata["power_histogram"]
        self.assertEqual(histogram["finite_pair_count"], 0)
        self.assertEqual(histogram["nonfinite_pair_count"], 2)
        self.assertEqual(histogram["threshold_recommendation"], None)


if __name__ == "__main__":
    unittest.main()
