"""Offline tests for Inspect Picks Exposure Plot and Power Histogram evidence."""
import io
import unittest

import numpy as np
from PIL import Image

from vision_inputs import (
    build_pick_qc_dashboard,
    build_structured_pick_statistics,
    select_high_power_tail_particles,
)


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

    def test_structured_statistics_report_upper_bound_sensitivity_without_recommendation(self):
        ncc = np.linspace(0.0, 1.0, 1000)
        power = np.linspace(100.0, 1100.0, 1000)
        stats = build_structured_pick_statistics(
            particle_count=1000,
            ncc_scores=ncc,
            power_scores=power,
            histogram_metadata={"bin_count": [48, 48], "finite_pair_count": 1000},
        )

        self.assertEqual(stats["particle_count"], 1000)
        self.assertEqual(stats["valid_ncc_power_pair_count"], 1000)
        self.assertEqual(
            list(stats["ncc_quantiles"]),
            ["P0.5", "P1", "P5", "P25", "P50", "P75", "P90", "P95", "P97.5", "P99", "P99.5"],
        )
        candidates = stats["candidate_upper_thresholds"]
        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0]["derived_from_power_percentile"], "P90")
        self.assertEqual(candidates[-1]["derived_from_power_percentile"], "P99.5")
        self.assertEqual(candidates[-1]["threshold_recommendation"], None)
        self.assertEqual(
            candidates[-1]["retained_count"] + candidates[-1]["removed_high_power_count"],
            1000,
        )

    def test_high_power_tail_selection_covers_requested_bins_with_distinct_micrographs(self):
        power = np.arange(1000, dtype=np.float64)
        micrograph_uids = np.arange(1000, dtype=np.int64)
        selected, tail_bins = select_high_power_tail_particles(
            power,
            micrograph_uids,
            samples_per_bin=3,
        )

        self.assertEqual([item["label"] for item in tail_bins], [
            "P95-P97.5", "P97.5-P99", "P99-P99.5", ">P99.5",
        ])
        self.assertLessEqual(len(selected), 12)
        self.assertGreater(len(selected), 0)
        self.assertTrue(all(item["selected_micrograph_count"] <= 3 for item in tail_bins))
        self.assertTrue(all(item["selected_particle_count"] <= 3 for item in tail_bins))

    def test_high_power_tail_selection_preserves_high_bit_cryosparc_uids(self):
        power = np.arange(1000, dtype=np.float64)
        high_bit_uids = [2**63 + index for index in range(1000)]

        selected, _tail_bins = select_high_power_tail_particles(
            power,
            high_bit_uids,
            samples_per_bin=2,
        )

        self.assertGreater(len(selected), 0)
        self.assertTrue(all(
            high_bit_uids[item["particle_index"]] >= 2**63
            for item in selected
        ))


if __name__ == "__main__":
    unittest.main()
