"""Offline tests for Inspect Picks Exposure Plot and Power Histogram evidence."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from vision_inputs import (
    _extract_particle_crop,
    _normalize_particle_pair,
    build_pick_qc_dashboard,
    build_structured_pick_statistics,
    load_micrograph_image,
    select_high_power_tail_particles,
    select_typical_power_matched_controls,
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

    def test_edge_outside_fraction_produces_fixed_size_padded_crop(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)

        for x_fraction, y_fraction in ((-0.2, 0.5), (1.2, 0.5), (0.5, -0.2), (0.5, 1.2)):
            crop = _extract_particle_crop(image, x_fraction, y_fraction, crop_size=4)
            self.assertEqual(crop.shape, (4, 4))
            self.assertTrue(np.isfinite(crop).all())

    def test_typical_control_prefers_same_micrograph_then_nearest_ncc(self):
        power = np.arange(100, dtype=np.float64)
        ncc = np.zeros(100, dtype=np.float64)
        uids = [1000 + index for index in range(100)]
        uids[42] = uids[58] = uids[95] = 777
        ncc[42] = 0.55
        ncc[58] = 0.61
        ncc[95] = 0.60

        controls, metadata = select_typical_power_matched_controls(
            [{"particle_index": 95, "tail_bin": "P99-P99.5"}],
            uids,
            ncc,
            power,
        )

        self.assertEqual(controls[0]["particle_index"], 58)
        self.assertEqual(controls[0]["matching_scope"], "same_micrograph")
        self.assertAlmostEqual(controls[0]["ncc_difference"], 0.01)
        self.assertEqual(metadata["same_micrograph_match_count"], 1)
        self.assertFalse(metadata["visual_selection_used"])

    def test_pair_normalization_uses_one_joint_grayscale_range(self):
        control = np.linspace(0.0, 10.0, 64, dtype=np.float32).reshape(8, 8)
        target = np.linspace(100.0, 110.0, 64, dtype=np.float32).reshape(8, 8)

        control_tile, target_tile, metadata = _normalize_particle_pair(control, target, 8)

        self.assertEqual(metadata["scope"], "per_pair_joint_percentile")
        self.assertEqual(metadata["percentiles"], [1, 99])
        self.assertLess(int(control_tile.max()), int(target_tile.min()))

    def test_j46_exact_path_precedes_api_and_dataset_fallback(self):
        class Project:
            def download_mrc(self, _path):
                raise AssertionError("API must not run when the J46 exact path is readable")

        with tempfile.TemporaryDirectory() as temp_dir:
            exact_path = Path(temp_dir) / "j46_exact.mrc"
            exact_path.touch()
            with patch("vision_inputs._read_mrc_path", return_value=np.ones((4, 5))) as read:
                image, metadata = load_micrograph_image(
                    Project(), str(exact_path), temp_dir, micrograph_uid=123, return_metadata=True
                )

        self.assertEqual(image.shape, (4, 5))
        read.assert_called_once_with(exact_path)
        self.assertEqual(metadata["micrograph_uid"], 123)
        self.assertEqual(metadata["selected_source_type"], "j46_exact")
        self.assertEqual(metadata["actual_source_path_or_blob"], str(exact_path))
        self.assertEqual(metadata["image_shape"], [4, 5])

    def test_cryosparc_api_precedes_dataset_fallback_for_relative_j46_blob(self):
        class Project:
            def download_mrc(self, path):
                self.path = path
                return None, np.ones((6, 7))

        project = Project()
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "same_name.mrc").touch()
            with patch("vision_inputs._read_mrc_path") as read:
                image, metadata = load_micrograph_image(
                    project, "J46/imported/same_name.mrc", temp_dir,
                    micrograph_uid=456, return_metadata=True,
                )

        self.assertEqual(image.shape, (6, 7))
        self.assertEqual(project.path, "J46/imported/same_name.mrc")
        read.assert_not_called()
        self.assertEqual(metadata["selected_source_type"], "cryosparc_api")
        self.assertEqual(metadata["actual_source_path_or_blob"], "J46/imported/same_name.mrc")

    def test_dataset_path_is_only_used_after_exact_and_api_fail(self):
        class Project:
            def download_mrc(self, _path):
                raise RuntimeError("offline")

        with tempfile.TemporaryDirectory() as temp_dir:
            fallback_path = Path(temp_dir) / "fallback.mrc"
            fallback_path.touch()
            with patch("vision_inputs._read_mrc_path", return_value=np.ones((8, 9))) as read:
                image, metadata = load_micrograph_image(
                    Project(), "J46/imported/fallback.mrc", temp_dir,
                    micrograph_uid=789, return_metadata=True,
                )

        self.assertEqual(image.shape, (8, 9))
        read.assert_called_once_with(fallback_path)
        self.assertEqual(metadata["selected_source_type"], "dataset_fallback")
        self.assertTrue(metadata["preceding_source_errors"])


if __name__ == "__main__":
    unittest.main()
