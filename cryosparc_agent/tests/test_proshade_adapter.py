import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from proshade_adapter import parse_proshade_output, run_proshade


SAMPLE_OUTPUT = """
ProSHADE default symmetry detection algorithm claims the symmetry to be C-12 with axes:
======================================================================================
   Type     Fold       X           Y          Z           Angle        Height      Average FSC
     C       +12      +0.00000   +0.00000   +1.00000     +0.52360      +0.99912      +0.99533
"""


class ProshadeAdapterTests(unittest.TestCase):
    def test_parses_recommended_symmetry_and_axis_metrics(self):
        parsed = parse_proshade_output(SAMPLE_OUTPUT)
        self.assertEqual(parsed["recommended_symmetry"], "C12")
        self.assertEqual(parsed["primary_axis"], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(parsed["average_fsc"], 0.99533)

    @patch("proshade_adapter.subprocess.run")
    def test_wraps_successful_cli_result_as_advisory_evidence(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["proshade"], 0, stdout=SAMPLE_OUTPUT, stderr=""
        )
        result = run_proshade(Path("/tmp/map.mrc"), binary=Path("/tmp/proshade"))
        self.assertTrue(result["success"])
        self.assertEqual(result["recommended_symmetry"], "C12")
        self.assertEqual(result["confidence"], "high")
        self.assertIn("--symmetry", result["command"])

    @patch("proshade_adapter.subprocess.run")
    def test_preserves_nonzero_exit_as_structured_failure(self, run):
        run.return_value = subprocess.CompletedProcess(
            ["proshade"], 2, stdout="", stderr="invalid map"
        )
        result = run_proshade(Path("/tmp/map.mrc"), binary=Path("/tmp/proshade"))
        self.assertFalse(result["success"])
        self.assertEqual(result["error_code"], "proshade_failed")
