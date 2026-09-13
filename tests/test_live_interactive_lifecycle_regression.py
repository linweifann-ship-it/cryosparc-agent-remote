import importlib.util
import unittest
from pathlib import Path


def load_cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "live_interactive_lifecycle_regression.py"
    spec = importlib.util.spec_from_file_location("live_interactive_regression", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveInteractiveRegressionTests(unittest.TestCase):
    def test_materializes_registry_sources(self):
        cli = load_cli()
        candidate = {"required_inputs": {"micrographs": [{"source_job_uid": "J13", "source_output": "micrographs"}], "particles": [{"source_job_uid": "J13", "source_output": "particles"}]}}
        self.assertEqual(cli.candidate_connections(candidate), {"micrographs": {"source_job_uid": "J13", "source_output": "micrographs"}, "particles": {"source_job_uid": "J13", "source_output": "particles"}})

    def test_missing_required_source_fails_before_create(self):
        cli = load_cli()
        with self.assertRaisesRegex(ValueError, "mandatory input 'particles'"):
            cli.candidate_connections({"required_inputs": {"particles": []}})


if __name__ == "__main__":
    unittest.main()
