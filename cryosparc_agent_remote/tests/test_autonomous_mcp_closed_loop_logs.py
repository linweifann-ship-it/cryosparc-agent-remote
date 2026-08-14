import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


def load_runner_module():
    if "mcp" not in sys.modules:
        mcp = types.ModuleType("mcp")
        mcp.ClientSession = object
        mcp.StdioServerParameters = object
        sys.modules["mcp"] = mcp
    if "mcp.client" not in sys.modules:
        sys.modules["mcp.client"] = types.ModuleType("mcp.client")
    if "mcp.client.stdio" not in sys.modules:
        stdio = types.ModuleType("mcp.client.stdio")
        stdio.stdio_client = object
        sys.modules["mcp.client.stdio"] = stdio

    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "autonomous_mcp_closed_loop.py"
    )
    spec = importlib.util.spec_from_file_location(
        "autonomous_mcp_closed_loop_for_log_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LocalJobLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner_module()

    def test_rejects_reused_job_uid_from_foreign_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "uid": "J132",
                        "project_uid": "P2",
                        "workspace_uids": ["W23"],
                        "job_type": "template_picker_gpu",
                        "type": "template_picker_gpu",
                        "created_at": {"$date": "2025-06-17T09:50:26.673Z"},
                    }
                )
            )
            (job_dir / "job.log").write_text("stale template picker traceback")

            result = self.runner.collect_job_logs(
                {
                    "project_uid": "P2",
                    "workspace_uid": "W6",
                    "job_uid": "J132",
                    "job_type": "homo_refine_new",
                    "timestamps": {
                        "created_at": "2026-07-16T06:41:22.684000+00:00"
                    },
                    "runtime": {"work_dir": str(job_dir)},
                },
                project_uid="P2",
                workspace_uid="W6",
            )

        self.assertEqual(result["status"], "stale_or_foreign_job_log")
        mismatch_fields = {item["field"] for item in result["mismatches"]}
        self.assertIn("workspace_uid", mismatch_fields)
        self.assertIn("job_type", mismatch_fields)
        self.assertIn("created_at", mismatch_fields)
        self.assertNotIn("files", result)
        self.assertNotIn("stale template picker traceback", json.dumps(result))

    def test_reads_logs_only_after_identity_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            created_at = "2026-07-16T06:41:22.684000+00:00"
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "uid": "J132",
                        "project_uid": "P2",
                        "workspace_uids": ["W6"],
                        "job_type": "homo_refine_new",
                        "created_at": created_at,
                    }
                )
            )
            (job_dir / "job.log").write_text("verified homo_refine_new failure")

            result = self.runner.collect_job_logs(
                {
                    "project_uid": "P2",
                    "workspace_uid": "W6",
                    "job_uid": "J132",
                    "job_type": "homo_refine_new",
                    "timestamps": {"created_at": created_at},
                    "runtime": {"work_dir": str(job_dir)},
                },
                project_uid="P2",
                workspace_uid="W6",
            )

        self.assertEqual(result["status"], "ok")
        self.assertIn("verified homo_refine_new failure", json.dumps(result["files"]))

    def test_resolves_relative_job_dir_against_trusted_project_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            job_dir = project_dir / "J132"
            job_dir.mkdir()
            created_at = "2026-07-16T06:41:22.684000+00:00"
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "uid": "J132",
                        "project_uid": "P2",
                        "workspace_uids": ["W6"],
                        "job_type": "homo_refine_new",
                        "created_at": created_at,
                    }
                )
            )
            (job_dir / "job.log").write_text("project dir resolved log")

            result = self.runner.collect_job_logs(
                {
                    "project_uid": "P2",
                    "workspace_uid": "W6",
                    "job_uid": "J132",
                    "job_type": "homo_refine_new",
                    "timestamps": {"created_at": created_at},
                    "runtime": {
                        "work_dir": "J132",
                        "project_dir": str(project_dir),
                    },
                },
                project_uid="P2",
                workspace_uid="W6",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["job_dir"], str(job_dir))

    def test_does_not_guess_fixed_paths_without_api_job_dir(self):
        result = self.runner.collect_job_logs(
            {
                "project_uid": "P2",
                "workspace_uid": "W6",
                "job_uid": "J132",
                "job_type": "homo_refine_new",
                "timestamps": {
                    "created_at": "2026-07-16T06:41:22.684000+00:00"
                },
                "runtime": {},
            },
            project_uid="P2",
            workspace_uid="W6",
        )

        self.assertEqual(result["status"], "no_verified_job_dir")

    def test_execution_error_uses_api_errors_run_raw_text(self):
        payload = {
            "success": False,
            "ready_for_model": True,
            "status": "failed",
            "run_errors": {
                "worker": [
                    "Traceback (most recent call last):",
                    "RuntimeError: GPU worker failed during refinement",
                ]
            },
            "local_job_logs": {
                "status": "no_verified_job_dir",
            },
        }

        error = self.runner.extract_execution_error("job_result", payload)

        self.assertIn("RuntimeError: GPU worker failed", error["raw_error_text"])
        self.assertEqual(error["source_used"], "api.run_errors")
        self.assertTrue(error["log_identity_verified"])
        self.assertIn("api.run_errors", error["sources_checked"])
        self.assertNotIn("error_type", error)

    def test_execution_error_rejects_stale_local_log_text(self):
        payload = {
            "success": False,
            "ready_for_model": True,
            "status": "failed",
            "run_errors": {},
            "local_job_logs": {
                "status": "stale_or_foreign_job_log",
                "observed_identity": {
                    "uid": "J132",
                    "workspace_uid": "W23",
                    "job_type": "template_picker_gpu",
                },
                "mismatches": [
                    {
                        "field": "job_type",
                        "expected": "homo_refine_new",
                        "observed": "template_picker_gpu",
                    }
                ],
            },
        }

        error = self.runner.extract_execution_error("job_result", payload)

        self.assertEqual(error["raw_error_text"], "error_text_unavailable")
        self.assertEqual(error["source_used"], "error_text_unavailable")
        self.assertFalse(error["log_identity_verified"])
        self.assertIn(
            "local_job_logs.stale_or_foreign_job_log",
            error["sources_checked"],
        )
        self.assertNotIn("template_picker_gpu traceback", json.dumps(error))

    def test_failure_context_carries_raw_error_without_retry_guidance(self):
        feedback = {
            "round": 4,
            "feedback_type": "job_result",
            "payload": {
                "success": False,
                "ready_for_model": True,
                "status": "failed",
                "run_errors": ["Traceback\nValueError: bad input array"],
                "local_job_logs": {"status": "no_verified_job_dir"},
            },
            "failed_decision": {
                "decision_type": "forward",
                "action": "homo_refine_new",
                "job_type": "homo_refine_new",
            },
        }

        context = self.runner.build_failure_context(feedback)

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("ValueError: bad input array", context["execution_error"]["raw_error_text"])
        self.assertEqual(
            context["attempt_history"][0]["raw_error_text"],
            context["execution_error"]["raw_error_text"],
        )
        self.assertNotIn("retry_guidance", context)
        self.assertNotIn("error_type", context["execution_error"])


if __name__ == "__main__":
    unittest.main()
