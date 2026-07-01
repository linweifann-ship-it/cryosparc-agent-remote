# Fixed tests for MCP-internal job status and model-facing result packages.
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from job_result import get_job_result_package


def workflow_state(status: str, old_active: bool = False) -> dict:
    """Return a minimal normalized workflow state containing one test job."""
    started_at = (
        datetime.now(timezone.utc) - timedelta(hours=24)
        if old_active
        else datetime.now(timezone.utc)
    )
    output_items = 0 if old_active else 100
    return {
        "schema_version": "1.0",
        "generated_at": "2026-06-27T00:00:00+00:00",
        "project_uid": "P2",
        "workspace_uid": "W3",
        "workflow_status": "running" if status == "queued" else status,
        "nodes": [
            {
                "workflow_node_id": "J30",
                "logical_node_id": "node_extract_micrographs_multi_002",
                "cryosparc_job_uid": "J30",
                "job_type": "extract_micrographs_multi",
                "title": "Agent forward_J8",
                "status": status,
                "updated_at": "2026-06-27T00:00:00+00:00",
                "timestamps": {
                    "created_at": started_at.isoformat(),
                    "queued_at": started_at.isoformat(),
                    "started_at": started_at.isoformat(),
                    "running_at": started_at.isoformat(),
                    "launched_at": started_at.isoformat(),
                    "completed_at": None,
                    "failed_at": None,
                    "killed_at": None,
                    "heartbeat_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                "parent_job_uids": ["J7"],
                "parent_workflow_node_ids": ["J7"],
                "parent_logical_node_ids": ["node_inspect_picks_v2_001"],
                "child_job_uids": [],
                "child_workflow_node_ids": [],
                "child_logical_node_ids": [],
                "inputs": {
                    "particles": [
                        {
                            "source_workflow_node_id": "J7",
                            "source_logical_node_id": "node_inspect_picks_v2_001",
                            "source_job_uid": "J7",
                            "source_output": "particles",
                            "result_names": ["location"],
                        }
                    ]
                },
                "outputs": {
                    "particles": {
                        "type": "particle",
                        "num_items": output_items,
                        "available": output_items > 0,
                        "result_names": ["blob", "location"],
                        "summary_keys": ["num_items"],
                        "latest_summary_stat_keys": ["num_items"],
                    }
                },
                "key_parameters": {
                    "box_size_pix": 400,
                    "compute_num_gpus": 4,
                },
                "has_error": False,
                "has_warning": False,
            }
        ],
        "edges": [],
        "root_nodes": [],
        "terminal_nodes": ["J30"],
        "running_nodes": ["J30"] if status == "queued" else [],
        "failed_nodes": [],
        "node_mapping": {"J30": "J30"},
    }


class JobResultPackageTests(unittest.TestCase):
    def test_queued_job_is_internal_only(self):
        with patch("job_result.extract_workflow_state", return_value=workflow_state("queued")):
            result = get_job_result_package("P2", "W3", "J30")

        self.assertTrue(result["success"])
        self.assertFalse(result["ready_for_model"])
        self.assertTrue(result["internal_only"])
        self.assertEqual(result["message_type"], "mcp_internal_job_status")
        self.assertEqual(result["status_group"], "active")

    def test_old_active_job_without_outputs_needs_attention(self):
        with patch(
            "job_result.extract_workflow_state",
            return_value=workflow_state("running", old_active=True),
        ):
            result = get_job_result_package("P2", "W3", "J30")

        self.assertTrue(result["success"])
        self.assertFalse(result["ready_for_model"])
        self.assertEqual(result["status_group"], "attention_required")
        self.assertTrue(result["monitoring"]["attention_required"])
        self.assertIn(
            "max_runtime_exceeded",
            result["monitoring"]["flags"],
        )
        self.assertIn(
            "no_registered_output_progress",
            result["monitoring"]["flags"],
        )

    def test_completed_job_returns_model_result_package(self):
        candidate_context = {
            "candidate_actions": [{"action_id": "forward_J31"}],
            "blocked_actions": [],
            "decision_hint": None,
        }
        with patch(
            "job_result.extract_workflow_state",
            return_value=workflow_state("completed"),
        ), patch(
            "job_result.get_next_candidate_context",
            return_value=candidate_context,
        ):
            result = get_job_result_package("P2", "W3", "J30")

        self.assertTrue(result["success"])
        self.assertTrue(result["ready_for_model"])
        self.assertFalse(result["internal_only"])
        self.assertEqual(result["message_type"], "mcp_job_result")
        self.assertEqual(result["next_candidate_actions"], [{"action_id": "forward_J31"}])
        self.assertEqual(result["metrics"]["num_items_by_output"]["particles"], 100)


if __name__ == "__main__":
    unittest.main()
