# Fixed tests for MCP-internal job status and model-facing result packages.
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from job_result import get_job_result_package


def workflow_state(
    status: str,
    old_active: bool = False,
    job_type: str = "extract_micrographs_multi",
    has_error: bool = False,
) -> dict:
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
                "job_type": job_type,
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
                    "failed_at": started_at.isoformat() if status == "failed" else None,
                    "killed_at": started_at.isoformat() if status == "killed" else None,
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
                "run_errors": {"raw": "Traceback\nRuntimeError: worker failed"},
                "has_error": has_error,
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

    def test_select_2d_waiting_requires_human_action(self):
        with patch(
            "job_result.extract_workflow_state",
            return_value=workflow_state("waiting", job_type="select_2D"),
        ):
            result = get_job_result_package("P2", "W3", "J30")

        self.assertTrue(result["success"])
        self.assertFalse(result["ready_for_model"])
        self.assertEqual(result["status_group"], "human_action_required")
        self.assertTrue(result["human_action_required"])
        self.assertEqual(
            result["human_action"]["action_type"],
            "cryosparc_interactive_selection",
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
        self.assertNotIn("selected_actions_rule", result["output_contract"])
        self.assertIn("decision_rule", result["output_contract"])
        self.assertEqual(result["metrics"]["num_items_by_output"]["particles"], 100)

    def test_failed_job_returns_model_visible_failure_context(self):
        with patch(
            "job_result.extract_workflow_state",
            return_value=workflow_state("failed", has_error=True),
        ):
            result = get_job_result_package("P2", "W3", "J30")

        self.assertFalse(result["success"])
        self.assertTrue(result["ready_for_model"])
        self.assertFalse(result["internal_only"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_context"]["job_uid"], "J30")
        self.assertEqual(
            result["failure_context"]["allowed_model_decisions"],
            ["retry", "rollback", "branch", "stop"],
        )
        self.assertIn("RuntimeError", result["failure_context"]["run_errors"]["raw"])

    def test_physical_race_loser_returns_completed_sibling_result(self):
        state = workflow_state("killed", job_type="homo_abinit")
        loser = state["nodes"][0]
        loser["cryosparc_job_uid"] = "J230"
        loser["workflow_node_id"] = "J230"
        loser["title"] = "Agent registry_J229_homo_abinit physical_1"
        loser["status"] = "killed"
        winner = {
            **loser,
            "cryosparc_job_uid": "J231",
            "workflow_node_id": "J231",
            "title": "Agent registry_J229_homo_abinit physical_2",
            "status": "completed",
            "has_error": False,
            "run_errors": {},
            "outputs": {
                "volume": {
                    "type": "volume",
                    "num_items": 1,
                    "available": True,
                    "result_names": ["map"],
                    "summary_keys": ["map/res_A"],
                    "latest_summary_stat_keys": ["map/res_A"],
                }
            },
        }
        state["nodes"].append(winner)
        state["node_mapping"]["J230"] = "J230"
        state["node_mapping"]["J231"] = "J231"

        with patch(
            "job_result.extract_workflow_state",
            return_value=state,
        ), patch(
            "job_result.get_next_candidate_context",
            return_value={"candidate_actions": [], "blocked_actions": [], "decision_hint": None},
        ):
            result = get_job_result_package("P2", "W3", "J230")

        self.assertTrue(result["ready_for_model"])
        self.assertEqual(result["job_uid"], "J231")
        self.assertEqual(result["race_resolution"]["requested_job_uid"], "J230")
        self.assertEqual(result["race_resolution"]["winner_job_uid"], "J231")


if __name__ == "__main__":
    unittest.main()
