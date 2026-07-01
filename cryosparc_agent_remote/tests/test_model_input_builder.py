# Fixed tests for V2 model input payload construction.
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from known_workflow_retriever import retrieve_known_workflow_steps
from model_input_builder import build_model_input_payload


def workflow_state(status: str = "completed") -> dict:
    """Return a minimal workflow state for V2 payload tests."""
    return {
        "schema_version": "1.0",
        "generated_at": "2026-06-29T00:00:00+00:00",
        "project_uid": "P2",
        "workspace_uid": "W3",
        "workflow_status": status,
        "nodes": [
            {
                "workflow_node_id": "J8",
                "logical_node_id": "node_extract_micrographs_multi_001",
                "cryosparc_job_uid": "J8",
                "job_type": "extract_micrographs_multi",
                "title": "Extract from picks",
                "status": status,
                "updated_at": "2026-06-29T00:00:00+00:00",
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
                            "result_names": ["ctf", "location"],
                        }
                    ]
                },
                "outputs": {
                    "particles": {
                        "type": "particle",
                        "num_items": 176623,
                        "available": status == "completed",
                        "result_names": ["blob", "ctf", "location"],
                        "summary_keys": ["blob/shape"],
                        "latest_summary_stat_keys": [],
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
        "terminal_nodes": ["J8"],
        "running_nodes": ["J8"] if status == "running" else [],
        "failed_nodes": [],
        "node_mapping": {"J8": "J8"},
    }


class ModelInputBuilderTests(unittest.TestCase):
    def test_completed_job_returns_v2_payload(self):
        with patch(
            "model_input_builder.extract_workflow_state",
            return_value=workflow_state("completed"),
        ):
            result = build_model_input_payload(
                "P2",
                "W3",
                current_job_uid="J8",
                dataset_info={"empiar_id": "EMPIAR-12099"},
            )

        self.assertEqual(result["schema_version"], "2.0")
        self.assertEqual(result["task_type"], "workflow_decision")
        self.assertNotIn("candidate_actions", result)
        self.assertEqual(result["dataset_info"]["known_workflow_steps"], None)
        self.assertEqual(result["current_state"]["last_node_id"], "J8")
        self.assertEqual(
            result["current_state"]["last_node_info"]["metrics"]["particles_count"],
            176623,
        )

    def test_running_job_returns_internal_status(self):
        with patch(
            "model_input_builder.extract_workflow_state",
            return_value=workflow_state("running"),
        ):
            result = build_model_input_payload(
                "P2",
                "W3",
                current_job_uid="J8",
                dataset_info={"empiar_id": "EMPIAR-12099"},
            )

        self.assertEqual(result["message_type"], "mcp_internal_job_status")
        self.assertFalse(result["ready_for_model"])
        self.assertNotIn("current_state", result)

    def test_known_workflow_retriever_returns_normalized_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "EMPIAR-12099_workflow.json"
            path.write_text(
                json.dumps(
                    {
                        "empiar_id": "EMPIAR-12099",
                        "steps": [
                            {
                                "node_id": "J1",
                                "job_type": "import_movies",
                                "parents": [],
                                "params": {"psize_A": 1.0},
                            }
                        ],
                    }
                )
            )

            steps = retrieve_known_workflow_steps(
                {"empiar_id": "EMPIAR-12099"},
                search_dirs=[tmpdir],
            )

        self.assertEqual(steps[0]["step_index"], 0)
        self.assertEqual(steps[0]["action"], "import_movies")
        self.assertEqual(steps[0]["parameter_template"], {"psize_A": 1.0})


if __name__ == "__main__":
    unittest.main()
