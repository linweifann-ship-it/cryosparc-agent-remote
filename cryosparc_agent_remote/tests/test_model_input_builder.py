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
                "workflow_node_id": "J7",
                "logical_node_id": "node_inspect_picks_v2_001",
                "cryosparc_job_uid": "J7",
                "job_type": "inspect_picks_v2",
                "title": "Inspect picks",
                "status": "completed",
                "updated_at": "2026-06-28T23:59:00+00:00",
                "timestamps": {},
                "parent_job_uids": [],
                "parent_workflow_node_ids": [],
                "parent_logical_node_ids": [],
                "child_job_uids": ["J8"],
                "child_workflow_node_ids": ["J8"],
                "child_logical_node_ids": ["node_extract_micrographs_multi_001"],
                "inputs": {},
                "outputs": {
                    "particles": {
                        "type": "particle",
                        "num_items": 176623,
                        "available": True,
                        "result_names": ["ctf", "location"],
                        "summary_keys": [],
                        "latest_summary_stat_keys": [],
                    }
                },
                "key_parameters": {},
                "runtime": {},
                "has_error": False,
                "has_warning": False,
            },
            {
                "workflow_node_id": "J6",
                "logical_node_id": "node_template_picker_gpu_001",
                "cryosparc_job_uid": "J6",
                "job_type": "template_picker_gpu",
                "title": "Unrelated failed branch",
                "status": "completed",
                "updated_at": "2026-06-28T23:58:00+00:00",
                "timestamps": {},
                "parent_job_uids": [],
                "parent_workflow_node_ids": [],
                "parent_logical_node_ids": [],
                "child_job_uids": [],
                "child_workflow_node_ids": [],
                "child_logical_node_ids": [],
                "inputs": {},
                "outputs": {
                    "templates": {
                        "type": "template",
                        "num_items": 6,
                        "available": True,
                        "result_names": ["template"],
                        "summary_keys": [],
                        "latest_summary_stat_keys": [],
                    }
                },
                "key_parameters": {},
                "runtime": {},
                "run_errors": {},
                "has_error": False,
                "has_warning": False,
            },
            {
                "workflow_node_id": "J8",
                "logical_node_id": "node_extract_micrographs_multi_001",
                "cryosparc_job_uid": "J8",
                "job_type": "extract_micrographs_multi",
                "title": "Extract from picks",
                "status": status,
                "updated_at": "2026-06-29T00:00:00+00:00",
                "timestamps": {
                    "created_at": "2026-06-29T00:00:00+00:00",
                    "queued_at": "2026-06-29T00:01:00+00:00",
                    "started_at": "2026-06-29T00:02:00+00:00",
                    "running_at": "2026-06-29T00:03:00+00:00",
                    "completed_at": "2026-06-29T00:04:00+00:00",
                    "failed_at": None,
                    "killed_at": None,
                    "updated_at": "2026-06-29T00:04:00+00:00",
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
                "runtime": {
                    "work_dir": "J8",
                    "lane": "g8m192_4090_slurm",
                    "worker_hostname": None,
                    "allocated_cpu": None,
                    "allocated_gpu": None,
                    "allocated_ram": None,
                    "allocated_ssd": None,
                },
                "run_errors": {},
                "has_error": False,
                "has_warning": False,
            },
            {
                "workflow_node_id": "J9",
                "logical_node_id": "node_extract_micrographs_multi_002",
                "cryosparc_job_uid": "J9",
                "job_type": "extract_micrographs_multi",
                "title": "Failed retry",
                "status": "building",
                "updated_at": "2026-06-29T00:05:00+00:00",
                "timestamps": {},
                "parent_job_uids": [],
                "parent_workflow_node_ids": [],
                "parent_logical_node_ids": [],
                "child_job_uids": [],
                "child_workflow_node_ids": [],
                "child_logical_node_ids": [],
                "inputs": {},
                "outputs": {},
                "key_parameters": {"box_size_pix": 384},
                "runtime": {},
                "run_errors": {"errors_build_inputs": "Inputs were empty."},
                "has_error": True,
                "has_warning": False,
            },
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

        self.assertEqual(result["schema_version"], "2.1")
        self.assertEqual(result["task_type"], "workflow_decision")
        self.assertNotIn("candidate_actions", result)
        self.assertNotIn("dataset_info", result)
        self.assertEqual(
            result["dataset_context"]["dataset_metadata"]["known_workflow_steps"],
            None,
        )
        self.assertEqual(result["current_state"]["last_node_id"], "J8")
        self.assertEqual(result["current_state"]["last_node_status"], "completed")
        self.assertEqual(
            result["current_state"]["state_features"]["particle_count"],
            176623,
        )
        self.assertEqual(
            result["current_state"]["last_node_info"]["metrics"]["particles_count"],
            176623,
        )
        node_info = result["current_state"]["last_node_info"]
        self.assertEqual(
            node_info["timestamps"]["completed_at"],
            "2026-06-29T00:04:00+00:00",
        )
        self.assertEqual(
            node_info["runtime"]["lane"],
            "g8m192_4090_slurm",
        )
        self.assertEqual(
            [
                {
                    item["job_uid"]: item["status"]
                    for item in result["current_state"]["recent_job_history"]
                }
            ][0],
            {"J7": "completed", "J8": "completed", "J9": "failure"},
        )
        self.assertNotIn(
            "J6",
            [
                item["job_uid"]
                for item in result["current_state"]["recent_job_history"]
            ],
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

    def test_known_workflow_retriever_handles_cryosparc_jobs_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empiar-10025-workflow-standard.json"
            path.write_text(
                json.dumps(
                    {
                        "jobs": {
                            "J14": {
                                "jobType": "homo_abinit",
                                "groups": [["J13.particles_selected", "particles"]],
                                "parameters": {
                                    "abinit_K": {"value": 1},
                                },
                            },
                            "J1": {
                                "jobType": "import_movies",
                                "groups": [],
                                "parameters": {
                                    "psize_A": {"value": 0.6575},
                                },
                            },
                        }
                    }
                )
            )

            steps = retrieve_known_workflow_steps(
                {"empiar_id": "EMPIAR-10025"},
                search_dirs=[tmpdir],
            )

        self.assertEqual(steps[0]["node_id"], "J1")
        self.assertEqual(steps[0]["action"], "import_movies")
        self.assertEqual(steps[0]["parameter_template"], {"psize_A": 0.6575})
        self.assertEqual(steps[1]["node_id"], "J14")
        self.assertEqual(steps[1]["action"], "homo_abinit")
        self.assertEqual(steps[1]["upstream_node_ids"], ["J13"])


if __name__ == "__main__":
    unittest.main()
