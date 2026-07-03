# Fixed dry-run tests for model decision validation and planning.
import unittest

from action_registry import (
    execute_model_decision_payload,
    generate_candidate_actions,
    validate_model_decision_payload,
)


def candidate_actions():
    return [
        {
            "action_id": "forward_J8",
            "action_type": "forward",
            "workflow_node_id": "J8",
            "reference_job_uid": "J8",
            "job_type": "extract_micrographs_multi",
            "description": "Dry-run candidate for fixed tests.",
            "execution_mode": "dry_run_only",
            "available": True,
            "blocked_by": [],
            "required_inputs": {},
            "parameter_template": {
                "compute_num_gpus": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 4,
                },
                "box_size_pix": {
                    "type": "integer",
                    "minimum": 32,
                    "default": 400,
                },
            },
            "default_parameters": {
                "compute_num_gpus": 4,
                "box_size_pix": 400,
            },
        }
    ]


def select_2d_workflow_state():
    """Return a minimal state where select_2D can feed homo_refine_new."""
    return {
        "schema_version": "1.0",
        "generated_at": "2026-07-02T00:00:00+00:00",
        "project_uid": "P2",
        "workspace_uid": "W4",
        "workflow_status": "completed",
        "nodes": [
            {
                "workflow_node_id": "J2",
                "logical_node_id": "node_import_volumes_001",
                "cryosparc_job_uid": "J2",
                "job_type": "import_volumes",
                "title": "Import volume",
                "status": "completed",
                "updated_at": "2026-07-02T00:00:00+00:00",
                "parent_job_uids": [],
                "parent_workflow_node_ids": [],
                "parent_logical_node_ids": [],
                "child_job_uids": [],
                "child_workflow_node_ids": [],
                "child_logical_node_ids": [],
                "inputs": {},
                "outputs": {
                    "imported_volume_1": {
                        "type": "volume",
                        "num_items": 1,
                        "available": True,
                        "result_names": ["map"],
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
                "workflow_node_id": "J37",
                "logical_node_id": "node_select_2D_001",
                "cryosparc_job_uid": "J37",
                "job_type": "select_2D",
                "title": "Select 2D",
                "status": "completed",
                "updated_at": "2026-07-02T00:00:00+00:00",
                "parent_job_uids": [],
                "parent_workflow_node_ids": [],
                "parent_logical_node_ids": [],
                "child_job_uids": [],
                "child_workflow_node_ids": [],
                "child_logical_node_ids": [],
                "inputs": {},
                "outputs": {
                    "particles_selected": {
                        "type": "particle",
                        "num_items": 133812,
                        "available": True,
                        "result_names": [
                            "alignments2D",
                            "blob",
                            "ctf",
                            "location",
                            "pick_stats",
                        ],
                        "summary_keys": [],
                        "latest_summary_stat_keys": [],
                    }
                },
                "key_parameters": {},
                "runtime": {},
                "has_error": False,
                "has_warning": False,
            },
        ],
        "edges": [],
        "root_nodes": ["J2", "J37"],
        "terminal_nodes": ["J2", "J37"],
        "running_nodes": [],
        "failed_nodes": [],
        "node_mapping": {"J2": "J2", "J37": "J37"},
    }


def homo_refine_workflow_state():
    """Return a minimal state where homo_refine_new can be run in parallel."""
    state = select_2d_workflow_state()
    state["nodes"].append(
        {
            "workflow_node_id": "J39",
            "logical_node_id": "node_homo_refine_new_001",
            "cryosparc_job_uid": "J39",
            "job_type": "homo_refine_new",
            "title": "Homogeneous Refinement",
            "status": "completed",
            "updated_at": "2026-07-02T01:00:00+00:00",
            "parent_job_uids": ["J37", "J2"],
            "parent_workflow_node_ids": ["J37", "J2"],
            "parent_logical_node_ids": [
                "node_select_2D_001",
                "node_import_volumes_001",
            ],
            "child_job_uids": [],
            "child_workflow_node_ids": [],
            "child_logical_node_ids": [],
            "inputs": {
                "particles": [
                    {
                        "source_workflow_node_id": "J37",
                        "source_logical_node_id": "node_select_2D_001",
                        "source_job_uid": "J37",
                        "source_output": "particles_selected",
                        "result_names": ["blob", "ctf", "location"],
                    }
                ],
                "volume": [
                    {
                        "source_workflow_node_id": "J2",
                        "source_logical_node_id": "node_import_volumes_001",
                        "source_job_uid": "J2",
                        "source_output": "imported_volume_1",
                        "result_names": ["map"],
                    }
                ],
            },
            "outputs": {
                "volume": {
                    "type": "volume",
                    "num_items": 1,
                    "available": True,
                    "result_names": ["map"],
                    "summary_keys": [],
                    "latest_summary_stat_keys": [],
                }
            },
            "key_parameters": {},
            "runtime": {},
            "has_error": False,
            "has_warning": False,
        }
    )
    state["terminal_nodes"] = ["J39"]
    state["node_mapping"]["J39"] = "J39"
    return state


def homo_refine_decision():
    """Return a model decision for homogeneous refinement."""
    return {
        "schema_version": "1.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "action_id": "forward_J37_homo_refine_new",
                "action_type": "forward",
                "workflow_node_id": "J37:homo_refine_new",
                "job_type": "homo_refine_new",
                "parameters": {
                    "compute_use_ssd": False,
                    "refine_ctf_global_refine": True,
                    "refine_defocus_refine": True,
                },
            }
        ],
        "rollback_target": None,
        "branch_plan": None,
        "reason": "Selected particles are ready for refinement.",
        "confidence": 0.94,
        "risk_flags": [],
        "evidence": ["J37 completed with selected particles."],
    }


def parallel_homo_refine_decision():
    """Return a model decision for another homogeneous refinement."""
    decision = homo_refine_decision()
    decision["selected_actions"][0].update(
        {
            "action_id": "forward_J39_homo_refine_new",
            "workflow_node_id": "J39:homo_refine_new",
        }
    )
    decision["reason"] = "Run another refinement with alternate parameters."
    return decision


def forward_decision(**overrides):
    decision = {
        "schema_version": "1.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "action_id": "forward_J8",
                "action_type": "forward",
                "workflow_node_id": "J8",
                "job_type": "extract_micrographs_multi",
                "parameters": {
                    "compute_num_gpus": 4,
                    "box_size_pix": 400,
                },
            }
        ],
        "rollback_target": None,
        "branch_plan": None,
        "reason": "Particles are ready for extraction.",
        "confidence": 0.94,
        "risk_flags": [],
        "evidence": ["Selected particles are available."],
    }
    decision.update(overrides)
    return decision


class ActionRegistryFixedTests(unittest.TestCase):
    def test_valid_forward_returns_planned_action(self):
        result = execute_model_decision_payload(
            forward_decision(),
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_mode"], "dry_run")
        self.assertEqual(result["execution_plan"]["plan_version"], "1.0")
        self.assertEqual(result["execution_plan"]["status"], "planned")
        self.assertEqual(result["execution_plan"]["action_count"], 1)
        self.assertFalse(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_id"],
            "forward_J8",
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["status"],
            "planned",
        )

    def test_parameter_above_maximum_is_invalid(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 16

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "parameter_above_maximum")

        execute_result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )
        self.assertFalse(execute_result["success"])
        self.assertIsNone(execute_result["execution_plan"])

    def test_null_required_parameter_is_reported_as_missing(self):
        decision = forward_decision()
        decision["selected_actions"][0] = {
            "job_type": "import_movies",
            "parameters": {
                "blob_paths": None,
                "psize_A": 0.982,
            },
        }

        result = validate_model_decision_payload(
            decision,
            candidate_actions=[],
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["issues"][0]["code"], "missing_required_parameter")

    def test_high_gpu_count_requires_approval(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 8

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        action = result["execution_plan"]["actions"][0]
        self.assertTrue(result["success"])
        self.assertTrue(result["execution_plan"]["approval_required"])
        self.assertTrue(action["approval_required"])
        self.assertIn("high_gpu_count", action["approval_reasons"])
        self.assertIn("high_gpu_count", result["execution_plan"]["approval_reasons"])

    def test_live_execution_requires_project_and_workspace(self):
        result = execute_model_decision_payload(
            forward_decision(),
            candidate_actions=candidate_actions(),
            dry_run=False,
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["execution_mode"], "missing_execution_context")
        self.assertEqual(result["issues"][0]["code"], "missing_execution_context")

    def test_high_gpu_live_execution_is_blocked_by_approval(self):
        decision = forward_decision()
        decision["selected_actions"][0]["parameters"]["compute_num_gpus"] = 8

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
            dry_run=False,
            project_uid="P1",
            workspace_uid="W1",
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["execution_mode"], "live_execution")
        self.assertEqual(
            result["execution_results"][0]["status"],
            "approval_required",
        )

    def test_unknown_action_id_becomes_generic_action(self):
        decision = forward_decision()
        decision["selected_actions"][0]["action_id"] = "forward_J999"

        result = validate_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["warnings"][0]["code"],
            "generic_action_without_candidate",
        )

        execute_result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )
        self.assertTrue(execute_result["success"])
        self.assertEqual(
            execute_result["execution_plan"]["actions"][0]["execution_mode"],
            "create_job",
        )

    def test_stop_decision_is_valid_and_planned(self):
        decision = forward_decision(
            decision_type="stop",
            selected_actions=[],
            reason="No safe next action is available.",
            confidence=0.8,
        )

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_plan"]["decision_type"], "stop")
        self.assertFalse(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_type"],
            "stop",
        )

    def test_rollback_decision_is_valid_and_planned(self):
        rollback_target = {
            "workflow_node_id": "J6",
            "job_type": "template_picker_gpu",
            "reason_code": "poor_particle_picking",
            "rollback_mode": "rerun_from_target",
        }
        decision = forward_decision(
            decision_type="rollback",
            selected_actions=[],
            rollback_target=rollback_target,
            reason="Picking should be adjusted before extraction.",
            confidence=0.83,
            risk_flags=["particle_quality_uncertain"],
        )

        result = execute_model_decision_payload(
            decision,
            candidate_actions=candidate_actions(),
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["execution_plan"]["approval_required"])
        self.assertEqual(
            result["execution_plan"]["approval_reasons"],
            ["rollback_decision"],
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["action_type"],
            "rollback",
        )
        self.assertEqual(
            result["execution_plan"]["actions"][0]["rollback_target"],
            rollback_target,
        )

    def test_select_2d_completed_generates_homo_refine_candidate(self):
        candidates, blocked = generate_candidate_actions(
            select_2d_workflow_state(),
            "J37",
        )

        self.assertFalse(blocked)
        self.assertEqual(candidates[0]["job_type"], "homo_refine_new")
        self.assertEqual(
            candidates[0]["required_inputs"]["particles"][0]["source_output"],
            "particles_selected",
        )
        self.assertEqual(
            candidates[0]["required_inputs"]["volume"][0]["source_output"],
            "imported_volume_1",
        )

    def test_homo_refine_compute_use_ssd_is_allowed(self):
        candidates, _ = generate_candidate_actions(
            select_2d_workflow_state(),
            "J37",
        )

        result = execute_model_decision_payload(
            homo_refine_decision(),
            candidate_actions=candidates,
        )

        self.assertTrue(result["success"])
        action = result["execution_plan"]["actions"][0]
        self.assertEqual(action["job_type"], "homo_refine_new")
        self.assertFalse(action["resolved_parameters"]["compute_use_ssd"])
        self.assertEqual(
            action["connections"]["particles"],
            ("J37", "particles_selected"),
        )
        self.assertEqual(
            action["connections"]["volume"],
            ("J2", "imported_volume_1"),
        )

    def test_homo_refine_completed_allows_parallel_homo_refine_candidate(self):
        candidates, blocked = generate_candidate_actions(
            homo_refine_workflow_state(),
            "J39",
        )

        self.assertFalse(blocked)
        self.assertEqual(candidates[0]["action_id"], "forward_J39_homo_refine_new")
        self.assertEqual(candidates[0]["job_type"], "homo_refine_new")
        self.assertEqual(
            candidates[0]["required_inputs"]["particles"][0]["source_job_uid"],
            "J37",
        )
        self.assertEqual(
            candidates[0]["required_inputs"]["volume"][0]["source_job_uid"],
            "J2",
        )

    def test_parallel_homo_refine_decision_is_planned(self):
        candidates, _ = generate_candidate_actions(
            homo_refine_workflow_state(),
            "J39",
        )

        result = execute_model_decision_payload(
            parallel_homo_refine_decision(),
            candidate_actions=candidates,
        )

        self.assertTrue(result["success"])
        action = result["execution_plan"]["actions"][0]
        self.assertEqual(action["job_type"], "homo_refine_new")
        self.assertEqual(
            action["connections"]["particles"],
            ("J37", "particles_selected"),
        )
        self.assertEqual(
            action["connections"]["volume"],
            ("J2", "imported_volume_1"),
        )

    def test_generic_action_can_use_model_supplied_connections(self):
        decision = forward_decision()
        decision["selected_actions"][0] = {
            "job_type": "unknown_future_job",
            "parameters": {"custom_param": 3},
            "connections": {
                "particles": {
                    "source_job_uid": "J37",
                    "source_output": "particles_selected",
                }
            },
        }

        result = execute_model_decision_payload(
            decision,
            candidate_actions=[],
        )

        self.assertTrue(result["success"])
        action = result["execution_plan"]["actions"][0]
        self.assertEqual(action["job_type"], "unknown_future_job")
        self.assertEqual(action["connections"]["particles"], ("J37", "particles_selected"))
        self.assertEqual(action["resolved_parameters"]["custom_param"], 3)


if __name__ == "__main__":
    unittest.main()
