# Fixed tests for adapting V2 model decisions into internal execution plans.
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from v2_decision_adapter import (
    adapt_v2_decision_to_internal,
    execute_v2_model_decision_payload,
)
from action_registry import validate_model_decision_payload, validate_parameters


def candidate_actions():
    """Return one internal candidate produced after J8 completes."""
    return [
        {
            "action_id": "forward_J9",
            "action_type": "forward",
            "workflow_node_id": "J9",
            "reference_job_uid": "J9",
            "reference_status": "completed",
            "job_type": "class_2D_new",
            "description": "Run 2D classification.",
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
                "class2D_K": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 50,
                },
            },
            "default_parameters": {
                "compute_num_gpus": 4,
                "class2D_K": 50,
            },
        }
    ]


def duplicate_class2d_candidates():
    """Return duplicate same-type candidates produced by repeated live tests."""
    actions = []
    for action_id, job_uid, status in [
        ("branch_J31", "J31", "running"),
        ("branch_J33", "J33", "completed"),
        ("branch_J9", "J9", "completed"),
    ]:
        action = candidate_actions()[0].copy()
        action.update(
            {
                "action_id": action_id,
                "action_type": "branch",
                "workflow_node_id": job_uid,
                "reference_job_uid": job_uid,
                "reference_status": status,
            }
        )
        actions.append(action)
    return actions


def candidate_context():
    """Return the internal candidate registry context."""
    return {
        "project_uid": "P2",
        "workspace_uid": "W3",
        "current_node_id": "J8",
        "candidate_actions": candidate_actions(),
        "blocked_actions": [],
        "decision_hint": None,
    }


def v2_forward_decision():
    """Return a compact V2 decision that does not expose action_id."""
    return {
        "schema_version": "2.0",
        "decision_type": "forward",
        "action": "class_2D_new",
        "parameters": {
            "compute_num_gpus": 4,
            "class2D_K": 50,
        },
        "reason": "Extracted particles are ready for 2D classification.",
        "confidence": 0.9,
        "risk_flags": [],
        "evidence": ["J8 completed with particles output."],
    }


class V2DecisionAdapterTests(unittest.TestCase):
    def test_v2_forward_maps_to_internal_candidate(self):
        result = adapt_v2_decision_to_internal(
            v2_forward_decision(),
            candidate_actions(),
        )

        self.assertTrue(result["success"])
        action = result["internal_decision"]["selected_actions"][0]
        self.assertEqual(action["action_id"], "forward_J9")
        self.assertEqual(action["job_type"], "class_2D_new")

    def test_unknown_v2_action_becomes_generic_plan(self):
        decision = v2_forward_decision()
        decision["action"] = "unknown_future_job"
        decision["job_type"] = "unknown_future_job"
        decision["connections"] = {
            "particles": {
                "source_job_uid": "J37",
                "source_output": "particles_selected",
            }
        }

        result = adapt_v2_decision_to_internal(decision, candidate_actions())

        self.assertTrue(result["success"])
        action = result["internal_decision"]["selected_actions"][0]
        self.assertEqual(action["action_id"], "generic_0_unknown_future_job")
        self.assertEqual(action["job_type"], "unknown_future_job")
        self.assertEqual(
            action["connections"]["particles"]["source_output"],
            "particles_selected",
        )

    def test_legacy_actions_shape_keeps_candidate_validation_path(self):
        decision = {
            "schema_version": "2.0",
            "decision_type": "branch",
            "actions": [{
                "action": "class_2D_new",
                "parameters": {"class2D_K": 64},
            }],
        }

        result = adapt_v2_decision_to_internal(decision, candidate_actions())

        self.assertTrue(result["success"])
        action = result["internal_decision"]["selected_actions"][0]
        self.assertEqual(action["action_id"], "forward_J9")
        self.assertEqual(action["parameters"]["class2D_K"], 64)

    def test_duplicate_same_type_candidates_prefer_original_completed_job(self):
        result = adapt_v2_decision_to_internal(
            v2_forward_decision(),
            duplicate_class2d_candidates(),
        )

        self.assertTrue(result["success"])
        action = result["internal_decision"]["selected_actions"][0]
        self.assertEqual(action["action_id"], "branch_J9")
        self.assertEqual(action["workflow_node_id"], "J9")
        self.assertEqual(result["internal_decision"]["decision_type"], "branch")
        self.assertEqual(
            result["internal_decision"]["branch_plan"]["max_parallel_branches"],
            1,
        )

    def test_v2_execute_dry_run_reuses_internal_executor(self):
        with patch(
            "v2_decision_adapter.get_candidate_actions",
            return_value=candidate_context(),
        ):
            result = execute_v2_model_decision_payload(
                v2_forward_decision(),
                project_uid="P2",
                workspace_uid="W3",
                current_node_id="J8",
                dry_run=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["execution_mode"], "v2_adapter")
        self.assertEqual(
            result["internal_decision"]["selected_actions"][0]["action_id"],
            "forward_J9",
        )
        self.assertEqual(
            result["execution_result"]["execution_plan"]["actions"][0]["job_type"],
            "class_2D_new",
        )


    def test_initial_import_inherits_dataset_parameters(self):
        from v2_decision_adapter import adapt_v2_decision_to_internal

        candidates = [{
            "action_id": "initial_import_micrographs",
            "action_type": "forward",
            "workflow_node_id": "initial:import_micrographs",
            "job_type": "import_micrographs",
            "default_parameters": {
                "blob_paths": "/data/*.mrc",
                "psize_A": 0.6575,
                "accel_kv": 300,
                "cs_mm": 2.7,
                "total_dose_e_per_A2": 53,
            },
        }]
        decision = {"decision_type": "forward", "action": "import_micrographs", "parameters": {}}
        result = adapt_v2_decision_to_internal(decision, candidates)
        self.assertTrue(result["success"])
        self.assertEqual(result["internal_decision"]["selected_actions"][0]["parameters"], {})

    def test_empty_workspace_initial_import_binds_candidate_and_keeps_defaults(self):
        defaults = {
            "blob_paths": "/data/initial/*.mrc",
            "psize_A": 0.6575,
            "accel_kv": 300,
            "cs_mm": 2.7,
            "total_dose_e_per_A2": 53,
        }
        candidates = [{
            "action_id": "initial_import_micrographs",
            "action_type": "forward",
            "workflow_node_id": "initial:import_micrographs",
            "job_type": "import_micrographs",
            "execution_mode": "create_job",
            "available": True,
            "parameter_template": {
                "blob_paths": {"type": "string", "required": True},
                "psize_A": {"type": "number", "minimum": 0},
                "accel_kv": {"type": "number", "minimum": 0},
                "cs_mm": {"type": "number", "minimum": 0},
                "total_dose_e_per_A2": {"type": "number", "minimum": 0},
            },
            "default_parameters": defaults,
        }]
        decision = {
            "decision_type": "forward",
            "action": "import_micrographs",
            "workflow_node_id": "generic:import_micrographs",
            "parameters": {},
        }

        adapter_result = adapt_v2_decision_to_internal(decision, candidates)
        self.assertTrue(adapter_result["success"])
        action = adapter_result["internal_decision"]["selected_actions"][0]
        self.assertEqual(action["action_id"], "initial_import_micrographs")
        self.assertEqual(action["workflow_node_id"], "initial:import_micrographs")
        self.assertEqual(action["parameters"], {})

        validation = validate_model_decision_payload(
            adapter_result["internal_decision"],
            candidate_actions=candidates,
        )
        self.assertTrue(validation["success"])
        self.assertEqual(
            validation["resolved_actions"][0]["resolved_parameters"],
            defaults,
        )

    def test_registry_boolean_defaults_are_normalized(self):
        candidates = [{
            "action_id": "forward_J19",
            "action_type": "forward",
            "workflow_node_id": "J19",
            "job_type": "homo_refine_new",
            "execution_mode": "create_job",
            "parameter_template": {
                "refine_do_marg": {"type": "boolean", "default": False},
                "compute_use_ssd": {"type": "boolean", "default": False},
                "refine_symmetry": {"type": "string", "default": "D7"},
            },
            "default_parameters": {
                "refine_do_marg": False,
                "compute_use_ssd": True,
                "refine_symmetry": "D7",
            },
        }]
        decision = {
            "decision_type": "forward",
            "action": "homo_refine_new",
            "parameters": {},
        }
        result = adapt_v2_decision_to_internal(decision, candidates)
        self.assertTrue(result["success"])
        params = result["internal_decision"]["selected_actions"][0]["parameters"]
        self.assertEqual(params, {})
        validation = validate_model_decision_payload(
            result["internal_decision"], candidate_actions=candidates
        )
        self.assertTrue(validation["success"])
        resolved = validation["resolved_actions"][0]["resolved_parameters"]
        self.assertIs(resolved["refine_do_marg"], False)
        self.assertIs(resolved["compute_use_ssd"], True)

    def test_unknown_parameter_is_not_resolved_for_execution(self):
        candidates = candidate_actions()
        decision = v2_forward_decision()
        decision["parameters"] = {"class2D_K": 64, "do_plots": 1}
        adapted = adapt_v2_decision_to_internal(decision, candidates)
        validation = validate_model_decision_payload(
            adapted["internal_decision"], candidate_actions=candidates
        )
        self.assertTrue(validation["success"])
        self.assertTrue(any(issue["code"] == "unknown_parameter" for issue in validation["warnings"]))
        self.assertNotIn("do_plots", validation["resolved_actions"][0]["resolved_parameters"])

    def test_model_override_wins_over_materialized_default(self):
        candidates = candidate_actions()
        decision = v2_forward_decision()
        decision["parameters"] = {"class2D_K": 64}
        adapted = adapt_v2_decision_to_internal(decision, candidates)
        validation = validate_model_decision_payload(
            adapted["internal_decision"], candidate_actions=candidates
        )
        self.assertTrue(validation["success"])
        self.assertEqual(validation["resolved_actions"][0]["resolved_parameters"]["class2D_K"], 64)

    def test_explicit_integer_boolean_remains_invalid(self):
        candidates = [{
            "action_id": "forward_J19",
            "action_type": "forward",
            "workflow_node_id": "J19",
            "job_type": "homo_refine_new",
            "parameter_template": {"refine_do_marg": {"type": "boolean"}},
            "default_parameters": {},
        }]
        decision = {
            "decision_type": "forward",
            "action": "homo_refine_new",
            "parameters": {"refine_do_marg": 1},
        }
        result = adapt_v2_decision_to_internal(decision, candidates)
        self.assertTrue(result["success"])
        params = result["internal_decision"]["selected_actions"][0]["parameters"]
        self.assertEqual(params["refine_do_marg"], 1)
        self.assertIs(type(params["refine_do_marg"]), int)
        issues, _ = validate_parameters(
            params,
            candidates[0]["parameter_template"],
            path="parameters",
        )
        self.assertTrue(any(item.code == "parameter_type_mismatch" for item in issues))

    def test_registry_patch_ctf_empty_model_parameters_follow_runtime_validation_path(self):
        """Exercise candidate -> V2 adapter -> validator -> resolved parameters."""
        candidates = [{
            "action_id": "registry_J312_patch_ctf_estimation_multi",
            "action_type": "forward",
            "workflow_node_id": "J312:patch_ctf_estimation_multi",
            "reference_job_uid": "J312",
            "job_type": "patch_ctf_estimation_multi",
            "execution_mode": "create_job",
            "available": True,
            "parameter_template": {
                "classic_mode": {"type": "boolean", "default": False},
                "do_phase_shift_refine_only": {"type": "boolean", "default": False},
                "compute_num_gpus": {"type": "integer", "default": 1},
            },
            "default_parameters": {
                "classic_mode": False,
                "do_phase_shift_refine_only": False,
                "compute_num_gpus": 1,
            },
        }]
        decision = {
            "schema_version": "2.0",
            "decision_type": "forward",
            "action": "patch_ctf_estimation_multi",
            "parameters": {},
        }
        adapted = adapt_v2_decision_to_internal(decision, candidates)
        self.assertTrue(adapted["success"])
        self.assertEqual(adapted["internal_decision"]["selected_actions"][0]["parameters"], {})
        validation = validate_model_decision_payload(
            adapted["internal_decision"], candidate_actions=candidates
        )
        self.assertTrue(validation["success"])
        resolved = validation["resolved_actions"][0]["resolved_parameters"]
        self.assertIs(resolved["classic_mode"], False)
        self.assertIs(resolved["do_phase_shift_refine_only"], False)
        self.assertNotIn("do_plots", resolved)

    def test_live_registry_style_patch_ctf_candidate_runs_full_adapter_path(self):
        from dynamic_candidates import build_registry_candidate

        spec = SimpleNamespace(
            type="patch_ctf_estimation_multi",
            title="Patch CTF",
            category="ctf_estimation",
            tags=["gpuEnabled"],
            interactive=False,
            params={
                "classic_mode": SimpleNamespace(type="boolean", anyOf=[], required_param=False, default=0, hidden=False, enum=None, ge=None, le=None),
                "do_phase_shift_refine_only": SimpleNamespace(type="boolean", anyOf=[], required_param=False, default=0, hidden=False, enum=None, ge=None, le=None),
                "do_plots": SimpleNamespace(type="integer", anyOf=[], required_param=False, default=1, hidden=True, enum=None, ge=None, le=None),
            },
        )
        candidate = build_registry_candidate(
            {"cryosparc_job_uid": "J312", "workflow_node_id": "J312"},
            spec,
            {"exposures": [{"source_job_uid": "J312", "source_output": "imported_micrographs"}]},
        )
        adapted = adapt_v2_decision_to_internal(
            {"decision_type": "forward", "action": "patch_ctf_estimation_multi", "parameters": {}},
            [candidate],
        )
        validation = validate_model_decision_payload(
            adapted["internal_decision"], candidate_actions=[candidate]
        )
        self.assertTrue(validation["success"])
        resolved = validation["resolved_actions"][0]["resolved_parameters"]
        self.assertIs(resolved["classic_mode"], False)
        self.assertIs(resolved["do_phase_shift_refine_only"], False)
        self.assertNotIn("do_plots", candidate["default_parameters"])
        self.assertNotIn("do_plots", resolved)

    def test_inspect_picks_empty_model_parameters_validate_with_boolean_defaults(self):
        candidates = [{
            "action_id": "registry_J314_inspect_picks_v2",
            "action_type": "forward",
            "workflow_node_id": "J314:inspect_picks_v2",
            "reference_job_uid": "J314",
            "job_type": "inspect_picks_v2",
            "execution_mode": "create_job",
            "available": True,
            "parameter_template": {
                "calibrate_ncc": {"type": "boolean", "default": True},
                "calibrate_pow": {"type": "boolean", "default": True},
                "do_auto_cluster": {"type": "boolean", "default": False},
            },
            "default_parameters": {
                "calibrate_ncc": True,
                "calibrate_pow": True,
                "do_auto_cluster": False,
            },
        }]
        adapted = adapt_v2_decision_to_internal(
            {"decision_type": "forward", "action": "inspect_picks_v2", "parameters": {}},
            candidates,
        )
        validation = validate_model_decision_payload(
            adapted["internal_decision"], candidate_actions=candidates
        )
        self.assertTrue(validation["success"])
        resolved = validation["resolved_actions"][0]["resolved_parameters"]
        self.assertIs(resolved["calibrate_ncc"], True)
        self.assertIs(resolved["calibrate_pow"], True)
        self.assertIs(resolved["do_auto_cluster"], False)
        self.assertNotIn("dilation_bins", resolved)
        self.assertNotIn("keep_threshold", resolved)

    def test_valid_request_input_is_terminal_control_flow_without_action(self):
        decision = {
            "schema_version": "2.0",
            "decision_type": "request_input",
            "requested_inputs": ["pick-inspection overlay or NCC/power score distribution"],
            "reason": "Visual evidence is unavailable, so safe thresholds cannot be chosen.",
            "confidence": 0.95,
            "risk_flags": ["missing_visual_evidence"],
            "evidence": ["The visual tool returned no image payload."],
        }
        adapted = adapt_v2_decision_to_internal(decision, candidate_actions())
        self.assertTrue(adapted["success"])
        internal = adapted["internal_decision"]
        self.assertEqual(internal["selected_actions"], [])
        validation = validate_model_decision_payload(
            internal, candidate_actions=candidate_actions()
        )
        self.assertTrue(validation["success"])
        self.assertEqual(validation["decision_type"], "request_input")
        from action_registry import execute_model_decision_payload
        execution = execute_model_decision_payload(
            internal,
            candidate_actions=candidate_actions(),
            dry_run=False,
            project_uid="P2",
            workspace_uid="W39",
        )
        self.assertTrue(execution["success"])
        self.assertEqual(execution["execution_mode"], "awaiting_input")
        self.assertEqual(execution["execution_results"], [])

    def test_stop_and_rollback_remain_non_action_control_flow(self):
        stop = adapt_v2_decision_to_internal(
            {"decision_type": "stop", "reason": "Done", "confidence": 0.8},
            candidate_actions(),
        )
        rollback = adapt_v2_decision_to_internal(
            {
                "decision_type": "rollback",
                "rollback_target": {
                    "workflow_node_id": "J8",
                    "job_type": "class_2D_new",
                    "reason_code": "quality_regression",
                },
                "reason": "Restore a prior node.",
                "confidence": 0.8,
            },
            candidate_actions(),
        )
        self.assertEqual(stop["internal_decision"]["selected_actions"], [])
        self.assertEqual(rollback["internal_decision"]["selected_actions"], [])
        self.assertTrue(validate_model_decision_payload(stop["internal_decision"])["success"])
        self.assertTrue(validate_model_decision_payload(rollback["internal_decision"])["success"])


if __name__ == "__main__":
    unittest.main()
