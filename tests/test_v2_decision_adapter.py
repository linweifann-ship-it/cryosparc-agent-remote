# Fixed tests for adapting V2 model decisions into internal execution plans.
import unittest
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
        self.assertEqual(
            result["internal_decision"]["selected_actions"][0]["parameters"]["psize_A"],
            0.6575,
        )

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
            "parameter_template": {
                "refine_do_marg": {"type": "boolean", "default": False},
                "compute_use_ssd": {"type": "boolean", "default": False},
                "refine_symmetry": {"type": "string", "default": "D7"},
            },
            "default_parameters": {
                "refine_do_marg": 0,
                "compute_use_ssd": 1,
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
        self.assertIs(params["refine_do_marg"], False)
        self.assertIs(params["compute_use_ssd"], True)

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


if __name__ == "__main__":
    unittest.main()
