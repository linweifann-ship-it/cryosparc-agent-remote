import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from missing_parameter_recovery import (
    candidate_actions_for_model_recovery,
    current_node_from_model_context,
    inject_model_parameter_recovery_guidance,
)


def action(job_type: str, parameter: str, spec: dict) -> dict:
    return {
        "action_id": f"registry_{job_type}",
        "job_type": job_type,
        "available": True,
        "blocked_by": [],
        "default_parameters": {},
        "parameter_template": {parameter: {"required": True, **spec}},
    }


def load_active_runner():
    mcp = types.ModuleType("mcp")
    mcp.ClientSession = object
    mcp.StdioServerParameters = object
    sys.modules.setdefault("mcp", mcp)
    sys.modules.setdefault("mcp.client", types.ModuleType("mcp.client"))
    stdio = types.ModuleType("mcp.client.stdio")
    stdio.stdio_client = object
    sys.modules.setdefault("mcp.client.stdio", stdio)
    path = Path(__file__).resolve().parents[1] / "scripts" / "autonomous_mcp_closed_loop.py"
    spec = importlib.util.spec_from_file_location("active_missing_parameter_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MissingParameterModelContextTests(unittest.TestCase):
    def test_first_model_context_includes_blob_picker_guidance_in_failure_context(self):
        model_input = {"failure_context": None}
        candidates = {"candidate_actions": [action("blob_picker_gpu", "diameter", {"type": "number"})]}
        feedback = inject_model_parameter_recovery_guidance(
            model_input, candidates, {"pixel_size_A": 0.6575}, {}
        )

        self.assertEqual(feedback["missing_required_parameters"][0]["parameter"], "diameter")
        guidance = model_input["failure_context"]["parameter_recovery_guidance"]
        self.assertEqual(guidance["parameters"][0]["job_type"], "blob_picker_gpu")
        self.assertTrue(guidance["policy"]["heuristic_estimates_allowed"])
        self.assertIn("do not invent observed facts", guidance["model_instruction"])

        runner = load_active_runner()
        prompt = runner.build_autonomous_prompt(model_input, candidates, round_index=1)
        first_model_payload = json.loads(prompt[1]["content"])
        self.assertEqual(
            first_model_payload["model_input"]["failure_context"]
            ["parameter_recovery_guidance"]["parameters"][0]["parameter"],
            "diameter",
        )

    def test_other_numeric_scientific_parameter_uses_the_same_first_call_guidance(self):
        model_input = {"failure_context": None}
        candidates = {"candidate_actions": [action("extract_micrographs_multi", "box_size_pix", {"type": "integer"})]}
        inject_model_parameter_recovery_guidance(model_input, candidates, {}, {})
        parameters = model_input["failure_context"]["parameter_recovery_guidance"]["parameters"]
        self.assertEqual(parameters[0]["parameter"], "box_size_pix")

    def test_first_call_guidance_requires_registry_units_and_default_preservation(self):
        model_input = {"failure_context": None}
        candidates = {"candidate_actions": [action("blob_picker_gpu", "diameter", {
            "type": "number", "title": "Minimum particle diameter (A)",
            "description": "Min Particle diameter (A)", "unit": "A",
            "unit_source": "registry_ui_contract",
        })]}
        inject_model_parameter_recovery_guidance(
            model_input, candidates, {"pixel_size_A": 0.6575}, {}
        )
        guidance = model_input["parameter_recovery_guidance"]
        self.assertEqual(guidance["parameters"][0]["unit"], "A")
        self.assertTrue(guidance["policy"]["registry_parameter_units_are_authoritative"])
        self.assertTrue(guidance["policy"]["pixel_size_is_not_particle_size_evidence"])
        self.assertTrue(guidance["policy"]["preserve_optional_registry_defaults_without_evidence"])
        self.assertIn("pixel size alone", guidance["model_instruction"])

        runner = load_active_runner()
        prompt = runner.build_autonomous_prompt(model_input, candidates, round_index=1)
        instruction = json.loads(prompt[1]["content"])["instruction"]
        self.assertIn("preserve optional Registry defaults", instruction)

    def test_model_context_candidates_win_over_restart_fallback_candidates(self):
        blob = action("blob_picker_gpu", "diameter", {"type": "number"})
        fallback = action("import_micrographs", "blob_paths", {"type": "string"})
        model_input = {"candidate_actions": [blob], "failure_context": None}
        fallback_context = {"candidate_actions": [fallback]}
        selected = candidate_actions_for_model_recovery(model_input, fallback_context)
        feedback = inject_model_parameter_recovery_guidance(
            model_input, {"candidate_actions": selected}, {}, {}
        )
        self.assertEqual(feedback["missing_required_parameters"][0]["job_type"], "blob_picker_gpu")
        self.assertEqual(
            current_node_from_model_context({"candidate_context": {"current_node_id": "J12"}}),
            "J12",
        )

    def test_non_scientific_string_requirement_is_not_presented_as_a_heuristic(self):
        model_input = {"failure_context": None}
        candidates = {"candidate_actions": [action("import_particles", "particle_meta_path", {"type": "string"})]}
        feedback = inject_model_parameter_recovery_guidance(model_input, candidates, {}, {})
        self.assertIsNone(feedback)
        self.assertIsNone(model_input["failure_context"])


if __name__ == "__main__":
    unittest.main()
