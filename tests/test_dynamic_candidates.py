import unittest
from types import SimpleNamespace

from dynamic_candidates import (
    build_registry_candidate,
    registry_parameter_template,
    resolve_registry_connections,
)


class DynamicCandidateTests(unittest.TestCase):
    def test_registry_parameter_types_and_constraints_are_preserved(self):
        spec = SimpleNamespace(
            type="blob_picker_gpu",
            title="Blob Picker",
            tags=["gpuEnabled", "multiGpu"],
            interactive=False,
            params={
                "diameter": SimpleNamespace(type="number", anyOf=[], required_param=True, default=None, hidden=False, enum=None, ge=10, le=500),
                "use_circle": SimpleNamespace(type="boolean", anyOf=[], required_param=False, default=1, hidden=False, enum=None, ge=None, le=None),
                "diameter_max": SimpleNamespace(type=None, anyOf=[SimpleNamespace(type="number"), SimpleNamespace(type="null")], required_param=False, default=None, hidden=False, enum=None, ge=None, le=None),
            },
        )
        template = registry_parameter_template(spec)
        self.assertEqual(template["diameter"]["type"], "number")
        self.assertTrue(template["diameter"]["required"])
        self.assertEqual(template["diameter"]["minimum"], 10)
        self.assertEqual(template["diameter_max"]["type"], "number")

    def test_optional_registry_slots_do_not_block_particle_input(self):
        spec = SimpleNamespace(
            inputs=SimpleNamespace(root={
                "particles": SimpleNamespace(
                    type="particle",
                    slots=["blob", "?ctf", "?location", "?pick_stats"],
                    count_min=1,
                ),
            }),
        )
        sources = [{
            "source_job_uid": "J6",
            "source_output": "particles",
            "result_names": ["blob", "location", "ctf", "pick_stats"],
        }]
        connections = resolve_registry_connections(spec, sources)
        self.assertEqual(connections["particles"][0]["source_job_uid"], "J6")

    def test_primary_particles_are_preferred_over_unused_branch(self):
        spec = SimpleNamespace(
            type="homo_refine_new",
            inputs=SimpleNamespace(root={
                "particles": SimpleNamespace(
                    type="particle", slots=["blob"], count_min=1,
                ),
            }),
        )
        sources = [
            {"source_job_uid": "J9", "source_output": "particles_class_0", "result_names": ["blob"]},
            {"source_job_uid": "J9", "source_output": "particles_unused", "result_names": ["blob"]},
        ]
        connections = resolve_registry_connections(spec, sources)
        self.assertEqual(connections["particles"][0]["source_output"], "particles_class_0")

    def test_recovery_scope_excludes_future_job_outputs(self):
        from dynamic_candidates import available_output_sources
        state = {"nodes": [
            {"status": "completed", "cryosparc_job_uid": "J9", "workflow_node_id": "J9", "logical_node_id": "n9",
             "job_type": "homo_abinit", "outputs": {"particles_class_0": {"available": True, "result_names": ["blob"]}}},
            {"status": "completed", "cryosparc_job_uid": "J14", "workflow_node_id": "J14", "logical_node_id": "n14",
             "job_type": "homo_refine_new", "outputs": {"particles": {"available": True, "result_names": ["blob"]}}},
        ]}
        sources = available_output_sources(state, max_job_uid="J9")
        self.assertEqual([source["source_job_uid"] for source in sources], ["J9"])

    def test_excluded_particle_output_is_fallback_when_only_match(self):
        spec = SimpleNamespace(
            type="homo_refine_new",
            inputs=SimpleNamespace(root={
                "particles": SimpleNamespace(
                    type="particle", slots=["blob"], count_min=1,
                ),
            }),
        )
        sources = [{"source_job_uid": "J9", "source_output": "particles_rejected", "result_names": ["blob"]}]
        connections = resolve_registry_connections(spec, sources)
        self.assertEqual(connections["particles"][0]["source_output"], "particles_rejected")

    def test_registry_gpu_metadata_is_executable(self):
        spec = SimpleNamespace(
            type="patch_ctf_estimation_multi", title="Patch CTF", tags=["gpuEnabled", "multiGpu"],
            interactive=False, category="ctf_estimation", params={},
        )
        candidate = build_registry_candidate(
            {"cryosparc_job_uid": "J6", "workflow_node_id": "J6"}, spec, {"exposures": []}
        )
        metadata = candidate["job_spec_metadata"]
        self.assertTrue(metadata["requires_gpu"])
        self.assertTrue(metadata["multi_gpu"])
        self.assertEqual(metadata["category"], "ctf_estimation")
        self.assertIn("default_lane", metadata)


if __name__ == "__main__":
    unittest.main()
