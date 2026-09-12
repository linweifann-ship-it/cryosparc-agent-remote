import unittest

from action_registry import apply_auto_select_2d_policy


class AutoSelect2DTests(unittest.TestCase):
    def test_selects_ceil_of_class_fraction_as_string(self):
        state = {"nodes": [{
            "workflow_node_id": "J20", "cryosparc_job_uid": "J20",
            "outputs": {"class_averages": {"num_items": 50}},
        }]}
        action = {
            "job_type": "select_2D", "execution_mode": "dry_run_only",
            "parameter_template": {"selected_templates": {"type": "string"}},
            "required_inputs": {"templates": [{"source_job_uid": "J20", "source_output": "class_averages"}]},
        }
        result = apply_auto_select_2d_policy(action, state)
        self.assertEqual(result["default_parameters"]["selected_templates"], ",".join(str(i) for i in range(43)))
        self.assertFalse(result["job_spec_metadata"]["interactive"])
        self.assertEqual(result["auto_policy"]["selected_class_count"], 43)

    def test_ranks_classes_by_resolution_for_particle_target(self):
        from unittest.mock import patch
        from action_registry import build_quality_ranked_class_plan

        class FakeJob:
            def __init__(self, data):
                self.data = data

            def load_output(self, _):
                return self.data

        class FakeProject:
            def find_job(self, uid):
                if uid == "J20":
                    return FakeJob({"blob/res_A": [10.0, 5.0, 8.0]})
                return FakeJob({"alignments2D/class": [0, 0, 0, 1, 1, 2]})

        action = {"required_inputs": {
            "templates": [{"source_job_uid": "J20", "source_output": "class_averages"}],
            "particles": [{"source_job_uid": "J21", "source_output": "particles"}],
        }}
        with patch("action_registry.cryosparc_client") as factory:
            factory.return_value.find_project.return_value = FakeProject()
            plan = build_quality_ranked_class_plan(
                action, {"project_uid": "P", "nodes": []}, 0.85
            )
        self.assertEqual(plan["selected_class_indices"], [1, 2, 0])
        self.assertEqual(plan["selected_particle_count"], 6)


if __name__ == "__main__":
    unittest.main()
