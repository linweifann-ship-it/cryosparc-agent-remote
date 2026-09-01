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


if __name__ == "__main__":
    unittest.main()
