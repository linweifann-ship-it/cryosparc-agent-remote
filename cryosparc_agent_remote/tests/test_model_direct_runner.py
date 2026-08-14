# Tests direct-model JSON parsing helpers without loading the model.
import unittest

from model_direct_runner import (
    build_workflow_decision_prompt,
    normalize_optional_path,
    parse_model_decision_text,
)


class ModelDirectRunnerTests(unittest.TestCase):
    def test_parse_plain_json(self):
        result = parse_model_decision_text(
            '{"schema_version":"3.0","decision_type":"stop","selected_actions":[]}'
        )

        self.assertEqual(result["decision_type"], "stop")

    def test_parse_markdown_wrapped_json(self):
        result = parse_model_decision_text(
            '```json\n{"schema_version":"3.0","decision_type":"forward","selected_actions":[]}\n```'
        )

        self.assertEqual(result["decision_type"], "forward")

    def test_parse_output_after_thinking_text(self):
        result = parse_model_decision_text(
            "/think hidden notes\n"
            '{"schema_version":"3.0","decision_type":"stop","selected_actions":[]}'
        )

        self.assertEqual(result["selected_actions"], [])

    def test_prompt_contains_v2_context_and_contract(self):
        messages = build_workflow_decision_prompt(
            {
                "schema_version": "2.1",
                "task_type": "workflow_decision",
                "dataset_context": {},
                "current_state": {"last_node_id": "J8"},
            }
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("output_contract", messages[1]["content"])
        self.assertIn("J8", messages[1]["content"])

    def test_empty_adapter_path_is_normalized(self):
        self.assertIsNone(normalize_optional_path(""))
        self.assertIsNone(normalize_optional_path("none"))
        self.assertEqual(normalize_optional_path("/tmp/adapter"), "/tmp/adapter")


if __name__ == "__main__":
    unittest.main()
