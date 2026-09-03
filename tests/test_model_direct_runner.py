# Tests direct-model JSON parsing helpers without loading the model.
import io
import json
import unittest
from unittest import mock
from urllib.error import HTTPError

from model_direct_runner import (
    build_workflow_decision_prompt,
    extract_usage_summary,
    normalize_optional_path,
    parse_model_decision_text,
    run_openai_compatible_model,
)


class ModelDirectRunnerTests(unittest.TestCase):
    def test_parse_plain_json(self):
        result = parse_model_decision_text(
            '{"schema_version":"2.0","decision_type":"stop","reason":"done"}'
        )

        self.assertEqual(result["decision_type"], "stop")

    def test_parse_markdown_wrapped_json(self):
        result = parse_model_decision_text(
            '```json\n{"schema_version":"2.0","decision_type":"forward"}\n```'
        )

        self.assertEqual(result["decision_type"], "forward")

    def test_parse_output_after_thinking_text(self):
        result = parse_model_decision_text(
            "/think hidden notes\n"
            '{"schema_version":"2.0","decision_type":"stop","reason":"review"}'
        )

        self.assertEqual(result["reason"], "review")

    def test_prompt_contains_v2_context_and_contract(self):
        messages = build_workflow_decision_prompt(
            {
                "schema_version": "2.0",
                "task_type": "workflow_decision",
                "dataset_info": {},
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

    def test_api_retries_transient_errors(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        errors = [
            HTTPError("https://example.test", 500, "server", {}, io.BytesIO(b"{}")),
            HTTPError("https://example.test", 429, "busy", {}, io.BytesIO(b"{}")),
            response,
        ]
        with mock.patch("model_direct_runner.request.urlopen", side_effect=errors) as urlopen:
            with mock.patch("model_direct_runner.time.sleep") as sleep:
                result = run_openai_compatible_model(
                    [{"role": "user", "content": "test"}],
                    "https://example.test/v1",
                    "key",
                    "model",
                    max_retries=2,
                    retry_backoff_seconds=2,
                )
        self.assertEqual(result["raw_text"], "OK")
        self.assertEqual(result["usage"]["present"], False)
        self.assertIn('"model":"model"', result["exact_serialized_request"])
        self.assertGreater(result["request_chars"], 0)
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_extract_usage_summary_reads_chat_cached_tokens(self):
        result = extract_usage_summary(
            {
                "usage": {
                    "prompt_tokens": 2048,
                    "completion_tokens": 10,
                    "total_tokens": 2058,
                    "prompt_tokens_details": {"cached_tokens": 1024},
                }
            }
        )

        self.assertTrue(result["present"])
        self.assertEqual(result["cached_tokens"], 1024)
        self.assertEqual(
            result["cache_field_path"],
            "usage.prompt_tokens_details.cached_tokens",
        )

    def test_api_payload_includes_prompt_cache_options_when_provided(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'

        with mock.patch("model_direct_runner.request.urlopen", return_value=response) as urlopen:
            result = run_openai_compatible_model(
                [{"role": "user", "content": "test"}],
                "https://example.test/v1",
                "key",
                "model",
                prompt_cache_key="cryoagent:P2:W9:workflow-v2",
                prompt_cache_options={"mode": "explicit", "ttl": "30m"},
            )

        request = urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(result["raw_text"], "OK")
        self.assertEqual(body["prompt_cache_key"], "cryoagent:P2:W9:workflow-v2")
        self.assertEqual(
            body["prompt_cache_options"],
            {"mode": "explicit", "ttl": "30m"},
        )


if __name__ == "__main__":
    unittest.main()
