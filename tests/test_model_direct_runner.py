# Tests direct-model JSON parsing helpers without loading the model.
import json
import io
import unittest
from unittest import mock
from urllib.error import HTTPError

from model_direct_runner import (
    build_workflow_decision_prompt,
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

    def test_visual_context_adds_openai_multimodal_image_url(self):
        messages = build_workflow_decision_prompt(
            {"schema_version": "2.0", "candidate_actions": []},
            {
                "kind": "pick_inspection",
                "contact_sheet": {"data_url": "data:image/png;base64,TEST"},
            },
        )
        visual_message = messages[-1]
        self.assertIsInstance(visual_message["content"], list)
        self.assertEqual(visual_message["content"][1]["type"], "image_url")
        self.assertEqual(
            visual_message["content"][1]["image_url"]["url"],
            "data:image/png;base64,TEST",
        )

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
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_cache_parameter_rejection_retries_without_cache(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}],"usage":{"prompt_tokens":3}}'
        rejected = HTTPError("https://example.test", 500, "server error", {}, io.BytesIO(b'{"error":"prompt_cache_breakpoint is not supported on this model"}'))
        messages = [{"role": "system", "content": [{"type": "text", "text": "stable", "prompt_cache_breakpoint": {"mode": "explicit"}}]}]
        with mock.patch("model_direct_runner.request.urlopen", side_effect=[rejected, response]) as urlopen:
            result = run_openai_compatible_model(
                messages, "https://example.test/v1", "key", "model",
                prompt_cache_key="workflow", prompt_cache_options={"mode": "explicit"},
            )
        retry_payload = json.loads(urlopen.call_args_list[-1].args[0].data.decode())
        self.assertNotIn("prompt_cache_key", retry_payload)
        self.assertNotIn("prompt_cache_options", retry_payload)
        self.assertNotIn("prompt_cache_breakpoint", retry_payload["messages"][0]["content"][0])
        self.assertEqual(urlopen.call_count, 2)
        self.assertTrue(result["cache_compatibility"]["fallback_used"])
        self.assertIsNone(result["cached_tokens"])

    def test_default_api_request_has_no_cache_hints(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        with mock.patch("model_direct_runner.request.urlopen", return_value=response) as urlopen:
            run_openai_compatible_model(
                [{"role": "user", "content": "test"}], "https://example.test/v1", "key", "model"
            )
        request_payload = json.loads(urlopen.call_args.args[0].data.decode())
        self.assertNotIn("prompt_cache_key", request_payload)
        self.assertNotIn("prompt_cache_options", request_payload)
        self.assertNotIn("prompt_cache_breakpoint", json.dumps(request_payload))

    def test_explicit_supported_request_preserves_cache_hints(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        messages = [{"role": "system", "content": [{"type": "text", "text": "stable", "prompt_cache_breakpoint": {"mode": "explicit"}}]}]
        with mock.patch("model_direct_runner.request.urlopen", return_value=response) as urlopen:
            run_openai_compatible_model(
                messages, "https://supported.example/v1", "key", "model",
                prompt_cache_key="workflow", prompt_cache_options={"mode": "explicit"},
            )
        request_payload = json.loads(urlopen.call_args.args[0].data.decode())
        self.assertEqual(request_payload["prompt_cache_key"], "workflow")
        self.assertEqual(request_payload["prompt_cache_options"]["mode"], "explicit")
        self.assertEqual(request_payload["messages"][0]["content"][0]["prompt_cache_breakpoint"]["mode"], "explicit")

    def test_tool_choice_is_omitted_when_tools_are_absent(self):
        request_payload = self._api_request_payload(tools=None, tool_choice="none")
        self.assertNotIn("tools", request_payload)
        self.assertNotIn("tool_choice", request_payload)

    def test_tool_choice_is_omitted_when_tools_are_empty(self):
        request_payload = self._api_request_payload(tools=[], tool_choice="none")
        self.assertNotIn("tools", request_payload)
        self.assertNotIn("tool_choice", request_payload)

    def test_tool_choice_is_preserved_with_nonempty_tools(self):
        tools = [{
            "type": "function",
            "function": {"name": "kb_lookup", "description": "lookup", "parameters": {"type": "object"}},
        }]
        request_payload = self._api_request_payload(tools=tools, tool_choice="auto")
        self.assertEqual(request_payload["tools"], tools)
        self.assertEqual(request_payload["tool_choice"], "auto")

    def test_kb_disabled_ofx_request_has_no_illegal_tool_choice(self):
        request_payload = self._api_request_payload(
            tools=None,
            tool_choice="none",
            api_base="https://api.ofox.io/v1",
            model_name="openai/gpt-5.6-sol",
        )
        self.assertNotIn("tool_choice", request_payload)
        self.assertNotIn("prompt_cache_key", request_payload)
        self.assertNotIn("prompt_cache_options", request_payload)

    def _api_request_payload(
        self,
        *,
        tools,
        tool_choice,
        api_base="https://example.test/v1",
        model_name="model",
    ):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        with mock.patch("model_direct_runner.request.urlopen", return_value=response) as urlopen:
            run_openai_compatible_model(
                [{"role": "user", "content": "test"}],
                api_base,
                "key",
                model_name,
                tools=tools,
                tool_choice=tool_choice,
            )
        return json.loads(urlopen.call_args.args[0].data.decode())


if __name__ == "__main__":
    unittest.main()
