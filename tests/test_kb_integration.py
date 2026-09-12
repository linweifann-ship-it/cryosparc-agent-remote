import asyncio
import io
import json
import unittest
from unittest import mock

from kb_bridge import normalize_dataset_id
from kb_tool_loop import build_kb_tool_definitions, run_kb_tool_call_loop
from model_direct_runner import run_openai_compatible_model


class KBIntegrationTests(unittest.TestCase):
    def test_dataset_id_normalization(self):
        self.assertEqual(normalize_dataset_id("EMPIAR-10025"), "10025")
        self.assertEqual(normalize_dataset_id(empiar_id="EMPIAR10025"), "10025")

    def test_only_readonly_kb_tools_are_exposed(self):
        names = {
            item["function"]["name"]
            for item in build_kb_tool_definitions()
        }
        self.assertIn("kb_get_decision_context", names)
        self.assertIn("kb_get_job_doc", names)
        self.assertNotIn("execute_v2_model_decision", names)

    def test_model_response_parses_tool_calls_without_content(self):
        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)
        response.read.return_value = json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "kb_get_job_doc",
                            "arguments": '{"job_type":"ctf_estimation_new"}',
                        },
                    }],
                }
            }]
        }).encode()
        with mock.patch("model_direct_runner.request.urlopen", return_value=response):
            result = run_openai_compatible_model(
                [{"role": "user", "content": "state"}],
                "https://example.test/v1",
                "key",
                "model",
                tools=build_kb_tool_definitions(),
                tool_choice="auto",
            )
        self.assertEqual(result["raw_text"], "")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "kb_get_job_doc")
        self.assertEqual(result["request_payload"]["tool_choice"], "auto")

    def test_model_calls_kb_before_final_decision(self):
        responses = [
            {
                "assistant_message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "kb_get_next_steps",
                            "arguments": '{"job_type":"import_movies"}',
                        },
                    }],
                },
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "kb_get_next_steps",
                        "arguments": '{"job_type":"import_movies"}',
                    },
                }],
                "raw_text": "",
            },
            {
                "assistant_message": {"role": "assistant", "content": '{"decision_type":"stop"}'},
                "tool_calls": [],
                "raw_text": '{"decision_type":"stop"}',
            },
        ]
        model_inputs = []
        tool_inputs = []

        async def model_call(messages, tools):
            model_inputs.append(messages)
            return responses.pop(0)

        async def executor(name, arguments):
            tool_inputs.append((name, arguments))
            return {"results": [{"next_job_type": "patch_ctf"}]}

        result = asyncio.run(run_kb_tool_call_loop(
            [{"role": "user", "content": "state"}],
            model_call,
            executor,
            max_tool_calls=2,
        ))
        self.assertEqual(result["raw_text"], '{"decision_type":"stop"}')
        self.assertEqual(tool_inputs, [("kb_get_next_steps", {"job_type": "import_movies"})])
        self.assertEqual(len(model_inputs), 2)
        self.assertEqual(model_inputs[1][-1]["role"], "tool")
        self.assertEqual(len(result["tool_trace"]), 1)

    def test_tool_call_limit_is_bounded(self):
        model_inputs = []

        async def model_call(messages, tools):
            model_inputs.append((messages, tools))
            if not tools:
                return {
                    "assistant_message": {"role": "assistant", "content": '{"decision_type":"stop"}'},
                    "tool_calls": [],
                    "raw_text": '{"decision_type":"stop"}',
                }
            return {
                "assistant_message": {"role": "assistant", "content": None},
                "tool_calls": [{
                    "id": "call",
                    "type": "function",
                    "function": {"name": "kb_get_job_doc", "arguments": "{}"},
                }],
                "raw_text": "",
            }

        async def executor(name, arguments):
            return {"results": []}

        result = asyncio.run(run_kb_tool_call_loop(
            [{"role": "user", "content": "state"}],
            model_call,
            executor,
            max_tool_calls=0,
        ))
        self.assertEqual(result["raw_text"], '{"decision_type":"stop"}')
        self.assertTrue(result["kb_limit_reached"])
        self.assertEqual(len(model_inputs), 2)
        self.assertEqual(model_inputs[-1][1], [])

    def test_required_policy_rejects_final_decision_without_retrieval(self):
        async def model_call(messages, tools):
            return {"tool_calls": [], "raw_text": '{"decision_type":"stop"}'}

        async def executor(name, arguments):
            raise AssertionError("Must not be called")

        with self.assertRaisesRegex(RuntimeError, "required"):
            asyncio.run(run_kb_tool_call_loop(
                [{"role": "user", "content": "state"}], model_call, executor, required=True,
            ))


if __name__ == "__main__":
    unittest.main()
