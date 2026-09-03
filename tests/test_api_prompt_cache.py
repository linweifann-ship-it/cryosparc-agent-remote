import importlib.util
import asyncio
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


def load_api_runner_module():
    if "mcp" not in sys.modules:
        mcp = types.ModuleType("mcp")
        mcp.ClientSession = object
        mcp.StdioServerParameters = object
        sys.modules["mcp"] = mcp
    if "mcp.client" not in sys.modules:
        sys.modules["mcp.client"] = types.ModuleType("mcp.client")
    if "mcp.client.stdio" not in sys.modules:
        stdio = types.ModuleType("mcp.client.stdio")
        stdio.stdio_client = object
        sys.modules["mcp.client.stdio"] = stdio

    module_dir = Path(__file__).resolve().parents[1] / "cryosparc_agent_remote"
    sys.path.insert(0, str(module_dir))
    module_path = module_dir / "autonomous_mcp_closed_loop.py"
    spec = importlib.util.spec_from_file_location(
        "api_autonomous_mcp_closed_loop_for_cache_tests",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ApiPromptCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_api_runner_module()

    def test_explicit_cache_breakpoint_marks_static_system_message(self):
        messages = self.runner.build_autonomous_prompt(
            model_input={"schema_version": "2.0"},
            candidate_context={"generated_at": "dynamic-time"},
            round_index=2,
            mark_static_cache_breakpoint=True,
        )

        static_content = messages[1]["content"]
        self.assertIsInstance(static_content, list)
        self.assertEqual(static_content[0]["type"], "text")
        self.assertIn("output_contract", static_content[0]["text"])
        self.assertEqual(
            static_content[0]["prompt_cache_breakpoint"],
            {"mode": "explicit"},
        )
        self.assertIn("dynamic-time", messages[2]["content"])
        self.assertNotIn("dynamic-time", static_content[0]["text"])

    def test_prompt_cache_options_follow_selected_mode(self):
        args = Namespace(
            api_prompt_cache_mode="explicit",
            api_prompt_cache_key=None,
            api_prompt_cache_ttl="30m",
            project="P2",
            workspace="W9",
        )

        self.assertEqual(
            self.runner.resolve_prompt_cache_key(args),
            "cryoagent:P2:W9:workflow-v2",
        )
        self.assertEqual(
            self.runner.build_prompt_cache_options(args),
            {"mode": "explicit", "ttl": "30m"},
        )

        args.api_prompt_cache_mode = "disabled"
        self.assertIsNone(self.runner.resolve_prompt_cache_key(args))
        self.assertIsNone(self.runner.build_prompt_cache_options(args))

    def test_cache_summary_distinguishes_hit_miss_and_unreported_usage(self):
        hit = self.runner.summarize_prompt_cache_calls(
            [{"usage": {"present": True, "prompt_tokens": 100, "cached_tokens": 80}}]
        )
        self.assertEqual(hit["status"], "hit")
        self.assertEqual(hit["cache_hit_ratio"], 0.8)

        miss = self.runner.summarize_prompt_cache_calls(
            [{"usage": {"present": True, "prompt_tokens": 100, "cached_tokens": 0}}]
        )
        self.assertEqual(miss["status"], "miss")
        self.assertEqual(miss["cached_tokens"], 0)

        unreported = self.runner.summarize_prompt_cache_calls(
            [{"usage": {"present": False, "cached_tokens": None}}]
        )
        self.assertEqual(unreported["status"], "usage_not_reported")
        self.assertIsNone(unreported["cache_hit_ratio"])

    def test_candidate_context_reuses_model_input_snapshot(self):
        context = self.runner.candidate_context_from_model_input(
            {
                "candidate_actions": [{"job_type": "class_2D_new"}],
                "blocked_actions": [{"job_type": "homo_refine_new"}],
                "candidate_context": {"current_node_id": "J7", "registry_version": "v2"},
            }
        )
        self.assertEqual(context["candidate_actions"][0]["job_type"], "class_2D_new")
        self.assertEqual(context["current_node_id"], "J7")

    def test_request_audit_keeps_fixed_layers_out_of_dynamic_hash(self):
        messages = self.runner.build_autonomous_prompt(
            {"state": "dynamic"}, {"candidate_actions": []}, 1, True
        )
        audit = self.runner.build_request_audit(messages, 1, "J7")
        self.assertEqual(audit["round"], 1)
        self.assertIn("prompt_cache_breakpoint", audit["breakpoint_prefix"])
        self.assertNotEqual(audit["system1_sha256"], audit["dynamic_input_sha256"])

    def test_call_tool_accepts_current_mcp_is_error_field(self):
        class Session:
            async def call_tool(self, tool_name, arguments):
                return SimpleNamespace(is_error=True, content=[])

        result = asyncio.run(self.runner.call_tool_json(Session(), "tool", {}))
        self.assertTrue(result["mcp_error"])


if __name__ == "__main__":
    unittest.main()
