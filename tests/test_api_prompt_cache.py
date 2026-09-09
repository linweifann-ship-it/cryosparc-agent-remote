import importlib.util
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
