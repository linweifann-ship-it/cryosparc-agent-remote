import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_runner():
    mcp = types.ModuleType("mcp")
    mcp.ClientSession = object
    mcp.StdioServerParameters = object
    sys.modules.setdefault("mcp", mcp)
    sys.modules.setdefault("mcp.client", types.ModuleType("mcp.client"))
    stdio = types.ModuleType("mcp.client.stdio")
    stdio.stdio_client = object
    sys.modules.setdefault("mcp.client.stdio", stdio)
    path = Path(__file__).resolve().parents[1] / "cryosparc_agent_remote" / "autonomous_mcp_closed_loop.py"
    spec = importlib.util.spec_from_file_location("runner_success_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AutonomousRunSuccessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_only_model_stop_is_successful(self):
        self.assertTrue(self.runner.is_successful_terminal_stop("model_stop"))
        for reason in ("model_request_input", "validation_failure_limit", "timeout", "exception", "max_rounds_reached"):
            self.assertFalse(self.runner.is_successful_terminal_stop(reason))


if __name__ == "__main__":
    unittest.main()
