"""Offline construction tests for the Deep Agents harness.

They intentionally do not call a model or any CryoSPARC API.
"""
import importlib.util
import sys
import unittest
from pathlib import Path


DEPS = ("deepagents", "langchain_openai", "langgraph")


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in DEPS), "Deep Agents optional dependencies are not installed")
class DeepAgentsHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cryosparc_agent_remote"))
        from deepagents_harness import build_agent
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
        from langgraph.checkpoint.memory import InMemorySaver

        @tool
        def get_workflow_decision_context(project_uid: str, workspace_uid: str) -> dict:
            """Minimal existing-MCP-shaped state tool for construction testing."""
            return {"project_uid": project_uid, "workspace_uid": workspace_uid}

        cls.agent = build_agent(
            ChatOpenAI(model="smoke-only", api_key="not-used", base_url="http://127.0.0.1:9/v1"),
            [get_workflow_decision_context],
            InMemorySaver(),
        )

    def test_builds_one_agent_without_task_tool(self):
        tools = set(self.agent.get_graph().nodes["tools"].data.tools_by_name)
        self.assertIn("get_workflow_decision_context", tools)
        self.assertNotIn("task", tools)
        self.assertNotIn("execute", tools)
        self.assertNotIn("write_file", tools)


class DeepAgentsRoundStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cryosparc_agent_remote"))
        from deepagents_harness import created_job_uids, terminal_observation_job_uid
        cls.created_job_uids = staticmethod(created_job_uids)
        cls.terminal_observation_job_uid = staticmethod(terminal_observation_job_uid)

    def test_terminal_mcp_observation_advances_current_node(self):
        observation = [{
            "type": "text",
            "text": '{"ready_for_model": true, "status": "completed", "job_uid": "J253"}',
        }]
        self.assertEqual(self.terminal_observation_job_uid(observation), "J253")

    def test_active_mcp_observation_does_not_advance_current_node(self):
        observation = '{"ready_for_model": false, "status": "running", "job_uid": "J253"}'
        self.assertIsNone(self.terminal_observation_job_uid(observation))

    def test_offloaded_terminal_observation_advances_to_created_job(self):
        observation = "Tool result too large, the result was saved in the filesystem"
        execution = [{"type": "text", "text": '''{
          "execution_result": {"execution_results": [
            {"success": true, "job_uid": "J278"}
          ]}
        }'''}]
        self.assertEqual(self.created_job_uids(execution), ["J278"])
        self.assertEqual(
            self.terminal_observation_job_uid(observation, execution), "J278"
        )

    def test_offloaded_multi_job_observation_does_not_choose_a_cursor(self):
        observation = "Tool result too large, the result was saved in the filesystem"
        execution = '{"execution_results": [' \
            '{"success": true, "job_uid": "J278"},' \
            '{"success": true, "job_uid": "J279"}]}'
        self.assertIsNone(self.terminal_observation_job_uid(observation, execution))


if __name__ == "__main__":
    unittest.main()
