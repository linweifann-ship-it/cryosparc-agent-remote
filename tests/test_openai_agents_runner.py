import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cryosparc_agent_remote.openai_agents_runner import (
    AgentsRunConfig,
    build_step_input,
    config_from_args,
    contains_stop,
    extract_usage,
    extract_created_jobs,
    summarize_prompt_cache,
)


class OpenAIAgentsRunnerTests(unittest.TestCase):
    def test_step_input_keeps_dynamic_values_out_of_static_instructions(self):
        config = AgentsRunConfig(
            project_uid="P2",
            workspace_uid="W1",
            start_node="J1",
            model="gpt-5.6-sol",
            base_url="https://api.ofox.ai/v1",
            api_key="sk-test",
            run_id="abc",
            output_dir=Path("runs/abc"),
            dataset_info={"empiar_id": "10025"},
            known_workflow_dirs=[],
            max_steps=1,
            max_turns_per_step=3,
            wait_timeout_seconds=9,
            poll_interval_seconds=1,
            server_python="python",
            project_dir=Path("/repo"),
            mcp_server="cryosparc_mcp_server.py",
            mcp_stdio_command=None,
            use_responses_api=False,
        )
        messages = build_step_input(config, "J7", 2)
        self.assertEqual([message["role"] for message in messages], ["system", "system", "user"])
        self.assertEqual(messages[1]["content"][0]["type"], "text")
        self.assertEqual(
            messages[1]["content"][0]["prompt_cache_breakpoint"], {"mode": "explicit"}
        )
        payload = json.loads(messages[2]["content"])
        self.assertEqual(payload["run_scope"]["project_uid"], "P2")
        self.assertEqual(payload["run_scope"]["current_node_id"], "J7")
        self.assertNotIn("output_contract", payload)

    def test_extract_created_jobs_deduplicates_nested_job_packages(self):
        event = {
            "new_items": [
                {"output": {"job_uid": "J9", "job_type": "import_movies", "status": "queued"}},
                {"output": {"job_uid": "J9", "job_type": "import_movies", "status": "completed"}},
            ]
        }
        self.assertEqual(
            extract_created_jobs(event),
            [{"project_uid": None, "workspace_uid": None, "job_uid": "J9", "job_type": "import_movies", "status": "queued", "queued": None}],
        )

    def test_summarize_prompt_cache_accepts_responses_and_chat_usage(self):
        summary = summarize_prompt_cache([
            {"input_tokens": 100, "output_tokens": 20, "input_tokens_details": {"cached_tokens": 40}},
            {"prompt_tokens": 50, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 25}},
        ])
        self.assertEqual(summary["input_tokens"], 150)
        self.assertEqual(summary["cached_input_tokens"], 65)
        self.assertEqual(summary["output_tokens"], 30)
        self.assertAlmostEqual(summary["cache_hit_ratio"], 65 / 150)

    def test_extract_usage_prefers_raw_provider_usage_without_double_counting(self):
        event = {
            "raw_responses": [
                {
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                    "raw_usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "input_tokens_details": {"cached_tokens": 80},
                    },
                }
            ]
        }
        summary = extract_usage(event)["summary"]
        self.assertEqual(summary["usage_record_count"], 1)
        self.assertEqual(summary["cached_input_tokens"], 80)

    def test_contains_stop_reads_final_json(self):
        self.assertTrue(contains_stop('{"decision_type":"stop"}'))
        self.assertFalse(contains_stop('{"decision_type":"forward"}'))

    def test_config_from_args_uses_env_style_defaults_without_key_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                project="P2",
                workspace="W1",
                start_node=None,
                model="gpt-5.6-sol",
                api_base="https://api.ofox.ai/v1",
                api_key="sk-test",
                run_id="run-x",
                output_dir=tmp,
                dataset_json="{}",
                dataset_json_file=None,
                known_workflow_dir=[],
                max_steps=1,
                max_turns_per_step=2,
                wait_timeout_seconds=3,
                poll_interval_seconds=1,
                server_python="python",
                project_dir="/repo",
                mcp_server="cryosparc_mcp_server.py",
                force_chat_completions=True,
            )
            config = config_from_args(args)
            self.assertEqual(config.run_id, "run-x")
            self.assertFalse(config.use_responses_api)


if __name__ == "__main__":
    unittest.main()
