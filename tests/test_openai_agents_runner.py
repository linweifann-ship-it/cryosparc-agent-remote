import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cryosparc_agent_remote.openai_agents_runner import (
    AgentsRunConfig,
    build_step_input,
    cancel_physical_race_job,
    config_from_args,
    contains_stop,
    extract_usage,
    extract_created_jobs,
    kill_race_losers,
    select_race_winner,
    summarize_prompt_cache,
)


class OpenAIAgentsRunnerTests(unittest.TestCase):
    def test_step_input_keeps_dynamic_values_out_of_static_instructions(self):
        dataset_info = {
            "input_type": "micrographs",
            "pixel_size_A": 0.6575,
            "accelerating_voltage_kv": 300,
            "spherical_aberration_mm": 2.7,
            "total_exposure_dose_e_per_A2": 53,
            "blob_paths": "/home/share/empiar/10025/data/14sep05c_averaged_196/*.mrc",
        }
        config = AgentsRunConfig(
            project_uid="P2",
            workspace_uid="W1",
            start_node="J1",
            model="gpt-5.6-sol",
            base_url="https://api.ofox.ai/v1",
            api_key="sk-test",
            run_id="abc",
            output_dir=Path("runs/abc"),
            dataset_info=dataset_info,
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
            prompt_cache_options_enabled=True,
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
        self.assertEqual(payload["mcp_arguments"]["dataset_info"], dataset_info)
        self.assertNotIn("output_contract", payload)
        prompt_text = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("first step", prompt_text.lower())
        self.assertNotIn("should import", prompt_text.lower())
        self.assertNotIn("应该", prompt_text)

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

    def test_extract_created_jobs_reads_json_encoded_mcp_output(self):
        event = {"new_items": [{"output": {"text": json.dumps({
            "job_uid": "J10", "job_type": "import_micrographs", "status": "queued"
        })}}]}
        self.assertEqual(extract_created_jobs(event)[0]["job_uid"], "J10")

    def test_extract_created_jobs_collapses_race_physical_jobs_to_logical_step(self):
        event = {
            "new_items": [
                {
                    "output": {
                        "success": True,
                        "project_uid": "P2",
                        "workspace_uid": "W16",
                        "job_uid": "J217",
                        "job_type": "patch_ctf_estimation_multi",
                        "status": "queued",
                        "queued": True,
                        "logical_workflow_step": True,
                        "logical_job": {
                            "logical_job_id": "logical-abc",
                            "logical_job_uid": "J217",
                            "physical_job_ids": ["J217", "J218"],
                            "winner_job_uid": None,
                        },
                        "diagnostics": {
                            "physical_jobs": [
                                {"job_uid": "J217", "resource_scheduling": {}},
                                {"job_uid": "J218", "resource_scheduling": {}},
                            ]
                        },
                    }
                }
            ]
        }

        self.assertEqual(
            extract_created_jobs(event),
            [
                {
                    "project_uid": "P2",
                    "workspace_uid": "W16",
                    "job_uid": "J217",
                    "job_type": "patch_ctf_estimation_multi",
                    "status": "queued",
                    "queued": True,
                    "logical_workflow_step": True,
                    "logical_job_id": "logical-abc",
                    "physical_job_ids": ["J217", "J218"],
                    "winner_job_uid": None,
                }
            ],
        )

    def test_select_race_winner_prefers_running_h20_over_queued_4090(self):
        self.assertEqual(
            select_race_winner({"J220": "launched", "J221": "running"}),
            "J221",
        )

    def test_kill_race_losers_kills_non_running_loser_when_winner_running(self):
        with patch(
            "cryosparc_agent_remote.openai_agents_runner.kill_job",
            return_value={"job_uid": "J220", "action": "kill", "success": True},
        ) as kill:
            result = kill_race_losers(
                "P2",
                "W16",
                "J221",
                {"J220": "launched", "J221": "running"},
            )

        kill.assert_called_once_with("P2", "W16", "J220", "launched")
        self.assertEqual(result[0]["job_uid"], "J220")

    def test_kill_race_losers_kills_running_loser(self):
        with patch(
            "cryosparc_agent_remote.openai_agents_runner.kill_job",
            return_value={"job_uid": "J220", "action": "kill", "success": True},
        ) as kill:
            result = kill_race_losers(
                "P2",
                "W16",
                "J221",
                {"J220": "running", "J221": "running"},
            )

        kill.assert_called_once_with("P2", "W16", "J220", "running")
        self.assertEqual(result[0]["job_uid"], "J220")

    def test_cancel_physical_race_job_falls_back_to_cancel(self):
        class FakeJob:
            def __init__(self):
                self.cancelled = False

            def kill(self):
                raise RuntimeError("cannot kill from this state")

            def cancel(self):
                self.cancelled = True

        job = FakeJob()
        result = cancel_physical_race_job(job, "J220", "queued")

        self.assertTrue(job.cancelled)
        self.assertEqual(result["method"], "cancel")
        self.assertTrue(result["success"])

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
                disable_prompt_cache_options=False,
            )
            config = config_from_args(args)
            self.assertEqual(config.run_id, "run-x")
            self.assertFalse(config.use_responses_api)
            self.assertTrue(config.prompt_cache_options_enabled)

    def test_config_accepts_large_max_steps_as_loop_guard(self):
        args = SimpleNamespace(
            project="P2",
            workspace="W16",
            start_node=None,
            model="openai/gpt-5.6-sol",
            api_base="https://api.ofox.io/v1",
            api_key="sk-test",
            run_id="p2w16-openaisdk-resource-scheduler-20260908",
            output_dir="runs",
            dataset_json='{"input_type":"micrographs"}',
            dataset_json_file=None,
            known_workflow_dir=[],
            max_steps=50,
            max_turns_per_step=12,
            wait_timeout_seconds=43200,
            poll_interval_seconds=60,
            server_python="/ssd1/linweifan/miniforge3/envs/cryoagent-model/bin/python",
            project_dir="/ssd1/linweifan/cryosparc-agent-openaisdk-adf6391-resource-scheduler",
            mcp_server="cryosparc_mcp_server.py",
            mcp_stdio_command=None,
            force_chat_completions=False,
            disable_prompt_cache_options=True,
        )

        config = config_from_args(args)

        self.assertEqual(config.max_steps, 50)
        self.assertEqual(config.workspace_uid, "W16")
        self.assertFalse(config.prompt_cache_options_enabled)


if __name__ == "__main__":
    unittest.main()
