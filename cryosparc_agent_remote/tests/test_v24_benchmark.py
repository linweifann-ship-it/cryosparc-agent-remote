import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from benchmark_core import (
    append_event,
    build_v24_messages,
    compare_benchmark_runs,
    compute_metrics,
    evaluate_model_round,
    sha256_json,
    write_compare_outputs,
    write_json,
)
from connection_resolver import resolve_action_connections
from model_clients import (
    DashScopeQwenClient,
    ModelClientConfig,
    OpenAIResponsesClient,
    missing_key_result,
)
from run_context_guard import RunContextGuard
from v24_contract import strict_parse_model_output, validate_v24_decision


def workflow_state(outputs_by_job=None):
    outputs_by_job = outputs_by_job or {
        "J1": {
            "particles": {
                "available": True,
                "result_names": ["blob", "ctf", "location"],
            }
        }
    }
    nodes = []
    for job_uid, outputs in outputs_by_job.items():
        nodes.append(
            {
                "workflow_node_id": job_uid,
                "cryosparc_job_uid": job_uid,
                "job_type": "source",
                "status": "completed",
                "outputs": {
                    name: {
                        "type": name,
                        "available": value.get("available", True),
                        "num_items": 10,
                        "result_names": value.get("result_names", []),
                        "summary_keys": [],
                        "latest_summary_stat_keys": [],
                    }
                    for name, value in outputs.items()
                },
            }
        )
    return {
        "project_uid": "P2",
        "workspace_uid": "W1",
        "nodes": nodes,
    }


def v24_forward(job_type="class_2D_new", parameters=None):
    return {
        "schema_version": "3.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "job_type": job_type,
                "parameters": parameters or {"class2D_K": 50},
            }
        ],
    }


class V24StrictParsingTests(unittest.TestCase):
    def test_plain_json_is_accepted(self):
        result = strict_parse_model_output(json.dumps(v24_forward()))
        self.assertEqual(result["schema_version"], "3.0")

    def test_markdown_wrapped_json_is_rejected(self):
        with self.assertRaises(json.JSONDecodeError):
            strict_parse_model_output("```json\n{}\n```")

    def test_forbidden_connections_field_is_rejected(self):
        decision = v24_forward()
        decision["selected_actions"][0]["connections"] = {"particles": ["J1", "particles"]}
        validation = validate_v24_decision(decision)
        self.assertFalse(validation["success"])
        self.assertEqual(validation["issues"][0]["path"], "selected_actions.0.connections")


class RunContextGuardTests(unittest.TestCase):
    def test_rejects_candidate_actions_and_foreign_history(self):
        guard = RunContextGuard("run1", "P2", "W1", "J10")
        model_input = {
            "schema_version": "2.1",
            "candidate_actions": [],
            "dataset_context": {"dataset_metadata": {"known_workflow_steps": None}},
            "current_state": {
                "last_node_status": "completed",
                "last_node_info": {"project_uid": "P2", "workspace_uid": "W1"},
                "recent_job_history": [{"job_uid": "J9", "job_type": "import_movies"}],
            },
        }
        result = guard.assert_model_input(model_input)
        self.assertFalse(result["success"])
        codes = {item["code"] for item in result["issues"]}
        self.assertIn("forbidden_model_input_key", codes)
        self.assertIn("foreign_history_job", codes)

    def test_prompt_does_not_include_candidate_actions(self):
        messages = build_v24_messages(
            {
                "schema_version": "2.1",
                "task_type": "workflow_decision",
                "dataset_context": {"dataset_metadata": {"known_workflow_steps": None}},
                "current_state": {"last_node_status": "completed"},
            },
            1,
        )
        result = RunContextGuard("run1", "P2", "W1", "J10").assert_messages(messages)
        self.assertTrue(result["success"])


class ConnectionResolverTests(unittest.TestCase):
    def test_unique_connection_resolves(self):
        result = resolve_action_connections(
            "class_2D_new",
            workflow_state(),
            allowed_job_uids={"J1"},
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["connections"]["particles"], ("J1", "particles"))

    def test_zero_connection_fails(self):
        result = resolve_action_connections(
            "class_2D_new",
            workflow_state({"J1": {"templates": {"result_names": ["template"]}}}),
            allowed_job_uids={"J1"},
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "connection_resolution_failed")

    def test_multiple_connections_are_ambiguous(self):
        result = resolve_action_connections(
            "class_2D_new",
            workflow_state(
                {
                    "J1": {"particles": {"result_names": ["location"]}},
                    "J2": {"particles": {"result_names": ["location"]}},
                }
            ),
            allowed_job_uids={"J1", "J2"},
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "ambiguous_connection")


class V24RoundEvaluationTests(unittest.TestCase):
    def test_validation_only_round_succeeds_without_job_creation(self):
        guard = RunContextGuard("run1", "P2", "W1", "J1")
        result = evaluate_model_round(
            json.dumps(v24_forward()),
            workflow_state=workflow_state(),
            guard=guard,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["stage"], "validation_only_complete")
        self.assertEqual(result["connection_result"]["status"], "resolved")

    def test_unknown_job_is_model_invalid_action(self):
        guard = RunContextGuard("run1", "P2", "W1", "J1")
        result = evaluate_model_round(
            json.dumps(v24_forward("not_a_job")),
            workflow_state=workflow_state(),
            guard=guard,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_category"], "model_invalid_action")


class ProviderClientTests(unittest.TestCase):
    def test_missing_openai_key_is_sanitized(self):
        result = missing_key_result(
            "openai",
            ModelClientConfig(provider="openai", model_label="gpt", openai_model="gpt-test"),
            "req1",
            "OPENAI_API_KEY",
        )
        self.assertFalse(result["success"])
        self.assertNotIn("sk-", json.dumps(result))

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-secret-value"})
    @patch("model_clients.post_json")
    def test_openai_client_uses_responses_payload_without_logging_key(self, post_json):
        post_json.return_value = {"success": True, "raw_text": "{}"}
        client = OpenAIResponsesClient(
            ModelClientConfig(provider="openai", model_label="gpt", openai_model="gpt-test")
        )
        client.generate([{"role": "user", "content": "x"}], "req1")
        kwargs = post_json.call_args.kwargs
        self.assertEqual(kwargs["url"], "https://api.openai.com/v1/responses")
        self.assertTrue(kwargs["body"]["store"] is False)
        self.assertEqual(kwargs["body"]["text"]["format"]["type"], "json_schema")
        self.assertNotIn("sk-secret-value", json.dumps(kwargs["body"]))

    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "dash-secret"})
    @patch("model_clients.post_json")
    def test_dashscope_client_uses_openai_compatible_chat_endpoint(self, post_json):
        post_json.return_value = {"success": True, "raw_text": "{}"}
        client = DashScopeQwenClient(
            ModelClientConfig(
                provider="dashscope_qwen",
                model_label="qwen",
                qwen_model="qwen-test",
            )
        )
        client.generate([{"role": "user", "content": "x"}], "req1")
        kwargs = post_json.call_args.kwargs
        self.assertTrue(kwargs["url"].endswith("/chat/completions"))
        self.assertEqual(kwargs["body"]["response_format"]["type"], "json_object")
        self.assertNotIn("dash-secret", json.dumps(kwargs["body"]))

    @patch.dict("os.environ", {"DASHSCOPE_API_KEY": "dash-secret"})
    @patch("model_clients.post_json")
    def test_dashscope_native_mode_records_finetuned_model_path(self, post_json):
        post_json.return_value = {"success": True, "raw_text": "{}"}
        client = DashScopeQwenClient(
            ModelClientConfig(
                provider="dashscope_qwen",
                model_label="qwen_ft",
                qwen_model="ft-qwen-checkpoint",
                finetune_checkpoint="ft-qwen-checkpoint",
                dashscope_api_mode="native",
            )
        )
        client.generate([{"role": "user", "content": "x"}], "req1")
        kwargs = post_json.call_args.kwargs
        self.assertIn("/api/v1/services/aigc/text-generation/generation", kwargs["url"])
        self.assertEqual(kwargs["body"]["model"], "ft-qwen-checkpoint")
        self.assertNotIn("dash-secret", json.dumps(kwargs["body"]))


class BenchmarkCompareTests(unittest.TestCase):
    def test_metrics_and_compare_outputs_are_repeatable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark_dir = root / "B1"
            for index, model in enumerate(["gpt", "qwen"]):
                run_dir = benchmark_dir / f"{model}_r{index}"
                run_dir.mkdir(parents=True)
                manifest = {
                    "benchmark_id": "B1",
                    "run_id": f"{model}_r{index}",
                    "model_label": model,
                    "provider": model,
                    "schema_version": "V24",
                    "prompt_hash": "p1",
                    "dataset_hash": "d1",
                    "start_node": "J1",
                    "runner_commit": "c1",
                    "mcp_commit": "c1",
                    "config_hash": "cfg",
                    "max_rounds": 1,
                }
                write_json(run_dir / "benchmark_manifest.json", manifest)
                append_event(run_dir / "events.jsonl", manifest["run_id"], 1, "model_response", status="success", input_tokens=1, output_tokens=1)
                append_event(run_dir / "events.jsonl", manifest["run_id"], 1, "parse_result", status="success")
                append_event(run_dir / "events.jsonl", manifest["run_id"], 1, "validation_result", status="success")
                append_event(run_dir / "events.jsonl", manifest["run_id"], 1, "connection_result", status="resolved")
                compute_metrics(run_dir)

            report1 = compare_benchmark_runs(benchmark_dir)
            write_compare_outputs(benchmark_dir, report1)
            report2 = compare_benchmark_runs(benchmark_dir)
            self.assertTrue(report1["comparability"]["directly_comparable"])
            self.assertEqual(sha256_json(report1), sha256_json(report2))
            self.assertTrue((benchmark_dir / "benchmark_report.md").exists())


if __name__ == "__main__":
    unittest.main()
