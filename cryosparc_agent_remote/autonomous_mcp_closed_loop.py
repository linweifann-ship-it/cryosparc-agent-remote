# Run an autonomous model -> MCP -> CryoSPARC closed-loop test.
import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_direct_runner import (
    parse_model_decision_text,
    resolve_api_key,
    run_openai_compatible_model,
)


DEFAULT_BASE_MODEL = "/ssd1/lisongyang/models/Qwen3.6-27B-ms-test"
DEFAULT_ADAPTER = "/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow"
DEFAULT_MODEL_PYTHON = "/ssd1/linweifan/miniforge3/envs/cryoagent-model/bin/python"
DEFAULT_SERVER_PYTHON = "/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python"
DEFAULT_PROJECT_DIR = "/ssd1/linweifan/cryosparc_agent"
DEFAULT_MCP_SERVER = "cryosparc_mcp_server.py"

SYSTEM_PROMPT = (
    "You are the only decision maker for an autonomous CryoSPARC workflow test. "
    "Return exactly one JSON object. Do not include markdown, comments, or "
    "thinking text. Codex and MCP will not repair missing decisions for you."
)

STATIC_DECISION_INSTRUCTIONS = {
    "instruction": (
        "Choose exactly one next action, rollback, request_input, or stop from the current "
        "CryoSPARC state. Only choose a job type present in candidate_actions_from_mcp. "
        "If you choose a job, include every required input connection you want MCP to use. "
        "MCP will return validation or execution errors without repairing your decision."
    ),
    "output_contract": {
        "schema_version": "2.0",
        "decision_type": "forward | branch | rollback | stop | request_input",
        "action": "CryoSPARC job type for forward/branch decisions.",
        "job_type": "Same as action when using compact format.",
        "parameters": "Only non-default parameters explicitly chosen by the model.",
        "connections": {
            "input_name": {
                "source_job_uid": "CryoSPARC source job, chosen by the model",
                "source_output": "CryoSPARC output group, chosen by the model",
            }
        },
        "reason": "Decision reason. Use your own evidence only.",
        "confidence": "Number from 0.0 to 1.0.",
        "risk_flags": [],
        "evidence": [],
    },
    "valid_examples": [
        {
            "schema_version": "2.0",
            "decision_type": "forward",
            "action": "patch_ctf_estimation_multi",
            "job_type": "patch_ctf_estimation_multi",
            "parameters": {"compute_num_gpus": 1},
            "connections": {
                "exposures": {
                    "source_job_uid": "J123",
                    "source_output": "imported_micrographs",
                }
            },
            "reason": "The current completed job provides micrographs.",
            "confidence": 0.8,
            "risk_flags": [],
            "evidence": ["Current output imported_micrographs is available."],
        },
        {
            "schema_version": "2.0",
            "decision_type": "stop",
            "reason": "No safe autonomous action is clear.",
            "confidence": 0.5,
            "risk_flags": ["needs_human_review"],
            "evidence": [],
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--current-node")
    parser.add_argument("--dataset-json", default="{}")
    parser.add_argument("--dataset-json-file")
    parser.add_argument("--known-workflow-dir", action="append", default=[])
    parser.add_argument("--backend", choices=["local", "api"], default="local")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--api-base", default="https://api.openai.com/v1")
    parser.add_argument("--api-model", default="gpt-5.6-luna")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--api-prompt-cache-mode",
        choices=["explicit", "implicit", "disabled"],
        default="explicit",
    )
    parser.add_argument("--api-prompt-cache-key")
    parser.add_argument("--api-prompt-cache-ttl", default="30m")
    parser.add_argument("--model-python", default=DEFAULT_MODEL_PYTHON)
    parser.add_argument("--model-srun-prefix")
    parser.add_argument("--server-python", default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--mcp-server", default=DEFAULT_MCP_SERVER)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--max-validation-failures", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--wait-timeout-seconds", type=int, default=43200)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--output-dir", default="reports/autonomous_mcp_closed_loop")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    run_dir = make_run_dir(Path(args.output_dir))
    dataset_info = load_dataset_info(args)
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "start_current_node": args.current_node,
        "backend": args.backend,
        "model_paths": {
            "base_model": args.base_model if args.backend == "local" else None,
            "adapter": args.adapter if args.backend == "local" else None,
            "api_base": args.api_base if args.backend == "api" else None,
            "api_model": args.api_model if args.backend == "api" else None,
        },
        "command": " ".join(sys.argv),
        "rounds": [],
        "stop_reason": None,
        "success": False,
    }

    server_env = os.environ.copy()
    if os.getenv("CRYOAGENT_GPU_LANE"):
        server_env["CRYOAGENT_GPU_LANE"] = os.environ["CRYOAGENT_GPU_LANE"]
    server_params = StdioServerParameters(
        command=args.server_python,
        args=[str(Path(args.project_dir) / args.mcp_server)],
        cwd=args.project_dir,
        env=server_env,
    )

    current_node = args.current_node
    feedback_to_model: dict[str, Any] | None = None
    validation_failures = 0

    model_worker = None
    if args.backend == "local":
        model_worker = ModelWorker(args, run_dir / "model_worker.stderr.log")
        model_worker.start()
        summary["model_worker"] = {
            "command": model_worker.command,
            "load_event": model_worker.load_event,
        }
    else:
        summary["api_backend"] = {
            "api_base": args.api_base,
            "api_model": args.api_model,
            "api_key_env": args.api_key_env,
            "prompt_cache_mode": args.api_prompt_cache_mode,
            "prompt_cache_key": resolve_prompt_cache_key(args),
            "prompt_cache_ttl": args.api_prompt_cache_ttl,
        }
    async with AsyncExitStack() as stack:
        if model_worker is not None:
            stack.callback(model_worker.close)
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        for round_index in range(1, args.max_rounds + 1):
            round_dir = run_dir / f"round_{round_index:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_log: dict[str, Any] = {
                "round": round_index,
                "started_at": utc_now(),
                "current_node": current_node,
                "files": {},
            }
            summary["rounds"].append(round_log)

            context_args = {
                "project_uid": args.project,
                "workspace_uid": args.workspace,
                "current_job_uid": current_node,
                "dataset_info": dataset_info,
                "known_workflow_dirs": args.known_workflow_dir or None,
            }
            model_input = await call_tool_json(
                session,
                "get_workflow_decision_context",
                context_args,
            )
            model_input["failure_context"] = build_failure_context(feedback_to_model)
            candidate_context = await call_tool_json(
                session,
                "get_candidate_actions",
                {
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "current_node_id": current_node,
                    "dataset_info": dataset_info,
                },
            )
            round_log["model_input"] = model_input
            round_log["candidate_context"] = candidate_context
            write_json(round_dir / "model_input.json", model_input, round_log)
            write_json(round_dir / "candidate_actions.json", candidate_context, round_log)

            if model_input.get("internal_only") or model_input.get("ready_for_model") is False:
                summary["stop_reason"] = "current_job_not_terminal"
                round_log["next_round_reason"] = model_input.get("message")
                break

            messages = build_autonomous_prompt(
                model_input=model_input,
                candidate_context=candidate_context,
                round_index=round_index,
                mark_static_cache_breakpoint=args.backend == "api"
                and args.api_prompt_cache_mode == "explicit",
            )
            messages_file = round_dir / "model_messages.json"
            write_json(messages_file, messages, round_log)

            generation_file = round_dir / "model_generation.json"
            try:
                if args.backend == "api":
                    model_call = await asyncio.to_thread(
                        run_openai_compatible_model,
                        messages,
                        args.api_base,
                        resolve_api_key(args.api_key, args.api_key_env),
                        args.api_model,
                        args.max_new_tokens,
                        args.temperature,
                        300,
                        prompt_cache_key=resolve_prompt_cache_key(args),
                        prompt_cache_options=build_prompt_cache_options(args),
                    )
                    model_call["request_id"] = f"round_{round_index:02d}"
                else:
                    assert model_worker is not None
                    model_call = model_worker.generate(
                        request_id=f"round_{round_index:02d}",
                        messages=messages,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
            except Exception as exc:
                error_payload = {
                    "round": round_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "api_base": args.api_base if args.backend == "api" else None,
                    "api_model": args.api_model if args.backend == "api" else None,
                }
                round_log["model_error"] = error_payload
                write_json(round_dir / "model_error.json", error_payload, round_log)
                summary["stop_reason"] = "model_call_error"
                summary["error"] = error_payload
                summary["finished_at"] = utc_now()
                write_json(run_dir / "summary.json", summary, None)
                print_run_summary(summary, run_dir)
                return
            round_log["model_call"] = model_call
            round_log["raw_model_output_file"] = str(generation_file)
            write_json(generation_file, model_call, round_log)
            generation = model_call
            raw_output = generation.get("raw_text", "")
            round_log["raw_model_output"] = raw_output

            try:
                decision = parse_model_decision_text(raw_output)
                parse_result = {"success": True, "decision": decision}
            except Exception as exc:
                decision = None
                parse_result = {
                    "success": False,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "raw_model_output": raw_output,
                }
            round_log["json_parse"] = parse_result
            write_json(round_dir / "json_parse.json", parse_result, round_log)

            if decision is None:
                validation_failures += 1
                feedback_to_model = make_feedback(
                    round_index,
                    "json_parse_failed",
                    parse_result,
                    model_input,
                    candidate_context,
                )
                if validation_failures >= args.max_validation_failures:
                    summary["stop_reason"] = "validation_failure_limit"
                    break
                continue

            decision_file = round_dir / "model_decision.json"
            write_json(decision_file, decision, round_log)

            validation_args = {
                "decision": decision,
                "project_uid": args.project,
                "workspace_uid": args.workspace,
                "current_node_id": current_node,
            }
            validation = await call_tool_json(
                session,
                "validate_v2_model_decision",
                validation_args,
            )
            round_log["mcp_validation_tool"] = {
                "tool_name": "validate_v2_model_decision",
                "arguments": validation_args,
                "result": validation,
            }
            write_json(round_dir / "validation.json", validation, round_log)

            if not validation.get("success"):
                validation_failures += 1
                feedback_to_model = make_feedback(
                    round_index,
                    "v2_validation_failed",
                    validation,
                    model_input,
                    candidate_context,
                    failed_decision=decision,
                )
                if validation_failures >= args.max_validation_failures:
                    summary["stop_reason"] = "validation_failure_limit"
                    break
                continue

            validation_failures = 0
            decision_type = decision.get("decision_type")
            if decision_type in {"stop", "complete", "request_input"}:
                summary["stop_reason"] = f"model_{decision_type}"
                round_log["next_round_reason"] = decision.get("reason")
                break

            execution_args = {
                "decision": decision,
                "project_uid": args.project,
                "workspace_uid": args.workspace,
                "current_node_id": current_node,
                "dry_run": False,
            }
            execution = await call_tool_json(
                session,
                "execute_v2_model_decision",
                execution_args,
            )
            round_log["mcp_execution_tool"] = {
                "tool_name": "execute_v2_model_decision",
                "arguments": execution_args,
                "result": execution,
            }
            write_json(round_dir / "execution.json", execution, round_log)

            created_jobs = find_created_jobs(execution)
            round_log["created_jobs"] = created_jobs
            if not created_jobs:
                feedback_to_model = make_feedback(
                    round_index,
                    "execution_failed_or_no_job",
                    execution,
                    model_input,
                    candidate_context,
                    failed_decision=decision,
                )
                round_log["next_round_reason"] = "Execution returned no created job; feedback sent to model."
                continue

            job_feedbacks = []
            for job in created_jobs:
                wait_args = {
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "job_uid": job["job_uid"],
                    "timeout_seconds": args.wait_timeout_seconds,
                    "poll_interval_seconds": args.poll_interval_seconds,
                    "include_next_candidates": True,
                }
                job_result = await call_tool_json(
                    session,
                    "wait_for_job_result_package",
                    wait_args,
                )
                job_result["local_job_logs"] = collect_job_logs(job["job_uid"])
                job_feedbacks.append(
                    {
                        "mcp_tool_name": "wait_for_job_result_package",
                        "arguments": wait_args,
                        "result": job_result,
                    }
                )
            round_log["job_results"] = job_feedbacks
            write_json(round_dir / "job_results.json", job_feedbacks, round_log)

            terminal_jobs = [
                feedback["result"]
                for feedback in job_feedbacks
                if feedback["result"].get("ready_for_model")
            ]
            if not terminal_jobs:
                summary["stop_reason"] = "job_wait_timeout_or_not_terminal"
                feedback_to_model = make_feedback(
                    round_index,
                    "job_not_terminal",
                    job_feedbacks,
                    model_input,
                    candidate_context,
                    failed_decision=decision,
                )
                break

            last_job = terminal_jobs[-1]
            current_node = last_job.get("job_uid") or current_node
            feedback_to_model = make_feedback(
                round_index,
                "job_result",
                last_job,
                model_input,
                candidate_context,
                failed_decision=decision if is_failed_job_result(last_job) else None,
            )
            round_log["next_round_reason"] = (
                f"此步骤由第 {round_index} 轮 Model 输出中的 action 字段触发。MCP 只负责执行。"
            )

        else:
            summary["stop_reason"] = "max_rounds_reached"

    summary["finished_at"] = utc_now()
    summary["success"] = summary["stop_reason"] in {"model_stop", "model_complete", "model_request_input", "max_rounds_reached"}
    write_json(run_dir / "summary.json", summary, None)
    print_run_summary(summary, run_dir)


def load_dataset_info(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_json_file:
        return json.loads(Path(args.dataset_json_file).read_text())
    return json.loads(args.dataset_json)


def build_autonomous_prompt(
    model_input: dict[str, Any],
    candidate_context: dict[str, Any],
    round_index: int,
    mark_static_cache_breakpoint: bool = False,
) -> list[dict[str, Any]]:
    dynamic_payload = {
        "round": round_index,
        "model_input": model_input,
        "candidate_actions_from_mcp": candidate_context,
        "failure_context": model_input.get("failure_context"),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": build_cacheable_text_content(
                json.dumps(
                    STATIC_DECISION_INSTRUCTIONS,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                mark_static_cache_breakpoint,
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                dynamic_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]


def build_cacheable_text_content(
    text: str,
    cache_breakpoint: bool,
) -> str | list[dict[str, Any]]:
    if not cache_breakpoint:
        return text
    return [
        {
            "type": "text",
            "text": text,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]


def resolve_prompt_cache_key(args: argparse.Namespace) -> str | None:
    if args.api_prompt_cache_mode == "disabled":
        return None
    if args.api_prompt_cache_key:
        return args.api_prompt_cache_key
    return f"cryoagent:{args.project}:{args.workspace}:workflow-v2"


def build_prompt_cache_options(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.api_prompt_cache_mode == "disabled":
        return None
    options: dict[str, Any] = {"ttl": args.api_prompt_cache_ttl}
    if args.api_prompt_cache_mode == "explicit":
        options["mode"] = "explicit"
    return options


class ModelWorker:
    def __init__(self, args: argparse.Namespace, stderr_path: Path):
        self.args = args
        self.stderr_path = stderr_path
        self.stderr_file = None
        self.command = [
            args.model_python,
            str(Path(args.project_dir) / "scripts" / "model_generate_once.py"),
            "--worker",
            "--base-model",
            args.base_model,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--device-map",
            args.device_map,
            "--torch-dtype",
            args.torch_dtype,
        ]
        if args.adapter:
            self.command.extend(["--adapter", args.adapter])
        if args.model_srun_prefix:
            self.command = shlex.split(args.model_srun_prefix) + self.command
        self.process: subprocess.Popen[str] | None = None
        self.load_event: dict[str, Any] | None = None

    def start(self) -> None:
        self.stderr_file = self.stderr_path.open("w")
        self.process = subprocess.Popen(
            self.command,
            cwd=self.args.project_dir,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_file,
            bufsize=1,
        )
        self.load_event = self._read_event()
        if self.load_event.get("event") != "model_loaded":
            raise RuntimeError(f"Model worker failed to load: {self.load_event}")

    def generate(
        self,
        request_id: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdin is not None
        request = {
            "request_id": request_id,
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        event = self._read_event()
        if event.get("event") != "generation":
            raise RuntimeError(f"Unexpected model worker event: {event}")
        event["success"] = True
        event["model_loaded_once"] = True
        return event

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdin:
                self.process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self.process.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=30)
        if self.stderr_file:
            self.stderr_file.close()

    def _read_event(self) -> dict[str, Any]:
        assert self.process is not None
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                return {
                    "event": "worker_exited",
                    "returncode": self.process.poll(),
                    "stderr_path": str(self.stderr_path),
                }
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue


async def call_tool_json(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    result = await session.call_tool(tool_name, arguments)
    if result.isError:
        return {
            "success": False,
            "mcp_error": True,
            "tool_name": tool_name,
            "content": [content.model_dump() for content in result.content],
        }
    if not result.content:
        return {"success": False, "tool_name": tool_name, "content": []}
    first = result.content[0]
    text = getattr(first, "text", None)
    if text is None:
        return {
            "success": False,
            "tool_name": tool_name,
            "content": [content.model_dump() for content in result.content],
        }
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"success": False, "tool_name": tool_name, "raw_text": text}


def make_feedback(
    round_index: int,
    feedback_type: str,
    payload: Any,
    model_input: dict[str, Any],
    candidate_context: dict[str, Any],
    failed_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "message_type": "mcp_feedback_to_model",
        "round": round_index,
        "feedback_type": feedback_type,
        "payload": payload,
        "failed_decision": failed_decision,
        "current_workflow_state": model_input,
        "candidate_actions": candidate_context,
        "instruction": "Use this feedback to decide the next v2 JSON action. MCP only reports the result.",
    }


def build_failure_context(feedback: dict[str, Any] | None) -> dict[str, Any] | None:
    if not feedback:
        return None
    feedback_type = feedback.get("feedback_type")
    payload = feedback.get("payload")
    failed_decision = feedback.get("failed_decision")
    if feedback_type == "job_result" and not is_failed_job_result(payload):
        return None
    return {
        "has_failure": True,
        "failure_round": feedback.get("round"),
        "failure_stage": map_failure_stage(feedback_type, payload),
        "failed_decision": failed_decision,
        "failed_action": extract_failed_action(failed_decision),
        "execution_error": extract_execution_error(feedback_type, payload),
        "missing_inputs": extract_named_items(payload, ("missing_inputs", "missing_required_inputs")),
        "invalid_connections": extract_named_items(payload, ("invalid_connections", "connection_errors")),
        "available_inputs": extract_named_items(
            feedback.get("current_workflow_state"),
            ("available_inputs", "outputs", "input_sources"),
        ),
        "candidate_actions": feedback.get("candidate_actions"),
        "attempt_history": [
            {
                "round": feedback.get("round"),
                "action": (failed_decision or {}).get("action") or (failed_decision or {}).get("job_type"),
                "failure_stage": map_failure_stage(feedback_type, payload),
                "message": extract_error_message(payload),
            }
        ],
        "retry_guidance": build_retry_guidance(failed_decision, payload),
    }


def map_failure_stage(feedback_type: str | None, payload: Any) -> str:
    if feedback_type == "json_parse_failed":
        return "json_parse"
    if feedback_type == "v2_validation_failed":
        return "validation"
    if feedback_type == "job_not_terminal":
        return "timeout"
    if feedback_type == "execution_failed_or_no_job":
        text = json.dumps(payload, ensure_ascii=False, default=str).lower()
        if "approval_required" in text:
            return "approval"
        if "connect" in text or "input" in text:
            return "connect_inputs"
        if "queue" in text or "missing required" in text:
            return "queue"
        return "no_created_job"
    if feedback_type == "job_result":
        return "run"
    return feedback_type or "unknown"


def extract_failed_action(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decision:
        return None
    return {
        "decision_type": decision.get("decision_type"),
        "action": decision.get("action"),
        "job_type": decision.get("job_type") or decision.get("action"),
        "parameters": decision.get("parameters") or {},
        "connections": decision.get("connections") or {},
    }


def extract_execution_error(feedback_type: str | None, payload: Any) -> dict[str, Any]:
    return {
        "source": infer_error_source(feedback_type, payload),
        "error_type": extract_error_type(payload),
        "message": extract_error_message(payload),
        "raw": payload,
    }


def infer_error_source(feedback_type: str | None, payload: Any) -> str:
    if feedback_type == "json_parse_failed":
        return "json_parser"
    text = json.dumps(payload, ensure_ascii=False, default=str).lower()
    if "cryosparc" in text or "http" in text or "traceback" in text:
        return "cryosparc"
    if "mcp" in text:
        return "mcp"
    return "runner"


def extract_error_type(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key in ("error_type", "type", "code", "status_code"):
            if payload.get(key):
                return str(payload[key])
    return None


def extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("message", "error", "detail", "stderr", "raw_text"):
            value = payload.get(key)
            if value:
                return str(value)
        for value in payload.values():
            message = extract_error_message(value)
            if message:
                return message
    if isinstance(payload, list):
        for value in payload:
            message = extract_error_message(value)
            if message:
                return message
    return ""


def extract_named_items(payload: Any, names: tuple[str, ...]) -> list[Any]:
    found: list[Any] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in names:
                found.extend(value if isinstance(value, list) else [value])
            else:
                found.extend(extract_named_items(value, names))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(extract_named_items(value, names))
    return found


def build_retry_guidance(decision: dict[str, Any] | None, payload: Any) -> dict[str, Any] | None:
    action = (decision or {}).get("action") or (decision or {}).get("job_type")
    message = extract_error_message(payload)
    if not action or not message:
        return None
    lowered = message.lower()
    if "missing required" in lowered or "required input" in lowered:
        return {
            "retry_same_action_allowed": False,
            "blocked_actions": [action],
            "reason": message,
        }
    return {
        "retry_same_action_allowed": True,
        "blocked_actions": [],
        "reason": message,
    }


def is_failed_job_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    status = str(result.get("status") or result.get("job_status") or "").lower()
    if status in {"failed", "killed", "error"}:
        return True
    return bool(result.get("has_error") or result.get("error"))


def find_created_jobs(execution: dict[str, Any]) -> list[dict[str, Any]]:
    execution_result = execution.get("execution_result") or {}
    results = execution_result.get("execution_results") or []
    created = []
    for item in results:
        job_uid = item.get("job_uid")
        if job_uid:
            created.append(
                {
                    "job_uid": job_uid,
                    "job_type": item.get("job_type"),
                    "status": item.get("status"),
                    "queued": item.get("queued"),
                    "planned_action": item.get("planned_action"),
                }
            )
    return created


def collect_job_logs(job_uid: str) -> dict[str, Any]:
    roots = [
        Path("/home/share/cryoSPARC/P2") / job_uid,
        Path("/ssd1/linweifan/cryosparc/cryosparc_master/run") / job_uid,
    ]
    logs = {}
    for root in roots:
        if not root.exists():
            continue
        for name in ("job.log", "job.json", "events.bson"):
            path = root / name
            if path.exists() and path.is_file():
                try:
                    logs[str(path)] = path.read_text(errors="replace")[-200000:]
                except UnicodeDecodeError:
                    logs[str(path)] = "<binary log file>"
                except OSError as exc:
                    logs[str(path)] = f"<read failed: {exc}>"
    return logs


def make_run_dir(output_dir: Path) -> Path:
    run_dir = output_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Any, round_log: dict[str, Any] | None) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    if round_log is not None:
        round_log.setdefault("files", {})[path.name] = str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def print_run_summary(summary: dict[str, Any], run_dir: Path) -> None:
    print(f"Run directory: {run_dir}")
    for round_log in summary["rounds"]:
        print(f"[轮次 {round_log['round']}]")
        print(f"Model 输入：{round_log.get('files', {}).get('model_messages.json')}")
        print(f"Model 返回：{round_log.get('raw_model_output', '')[:500]}")
        validation = round_log.get("mcp_validation_tool", {}).get("result")
        print(f"JSON 校验：{json.dumps(validation, ensure_ascii=False, default=str)[:500]}")
        execution_tool = round_log.get("mcp_execution_tool", {})
        print(f"MCP 调用：{execution_tool.get('tool_name')}")
        print(f"cryoSPARC 结果：{json.dumps(round_log.get('job_results') or round_log.get('created_jobs') or round_log.get('mcp_execution_tool', {}).get('result'), ensure_ascii=False, default=str)[:500]}")
        print(f"下一轮原因：{round_log.get('next_round_reason')}")
        print(f"日志位置：{round_log.get('files')}")
    print(f"Stop reason: {summary['stop_reason']}")
    print(f"Summary: {run_dir / 'summary.json'}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
