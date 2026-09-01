# Run an autonomous model -> MCP -> CryoSPARC closed-loop test.
import argparse
import asyncio
import json
import re
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

from model_direct_runner import parse_model_decision_text


DEFAULT_BASE_MODEL = "/ssd1/lisongyang/models/Qwen3.6-27B-ms-test"
DEFAULT_ADAPTER = "/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow"
DEFAULT_MODEL_PYTHON = "/ssd1/linweifan/miniforge3/envs/cryoagent-model/bin/python"
DEFAULT_SERVER_PYTHON = "/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python"
DEFAULT_PROJECT_DIR = "/ssd1/linweifan/cryosparc_agent"
DEFAULT_MCP_SERVER = "cryosparc_mcp_server.py"

JOB_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "patch_ctf_estimation_multi": {
        "valid_inputs": [
            {
                "name": "exposures",
                "required": True,
                "accepted_source_outputs": [
                    "imported_micrographs",
                    "micrographs",
                    "exposures",
                ],
            }
        ]
    },
    "blob_picker_gpu": {
        "valid_inputs": [
            {
                "name": "micrographs",
                "required": True,
                "accepted_source_outputs": [
                    "imported_micrographs",
                    "micrographs",
                    "exposures",
                ],
            }
        ]
    },
    "template_picker_gpu": {
        "valid_inputs": [
            {
                "name": "templates",
                "required": True,
                "accepted_source_outputs": ["templates", "templates_selected"],
            },
            {
                "name": "micrographs",
                "required": True,
                "accepted_source_outputs": [
                    "imported_micrographs",
                    "micrographs",
                    "exposures",
                ],
            },
        ]
    },
    "inspect_picks_v2": {
        "valid_inputs": [
            {
                "name": "particles",
                "required": True,
                "accepted_source_outputs": ["particles"],
            },
            {
                "name": "micrographs",
                "required": True,
                "accepted_source_outputs": [
                    "imported_micrographs",
                    "micrographs",
                    "exposures",
                ],
            },
        ]
    },
    "extract_micrographs_multi": {
        "valid_inputs": [
            {
                "name": "micrographs",
                "required": True,
                "accepted_source_outputs": [
                    "imported_micrographs",
                    "micrographs",
                    "exposures",
                ],
            },
            {
                "name": "particles",
                "required": True,
                "accepted_source_outputs": ["particles"],
            },
        ]
    },
    "class_2D_new": {
        "valid_inputs": [
            {
                "name": "particles",
                "required": True,
                "accepted_source_outputs": ["particles", "particles_selected"],
            }
        ]
    },
    "select_2D": {
        "valid_inputs": [
            {
                "name": "particles",
                "required": True,
                "accepted_source_outputs": ["particles"],
            },
            {
                "name": "templates",
                "required": True,
                "accepted_source_outputs": ["class_averages", "templates"],
            },
        ]
    },
}

SYSTEM_PROMPT = (
    "You are the only decision maker for an autonomous CryoSPARC workflow test. "
    "Return exactly one JSON object. Do not include markdown, comments, or "
    "thinking text. Codex and MCP will not repair missing decisions for you."
)

STATIC_DECISION_INSTRUCTIONS = {
    "instruction": (
        "Choose exactly one next action, rollback, or stop from the current CryoSPARC state. "
        "Do not repeat completed Import Micrographs. If you choose a job, include every "
        "required input connection you want MCP to use. MCP will return validation or "
        "execution errors without repairing your decision."
    ),
    "output_contract": {
        "schema_version": "2.0",
        "decision_type": "forward | branch | rollback | stop",
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
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
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
        "model_paths": {"base_model": args.base_model, "adapter": args.adapter},
        "command": " ".join(sys.argv),
        "rounds": [],
        "stop_reason": None,
        "success": False,
    }

    server_params = StdioServerParameters(
        command=args.server_python,
        args=[str(Path(args.project_dir) / args.mcp_server)],
        cwd=args.project_dir,
    )

    current_node = args.current_node
    feedback_to_model: dict[str, Any] | None = None
    validation_failures = 0

    model_worker = ModelWorker(args, run_dir / "model_worker.stderr.log")
    model_worker.start()
    summary["model_worker"] = {
        "command": model_worker.command,
        "load_event": model_worker.load_event,
    }
    async with AsyncExitStack() as stack:
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
            )
            messages_file = round_dir / "model_messages.json"
            write_json(messages_file, messages, round_log)

            generation_file = round_dir / "model_generation.json"
            model_call = model_worker.generate(
                request_id=f"round_{round_index:02d}",
                messages=messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
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
            if decision_type in {"stop", "complete"}:
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
                job_result["local_job_logs"] = collect_job_logs(
                    job_result,
                    project_uid=args.project,
                    workspace_uid=args.workspace,
                )
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
    summary["success"] = summary["stop_reason"] in {"model_stop", "model_complete", "max_rounds_reached"}
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
) -> list[dict[str, str]]:
    dynamic_payload = {
        "round": round_index,
        "model_input": model_input,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": json.dumps(
                STATIC_DECISION_INSTRUCTIONS,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
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
    attempted_job_schema = build_attempted_job_schema(failed_decision)
    invalid_connections = extract_invalid_connections(
        payload,
        failed_decision,
        attempted_job_schema,
    )
    missing_inputs = extract_missing_inputs(
        payload,
        failed_decision,
        attempted_job_schema,
        invalid_connections,
    )
    execution_error = extract_execution_error(feedback_type, payload)
    return {
        "has_failure": True,
        "failure_round": feedback.get("round"),
        "failure_stage": map_failure_stage(feedback_type, payload),
        "failed_decision": failed_decision,
        "failed_action": extract_failed_action(failed_decision),
        "attempted_job": attempted_job_schema,
        "execution_error": execution_error,
        "missing_inputs": missing_inputs,
        "invalid_connections": invalid_connections,
        "available_inputs": extract_named_items(
            feedback.get("current_workflow_state"),
            ("available_inputs", "outputs", "input_sources"),
        ),
        "attempt_history": [
            {
                "round": feedback.get("round"),
                "action": (failed_decision or {}).get("action") or (failed_decision or {}).get("job_type"),
                "failure_stage": map_failure_stage(feedback_type, payload),
                "message": execution_error["raw_error_text"],
                "raw_error_text": execution_error["raw_error_text"],
                "source_used": execution_error["source_used"],
                "missing_inputs": missing_inputs,
                "invalid_connections": invalid_connections,
            }
        ],
    }


def map_failure_stage(feedback_type: str | None, payload: Any) -> str:
    if feedback_type == "json_parse_failed":
        return "json_parse"
    if feedback_type == "v2_validation_failed":
        return "validation"
    if feedback_type == "job_not_terminal":
        return "timeout"
    if feedback_type == "execution_failed_or_no_job":
        result = first_execution_result(payload)
        text = json.dumps(result or payload, ensure_ascii=False, default=str).lower()
        status = str((result or {}).get("status") or "").lower()
        if status == "approval_required":
            return "approval"
        if "inputs/" in text or "could not find" in text or "connect" in text:
            return "connect_inputs"
        if ":enqueue" in text or "queue" in text or "too_short" in text:
            return "queue"
        if "input" in text:
            return "connect_inputs"
        return "no_created_job"
    if feedback_type == "job_result":
        return "run"
    return feedback_type or "unknown"


def extract_failed_action(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if not decision:
        return None
    selected = first_selected_action(decision)
    return {
        "decision_type": decision.get("decision_type"),
        "action": decision.get("action") or (selected or {}).get("job_type"),
        "job_type": (
            decision.get("job_type")
            or decision.get("action")
            or (selected or {}).get("job_type")
        ),
        "parameters": decision.get("parameters") or (selected or {}).get("parameters") or {},
        "connections": decision.get("connections") or (selected or {}).get("connections") or {},
    }


def extract_execution_error(feedback_type: str | None, payload: Any) -> dict[str, Any]:
    primary = first_execution_result(payload) or payload
    raw_error = collect_raw_error_text(primary)
    return {
        "source": infer_error_source(feedback_type, primary),
        "message": raw_error["raw_error_text"],
        "raw_error_text": raw_error["raw_error_text"],
        "raw_error_lines": raw_error["raw_error_lines"],
        "source_used": raw_error["source_used"],
        "sources_checked": raw_error["sources_checked"],
        "log_identity_verified": raw_error["log_identity_verified"],
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


def extract_error_message(payload: Any) -> str:
    primary = first_execution_result(payload)
    if primary is not None and primary is not payload:
        return extract_error_message(primary)
    if isinstance(payload, dict):
        for key in ("error", "detail", "message", "stderr", "raw_text"):
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


def collect_raw_error_text(payload: Any) -> dict[str, Any]:
    sources_checked: list[str] = []
    candidates = raw_error_candidates(payload, sources_checked)
    for source, text, identity_verified in candidates:
        cleaned = trim_error_text(text)
        if cleaned:
            return {
                "raw_error_text": cleaned,
                "raw_error_lines": cleaned.splitlines(),
                "source_used": source,
                "sources_checked": dedupe_strings(sources_checked),
                "log_identity_verified": identity_verified,
            }
    return {
        "raw_error_text": "error_text_unavailable",
        "raw_error_lines": ["error_text_unavailable"],
        "source_used": "error_text_unavailable",
        "sources_checked": dedupe_strings(sources_checked),
        "log_identity_verified": False,
    }


def raw_error_candidates(
    payload: Any,
    sources_checked: list[str],
) -> list[tuple[str, str, bool]]:
    candidates: list[tuple[str, str, bool]] = []
    if isinstance(payload, dict):
        run_errors = payload.get("run_errors")
        if run_errors is not None:
            sources_checked.append("api.run_errors")
            for text in flatten_error_text(run_errors):
                candidates.append(("api.run_errors", text, True))

        local_logs = payload.get("local_job_logs")
        if isinstance(local_logs, dict):
            status = local_logs.get("status")
            sources_checked.append(f"local_job_logs.{status or 'unknown'}")
            if status == "ok" and isinstance(local_logs.get("files"), dict):
                for path, text in local_logs["files"].items():
                    source = log_source_name(str(path))
                    sources_checked.append(source)
                    for error_text in extract_relevant_error_sections(str(text)):
                        candidates.append((source, error_text, True))
            elif status == "stale_or_foreign_job_log":
                sources_checked.append("local_job_logs.stale_or_foreign_job_log")

        for key in ("stderr", "raw_text", "error", "detail", "message"):
            if key in payload:
                sources_checked.append(key)
                for text in flatten_error_text(payload.get(key)):
                    candidates.append((key, text, False))

        primary = first_execution_result(payload)
        if primary is not None and primary is not payload:
            candidates.extend(raw_error_candidates(primary, sources_checked))
        else:
            for key, value in payload.items():
                if key in {"run_errors", "local_job_logs", "raw"}:
                    continue
                if isinstance(value, (dict, list)):
                    candidates.extend(raw_error_candidates(value, sources_checked))
    elif isinstance(payload, list):
        for value in payload:
            candidates.extend(raw_error_candidates(value, sources_checked))
    return candidates


def flatten_error_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: list[str] = []
        for nested in value.values():
            texts.extend(flatten_error_text(nested))
        if not texts and value:
            return [json.dumps(value, ensure_ascii=False, default=str)]
        return texts
    if isinstance(value, list):
        if all(isinstance(nested, str) for nested in value):
            return ["\n".join(value)]
        texts: list[str] = []
        for nested in value:
            texts.extend(flatten_error_text(nested))
        return texts
    return [str(value)]


def extract_relevant_error_sections(text: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines()
    error_markers = (
        "traceback",
        "error",
        "exception",
        "failed",
        "stderr",
        "runtimeerror",
        "valueerror",
    )
    matching_indexes = [
        index
        for index, line in enumerate(lines)
        if any(marker in line.lower() for marker in error_markers)
    ]
    if not matching_indexes:
        return []
    sections = []
    for index in matching_indexes:
        start = max(0, index - 20)
        end = min(len(lines), index + 80)
        sections.append("\n".join(lines[start:end]))
    return dedupe_strings(sections)


def trim_error_text(text: str, max_chars: int = 20000, max_lines: int = 300) -> str:
    lines = [line.rstrip() for line in str(text).splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    trimmed = "\n".join(lines[-max_lines:])
    return trimmed[-max_chars:]


def log_source_name(path: str) -> str:
    name = Path(path).name
    if name in {"job.log", "job.json", "events.bson"}:
        return f"verified_log.{name}"
    if "stderr" in name.lower():
        return f"verified_log.{name}"
    return f"verified_log.{name or 'unknown'}"


def dedupe_strings(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


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


def build_attempted_job_schema(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    failed_action = extract_failed_action(decision)
    job_type = (failed_action or {}).get("job_type")
    if not job_type:
        return None
    schema = JOB_INPUT_SCHEMAS.get(job_type, {"valid_inputs": []})
    valid_inputs = [
        {
            "name": item["name"],
            "required": bool(item.get("required")),
            "accepted_source_outputs": item.get("accepted_source_outputs", []),
        }
        for item in schema.get("valid_inputs", [])
    ]
    return {
        "job_type": job_type,
        "valid_inputs": valid_inputs,
        "required_inputs": [
            item["name"]
            for item in valid_inputs
            if item.get("required")
        ],
    }


def extract_invalid_connections(
    payload: Any,
    decision: dict[str, Any] | None,
    attempted_job_schema: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    invalid.extend(
        item
        for item in extract_named_items(payload, ("invalid_connections", "connection_errors"))
        if isinstance(item, dict)
    )
    failed_action = extract_failed_action(decision) or {}
    connections = failed_action.get("connections") or {}
    valid_inputs = valid_input_names(attempted_job_schema)
    valid_input_specs = valid_input_specs_by_name(attempted_job_schema)

    for input_name, connection in connections.items():
        if valid_inputs and input_name not in valid_inputs:
            invalid.append(
                {
                    "target_input": input_name,
                    "reason": (
                        f"{failed_action.get('job_type')} has no input named "
                        f"{input_name!r}."
                    ),
                    "provided_connection": connection,
                    "valid_target_inputs": sorted(valid_inputs),
                }
            )
            continue
        spec = valid_input_specs.get(input_name) or {}
        accepted = set(spec.get("accepted_source_outputs") or [])
        for value in normalize_connection_values_for_context(connection):
            source_output = value.get("source_output")
            if accepted and source_output and source_output not in accepted:
                invalid.append(
                    {
                        "target_input": input_name,
                        "reason": (
                            f"source_output {source_output!r} is not listed "
                            f"as accepted for input {input_name!r}."
                        ),
                        "provided_connection": value,
                        "accepted_source_outputs": sorted(accepted),
                    }
                )

    message = extract_error_message(payload)
    missing_input_name = parse_missing_input_name(message)
    if missing_input_name and (
        not valid_inputs or missing_input_name not in valid_inputs
    ):
        invalid.append(
            {
                "target_input": missing_input_name,
                "reason": (
                    f"CryoSPARC reported that the attempted job has no input "
                    f"named {missing_input_name!r}."
                ),
                "valid_target_inputs": sorted(valid_inputs),
            }
        )
    return dedupe_dicts(invalid)


def extract_missing_inputs(
    payload: Any,
    decision: dict[str, Any] | None,
    attempted_job_schema: dict[str, Any] | None,
    invalid_connections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    missing.extend(
        item
        for item in extract_named_items(payload, ("missing_inputs", "missing_required_inputs"))
        if isinstance(item, dict)
    )
    failed_action = extract_failed_action(decision) or {}
    connections = failed_action.get("connections") or {}
    valid_inputs = valid_input_names(attempted_job_schema)
    invalid_names = {
        item.get("target_input")
        for item in invalid_connections
        if item.get("target_input")
    }
    connected_valid_inputs = {
        name
        for name in connections
        if name in valid_inputs and name not in invalid_names
    }
    for spec in (attempted_job_schema or {}).get("valid_inputs", []):
        name = spec.get("name")
        if spec.get("required") and name not in connected_valid_inputs:
            missing.append(
                {
                    "target_input": name,
                    "required": True,
                    "accepted_source_outputs": spec.get("accepted_source_outputs", []),
                    "reason": "Required input has no valid connection in the failed action.",
                }
            )

    for input_name in parse_too_short_inputs(extract_error_message(payload)):
        spec = valid_input_specs_by_name(attempted_job_schema).get(input_name, {})
        missing.append(
            {
                "target_input": input_name,
                "required": True,
                "accepted_source_outputs": spec.get("accepted_source_outputs", []),
                "reason": "CryoSPARC validation reported an empty required connection list.",
            }
        )
    return dedupe_dicts(missing)


def build_retry_guidance(
    decision: dict[str, Any] | None,
    payload: Any,
    attempted_job_schema: dict[str, Any] | None,
    missing_inputs: list[dict[str, Any]],
    invalid_connections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    failed_action = extract_failed_action(decision) or {}
    action = failed_action.get("action") or failed_action.get("job_type")
    message = extract_error_message(payload)
    if not action or not message:
        return None
    return {
        "action_observed": action,
        "job_type_observed": failed_action.get("job_type"),
        "failure_stage_observed": map_failure_stage("execution_failed_or_no_job", payload),
        "objective_blocking_error": message,
        "valid_inputs_observed": (attempted_job_schema or {}).get("valid_inputs", []),
        "invalid_connections_observed": invalid_connections,
        "missing_inputs_observed": missing_inputs,
        "note": (
            "This field only reports observed blockers, valid input names, and "
            "missing or invalid connections. It does not recommend retrying, "
            "changing action, rolling back, or stopping."
        ),
    }


def first_selected_action(decision: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(decision, dict):
        return None
    selected = decision.get("selected_actions")
    if isinstance(selected, list) and selected and isinstance(selected[0], dict):
        return selected[0]
    return None


def valid_input_names(attempted_job_schema: dict[str, Any] | None) -> set[str]:
    return {
        item.get("name")
        for item in (attempted_job_schema or {}).get("valid_inputs", [])
        if item.get("name")
    }


def valid_input_specs_by_name(
    attempted_job_schema: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    return {
        item["name"]: item
        for item in (attempted_job_schema or {}).get("valid_inputs", [])
        if item.get("name")
    }


def normalize_connection_values_for_context(raw_value: Any) -> list[dict[str, str]]:
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    normalized = []
    for value in values:
        if isinstance(value, dict):
            source_job = (
                value.get("source_job_uid")
                or value.get("source_job")
                or value.get("job_uid")
            )
            source_output = value.get("source_output") or value.get("output")
            normalized.append(
                {
                    "source_job_uid": str(source_job) if source_job else "",
                    "source_output": str(source_output) if source_output else "",
                }
            )
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            normalized.append(
                {
                    "source_job_uid": str(value[0]),
                    "source_output": str(value[1]),
                }
            )
    return normalized


def parse_missing_input_name(message: str) -> str | None:
    patterns = [
        r'input\s+\\?"([^"\\]+)\\?"',
        r"input\s+'([^']+)'",
        r"inputs/([A-Za-z0-9_]+)",
        r'"inputs",\s*"([A-Za-z0-9_]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return None


def parse_too_short_inputs(message: str) -> list[str]:
    names: list[str] = []
    names.extend(re.findall(r'"inputs",\s*"([A-Za-z0-9_]+)"', message))
    names.extend(re.findall(r"inputs/([A-Za-z0-9_]+)", message))
    return sorted(set(names))


def dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def first_execution_result(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    execution_result = payload.get("execution_result")
    if isinstance(execution_result, dict):
        results = execution_result.get("execution_results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                return first
    results = payload.get("execution_results")
    if isinstance(results, list) and results:
        first = results[0]
        if isinstance(first, dict):
            return first
    return None


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


def collect_job_logs(
    job_result: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
) -> dict[str, Any]:
    expected = expected_job_identity(job_result, project_uid, workspace_uid)
    job_dir = resolve_api_job_dir(job_result)
    if not job_dir:
        return {
            "status": "no_verified_job_dir",
            "reason": "Current CryoSPARC API result did not expose an absolute job_dir.",
            "expected_identity": expected,
        }
    if not job_dir.exists() or not job_dir.is_dir():
        return {
            "status": "job_dir_not_found",
            "job_dir": str(job_dir),
            "expected_identity": expected,
        }

    job_json_path = job_dir / "job.json"
    if not job_json_path.exists():
        return {
            "status": "job_json_missing",
            "job_dir": str(job_dir),
            "expected_identity": expected,
        }

    try:
        job_json = json.loads(job_json_path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "job_json_unreadable",
            "job_dir": str(job_dir),
            "job_json_path": str(job_json_path),
            "error": str(exc),
            "expected_identity": expected,
        }

    validation = validate_job_log_identity(job_json, expected)
    if not validation["valid"]:
        return {
            "status": "stale_or_foreign_job_log",
            "job_dir": str(job_dir),
            "job_json_path": str(job_json_path),
            "expected_identity": expected,
            "observed_identity": validation["observed_identity"],
            "mismatches": validation["mismatches"],
        }

    logs = {}
    for path in iter_verified_log_paths(job_dir):
        if path.exists() and path.is_file():
            try:
                logs[str(path)] = path.read_text(errors="replace")[-200000:]
            except UnicodeDecodeError:
                logs[str(path)] = "<binary log file>"
            except OSError as exc:
                logs[str(path)] = f"<read failed: {exc}>"
    return {
        "status": "ok",
        "job_dir": str(job_dir),
        "expected_identity": expected,
        "files": logs,
    }


def iter_verified_log_paths(job_dir: Path) -> list[Path]:
    names = ["job.log", "job.json", "events.bson"]
    paths = [job_dir / name for name in names]
    for pattern in ("*.err", "*.stderr", "*stderr*", "slurm-*.out", "*.out"):
        paths.extend(sorted(job_dir.glob(pattern)))
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_api_job_dir(job_result: dict[str, Any]) -> Path | None:
    runtime = job_result.get("runtime") or {}
    raw_job_dir = runtime.get("work_dir") or runtime.get("job_dir")
    raw_project_dir = runtime.get("project_dir") or runtime.get("project_path")
    if not raw_job_dir:
        last_node_info = (
            (job_result.get("current_state") or {}).get("last_node_info") or {}
        )
        last_runtime = last_node_info.get("runtime") or {}
        raw_job_dir = (
            last_runtime.get("work_dir")
            or last_runtime.get("job_dir")
        )
        raw_project_dir = raw_project_dir or last_runtime.get("project_dir")
        raw_project_dir = raw_project_dir or last_runtime.get("project_path")
    if not raw_job_dir:
        return None
    path = Path(str(raw_job_dir)).expanduser()
    if path.is_absolute():
        return path
    if raw_project_dir:
        project_dir = Path(str(raw_project_dir)).expanduser()
        if project_dir.is_absolute():
            return project_dir / path
    if not path.is_absolute():
        return None
    return path


def expected_job_identity(
    job_result: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
) -> dict[str, Any]:
    timestamps = job_result.get("timestamps") or {}
    return {
        "uid": job_result.get("job_uid"),
        "project_uid": job_result.get("project_uid") or project_uid,
        "workspace_uid": job_result.get("workspace_uid") or workspace_uid,
        "job_type": job_result.get("job_type"),
        "created_at": normalize_timestamp(timestamps.get("created_at")),
    }


def validate_job_log_identity(
    job_json: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    observed = {
        "uid": job_json.get("uid"),
        "project_uid": job_json.get("project_uid"),
        "workspace_uid": observed_workspace_uid(job_json),
        "job_type": job_json.get("job_type") or job_json.get("type"),
        "created_at": normalize_timestamp(job_json.get("created_at")),
    }
    mismatches = []
    for key in ("uid", "project_uid", "workspace_uid", "job_type", "created_at"):
        if expected.get(key) and observed.get(key) != expected.get(key):
            mismatches.append(
                {
                    "field": key,
                    "expected": expected.get(key),
                    "observed": observed.get(key),
                }
            )
    return {
        "valid": not mismatches,
        "observed_identity": observed,
        "mismatches": mismatches,
    }


def observed_workspace_uid(job_json: dict[str, Any]) -> str | None:
    workspace_uid = job_json.get("workspace_uid")
    if workspace_uid:
        return str(workspace_uid)
    workspace_uids = job_json.get("workspace_uids")
    if isinstance(workspace_uids, list) and len(workspace_uids) == 1:
        return str(workspace_uids[0])
    return None


def normalize_timestamp(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, dict) and "$date" in value:
        value = value["$date"]
    if not isinstance(value, str):
        return str(value)
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


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
