# Strict Schema V24 validation-only model -> MCP benchmark runner.
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_core import (
    append_event,
    build_manifest,
    build_v24_messages,
    compute_metrics,
    evaluate_model_round,
    sha256_json,
    write_json,
)
from model_clients import ModelClientConfig, create_model_client, write_model_client_log
from run_context_guard import RunContextGuard


DEFAULT_BASE_MODEL = "/ssd1/lisongyang/models/Qwen3.6-27B-ms-test"
DEFAULT_ADAPTER = "/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow"
DEFAULT_SERVER_PYTHON = "/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python"
DEFAULT_PROJECT_DIR = "/ssd1/linweifan/cryosparc_agent"
DEFAULT_MCP_SERVER = "cryosparc_mcp_server.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--current-node")
    parser.add_argument("--dataset-json", default="{}")
    parser.add_argument("--dataset-json-file")
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--model-provider", choices=["local", "openai", "dashscope_qwen"], required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--finetune-checkpoint")
    parser.add_argument("--openai-model")
    parser.add_argument("--qwen-model")
    parser.add_argument("--dashscope-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--dashscope-native-url", default="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation")
    parser.add_argument("--dashscope-api-mode", choices=["compatible", "native"], default="compatible")
    parser.add_argument("--qwen-response-format", choices=["json_object", "json_schema"], default="json_object")
    parser.add_argument("--api-timeout-seconds", type=int, default=120)
    parser.add_argument("--server-python", default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--mcp-server", default=DEFAULT_MCP_SERVER)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--output-dir", default="reports/v24_benchmarks")
    parser.add_argument("--validation-only", action="store_true", default=True)
    parser.add_argument("--create-real-job", action="store_true", default=False)
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    if args.create_real_job:
        raise SystemExit("Refusing to create real CryoSPARC jobs in strict V24 benchmark runner.")
    dataset_info = load_dataset_info(args)
    run_id = make_run_id(args)
    run_dir = Path(args.output_dir) / args.benchmark_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events_file = run_dir / "events.jsonl"
    model_log_file = run_dir / "model_client.log"
    (run_dir / "launcher.log").write_text("strict_v24_benchmark_runner started\n", encoding="utf-8")

    config_payload = vars(args).copy()
    config_payload["dataset_json"] = dataset_info
    config_hash = sha256_json(config_payload)
    dataset_hash = sha256_json(dataset_info)
    commit = git_commit(Path.cwd())
    dirty = git_dirty(Path.cwd())

    client = create_model_client(
        ModelClientConfig(
            provider=args.model_provider,
            model_label=args.model_label,
            temperature=args.temperature,
            seed=args.seed,
            api_timeout_seconds=args.api_timeout_seconds,
            base_model=args.base_model,
            adapter=args.adapter,
            openai_model=args.openai_model,
            qwen_model=args.qwen_model,
            dashscope_base_url=args.dashscope_base_url,
            dashscope_native_url=args.dashscope_native_url,
            dashscope_api_mode=args.dashscope_api_mode,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            qwen_response_format=args.qwen_response_format,
            finetune_checkpoint=args.finetune_checkpoint,
        )
    )
    guard = RunContextGuard(
        run_id=run_id,
        project_uid=args.project,
        workspace_uid=args.workspace,
        start_node=args.current_node,
    )

    server_params = StdioServerParameters(
        command=args.server_python,
        args=[str(Path(args.project_dir) / args.mcp_server)],
        cwd=args.project_dir,
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "benchmark_id": args.benchmark_id,
        "provider": args.model_provider,
        "model_label": args.model_label,
        "validation_only": True,
        "created_real_jobs": [],
        "rounds": [],
        "stop_reason": None,
    }
    current_node = args.current_node

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        for round_index in range(1, args.max_rounds + 1):
            round_dir = run_dir / f"round_{round_index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            model_input = await call_tool_json(
                session,
                "get_workflow_decision_context",
                {
                    "project_uid": args.project,
                    "workspace_uid": args.workspace,
                    "current_job_uid": current_node,
                    "dataset_info": dataset_info,
                    "known_workflow_dirs": None,
                },
            )
            write_json(round_dir / "model_input.json", model_input)
            isolation = guard.assert_model_input(model_input)
            write_json(round_dir / "isolation.json", isolation)
            if not isolation["success"]:
                append_event(events_file, run_id, round_index, "round_stopped", status="context_isolation_failed", error_category="context_isolation_failed", source_file=str(round_dir / "isolation.json"))
                summary["stop_reason"] = "context_isolation_failed"
                break
            if model_input.get("internal_only") or model_input.get("ready_for_model") is False:
                write_json(round_dir / "job_status.json", model_input)
                category = "approval_required" if model_input.get("human_action_required") else None
                append_event(events_file, run_id, round_index, "job_status", status=model_input.get("status_group") or model_input.get("status"), error_category=category, source_file=str(round_dir / "job_status.json"))
                summary["stop_reason"] = model_input.get("status_group") or "current_job_not_terminal"
                break

            messages = build_v24_messages(model_input, round_index)
            message_isolation = guard.assert_messages(messages)
            write_json(round_dir / "model_messages.json", messages)
            write_json(round_dir / "message_isolation.json", message_isolation)
            if not message_isolation["success"]:
                append_event(events_file, run_id, round_index, "round_stopped", status="context_isolation_failed", error_category="context_isolation_failed", source_file=str(round_dir / "message_isolation.json"))
                summary["stop_reason"] = "context_isolation_failed"
                break

            prompt_hash = sha256_json(messages)
            if round_index == 1:
                manifest = build_manifest(args, run_id, prompt_hash, dataset_hash, config_hash, dirty, commit)
                write_json(run_dir / "benchmark_manifest.json", manifest)
                write_json(run_dir / "run_config.json", config_payload)

            append_event(events_file, run_id, round_index, "model_request", status="sent", source_file=str(round_dir / "model_messages.json"))
            model_call = client.generate(messages, request_id=f"{run_id}-round-{round_index:03d}")
            write_json(round_dir / "model_generation.json", model_call)
            write_model_client_log(model_log_file, {"round": round_index, **redact_model_call_for_log(model_call)})
            usage = model_call.get("usage") or {}
            append_event(
                events_file,
                run_id,
                round_index,
                "model_response",
                model_id=model_call.get("exact_model_id"),
                status="success" if model_call.get("success") else "api_failure",
                error_category=None if model_call.get("success") else "api_failure",
                latency=model_call.get("latency"),
                input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
                output_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
                source_file=str(round_dir / "model_generation.json"),
            )
            if not model_call.get("success"):
                summary["stop_reason"] = "api_failure"
                break

            workflow_state = await call_tool_json(
                session,
                "get_workflow_state",
                {"project_uid": args.project, "workspace_uid": args.workspace},
            )
            write_json(round_dir / "workflow_state.json", workflow_state)
            evaluation = evaluate_model_round(
                model_call.get("raw_text", ""),
                workflow_state=workflow_state,
                guard=guard,
            )
            write_round_evidence(round_dir, evaluation)
            append_event(
                events_file,
                run_id,
                round_index,
                "parse_result",
                status="success" if evaluation.get("parse_result", {}).get("success") else "failed",
                error_category=evaluation.get("error_category") if evaluation.get("stage") == "parse" else None,
                source_file=str(round_dir / "parse_result.json"),
            )
            if evaluation.get("validation"):
                append_event(
                    events_file,
                    run_id,
                    round_index,
                    "validation_result",
                    action=first_action(evaluation.get("decision")),
                    status="success" if evaluation["validation"].get("success") else "failed",
                    error_category=evaluation.get("error_category") if evaluation.get("stage") in {"schema_validation", "action_validation"} else None,
                    source_file=str(round_dir / "validation.json"),
                )
            if evaluation.get("connection_result"):
                append_event(
                    events_file,
                    run_id,
                    round_index,
                    "connection_result",
                    action=first_action(evaluation.get("decision")),
                    status=evaluation["connection_result"].get("status"),
                    error_category=evaluation.get("error_category") if evaluation.get("stage") == "connection" else None,
                    source_file=str(round_dir / "connection_result.json"),
                )

            summary["rounds"].append({"round": round_index, "stage": evaluation.get("stage"), "success": evaluation.get("success")})
            if evaluation.get("stage") == "stop":
                summary["stop_reason"] = "model_stop"
                append_event(events_file, run_id, round_index, "round_stopped", status="model_stop", source_file=str(round_dir / "model_decision.json"))
                break
            summary["stop_reason"] = evaluation.get("stage")
            append_event(events_file, run_id, round_index, "round_stopped", status=evaluation.get("stage"), error_category=evaluation.get("error_category"), source_file=str(round_dir / "summary.json"))
            break
        else:
            summary["stop_reason"] = "max_rounds_reached"

    append_event(events_file, run_id, 0, "run_stopped", status=summary["stop_reason"], source_file=str(run_dir / "summary.json"))
    metrics = compute_metrics(run_dir)
    summary["metrics"] = metrics
    write_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, ensure_ascii=False, indent=2, default=str))


def load_dataset_info(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_json_file:
        return json.loads(Path(args.dataset_json_file).read_text(encoding="utf-8"))
    return json.loads(args.dataset_json)


def make_run_id(args: argparse.Namespace) -> str:
    return f"{args.model_label}_r{args.repeat_index}_{uuid.uuid4().hex[:10]}"


async def call_tool_json(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await session.call_tool(tool_name, arguments)
    if result.isError:
        return {"success": False, "mcp_error": True, "tool_name": tool_name, "content": [content.model_dump() for content in result.content]}
    if not result.content:
        return {"success": False, "tool_name": tool_name, "content": []}
    text = getattr(result.content[0], "text", None)
    if text is None:
        return {"success": False, "tool_name": tool_name, "content": [content.model_dump() for content in result.content]}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"success": False, "tool_name": tool_name, "raw_text": text}


def write_round_evidence(round_dir: Path, evaluation: dict[str, Any]) -> None:
    write_json(round_dir / "parse_result.json", evaluation.get("parse_result"))
    if evaluation.get("decision") is not None:
        write_json(round_dir / "model_decision.json", evaluation["decision"])
    if evaluation.get("validation") is not None:
        write_json(round_dir / "validation.json", evaluation["validation"])
    if evaluation.get("connection_result") is not None:
        write_json(round_dir / "connection_result.json", evaluation["connection_result"])
    write_json(round_dir / "execution.json", {"validation_only": True, "created_real_job": False})
    write_json(round_dir / "job_results.json", {"validation_only": True, "created_real_jobs": []})
    write_json(round_dir / "summary.json", {"stage": evaluation.get("stage"), "success": evaluation.get("success"), "error_category": evaluation.get("error_category")})


def first_action(decision: dict[str, Any] | None) -> str | None:
    if not decision:
        return None
    actions = decision.get("selected_actions") or []
    if not actions:
        return None
    return actions[0].get("job_type")


def redact_model_call_for_log(model_call: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": model_call.get("success"),
        "provider": model_call.get("provider"),
        "model_label": model_call.get("model_label"),
        "exact_model_id": model_call.get("exact_model_id"),
        "request_id": model_call.get("request_id"),
        "usage": model_call.get("usage"),
        "latency": model_call.get("latency"),
        "error": model_call.get("error"),
    }


def git_commit(cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
    except Exception:
        return None


def git_dirty(cwd: Path) -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], cwd=cwd, text=True).strip())
    except Exception:
        return True


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
