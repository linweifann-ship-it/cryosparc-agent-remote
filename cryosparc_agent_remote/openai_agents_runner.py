# OpenAI Agents SDK closed-loop runner over the existing cryoSPARC MCP server.
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import uuid
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_BASE_URL = "https://api.ofox.ai/v1"
DEFAULT_SERVER_PYTHON = "/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python"
DEFAULT_PROJECT_DIR = "/ssd1/linweifan/cryosparc_agent"
DEFAULT_MCP_SERVER = "cryosparc_mcp_server.py"

STATIC_AGENT_INSTRUCTIONS = (
    "You are the only workflow and scientific decision maker for an autonomous CryoSPARC "
    "workflow. MCP provides state, validation, execution, observations, and errors; use its "
    "tools rather than asking the runner to infer a workflow step. Use only current MCP state "
    "or tool evidence: never invent jobs, outputs, metrics, visual observations, connections, "
    "rollback targets, or execution results. Return a complete valid MCP v3.0 minimal decision; "
    "do not assume MCP will repair a decision. Dynamic project, workspace, node, job, and run "
    "facts belong only in the user payload."
)

STATIC_OUTPUT_CONTRACT = {
    "schema_version": "3.0",
    "decision_type": "forward | branch | stop",
    "selected_actions": [
        {
            "job_type": "CryoSPARC job type selected from MCP-visible candidates.",
            "parameters": "Only non-default parameters explicitly chosen by the model.",
        }
    ],
    "stop_shape": {
        "schema_version": "3.0",
        "decision_type": "stop",
        "selected_actions": [],
    },
}

# This entire object is independent of a particular cryoSPARC workspace.  It
# is deliberately the second fixed system layer: its final text content block
# is the explicit cache boundary immediately before per-step state.
STATIC_MCP_PROTOCOL = {
    "required_mcp_sequence": [
        "get_workflow_decision_context",
        "validate_v2_model_decision when you have a candidate decision",
        "execute_v2_model_decision with dry_run=false for executable decisions",
        "wait_for_job_result_package for created jobs",
    ],
    "output_contract": STATIC_OUTPUT_CONTRACT,
    "decision_rules": [
        "Use forward for one evidence-supported next action, branch only for a meaningful scientific comparison, and stop when no safe justified action remains.",
        "selected_actions must contain only MCP-visible executable job types; parameters contain only non-default overrides.",
        "MCP validates source outputs, input compatibility, and execution. Do not claim a connection or job result not returned by MCP.",
        "For missing evidence or an unavailable safe action, obtain the relevant MCP state or stop; this v3 contract has no request_input or rollback decision type.",
    ],
    "interactive_qc_rules": {
        "inspect_picks": "Use supplied micrograph or pick overlays, NCC or CC, power, score distributions, particle counts, and other available evidence. Adjust or accept thresholds only when evidence supports it; do not claim visual inspection without supplied visual evidence.",
        "select_2d": "Use supplied class averages, class counts, and statistics to retain structurally suitable classes and reject clear junk, contamination, aggregation, background, or misalignment. Multiple Select 2D steps are allowed when MCP exposes them.",
        "human_review": "A traditionally manual step is not automatically human-only when sufficient machine-readable or visual evidence is supplied. Respect MCP-required human gates; otherwise stop rather than guess when evidence is insufficient.",
    },
    "failure_and_science_rules": [
        "Base recovery only on real MCP execution results or observations. Do not repeat an unchanged failed action without new evidence or a legal changed parameter or input.",
        "Respect scientific dependencies: CTF estimation normally precedes CTF-dependent picking; assess available pick QC before large extraction; assess available 2D class QC before 3D reconstruction; refinement requires a valid volume and compatible particles.",
        "Completed means computation succeeded, not scientific quality. Weigh metrics, images, counts, and observations together; particle count alone is not quality, and clearly poor 2D classes must not be retained merely to increase count.",
        "Inspect Picks and Select 2D are high-value QC checkpoints, not unconditional gates when existing evidence supports another scientifically justified path.",
    ],
}

CLOSED_LOOP_MCP_TOOLS = [
    "get_workflow_decision_context",
    "validate_v2_model_decision",
    "execute_v2_model_decision",
    "wait_for_job_result_package",
]


@dataclass
class AgentsRunConfig:
    project_uid: str
    workspace_uid: str
    start_node: str | None
    model: str
    base_url: str
    api_key: str
    run_id: str
    output_dir: Path
    dataset_info: dict[str, Any]
    known_workflow_dirs: list[str]
    max_steps: int
    max_turns_per_step: int
    wait_timeout_seconds: int
    poll_interval_seconds: int
    server_python: str
    project_dir: Path
    mcp_server: str
    mcp_stdio_command: list[str] | None
    use_responses_api: bool


def build_step_input(
    config: AgentsRunConfig, current_node: str | None, step: int
) -> list[dict[str, Any]]:
    """Build two stable system layers followed by the dynamic workflow state."""
    payload = {
        "run_scope": {
            "project_uid": config.project_uid,
            "workspace_uid": config.workspace_uid,
            "current_node_id": current_node,
            "step": step,
        },
        "mcp_arguments": {
            "project_uid": config.project_uid,
            "workspace_uid": config.workspace_uid,
            "current_job_uid": current_node,
            "dataset_info": config.dataset_info,
            "known_workflow_dirs": config.known_workflow_dirs or None,
            "wait_timeout_seconds": config.wait_timeout_seconds,
            "poll_interval_seconds": config.poll_interval_seconds,
        },
    }
    static_content_type = "input_text" if config.use_responses_api else "text"
    return [
        {"role": "system", "content": STATIC_AGENT_INSTRUCTIONS},
        {
            "role": "system",
            "content": [
                {
                    "type": static_content_type,
                    "text": json.dumps(
                        STATIC_MCP_PROTOCOL,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "role": "user",
            "content": json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        },
    ]


async def run_agents_closed_loop(config: AgentsRunConfig) -> dict[str, Any]:
    Agent, Runner, ModelSettings, RunConfig, ResponsesModel, ChatModel, AsyncOpenAI, _function_tool = import_agents_core()
    MCPServerStdio, create_static_tool_filter = import_mcp_stdio()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "model_requests.jsonl", {"event": "run_started", "run_id": config.run_id})

    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    model_obj: Any = (
        ResponsesModel(model=config.model, openai_client=client)
        if config.use_responses_api
        else ChatModel(model=config.model, openai_client=client)
    )
    api_mode = "responses" if config.use_responses_api else "chat_completions"

    mcp_params = build_mcp_stdio_params(config)
    mcp_server = MCPServerStdio(
        params=mcp_params,
        cache_tools_list=True,
        tool_filter=create_static_tool_filter(allowed_tool_names=CLOSED_LOOP_MCP_TOOLS),
        name="existing-cryoSPARC-mcp",
        client_session_timeout_seconds=None,
    )
    agent = Agent(
        name="cryoSPARC Agents SDK loop",
        instructions=None,
        mcp_servers=[mcp_server],
        model=model_obj,
        model_settings=ModelSettings(
            max_tokens=2048,
            include_usage=True,
            # The SDK maps this field to the provider's native cache options.
            # Keep the provider-specific cache key in extra_body.
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
            preserve_raw_usage=True,
            extra_body={
                "prompt_cache_key": f"cryoagent:{config.project_uid}:{config.workspace_uid}:agents-sdk-v1",
            },
        ),
    )
    run_config = RunConfig(tracing_disabled=True)

    summary: dict[str, Any] = {
        "run_id": config.run_id,
        "created_at": utc_now(),
        "project_uid": config.project_uid,
        "workspace_uid": config.workspace_uid,
        "start_node": config.start_node,
        "model": config.model,
        "base_url": config.base_url,
        "api_mode": api_mode,
        "mcp_server": str(config.project_dir / config.mcp_server),
        "steps": [],
        "created_jobs": [],
        "usage": [],
        "success": False,
        "stop_reason": None,
    }

    current_node = config.start_node
    await mcp_server.connect()
    try:
        tools = await mcp_server.list_tools()
        summary["mcp_tools"] = sorted(tool.name for tool in tools)
        write_jsonl(config.output_dir / "mcp_calls.jsonl", {"event": "tools_listed", "tools": summary["mcp_tools"]})
        for step in range(1, config.max_steps + 1):
            user_input = build_step_input(config, current_node, step)
            request_event = {"step": step, "input": user_input, "api_mode": api_mode}
            write_jsonl(config.output_dir / "model_requests.jsonl", request_event)
            result = await Runner.run(
                agent,
                input=user_input,
                max_turns=config.max_turns_per_step,
                run_config=run_config,
            )
            event = serialize_run_result(result)
            event["step"] = step
            write_jsonl(config.output_dir / "model_responses.jsonl", event)
            append_tool_events(config.output_dir, step, event)
            summary["steps"].append(summarize_step_event(event))
            summary["usage"].append(extract_usage(event))

            created_jobs = extract_created_jobs(event)
            summary["created_jobs"].extend(created_jobs)
            for job in created_jobs:
                current_node = job.get("job_uid") or current_node
                write_jsonl(config.output_dir / "jobs.jsonl", {"step": step, **job})

            observations = extract_observations(event)
            for observation in observations:
                write_jsonl(config.output_dir / "observations.jsonl", {"step": step, "observation": observation})

            final_text = str(event.get("final_output") or "")
            if contains_stop(final_text):
                summary["stop_reason"] = "model_stop"
                summary["success"] = True
                break
            if not created_jobs:
                summary["stop_reason"] = "no_created_job"
                break
        else:
            summary["stop_reason"] = "max_steps_reached"
            summary["success"] = True
    finally:
        await mcp_server.cleanup()

    summary["finished_at"] = utc_now()
    summary["prompt_cache"] = summarize_prompt_cache(
        [
            usage
            for step_usage in summary["usage"]
            for usage in step_usage.get("usage_records", [])
        ]
    )
    write_json(config.output_dir / "summary.json", summary)
    return summary


def build_mcp_stdio_params(config: AgentsRunConfig) -> dict[str, Any]:
    if config.mcp_stdio_command:
        return {
            "command": config.mcp_stdio_command[0],
            "args": config.mcp_stdio_command[1:],
            "cwd": str(Path.cwd()),
            "env": os.environ.copy(),
        }
    return {
        "command": config.server_python,
        "args": [str(config.project_dir / config.mcp_server)],
        "cwd": str(config.project_dir),
        "env": os.environ.copy(),
    }


async def smoke_api(config: AgentsRunConfig) -> dict[str, Any]:
    Agent, Runner, ModelSettings, RunConfig, ResponsesModel, ChatModel, AsyncOpenAI, function_tool = import_agents_core()
    client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
    modes = []
    errors = []

    @function_tool
    def echo_tool(value: str) -> dict[str, str]:
        return {"value": value}

    for use_responses, mode_name in ((True, "responses"), (False, "chat_completions")):
        try:
            if use_responses:
                await client.responses.create(
                    model=config.model,
                    input='{"task":"return {\\"ok\\":true}"}',
                    max_output_tokens=64,
                )
                model_obj = ResponsesModel(model=config.model, openai_client=client)
            else:
                model_obj = ChatModel(model=config.model, openai_client=client)
            agent = Agent(
                name=f"api smoke {mode_name}",
                instructions="Call echo_tool with value='ok', then return a tiny JSON object. No markdown.",
                tools=[echo_tool],
                model=model_obj,
                model_settings=ModelSettings(max_tokens=64, include_usage=True),
            )
            result = await Runner.run(
                agent,
                input='{"task":"return {\"ok\":true}"}',
                max_turns=2,
                run_config=RunConfig(tracing_disabled=True),
            )
            modes.append({"mode": mode_name, "success": True, "result": serialize_run_result(result)})
            if use_responses:
                break
        except Exception as exc:
            errors.append({"mode": mode_name, "error_type": type(exc).__name__, "error": str(exc)})
    return {"success": bool(modes), "working_modes": modes, "errors": errors}


def import_agents_core():
    try:
        from openai import AsyncOpenAI
        from agents import (
            Agent,
            ModelSettings,
            OpenAIChatCompletionsModel,
            OpenAIResponsesModel,
            Runner,
            RunConfig,
            function_tool,
        )
    except Exception as exc:
        raise RuntimeError(
            "OpenAI Agents SDK is required. Install it in the runtime environment "
            "with Python 3.10+ and package openai-agents."
        ) from exc
    return Agent, Runner, ModelSettings, RunConfig, OpenAIResponsesModel, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool


def import_mcp_stdio():
    try:
        from agents.mcp import MCPServerStdio, create_static_tool_filter
    except Exception as exc:
        raise RuntimeError(
            "OpenAI Agents SDK MCP stdio support is required. Install package mcp "
            "in the same Python runtime used to run this script."
        ) from exc
    return MCPServerStdio, create_static_tool_filter


def serialize_run_result(result: Any) -> dict[str, Any]:
    return {
        "final_output": getattr(result, "final_output", None),
        "last_agent": getattr(getattr(result, "last_agent", None), "name", None),
        "new_items": [serialize_unknown(item) for item in getattr(result, "new_items", [])],
        "raw_responses": [serialize_unknown(item) for item in getattr(result, "raw_responses", [])],
    }


def serialize_unknown(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [serialize_unknown(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_unknown(item) for key, item in value.items()}
    if is_dataclass(value):
        return {
            field.name: serialize_unknown(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if hasattr(value, "model_dump"):
        try:
            return serialize_unknown(value.model_dump())
        except Exception:
            pass
    result = {"type": type(value).__name__}
    for name in (
        "agent",
        "raw_item",
        "output",
        "input",
        "tool_name",
        "arguments",
        "usage",
        "response_id",
        "model",
        "raw_usage",
    ):
        if hasattr(value, name):
            attr = getattr(value, name)
            if name == "agent" and hasattr(attr, "name"):
                attr = attr.name
            result[name] = serialize_unknown(attr)
    return result


def append_tool_events(output_dir: Path, step: int, event: dict[str, Any]) -> None:
    for item in event.get("new_items", []):
        text = json.dumps(item, ensure_ascii=False, default=str)
        if "tool" not in text.lower() and "mcp" not in text.lower():
            continue
        write_jsonl(output_dir / "mcp_calls.jsonl", {"step": step, "item": item})


def extract_created_jobs(event: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for item in walk(event):
        if isinstance(item, dict) and item.get("job_uid"):
            jobs.append({
                "project_uid": item.get("project_uid"),
                "workspace_uid": item.get("workspace_uid"),
                "job_uid": item.get("job_uid"),
                "job_type": item.get("job_type"),
                "status": item.get("status"),
                "queued": item.get("queued"),
            })
    seen = set()
    unique = []
    for job in jobs:
        key = job["job_uid"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(job)
    return unique


def extract_observations(event: dict[str, Any]) -> list[dict[str, Any]]:
    observations = []
    for item in walk(event):
        if isinstance(item, dict) and item.get("message_type") in {"mcp_job_result", "mcp_internal_job_status"}:
            observations.append(item)
    return observations


def extract_usage(event: dict[str, Any]) -> dict[str, Any]:
    usages: list[dict[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            raw_usage = value.get("raw_usage")
            usage = raw_usage if isinstance(raw_usage, dict) else value.get("usage")
            if isinstance(usage, dict):
                usages.append(usage)
            for key, child in value.items():
                if key not in {"raw_usage", "usage"}:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(event)
    return {"usage_records": usages, "summary": summarize_prompt_cache(usages)}


def summarize_prompt_cache(usages: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    for usage in usages:
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        input_tokens += prompt_tokens if isinstance(prompt_tokens, int) else 0
        output_tokens += completion_tokens if isinstance(completion_tokens, int) else 0
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        cached_tokens += cached if isinstance(cached, int) else 0
    ratio = (cached_tokens / input_tokens) if input_tokens else None
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "cache_hit_ratio": ratio,
        "usage_record_count": len(usages),
    }


def summarize_step_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_output": event.get("final_output"),
        "created_jobs": extract_created_jobs(event),
        "prompt_cache": summarize_prompt_cache(extract_usage(event).get("usage_records", [])),
    }


def walk(value: Any):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def contains_stop(text: str) -> bool:
    try:
        parsed = json.loads(text)
    except Exception:
        return '"decision_type":"stop"' in text.replace(" ", "") or '"decision_type": "stop"' in text
    return parsed.get("decision_type") in {"stop", "complete", "request_input"}


def parse_common_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--start-node")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--dataset-json", default="{}")
    parser.add_argument("--dataset-json-file")
    parser.add_argument("--known-workflow-dir", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-turns-per-step", type=int, default=12)
    parser.add_argument("--wait-timeout-seconds", type=int, default=43200)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--server-python", default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--mcp-server", default=DEFAULT_MCP_SERVER)
    parser.add_argument(
        "--mcp-stdio-command",
        help=(
            "Optional complete MCP stdio command. Use this to launch a remote "
            "existing MCP server through ssh without changing MCP tool code."
        ),
    )
    parser.add_argument("--force-chat-completions", action="store_true")
    parser.add_argument("--api-smoke-only", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace, output_dir: Path | None = None) -> AgentsRunConfig:
    if not args.api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key.")
    run_id = args.run_id or uuid.uuid4().hex
    dataset_info = json.loads(Path(args.dataset_json_file).read_text()) if args.dataset_json_file else json.loads(args.dataset_json)
    return AgentsRunConfig(
        project_uid=args.project,
        workspace_uid=args.workspace,
        start_node=args.start_node,
        model=args.model,
        base_url=args.api_base,
        api_key=args.api_key,
        run_id=run_id,
        output_dir=output_dir or Path(args.output_dir) / run_id,
        dataset_info=dataset_info,
        known_workflow_dirs=args.known_workflow_dir,
        max_steps=args.max_steps,
        max_turns_per_step=args.max_turns_per_step,
        wait_timeout_seconds=args.wait_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        server_python=args.server_python,
        project_dir=Path(args.project_dir),
        mcp_server=args.mcp_server,
        mcp_stdio_command=(
            shlex.split(args.mcp_stdio_command)
            if getattr(args, "mcp_stdio_command", None)
            else None
        ),
        use_responses_api=not args.force_chat_completions,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")


def write_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> None:
    args = parse_common_args(argv)
    config = config_from_args(args)
    if args.api_smoke_only:
        result = asyncio.run(smoke_api(config))
        write_json(config.output_dir / "api_smoke.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    summary = asyncio.run(run_agents_closed_loop(config))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
