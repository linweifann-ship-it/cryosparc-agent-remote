"""Single-agent Deep Agents harness over the existing CryoSPARC MCP server.

This module deliberately owns no CryoSPARC domain logic.  State collection,
V2 validation, execution, and job-result packaging remain MCP tools supplied
by :mod:`cryosparc_mcp_server`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from autonomous_mcp_closed_loop import STATIC_DECISION_INSTRUCTIONS, SYSTEM_PROMPT
from model_direct_runner import parse_model_decision_text, resolve_api_key


_HIDDEN_BUILTINS = (
    "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute", "task"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Turn LangChain/MCP messages and events into stable JSON log records."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    return repr(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n")


def configure_single_agent_profile() -> None:
    """Use official Deep Agents profile controls to disable all subagents/tools.

    Deep Agents auto-adds a general-purpose subagent unless its profile is
    disabled.  This profile leaves LangGraph checkpointing and summarization
    intact while exposing only the existing CryoSPARC MCP tools to this model.
    """
    from deepagents import (
        GeneralPurposeSubagentProfile,
        HarnessProfile,
        register_harness_profile,
    )

    profile = HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=_HIDDEN_BUILTINS,
    )
    register_harness_profile("openai", profile)
    # This provider exists only in --smoke, where no real model is contacted.
    register_harness_profile("fakemessageslistchatmodel", profile)


def build_chat_model(args: argparse.Namespace) -> Any:
    """Build, but do not call, a LangChain ChatOpenAI-compatible model."""
    from langchain_openai import ChatOpenAI

    api_base = args.api_base or os.getenv("OPENAI_BASE_URL") or os.getenv("CRYOAGENT_API_BASE")
    api_model = args.api_model or os.getenv("OPENAI_MODEL") or os.getenv("CRYOAGENT_API_MODEL")
    if not api_base or not api_model:
        raise ValueError(
            "Set --api-base/--api-model or OPENAI_BASE_URL/OPENAI_MODEL "
            "(CRYOAGENT_API_BASE/CRYOAGENT_API_MODEL are also supported)."
        )
    return ChatOpenAI(
        model=api_model,
        api_key=resolve_api_key(args.api_key, args.api_key_env),
        base_url=api_base,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        timeout=args.model_timeout_seconds,
    )


def mcp_connection(args: argparse.Namespace) -> dict[str, Any]:
    server_env = dict(os.environ)
    if os.getenv("CRYOAGENT_GPU_LANE"):
        server_env["CRYOAGENT_GPU_LANE"] = os.environ["CRYOAGENT_GPU_LANE"]
    return {
        "cryosparc": {
            "transport": "stdio",
            "command": args.server_python,
            "args": [str(Path(args.project_dir) / args.mcp_server)],
            "cwd": args.project_dir,
            "env": server_env,
        }
    }


async def discover_mcp_tools(args: argparse.Namespace) -> tuple[Any, list[Any]]:
    """Load standard MCP tools through LangChain's MCP adapter; no API wrapper."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(mcp_connection(args))
    tools = await client.get_tools()
    names = {tool.name for tool in tools}
    required = {
        "get_workflow_decision_context",
        "validate_v2_model_decision",
        "execute_v2_model_decision",
        "get_job_result_package",
        "wait_for_job_result_package",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError("CryoSPARC MCP server is missing required tools: " + ", ".join(missing))
    return client, tools


def build_agent(model: Any, mcp_tools: list[Any], checkpointer: Any) -> Any:
    """Create exactly one Deep Agent with the native LangGraph checkpointer."""
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.middleware import FilesystemMiddleware

    configure_single_agent_profile()
    backend = StateBackend()
    return create_deep_agent(
        model=model,
        tools=mcp_tools,
        system_prompt=SYSTEM_PROMPT + "\n\n" + json.dumps(STATIC_DECISION_INSTRUCTIONS, ensure_ascii=False),
        # Deep Agents requires its read_file primitive internally.  The active
        # harness profile removes it from model-visible tools; no host/shell or
        # mutable filesystem tool is enabled for this scientific agent.
        middleware=[FilesystemMiddleware(backend=backend, tools=["read_file"])],
        subagents=[],
        backend=backend,
        checkpointer=checkpointer,
        name="cryosparc-deepagents-main",
    )


def round_instruction(args: argparse.Namespace, round_index: int) -> str:
    execution_mode = "false" if args.execute else "true"
    return json.dumps(
        {
            "task": "Run exactly one CryoSPARC State -> Decision -> Validation -> Execution -> Observation round.",
            "round": round_index,
            "project_uid": args.project,
            "workspace_uid": args.workspace,
            "current_job_uid": args.current_node,
            "dataset_info": load_dataset_info(args),
            "known_workflow_dirs": args.known_workflow_dir or None,
            "required_protocol": [
                "Call get_workflow_decision_context first with the supplied dataset_info. Treat its live workflow state and candidate actions as authoritative.",
                "Make one V2 decision using the preserved output contract and scientific/rubric rules in the system prompt.",
                "Call validate_v2_model_decision with that exact V2 decision and the supplied dataset_info before any execution.",
                f"Only if validation succeeds, call execute_v2_model_decision with the supplied dataset_info and dry_run={execution_mode}. Never set dry_run=false unless this run was explicitly started with --execute.",
                "For a live created job, call wait_for_job_result_package and use its terminal result as the observation. For dry run, the execution plan is the observation.",
                "Finish with exactly the V2 decision JSON object and no Markdown. Do not invent jobs, inputs, connections, state, or an observation.",
            ],
        },
        ensure_ascii=False,
    )


def load_dataset_info(args: argparse.Namespace) -> dict[str, Any]:
    from dataset_info import normalize_dataset_info

    return normalize_dataset_info(
        json.loads(Path(args.dataset_json_file).read_text())
        if args.dataset_json_file else json.loads(args.dataset_json)
    )


def messages_from_state(state: Any) -> list[Any]:
    values = getattr(state, "values", state)
    if isinstance(values, dict):
        return list(values.get("messages") or [])
    return []


def extract_round_records(messages: Iterable[Any]) -> dict[str, Any]:
    records: dict[str, Any] = {
        "model_input": [], "model_output": [], "tool_calls": [], "token_usage": [],
        "v2_decision": None, "validation_result": None, "execution_result": None,
        "observation": None,
    }
    for message in messages:
        payload = jsonable(message)
        message_type = payload.get("type") if isinstance(payload, dict) else None
        if message_type == "human":
            records["model_input"].append(payload)
        elif message_type == "ai":
            records["model_output"].append(payload)
            usage = payload.get("usage_metadata") or payload.get("response_metadata", {}).get("token_usage")
            if usage:
                records["token_usage"].append(usage)
        elif message_type == "tool":
            record = {"name": payload.get("name"), "content": payload.get("content"), "tool_call_id": payload.get("tool_call_id")}
            records["tool_calls"].append(record)
            content = payload.get("content")
            if record["name"] == "validate_v2_model_decision":
                records["validation_result"] = content
            elif record["name"] == "execute_v2_model_decision":
                records["execution_result"] = content
            elif record["name"] in {"get_job_result_package", "wait_for_job_result_package"}:
                records["observation"] = content
    for output in reversed(records["model_output"]):
        content = output.get("content")
        if not isinstance(content, str):
            continue
        try:
            records["v2_decision"] = parse_model_decision_text(content)
            break
        except Exception:
            pass
    if records["observation"] is None:
        records["observation"] = records["execution_result"]
    return records


async def run(args: argparse.Namespace) -> int:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "run.json", {"created_at": utc_now(), "command": sys.argv, "execute": args.execute})
    _client, mcp_tools = await discover_mcp_tools(args)
    write_json(run_dir / "mcp_tools.json", sorted(tool.name for tool in mcp_tools))
    if args.smoke:
        # A fake LangChain model verifies graph construction without requiring
        # API configuration or sending any request to a model provider.
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage
        from langgraph.checkpoint.memory import InMemorySaver

        agent = build_agent(
            FakeMessagesListChatModel(responses=[AIMessage(content="smoke")]),
            mcp_tools,
            InMemorySaver(),
        )
        visible_graph_tools = set(agent.get_graph().nodes["tools"].data.tools_by_name)
        if "task" in visible_graph_tools:
            raise RuntimeError("Deep Agents smoke failed: task/subagent tool is enabled.")
        write_json(run_dir / "smoke.json", {"success": True, "mcp_tool_count": len(mcp_tools), "agent_initialized": True, "task_tool_enabled": False})
        print(f"Deep Agents smoke passed: initialized one agent with {len(mcp_tools)} MCP tools ({run_dir})")
        return 0

    model = build_chat_model(args)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else run_dir / "langgraph_checkpoints.sqlite"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"configurable": {"thread_id": args.thread_id or run_dir.name}}
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        agent = build_agent(model, mcp_tools, checkpointer)
        write_json(run_dir / "agent.json", {"name": "cryosparc-deepagents-main", "subagents": 0, "thread_id": config["configurable"]["thread_id"], "checkpoint_path": str(checkpoint_path)})
        for round_index in range(1, args.max_rounds + 1):
            await agent.ainvoke({"messages": [{"role": "user", "content": round_instruction(args, round_index)}]}, config=config)
            state = await agent.aget_state(config)
            records = extract_round_records(messages_from_state(state))
            write_json(run_dir / f"round_{round_index:02d}.json", records)
            decision = records["v2_decision"] or {}
            if not args.execute or decision.get("decision_type") in {"stop", "request_input", "complete"}:
                break
    print(f"Deep Agents run completed: {run_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-agent Deep Agents harness for the existing CryoSPARC MCP server.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--current-node")
    parser.add_argument("--dataset-json", default="{}")
    parser.add_argument("--dataset-json-file")
    parser.add_argument("--known-workflow-dir", action="append", default=[])
    parser.add_argument("--api-base")
    parser.add_argument("--api-model")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--model-timeout-seconds", type=int, default=300)
    parser.add_argument("--server-python", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--mcp-server", default="cryosparc_mcp_server.py")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--thread-id", help="Stable LangGraph thread ID to resume from the run's SQLite checkpoint.")
    parser.add_argument("--checkpoint-path", help="Existing/new LangGraph SQLite checkpoint database; pair with --thread-id to resume.")
    parser.add_argument("--output-dir", default="reports/deepagents")
    parser.add_argument("--execute", action="store_true", help="Allow MCP live execution. The default is MCP dry-run only.")
    parser.add_argument("--smoke", action="store_true", help="Import/initialize one fake-model agent and discover MCP tools; never calls a real model or CryoSPARC workflow.")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
