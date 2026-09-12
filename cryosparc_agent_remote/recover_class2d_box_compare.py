"""Resume after a Class 2D box-size sweep and let a vision model choose Select 2D.

This entry point is intentionally bounded: it compares only the supplied completed
Class 2D jobs and submits at most one Select 2D job.
"""
import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autonomous_mcp_closed_loop import call_tool_json  # noqa: E402
from model_direct_runner import (  # noqa: E402
    parse_model_decision_text,
    resolve_api_key,
    run_openai_compatible_model,
)


DEFAULT_SERVER_PYTHON = "/ssd1/linweifan/miniforge3/envs/cryosparc-agent/bin/python"
DEFAULT_PROJECT_DIR = "/home/lisongyang/cryoagent/cryosparc_agent"
DEFAULT_MCP_SERVER = "cryosparc_mcp_server.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--class2d-jobs", default="J53,J54,J55")
    parser.add_argument("--box-sizes", default="320,384,440")
    parser.add_argument("--api-base", default="https://api.ofox.ai/v1")
    parser.add_argument("--api-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--server-python", default=DEFAULT_SERVER_PYTHON)
    parser.add_argument("--project-dir", default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--mcp-server", default=DEFAULT_MCP_SERVER)
    parser.add_argument("--max-classes", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default="/home/lisongyang/cryoagent/logs/class2d_box_compare_recovery")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_messages(project: str, workspace: str, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for item in evidence:
        visual = item["visual"]
        records.append({
            "job_uid": item["job_uid"],
            "box_size_pix": item["box_size_pix"],
            "image_count": visual.get("image_count"),
            "class_statistics": visual.get("class_statistics", []),
            "label_format": visual.get("contact_sheet", {}).get("label_format"),
        })
    instruction = {
        "task": "compare_class2d_box_size_trials",
        "project_uid": project,
        "workspace_uid": workspace,
        "trials": records,
        "required_output": {
            "decision_type": "forward",
            "action": "select_2D",
            "connections": {
                "particles": {"source_job_uid": "<chosen_class2d_job>", "source_output": "particles"},
                "templates": {"source_job_uid": "<chosen_class2d_job>", "source_output": "class_averages"},
            },
            "parameters": {"selected_templates": "<comma-separated class IDs, e.g. 0,1,4,7>"},
        },
        "rules": [
            "Compare all three attached class-average contact sheets, not only metadata.",
            "Choose exactly one trial job and use its box size for Select 2D.",
            "Select visually interpretable, high-quality classes; do not select obvious junk.",
            "Return exactly one JSON object, with no markdown or explanation.",
            "Do not return action_id, workflow_node_id, or any obsolete schema fields.",
        ],
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Return exactly one valid V2 JSON decision. No thinking text."},
        {"role": "user", "content": json.dumps(instruction, ensure_ascii=False)},
    ]
    for item in evidence:
        sheet = item["visual"]["contact_sheet"]
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": f"Class 2D evidence: job {item['job_uid']}, box_size_pix={item['box_size_pix']}"},
                {"type": "image_url", "image_url": {"url": sheet["data_url"]}},
            ],
        })
    return messages


async def main_async() -> None:
    args = parse_args()
    job_ids = [value.strip() for value in args.class2d_jobs.split(",") if value.strip()]
    sizes = [int(value.strip()) for value in args.box_sizes.split(",") if value.strip()]
    if len(job_ids) != len(sizes):
        raise SystemExit("--class2d-jobs and --box-sizes must have the same length")
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"created_at": utc_now(), "project_uid": args.project, "workspace_uid": args.workspace, "jobs": job_ids, "box_sizes": sizes, "success": False}
    server_env = os.environ.copy()
    params = StdioServerParameters(command=args.server_python, args=[str(Path(args.project_dir) / args.mcp_server)], cwd=args.project_dir, env=server_env)
    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed_tools = await session.list_tools()
        tool_names = sorted(tool.name for tool in listed_tools.tools)
        write_json(run_dir / "mcp_tools.json", tool_names)
        required_tools = {"get_class_average_visual_context", "validate_v2_model_decision", "execute_v2_model_decision"}
        missing_tools = sorted(required_tools - set(tool_names))
        if missing_tools:
            raise RuntimeError(
                "MCP server is missing required tools: " + ", ".join(missing_tools)
                + f". Check --project-dir/--mcp-server; active server is {Path(args.project_dir) / args.mcp_server}."
            )
        context = await call_tool_json(session, "get_workflow_decision_context", {"project_uid": args.project, "workspace_uid": args.workspace, "current_job_uid": job_ids[-1]})
        write_json(run_dir / "decision_context.json", context)
        evidence = []
        for job_uid, box_size in zip(job_ids, sizes):
            visual = await call_tool_json(session, "get_class_average_visual_context", {"project_uid": args.project, "job_uid": job_uid, "max_classes": args.max_classes})
            if not visual.get("success", True) or "contact_sheet" not in visual:
                raise RuntimeError(f"Could not build visual evidence for {job_uid}: {visual}")
            evidence.append({"job_uid": job_uid, "box_size_pix": box_size, "visual": visual})
        write_json(run_dir / "visual_evidence.json", evidence)
        messages = make_messages(args.project, args.workspace, evidence)
        write_json(run_dir / "model_messages.json", messages)
        result = run_openai_compatible_model(messages, args.api_base, resolve_api_key(args.api_key, args.api_key_env), args.api_model, args.max_new_tokens, args.temperature, 300)
        write_json(run_dir / "model_generation.json", result)
        decision = parse_model_decision_text(result["raw_text"])
        normalize_decision(decision)
        write_json(run_dir / "model_decision.json", decision)
        chosen_uid = (decision.get("connections", {}).get("particles", {}) or {}).get("source_job_uid")
        if chosen_uid not in job_ids:
            raise RuntimeError(f"Model must choose one of {job_ids}; got {chosen_uid!r}")
        validation_args = {"decision": decision, "project_uid": args.project, "workspace_uid": args.workspace, "current_node_id": chosen_uid}
        validation = await call_tool_json(session, "validate_v2_model_decision", validation_args)
        write_json(run_dir / "validation.json", validation)
        summary["chosen_class2d_job"] = chosen_uid
        summary["chosen_box_size_pix"] = sizes[job_ids.index(chosen_uid)]
        if not validation.get("success"):
            write_json(run_dir / "summary.json", summary | {"stop_reason": "validation_failed"})
            raise RuntimeError(f"Select 2D decision failed validation: {validation}")
        if args.dry_run:
            execution = {"success": True, "dry_run": True, "message": "Validation passed; no job submitted."}
        else:
            execution = await call_tool_json(session, "execute_v2_model_decision", {**validation_args, "dry_run": False, "allow_approval_required_create": False})
        write_json(run_dir / "execution.json", execution)
        summary["execution"] = execution
        summary["success"] = bool(execution.get("success"))
        write_json(run_dir / "summary.json", summary)
        print(json.dumps({"run_dir": str(run_dir), "chosen_class2d_job": chosen_uid, "chosen_box_size_pix": summary["chosen_box_size_pix"], "execution": execution}, ensure_ascii=False, indent=2))



def normalize_decision(decision: dict[str, Any]) -> None:
    """Normalize harmless formatting variations without changing model choices."""
    parameters = decision.get("parameters")
    if isinstance(parameters, dict) and isinstance(parameters.get("selected_templates"), list):
        parameters["selected_templates"] = ",".join(str(value) for value in parameters["selected_templates"])
    connections = decision.setdefault("connections", {})
    particles = connections.setdefault("particles", {})
    templates = connections.setdefault("templates", {})
    chosen_uid = particles.get("source_job_uid") or templates.get("source_job_uid")
    if chosen_uid:
        particles.setdefault("source_job_uid", chosen_uid)
        particles.setdefault("source_output", "particles")
        templates.setdefault("source_job_uid", chosen_uid)
        templates.setdefault("source_output", "class_averages")


if __name__ == "__main__":
    asyncio.run(main_async())
