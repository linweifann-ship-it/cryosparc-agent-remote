# Runs one direct-model V2 closed-loop step against a CryoSPARC workflow.
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from job_result import wait_for_job_result_package
from model_direct_runner import (
    build_workflow_decision_prompt,
    parse_model_decision_text,
    resolve_api_key,
    run_openai_compatible_model,
    run_qwen_lora_model,
)
from model_input_builder import build_model_input_payload
from action_registry import get_candidate_actions
from v2_decision_adapter import execute_v2_model_decision_payload


DEFAULT_BACKEND = "local"
DEFAULT_BASE_MODEL = "/ssd1/lisongyang/models/Qwen3.6-27B-ms-test"
DEFAULT_ADAPTER = "/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_API_MODEL = "gpt-5.6-luna"


def parse_args() -> argparse.Namespace:
    """Parse workflow, model, and execution options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--current-node")
    parser.add_argument("--dataset-json", default="{}")
    parser.add_argument("--dataset-json-file")
    parser.add_argument("--known-workflow-dir", action="append", default=[])
    parser.add_argument("--backend", choices=["local", "api"], default=DEFAULT_BACKEND)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--api-timeout-seconds", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--output-dir", default="reports/model_closed_loop")
    return parser.parse_args()


def main() -> None:
    """Run one model -> MCP -> optional CryoSPARC execution loop."""
    args = parse_args()
    dataset_info = load_dataset_info(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_input = build_model_input_payload(
        project_uid=args.project,
        workspace_uid=args.workspace,
        current_job_uid=args.current_node,
        dataset_info=dataset_info,
        known_workflow_dirs=args.known_workflow_dir or None,
    )
    candidate_context = get_candidate_actions(
        project_uid=args.project,
        workspace_uid=args.workspace,
        current_node_id=model_input.get("current_state", {}).get("last_node_id"),
    ) if model_input.get("current_state", {}).get("last_node_id") else None
    if candidate_context:
        model_input["candidate_actions"] = candidate_context["candidate_actions"]
        model_input["blocked_actions"] = candidate_context["blocked_actions"]
        model_input["candidate_context"] = {
            "registry_version": candidate_context["registry_version"],
            "current_node_id": candidate_context["current_node_id"],
            "decision_hint": candidate_context["decision_hint"],
        }

    result: dict[str, Any] = {
        "success": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "current_node": args.current_node,
        "live": args.live,
        "backend": args.backend,
        "model_paths": {
            "base_model": args.base_model,
            "adapter": args.adapter,
            "api_base": args.api_base if args.backend == "api" else None,
            "api_model": args.api_model if args.backend == "api" else None,
        },
        "model_input": model_input,
        "model_called": False,
        "raw_model_output": None,
        "model_decision": None,
        "execution": None,
        "job_results": [],
        "next_model_inputs": [],
        "issues": [],
    }

    if model_input.get("internal_only") or model_input.get("ready_for_model") is False:
        result["success"] = True
        result["issues"].append(
            {
                "code": "job_not_ready_for_model",
                "message": "Current job is active; MCP keeps this status internal.",
            }
        )
        print_and_save(result, output_dir)
        return

    messages = build_workflow_decision_prompt(model_input)
    generation = run_generation(args, messages)
    result["model_called"] = True
    result["raw_model_output"] = generation["raw_text"]

    try:
        decision = parse_model_decision_text(generation["raw_text"])
    except Exception as exc:
        result["issues"].append(
            {
                "code": "model_output_parse_failed",
                "message": str(exc),
                "raw_model_output": generation["raw_text"],
            }
        )
        print_and_save(result, output_dir)
        return

    result["model_decision"] = decision
    execution = execute_v2_model_decision_payload(
        decision,
        project_uid=args.project,
        workspace_uid=args.workspace,
        current_node_id=args.current_node,
        dry_run=not args.live,
    )
    result["execution"] = execution

    if args.live:
        result["job_results"] = wait_for_created_jobs(args, execution)
        for job_package in result["job_results"]:
            if job_package.get("ready_for_model"):
                result["next_model_inputs"].append(
                    build_model_input_payload(
                        project_uid=args.project,
                        workspace_uid=args.workspace,
                        current_job_uid=job_package["job_uid"],
                        dataset_info=dataset_info,
                        known_workflow_dirs=args.known_workflow_dir or None,
                    )
                )

    result["success"] = execution.get("success", False)
    print_and_save(result, output_dir)


def run_generation(args: argparse.Namespace, messages: list[dict[str, str]]) -> dict:
    """Dispatch generation to a local model or an OpenAI-compatible API."""
    if args.backend == "api":
        return run_openai_compatible_model(
            messages=messages,
            api_base=args.api_base,
            api_key=resolve_api_key(args.api_key, args.api_key_env),
            model_name=args.api_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout_seconds=args.api_timeout_seconds,
        )
    return run_qwen_lora_model(
        messages=messages,
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )


def load_dataset_info(args: argparse.Namespace) -> dict[str, Any]:
    """Load dataset metadata from inline JSON or a JSON file."""
    if args.dataset_json_file:
        return json.loads(Path(args.dataset_json_file).read_text())
    return json.loads(args.dataset_json)


def wait_for_created_jobs(
    args: argparse.Namespace,
    execution: dict[str, Any],
) -> list[dict[str, Any]]:
    """Poll jobs created by live execution until terminal state or timeout."""
    job_results = []
    for execution_result in execution.get("execution_result", {}).get(
        "execution_results",
        [],
    ):
        job_uid = execution_result.get("job_uid")
        if not job_uid:
            job_results.append(execution_result)
            continue
        job_results.append(
            wait_for_job_result_package(
                project_uid=args.project,
                workspace_uid=args.workspace,
                job_uid=job_uid,
                timeout_seconds=args.wait_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                include_next_candidates=False,
            )
        )
    return job_results


def print_and_save(result: dict[str, Any], output_dir: Path) -> None:
    """Print the full result and save it for later inspection."""
    path = output_dir / f"model_closed_loop_{timestamp_slug()}.json"
    result["report_path"] = str(path)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def timestamp_slug() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    main()
