# Calls a local or OpenAI-compatible model on a prebuilt V2 payload and writes the decision JSON.
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_direct_runner import (
    build_workflow_decision_prompt,
    parse_model_decision_text,
    resolve_api_key,
    run_openai_compatible_model,
    run_qwen_lora_model,
)


DEFAULT_BACKEND = "local"
DEFAULT_BASE_MODEL = "/ssd1/lisongyang/models/Qwen3.6-27B-ms-test"
DEFAULT_ADAPTER = "/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_API_MODEL = "gpt-5.6-luna"


def parse_args() -> argparse.Namespace:
    """Parse model paths and input/output JSON files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input-json-file", required=True)
    parser.add_argument("--decision-output-file", required=True)
    parser.add_argument("--report-output-file")
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
    return parser.parse_args()


def main() -> None:
    """Run model generation and save both parsed decision and raw output."""
    args = parse_args()
    model_input = json.loads(Path(args.model_input_json_file).read_text())
    messages = build_workflow_decision_prompt(model_input)
    generation = run_generation(args, messages)
    report = {
        "success": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "model_input_json_file": args.model_input_json_file,
        "decision_output_file": args.decision_output_file,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "api_base": args.api_base if args.backend == "api" else None,
        "api_model": args.api_model if args.backend == "api" else None,
        "raw_model_output": generation["raw_text"],
        "decision": None,
        "issues": [],
    }

    try:
        decision = parse_model_decision_text(generation["raw_text"])
    except Exception as exc:
        report["issues"].append(
            {
                "code": "model_output_parse_failed",
                "message": str(exc),
            }
        )
        save_report(report, args.report_output_file)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        raise SystemExit(1)

    Path(args.decision_output_file).write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, default=str)
    )
    report["success"] = True
    report["decision"] = decision
    save_report(report, args.report_output_file)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


def run_generation(args: argparse.Namespace, messages: List[Dict[str, str]]) -> Dict:
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


def save_report(report: dict, report_output_file: str) -> None:
    """Save the model generation report when a path is provided."""
    if report_output_file:
        Path(report_output_file).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str)
        )


if __name__ == "__main__":
    main()
