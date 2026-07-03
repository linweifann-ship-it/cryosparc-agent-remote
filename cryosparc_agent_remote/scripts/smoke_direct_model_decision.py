# Calls the local model on a prebuilt V2 payload and writes the decision JSON.
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_direct_runner import (
    build_workflow_decision_prompt,
    parse_model_decision_text,
    run_qwen_lora_model,
)


DEFAULT_BASE_MODEL = "/ssd1/lisongyang/models/Qwen3.6-27B-ms-test"
DEFAULT_ADAPTER = "/ssd1/lisongyang/outputs/cryoagent-fsdp-lora-h20-v2-no-workflow"


def parse_args() -> argparse.Namespace:
    """Parse model paths and input/output JSON files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-input-json-file", required=True)
    parser.add_argument("--decision-output-file", required=True)
    parser.add_argument("--report-output-file")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
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
    generation = run_qwen_lora_model(
        messages=messages,
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )
    report = {
        "success": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_input_json_file": args.model_input_json_file,
        "decision_output_file": args.decision_output_file,
        "base_model": args.base_model,
        "adapter": args.adapter,
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


def save_report(report: dict, report_output_file: str | None) -> None:
    """Save the model generation report when a path is provided."""
    if report_output_file:
        Path(report_output_file).write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str)
        )


if __name__ == "__main__":
    main()
