# Core logging, validation, and metrics helpers for V24 benchmark runs.
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from action_registry import validate_parameters
from connection_resolver import resolve_connections_for_decision
from job_specs import get_job_spec, list_supported_job_types
from run_context_guard import RunContextGuard
from v24_contract import (
    SCHEMA_LABEL,
    classify_validation_error,
    strict_parse_model_output,
    validate_v24_decision,
)


EVENT_TYPES = {
    "model_request",
    "model_response",
    "parse_result",
    "validation_result",
    "connection_result",
    "execution_started",
    "job_status",
    "job_completed",
    "round_stopped",
    "run_stopped",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def append_event(
    path: Path,
    run_id: str,
    round_index: int,
    event_type: str,
    **fields: Any,
) -> dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event_type: {event_type}")
    event = {
        "timestamp": utc_now(),
        "run_id": run_id,
        "round": round_index,
        "event_type": event_type,
        "model_id": fields.pop("model_id", None),
        "action": fields.pop("action", None),
        "job_uid": fields.pop("job_uid", None),
        "status": fields.pop("status", None),
        "error_category": fields.pop("error_category", None),
        "latency": fields.pop("latency", None),
        "input_tokens": fields.pop("input_tokens", None),
        "output_tokens": fields.pop("output_tokens", None),
        "source_file": fields.pop("source_file", None),
    }
    event.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    return event


def build_v24_messages(model_input: dict[str, Any], round_index: int) -> list[dict[str, str]]:
    payload = {
        "round": round_index,
        "instruction": (
            "Choose the next CryoSPARC workflow action from the V24 model input. "
            "Return one strict JSON object only. Do not include markdown, hidden "
            "thinking, explanations, confidence, evidence, workflow_node_id, "
            "connections, candidate_actions, failure_context, or retry_guidance. "
            "If human UI work is required or no valid next action is available, "
            "return a stop decision."
        ),
        "model_input": model_input,
        "output_contract": {
            "schema_version": "3.0",
            "decision_type": "forward | branch | stop",
            "selected_actions": [
                {
                    "job_type": "CryoSPARC job type.",
                    "parameters": "Object containing only model-chosen parameter overrides.",
                }
            ],
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the only decision maker for a Schema V24 CryoSPARC "
                "benchmark. Return exactly one directly parseable JSON object."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def evaluate_model_round(
    raw_text: str,
    workflow_state: dict[str, Any],
    guard: RunContextGuard,
) -> dict[str, Any]:
    try:
        decision = strict_parse_model_output(raw_text)
    except Exception as exc:
        return {
            "success": False,
            "stage": "parse",
            "error_category": "model_invalid_json",
            "parse_result": {
                "success": False,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            "decision": None,
        }

    schema_validation = validate_v24_decision(decision)
    if not schema_validation["success"]:
        return {
            "success": False,
            "stage": "schema_validation",
            "error_category": classify_validation_error(schema_validation),
            "parse_result": {"success": True},
            "decision": decision,
            "validation": schema_validation,
        }

    action_validation = validate_actions_and_parameters(decision)
    if not action_validation["success"]:
        return {
            "success": False,
            "stage": "action_validation",
            "error_category": action_validation["error_category"],
            "parse_result": {"success": True},
            "decision": decision,
            "validation": action_validation,
        }

    if decision.get("decision_type") == "stop":
        return {
            "success": True,
            "stage": "stop",
            "parse_result": {"success": True},
            "decision": decision,
            "validation": action_validation,
            "connection_result": None,
        }

    connection_result = resolve_connections_for_decision(
        decision,
        workflow_state=workflow_state,
        allowed_job_uids=guard.allowed_job_uids,
    )
    if not connection_result["success"]:
        return {
            "success": False,
            "stage": "connection",
            "error_category": connection_result["status"],
            "parse_result": {"success": True},
            "decision": decision,
            "validation": action_validation,
            "connection_result": connection_result,
        }

    return {
        "success": True,
        "stage": "validation_only_complete",
        "parse_result": {"success": True},
        "decision": decision,
        "validation": action_validation,
        "connection_result": connection_result,
    }


def validate_actions_and_parameters(decision: dict[str, Any]) -> dict[str, Any]:
    supported = set(list_supported_job_types())
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if decision.get("decision_type") == "stop":
        return {
            "success": True,
            "valid_actions": True,
            "issues": [],
            "warnings": [],
        }
    for index, action in enumerate(decision.get("selected_actions") or []):
        job_type = action.get("job_type")
        if job_type not in supported:
            issues.append(
                {
                    "severity": "error",
                    "code": "unsupported_job_type",
                    "message": f"Unsupported job_type {job_type!r}.",
                    "path": f"selected_actions.{index}.job_type",
                }
            )
            continue
        template = get_job_spec(job_type).get("parameter_template") or {}
        parameter_issues, _ = validate_parameters(
            action.get("parameters") or {},
            template,
            path=f"selected_actions.{index}.parameters",
        )
        for item in parameter_issues:
            dumped = item.model_dump() if hasattr(item, "model_dump") else item
            if dumped.get("code") == "unknown_parameter":
                dumped["severity"] = "warning"
                warnings.append(dumped)
            else:
                issues.append(dumped)
    return {
        "success": not issues,
        "valid_actions": not issues,
        "issues": issues,
        "warnings": warnings,
        "error_category": "model_invalid_action"
        if any(item.get("code") == "unsupported_job_type" for item in issues)
        else "model_invalid_parameter",
    }


def build_manifest(args: Any, run_id: str, prompt_hash: str, dataset_hash: str, config_hash: str, dirty: bool, commit: str | None) -> dict[str, Any]:
    return {
        "benchmark_id": args.benchmark_id,
        "run_id": run_id,
        "model_label": args.model_label,
        "provider": args.model_provider,
        "exact_model_id": getattr(args, "openai_model", None) if args.model_provider == "openai" else getattr(args, "qwen_model", None) if args.model_provider == "dashscope_qwen" else getattr(args, "adapter", None),
        "finetune_checkpoint": getattr(args, "finetune_checkpoint", None) or getattr(args, "adapter", None),
        "schema_version": SCHEMA_LABEL,
        "prompt_hash": prompt_hash,
        "dataset_hash": dataset_hash,
        "start_node": args.current_node,
        "runner_commit": commit,
        "mcp_commit": commit,
        "config_hash": config_hash,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_rounds": args.max_rounds,
        "dirty": dirty,
    }


def compute_metrics(run_dir: Path) -> dict[str, Any]:
    events = read_events(run_dir / "events.jsonl")
    model_responses = [event for event in events if event["event_type"] == "model_response"]
    parse_results = [event for event in events if event["event_type"] == "parse_result"]
    validations = [event for event in events if event["event_type"] == "validation_result"]
    connections = [event for event in events if event["event_type"] == "connection_result"]
    errors = [event for event in events if event.get("error_category")]
    rounds = sorted({event["round"] for event in events if event["round"]})
    input_tokens = sum_number(event.get("input_tokens") for event in model_responses)
    output_tokens = sum_number(event.get("output_tokens") for event in model_responses)
    api_calls = len(model_responses)
    metrics = {
        "v24_valid_rate": rate(validations, lambda event: event.get("status") == "success"),
        "action_parameter_valid_rate": rate(validations, lambda event: event.get("status") == "success"),
        "system_connection_success_rate": rate(connections, lambda event: event.get("status") == "resolved"),
        "job_success_rate": None,
        "final_completed": any(event.get("status") == "model_stop" for event in events),
        "final_fsc": None,
        "total_rounds": len(rounds),
        "valid_jobs": 0,
        "failed_jobs": 0,
        "repeated_jobs": None,
        "invalid_branches": count_errors(errors, "model_invalid_action"),
        "failure_recovery_rate": None,
        "human_intervention_count": count_errors(errors, "approval_required"),
        "total_elapsed_seconds": elapsed_seconds(events),
        "model_elapsed_seconds": sum_number(event.get("latency") for event in model_responses),
        "compute_elapsed_seconds": 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "api_call_count": api_calls,
        "api_cost": None,
        "error_counts": error_counts(errors),
        "parse_success_rate": rate(parse_results, lambda event: event.get("status") == "success"),
    }
    write_json(run_dir / "metrics.json", metrics)
    return metrics


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rate(events: list[dict[str, Any]], predicate) -> float | None:
    if not events:
        return None
    return sum(1 for event in events if predicate(event)) / len(events)


def sum_number(values) -> int | float | None:
    numbers = [value for value in values if isinstance(value, (int, float))]
    return sum(numbers) if numbers else None


def count_errors(events: list[dict[str, Any]], category: str) -> int:
    return sum(1 for event in events if event.get("error_category") == category)


def error_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        category = event.get("error_category") or "unknown"
        counts[category] = counts.get(category, 0) + 1
    return counts


def elapsed_seconds(events: list[dict[str, Any]]) -> float | None:
    if len(events) < 2:
        return None
    try:
        start = datetime.fromisoformat(events[0]["timestamp"])
        end = datetime.fromisoformat(events[-1]["timestamp"])
        return (end - start).total_seconds()
    except Exception:
        return None


def compare_benchmark_runs(benchmark_dir: Path) -> dict[str, Any]:
    run_dirs = sorted(path for path in benchmark_dir.iterdir() if path.is_dir())
    rows = []
    manifests = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "benchmark_manifest.json"
        metrics_path = run_dir / "metrics.json"
        if not manifest_path.exists() or not metrics_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        rows.append({**manifest, **metrics, "run_dir": str(run_dir)})

    comparability = check_comparability(manifests)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("model_label") or "unknown", []).append(row)
    model_stats = {
        model: summarize_rows(items)
        for model, items in grouped.items()
    }
    return {
        "benchmark_id": benchmark_dir.name,
        "run_count": len(rows),
        "comparability": comparability,
        "models": model_stats,
        "runs": rows,
    }


def check_comparability(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "start_node",
        "schema_version",
        "prompt_hash",
        "dataset_hash",
        "runner_commit",
        "mcp_commit",
        "max_rounds",
    ]
    mismatches = []
    if manifests:
        baseline = manifests[0]
        for key in keys:
            values = sorted({str(item.get(key)) for item in manifests})
            if len(values) > 1:
                mismatches.append({"field": key, "values": values})
    return {
        "directly_comparable": not mismatches,
        "status": "directly_comparable" if not mismatches else "not_directly_comparable",
        "mismatches": mismatches,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "v24_valid_rate",
        "action_parameter_valid_rate",
        "system_connection_success_rate",
        "total_rounds",
        "api_call_count",
        "input_tokens",
        "output_tokens",
    ]
    summary: dict[str, Any] = {"run_count": len(rows)}
    for key in numeric_keys:
        values = [row.get(key) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            summary[key] = {"mean": None, "stddev": None, "best": None, "worst": None}
            continue
        summary[key] = {
            "mean": statistics.mean(values),
            "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "best": max(values),
            "worst": min(values),
        }
    summary["success_rate"] = rate(rows, lambda row: bool(row.get("final_completed")))
    return summary


def write_compare_outputs(benchmark_dir: Path, report: dict[str, Any]) -> None:
    write_json(benchmark_dir / "benchmark_summary.json", report)
    rows = report.get("runs") or []
    if rows:
        with (benchmark_dir / "benchmark_table.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        (benchmark_dir / "benchmark_table.csv").write_text("", encoding="utf-8")
    (benchmark_dir / "benchmark_report.md").write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark {report.get('benchmark_id')}",
        "",
        f"- Runs: {report.get('run_count')}",
        f"- Comparability: {report.get('comparability', {}).get('status')}",
        "",
        "## Models",
    ]
    for model, stats in (report.get("models") or {}).items():
        valid = stats.get("v24_valid_rate", {}).get("mean")
        conn = stats.get("system_connection_success_rate", {}).get("mean")
        success = stats.get("success_rate")
        lines.extend(
            [
                f"### {model}",
                f"- run_count: {stats.get('run_count')}",
                f"- v24_valid_rate_mean: {valid}",
                f"- system_connection_success_rate_mean: {conn}",
                f"- success_rate: {success}",
            ]
        )
    if report.get("comparability", {}).get("mismatches"):
        lines.extend(["", "## Not Directly Comparable"])
        for item in report["comparability"]["mismatches"]:
            lines.append(f"- {item['field']}: {', '.join(item['values'])}")
    lines.extend(["", "## Evidence Paths"])
    for row in report.get("runs") or []:
        lines.append(f"- {row.get('model_label')} {row.get('run_id')}: {row.get('run_dir')}")
    return "\n".join(lines) + "\n"

