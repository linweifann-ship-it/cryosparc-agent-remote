#!/usr/bin/env python3
"""Prepare workflow-level SFT data from batch-exported cryoSPARC workflow/log JSONs.

This variant targets a batch export layout like:

  workflow/<EMPIAR_ID>.json
  Log/<EMPIAR_ID>/j17.json
  Log/<EMPIAR_ID>/J17_some_plot.png

It reuses the canonical node/decision/chat builders from
prepare_cryosparc_workflow_sft_data.py and only swaps in:
1. workflow discovery for plain <digits>.json files
2. log pairing via Log/<workflow_stem>/
3. parsing of exported per-job JSON records instead of PDF/ZIP logs
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import prepare_cryosparc_workflow_sft_data as base
from parse_emdb_reference_xml import parse_emdb_xml


JSON_LOG_RE = re.compile(r"^j(?P<node_num>\d+)\.json$", re.I)
RESOURCE_LINE_RE = re.compile(r"^\[\d+\]\s+\[(?P<ts>[^\]]+)\]\s+(?P<msg>.+)$")


def discover_batch_workflow_paths(workflow_root: Path) -> list[Path]:
    return sorted(
        path
        for path in workflow_root.glob("*.json")
        if path.is_file() and re.fullmatch(r"\d+[A-Za-z]*", path.stem)
    )


def discover_batch_node_logs(log_dir: Path | None) -> dict[str, Path]:
    if log_dir is None or not log_dir.exists():
        return {}

    found: dict[str, Path] = {}
    for path in sorted(log_dir.iterdir()):
        if not path.is_file():
            continue
        match = JSON_LOG_RE.match(path.name)
        if not match:
            continue
        node_id = f"J{match.group('node_num')}"
        found[node_id] = path
    return found


def _parse_structured_log_lines(text: str) -> list[tuple[str | None, str]]:
    lines: list[tuple[str | None, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = RESOURCE_LINE_RE.match(line)
        if match:
            lines.append((match.group("ts"), match.group("msg")))
        else:
            lines.append((None, line))
    return lines


def _extract_log_evidence(text: str) -> tuple[list[str], dict[str, Any], list[str]]:
    evidence: list[str] = []
    risk_flags: set[str] = set()
    resources: dict[str, Any] = {}

    worker = None
    gpus = None
    cpus = None
    ram = None

    for _ts, line in _parse_structured_log_lines(text):
        if len(evidence) < 10 and (
            "Resources allocated" in line
            or "Running on lane" in line
            or "Working in directory" in line
            or "Job ready to run" in line
            or "completed" in line.lower()
            or "resolution" in line.lower()
            or "FSC" in line
        ):
            evidence.append(line[:240])

        lowered = line.lower()
        if "error" in lowered:
            risk_flags.add("runtime_error_signal")
        if "warning" in lowered:
            risk_flags.add("runtime_warning_signal")
        if "out of memory" in lowered or "oom" in lowered:
            risk_flags.add("oom_signal")

        if ":" in line:
            label, value = [part.strip() for part in line.split(":", 1)]
            if label == "Worker":
                worker = value
            elif label == "GPU":
                gpus = value
            elif label == "CPU":
                cpus = value
            elif label == "RAM":
                ram = value

    if worker:
        resources["worker"] = worker
    if gpus:
        resources["gpu_ids"] = gpus
    if cpus:
        resources["cpu_ids"] = cpus
    if ram:
        resources["ram_ids"] = ram

    return evidence[:10], resources, sorted(risk_flags)


def _event_image_evidence(event_log_entries: list[dict[str, Any]]) -> list[str]:
    evidence: list[str] = []
    for entry in event_log_entries:
        images = entry.get("images") or []
        text = str(entry.get("text") or "").strip()
        if not images:
            continue
        image_names = [img.get("filename") for img in images if img.get("filename")]
        if image_names:
            snippet = f"{text} | images: {', '.join(image_names[:2])}"
            evidence.append(snippet[:240])
        if len(evidence) >= 6:
            break
    return evidence


def parse_exported_job_json(path: Path) -> dict[str, Any]:
    job_document = json.loads(path.read_text(encoding="utf-8"))

    inputs = base.compress_connections(job_document.get("input_slot_groups", []))
    outputs, output_metrics, output_evidence = base.compress_output_groups(
        job_document.get("output_result_groups", [])
    )
    parameters = base.compress_params(job_document.get("params_spec", {}), job_document.get("params_base", {}))

    structured_log_text = (
        job_document.get("structured_event_log_text")
        or job_document.get("worker_log")
        or job_document.get("event_log_text")
        or ""
    )
    log_evidence, runtime_resources, log_risk_flags = _extract_log_evidence(structured_log_text)
    image_evidence = _event_image_evidence(job_document.get("event_log_entries") or [])

    risk_flags: set[str] = set(log_risk_flags)
    status = str(job_document.get("status") or "").lower()
    if "failed" in status or "error" in status:
        risk_flags.add("job_has_error")
    if "warning" in status:
        risk_flags.add("job_has_warning")

    derived_metrics = dict(output_metrics)
    runtime_seconds = base.parse_iso_seconds(job_document.get("started_at"), job_document.get("completed_at"))
    if runtime_seconds is not None:
        derived_metrics["total_runtime_seconds"] = runtime_seconds

    actual_parent_job_ids = sorted(
        {
            connection.get("job_uid")
            for group in job_document.get("input_slot_groups", [])
            for connection in group.get("connections", [])
            if connection.get("job_uid")
        },
        key=base.numeric_job_sort_key,
    )

    return {
        "project": {
            "id": job_document.get("project_uid"),
            "title": None,
        },
        "job": {
            "id": job_document.get("job_uid"),
            "title": job_document.get("title"),
            "type": job_document.get("job_type"),
            "status": job_document.get("status"),
            "cryosparc_version": None,
            "created_by": None,
            "timestamps": {
                "Created": job_document.get("created_at"),
                "Started": job_document.get("started_at"),
                "Completed": job_document.get("completed_at"),
            },
        },
        "inputs": inputs,
        "parameters": parameters,
        "outputs": outputs,
        "derived_metrics": derived_metrics,
        "evidence": (output_evidence + log_evidence + image_evidence)[:12],
        "risk_flags": sorted(risk_flags),
        "runtime_resources": runtime_resources,
        "actual_parent_job_ids": actual_parent_job_ids,
        "raw_job_document": job_document,
    }


def parse_node_log_v2(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_exported_job_json(path)
    return base.parse_node_log(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-root", required=True, help="Directory like workflow/<EMPIAR_ID>.json")
    parser.add_argument("--log-root", required=True, help="Directory like Log/<EMPIAR_ID>/j17.json")
    parser.add_argument("--workflow-labels", default="workflow_label.json")
    parser.add_argument("--reference-xml-root", default=None)
    parser.add_argument("--node-records-jsonl", default="workflow_node_records.jsonl")
    parser.add_argument("--decision-records-jsonl", default="workflow_decision_records.jsonl")
    parser.add_argument("--sft-jsonl", default="workflow_sft_data.jsonl")
    args = parser.parse_args()

    workflow_root = Path(args.workflow_root)
    log_root = Path(args.log_root)
    workflow_paths = discover_batch_workflow_paths(workflow_root)
    if not workflow_paths:
        raise SystemExit(f"No batch workflow JSON files found under: {workflow_root}")

    labels_path = Path(args.workflow_labels)
    labels = base.load_workflow_labels(labels_path) if labels_path.exists() else {}
    reference_xml_root = Path(args.reference_xml_root) if args.reference_xml_root else None

    base.parse_node_log = parse_node_log_v2  # Reuse canonical builders with the new parser.

    all_node_records: list[dict[str, Any]] = []
    all_decision_records: list[dict[str, Any]] = []
    all_sft_records: list[dict[str, Any]] = []
    missing_log_dirs: list[str] = []
    missing_reference_xmls: list[str] = []

    for workflow_path in workflow_paths:
        workflow = base.load_workflow(workflow_path)
        dataset_id = base.workflow_dataset_id(workflow_path, workflow)
        dataset_label = labels.get(dataset_id, {"dataset_id": dataset_id})
        log_dir = log_root / workflow_path.stem
        if not log_dir.exists():
            missing_log_dirs.append(dataset_id)
            node_logs = {}
        else:
            node_logs = discover_batch_node_logs(log_dir)

        reference_xml_path = base.find_reference_xml(
            workflow_path=workflow_path,
            workflow=workflow,
            explicit_reference_root=reference_xml_root,
            explicit_reference_path=None,
        )
        reference_data = parse_emdb_xml(reference_xml_path) if reference_xml_path else None
        if reference_xml_path is None:
            missing_reference_xmls.append(dataset_id)

        node_records = base.build_node_records(
            workflow_path=workflow_path,
            workflow=workflow,
            dataset_label=dataset_label,
            node_logs=node_logs,
            reference_data=reference_data,
        )
        decision_records = base.build_decision_records_for_workflow(
            workflow_path=workflow_path,
            workflow=workflow,
            dataset_label=dataset_label,
            node_records=node_records,
        )

        all_node_records.extend(
            node_records[node_id] for node_id in sorted(node_records.keys(), key=base.numeric_job_sort_key)
        )
        all_decision_records.extend(decision_records)
        all_sft_records.extend(
            {
                "id": decision_record["sample_id"],
                "dataset_id": decision_record["dataset_id"],
                "workflow_id": decision_record["workflow_id"],
                "messages": base.build_chat_messages(
                    decision_record["model_input"],
                    decision_record["model_output"],
                ),
                "metadata": {
                    "data_source": decision_record["data_source"],
                    "decision_index": decision_record["decision_index"],
                    "decision_nodes": decision_record["decision_nodes"],
                },
            }
            for decision_record in decision_records
        )

    base.write_jsonl(Path(args.node_records_jsonl), all_node_records)
    base.write_jsonl(Path(args.decision_records_jsonl), all_decision_records)
    base.write_jsonl(Path(args.sft_jsonl), all_sft_records)

    print(
        json.dumps(
            {
                "workflow_count": len(workflow_paths),
                "node_record_count": len(all_node_records),
                "decision_record_count": len(all_decision_records),
                "sft_record_count": len(all_sft_records),
                "missing_log_dirs": missing_log_dirs,
                "missing_reference_xmls": missing_reference_xmls,
                "node_records_jsonl": str(Path(args.node_records_jsonl).resolve()),
                "decision_records_jsonl": str(Path(args.decision_records_jsonl).resolve()),
                "sft_jsonl": str(Path(args.sft_jsonl).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
