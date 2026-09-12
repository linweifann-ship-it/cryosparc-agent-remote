#!/usr/bin/env python3
"""Prepare workflow-level SFT data for a cryoSPARC agent.

This script consumes workflow definitions, workflow labels, and per-node job logs
to build:

1. Canonical workflow node records
2. Decision records for the workflow agent task
3. Chat-format SFT samples for model training

The target task is:

- Input: current completed nodes, current node state, workflow labels,
  candidate actions
- Output: structured JSON describing the next workflow action(s)
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from parse_emdb_reference_xml import parse_emdb_xml
from prepare_cryosparc_sft_data import parse_pdf_record, summarize_for_sft


WORKFLOW_LOG_RE = re.compile(r"J(?P<node_num>\d+)-EventLog\.(?P<suffix>pdf|zip)$", re.I)
EMPIAR_ID_RE = re.compile(r"empiar[-_]?(\d+)", re.I)
EMD_ID_RE = re.compile(r"EMD[-_]?(\d+)", re.I)
ISO_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def normalize_dataset_id(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    match = EMPIAR_ID_RE.search(text)
    if match:
        return f"EMPIAR-{match.group(1)}"
    if text.isdigit():
        return f"EMPIAR-{text}"
    return text.upper().replace("_", "-")


def numeric_job_sort_key(job_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", job_id)
    if match:
        return (int(match.group(1)), job_id)
    return (10**9, job_id)


def clean_json_text(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("."):
        stripped = stripped[1:].lstrip()
    start_positions = [pos for pos in (stripped.find("["), stripped.find("{")) if pos >= 0]
    if start_positions:
        stripped = stripped[min(start_positions) :]
    return stripped


def load_workflow_labels(path: Path) -> dict[str, dict[str, Any]]:
    raw = clean_json_text(path.read_text(encoding="utf-8"))
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected label file to contain a list: {path}")

    labels: dict[str, dict[str, Any]] = {}
    for item in data:
        empiar_id = item.get("EMPIAR_ID")
        dataset_id = normalize_dataset_id(empiar_id)
        if not dataset_id:
            continue
        labels[dataset_id] = {
            "dataset_id": dataset_id,
            "empiar_id": empiar_id,
            "input_type": item.get("input"),
            "sample_type": item.get("types"),
            "target_resolution": item.get("resolution"),
            "num_of_maps": item.get("num_of_maps"),
        }
    return labels


def discover_workflow_paths(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*workflow.json")))
        elif any(char in item for char in "*?[]"):
            paths.extend(sorted(Path().glob(item)))
        else:
            paths.append(path)

    results = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".json" and "workflow" in path.name.lower():
            results.append(path)
    return sorted(set(results))


def discover_workflow_paths_from_root(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("*workflow.json")
        if path.is_file() and not path.name.endswith("-standard.json")
    )


def load_workflow(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def workflow_dataset_id(path: Path, workflow: dict[str, Any]) -> str:
    path_dataset_id = normalize_dataset_id(path.stem)
    if path_dataset_id and path_dataset_id.startswith("EMPIAR-"):
        return path_dataset_id
    title_dataset_id = normalize_dataset_id(workflow.get("title"))
    if title_dataset_id:
        return title_dataset_id
    raise ValueError(f"Could not infer dataset id from workflow: {path}")


def infer_reference_emd_id(workflow: dict[str, Any]) -> str | None:
    for _node_id, job in workflow.get("jobs", {}).items():
        title = str(job.get("title") or "")
        match = EMD_ID_RE.search(title)
        if match:
            return f"EMD-{match.group(1)}"
        for _param_name, meta in job.get("parameters", {}).items():
            value = str(meta.get("value") or "")
            match = EMD_ID_RE.search(value)
            if match:
                return f"EMD-{match.group(1)}"
    return None


def find_log_dir(workflow_path: Path, dataset_id: str, explicit_log_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_log_root is not None:
        candidates.extend(
            [
                explicit_log_root / f"{dataset_id}-JobLog",
                explicit_log_root / f"{dataset_id.lower()}-JobLog",
                explicit_log_root / f"{dataset_id.lower()}-joblog",
            ]
        )

    workflow_parent = workflow_path.parent
    candidates.extend(
        [
            workflow_parent / f"{dataset_id}-JobLog",
            workflow_parent / f"{dataset_id.lower()}-JobLog",
            workflow_parent / f"{dataset_id.lower()}-joblog",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def pair_workflows_with_logs(
    workflow_paths: list[Path],
    explicit_log_root: Path | None,
) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    for workflow_path in workflow_paths:
        workflow = load_workflow(workflow_path)
        dataset_id = workflow_dataset_id(workflow_path, workflow)
        log_dir = find_log_dir(workflow_path, dataset_id, explicit_log_root)
        pairs.append((workflow_path, log_dir))
    return pairs


def find_reference_xml(
    workflow_path: Path,
    workflow: dict[str, Any],
    explicit_reference_root: Path | None,
    explicit_reference_path: Path | None,
) -> Path | None:
    if explicit_reference_path is not None:
        return explicit_reference_path if explicit_reference_path.exists() else None

    emd_id = infer_reference_emd_id(workflow)
    if emd_id is None:
        return None

    candidates: list[Path] = []
    if explicit_reference_root is not None:
        candidates.extend(
            [
                explicit_reference_root / f"{emd_id.lower()}.xml",
                explicit_reference_root / f"{emd_id.upper()}.xml",
            ]
        )

    workflow_parent = workflow_path.parent
    candidates.extend(
        [
            workflow_parent / f"{emd_id.lower()}.xml",
            workflow_parent / f"{emd_id.upper()}.xml",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def workflow_upstreams(workflow: dict[str, Any]) -> dict[str, list[str]]:
    upstreams: dict[str, list[str]] = {}
    for node_id, job in workflow["jobs"].items():
        parents = []
        for group in job.get("groups", []):
            if not group:
                continue
            source = group[0]
            source_node = str(source).split(".", 1)[0]
            if source_node.startswith("J"):
                parents.append(source_node)
        upstreams[node_id] = sorted(set(parents), key=numeric_job_sort_key)
    return upstreams


def workflow_children(upstreams: dict[str, list[str]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, parents in upstreams.items():
        for parent in parents:
            children[parent].append(node_id)
    for node_id in list(children):
        children[node_id] = sorted(children[node_id], key=numeric_job_sort_key)
    return children


def topological_batches(nodes: Iterable[str], upstreams: dict[str, list[str]]) -> list[list[str]]:
    remaining = set(nodes)
    completed: set[str] = set()
    batches: list[list[str]] = []

    while remaining:
        ready = sorted(
            [node for node in remaining if set(upstreams.get(node, [])) <= completed],
            key=numeric_job_sort_key,
        )
        if not ready:
            raise ValueError("Workflow graph contains a cycle or unresolved dependency.")
        batches.append(ready)
        completed.update(ready)
        remaining.difference_update(ready)

    return batches


def timestamps_from_job(job: dict[str, Any]) -> dict[str, str | None]:
    return {
        "Created": job.get("created_at"),
        "Queued": job.get("queued_at"),
        "Launched": job.get("launched_at"),
        "Started": job.get("started_at"),
        "Waiting": job.get("waiting_at"),
        "Completed": job.get("completed_at"),
        "Failed": job.get("failed_at"),
    }


def parse_iso_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    if not ISO_DT_RE.match(start) or not ISO_DT_RE.match(end):
        return None
    started = datetime.fromisoformat(start.replace("Z", "+00:00"))
    finished = datetime.fromisoformat(end.replace("Z", "+00:00"))
    return round((finished - started).total_seconds(), 3)


def parse_runtime_log_texts(archive: zipfile.ZipFile) -> dict[str, str]:
    payload: dict[str, str] = {}
    for name in archive.namelist():
        lower_name = name.lower()
        if lower_name.endswith(".log") and "job_log_" in lower_name:
            payload[name] = archive.read(name).decode("utf-8", errors="ignore")
    return payload


def extract_log_signals(log_texts: dict[str, str]) -> list[str]:
    evidence: list[str] = []

    patterns = (
        (re.compile(r"Queue message (.+)"), None),
        (re.compile(r"Running job\s+J\d+\s+of type\s+(\S+)"), None),
        (re.compile(r"Allocated Resources\s+:\s+(.+)"), None),
    )

    for text in log_texts.values():
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[:200]:
            for pattern, _ in patterns:
                match = pattern.search(line)
                if match:
                    evidence.append(line[:240])
                    break

    return evidence[:10]


def compress_connections(input_slot_groups: list[dict[str, Any]]) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for group in input_slot_groups:
        group_name = group.get("name") or group.get("type") or "input"
        group_record = {"dataset": None, "fields": {}}
        for connection in group.get("connections", []):
            dataset = f"{connection.get('job_uid')}.{connection.get('group_name')}"
            if group_record["dataset"] is None:
                group_record["dataset"] = dataset
            for slot in connection.get("slots", []):
                field_name = slot.get("slot_name") or slot.get("result_name")
                field_value = (
                    f"{slot.get('job_uid')}.{slot.get('group_name')}.{slot.get('result_name')}"
                    if slot.get("job_uid") and slot.get("group_name") and slot.get("result_name")
                    else slot.get("result_type")
                )
                if field_name:
                    existing = group_record["fields"].get(field_name)
                    if existing is None:
                        group_record["fields"][field_name] = field_value
                    elif isinstance(existing, list):
                        existing.append(field_value)
                    else:
                        group_record["fields"][field_name] = [existing, field_value]
        inputs[group_name] = group_record
    return inputs


def compress_output_groups(output_result_groups: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    outputs: dict[str, Any] = {}
    derived_metrics: dict[str, Any] = {}
    evidence: list[str] = []

    for group in output_result_groups:
        group_name = group.get("name") or group.get("type") or "output"
        fields: dict[str, Any] = {}
        for item in group.get("contains", []):
            fields[item.get("name", "result")] = {
                "type": item.get("type"),
                "passthrough": item.get("passthrough"),
            }

        outputs[group_name] = {
            "dataset": group.get("uid"),
            "fields": fields,
            "num_items": group.get("num_items"),
        }

        if group.get("num_items") is not None:
            derived_metrics[f"{group_name}_num_items"] = group["num_items"]
            evidence.append(f"Output group {group_name} contains {group['num_items']} items.")

        latest_stats = group.get("latest_summary_stats") or {}
        if isinstance(latest_stats, dict):
            if "radwn_final_A" in latest_stats:
                derived_metrics[f"{group_name}_radwn_final_A"] = latest_stats["radwn_final_A"]
                evidence.append(
                    f"Output group {group_name} final resolution metric is {latest_stats['radwn_final_A']} A."
                )
            if "radwn_A" in latest_stats:
                derived_metrics[f"{group_name}_radwn_A"] = latest_stats["radwn_A"]
            if "extent" in latest_stats:
                derived_metrics[f"{group_name}_extent"] = latest_stats["extent"]

    return outputs, derived_metrics, evidence[:10]


def compress_params(params_spec: dict[str, Any], params_base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if params_spec:
        result: dict[str, dict[str, Any]] = {"General": {}}
        for name, spec in params_spec.items():
            value = spec.get("value") if isinstance(spec, dict) else spec
            result["General"][name] = value
        return result

    result: dict[str, dict[str, Any]] = defaultdict(dict)
    for name, meta in params_base.items():
        if meta.get("hidden"):
            continue
        value = meta.get("value")
        if value is None:
            continue
        section = meta.get("section") or "General"
        result[section][name] = value
    return dict(result)


def parse_zip_record(zip_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as archive:
        job_document_name = next(
            name for name in archive.namelist() if name.startswith("job_document_") and name.endswith(".json")
        )
        job_document = json.loads(archive.read(job_document_name))
        log_texts = parse_runtime_log_texts(archive)

    inputs = compress_connections(job_document.get("input_slot_groups", []))
    outputs, output_metrics, output_evidence = compress_output_groups(job_document.get("output_result_groups", []))
    parameters = compress_params(job_document.get("params_spec", {}), job_document.get("params_base", {}))
    log_evidence = extract_log_signals(log_texts)

    risk_flags: set[str] = set()
    if job_document.get("has_error"):
        risk_flags.add("job_has_error")
    if job_document.get("has_warning"):
        risk_flags.add("job_has_warning")
    if job_document.get("queue_status") == "waiting_resources":
        risk_flags.add("resource_wait")
    if job_document.get("errors_run"):
        risk_flags.add("runtime_error_signal")

    derived_metrics = dict(output_metrics)
    runtime_seconds = parse_iso_seconds(job_document.get("started_at"), job_document.get("completed_at"))
    if runtime_seconds is not None:
        derived_metrics["total_runtime_seconds"] = runtime_seconds

    resources = {
        "platform_node": job_document.get("instance_information", {}).get("platform_node"),
        "gpu_names": [
            gpu.get("name") for gpu in job_document.get("instance_information", {}).get("gpu_info", [])
        ],
        "total_memory": job_document.get("instance_information", {}).get("total_memory"),
    }

    return {
        "project": {
            "id": job_document.get("project_uid"),
            "title": None,
        },
        "job": {
            "id": job_document.get("uid"),
            "title": job_document.get("title"),
            "type": job_document.get("job_type"),
            "status": job_document.get("status"),
            "cryosparc_version": job_document.get("version"),
            "created_by": job_document.get("created_by_user_id"),
            "timestamps": timestamps_from_job(job_document),
        },
        "inputs": inputs,
        "parameters": parameters,
        "outputs": outputs,
        "derived_metrics": derived_metrics,
        "evidence": (output_evidence + log_evidence)[:12],
        "risk_flags": sorted(risk_flags),
        "runtime_resources": resources,
        "actual_parent_job_ids": sorted(job_document.get("parents", []), key=numeric_job_sort_key),
        "raw_job_document": job_document,
    }


def parse_pdf_log_record(pdf_path: Path) -> dict[str, Any]:
    parsed = parse_pdf_record(pdf_path)
    summary = summarize_for_sft(parsed)

    risk_flags: list[str] = []
    for event in parsed.get("events", []):
        message = event.get("message", "")
        if "Error" in message or "ERROR" in message:
            risk_flags.append("runtime_error_signal")
        if "Warning" in message or "WARNING" in message:
            risk_flags.append("runtime_warning_signal")

    evidence = [item["message"] for item in parsed.get("derived_metrics", {}).get("key_events", [])[:8]]

    return {
        "project": summary["project"],
        "job": summary["job"],
        "inputs": summary["inputs"],
        "parameters": summary["parameters"],
        "outputs": summary["outputs"],
        "derived_metrics": summary["derived_metrics"],
        "evidence": evidence,
        "risk_flags": sorted(set(risk_flags)),
        "runtime_resources": {},
        "actual_parent_job_ids": [],
        "raw_parsed_record": parsed,
    }


def parse_node_log(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".zip":
        return parse_zip_record(path)
    if path.suffix.lower() == ".pdf":
        return parse_pdf_log_record(path)
    raise ValueError(f"Unsupported log file: {path}")


def discover_node_logs(log_dir: Path | None) -> dict[str, Path]:
    if log_dir is None or not log_dir.exists():
        return {}

    found: dict[str, Path] = {}
    candidates = sorted(log_dir.iterdir(), key=lambda path: (path.suffix.lower() != ".zip", path.name))
    for path in candidates:
        match = WORKFLOW_LOG_RE.search(path.name)
        if not match:
            continue
        node_id = f"J{match.group('node_num')}"
        if node_id not in found or path.suffix.lower() == ".zip":
            found[node_id] = path
    return found


def workflow_parameter_template(job: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, meta in job.get("parameters", {}).items():
        result[key] = meta.get("value")
    return result


def build_node_records(
    workflow_path: Path,
    workflow: dict[str, Any],
    dataset_label: dict[str, Any] | None,
    node_logs: dict[str, Path],
    reference_data: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    dataset_id = workflow_dataset_id(workflow_path, workflow)
    upstreams = workflow_upstreams(workflow)
    children = workflow_children(upstreams)
    node_records: dict[str, dict[str, Any]] = {}

    for node_id, job in sorted(workflow["jobs"].items(), key=lambda item: numeric_job_sort_key(item[0])):
        log_path = node_logs.get(node_id)
        runtime_record = parse_node_log(log_path) if log_path else None
        workflow_params = workflow_parameter_template(job)
        reference_evidence: list[str] = []
        if reference_data:
            summary = reference_data.get("summary", {})
            if summary.get("reference_id") and summary.get("resolution_a") is not None:
                reference_evidence.append(
                    f"Reference {summary['reference_id']} reports resolution {summary['resolution_a']} A."
                )
            if summary.get("symmetry"):
                reference_evidence.append(f"Reference symmetry is {summary['symmetry']}.")
        node_records[node_id] = {
            "dataset_id": dataset_id,
            "workflow_id": workflow.get("_id") or f"wf_{dataset_id.lower().replace('-', '_')}",
            "workflow_title": workflow.get("title"),
            "workflow_version": workflow.get("workflowVersion"),
            "workflow_node_id": node_id,
            "workflow_job_type": job.get("jobType"),
            "workflow_title_for_node": job.get("title"),
            "workflow_description_for_node": job.get("description"),
            "workflow_upstream_nodes": upstreams.get(node_id, []),
            "workflow_child_nodes": children.get(node_id, []),
            "workflow_groups": job.get("groups", []),
            "workflow_parameters": workflow_params,
            "dataset": dataset_label or {"dataset_id": dataset_id},
            "reference_data": reference_data,
            "log_source_path": str(log_path) if log_path else None,
            "log_source_type": log_path.suffix.lower().lstrip(".") if log_path else None,
            "project": runtime_record.get("project") if runtime_record else {"id": None, "title": None},
            "job": runtime_record.get("job")
            if runtime_record
            else {
                "id": None,
                "title": job.get("title"),
                "type": job.get("jobType"),
                "status": None,
                "cryosparc_version": workflow.get("csVersion"),
                "created_by": workflow.get("createdBy"),
                "timestamps": {},
            },
            "inputs": runtime_record.get("inputs") if runtime_record else {},
            "parameters": runtime_record.get("parameters") if runtime_record else {"General": workflow_params},
            "outputs": runtime_record.get("outputs") if runtime_record else {},
            "derived_metrics": runtime_record.get("derived_metrics") if runtime_record else {},
            "evidence": (reference_evidence + (runtime_record.get("evidence", []) if runtime_record else []))[:12],
            "risk_flags": runtime_record.get("risk_flags", []) if runtime_record else [],
            "runtime_resources": runtime_record.get("runtime_resources", {}) if runtime_record else {},
            "actual_parent_job_ids": runtime_record.get("actual_parent_job_ids", []) if runtime_record else [],
            "status": runtime_record.get("job", {}).get("status") if runtime_record else None,
            "actual_job_id": runtime_record.get("job", {}).get("id") if runtime_record else None,
            "actual_project_id": runtime_record.get("project", {}).get("id") if runtime_record else None,
        }

    return node_records


def summarize_completed_node(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "workflow_node_id": record["workflow_node_id"],
        "job_type": record["workflow_job_type"],
        "actual_job_id": record.get("actual_job_id"),
        "status": record.get("status"),
    }


def pseudo_state(
    dataset: dict[str, Any],
    reference_data: dict[str, Any] | None,
    workflow: dict[str, Any],
    status: str,
    recent_completed_nodes: list[str],
    evidence: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "workflow_node_id": None,
        "job_type": None,
        "actual_job_id": None,
        "status": status,
        "project": {"id": None, "title": workflow.get("title")},
        "job": {
            "id": None,
            "title": workflow.get("title"),
            "type": None,
            "status": status,
            "cryosparc_version": workflow.get("csVersion"),
            "created_by": workflow.get("createdBy"),
            "timestamps": {},
        },
        "inputs": {},
        "parameters": {},
        "outputs": {},
        "derived_metrics": {},
        "quality_flags": [],
        "evidence": evidence or [],
        "recent_completed_nodes": recent_completed_nodes,
        "dataset": dataset,
        "reference_data": reference_data,
    }


def state_from_previous_batch(
    previous_batch: list[str],
    node_records: dict[str, dict[str, Any]],
    dataset: dict[str, Any],
    workflow: dict[str, Any],
    reference_data: dict[str, Any] | None,
) -> dict[str, Any]:
    if not previous_batch:
        return pseudo_state(
            dataset=dataset,
            reference_data=reference_data,
            workflow=workflow,
            status="not_started",
            recent_completed_nodes=[],
            evidence=["Workflow has not started yet."],
        )

    representative = node_records[previous_batch[-1]]
    state = {
        "workflow_node_id": representative["workflow_node_id"],
        "job_type": representative["workflow_job_type"],
        "actual_job_id": representative.get("actual_job_id"),
        "status": representative.get("status") or "completed",
        "project": representative.get("project", {}),
        "job": representative.get("job", {}),
        "inputs": representative.get("inputs", {}),
        "parameters": representative.get("parameters", {}),
        "outputs": representative.get("outputs", {}),
        "derived_metrics": representative.get("derived_metrics", {}),
        "quality_flags": representative.get("risk_flags", []),
        "evidence": representative.get("evidence", []),
        "recent_completed_nodes": previous_batch,
        "reference_data": representative.get("reference_data", reference_data),
    }
    return state


def build_candidate_action(node_id: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_id": f"{'branch' if False else 'forward'}_{node_id}",
        "action_type": "forward",
        "workflow_node_id": node_id,
        "job_type": record["workflow_job_type"],
        "allowed_inputs": [group[0] for group in record.get("workflow_groups", []) if group],
        "parameter_template": record["workflow_parameters"],
    }


def build_candidate_actions_for_batch(batch: list[str], node_records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    action_type = "branch" if len(batch) > 1 else "forward"
    for node_id in batch:
        record = node_records[node_id]
        action = build_candidate_action(node_id, record)
        action["action_type"] = action_type
        action["action_id"] = f"{action_type}_{node_id}"
        actions.append(action)
    return actions


def build_reason(batch: list[str], node_records: dict[str, dict[str, Any]], stop: bool = False) -> str:
    if stop:
        return "Workflow is complete and no further action is required."
    if len(batch) == 1:
        record = node_records[batch[0]]
        return f"Upstream dependencies are satisfied for {record['workflow_job_type']}."
    job_types = ", ".join(node_records[node_id]["workflow_job_type"] for node_id in batch)
    return f"Multiple ready workflow nodes can run in parallel: {job_types}."


def build_model_input(
    schema_version: str,
    dataset: dict[str, Any],
    workflow: dict[str, Any],
    completed_nodes: list[str],
    current_state: dict[str, Any],
    candidate_actions: list[dict[str, Any]],
    node_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "task_type": "workflow_decision",
        "dataset": dataset,
        "reference_data": current_state.get("reference_data"),
        "workflow": {
            "workflow_id": workflow.get("_id") or f"wf_{dataset['dataset_id'].lower().replace('-', '_')}",
            "workflow_title": workflow.get("title"),
            "workflow_version": workflow.get("workflowVersion"),
            "current_workflow_node_id": current_state.get("workflow_node_id"),
            "completed_nodes": [summarize_completed_node(node_records[node_id]) for node_id in completed_nodes],
            "available_upstream_outputs": sorted(
                {
                    source
                    for node_id in completed_nodes
                    for source, _group_name in node_records[node_id].get("workflow_groups", [])
                }
            ),
        },
        "current_state": current_state,
        "candidate_actions": candidate_actions,
        "constraints": {
            "max_branches": 8,
            "must_choose_from_candidates": True,
            "return_json_only": True,
        },
    }


def build_model_output(
    schema_version: str,
    decision_type: str,
    selected_actions: list[dict[str, Any]],
    rollback_target: dict[str, Any] | None,
    branch_plan: dict[str, Any] | None,
    reason: str,
    risk_flags: list[str],
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "decision_type": decision_type,
        "selected_actions": selected_actions,
        "rollback_target": rollback_target,
        "branch_plan": branch_plan,
        "reason": reason,
        "confidence": 0.99,
        "risk_flags": risk_flags,
        "evidence": evidence[:8],
    }


def build_decision_records_for_workflow(
    workflow_path: Path,
    workflow: dict[str, Any],
    dataset_label: dict[str, Any] | None,
    node_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    schema_version = "1.0"
    dataset = dataset_label or {"dataset_id": workflow_dataset_id(workflow_path, workflow)}
    reference_data = next(iter(node_records.values())).get("reference_data") if node_records else None
    upstreams = workflow_upstreams(workflow)
    batches = topological_batches(workflow["jobs"].keys(), upstreams)
    records: list[dict[str, Any]] = []
    completed_nodes: list[str] = []
    previous_batch: list[str] = []

    for batch_index, batch in enumerate(batches):
        current_state = state_from_previous_batch(previous_batch, node_records, dataset, workflow, reference_data)
        candidate_actions = build_candidate_actions_for_batch(batch, node_records)
        decision_type = "branch" if len(batch) > 1 else "forward"
        reason = build_reason(batch, node_records)
        evidence = current_state.get("evidence", [])
        risk_flags = current_state.get("quality_flags", [])

        selected_actions = [
            {
                "action_id": action["action_id"],
                "action_type": decision_type,
                "workflow_node_id": action["workflow_node_id"],
                "job_type": action["job_type"],
                "parameters": action["parameter_template"],
            }
            for action in candidate_actions
        ]

        branch_plan = None
        if decision_type == "branch":
            branch_plan = {
                "branch_type": "parallel_required_nodes",
                "max_parallel_branches": len(selected_actions),
                "notes": "These workflow nodes are concurrently ready in the successful reference workflow.",
            }

        model_input = build_model_input(
            schema_version=schema_version,
            dataset=dataset,
            workflow=workflow,
            completed_nodes=completed_nodes,
            current_state=current_state,
            candidate_actions=candidate_actions,
            node_records=node_records,
        )
        model_output = build_model_output(
            schema_version=schema_version,
            decision_type=decision_type,
            selected_actions=selected_actions,
            rollback_target=None,
            branch_plan=branch_plan,
            reason=reason,
            risk_flags=risk_flags,
            evidence=evidence,
        )

        records.append(
            {
                "sample_id": f"{dataset['dataset_id']}:step:{batch_index:03d}",
                "data_source": "real_success",
                "dataset_id": dataset["dataset_id"],
                "workflow_id": workflow.get("_id") or f"wf_{dataset['dataset_id'].lower().replace('-', '_')}",
                "workflow_title": workflow.get("title"),
                "decision_index": batch_index,
                "decision_nodes": batch,
                "model_input": model_input,
                "model_output": model_output,
            }
        )

        completed_nodes.extend(batch)
        previous_batch = batch

    stop_state = state_from_previous_batch(previous_batch, node_records, dataset, workflow, reference_data)
    stop_input = build_model_input(
        schema_version=schema_version,
        dataset=dataset,
        workflow=workflow,
        completed_nodes=completed_nodes,
        current_state=stop_state,
        candidate_actions=[],
        node_records=node_records,
    )
    stop_output = build_model_output(
        schema_version=schema_version,
        decision_type="stop",
        selected_actions=[],
        rollback_target=None,
        branch_plan=None,
        reason=build_reason([], node_records, stop=True),
        risk_flags=stop_state.get("quality_flags", []),
        evidence=stop_state.get("evidence", []),
    )
    records.append(
        {
            "sample_id": f"{dataset['dataset_id']}:step:{len(batches):03d}",
            "data_source": "real_success",
            "dataset_id": dataset["dataset_id"],
            "workflow_id": workflow.get("_id") or f"wf_{dataset['dataset_id'].lower().replace('-', '_')}",
            "workflow_title": workflow.get("title"),
            "decision_index": len(batches),
            "decision_nodes": [],
            "model_input": stop_input,
            "model_output": stop_output,
        }
    )

    return records


def build_chat_messages(model_input: dict[str, Any], model_output: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a cryoSPARC workflow assistant. Read the workflow state and "
                "return only valid JSON that follows the required decision schema."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(model_input, ensure_ascii=False, indent=2),
        },
        {
            "role": "assistant",
            "content": json.dumps(model_output, ensure_ascii=False, indent=2),
        },
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflows",
        nargs="*",
        help="Workflow JSON files, directories, or globs.",
    )
    parser.add_argument(
        "--workflow-root",
        default=None,
        help="Directory containing batch workflow files named like empiar-xxxxx-workflow.json.",
    )
    parser.add_argument(
        "--workflow-labels",
        default="workflow_label.json",
        help="Workflow label JSON file.",
    )
    parser.add_argument(
        "--log-root",
        default=None,
        help="Optional root directory containing JobLog directories named like EMPIAR-xxxxx-JobLog.",
    )
    parser.add_argument(
        "--workflow-path",
        default=None,
        help="Explicit single workflow JSON path.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Explicit single JobLog directory paired with --workflow-path.",
    )
    parser.add_argument(
        "--reference-xml-path",
        default=None,
        help="Explicit single EMDB XML path paired with --workflow-path.",
    )
    parser.add_argument(
        "--reference-xml-root",
        default=None,
        help="Directory containing EMDB XML files named like emd-6287.xml.",
    )
    parser.add_argument(
        "--node-records-jsonl",
        default="workflow_node_records.jsonl",
        help="Output JSONL path for canonical node records.",
    )
    parser.add_argument(
        "--decision-records-jsonl",
        default="workflow_decision_records.jsonl",
        help="Output JSONL path for raw decision records.",
    )
    parser.add_argument(
        "--sft-jsonl",
        default="workflow_sft_data.jsonl",
        help="Output JSONL path for chat-format SFT samples.",
    )
    args = parser.parse_args()

    workflow_paths: list[Path] = []
    workflow_log_pairs: list[tuple[Path, Path | None]] = []

    if args.workflow_path or args.log_dir or args.reference_xml_path:
        if not (args.workflow_path and args.log_dir):
            raise SystemExit("Both --workflow-path and --log-dir must be provided together.")
        workflow_path = Path(args.workflow_path)
        log_dir = Path(args.log_dir)
        if not workflow_path.exists():
            raise SystemExit(f"Workflow file not found: {workflow_path}")
        if not log_dir.exists() or not log_dir.is_dir():
            raise SystemExit(f"Log directory not found: {log_dir}")
        workflow_log_pairs = [(workflow_path, log_dir)]
        workflow_paths = [workflow_path]
    else:
        if args.workflow_root:
            workflow_paths.extend(discover_workflow_paths_from_root(Path(args.workflow_root)))
        if args.workflows:
            workflow_paths.extend(discover_workflow_paths(args.workflows))
        workflow_paths = sorted(set(workflow_paths))
        workflow_paths = [path for path in workflow_paths if not path.name.endswith("-standard.json")]
        if workflow_paths:
            workflow_log_pairs = pair_workflows_with_logs(
                workflow_paths,
                Path(args.log_root) if args.log_root else None,
            )

    if not workflow_paths:
        raise SystemExit(
            "No workflow JSON files found. Use positional workflow paths, --workflow-root, or --workflow-path."
        )

    label_path = Path(args.workflow_labels)
    labels = load_workflow_labels(label_path) if label_path.exists() else {}
    explicit_reference_path = Path(args.reference_xml_path) if args.reference_xml_path else None
    explicit_reference_root = Path(args.reference_xml_root) if args.reference_xml_root else None

    all_node_records: list[dict[str, Any]] = []
    all_decision_records: list[dict[str, Any]] = []
    all_sft_records: list[dict[str, Any]] = []
    missing_log_dirs: list[str] = []
    missing_reference_xmls: list[str] = []

    for workflow_path, explicit_log_dir in workflow_log_pairs:
        workflow = load_workflow(workflow_path)
        dataset_id = workflow_dataset_id(workflow_path, workflow)
        dataset_label = labels.get(dataset_id, {"dataset_id": dataset_id})
        log_dir = explicit_log_dir
        if log_dir is None:
            missing_log_dirs.append(dataset_id)
        node_logs = discover_node_logs(log_dir)
        reference_xml_path = find_reference_xml(
            workflow_path,
            workflow,
            explicit_reference_root,
            explicit_reference_path if args.workflow_path else None,
        )
        reference_data = parse_emdb_xml(reference_xml_path) if reference_xml_path else None
        if reference_xml_path is None:
            missing_reference_xmls.append(dataset_id)
        node_records = build_node_records(
            workflow_path,
            workflow,
            dataset_label,
            node_logs,
            reference_data,
        )
        decision_records = build_decision_records_for_workflow(
            workflow_path=workflow_path,
            workflow=workflow,
            dataset_label=dataset_label,
            node_records=node_records,
        )

        all_node_records.extend(
            node_records[node_id] for node_id in sorted(node_records.keys(), key=numeric_job_sort_key)
        )
        all_decision_records.extend(decision_records)
        all_sft_records.extend(
            {
                "id": decision_record["sample_id"],
                "dataset_id": decision_record["dataset_id"],
                "workflow_id": decision_record["workflow_id"],
                "messages": build_chat_messages(
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

    write_jsonl(Path(args.node_records_jsonl), all_node_records)
    write_jsonl(Path(args.decision_records_jsonl), all_decision_records)
    write_jsonl(Path(args.sft_jsonl), all_sft_records)

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
