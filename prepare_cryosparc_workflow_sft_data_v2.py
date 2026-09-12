#!/usr/bin/env python3
"""Prepare V2 SFT data with dataset_context + current_state inputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from extract_last_node_info_v2 import build_last_node_info
from parse_emdb_reference_xml import parse_emdb_xml


JSON_LOG_RE = re.compile(r"^j(?P<node_num>\d+)\.json$", re.I)
EMPIAR_ID_RE = re.compile(r"empiar[-_]?(\d+)", re.I)
EMD_ID_RE = re.compile(r"EMD[-_]?(\d+)", re.I)
WORKFLOW_STEM_RE = re.compile(r"^(?P<digits>\d+)(?P<suffix>[A-Za-z]+)?$")


def _parse_empiar_like_id(value: str | int | None) -> tuple[str, str] | None:
    if value is None:
        return None
    text = str(value).strip()
    match = EMPIAR_ID_RE.search(text)
    if match:
        suffix_match = re.match(r"^([A-Za-z]+)", text[match.end():].strip())
        suffix = suffix_match.group(1).upper() if suffix_match else ""
        return (match.group(1), suffix)
    stem_match = WORKFLOW_STEM_RE.fullmatch(text)
    if stem_match:
        suffix = (stem_match.group("suffix") or "").upper()
        return (stem_match.group("digits"), suffix)
    return None


def normalize_dataset_id(value: str | int | None, preserve_suffix: bool = False) -> str | None:
    parsed = _parse_empiar_like_id(value)
    if parsed:
        digits, suffix = parsed
        suffix_text = suffix if preserve_suffix else ""
        return f"EMPIAR-{digits}{suffix_text}"

    if value is None:
        return None
    return str(value).strip().upper().replace("_", "-")


def numeric_job_sort_key(job_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", job_id)
    if match:
        return (int(match.group(1)), job_id)
    return (10**9, job_id)


def clean_json_text(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith('.'):
        stripped = stripped[1:].lstrip()
    start_positions = [pos for pos in (stripped.find('['), stripped.find('{')) if pos >= 0]
    if start_positions:
        stripped = stripped[min(start_positions):]
    return stripped


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def load_workflow_labels(path: Path) -> dict[str, dict[str, Any]]:
    raw = clean_json_text(path.read_text(encoding='utf-8'))
    data = json.loads(raw)
    labels: dict[str, dict[str, Any]] = {}
    for item in data:
        empiar_id = item.get('EMPIAR_ID')
        dataset_id = normalize_dataset_id(empiar_id)
        if not dataset_id:
            continue
        labels[dataset_id] = {
            'dataset_id': dataset_id,
            'empiar_id': empiar_id,
            'input_type': item.get('input'),
            'sample_type': item.get('types'),
            'target_resolution': item.get('resolution'),
            'num_of_maps': item.get('num_of_maps'),
        }
    return labels


def discover_batch_workflow_paths(workflow_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in workflow_root.glob('*.json')
            if path.is_file() and WORKFLOW_STEM_RE.fullmatch(path.stem)
        ),
        key=lambda path: (
            normalize_dataset_id(path.stem) or path.stem,
            normalize_dataset_id(path.stem, preserve_suffix=True) or path.stem,
        ),
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
        found[f"J{match.group('node_num')}"] = path
    return found


def load_workflow(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _sorted_workflow_node_ids(workflow: dict[str, Any]) -> list[str]:
    return sorted(workflow.get('jobs', {}).keys(), key=numeric_job_sort_key)


def _log_job_upstreams(job_documents: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    upstreams: dict[str, list[str]] = {}
    known_ids = set(job_documents.keys())
    for log_node_id, job_document in job_documents.items():
        parents: set[str] = set()
        for group in job_document.get('input_slot_groups', []) or []:
            for connection in group.get('connections', []) or []:
                parent_id = connection.get('job_uid')
                if parent_id in known_ids:
                    parents.add(parent_id)
        upstreams[log_node_id] = sorted(parents, key=numeric_job_sort_key)
    return upstreams


def build_workflow_to_log_node_id_map(
    workflow: dict[str, Any],
    node_logs: dict[str, Path],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    workflow_node_ids = _sorted_workflow_node_ids(workflow)
    log_node_ids = sorted(node_logs.keys(), key=numeric_job_sort_key)
    identity_map = {node_id: node_id for node_id in workflow_node_ids}
    if not workflow_node_ids:
        return {}, {}

    job_documents: dict[str, dict[str, Any]] = {}
    for log_node_id in log_node_ids:
        path = node_logs[log_node_id]
        job_documents[log_node_id] = json.loads(path.read_text(encoding='utf-8'))

    if workflow_node_ids == log_node_ids:
        return identity_map, job_documents

    if len(workflow_node_ids) != len(log_node_ids):
        return identity_map, job_documents

    workflow_batches = topological_batches(workflow_node_ids, workflow_upstreams(workflow))
    log_batches = topological_batches(log_node_ids, _log_job_upstreams(job_documents))
    if len(workflow_batches) != len(log_batches):
        return identity_map, job_documents

    mapping: dict[str, str] = {}
    for workflow_batch, log_batch in zip(workflow_batches, log_batches):
        if len(workflow_batch) != len(log_batch):
            return identity_map, job_documents

        workflow_sorted = sorted(workflow_batch, key=numeric_job_sort_key)
        log_sorted = sorted(log_batch, key=numeric_job_sort_key)
        workflow_types = sorted((workflow.get('jobs', {}).get(node_id, {}) or {}).get('jobType') for node_id in workflow_sorted)
        log_types = sorted((job_documents.get(node_id, {}) or {}).get('job_type') for node_id in log_sorted)
        if workflow_types != log_types:
            return identity_map, job_documents

        for workflow_node_id, log_node_id in zip(workflow_sorted, log_sorted):
            workflow_job_type = (workflow.get('jobs', {}).get(workflow_node_id, {}) or {}).get('jobType')
            log_job_type = (job_documents.get(log_node_id, {}) or {}).get('job_type')
            if workflow_job_type != log_job_type:
                return identity_map, job_documents
            mapping[workflow_node_id] = log_node_id

    return mapping, job_documents


def canonicalize_node_id(node_id: str | None, node_id_map: dict[str, str]) -> str | None:
    if node_id is None:
        return None
    return node_id_map.get(node_id, node_id)


def canonicalize_node_ids(node_ids: Iterable[str], node_id_map: dict[str, str]) -> list[str]:
    return [node_id_map.get(node_id, node_id) for node_id in node_ids]


def workflow_dataset_id(path: Path, workflow: dict[str, Any]) -> str:
    path_dataset_id = normalize_dataset_id(path.stem)
    if path_dataset_id and path_dataset_id.startswith('EMPIAR-'):
        return path_dataset_id
    title_dataset_id = normalize_dataset_id(workflow.get('title'))
    if title_dataset_id:
        return title_dataset_id
    raise ValueError(f'Could not infer dataset id from workflow: {path}')


def workflow_instance_id(path: Path, workflow: dict[str, Any]) -> str:
    path_instance_id = normalize_dataset_id(path.stem, preserve_suffix=True)
    if path_instance_id and path_instance_id.startswith('EMPIAR-'):
        return path_instance_id
    title_instance_id = normalize_dataset_id(workflow.get('title'), preserve_suffix=True)
    if title_instance_id:
        return title_instance_id
    raise ValueError(f'Could not infer workflow instance id from workflow: {path}')


def infer_reference_emd_id(workflow: dict[str, Any]) -> str | None:
    for _node_id, job in workflow.get('jobs', {}).items():
        title = str(job.get('title') or '')
        match = EMD_ID_RE.search(title)
        if match:
            return f"EMD-{match.group(1)}"
        for _param_name, meta in job.get('parameters', {}).items():
            value = str(meta.get('value') or '')
            match = EMD_ID_RE.search(value)
            if match:
                return f"EMD-{match.group(1)}"
    return None


def find_reference_xml(workflow_path: Path, workflow: dict[str, Any], explicit_reference_root: Path | None) -> Path | None:
    emd_id = infer_reference_emd_id(workflow)
    if emd_id is None:
        return None
    candidates: list[Path] = []
    if explicit_reference_root is not None:
        candidates.extend([explicit_reference_root / f"{emd_id.lower()}.xml", explicit_reference_root / f"{emd_id.upper()}.xml"])
    workflow_parent = workflow_path.parent
    candidates.extend([workflow_parent / f"{emd_id.lower()}.xml", workflow_parent / f"{emd_id.upper()}.xml"])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def workflow_upstreams(workflow: dict[str, Any]) -> dict[str, list[str]]:
    upstreams: dict[str, list[str]] = {}
    for node_id, job in workflow['jobs'].items():
        parents = []
        for group in job.get('groups', []):
            if not group:
                continue
            source = group[0]
            source_node = str(source).split('.', 1)[0]
            if source_node.startswith('J'):
                parents.append(source_node)
        upstreams[node_id] = sorted(set(parents), key=numeric_job_sort_key)
    return upstreams


def topological_batches(nodes: Iterable[str], upstreams: dict[str, list[str]]) -> list[list[str]]:
    remaining = set(nodes)
    completed: set[str] = set()
    batches: list[list[str]] = []
    while remaining:
        ready = sorted([node for node in remaining if set(upstreams.get(node, [])) <= completed], key=numeric_job_sort_key)
        if not ready:
            raise ValueError('Workflow graph contains a cycle or unresolved dependency.')
        batches.append(ready)
        completed.update(ready)
        remaining.difference_update(ready)
    return batches


def workflow_parameter_template(job: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, meta in job.get('parameters', {}).items():
        result[key] = meta.get('value')
    return result


def build_reason(batch: list[str], workflow: dict[str, Any], stop: bool = False) -> str:
    if stop:
        return 'Workflow is complete and no further action is required.'
    if len(batch) == 1:
        return f"Upstream dependencies are satisfied for {workflow['jobs'][batch[0]].get('jobType')}."
    job_types = ', '.join(workflow['jobs'][node_id].get('jobType') or 'unknown' for node_id in batch)
    return f"Multiple ready workflow nodes can run in parallel: {job_types}."


def simplify_selected_actions(selected_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'job_type': action.get('job_type'),
            'parameters': canonicalize(action.get('parameters', {})),
        }
        for action in selected_actions
    ]


def build_model_output(
    schema_version: str,
    decision_type: str,
    selected_actions: list[dict[str, Any]],
    rollback_target: dict[str, Any] | None,
    branch_plan: dict[str, Any] | None,
    reason: str,
    risk_flags: list[str],
    evidence: list[str],
    output_schema: str,
) -> dict[str, Any]:
    if output_schema == 'minimal_v3':
        return {
            'schema_version': '3.0',
            'decision_type': decision_type,
            'selected_actions': simplify_selected_actions(selected_actions),
        }

    return {
        'schema_version': schema_version,
        'decision_type': decision_type,
        'selected_actions': selected_actions,
        'rollback_target': rollback_target,
        'branch_plan': branch_plan,
        'reason': reason,
        'confidence': 0.99,
        'risk_flags': risk_flags,
        'evidence': evidence[:8],
    }


def build_chat_messages(model_input: dict[str, Any], model_output: dict[str, Any]) -> list[dict[str, str]]:
    system_prompt = (
        'You are a cryoSPARC workflow assistant. Return only valid JSON. '
        'The output must be exactly one JSON object with these top-level keys only: '
        'schema_version, decision_type, selected_actions. '
        'schema_version must be the string 3.0. '
        'decision_type must be one of: forward, branch, stop. '
        'selected_actions must be a list of objects, and each object must contain only: job_type and parameters. '
        'Do not output any other top-level keys such as decision, next_action, action, reason, explanation, confidence, evidence, rollback_target, branch_plan, or workflow_node_id. '
        'Do not output markdown or prose.'
    )
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': json.dumps(model_input, ensure_ascii=False, indent=2)},
        {'role': 'assistant', 'content': json.dumps(model_output, ensure_ascii=False, indent=2)},
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write('\n')


def _normalize_resolution(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def build_dataset_abstract(reference_data: dict[str, Any] | None) -> str | None:
    if not reference_data:
        return None

    summary = reference_data.get("summary") or {}
    citation = reference_data.get("citation") or {}
    sample = reference_data.get("sample") or {}

    parts: list[str] = []
    if citation.get("title"):
        parts.append(str(citation["title"]).strip())
    elif summary.get("title"):
        parts.append(str(summary["title"]).strip())

    if sample.get("name"):
        parts.append(f"Sample: {sample['name']}.")
    if sample.get("symmetry"):
        parts.append(f"Symmetry: {sample['symmetry']}.")
    if summary.get("resolution_a") is not None:
        parts.append(f"Resolution: {summary['resolution_a']} A.")
    if citation.get("journal") and citation.get("year"):
        parts.append(f"Published in {citation['journal']} ({citation['year']}).")

    text = " ".join(part.strip() for part in parts if part)
    return text or None


def build_known_workflow_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    upstreams = workflow_upstreams(workflow)
    steps: list[dict[str, Any]] = []
    for step_index, (node_id, job) in enumerate(
        sorted(workflow["jobs"].items(), key=lambda item: numeric_job_sort_key(item[0]))
    ):
        steps.append(
            {
                "step_index": step_index,
                "node_id": node_id,
                "action": job.get("jobType"),
                "title": job.get("title"),
                "description": job.get("description"),
                "upstream_node_ids": upstreams.get(node_id, []),
                "parameter_template": workflow_parameter_template(job),
            }
        )
    return steps


def _dataset_fact_job_types() -> set[str]:
    return {
        "import_micrographs",
        "import_movies",
        "import_particles",
        "import_volumes",
    }


def build_dataset_parameter_facts(workflow: dict[str, Any]) -> dict[str, Any]:
    facts_by_job_type: dict[str, dict[str, Any]] = {}
    merged_facts: dict[str, Any] = {}

    for _node_id, job in sorted(workflow.get("jobs", {}).items(), key=lambda item: numeric_job_sort_key(item[0])):
        job_type = job.get("jobType")
        if job_type not in _dataset_fact_job_types():
            continue

        params = canonicalize(workflow_parameter_template(job))
        facts_by_job_type[job_type] = params
        for key, value in params.items():
            if key not in merged_facts:
                merged_facts[key] = value

    return {
        "job_type_facts": facts_by_job_type,
        "merged_facts": canonicalize(merged_facts),
    }


def build_dataset_info(
    dataset_id: str,
    dataset_label: dict[str, Any],
    reference_data: dict[str, Any] | None,
    workflow: dict[str, Any],
    include_known_workflow: bool = True,
) -> dict[str, Any]:
    empiar_id = dataset_label.get("empiar_id")
    emdb_id = (
        reference_data.get("reference_id")
        if reference_data
        else infer_reference_emd_id(workflow)
    )
    return {
        "empiar_id": dataset_id,
        "emdb_id": emdb_id,
        "resolution": _normalize_resolution(dataset_label.get("target_resolution")),
        "input_type": dataset_label.get("input_type"),
        "macromolecules_type": dataset_label.get("sample_type"),
        "num_of_maps": dataset_label.get("num_of_maps"),
        "abstract": build_dataset_abstract(reference_data),
        "known_workflow_steps": build_known_workflow_steps(workflow) if include_known_workflow else None,
        "label_empiar_id": empiar_id,
    }


def build_dataset_context(
    dataset_id: str,
    dataset_label: dict[str, Any],
    reference_data: dict[str, Any] | None,
    workflow: dict[str, Any],
    include_known_workflow: bool = True,
) -> dict[str, Any]:
    dataset_info = build_dataset_info(
        dataset_id=dataset_id,
        dataset_label=dataset_label,
        reference_data=reference_data,
        workflow=workflow,
        include_known_workflow=include_known_workflow,
    )
    dataset_parameter_facts = build_dataset_parameter_facts(workflow)
    return {
        "dataset_metadata": dataset_info,
        "dataset_parameter_facts": dataset_parameter_facts["merged_facts"],
        "dataset_parameter_facts_by_job_type": dataset_parameter_facts["job_type_facts"],
    }


def _collect_output_group_names(last_node_info: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for group in last_node_info.get("outputs", {}).get("groups", []) or []:
        name = group.get("name")
        if name:
            names.append(str(name))
    return sorted(set(names))


def _collect_output_field_names(last_node_info: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for group in last_node_info.get("outputs", {}).get("groups", []) or []:
        for field_name in group.get("field_names", []) or []:
            if field_name:
                names.append(str(field_name))
    return sorted(set(names))


def build_state_features(last_node_info: dict[str, Any]) -> dict[str, Any]:
    output_group_names = _collect_output_group_names(last_node_info)
    output_field_names = _collect_output_field_names(last_node_info)
    metrics = last_node_info.get("metrics", {}) or {}

    output_group_name_set = set(output_group_names)
    output_field_name_set = set(output_field_names)

    return {
        "has_templates": any(name.startswith("templates") or name == "templates" for name in output_group_names),
        "has_ctf": "ctf" in output_field_name_set,
        "has_ctf_stats": "ctf_stats" in output_field_name_set,
        "has_initial_volume": any("volume" in name for name in output_group_names) or "map" in output_field_name_set,
        "has_selected_particles": "particles_selected" in output_group_name_set,
        "has_particle_alignments_2d": "alignments2D" in output_field_name_set,
        "has_particle_alignments_3d": "alignments3D" in output_field_name_set,
        "has_filament_metadata": "filament" in output_field_name_set,
        "micrograph_count": metrics.get("micrograph_count"),
        "particle_count": metrics.get("particle_count"),
        "selected_particle_count": metrics.get("selected_particle_count"),
        "rejected_particle_count": metrics.get("rejected_particle_count"),
        "volume_count": metrics.get("volume_count"),
        "class_count": metrics.get("class_count"),
        "mask_count": metrics.get("mask_count"),
        "last_output_group_names": output_group_names,
        "last_output_field_names": output_field_names,
    }


def _build_current_state_not_started(workflow: dict[str, Any]) -> dict[str, Any]:
    last_node_info = {
        "job_type": None,
        "job_uid": None,
        "job_title": workflow.get("title"),
        "project_uid": None,
        "status": "not_started",
        "timestamps": {},
        "inputs": {"groups": []},
        "parameters": {},
        "outputs": {"groups": []},
        "metrics": {},
        "runtime": {},
        "evidence_text": ["Workflow has not started yet."],
        "warning_lines": [],
        "image_refs": {
            "ui_tile_images": [],
            "output_group_images": {},
            "event_images": [],
            "event_image_count": 0,
            "event_image_kind_counts": {},
        },
    }
    return {
        "last_node_id": None,
        "last_action": None,
        "last_node_status": "not_started",
        "last_node_info": last_node_info,
        "state_features": build_state_features(last_node_info),
    }


def _build_fallback_last_node_info(node_id: str, workflow: dict[str, Any], canonical_node_id: str | None = None) -> dict[str, Any]:
    job = workflow.get("jobs", {}).get(node_id, {})
    resolved_node_id = canonical_node_id or node_id
    return {
        "job_type": job.get("jobType"),
        "job_uid": resolved_node_id,
        "job_title": job.get("title") or workflow.get("title"),
        "project_uid": None,
        "status": "completed",
        "timestamps": {},
        "inputs": {"groups": job.get("groups", [])},
        "parameters": workflow_parameter_template(job),
        "outputs": {"groups": []},
        "metrics": {},
        "runtime": {},
        "evidence_text": [f"No exported runtime log was available for {resolved_node_id}; using workflow template information."],
        "warning_lines": ["missing_runtime_log"],
        "image_refs": {
            "ui_tile_images": [],
            "output_group_images": {},
            "event_images": [],
            "event_image_count": 0,
            "event_image_kind_counts": {},
        },
    }


def _build_current_state_from_previous_batch(
    previous_batch: list[str],
    node_runtime_records: dict[str, dict[str, Any]],
    workflow: dict[str, Any],
    node_id_map: dict[str, str],
) -> dict[str, Any]:
    if not previous_batch:
        return _build_current_state_not_started(workflow)

    representative_node_id = previous_batch[-1]
    representative = None
    for node_id in reversed(previous_batch):
        candidate = node_runtime_records.get(node_id)
        if candidate is not None:
            representative_node_id = canonicalize_node_id(node_id, node_id_map) or node_id
            representative = dict(candidate)
            break

    if representative is None:
        representative = _build_fallback_last_node_info(
            previous_batch[-1],
            workflow,
            canonical_node_id=canonicalize_node_id(previous_batch[-1], node_id_map),
        )
        representative_node_id = representative.get("job_uid") or representative_node_id

    current_state = {
        "last_node_id": representative_node_id,
        "last_action": representative.get("job_type"),
        "last_node_status": representative.get("status") or "completed",
        "last_node_info": representative,
    }
    current_state["last_node_info"]["recent_batch_node_ids"] = canonicalize_node_ids(previous_batch, node_id_map)
    current_state["state_features"] = build_state_features(current_state["last_node_info"])
    return current_state


def build_v2_model_input(
    dataset_context: dict[str, Any],
    current_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "2.1",
        "task_type": "workflow_decision",
        "dataset_context": dataset_context,
        "current_state": current_state,
    }


def build_v2_decision_records_for_workflow(
    workflow_path: Path,
    workflow: dict[str, Any],
    dataset_context: dict[str, Any],
    node_runtime_records: dict[str, dict[str, Any]],
    workflow_instance_id: str,
    node_id_map: dict[str, str],
    output_schema: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    upstreams = workflow_upstreams(workflow)
    batches = topological_batches(workflow["jobs"].keys(), upstreams)
    previous_batch: list[str] = []

    for batch_index, batch_nodes in enumerate(batches):
        current_state = _build_current_state_from_previous_batch(previous_batch, node_runtime_records, workflow, node_id_map)
        model_input = build_v2_model_input(dataset_context, current_state)

        decision_type = "branch" if len(batch_nodes) > 1 else "forward"
        selected_actions = []
        for node_id in batch_nodes:
            job = workflow["jobs"][node_id]
            selected_actions.append(
                {
                    "action_id": f"{decision_type}_{canonicalize_node_id(node_id, node_id_map) or node_id}",
                    "action_type": decision_type,
                    "workflow_node_id": canonicalize_node_id(node_id, node_id_map) or node_id,
                    "job_type": job.get("jobType"),
                    "parameters": workflow_parameter_template(job),
                }
            )

        branch_plan = None
        if decision_type == "branch":
            branch_plan = {
                "branch_type": "parallel_required_nodes",
                "max_parallel_branches": len(selected_actions),
                "notes": "These workflow nodes are concurrently ready in the known workflow.",
            }

        model_output = build_model_output(
            schema_version="1.0",
            decision_type=decision_type,
            selected_actions=selected_actions,
            rollback_target=None,
            branch_plan=branch_plan,
            reason=build_reason(batch_nodes, workflow),
            risk_flags=current_state["last_node_info"].get("warning_lines", []),
            evidence=current_state["last_node_info"].get("evidence_text", []),
            output_schema=output_schema,
        )

        records.append(
            {
                "sample_id": f"{workflow_instance_id}:v2_step:{batch_index:03d}",
                "data_source": "real_success_v2",
                "dataset_id": dataset_context["dataset_metadata"]["empiar_id"],
                "workflow_id": workflow.get("_id") or f"wf_{workflow_instance_id.lower().replace('-', '_')}",
                "workflow_instance_id": workflow_instance_id,
                "workflow_title": workflow.get("title"),
                "decision_index": batch_index,
                "decision_nodes": canonicalize_node_ids(batch_nodes, node_id_map),
                "model_input": model_input,
                "model_output": model_output,
            }
        )

        previous_batch = batch_nodes

    stop_state = _build_current_state_from_previous_batch(previous_batch, node_runtime_records, workflow, node_id_map)
    stop_input = build_v2_model_input(dataset_context, stop_state)
    stop_output = build_model_output(
        schema_version="1.0",
        decision_type="stop",
        selected_actions=[],
        rollback_target=None,
        branch_plan=None,
        reason="Workflow is complete and no further action is required.",
        risk_flags=stop_state["last_node_info"].get("warning_lines", []),
        evidence=stop_state["last_node_info"].get("evidence_text", []),
        output_schema=output_schema,
    )
    records.append(
        {
            "sample_id": f"{workflow_instance_id}:v2_step:{len(batches):03d}",
            "data_source": "real_success_v2",
            "dataset_id": dataset_context["dataset_metadata"]["empiar_id"],
            "workflow_id": workflow.get("_id") or f"wf_{workflow_instance_id.lower().replace('-', '_')}",
            "workflow_instance_id": workflow_instance_id,
            "workflow_title": workflow.get("title"),
            "decision_index": len(batches),
            "decision_nodes": [],
            "model_input": stop_input,
            "model_output": stop_output,
        }
    )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--workflow-labels", required=True)
    parser.add_argument("--reference-xml-root", required=True)
    parser.add_argument("--max-workflows", type=int, default=None)
    parser.add_argument("--hide-known-workflow", action="store_true")
    parser.add_argument("--output-schema", choices=["full_v1", "minimal_v3"], default="minimal_v3")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    workflow_root = Path(args.workflow_root)
    log_root = Path(args.log_root)
    labels_path = Path(args.workflow_labels)
    reference_root = Path(args.reference_xml_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    workflow_paths = discover_batch_workflow_paths(workflow_root)
    if args.max_workflows is not None:
        workflow_paths = workflow_paths[: args.max_workflows]

    labels = load_workflow_labels(labels_path)
    all_records: list[dict[str, Any]] = []
    all_sft_records: list[dict[str, Any]] = []

    for workflow_path in workflow_paths:
        workflow = load_workflow(workflow_path)
        dataset_id = workflow_dataset_id(workflow_path, workflow)
        workflow_id = workflow_instance_id(workflow_path, workflow)
        dataset_label = labels.get(dataset_id, {"dataset_id": dataset_id})
        log_dir = log_root / workflow_path.stem
        node_logs = discover_batch_node_logs(log_dir)
        node_id_map, job_documents = build_workflow_to_log_node_id_map(workflow, node_logs)

        reference_xml_path = find_reference_xml(workflow_path=workflow_path, workflow=workflow, explicit_reference_root=reference_root)
        reference_data = parse_emdb_xml(reference_xml_path) if reference_xml_path else None
        dataset_context = build_dataset_context(
            dataset_id,
            dataset_label,
            reference_data,
            workflow,
            include_known_workflow=not args.hide_known_workflow,
        )

        node_runtime_records: dict[str, dict[str, Any]] = {}
        for workflow_node_id, log_node_id in node_id_map.items():
            job_document = job_documents.get(log_node_id)
            if job_document is None:
                path = node_logs.get(log_node_id)
                if path is None:
                    continue
                job_document = json.loads(path.read_text(encoding="utf-8"))
            node_runtime_records[workflow_node_id] = build_last_node_info(job_document)

        decision_records = build_v2_decision_records_for_workflow(
            workflow_path=workflow_path,
            workflow=workflow,
            dataset_context=dataset_context,
            node_runtime_records=node_runtime_records,
            workflow_instance_id=workflow_id,
            node_id_map=node_id_map,
            output_schema=args.output_schema,
        )

        all_records.extend(decision_records)
        all_sft_records.extend(
            {
                "id": record["sample_id"],
                "dataset_id": record["dataset_id"],
                "workflow_id": record["workflow_id"],
                "messages": build_chat_messages(record["model_input"], record["model_output"]),
                "metadata": {
                    "data_source": record["data_source"],
                    "decision_index": record["decision_index"],
                    "decision_nodes": record["decision_nodes"],
                    "workflow_instance_id": record["workflow_instance_id"],
                },
            }
            for record in decision_records
        )

    records_path = output_root / "workflow_decision_records_v2.jsonl"
    sft_path = output_root / "workflow_sft_data_v2.jsonl"
    write_jsonl(records_path, all_records)
    write_jsonl(sft_path, all_sft_records)

    print(
        json.dumps(
            {
                "workflow_count": len(workflow_paths),
                "decision_record_count": len(all_records),
                "sft_record_count": len(all_sft_records),
                "decision_records_jsonl": str(records_path.resolve()),
                "sft_jsonl": str(sft_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
