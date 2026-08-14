# Builds model-facing workflow decision payloads from CryoSPARC state.
from typing import Any

from job_result import TERMINAL_STATUSES, build_internal_status_package
from workflow_state import extract_workflow_state, find_node, job_uid_number


MODEL_INPUT_SCHEMA_VERSION = "2.1"
MODEL_TASK_TYPE = "workflow_decision"

FAILURE_STATUSES = {"failed", "killed", "aborted"}
INCOMPLETE_ATTEMPT_STATUSES = {"building"}
HISTORY_FAILURE_STATUSES = FAILURE_STATUSES | INCOMPLETE_ATTEMPT_STATUSES

MICROGRAPH_OUTPUT_NAMES = {"micrographs", "imported_micrographs", "exposures"}
PARTICLE_OUTPUT_NAMES = {"particles", "particles_selected"}
SELECTED_PARTICLE_OUTPUT_NAMES = {"particles_selected", "selected_particles"}
REJECTED_PARTICLE_OUTPUT_NAMES = {
    "particles_rejected",
    "particles_excluded",
    "particles_unused",
}
VOLUME_OUTPUT_NAMES = {"volume", "volumes", "map", "maps"}
CLASS_OUTPUT_NAMES = {"classes", "class_averages", "templates"}
MASK_OUTPUT_NAMES = {"mask", "masks"}

CTF_FIELD_NAMES = {"ctf"}
CTF_STATS_FIELD_NAMES = {"ctf_stats"}
TEMPLATE_NAMES = {"templates", "template", "class_averages"}
VOLUME_FIELD_NAMES = {"map", "volume", "volume_blob"}
ALIGNMENTS_2D_FIELD_NAMES = {"alignments2D", "alignments_2d"}
ALIGNMENTS_3D_FIELD_NAMES = {"alignments3D", "alignments_3d", "alignments3D_multi"}
FILAMENT_FIELD_HINTS = ("filament", "helix", "helical")


def build_model_input_payload(
    project_uid: str,
    workspace_uid: str,
    current_job_uid: str | None = None,
    dataset_info: dict[str, Any] | None = None,
    known_workflow_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build the model payload for the model, or internal status if the job is active.

    The model-facing payload is returned only when the current job is
    completed/failed/killed or when the workflow has not started.
    """
    workflow_state = extract_workflow_state(project_uid, workspace_uid)
    node = resolve_current_node(workflow_state, current_job_uid)
    dataset_context = build_dataset_context(
        dataset_info or {},
        known_workflow_dirs=known_workflow_dirs,
    )

    if node is None:
        return build_not_started_payload(project_uid, workspace_uid, dataset_context)

    if node["status"] not in TERMINAL_STATUSES:
        package = build_internal_status_package(workflow_state, node)
        package["model_input_schema_version"] = MODEL_INPUT_SCHEMA_VERSION
        return package

    context_nodes = filter_model_context_nodes(workflow_state, node)
    return {
        "schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "task_type": MODEL_TASK_TYPE,
        "dataset_context": dataset_context,
        "current_state": build_current_state(
            workflow_state=workflow_state,
            current_node=node,
            context_nodes=context_nodes,
            project_uid=project_uid,
            workspace_uid=workspace_uid,
        ),
    }


def build_dataset_context(
    dataset_info: dict[str, Any],
    known_workflow_dirs: list[str] | None,
) -> dict[str, Any]:
    """Normalize dataset metadata and facts for the v2.1 model contract."""
    _ = known_workflow_dirs
    metadata = {
        "empiar_id": dataset_info.get("empiar_id"),
        "emdb_id": dataset_info.get("emdb_id"),
        "resolution": dataset_info.get("resolution"),
        "input_type": dataset_info.get("input_type"),
        "macromolecules_type": dataset_info.get("macromolecules_type"),
        "num_of_maps": dataset_info.get("num_of_maps"),
        "abstract": dataset_info.get("abstract"),
        "known_workflow_steps": dataset_info.get("known_workflow_steps"),
        "label_empiar_id": dataset_info.get("label_empiar_id"),
    }
    facts = build_dataset_parameter_facts(dataset_info)
    return {
        "dataset_metadata": metadata,
        "dataset_parameter_facts": facts,
        "dataset_parameter_facts_by_job_type": build_dataset_parameter_facts_by_job_type(
            facts
        ),
    }


def build_dataset_parameter_facts(dataset_info: dict[str, Any]) -> dict[str, Any]:
    """Expose only facts that were supplied or previously extracted."""
    aliases = {
        "accel_kv": ("accel_kv", "accelerating_voltage_kv"),
        "blob_paths": ("blob_paths",),
        "blob_exists": ("blob_exists",),
        "cs_mm": ("cs_mm", "spherical_aberration_mm"),
        "ctf_exists": ("ctf_exists",),
        "enable_validation": ("enable_validation",),
        "particle_meta_path": ("particle_meta_path",),
        "particle_blob_path": ("particle_blob_path",),
        "psize_A": ("psize_A", "pixel_size_A"),
        "total_dose_e_per_A2": ("total_dose_e_per_A2", "total_exposure_dose_e_per_A2"),
        "volume_blob_path": ("volume_blob_path",),
    }
    facts: dict[str, Any] = {}
    for target_key, source_keys in aliases.items():
        value = first_present(dataset_info, source_keys)
        if value is not None:
            facts[target_key] = value
    return facts


def first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def build_dataset_parameter_facts_by_job_type(
    facts: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Group confirmed facts by the job types that commonly consume them."""
    grouped: dict[str, dict[str, Any]] = {}

    volume_facts = subset(
        facts,
        ("volume_blob_path",),
    )
    if volume_facts:
        grouped["import_volumes"] = volume_facts

    particle_facts = subset(
        facts,
        (
            "accel_kv",
            "blob_exists",
            "cs_mm",
            "ctf_exists",
            "enable_validation",
            "particle_meta_path",
            "particle_blob_path",
            "psize_A",
        ),
    )
    if particle_facts:
        grouped["import_particles"] = particle_facts

    micrograph_facts = subset(
        facts,
        (
            "accel_kv",
            "blob_paths",
            "cs_mm",
            "psize_A",
            "total_dose_e_per_A2",
        ),
    )
    if micrograph_facts:
        grouped["import_micrographs"] = micrograph_facts
        grouped["import_movies"] = micrograph_facts.copy()

    return grouped


def subset(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in keys
        if key in payload
    }


def resolve_current_node(
    workflow_state: dict[str, Any],
    current_job_uid: str | None,
) -> dict[str, Any] | None:
    """Find the requested node or fall back to the latest job in the workspace."""
    if current_job_uid:
        return find_node(workflow_state, current_job_uid)
    if not workflow_state["nodes"]:
        return None
    return max(
        workflow_state["nodes"],
        key=lambda node: job_uid_number(node["cryosparc_job_uid"]),
    )


def build_not_started_payload(
    project_uid: str,
    workspace_uid: str,
    dataset_context: dict[str, Any],
) -> dict[str, Any]:
    """Build a valid payload for workflows that have not started."""
    return {
        "schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "task_type": MODEL_TASK_TYPE,
        "dataset_context": dataset_context,
        "current_state": {
            "last_node_id": None,
            "last_action": None,
            "last_node_status": "not_started",
            "state_features": empty_state_features(),
            "recent_job_history": [],
            "last_node_info": {
                "job_type": None,
                "job_uid": None,
                "job_title": None,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "status": "not_started",
                "timestamps": empty_timestamps(),
                "inputs": {"groups": []},
                "parameters": {},
                "outputs": {"groups": []},
                "metrics": {},
                "runtime": empty_runtime(),
                "evidence_text": ["Workflow has not started yet."],
                "warning_lines": [],
                "image_refs": empty_image_refs(),
                "recent_batch_node_ids": [],
                "error_info": None,
            },
        },
    }


def build_current_state(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    context_nodes: list[dict[str, Any]],
    project_uid: str,
    workspace_uid: str,
) -> dict[str, Any]:
    return {
        "last_node_id": current_node["workflow_node_id"],
        "last_action": current_node["job_type"],
        "last_node_status": normalize_model_status(current_node["status"]),
        "last_node_info": build_last_node_info(
            current_node,
            project_uid=project_uid,
            workspace_uid=workspace_uid,
        ),
        "state_features": derive_state_features(current_node),
        "recent_job_history": build_recent_job_history(
            context_nodes,
            limit=6,
        ),
    }


def normalize_model_status(status: str) -> str:
    if status == "completed":
        return "completed"
    if status in FAILURE_STATUSES:
        return "failure"
    if status == "not_started":
        return "not_started"
    return "failure"


def filter_model_context_nodes(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep the current branch and failure attempts without unrelated branches."""
    ancestor_ids = collect_ancestor_ids(workflow_state, current_node)
    relevant_ids = ancestor_ids | {current_node["workflow_node_id"]}
    relevant_numbers = [
        job_uid_number(job_uid)
        for job_uid in relevant_ids
    ]
    min_relevant_number = min(relevant_numbers) if relevant_numbers else 0
    current_number = job_uid_number(current_node["cryosparc_job_uid"])

    filtered = []
    for node in workflow_state.get("nodes", []):
        node_id = node["workflow_node_id"]
        node_number = job_uid_number(node["cryosparc_job_uid"])
        if node_id in relevant_ids:
            filtered.append(node)
            continue
        if is_failure_attempt(node) and node_number >= min_relevant_number:
            filtered.append(node)
            continue
        if is_failure_attempt(node) and node_number > current_number:
            filtered.append(node)

    return sorted(
        filtered,
        key=lambda item: job_uid_number(item["cryosparc_job_uid"]),
    )


def collect_ancestor_ids(
    workflow_state: dict[str, Any],
    node: dict[str, Any],
) -> set[str]:
    by_uid = {
        item["workflow_node_id"]: item
        for item in workflow_state.get("nodes", [])
    }
    pending = list(node.get("parent_workflow_node_ids") or [])
    ancestors: set[str] = set()
    while pending:
        node_id = pending.pop()
        if node_id in ancestors:
            continue
        ancestors.add(node_id)
        parent = by_uid.get(node_id)
        if parent:
            pending.extend(parent.get("parent_workflow_node_ids") or [])
    return ancestors


def is_failure_attempt(node: dict[str, Any]) -> bool:
    return (
        node.get("status") in HISTORY_FAILURE_STATUSES
        or bool(node.get("has_error"))
        or bool(extract_node_error_text(node))
    )


def build_recent_job_history(
    context_nodes: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Build the compact success/failure history required by schema 2.1."""
    history = []
    attempt_counts: dict[str, int] = {}
    for node in context_nodes:
        status = normalize_history_status(node)
        if status is None:
            continue
        job_type = node.get("job_type")
        attempt_counts[job_type] = attempt_counts.get(job_type, 0) + 1
        history.append(
            {
                "job_uid": node.get("cryosparc_job_uid"),
                "job_type": job_type,
                "attempt_index": attempt_counts[job_type],
                "status": status,
                "key_parameters": node.get("key_parameters") or {},
                "output_summary": build_output_summary(node),
                "error_summary": build_error_summary(node) if status == "failure" else None,
            }
        )
    return history[-limit:]


def normalize_history_status(node: dict[str, Any]) -> str | None:
    if node.get("status") == "completed":
        return "completed"
    if is_failure_attempt(node):
        return "failure"
    return None


def build_last_node_info(
    node: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
) -> dict[str, Any]:
    """Convert a normalized CryoSPARC node into model-facing last_node_info."""
    return {
        "job_type": node["job_type"],
        "job_uid": node["cryosparc_job_uid"],
        "job_title": node["title"],
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "status": node["status"],
        "timestamps": build_timestamps(node),
        "inputs": build_input_groups(node),
        "parameters": node["key_parameters"],
        "outputs": build_output_groups(node),
        "metrics": build_metrics(node),
        "runtime": build_runtime(node),
        "evidence_text": build_evidence_text(node),
        "warning_lines": build_warning_lines(node),
        "image_refs": empty_image_refs(),
        "recent_batch_node_ids": node["parent_workflow_node_ids"],
        "error_info": build_error_info(node),
    }


def build_timestamps(node: dict[str, Any]) -> dict[str, Any]:
    """Expose the agreed timestamp fields, using null when unavailable."""
    timestamps = node.get("timestamps") or {}
    return {
        "created_at": timestamps.get("created_at"),
        "queued_at": timestamps.get("queued_at"),
        "started_at": timestamps.get("started_at"),
        "running_at": timestamps.get("running_at"),
        "completed_at": timestamps.get("completed_at"),
        "failed_at": timestamps.get("failed_at"),
        "killed_at": timestamps.get("killed_at"),
        "updated_at": timestamps.get("updated_at") or node["updated_at"],
    }


def empty_timestamps() -> dict[str, Any]:
    """Return all agreed timestamp fields for not-started workflows."""
    return {
        "created_at": None,
        "queued_at": None,
        "started_at": None,
        "running_at": None,
        "completed_at": None,
        "failed_at": None,
        "killed_at": None,
        "updated_at": None,
    }


def build_runtime(node: dict[str, Any]) -> dict[str, Any]:
    """Expose agreed runtime/resource fields from the normalized node."""
    runtime = node.get("runtime") or {}
    result = empty_runtime()
    result.update(
        {
            key: runtime.get(key)
            for key in result
        }
    )
    return result


def empty_runtime() -> dict[str, Any]:
    """Return all agreed runtime fields with null defaults."""
    return {
        "work_dir": None,
        "lane": None,
        "worker_hostname": None,
        "allocated_cpu": None,
        "allocated_gpu": None,
        "allocated_ram": None,
        "allocated_ssd": None,
    }


def build_input_groups(node: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert CryoSPARC input connections into compact V2 input groups."""
    groups = []
    for name, connections in node["inputs"].items():
        groups.append(
            {
                "name": name,
                "title": name,
                "count_min": 0,
                "count_max": None,
                "slot_names": sorted(
                    {
                        result_name
                        for connection in connections
                        for result_name in connection["result_names"]
                    }
                ),
                "connected_job_uids": [
                    connection["source_job_uid"]
                    for connection in connections
                ],
            }
        )
    return {"groups": groups}


def build_output_groups(node: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Convert CryoSPARC outputs into compact V2 output groups."""
    groups = []
    for name, output in node["outputs"].items():
        groups.append(
            {
                "name": name,
                "description": f"CryoSPARC output group {name}.",
                "num_items": output["num_items"],
                "field_names": output["result_names"],
                "scalar_stats": {},
                "summary_stat_keys": output["summary_keys"],
            }
        )
    return {"groups": groups}


def build_metrics(node: dict[str, Any]) -> dict[str, Any]:
    """Expose simple output counts that are useful for model decisions."""
    features = derive_state_features(node)
    metrics = {
        "completed": node["status"] == "completed",
        "failed": node["status"] in FAILURE_STATUSES,
        "has_error": node["has_error"],
        "has_warning": node["has_warning"],
        "micrograph_count": features["micrograph_count"],
        "particle_count": features["particle_count"],
        "selected_particle_count": features["selected_particle_count"],
        "rejected_particle_count": features["rejected_particle_count"],
        "volume_count": features["volume_count"],
        "class_count": features["class_count"],
        "mask_count": features["mask_count"],
    }
    for name, output in node["outputs"].items():
        metrics[f"{name}_count"] = output["num_items"]
    return metrics


def build_output_summary(node: dict[str, Any]) -> dict[str, Any]:
    groups = output_groups_from_node(node)
    return {
        "output_group_names": [
            group["name"]
            for group in groups
            if group.get("name")
        ],
        "output_field_names": flatten_field_names(groups),
        "num_items_by_group": {
            group["name"]: group.get("num_items")
            for group in groups
            if group.get("name")
        },
    }


def build_error_summary(node: dict[str, Any]) -> dict[str, Any]:
    code = error_code_for_node(node)
    message = extract_node_error_text(node) or f"Job {node.get('cryosparc_job_uid')} status is {node.get('status')}."
    return {
        "error_code": code,
        "message": trim_text(message, max_chars=500),
    }


def build_error_info(node: dict[str, Any]) -> dict[str, Any] | None:
    if normalize_model_status(node.get("status")) != "failure" and not node.get("has_error"):
        return None
    raw_text = extract_node_error_text(node)
    truncated_text = trim_text(raw_text, max_chars=2000) if raw_text else None
    return {
        "error_type": "CryoSPARCJobError",
        "error_code": error_code_for_node(node),
        "message": truncated_text or f"Job status is {node.get('status')}.",
        "raw_error_text": truncated_text,
        "raw_error_truncated": bool(raw_text and len(raw_text) > 2000),
        "stderr_tail": None,
        "stderr_truncated": False,
        "log_path": None,
        "source": "cryosparc_job_log" if raw_text else "mcp_tool_execution",
    }


def error_code_for_node(node: dict[str, Any]) -> str:
    status = node.get("status")
    if status == "killed":
        return "CRYOSPARC_JOB_KILLED"
    if status in {"failed", "aborted"}:
        return "CRYOSPARC_JOB_FAILED"
    if status in INCOMPLETE_ATTEMPT_STATUSES:
        return "INCOMPLETE_JOB_ATTEMPT"
    if node.get("has_error"):
        return "CRYOSPARC_JOB_ERROR"
    return "UNKNOWN_JOB_ERROR"


def extract_node_error_text(node: dict[str, Any]) -> str:
    run_errors = node.get("run_errors") or {}
    parts: list[str] = []
    collect_text(run_errors, parts)
    return "\n".join(part for part in parts if part)


def collect_text(value: Any, parts: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        if value.strip():
            parts.append(value.strip())
        return
    if isinstance(value, dict):
        for nested in value.values():
            collect_text(nested, parts)
        return
    if isinstance(value, list):
        for nested in value:
            collect_text(nested, parts)
        return
    parts.append(str(value))


def trim_text(text: str, max_chars: int) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def output_groups_from_node(node: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = node.get("outputs") or {}
    if isinstance(outputs, dict) and isinstance(outputs.get("groups"), list):
        return outputs["groups"]

    groups = []
    if isinstance(outputs, dict):
        for name, output in outputs.items():
            if not isinstance(output, dict):
                continue
            groups.append(
                {
                    "name": name,
                    "num_items": output.get("num_items"),
                    "field_names": output.get("field_names")
                    or output.get("result_names")
                    or [],
                    "summary_stat_keys": output.get("summary_stat_keys")
                    or output.get("summary_keys")
                    or output.get("latest_summary_stat_keys")
                    or [],
                }
            )
    return groups


def output_name_matches(name: str, names: set[str]) -> bool:
    if name in names:
        return True
    if name.startswith("volume_") or name.startswith("volumes_"):
        return bool(names & VOLUME_OUTPUT_NAMES)
    if name.startswith("mask_"):
        return bool(names & MASK_OUTPUT_NAMES)
    if name.startswith("templates_"):
        return bool(names & CLASS_OUTPUT_NAMES)
    if name.startswith("particles_rejected") or name.startswith("particles_excluded"):
        return bool(names & REJECTED_PARTICLE_OUTPUT_NAMES)
    return False


def count_by_names(groups: list[dict[str, Any]], names: set[str]) -> int | None:
    matching = [
        group.get("num_items")
        for group in groups
        if output_name_matches(str(group.get("name")), names)
        and isinstance(group.get("num_items"), int)
    ]
    if not matching:
        return None
    return sum(matching)


def flatten_field_names(groups: list[dict[str, Any]]) -> list[str]:
    fields: set[str] = set()
    for group in groups:
        for field in group.get("field_names") or []:
            fields.add(str(field))
    return sorted(fields)


def derive_state_features(node: dict[str, Any]) -> dict[str, Any]:
    groups = output_groups_from_node(node)
    group_names = sorted(str(group.get("name")) for group in groups if group.get("name"))
    field_names = flatten_field_names(groups)
    name_set = set(group_names)
    field_set = set(field_names)
    lower_tokens = {item.lower() for item in name_set | field_set}

    return {
        "has_templates": bool(
            ((name_set | field_set) & TEMPLATE_NAMES)
            or any(name.startswith("templates_") for name in name_set)
        ),
        "has_ctf": bool(field_set & CTF_FIELD_NAMES),
        "has_ctf_stats": bool(field_set & CTF_STATS_FIELD_NAMES),
        "has_initial_volume": bool(
            (name_set & VOLUME_OUTPUT_NAMES)
            or (field_set & VOLUME_FIELD_NAMES)
            or any(name.startswith(("volume_", "volumes_")) for name in name_set)
        ),
        "has_selected_particles": bool(name_set & SELECTED_PARTICLE_OUTPUT_NAMES),
        "has_particle_alignments_2d": bool(field_set & ALIGNMENTS_2D_FIELD_NAMES),
        "has_particle_alignments_3d": bool(field_set & ALIGNMENTS_3D_FIELD_NAMES),
        "has_filament_metadata": any(
            hint in token
            for token in lower_tokens
            for hint in FILAMENT_FIELD_HINTS
        ),
        "micrograph_count": count_by_names(groups, MICROGRAPH_OUTPUT_NAMES),
        "particle_count": count_by_names(groups, PARTICLE_OUTPUT_NAMES),
        "selected_particle_count": count_by_names(groups, SELECTED_PARTICLE_OUTPUT_NAMES),
        "rejected_particle_count": count_by_names(groups, REJECTED_PARTICLE_OUTPUT_NAMES),
        "volume_count": count_by_names(groups, VOLUME_OUTPUT_NAMES),
        "class_count": count_by_names(groups, CLASS_OUTPUT_NAMES),
        "mask_count": count_by_names(groups, MASK_OUTPUT_NAMES),
        "last_output_group_names": group_names,
        "last_output_field_names": field_names,
    }


def empty_state_features() -> dict[str, Any]:
    return {
        "has_templates": False,
        "has_ctf": False,
        "has_ctf_stats": False,
        "has_initial_volume": False,
        "has_selected_particles": False,
        "has_particle_alignments_2d": False,
        "has_particle_alignments_3d": False,
        "has_filament_metadata": False,
        "micrograph_count": None,
        "particle_count": None,
        "selected_particle_count": None,
        "rejected_particle_count": None,
        "volume_count": None,
        "class_count": None,
        "mask_count": None,
        "last_output_group_names": [],
        "last_output_field_names": [],
    }


def build_evidence_text(node: dict[str, Any]) -> list[str]:
    """Create short evidence lines instead of passing raw logs."""
    lines = [
        f"Job {node['cryosparc_job_uid']} ({node['job_type']}) status: {node['status']}."
    ]
    for name, output in node["outputs"].items():
        lines.append(f"Output {name}: {output['num_items']} items.")
    return lines


def build_warning_lines(node: dict[str, Any]) -> list[str]:
    """Represent basic warning/error flags as short text lines."""
    lines = []
    if node["has_warning"]:
        lines.append("CryoSPARC job has_warning is true.")
    if node["has_error"]:
        lines.append("CryoSPARC job has_error is true.")
    return lines


def empty_image_refs() -> dict[str, Any]:
    """Return the V2 placeholder for image references."""
    return {
        "ui_tile_images": [],
        "output_group_images": {},
        "event_images": [],
        "event_image_count": 0,
        "event_image_kind_counts": {},
    }
