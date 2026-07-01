# Builds V2 model-facing workflow decision payloads from CryoSPARC state.
from typing import Any

from job_result import TERMINAL_STATUSES, build_internal_status_package
from known_workflow_retriever import retrieve_known_workflow_steps
from workflow_state import extract_workflow_state, find_node, job_uid_number


MODEL_INPUT_SCHEMA_VERSION = "2.0"


def build_model_input_payload(
    project_uid: str,
    workspace_uid: str,
    current_job_uid: str | None = None,
    dataset_info: dict[str, Any] | None = None,
    known_workflow_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build the V2 payload for the model, or internal status if the job is active.

    The model-facing V2 payload is returned only when the current job is
    completed/failed/killed or when the workflow has not started.
    """
    workflow_state = extract_workflow_state(project_uid, workspace_uid)
    node = resolve_current_node(workflow_state, current_job_uid)
    dataset = build_dataset_info(
        dataset_info or {},
        known_workflow_dirs=known_workflow_dirs,
    )

    if node is None:
        return build_not_started_payload(project_uid, workspace_uid, dataset)

    if node["status"] not in TERMINAL_STATUSES:
        package = build_internal_status_package(workflow_state, node)
        package["model_input_schema_version"] = MODEL_INPUT_SCHEMA_VERSION
        return package

    return {
        "schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "task_type": "workflow_decision",
        "dataset_info": dataset,
        "current_state": {
            "last_node_id": node["workflow_node_id"],
            "last_action": node["job_type"],
            "last_node_status": node["status"],
            "last_node_info": build_last_node_info(
                node,
                project_uid=project_uid,
                workspace_uid=workspace_uid,
            ),
        },
    }


def build_dataset_info(
    dataset_info: dict[str, Any],
    known_workflow_dirs: list[str] | None,
) -> dict[str, Any]:
    """Normalize dataset metadata and attach known workflow steps if available."""
    dataset = {
        "empiar_id": dataset_info.get("empiar_id"),
        "emdb_id": dataset_info.get("emdb_id"),
        "resolution": dataset_info.get("resolution"),
        "input_type": dataset_info.get("input_type"),
        "macromolecules_type": dataset_info.get("macromolecules_type"),
        "num_of_maps": dataset_info.get("num_of_maps"),
        "abstract": dataset_info.get("abstract"),
        "known_workflow_steps": dataset_info.get("known_workflow_steps"),
    }
    if dataset["known_workflow_steps"] is None:
        dataset["known_workflow_steps"] = retrieve_known_workflow_steps(
            dataset,
            search_dirs=known_workflow_dirs,
        )
    return dataset


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
    dataset: dict[str, Any],
) -> dict[str, Any]:
    """Build a valid V2 payload for workflows that have not started."""
    return {
        "schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "task_type": "workflow_decision",
        "dataset_info": dataset,
        "current_state": {
            "last_node_id": None,
            "last_action": None,
            "last_node_status": "not_started",
            "last_node_info": {
                "job_type": None,
                "job_uid": None,
                "job_title": None,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "status": "not_started",
                "timestamps": {},
                "inputs": {"groups": []},
                "parameters": {},
                "outputs": {"groups": []},
                "metrics": {},
                "runtime": {},
                "evidence_text": ["Workflow has not started yet."],
                "warning_lines": [],
                "image_refs": empty_image_refs(),
            },
        },
    }


def build_last_node_info(
    node: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
) -> dict[str, Any]:
    """Convert a normalized CryoSPARC node into V2 last_node_info."""
    return {
        "job_type": node["job_type"],
        "job_uid": node["cryosparc_job_uid"],
        "job_title": node["title"],
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "status": node["status"],
        "timestamps": {
            "updated_at": node["updated_at"],
        },
        "inputs": build_input_groups(node),
        "parameters": node["key_parameters"],
        "outputs": build_output_groups(node),
        "metrics": build_metrics(node),
        "runtime": {},
        "evidence_text": build_evidence_text(node),
        "warning_lines": build_warning_lines(node),
        "image_refs": empty_image_refs(),
        "recent_batch_node_ids": node["parent_workflow_node_ids"],
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
    metrics = {
        "completed": node["status"] == "completed",
        "failed": node["status"] in {"failed", "killed"},
        "has_error": node["has_error"],
        "has_warning": node["has_warning"],
    }
    for name, output in node["outputs"].items():
        metrics[f"{name}_count"] = output["num_items"]
    return metrics


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
