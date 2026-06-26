# Extracts a normalized, hashable CryoSPARC workspace DAG for decision making.
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from cryosparc_client import cryosparc_client


WORKFLOW_STATE_SCHEMA_VERSION = "1.0"

ACTIVE_STATUSES = {"building", "queued", "launched", "started", "running", "waiting"}

# Only decision-relevant parameters are copied into the workflow snapshot.
KEY_PARAMETERS: dict[str, list[str]] = {
    "import_micrographs": ["blob_paths", "psize_A", "accel_kv"],
    "import_volumes": ["emdb_id", "volume_blob_path", "volume_psize"],
    "create_templates": ["n_templates", "angular_step", "max_tilt"],
    "patch_ctf_estimation_multi": [
        "compute_num_gpus",
        "df_search_min",
        "df_search_max",
    ],
    "template_picker_gpu": [
        "diameter",
        "min_distance",
        "use_ctf",
    ],
    "inspect_picks_v2": [
        "min_score",
        "max_score",
        "min_power",
        "max_power",
    ],
    "extract_micrographs_multi": [
        "box_size_pix",
        "compute_num_gpus",
    ],
    "class_2D_new": [
        "class2D_K",
        "compute_num_gpus",
    ],
    "select_2D": [
        "selected_templates",
        "resolution_better_than",
        "particle_count_above",
    ],
    "homo_refine_new": [
        "refine_symmetry",
        "refine_defocus_refine",
        "refine_ctf_global_refine",
    ],
}


def content_hash(prefix: str, payload: Any) -> str:
    """Create a deterministic short ID from decision-relevant content."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def build_logical_node_ids(jobs: list[Any]) -> dict[str, str]:
    """Assign readable logical node labels for diagnostics."""
    counts: dict[str, int] = {}
    mapping: dict[str, str] = {}

    for job in sorted(jobs, key=lambda item: job_uid_number(item.uid)):
        counts[job.type] = counts.get(job.type, 0) + 1
        mapping[job.uid] = f"node_{job.type}_{counts[job.type]:03d}"

    return mapping


def extract_workflow_state(project_uid: str, workspace_uid: str) -> dict[str, Any]:
    """Read CryoSPARC jobs and return a normalized workspace DAG snapshot."""
    cs = cryosparc_client()
    jobs = list(
        cs.find_jobs(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
        )
    )
    jobs.sort(key=lambda item: job_uid_number(item.uid))
    logical_ids = build_logical_node_ids(jobs)
    job_uids = set(logical_ids)

    nodes = [
        extract_job_node(job, logical_ids, job_uids)
        for job in jobs
    ]
    edges = build_edges(nodes)

    root_nodes = [
        node["workflow_node_id"]
        for node in nodes
        if not node["parent_job_uids"]
    ]
    terminal_nodes = [
        node["workflow_node_id"]
        for node in nodes
        if not node["child_job_uids"]
    ]
    running_nodes = [
        node["workflow_node_id"]
        for node in nodes
        if node["status"] in ACTIVE_STATUSES
    ]
    failed_nodes = [
        node["workflow_node_id"]
        for node in nodes
        if node["status"] in {"failed", "killed"}
    ]

    snapshot_payload = {
        "schema_version": WORKFLOW_STATE_SCHEMA_VERSION,
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "nodes": [
            snapshot_node(node)
            for node in nodes
        ],
        "edges": edges,
    }

    return {
        "schema_version": WORKFLOW_STATE_SCHEMA_VERSION,
        "state_snapshot_id": content_hash("state", snapshot_payload),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "workflow_status": derive_workflow_status(nodes),
        "nodes": nodes,
        "edges": edges,
        "root_nodes": root_nodes,
        "terminal_nodes": terminal_nodes,
        "running_nodes": running_nodes,
        "failed_nodes": failed_nodes,
        "node_mapping": {
            node["workflow_node_id"]: node["cryosparc_job_uid"]
            for node in nodes
        },
    }


def extract_job_node(
    job: Any,
    logical_ids: dict[str, str],
    workspace_job_uids: set[str],
) -> dict[str, Any]:
    """Convert one CryoSPARC job object into a JSON-safe workflow node."""
    inputs: dict[str, list[dict[str, Any]]] = {}
    for input_name, input_spec in job.inputs.items():
        inputs[input_name] = [
            {
                "source_workflow_node_id": connection.job_uid,
                "source_logical_node_id": logical_ids.get(connection.job_uid),
                "source_job_uid": connection.job_uid,
                "source_output": connection.output,
                "result_names": sorted(
                    {
                        result.result
                        for result in connection.results
                    }
                ),
            }
            for connection in input_spec.connections
        ]

    outputs = {
        output_name: {
            "type": output.type,
            "num_items": output.num_items,
            "available": output.num_items > 0,
            "result_names": sorted(
                {
                    result.name
                    for result in output.results
                }
            ),
            "summary_keys": sorted(output.summary.keys()),
            "latest_summary_stat_keys": sorted(
                output.latest_summary_stats.keys()
            ),
        }
        for output_name, output in job.outputs.items()
    }

    params = job.params.model_dump()
    key_parameters = {
        name: to_json_safe(params[name])
        for name in KEY_PARAMETERS.get(job.type, [])
        if name in params
    }

    parent_job_uids = sorted(
        uid
        for uid in job.model.parents
        if uid in workspace_job_uids
    )
    child_job_uids = sorted(
        uid
        for uid in job.model.children
        if uid in workspace_job_uids
    )

    return {
        "workflow_node_id": job.uid,
        "logical_node_id": logical_ids[job.uid],
        "cryosparc_job_uid": job.uid,
        "job_type": job.type,
        "title": job.title,
        "status": job.status,
        "updated_at": job.model.updated_at.isoformat(),
        "parent_job_uids": parent_job_uids,
        "parent_workflow_node_ids": [
            uid
            for uid in parent_job_uids
        ],
        "parent_logical_node_ids": [
            logical_ids[uid]
            for uid in parent_job_uids
        ],
        "child_job_uids": child_job_uids,
        "child_workflow_node_ids": [
            uid
            for uid in child_job_uids
        ],
        "child_logical_node_ids": [
            logical_ids[uid]
            for uid in child_job_uids
        ],
        "inputs": inputs,
        "outputs": outputs,
        "key_parameters": key_parameters,
        "has_error": job.model.has_error,
        "has_warning": job.model.has_warning,
    }


def build_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build explicit DAG edges from normalized job input connections."""
    edges: list[dict[str, Any]] = []

    for node in nodes:
        for target_input, connections in node["inputs"].items():
            for connection in connections:
                edges.append(
                    {
                        "source_workflow_node_id": connection["source_workflow_node_id"],
                        "source_job_uid": connection["source_job_uid"],
                        "source_output": connection["source_output"],
                        "target_workflow_node_id": node["workflow_node_id"],
                        "target_job_uid": node["cryosparc_job_uid"],
                        "target_input": target_input,
                    }
                )

    return sorted(
        edges,
        key=lambda edge: (
            edge["source_job_uid"],
            edge["source_output"],
            edge["target_job_uid"],
            edge["target_input"],
        ),
    )


def snapshot_node(node: dict[str, Any]) -> dict[str, Any]:
    """Keep only stable, decision-relevant fields for snapshot hashing."""
    return {
        "workflow_node_id": node["workflow_node_id"],
        "logical_node_id": node["logical_node_id"],
        "cryosparc_job_uid": node["cryosparc_job_uid"],
        "job_type": node["job_type"],
        "status": node["status"],
        "parent_job_uids": node["parent_job_uids"],
        "child_job_uids": node["child_job_uids"],
        "inputs": node["inputs"],
        "outputs": node["outputs"],
        "key_parameters": node["key_parameters"],
        "has_error": node["has_error"],
        "has_warning": node["has_warning"],
    }


def derive_workflow_status(nodes: list[dict[str, Any]]) -> str:
    """Summarize the workspace status from individual job states."""
    statuses = {node["status"] for node in nodes}
    if statuses & {"failed", "killed"}:
        return "attention_required"
    if statuses & ACTIVE_STATUSES:
        return "running"
    if nodes and statuses == {"completed"}:
        return "completed"
    if not nodes:
        return "empty"
    return "incomplete"


def find_node(
    workflow_state: dict[str, Any],
    node_id: str,
) -> dict[str, Any] | None:
    """Find a node by CryoSPARC job UID, workflow ID, or logical diagnostic ID."""
    for node in workflow_state["nodes"]:
        if node_id in {
            node["workflow_node_id"],
            node["cryosparc_job_uid"],
            node["logical_node_id"],
        }:
            return node
    return None


def job_uid_number(job_uid: str) -> int:
    """Sort CryoSPARC job UIDs by numeric order when possible."""
    try:
        return int(job_uid.removeprefix("J"))
    except ValueError:
        return 0


def to_json_safe(value: Any) -> Any:
    """Convert CryoSPARC SDK values into JSON-serializable structures."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): to_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            to_json_safe(item)
            for item in value
        ]
    return str(value)
