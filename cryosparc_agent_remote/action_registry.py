# Builds candidate CryoSPARC actions, validates model decisions, and plans safe execution.
from typing import Any
import math
import os

from schemas import (
    Action,
    ModelDecision,
    ResolvedAction,
    ValidationIssue,
    ValidationResult,
    parse_model_decision,
)
from cryosparc_client import cryosparc_client
from job_executor import execute_job_action, plan_job_action
from job_specs import get_parameter_template
from dynamic_candidates import build_registry_next_actions
from workflow_policy import annotate_candidates, infer_current_stage, POLICY_VERSION
from quality_policy import assess_node, build_retry_candidate
from workflow_state import content_hash, extract_workflow_state, find_node


REGISTRY_VERSION = "workflow_v1"

def get_candidate_actions(
    project_uid: str,
    workspace_uid: str,
    current_node_id: str | None = None,
    dataset_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read the current workflow and return selectable next actions."""
    workflow_state = extract_workflow_state(project_uid, workspace_uid)
    current_node = (
        find_node(workflow_state, current_node_id)
        if current_node_id
        else None
    )
    canonical_current_node_id = (
        current_node["workflow_node_id"]
        if current_node
        else current_node_id
    )
    candidate_actions, blocked_actions = generate_candidate_actions(
        workflow_state,
        canonical_current_node_id,
        project_uid=project_uid,
        include_registry=True,
    )
    if canonical_current_node_id is None:
        candidate_actions.extend(build_initial_import_candidates(dataset_info or {}))
    if current_node and current_node["status"] in {"failed", "killed"}:
        candidate_actions.append(build_retry_candidate(current_node))
    quality_assessment = (
        assess_node(current_node)
        if current_node
        else {"policy_version": "cryoem_quality_v1", "quality_status": "not_started"}
    )
    candidate_payload = {
        "state_snapshot_id": workflow_state["state_snapshot_id"],
        "registry_version": REGISTRY_VERSION,
        "current_node_id": canonical_current_node_id,
        "candidate_actions": candidate_actions,
    }
    return {
        "schema_version": "1.0",
        "registry_version": REGISTRY_VERSION,
        "state_snapshot_id": workflow_state["state_snapshot_id"],
        "candidate_set_id": content_hash("candidates", candidate_payload),
        "generated_at": workflow_state["generated_at"],
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "current_node_id": canonical_current_node_id,
        "requested_current_node_id": current_node_id,
        "workflow_status": workflow_state["workflow_status"],
        "candidate_actions": candidate_actions,
        "blocked_actions": blocked_actions,
        "decision_hint": "stop" if not candidate_actions else None,
        "workflow_policy": infer_current_stage(
            workflow_state["nodes"],
            current_node_id=canonical_current_node_id,
        ),
        "workflow_policy_version": POLICY_VERSION,
        "quality_assessment": quality_assessment,
    }


def build_initial_import_candidates(
    dataset_info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build import candidates from explicit dataset file facts at startup."""
    files = dataset_info.get("available_input_files") or {}
    candidates = []
    sources = [
        ("micrograph_blob_paths", "import_micrographs", "micrograph_blob_paths"),
        ("movie_blob_paths", "import_movies", "movie_blob_paths"),
        ("volume_blob_path", "import_volumes", "volume_blob_path"),
    ]
    for file_key, job_type, parameter_name in sources:
        value = files.get(file_key)
        if not value:
            continue
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else value[0]
        template = get_parameter_template(job_type)
        defaults = {parameter_name if parameter_name in template else "blob_paths": value}
        for name in ("psize_A", "accel_kv", "cs_mm", "total_dose_e_per_A2"):
            if dataset_info.get(name) is not None and name in template:
                defaults[name] = dataset_info[name]
        candidates.append({
            "action_id": f"initial_{job_type}",
            "action_type": "forward",
            "workflow_node_id": f"initial:{job_type}",
            "reference_job_uid": None,
            "reference_status": None,
            "job_type": job_type,
            "description": f"Import dataset input using {job_type}.",
            "execution_mode": "create_job",
            "available": True,
            "blocked_by": [],
            "required_inputs": {},
            "missing_required_inputs": [],
            "parameter_template": template,
            "default_parameters": defaults,
            "workflow_stage": "import",
            "workflow_policy_recommendation": "preferred_next_stage",
        })
    return candidates


def generate_candidate_actions(
    workflow_state: dict[str, Any],
    current_node_id: str | None,
    project_uid: str | None = None,
    include_registry: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn the selected node's child jobs into forward or branch candidates."""
    if not current_node_id:
        return [], []

    current_node = find_node(workflow_state, current_node_id)
    if current_node is None:
        return [], [
            {
                "action_id": None,
                "available": False,
                "blocked_by": [f"Unknown current_node_id: {current_node_id}"],
            }
        ]

    child_uids = current_node["child_job_uids"]
    child_nodes = [
        find_node(workflow_state, child_uid)
        for child_uid in child_uids
    ]
    child_nodes = [
        node
        for node in child_nodes
        if node is not None and node["status"] not in {"failed", "killed"}
    ]

    is_branch = len(child_nodes) > 1
    candidates: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for child in child_nodes:
        action_type = "branch" if is_branch else "forward"
        action = build_candidate_action(
            workflow_state,
            child,
            action_type,
        )
        if action["available"]:
            candidates.append(action)
        else:
            blocked.append(action)

    synthetic_actions, synthetic_blocked = build_synthetic_next_actions(
        workflow_state,
        current_node,
        project_uid=project_uid,
    )
    candidates.extend(synthetic_actions)
    blocked.extend(synthetic_blocked)

    if include_registry:
        registry_actions, registry_blocked = build_registry_next_actions(
            workflow_state,
            current_node,
            candidate_job_types={node["job_type"] for node in child_nodes},
        )
        blocked_job_types = {action.get("job_type") for action in registry_blocked}
        candidates = [
            action for action in candidates
            if action.get("job_type") not in blocked_job_types
        ]
        candidates = merge_candidates(candidates, registry_actions)
        blocked = merge_candidates(blocked, registry_blocked)

    candidates = [
        apply_auto_select_2d_policy(action, workflow_state)
        for action in candidates
    ]
    blocked = [
        apply_auto_select_2d_policy(action, workflow_state)
        for action in blocked
    ]
    policy = infer_current_stage(
        workflow_state["nodes"],
        current_node_id=current_node["workflow_node_id"],
    )
    candidates = annotate_candidates(candidates, policy["current_stage"])
    blocked = annotate_candidates(blocked, policy["current_stage"])
    candidates.sort(key=lambda action: action["action_id"])
    blocked.sort(key=lambda action: action["action_id"])
    return candidates, blocked


def apply_auto_select_2d_policy(
    action: dict[str, Any],
    workflow_state: dict[str, Any],
) -> dict[str, Any]:
    """Make select_2D non-interactive with a temporary class-count policy."""
    if action.get("job_type") != "select_2D":
        return action
    raw_fraction = os.getenv("CRYOAGENT_AUTO_SELECT_2D_FRACTION", "0.85")
    try:
        fraction = float(raw_fraction)
    except ValueError:
        return action
    if not 0 < fraction <= 1:
        return action
    class_count = class_average_count(action, workflow_state)
    if class_count is None:
        return action
    keep_count = max(1, min(class_count, int(math.ceil(class_count * fraction))))
    selected = ",".join(str(index) for index in range(keep_count))
    enriched = dict(action)
    template = dict(enriched.get("parameter_template") or {})
    template["selected_templates"] = {"type": "string", "default": selected}
    enriched["parameter_template"] = template
    defaults = dict(enriched.get("default_parameters") or {})
    defaults["selected_templates"] = selected
    enriched["default_parameters"] = defaults
    enriched["execution_mode"] = "create_job"
    enriched["approval_required"] = False
    enriched["approval_reasons"] = []
    enriched["auto_policy"] = {
        "name": "select_2d_class_fraction",
        "fraction": fraction,
        "class_count": class_count,
        "selected_class_count": keep_count,
        "selection_order": "class_index",
        "quality_ranked": False,
        "note": "Temporary deterministic selection until image-aware class ranking is available.",
    }
    metadata = dict(enriched.get("job_spec_metadata") or {})
    metadata.update({"interactive": False, "requires_approval": False})
    enriched["job_spec_metadata"] = metadata
    enriched["description"] = (
        f"Automatically select {keep_count}/{class_count} 2D classes "
        f"({fraction:.0%} target by class count)."
    )
    return enriched


def class_average_count(
    action: dict[str, Any],
    workflow_state: dict[str, Any],
) -> int | None:
    """Find the available class-average count for a select_2D candidate."""
    template_sources = (action.get("required_inputs") or {}).get("templates") or []
    for source in template_sources:
        node = find_node(workflow_state, source.get("source_job_uid", ""))
        if node is None:
            continue
        output = node.get("outputs", {}).get(source.get("source_output")) or {}
        count = output.get("num_items")
        if isinstance(count, int) and count > 0:
            return count
    return None


def build_synthetic_next_actions(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
    project_uid: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create safe next-step candidates when no child job exists yet."""
    if current_node["status"] != "completed":
        return [], []
    if current_node["job_type"] == "homo_refine_new":
        return build_parallel_homo_refine_actions(current_node)
    micrograph_source = micrograph_source_from_node(current_node)
    if micrograph_source:
        template_source = find_available_template_source(
            workflow_state,
            current_node,
        )
        actions = [
            build_synthetic_blob_picker_candidate(
                current_node,
                micrograph_source,
            )
        ]
        template_picker = build_synthetic_template_picker_candidate(
            current_node,
            micrograph_source,
            template_source,
        )
        blocked = []
        if template_picker["available"]:
            actions.append(template_picker)
        else:
            blocked.append(template_picker)
        return actions, blocked
    if current_node["job_type"] != "select_2D":
        return [], []
    particles = current_node["outputs"].get("particles_selected")
    if not particles or not particles["available"]:
        return [], [
            build_synthetic_homo_refine_candidate(
                current_node,
                required_inputs={},
                blocked_by=["Current select_2D job has no particles_selected output."],
            )
        ]

    volume_source = find_completed_volume_source(workflow_state, project_uid)
    required_inputs = {
        "particles": [
            {
                "source_workflow_node_id": current_node["workflow_node_id"],
                "source_logical_node_id": current_node["logical_node_id"],
                "source_job_uid": current_node["cryosparc_job_uid"],
                "source_output": "particles_selected",
                "result_names": particles["result_names"],
            }
        ],
    }
    blocked_by = []
    if volume_source:
        required_inputs["volume"] = [volume_source]
    else:
        blocked_by.append(
            "No completed imported volume output was found in this workspace or project."
        )

    action = build_synthetic_homo_refine_candidate(
        current_node,
        required_inputs=required_inputs,
        blocked_by=blocked_by,
    )
    return ([action], []) if action["available"] else ([], [action])


def merge_candidates(
    existing: list[dict[str, Any]],
    additions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate reference/synthetic and Registry candidates by job type."""
    seen = {action.get("job_type") for action in existing}
    return existing + [
        action for action in additions if action.get("job_type") not in seen
    ]


def micrograph_source_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return a usable micrograph-like output from a completed workflow node."""
    if node["status"] != "completed":
        return None
    for output_name in ("exposures", "micrographs", "imported_micrographs"):
        output = node["outputs"].get(output_name)
        if not output or not output["available"]:
            continue
        result_names = output.get("result_names") or []
        if "micrograph_blob" not in result_names:
            continue
        return {
            "source_workflow_node_id": node["workflow_node_id"],
            "source_logical_node_id": node["logical_node_id"],
            "source_job_uid": node["cryosparc_job_uid"],
            "source_output": output_name,
            "result_names": result_names,
        }
    return None


def find_available_template_source(
    workflow_state: dict[str, Any],
    current_node: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a usable template-like output from the current workflow."""
    current_source = template_source_from_node(current_node)
    if current_source:
        return current_source
    for node in workflow_state["nodes"]:
        if node["cryosparc_job_uid"] == current_node["cryosparc_job_uid"]:
            continue
        source = template_source_from_node(node)
        if source:
            return source
    return None


def template_source_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return a template-like output from a completed workflow node."""
    if node["status"] != "completed":
        return None
    for output_name, output in node["outputs"].items():
        if not output or not output["available"]:
            continue
        if output_name not in {"templates", "templates_selected"}:
            continue
        result_names = output.get("result_names") or []
        return {
            "source_workflow_node_id": node["workflow_node_id"],
            "source_logical_node_id": node["logical_node_id"],
            "source_job_uid": node["cryosparc_job_uid"],
            "source_output": output_name,
            "result_names": result_names,
        }
    return None


def build_synthetic_blob_picker_candidate(
    current_node: dict[str, Any],
    micrograph_source: dict[str, Any],
) -> dict[str, Any]:
    """Build a generic blob_picker_gpu candidate from available micrographs."""
    job_type = "blob_picker_gpu"
    return {
        "action_id": f"forward_{current_node['cryosparc_job_uid']}_blob_picker_gpu",
        "action_type": "forward",
        "workflow_node_id": f"{current_node['workflow_node_id']}:blob_picker_gpu",
        "reference_job_uid": None,
        "reference_status": None,
        "job_type": job_type,
        "description": (
            "Create a Blob Picker job from the current available micrographs."
        ),
        "execution_mode": "create_job",
        "available": True,
        "blocked_by": [],
        "required_inputs": {"micrographs": [micrograph_source]},
        "missing_required_inputs": [],
        "parameter_template": get_parameter_template(job_type),
        "default_parameters": {},
    }


def build_synthetic_template_picker_candidate(
    current_node: dict[str, Any],
    micrograph_source: dict[str, Any],
    template_source: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a Template Picker candidate when templates are available."""
    job_type = "template_picker_gpu"
    required_inputs = {
        "micrographs": [micrograph_source],
        "templates": [template_source] if template_source else [],
    }
    blocked_by = []
    missing_required_inputs = []
    if not template_source:
        blocked_by.append(
            "Required input templates has no available source in the current branch."
        )
        missing_required_inputs.append("templates")
    return {
        "action_id": (
            f"forward_{current_node['cryosparc_job_uid']}_template_picker_gpu"
            if template_source
            else f"blocked_{current_node['cryosparc_job_uid']}_template_picker_gpu"
        ),
        "action_type": "forward",
        "workflow_node_id": f"{current_node['workflow_node_id']}:template_picker_gpu",
        "reference_job_uid": None,
        "reference_status": None,
        "job_type": job_type,
        "description": (
            "Template Picker requires templates in addition to micrographs."
        ),
        "execution_mode": "create_job",
        "available": template_source is not None,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "missing_required_inputs": missing_required_inputs,
        "parameter_template": get_parameter_template(job_type),
        "default_parameters": {},
    }


def build_parallel_homo_refine_actions(
    current_node: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Allow another homogeneous refinement using the same upstream inputs."""
    required_inputs = {
        name: connections
        for name, connections in current_node["inputs"].items()
        if name in {"particles", "volume", "mask"} and connections
    }
    blocked_by = []
    if "particles" not in required_inputs:
        blocked_by.append("Current homo_refine_new job has no particles input.")
    if "volume" not in required_inputs:
        blocked_by.append("Current homo_refine_new job has no volume input.")

    action = build_synthetic_homo_refine_candidate(
        current_node,
        required_inputs=required_inputs,
        blocked_by=blocked_by,
        description=(
            "Create another homogeneous refinement job in parallel using the "
            "same upstream particles and volume inputs."
        ),
    )
    return ([action], []) if action["available"] else ([], [action])


def build_synthetic_homo_refine_candidate(
    current_node: dict[str, Any],
    required_inputs: dict[str, list[dict[str, Any]]],
    blocked_by: list[str],
    description: str | None = None,
) -> dict[str, Any]:
    """Build a synthetic homo_refine_new creation candidate."""
    job_type = "homo_refine_new"
    return {
        "action_id": f"forward_{current_node['cryosparc_job_uid']}_homo_refine_new",
        "action_type": "forward",
        "workflow_node_id": f"{current_node['workflow_node_id']}:homo_refine_new",
        "reference_job_uid": None,
        "reference_status": None,
        "job_type": job_type,
        "description": description or (
            "Create a new homogeneous refinement job from selected particles "
            "and an available imported volume."
        ),
        "execution_mode": "create_job",
        "available": not blocked_by,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "parameter_template": get_parameter_template(job_type),
        "default_parameters": {},
    }


def find_completed_volume_source(
    workflow_state: dict[str, Any],
    project_uid: str | None,
) -> dict[str, Any] | None:
    """Find a completed imported volume in the workspace or parent project."""
    for node in workflow_state["nodes"]:
        source = volume_source_from_node(node)
        if source:
            return source
    if not project_uid:
        return None
    try:
        cs = cryosparc_client()
        jobs = sorted(
            cs.find_jobs(project_uid=project_uid),
            key=lambda job: int(job.uid[1:]) if job.uid.startswith("J") and job.uid[1:].isdigit() else 10**12,
        )
    except Exception:
        return None
    for job in jobs:
        if job.status != "completed":
            continue
        for output_name, output in job.outputs.items():
            if output_name.startswith("imported_volume") and output.num_items > 0:
                return {
                    "source_workflow_node_id": job.uid,
                    "source_logical_node_id": None,
                    "source_job_uid": job.uid,
                    "source_output": output_name,
                    "result_names": sorted(
                        result.name
                        for result in output.results
                    ),
                }
    return None


def volume_source_from_node(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return a volume output from a normalized completed job node."""
    if node["status"] != "completed":
        return None
    for output_name, output in node["outputs"].items():
        if output_name.startswith("imported_volume") and output["available"]:
            return {
                "source_workflow_node_id": node["workflow_node_id"],
                "source_logical_node_id": node["logical_node_id"],
                "source_job_uid": node["cryosparc_job_uid"],
                "source_output": output_name,
                "result_names": output["result_names"],
            }
    return None


def build_candidate_action(
    workflow_state: dict[str, Any],
    target_node: dict[str, Any],
    action_type: str,
) -> dict[str, Any]:
    """Build one candidate action and record why it is unavailable, if blocked."""
    required_inputs: dict[str, list[dict[str, Any]]] = {}
    blocked_by: list[str] = []

    for input_name, connections in target_node["inputs"].items():
        required_inputs[input_name] = connections
        if not connections:
            blocked_by.append(
                f"Required input {input_name} has no available source."
            )
        for connection in connections:
            source_node = find_node(
                workflow_state,
                connection["source_job_uid"],
            )
            if source_node is None:
                blocked_by.append(
                    f"Missing source job {connection['source_job_uid']}"
                )
                continue
            if source_node["status"] != "completed":
                blocked_by.append(
                    f"Source job {source_node['cryosparc_job_uid']} is "
                    f"{source_node['status']}, not completed"
                )
            source_output = source_node["outputs"].get(
                connection["source_output"]
            )
            if not source_output or not source_output["available"]:
                blocked_by.append(
                    f"Output {connection['source_job_uid']}."
                    f"{connection['source_output']} is unavailable"
                )

    parameter_template = build_parameter_template(target_node)

    return {
        "action_id": f"{action_type}_{target_node['cryosparc_job_uid']}",
        "action_type": action_type,
        "workflow_node_id": target_node["workflow_node_id"],
        "reference_job_uid": target_node["cryosparc_job_uid"],
        "reference_status": target_node["status"],
        "job_type": target_node["job_type"],
        "description": (
            f"Reproduce the validated P2-W3 workflow node "
            f"{target_node['cryosparc_job_uid']}."
        ),
        "execution_mode": "dry_run_only",
        "available": not blocked_by,
        "blocked_by": blocked_by,
        "required_inputs": required_inputs,
        "parameter_template": parameter_template,
        "default_parameters": target_node["key_parameters"],
    }


def build_parameter_template(
    target_node: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Combine static parameter rules with defaults copied from the reference job."""
    template = get_parameter_template(target_node["job_type"])
    for name, value in target_node["key_parameters"].items():
        template.setdefault(name, {"type": infer_json_type(value)})
        template[name]["default"] = value
    return template


def validate_model_decision_payload(
    payload: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
    expected_state_snapshot_id: str | None = None,
    expected_candidate_set_id: str | None = None,
) -> dict[str, Any]:
    """Validate a model decision without creating or enqueueing CryoSPARC jobs."""
    decision, schema_issues = parse_model_decision(payload)
    if decision is None:
        result = ValidationResult(
            success=False,
            valid_schema=False,
            valid_actions=False,
            issues=schema_issues,
        )
        return result.model_dump()

    result = validate_decision_against_registry(
        decision,
        candidate_actions=candidate_actions,
    )

    context_issues = validate_context_ids(
        decision,
        expected_state_snapshot_id=expected_state_snapshot_id,
        expected_candidate_set_id=expected_candidate_set_id,
    )
    if context_issues:
        result.success = False
        result.valid_actions = False
        result.issues = context_issues + result.issues

    return result.model_dump()


def execute_model_decision_payload(
    payload: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
    expected_state_snapshot_id: str | None = None,
    expected_candidate_set_id: str | None = None,
    dry_run: bool = True,
    project_uid: str | None = None,
    workspace_uid: str | None = None,
    allow_approval_required_create: bool = False,
) -> dict[str, Any]:
    """
    Convert a validated decision into an execution plan.

    The current implementation is intentionally dry-run first. It verifies the
    decision and returns the jobs that would be executed, but it does not create
    or queue CryoSPARC jobs.
    """
    validation = validate_model_decision_payload(
        payload,
        candidate_actions=candidate_actions,
        expected_state_snapshot_id=expected_state_snapshot_id,
        expected_candidate_set_id=expected_candidate_set_id,
    )
    if not validation["success"]:
        return {
            "success": False,
            "dry_run": dry_run,
            "execution_mode": "validation_failed",
            "decision_type": validation.get("decision_type"),
            "validation": validation,
            "execution_plan": None,
            "execution_results": [],
            "issues": validation["issues"],
            "warnings": validation["warnings"],
        }

    execution_plan = build_execution_plan(payload, validation, candidate_actions)
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "execution_mode": "dry_run",
            "decision_type": validation["decision_type"],
            "validation": validation,
            "execution_plan": execution_plan,
            "execution_results": [],
            "issues": [],
            "warnings": validation["warnings"],
            "message": "Dry run only; no CryoSPARC jobs were created or queued.",
        }

    if validation["decision_type"] == "request_input":
        return {
            "success": True,
            "dry_run": dry_run,
            "execution_mode": "awaiting_input",
            "decision_type": "request_input",
            "validation": validation,
            "execution_plan": execution_plan,
            "execution_results": [],
            "issues": [],
            "warnings": validation["warnings"],
            "message": "Additional input is required; no CryoSPARC job was created or queued.",
        }

    if not project_uid or not workspace_uid:
        return {
            "success": False,
            "dry_run": False,
            "execution_mode": "missing_execution_context",
            "decision_type": validation["decision_type"],
            "validation": validation,
            "execution_plan": execution_plan,
            "execution_results": [],
            "issues": [
                {
                    "severity": "error",
                    "code": "missing_execution_context",
                    "message": "project_uid and workspace_uid are required for live execution.",
                    "path": None,
                }
            ],
            "warnings": validation["warnings"],
        }

    execution_results = [
        execute_job_action(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            planned_action=action,
            dry_run=False,
            allow_approval_required_create=allow_approval_required_create,
        )
        for action in execution_plan["actions"]
    ]
    return {
        "success": all(result["success"] for result in execution_results),
        "dry_run": False,
        "execution_mode": "live_execution",
        "model_visible": False,
        "next_model_input": None,
        "decision_type": validation["decision_type"],
        "validation": validation,
        "execution_plan": execution_plan,
        "execution_results": execution_results,
        "issues": [
            issue
            for result in execution_results
            for issue in result.get("issues", [])
        ],
        "warnings": validation["warnings"],
        "message": (
            "Live execution status is MCP-internal. Wait for the created job to "
            "complete, then call get_job_result_package before asking the model "
            "for the next decision."
        ),
    }


def build_execution_plan(
    payload: dict[str, Any],
    validation: dict[str, Any],
    candidate_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize validated decisions into a future execution-ready plan."""
    decision_type = validation["decision_type"]
    candidates_by_id = {
        action["action_id"]: action
        for action in candidate_actions or []
    }

    if decision_type in {"forward", "branch"}:
        actions = []
        for idx, action in enumerate(validation["resolved_actions"]):
            base_action = {
                "plan_step": idx + 1,
                "action_id": action["action_id"],
                "action_type": action["action_type"],
                "workflow_node_id": action["workflow_node_id"],
                "job_type": action["job_type"],
                "execution_mode": action["execution_mode"],
                "mcp_tool_name": action["mcp_tool_name"],
                "approval_required": False,
                "approval_reasons": [],
                "resolved_parameters": action["resolved_parameters"],
                "connections": action.get("connections"),
                "rollback_target": None,
                "status": "planned",
            }
            actions.append(
                plan_job_action(
                    base_action,
                    candidate_action=candidates_by_id.get(action["action_id"]),
                )
            )
        return make_execution_plan(payload, validation, actions)

    if decision_type == "rollback":
        rollback_target = payload.get("rollback_target")
        actions = [
            {
                "plan_step": 1,
                "action_id": None,
                "action_type": "rollback",
                "workflow_node_id": (
                    rollback_target or {}
                ).get("workflow_node_id"),
                "job_type": (rollback_target or {}).get("job_type"),
                "execution_mode": "dry_run_only",
                "mcp_tool_name": None,
                "approval_required": True,
                "approval_reasons": ["rollback_decision"],
                "resolved_parameters": {},
                "rollback_target": payload.get("rollback_target"),
                "status": "planned",
            }
        ]
        return make_execution_plan(payload, validation, actions)

    if decision_type == "stop":
        actions = [
            {
                "plan_step": 1,
                "action_id": None,
                "action_type": "stop",
                "workflow_node_id": None,
                "job_type": None,
                "execution_mode": "dry_run_only",
                "mcp_tool_name": None,
                "approval_required": False,
                "approval_reasons": [],
                "resolved_parameters": {},
                "rollback_target": None,
                "status": "planned",
            }
        ]
        return make_execution_plan(payload, validation, actions)

    if decision_type == "request_input":
        actions = [
            {
                "plan_step": 1,
                "action_id": None,
                "action_type": "request_input",
                "workflow_node_id": None,
                "job_type": None,
                "execution_mode": "dry_run_only",
                "mcp_tool_name": None,
                "approval_required": False,
                "approval_reasons": [],
                "resolved_parameters": {},
                "rollback_target": None,
                "status": "planned",
            }
        ]
        return make_execution_plan(payload, validation, actions)

    return make_execution_plan(payload, validation, [])


def make_execution_plan(
    payload: dict[str, Any],
    validation: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the stable top-level ExecutionPlan object."""
    approval_reasons = sorted(
        {
            reason
            for action in actions
            for reason in action["approval_reasons"]
        }
    )
    plan_payload = {
        "decision_type": validation["decision_type"],
        "actions": actions,
    }
    return {
        "plan_id": content_hash("plan", plan_payload),
        "plan_version": "1.0",
        "status": "planned",
        "dry_run_only": True,
        "decision_type": validation["decision_type"],
        "action_count": len(actions),
        "approval_required": any(
            action["approval_required"]
            for action in actions
        ),
        "approval_reasons": approval_reasons,
        "actions": actions,
        "execution_results": [],
    }


def validate_context_ids(
    decision: ModelDecision,
    expected_state_snapshot_id: str | None,
    expected_candidate_set_id: str | None,
) -> list[ValidationIssue]:
    """Reject model decisions generated from stale workflow or candidate state."""
    issues: list[ValidationIssue] = []
    if expected_state_snapshot_id:
        if not decision.state_snapshot_id:
            issues.append(ValidationIssue(
                code="missing_state_snapshot_id",
                message="Decision must return the supplied state_snapshot_id",
                path="state_snapshot_id",
            ))
        elif decision.state_snapshot_id != expected_state_snapshot_id:
            issues.append(ValidationIssue(
                code="stale_workflow_state",
                message="Decision state_snapshot_id does not match the current workflow state",
                path="state_snapshot_id",
            ))
    if expected_candidate_set_id:
        if not decision.candidate_set_id:
            issues.append(ValidationIssue(
                code="missing_candidate_set_id",
                message="Decision must return the supplied candidate_set_id",
                path="candidate_set_id",
            ))
        elif decision.candidate_set_id != expected_candidate_set_id:
            issues.append(ValidationIssue(
                code="stale_candidate_set",
                message="Decision candidate_set_id does not match the current candidate actions",
                path="candidate_set_id",
            ))
    return issues


def validate_decision_against_registry(
    decision: ModelDecision,
    candidate_actions: list[dict[str, Any]] | None = None,
) -> ValidationResult:
    """Validate actions, using candidates as optional helpers."""
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    resolved_actions: list[ResolvedAction] = []

    if decision.decision_type in {"rollback", "stop", "request_input"}:
        return ValidationResult(
            success=True,
            valid_schema=True,
            valid_actions=True,
            decision_type=decision.decision_type,
            dry_run=True,
        )

    candidates_by_id = {
        action["action_id"]: action
        for action in candidate_actions or []
    }

    for idx, action in enumerate(decision.selected_actions):
        action_issues, action_warnings, resolved_action = (
            validate_action_against_candidates(
                action,
                candidates_by_id,
                idx,
            )
        )
        issues.extend(action_issues)
        warnings.extend(action_warnings)
        if resolved_action:
            resolved_actions.append(resolved_action)

    return ValidationResult(
        success=not issues,
        valid_schema=True,
        valid_actions=not issues,
        decision_type=decision.decision_type,
        dry_run=True,
        issues=issues,
        warnings=warnings,
        resolved_actions=resolved_actions,
    )


def validate_action_against_candidates(
    action: Action,
    candidates_by_id: dict[str, dict[str, Any]],
    index: int,
) -> tuple[list[ValidationIssue], list[ValidationIssue], ResolvedAction | None]:
    """Validate one selected action and resolve its final parameters."""
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    path = f"selected_actions.{index}"
    candidate = candidates_by_id.get(action.action_id or "")

    if candidate is not None:
        for field_name in ("action_type", "workflow_node_id", "job_type"):
            expected = candidate[field_name]
            actual = getattr(action, field_name)
            if actual != expected:
                issues.append(
                    ValidationIssue(
                        code=f"{field_name}_mismatch",
                        message=(
                            f"{field_name} must be {expected!r} for "
                            f"action_id {action.action_id!r}"
                        ),
                        path=f"{path}.{field_name}",
                    )
                )
        parameter_template = candidate["parameter_template"]
        execution_mode = candidate["execution_mode"]
    else:
        parameter_template = get_parameter_template(action.job_type)
        execution_mode = "create_job"
        warnings.append(
            ValidationIssue(
                severity="warning",
                code="generic_action_without_candidate",
                message=(
                    f"job_type {action.job_type!r} did not match an internal "
                    "candidate; MCP will plan it as a generic CryoSPARC job."
                ),
                path=path,
            )
        )

    parameter_issues, resolved_parameters = validate_parameters(
        action.parameters,
        parameter_template,
        path=f"{path}.parameters",
    )
    for parameter_issue in parameter_issues:
        if parameter_issue.code == "unknown_parameter":
            parameter_issue.severity = "warning"
            warnings.append(parameter_issue)
        else:
            issues.append(parameter_issue)

    if issues:
        return issues, warnings, None

    return (
        issues,
        warnings,
        ResolvedAction(
            action_id=action.action_id or f"generic_{index}_{action.job_type}",
            action_type=action.action_type,
            workflow_node_id=action.workflow_node_id or f"generic:{action.job_type}",
            job_type=action.job_type,
            execution_mode=execution_mode,
            resolved_parameters=resolved_parameters,
            connections=action.connections,
            mcp_tool_name=None,
        ),
    )


def validate_parameters(
    parameters: dict[str, Any],
    template: dict[str, dict[str, Any]],
    path: str,
) -> tuple[list[ValidationIssue], dict[str, Any]]:
    """Merge defaults with model-supplied parameters and validate the result."""
    issues: list[ValidationIssue] = []
    resolved = {
        name: spec["default"]
        for name, spec in template.items()
        if "default" in spec
    }
    resolved.update(parameters)

    for name, spec in template.items():
        if spec.get("required") and resolved.get(name) is None:
            issues.append(
                ValidationIssue(
                    code="missing_required_parameter",
                    message=f"Missing required parameter {name!r}",
                    path=f"{path}.{name}",
                )
            )

    for name, value in parameters.items():
        if value is None:
            continue
        spec = template.get(name)
        if spec is None:
            issues.append(
                ValidationIssue(
                    code="unknown_parameter",
                    message=f"Parameter {name!r} is not allowed for this action",
                    path=f"{path}.{name}",
                )
            )
            continue

        issues.extend(
            validate_parameter_value(
                name,
                value,
                spec,
                f"{path}.{name}",
            )
        )

    return issues, resolved


def validate_parameter_value(
    name: str,
    value: Any,
    spec: dict[str, Any],
    path: str,
) -> list[ValidationIssue]:
    """Validate one parameter's type and simple numeric constraints."""
    issues: list[ValidationIssue] = []
    expected_type = spec.get("type")

    if expected_type and not matches_type(value, expected_type):
        return [
            ValidationIssue(
                code="parameter_type_mismatch",
                message=f"Parameter {name!r} must be {expected_type}",
                path=path,
            )
        ]

    if "enum" in spec and value not in spec["enum"]:
        issues.append(
            ValidationIssue(
                code="parameter_enum_mismatch",
                message=f"Parameter {name!r} must be one of {spec['enum']}",
                path=path,
            )
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = spec.get("minimum")
        maximum = spec.get("maximum")
        if minimum is not None and value < minimum:
            issues.append(
                ValidationIssue(
                    code="parameter_below_minimum",
                    message=f"Parameter {name!r} must be >= {minimum}",
                    path=path,
                )
            )
        if maximum is not None and value > maximum:
            issues.append(
                ValidationIssue(
                    code="parameter_above_maximum",
                    message=f"Parameter {name!r} must be <= {maximum}",
                    path=path,
                )
            )

    return issues


def matches_type(value: Any, expected_type: str) -> bool:
    """Map JSON schema-style type names to Python runtime checks."""
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return True


def infer_json_type(value: Any) -> str:
    """Infer a compact JSON type name from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return "string"
