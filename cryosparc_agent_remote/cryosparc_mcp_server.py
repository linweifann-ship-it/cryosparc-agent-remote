# Exposes CryoSPARC helper functions as MCP tools for model-driven workflows.
from mcp.server.fastmcp import FastMCP
from typing import Any

from action_registry import (
    execute_model_decision_payload,
    get_candidate_actions as registry_get_candidate_actions,
    validate_model_decision_payload,
)
from cryosparc_cli_tools import (
    cryosparc_create_import_movies_job,
    cryosparc_status,
    cryosparc_test_workers,
    cryosparc_version,
    cryosparc_worker_gpulist,
)
from job_specs import list_supported_job_types
from job_result import get_job_result_package as registry_get_job_result_package
from job_result import wait_for_job_result_package as registry_wait_for_job_result_package
from model_input_builder import build_model_input_payload as registry_build_model_input_payload
from v2_decision_adapter import (
    adapt_v2_decision_to_internal,
    execute_v2_model_decision_payload,
)
from workflow_policy import inspect_picks_gate_violation
from workflow_state import extract_workflow_state
from vision_inputs import (
    build_class_average_visual_context,
    build_pick_inspection_visual_context,
)
from kb_bridge import call_kb_tool, get_decision_context
from cryosift_adapter import evaluate_2d_classes_with_cryosift as evaluate_2d_classes_with_cryosift_impl


mcp = FastMCP("cryoSPARC Tools")


# Read-only knowledge-base tools. They are prefixed to distinguish historical
# evidence from live CryoSPARC execution tools.
@mcp.tool()
def kb_search_cryoem_kb(query: str, top_k: int = 5, kb_types: list[str] | None = None) -> dict:
    return call_kb_tool("search_cryoem_kb", {"query": query, "top_k": top_k, "kb_types": kb_types})


@mcp.tool()
def kb_get_dataset_summary(dataset_id: str) -> dict:
    return call_kb_tool("get_dataset_summary", {"dataset_id": dataset_id})


@mcp.tool()
def kb_get_workflow(dataset_id: str) -> dict:
    return call_kb_tool("get_workflow", {"dataset_id": dataset_id})


@mcp.tool()
def kb_get_maps(dataset_id: str | None = None, multi_map: bool | str | None = None, top_k: int = 20) -> dict:
    return call_kb_tool("get_maps", {"dataset_id": dataset_id, "multi_map": multi_map, "top_k": top_k})


@mcp.tool()
def kb_get_failures(failure_class: str | None = None, job_type: str | None = None, dataset_id: str | None = None, top_k: int = 20) -> dict:
    return call_kb_tool("get_failures", {"failure_class": failure_class, "job_type": job_type, "dataset_id": dataset_id, "top_k": top_k})


@mcp.tool()
def kb_get_images(dataset_id: str | None = None, job_id: str | None = None, image_type: str | None = None, top_k: int = 20) -> dict:
    return call_kb_tool("get_images", {"dataset_id": dataset_id, "job_id": job_id, "image_type": image_type, "top_k": top_k})


@mcp.tool()
def kb_find_similar_cases(input_type: str | None = None, molecule_type: str | None = None, multi_map: bool | str | None = None, top_k: int = 10) -> dict:
    return call_kb_tool("find_similar_cases", {"input_type": input_type, "molecule_type": molecule_type, "multi_map": multi_map, "top_k": top_k})


@mcp.tool()
def kb_get_next_steps(job_type: str, top_k: int = 10) -> dict:
    return call_kb_tool("get_next_steps", {"job_type": job_type, "top_k": top_k})


@mcp.tool()
def kb_get_job_doc(job_type: str | None = None, query: str | None = None, top_k: int = 10) -> dict:
    return call_kb_tool("get_job_doc", {"job_type": job_type, "query": query, "top_k": top_k})


@mcp.tool()
def kb_get_manual_annotations(annotation_type: str = "all", dataset_id: str | None = None, job_type: str | None = None, top_k: int = 20) -> dict:
    return call_kb_tool("get_manual_annotations", {"annotation_type": annotation_type, "dataset_id": dataset_id, "job_type": job_type, "top_k": top_k})


@mcp.tool()
def kb_get_decision_context(dataset_info: dict | None = None, current_state: dict | None = None, candidate_actions: list[dict] | None = None, top_k: int = 5) -> dict:
    return get_decision_context(dataset_info, current_state, candidate_actions, top_k)


# Basic read-only health and environment tools.
@mcp.tool()
def get_cryosparc_status() -> dict:
    """
    Check whether CryoSPARC master services are running.
    """
    return cryosparc_status()


@mcp.tool()
def get_cryosparc_version() -> dict:
    """
    Get the installed CryoSPARC version.
    """
    return cryosparc_version()


@mcp.tool()
def get_cryosparc_worker_gpulist() -> dict:
    """
    Get GPU information visible to the CryoSPARC worker environment.
    """
    return cryosparc_worker_gpulist()


# Explicit operational helpers; these may create CryoSPARC-side validation jobs.
@mcp.tool()
def test_cryosparc_workers(
    project_uid: str,
    test: str = "launch",
    target: str | None = None,
    test_pytorch: bool = False,
    timeout: int = 600,
) -> dict:
    """
    Run CryoSPARC worker validation jobs in a project.
    """
    return cryosparc_test_workers(
        project_uid=project_uid,
        test=test,
        target=target,
        test_pytorch=test_pytorch,
        timeout=timeout,
    )


# Job creation wrappers for directly supported CryoSPARC job types.
@mcp.tool()
def create_cryosparc_import_movies_job(
    project_uid: str,
    workspace_uid: str,
    blob_paths: str,
    title: str = "Import Movies",
    desc: str = "Created by cryosparc_agent MCP tool.",
    params: dict[str, Any] | None = None,
) -> dict:
    """
    Create an Import Movies job and set its blob_paths parameter.
    """
    return cryosparc_create_import_movies_job(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        blob_paths=blob_paths,
        title=title,
        desc=desc,
        params=params,
    )


# Workflow state and model-alignment tools.
@mcp.tool()
def get_supported_job_types() -> dict:
    """
    Return job types with explicit local metadata for generic execution plans.
    """
    return {
        "success": True,
        "job_types": list_supported_job_types(),
    }


@mcp.tool()
def get_candidate_actions(
    project_uid: str,
    workspace_uid: str,
    current_node_id: str | None = None,
    dataset_info: dict[str, Any] | None = None,
) -> dict:
    """
    Return the candidate actions currently recognized by the MCP adapter.
    """
    return registry_get_candidate_actions(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_node_id=current_node_id,
        dataset_info=dataset_info,
    )


@mcp.tool()
def get_workflow_state(
    project_uid: str,
    workspace_uid: str,
) -> dict:
    """
    Read a CryoSPARC workspace and return a normalized workflow state snapshot.
    """
    return extract_workflow_state(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
    )


@mcp.tool()
def get_class_average_visual_context(
    project_uid: str,
    job_uid: str,
    max_classes: int = 50,
) -> dict:
    """Return a class-id-labelled contact sheet for a completed 2D job."""
    return build_class_average_visual_context(
        project_uid=project_uid,
        job_uid=job_uid,
        max_classes=max_classes,
    )


@mcp.tool()
def get_pick_inspection_visual_context(
    project_uid: str,
    job_uid: str,
    max_micrographs: int = 6,
    max_picks_per_micrograph: int = 400,
    micrograph_root: str | None = None,
) -> dict:
    """Return micrograph thumbnails with Blob Picker locations overlaid."""
    return build_pick_inspection_visual_context(
        project_uid=project_uid,
        job_uid=job_uid,
        max_micrographs=max_micrographs,
        max_picks_per_micrograph=max_picks_per_micrograph,
        micrograph_root=micrograph_root,
    )


@mcp.tool()
def evaluate_2d_classes_with_cryosift(
    project_uid: str,
    job_uid: str,
    threshold: float = 3.0,
    output_dir: str | None = None,
    timeout_seconds: int = 1800,
) -> dict:
    """Score completed Class 2D averages with optional CryoSift CNN."""
    return evaluate_2d_classes_with_cryosift_impl(project_uid, job_uid, threshold, output_dir, timeout_seconds)


@mcp.tool()
def validate_model_decision(
    decision: dict[str, Any],
    project_uid: str | None = None,
    workspace_uid: str | None = None,
    current_node_id: str | None = None,
    candidate_actions: list[dict[str, Any]] | None = None,
) -> dict:
    """
    Validate an upstream model decision JSON against the schema and action registry.
    """
    candidate_context = None
    if project_uid and workspace_uid:
        candidate_context = registry_get_candidate_actions(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            current_node_id=current_node_id,
        )
        candidate_actions = candidate_context["candidate_actions"]

    return validate_model_decision_payload(
        decision,
        candidate_actions=candidate_actions,
    )


@mcp.tool()
def execute_model_decision(
    decision: dict[str, Any],
    project_uid: str | None = None,
    workspace_uid: str | None = None,
    current_node_id: str | None = None,
    candidate_actions: list[dict[str, Any]] | None = None,
    dry_run: bool = True,
) -> dict:
    """
    Validate a model decision and return the execution plan.

    Defaults to dry-run mode. The current implementation never creates or
    queues CryoSPARC jobs.
    """
    candidate_context = None
    if project_uid and workspace_uid:
        candidate_context = registry_get_candidate_actions(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            current_node_id=current_node_id,
        )
        candidate_actions = candidate_context["candidate_actions"]

    return execute_model_decision_payload(
        decision,
        candidate_actions=candidate_actions,
        dry_run=dry_run,
        project_uid=project_uid,
        workspace_uid=workspace_uid,
    )


@mcp.tool()
def get_job_result_package(
    project_uid: str,
    workspace_uid: str,
    job_uid: str,
    include_next_candidates: bool = True,
) -> dict:
    """
    Return model-facing results only after a CryoSPARC job reaches a terminal state.

    Queue/running states are returned as MCP-internal status packages with
    ready_for_model=false.
    """
    return registry_get_job_result_package(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        job_uid=job_uid,
        include_next_candidates=include_next_candidates,
    )


@mcp.tool()
def wait_for_job_result_package(
    project_uid: str,
    workspace_uid: str,
    job_uid: str,
    timeout_seconds: int = 0,
    poll_interval_seconds: int = 30,
    include_next_candidates: bool = True,
) -> dict:
    """
    Poll a CryoSPARC job and return a model-facing result package when finished.
    """
    return registry_wait_for_job_result_package(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        job_uid=job_uid,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        include_next_candidates=include_next_candidates,
    )


@mcp.tool()
def build_model_input_payload(
    project_uid: str,
    workspace_uid: str,
    current_job_uid: str | None = None,
    dataset_info: dict[str, Any] | None = None,
    known_workflow_dirs: list[str] | None = None,
) -> dict:
    """
    Build the V2 model-facing workflow decision payload.

    Active jobs return MCP-internal status with ready_for_model=false instead of
    a model-facing payload.
    """
    payload = registry_build_model_input_payload(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_job_uid=current_job_uid,
        dataset_info=dataset_info,
        known_workflow_dirs=known_workflow_dirs,
    )
    return attach_candidate_context(payload, project_uid, workspace_uid)


def attach_candidate_context(
    payload: dict, project_uid: str, workspace_uid: str
) -> dict:
    """Attach live Registry candidates to model-facing MCP context."""
    current = payload.get("current_state", {}).get("last_node_id")
    if payload.get("ready_for_model") is False:
        return payload
    context = registry_get_candidate_actions(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_node_id=current,
        dataset_info=payload.get("dataset_info"),
    )
    payload["candidate_actions"] = context["candidate_actions"]
    payload["blocked_actions"] = context["blocked_actions"]
    payload["candidate_context"] = {
        "registry_version": context["registry_version"],
        "current_node_id": context["current_node_id"],
        "decision_hint": context["decision_hint"],
    }
    return payload


@mcp.tool()
def get_workflow_decision_context(
    project_uid: str,
    workspace_uid: str,
    current_job_uid: str | None = None,
    dataset_info: dict[str, Any] | None = None,
    known_workflow_dirs: list[str] | None = None,
) -> dict:
    """
    Alias for build_model_input_payload using V2 workflow decision terminology.
    """
    payload = registry_build_model_input_payload(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_job_uid=current_job_uid,
        dataset_info=dataset_info,
        known_workflow_dirs=known_workflow_dirs,
    )
    return attach_candidate_context(payload, project_uid, workspace_uid)


@mcp.tool()
def validate_v2_model_decision(
    decision: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
    current_node_id: str | None = None,
    dataset_info: dict[str, Any] | None = None,
) -> dict:
    """
    Adapt a V2 model decision to the internal decision schema and validate it.
    """
    candidate_context = registry_get_candidate_actions(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_node_id=current_node_id,
        dataset_info=dataset_info,
    )
    adapter_result = adapt_v2_decision_to_internal(
        decision,
        candidate_context["candidate_actions"],
    )
    if not adapter_result["success"]:
        return adapter_result
    validation = validate_model_decision_payload(
        adapter_result["internal_decision"],
        candidate_actions=candidate_context["candidate_actions"],
    )
    validation = apply_workflow_goal_guard(
        decision,
        validation,
        candidate_context.get("workflow_guidance") or {},
    )
    validation = apply_mandatory_trial_guard(
        adapter_result.get("internal_decision") or decision,
        validation,
        candidate_context.get("candidate_actions") or [],
    )
    validation = apply_inspect_parameter_guard(decision, validation)
    validation = apply_inspect_picks_completion_guard(
        decision,
        validation,
        candidate_context.get("workflow_guidance") or {},
    )
    return {
        "success": validation["success"],
        "adapter_result": adapter_result,
        "validation": validation,
        "workflow_guidance": candidate_context.get("workflow_guidance") or {},
    }


def apply_inspect_picks_completion_guard(
    decision: dict[str, Any], validation: dict[str, Any], guidance: dict[str, Any]
) -> dict[str, Any]:
    """Reject generic or stale-cursor downstream actions before Pick QC."""
    violation = inspect_picks_gate_violation(decision, guidance)
    if not validation.get("success") or violation is None:
        return validation
    result = dict(validation)
    result["success"] = False
    result["valid_actions"] = False
    result["issues"] = list(result.get("issues") or []) + [violation]
    return result


def apply_inspect_parameter_guard(
    decision: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    """Require an explicit automated filtering choice for Inspect Picks."""
    if not validation.get("success") or decision.get("job_type") != "inspect_picks_v2":
        return validation
    raw_parameters = decision.get("parameters") or {}
    if isinstance(decision.get("selected_actions"), list):
        raw_parameters = {}
        for action in decision["selected_actions"]:
            if isinstance(action, dict) and action.get("job_type") == "inspect_picks_v2":
                raw_parameters.update(action.get("parameters") or {})
    filtering_keys = {
        "ncc_score_thresh", "lpower_thresh_min", "lpower_thresh_max",
        "curv_thresh", "sinu_thresh", "do_auto_cluster", "keep_threshold",
    }
    chosen = filtering_keys.intersection(raw_parameters)
    if chosen and not (chosen == {"do_auto_cluster"} and raw_parameters.get("do_auto_cluster") is False):
        return validation
    result = dict(validation)
    result["success"] = False
    result["valid_actions"] = False
    result["issues"] = list(result.get("issues") or []) + [{
        "severity": "error",
        "code": "inspect_parameters_required",
        "message": (
            "Inspect Picks must include an explicit automated filtering decision: "
            "a numeric threshold or do_auto_cluster=true."
        ),
        "path": "parameters",
    }]
    return result


def apply_mandatory_trial_guard(
    decision: dict[str, Any],
    validation: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require every generated box-size trial before allowing Class 2D."""
    trials = {
        action.get("action_id")
        for action in candidates
        if action.get("available") and (
            action.get("box_size_trial") or action.get("class_2d_trial")
        )
    }
    if not trials or not validation.get("success"):
        return validation
    requested = decision.get("selected_actions")
    if not isinstance(requested, list):
        requested = [decision] if decision.get("action_id") else []
    selected = {
        action.get("action_id")
        for action in requested
        if isinstance(action, dict)
    }
    if trials.issubset(selected) and decision.get("decision_type") in {"forward", "branch"}:
        return validation
    result = dict(validation)
    result["success"] = False
    result["valid_actions"] = False
    result["issues"] = list(result.get("issues") or []) + [{
        "severity": "error",
        "code": "mandatory_box_size_trials_missing",
        "message": (
            "All available extraction/Class 2D trial actions must be selected together "
            "in one forward decision before proceeding."
        ),
        "path": "selected_actions",
    }]
    return result


def apply_workflow_goal_guard(
    decision: dict[str, Any],
    validation: dict[str, Any],
    guidance: dict[str, Any],
) -> dict[str, Any]:
    """Prevent premature stop when an explicit resolution target is unmet."""
    if (
        validation.get("success")
        and decision.get("decision_type") == "stop"
        and guidance.get("must_continue_for_target")
    ):
        validation = dict(validation)
        validation["success"] = False
        validation["valid_actions"] = False
        issues = list(validation.get("issues") or [])
        issues.append({
            "severity": "error",
            "code": "resolution_target_not_met",
            "message": (
                "Cannot stop: the explicit target resolution has not been met and "
                "a refinement candidate is available."
            ),
            "path": "decision_type",
        })
        validation["issues"] = issues
    return validation


@mcp.tool()
def execute_v2_model_decision(
    decision: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
    current_node_id: str | None = None,
    dataset_info: dict[str, Any] | None = None,
    dry_run: bool = True,
    allow_approval_required_create: bool = False,
) -> dict:
    """
    Adapt and execute a V2 model decision through the existing internal executor.
    """
    return execute_v2_model_decision_payload(
        decision,
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_node_id=current_node_id,
        dataset_info=dataset_info,
        dry_run=dry_run,
        allow_approval_required_create=allow_approval_required_create,
    )


if __name__ == "__main__":
    mcp.run()
