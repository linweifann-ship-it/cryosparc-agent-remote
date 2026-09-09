# Exposes CryoSPARC helper functions as MCP tools for model-driven workflows.
from mcp.server.fastmcp import FastMCP
from typing import Any
import json

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
from workflow_state import extract_workflow_state


mcp = FastMCP("cryoSPARC Tools")


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
        normalize_decision_argument(decision),
        candidate_actions=candidate_actions,
        expected_state_snapshot_id=(
            candidate_context["state_snapshot_id"] if candidate_context else None
        ),
        expected_candidate_set_id=(
            candidate_context["candidate_set_id"] if candidate_context else None
        ),
        allow_internal_schema=False,
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

    Defaults to dry-run mode. Live execution requires dry_run=false and the
    necessary execution context and approval policy.
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
        normalize_decision_argument(decision),
        candidate_actions=candidate_actions,
        expected_state_snapshot_id=(
            candidate_context["state_snapshot_id"] if candidate_context else None
        ),
        expected_candidate_set_id=(
            candidate_context["candidate_set_id"] if candidate_context else None
        ),
        dry_run=dry_run,
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        allow_internal_schema=False,
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
) -> dict:
    """
    Adapt a V2 model decision to the internal decision schema and validate it.
    """
    candidate_context = registry_get_candidate_actions(
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_node_id=current_node_id,
    )
    normalized_decision = normalize_decision_argument(decision)
    adapter_result = adapt_v2_decision_to_internal(
        normalized_decision,
        candidate_context["candidate_actions"],
    )
    if not adapter_result["success"]:
        return adapter_result
    validation = validate_model_decision_payload(
        adapter_result["internal_decision"],
        candidate_actions=candidate_context["candidate_actions"],
        allow_internal_schema=True,
    )
    return {
        "success": validation["success"],
        "adapter_result": adapter_result,
        "validation": validation,
    }


@mcp.tool()
def execute_v2_model_decision(
    decision: dict[str, Any],
    project_uid: str,
    workspace_uid: str,
    current_node_id: str | None = None,
    dry_run: bool = True,
    allow_approval_required_create: bool = False,
) -> dict:
    """
    Adapt and execute a V2 model decision through the existing internal executor.
    """
    return execute_v2_model_decision_payload(
        normalize_decision_argument(decision),
        project_uid=project_uid,
        workspace_uid=workspace_uid,
        current_node_id=current_node_id,
        dry_run=dry_run,
        allow_approval_required_create=allow_approval_required_create,
    )


def normalize_decision_argument(decision: Any) -> dict[str, Any]:
    """Accept JSON object arguments and provider-serialized JSON strings."""
    if isinstance(decision, dict):
        return decision
    if isinstance(decision, str):
        parsed = json.loads(decision)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError("decision must be a JSON object or JSON object string")


if __name__ == "__main__":
    mcp.run()
