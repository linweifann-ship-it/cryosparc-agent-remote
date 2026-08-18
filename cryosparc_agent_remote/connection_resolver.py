# Deterministic internal CryoSPARC connection resolver for V24 benchmark runs.
from __future__ import annotations

from typing import Any

from job_specs import get_job_spec


INPUT_OUTPUT_ALIASES: dict[str, set[str]] = {
    "movies": {"movies", "movie_blob"},
    "exposures": {"exposures", "micrographs", "imported_micrographs", "micrograph_blob"},
    "micrographs": {"micrographs", "imported_micrographs", "exposures", "micrograph_blob"},
    "particles": {"particles", "particles_selected", "selected_particles", "location"},
    "templates": {"templates", "templates_selected", "class_averages", "template"},
    "volume": {"volume", "volumes", "map", "imported_volume_1", "imported_volume"},
    "mask": {"mask", "masks"},
}


def resolve_connections_for_decision(
    decision: dict[str, Any],
    workflow_state: dict[str, Any],
    allowed_job_uids: set[str],
) -> dict[str, Any]:
    """Resolve every selected action using only allowed upstream jobs."""
    results = []
    for index, action in enumerate(decision.get("selected_actions") or []):
        result = resolve_action_connections(
            job_type=action.get("job_type"),
            workflow_state=workflow_state,
            allowed_job_uids=allowed_job_uids,
            action_index=index,
        )
        results.append(result)
    success = all(item["success"] for item in results)
    status = "resolved" if success else first_error_status(results)
    return {
        "success": success,
        "status": status,
        "allowed_job_uids": sorted(allowed_job_uids),
        "actions": results,
    }


def resolve_action_connections(
    job_type: str,
    workflow_state: dict[str, Any],
    allowed_job_uids: set[str],
    action_index: int = 0,
) -> dict[str, Any]:
    spec = get_job_spec(job_type)
    required_inputs = spec.get("required_inputs") or []
    connections: dict[str, tuple[str, str] | list[tuple[str, str]]] = {}
    input_results = []
    for input_name in required_inputs:
        matches = find_matches(input_name, workflow_state, allowed_job_uids)
        if len(matches) == 1:
            match = matches[0]
            connections[input_name] = (match["source_job_uid"], match["source_output"])
            input_results.append(
                {
                    "input_name": input_name,
                    "status": "resolved",
                    "matches": matches,
                }
            )
        elif not matches:
            input_results.append(
                {
                    "input_name": input_name,
                    "status": "connection_resolution_failed",
                    "matches": [],
                    "available_upstream_outputs": available_outputs(workflow_state, allowed_job_uids),
                }
            )
        else:
            input_results.append(
                {
                    "input_name": input_name,
                    "status": "ambiguous_connection",
                    "matches": matches,
                }
            )
    success = all(item["status"] == "resolved" for item in input_results)
    return {
        "success": success,
        "action_index": action_index,
        "job_type": job_type,
        "status": "resolved" if success else first_input_error(input_results),
        "connections": connections if success else {},
        "input_results": input_results,
    }


def find_matches(
    input_name: str,
    workflow_state: dict[str, Any],
    allowed_job_uids: set[str],
) -> list[dict[str, Any]]:
    accepted = INPUT_OUTPUT_ALIASES.get(input_name, {input_name})
    matches = []
    for node in workflow_state.get("nodes", []):
        job_uid = node.get("cryosparc_job_uid") or node.get("workflow_node_id")
        if allowed_job_uids and job_uid not in allowed_job_uids:
            continue
        if node.get("status") != "completed":
            continue
        for output_name, output in (node.get("outputs") or {}).items():
            if not output or not output.get("available"):
                continue
            result_names = set(output.get("result_names") or [])
            output_tokens = {output_name, *result_names}
            if accepted & output_tokens:
                matches.append(
                    {
                        "source_job_uid": job_uid,
                        "source_output": output_name,
                        "result_names": sorted(result_names),
                    }
                )
    return sorted(matches, key=lambda item: (item["source_job_uid"], item["source_output"]))


def available_outputs(
    workflow_state: dict[str, Any],
    allowed_job_uids: set[str],
) -> list[str]:
    outputs = []
    for node in workflow_state.get("nodes", []):
        job_uid = node.get("cryosparc_job_uid") or node.get("workflow_node_id")
        if allowed_job_uids and job_uid not in allowed_job_uids:
            continue
        for output_name in (node.get("outputs") or {}):
            outputs.append(f"{job_uid}.{output_name}")
    return sorted(outputs)


def first_input_error(input_results: list[dict[str, Any]]) -> str:
    for item in input_results:
        if item["status"] != "resolved":
            return item["status"]
    return "unknown"


def first_error_status(results: list[dict[str, Any]]) -> str:
    for item in results:
        if not item["success"]:
            return item["status"]
    return "unknown"

