# Retrieves known reference workflows for V2 model input payloads.
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_WORKFLOW_DIR_ENV = "CRYOAGENT_KNOWN_WORKFLOW_DIRS"


def retrieve_known_workflow_steps(
    dataset_info: dict[str, Any],
    search_dirs: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    """Find and normalize known workflow steps for a dataset, or return None."""
    identifiers = dataset_identifiers(dataset_info)
    if not identifiers:
        return None

    for workflow_file in iter_workflow_files(search_dirs):
        if not file_may_match(workflow_file, identifiers):
            continue
        data = read_json(workflow_file)
        if data is None or not content_matches(data, identifiers):
            continue
        steps = extract_steps(data)
        if steps:
            return normalize_steps(steps)
    return None


def dataset_identifiers(dataset_info: dict[str, Any]) -> set[str]:
    """Return normalized dataset IDs that can be used for workflow lookup."""
    identifiers = set()
    for key in ("empiar_id", "emdb_id"):
        value = dataset_info.get(key)
        if isinstance(value, str) and value.strip():
            identifiers.add(normalize_identifier(value))
    return identifiers


def iter_workflow_files(search_dirs: list[str] | None) -> list[Path]:
    """List JSON files from explicit directories or CRYOAGENT_KNOWN_WORKFLOW_DIRS."""
    dirs = search_dirs or env_search_dirs()
    files: list[Path] = []
    for item in dirs:
        root = Path(item).expanduser()
        if root.is_file() and root.suffix.lower() == ".json":
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.json")))
    return files


def env_search_dirs() -> list[str]:
    """Read search directories from a colon-separated environment variable."""
    raw = os.getenv(DEFAULT_WORKFLOW_DIR_ENV, "")
    return [
        item
        for item in raw.split(os.pathsep)
        if item
    ]


def file_may_match(path: Path, identifiers: set[str]) -> bool:
    """Use the file path as a cheap first-pass match."""
    normalized_path = normalize_identifier(str(path))
    return any(identifier in normalized_path for identifier in identifiers)


def content_matches(data: Any, identifiers: set[str]) -> bool:
    """Check common metadata fields inside a workflow JSON."""
    metadata = data if isinstance(data, dict) else {}
    candidates = [
        metadata.get("empiar_id"),
        metadata.get("emdb_id"),
        metadata.get("dataset_id"),
        metadata.get("dataset"),
    ]
    dataset_info = metadata.get("dataset_info")
    if isinstance(dataset_info, dict):
        candidates.extend([
            dataset_info.get("empiar_id"),
            dataset_info.get("emdb_id"),
            dataset_info.get("dataset_id"),
        ])

    normalized = {
        normalize_identifier(value)
        for value in candidates
        if isinstance(value, str) and value.strip()
    }
    if not normalized:
        return True
    return bool(identifiers & normalized)


def read_json(path: Path) -> Any | None:
    """Read a JSON file, returning None on parse errors."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def extract_steps(data: Any) -> list[Any]:
    """Pull workflow steps from common JSON layouts."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in (
        "known_workflow_steps",
        "workflow_steps",
        "steps",
        "jobs",
        "nodes",
    ):
        value = data.get(key)
        if isinstance(value, list):
            return value
    workflow = data.get("workflow")
    if isinstance(workflow, dict):
        return extract_steps(workflow)
    return []


def normalize_steps(steps: list[Any]) -> list[dict[str, Any]]:
    """Convert reference workflow records into the V2 known_workflow_steps shape."""
    normalized = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        normalized.append(
            {
                "step_index": int(step.get("step_index", idx)),
                "node_id": str(
                    step.get("node_id")
                    or step.get("workflow_node_id")
                    or step.get("job_uid")
                    or step.get("uid")
                    or f"step_{idx}"
                ),
                "action": step.get("action") or step.get("job_type") or step.get("type"),
                "title": step.get("title"),
                "description": step.get("description") or step.get("desc"),
                "upstream_node_ids": list(
                    step.get("upstream_node_ids")
                    or step.get("parent_node_ids")
                    or step.get("parent_job_uids")
                    or step.get("parents")
                    or []
                ),
                "parameter_template": (
                    step.get("parameter_template")
                    or step.get("parameters")
                    or step.get("params")
                    or {}
                ),
            }
        )
    return sorted(normalized, key=lambda item: item["step_index"])


def normalize_identifier(value: str) -> str:
    """Normalize dataset IDs for loose file and metadata matching."""
    return (
        value.strip()
        .lower()
        .replace("_", "-")
        .replace(" ", "")
    )
