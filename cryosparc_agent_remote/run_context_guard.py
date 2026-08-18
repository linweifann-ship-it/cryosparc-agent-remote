# Run isolation checks for Schema V24 benchmark runs.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


FORBIDDEN_MODEL_INPUT_KEYS = {
    "candidate_actions",
    "failure_context",
    "retry_guidance",
}


@dataclass
class RunContextGuard:
    run_id: str
    project_uid: str
    workspace_uid: str
    start_node: str | None
    created_job_uids: set[str] = field(default_factory=set)

    @property
    def allowed_job_uids(self) -> set[str]:
        allowed = set(self.created_job_uids)
        if self.start_node:
            allowed.add(self.start_node)
        return allowed

    def add_created_jobs(self, job_uids: list[str]) -> None:
        self.created_job_uids.update(uid for uid in job_uids if uid)

    def assert_model_input(self, model_input: dict[str, Any]) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        for key in FORBIDDEN_MODEL_INPUT_KEYS:
            if contains_key(model_input, key):
                issues.append(issue("forbidden_model_input_key", f"Found {key}.", key))

        metadata = (
            model_input.get("dataset_context", {})
            .get("dataset_metadata", {})
        )
        if metadata.get("known_workflow_steps") is not None:
            issues.append(
                issue(
                    "known_workflow_steps_not_null",
                    "known_workflow_steps must be null for V24 benchmark runs.",
                    "dataset_context.dataset_metadata.known_workflow_steps",
                )
            )

        current_state = model_input.get("current_state") or {}
        history = current_state.get("recent_job_history") or []
        for index, item in enumerate(history):
            job_uid = item.get("job_uid") if isinstance(item, dict) else None
            if job_uid and self.allowed_job_uids and job_uid not in self.allowed_job_uids:
                issues.append(
                    issue(
                        "foreign_history_job",
                        f"recent_job_history contains foreign job {job_uid}.",
                        f"current_state.recent_job_history.{index}.job_uid",
                    )
                )

        last_info = current_state.get("last_node_info") or {}
        if last_info.get("project_uid") not in {None, self.project_uid}:
            issues.append(issue("project_mismatch", "Current node project mismatch.", "current_state.last_node_info.project_uid"))
        if last_info.get("workspace_uid") not in {None, self.workspace_uid}:
            issues.append(issue("workspace_mismatch", "Current node workspace mismatch.", "current_state.last_node_info.workspace_uid"))

        status = current_state.get("last_node_status")
        if status not in {"not_started", "completed", "failure"}:
            issues.append(issue("not_decidable_status", f"Status {status!r} is not decidable.", "current_state.last_node_status"))

        return {"success": not issues, "allowed_job_uids": sorted(self.allowed_job_uids), "issues": issues}

    def assert_messages(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        text = json.dumps(messages, ensure_ascii=False, default=str)
        issues = []
        for forbidden in FORBIDDEN_MODEL_INPUT_KEYS:
            if f'"{forbidden}"' in text:
                issues.append(issue("forbidden_prompt_key", f"Prompt contains {forbidden}.", forbidden))
        return {"success": not issues, "issues": issues}


def contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(nested, key) for nested in value.values())
    if isinstance(value, list):
        return any(contains_key(nested, key) for nested in value)
    return False


def issue(code: str, message: str, path: str | None) -> dict[str, Any]:
    return {
        "severity": "error",
        "code": code,
        "message": message,
        "path": path,
    }

