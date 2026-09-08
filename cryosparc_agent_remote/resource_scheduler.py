"""Resource-aware scheduling policy for CryoSPARC job submission.

This module deliberately sits below the Model/MCP scientific decision layer.
It may change compute resources and queue placement, but it must not change
workflow shape, inputs, or scientific parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any


ACTIVE_PRE_RUNNING_STATUSES = {"queued", "launched"}
RUNNING_STATUSES = {"running", "started"}
FAILED_STATUSES = {"failed", "killed", "error"}
RESOURCE_PARAMETER_KEYS = {"compute_num_gpus", "compute_num_cores"}

DEFAULT_GPU_LANES = ("g8m192_4090_slurm", "h20_slurm")
CPU_FALLBACK_CAPABILITIES = {
    "extract_micrographs_multi": {
        "replacement_job_type": "extract_micrographs_cpu_parallel",
        "compute_num_cores": 32,
    },
}
RACE_MODE_JOB_TYPES = {
    "homo_refine_new",
    "heterogeneous_refinement",
    "nonuniform_refine_new",
}


@dataclass(frozen=True)
class ResourceLane:
    partition: str
    node: str | None = None
    gpu_type: str | None = None
    total_gpus: int = 0
    free_gpus: int = 0
    allocated_gpus: int = 0
    free_cpus: int = 0
    node_state: str = "unknown"
    queue_state: str = "unknown"


@dataclass(frozen=True)
class SchedulingPolicy:
    queue_probe: bool = True
    dynamic_downscale: bool = True
    gpu_lane_fallback: bool = True
    cpu_fallback: bool = False
    race_mode: bool = False
    preferred_lanes: tuple[str, ...] = DEFAULT_GPU_LANES
    compatible_lanes: tuple[str, ...] = DEFAULT_GPU_LANES
    minimum_gpus: int = 1
    maximum_replacements: int = 3
    queue_start_timeout_seconds: int = 600
    race_poll_interval_seconds: int = 20


@dataclass
class LogicalJobRecord:
    logical_job_id: str
    source_decision_hash: str
    scientific_fingerprint: str
    physical_job_ids: list[str] = field(default_factory=list)
    resource_snapshots: list[dict[str, Any]] = field(default_factory=list)
    submissions: list[dict[str, Any]] = field(default_factory=list)
    cancellations: list[dict[str, Any]] = field(default_factory=list)
    replacements: list[dict[str, Any]] = field(default_factory=list)
    winner: str | None = None


def policy_for_job(
    job_type: str,
    spec: dict[str, Any],
    params: dict[str, Any],
    requested_lane: str | None = None,
) -> SchedulingPolicy:
    """Build a resource policy from executor metadata, not model output."""
    preferred = configured_gpu_lanes(requested_lane or spec.get("default_lane"))
    template = spec.get("parameter_template") or {}
    gpu_rule = template.get("compute_num_gpus") or {}
    min_gpus = int(gpu_rule.get("minimum") or 1)
    return SchedulingPolicy(
        cpu_fallback=job_type in CPU_FALLBACK_CAPABILITIES,
        race_mode=job_type in RACE_MODE_JOB_TYPES,
        preferred_lanes=tuple(preferred),
        compatible_lanes=tuple(preferred),
        minimum_gpus=min_gpus,
    )


def configured_gpu_lanes(default_lane: str | None = None) -> list[str]:
    """Return scheduler-owned lane preferences from environment/defaults."""
    raw = os.getenv("CRYOAGENT_GPU_LANES")
    lanes = [item.strip() for item in raw.split(",")] if raw else []
    single = os.getenv("CRYOAGENT_GPU_LANE")
    if single:
        lanes.insert(0, single.strip())
    if default_lane:
        lanes.insert(0, default_lane)
    lanes.extend(DEFAULT_GPU_LANES)
    return dedupe([lane for lane in lanes if lane])


def probe_cluster_resources() -> dict[str, Any]:
    """
    Read the latest resource snapshot.

    Live cluster adapters can populate CRYOAGENT_RESOURCE_SNAPSHOT_JSON before
    execution. Without a snapshot, scheduler falls back to a conservative plan
    and records that probing was unavailable.
    """
    raw = os.getenv("CRYOAGENT_RESOURCE_SNAPSHOT_JSON")
    if raw:
        try:
            snapshot = json.loads(raw)
            snapshot.setdefault("probe_source", "CRYOAGENT_RESOURCE_SNAPSHOT_JSON")
            snapshot.setdefault("captured_at", time())
            return snapshot
        except json.JSONDecodeError as exc:
            return {
                "probe_source": "CRYOAGENT_RESOURCE_SNAPSHOT_JSON",
                "captured_at": time(),
                "probe_error": str(exc),
                "gpu_lanes": [],
                "cpu": {},
                "queue": {},
            }
    return {
        "probe_source": "unavailable",
        "captured_at": time(),
        "gpu_lanes": [],
        "cpu": {},
        "queue": {},
    }


def build_scheduling_plan(
    job_type: str,
    spec: dict[str, Any],
    params: dict[str, Any],
    requested_lane: str | None = None,
    resource_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select queue resources without mutating the scientific decision."""
    policy = policy_for_job(job_type, spec, params, requested_lane)
    snapshot = resource_snapshot if resource_snapshot is not None else probe_cluster_resources()
    requested_gpus = requested_gpu_count(params, policy.minimum_gpus)
    selected = select_gpu_lane(snapshot, policy, requested_gpus)
    selected_lane = selected.get("partition") if selected else requested_lane or first_or_none(policy.preferred_lanes)
    queue = {
        "lane": selected_lane,
        "hostname": None,
        "gpus": [],
        "cluster_vars": {},
        "will_queue": not spec.get("interactive", False),
    }
    reason = "selected_idle_gpu_lane" if selected else "no_probe_or_no_idle_lane_using_preferred"
    resource_config = {
        "mode": "gpu" if spec.get("requires_gpu") else "cpu_or_interactive",
        "lane": queue["lane"],
        "hostname": selected.get("node") if selected else queue["hostname"],
        "compute_num_gpus": requested_gpus if spec.get("requires_gpu") else None,
    }
    if not spec.get("requires_gpu"):
        reason = "job_does_not_require_gpu"
        resource_config["mode"] = "cpu"

    return {
        "policy": policy_to_dict(policy),
        "snapshot": snapshot,
        "selected_lane": selected_lane,
        "selected_resource": selected,
        "resource_config": resource_config,
        "queue": queue,
        "parameter_overrides": {},
        "reason": reason,
    }


def select_gpu_lane(
    snapshot: dict[str, Any],
    policy: SchedulingPolicy,
    requested_gpus: int,
) -> dict[str, Any] | None:
    lanes = parse_gpu_lanes(snapshot)
    if not lanes:
        return None
    by_partition = {lane.partition: lane for lane in lanes}
    search_order = list(policy.preferred_lanes)
    if policy.gpu_lane_fallback:
        search_order.extend(policy.compatible_lanes)
    for partition in dedupe(search_order):
        lane = by_partition.get(partition)
        if lane and lane.free_gpus >= requested_gpus and lane.node_state.lower() not in {"down", "drain"}:
            return asdict(lane)
    return None


def choose_replacement(
    record: LogicalJobRecord,
    job_type: str,
    spec: dict[str, Any],
    params: dict[str, Any],
    current_resource_config: dict[str, Any],
    status: str,
    waited_seconds: int,
    resource_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Choose one legal replacement after queue-start timeout."""
    policy = policy_for_job(job_type, spec, params, current_resource_config.get("lane"))
    if len(record.replacements) >= policy.maximum_replacements:
        return {"action": "none", "reason": "maximum_replacements_reached"}
    if status not in ACTIVE_PRE_RUNNING_STATUSES:
        return {"action": "none", "reason": "job_not_waiting_for_start"}
    if waited_seconds < policy.queue_start_timeout_seconds:
        return {"action": "none", "reason": "scheduler_timeout_not_reached"}

    current_gpus = int(current_resource_config.get("compute_num_gpus") or requested_gpu_count(params, policy.minimum_gpus))
    if policy.dynamic_downscale and current_gpus > policy.minimum_gpus:
        next_gpus = max(policy.minimum_gpus, current_gpus // 2)
        return {
            "action": "replace",
            "strategy": "dynamic_downscale",
            "resource_config": {**current_resource_config, "compute_num_gpus": next_gpus},
            "parameter_overrides": {"compute_num_gpus": next_gpus},
            "cancel_original": True,
        }

    alternate = select_gpu_lane(resource_snapshot, policy, max(policy.minimum_gpus, current_gpus))
    if alternate and alternate.get("partition") != current_resource_config.get("lane"):
        return {
            "action": "replace",
            "strategy": "gpu_lane_fallback",
            "resource_config": {**current_resource_config, "lane": alternate["partition"], "hostname": alternate.get("node")},
            "parameter_overrides": {},
            "cancel_original": True,
        }

    if policy.cpu_fallback and cpu_available(resource_snapshot):
        fallback = CPU_FALLBACK_CAPABILITIES[job_type]
        return {
            "action": "replace",
            "strategy": "cpu_fallback",
            "replacement_job_type": fallback["replacement_job_type"],
            "resource_config": {
                "mode": "cpu",
                "compute_num_cores": fallback["compute_num_cores"],
                "lane": None,
                "hostname": None,
            },
            "parameter_overrides": {"compute_num_cores": fallback["compute_num_cores"]},
            "cancel_original": True,
        }

    return {"action": "none", "reason": "no_legal_replacement_available"}


def reconcile_race_jobs(
    left_job_id: str,
    right_job_id: str,
    statuses: dict[str, str],
) -> dict[str, Any]:
    """Apply first-running-wins without killing already-running work."""
    left = normalize_status(statuses.get(left_job_id))
    right = normalize_status(statuses.get(right_job_id))
    if left in RUNNING_STATUSES and right in ACTIVE_PRE_RUNNING_STATUSES:
        return {"action": "cancel", "cancel_job_id": right_job_id, "winner": left_job_id}
    if right in RUNNING_STATUSES and left in ACTIVE_PRE_RUNNING_STATUSES:
        return {"action": "cancel", "cancel_job_id": left_job_id, "winner": right_job_id}
    if left in RUNNING_STATUSES and right in RUNNING_STATUSES:
        return {"action": "none", "reason": "both_jobs_running_do_not_cancel"}
    if left in FAILED_STATUSES and right not in FAILED_STATUSES:
        return {"action": "keep", "keep_job_id": right_job_id, "reason": "left_failed_before_winning"}
    if right in FAILED_STATUSES and left not in FAILED_STATUSES:
        return {"action": "keep", "keep_job_id": left_job_id, "reason": "right_failed_before_winning"}
    return {"action": "none", "reason": "no_race_winner_yet"}


def make_logical_job_record(
    planned_action: dict[str, Any],
    source_decision: dict[str, Any] | None = None,
) -> LogicalJobRecord:
    fingerprint = scientific_fingerprint(planned_action)
    source_hash = stable_hash(source_decision or planned_action)
    return LogicalJobRecord(
        logical_job_id=f"logical-{source_hash[:12]}",
        source_decision_hash=source_hash,
        scientific_fingerprint=fingerprint,
    )


def register_submission(
    record: LogicalJobRecord,
    job_uid: str,
    resource_config: dict[str, Any],
    snapshot: dict[str, Any],
    reason: str,
) -> None:
    config_id = stable_hash(resource_config)
    if any(item.get("resource_config_id") == config_id for item in record.submissions):
        return
    record.physical_job_ids.append(job_uid)
    record.resource_snapshots.append(deepcopy(snapshot))
    record.submissions.append(
        {
            "job_uid": job_uid,
            "resource_config_id": config_id,
            "resource_config": deepcopy(resource_config),
            "submission_time": time(),
            "reason": reason,
        }
    )


def save_logical_job_record(record: LogicalJobRecord, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(record), sort_keys=True, indent=2), encoding="utf-8")


def load_logical_job_record(path: str | Path) -> LogicalJobRecord:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LogicalJobRecord(**data)


def apply_resource_overrides(
    params: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    updated = deepcopy(params)
    updated.update(overrides)
    return updated


def scientific_fingerprint(planned_action: dict[str, Any]) -> str:
    payload = {
        "job_type": planned_action.get("job_type"),
        "connections": planned_action.get("connections"),
        "parameters": {
            key: value
            for key, value in (planned_action.get("resolved_parameters") or {}).items()
            if key not in RESOURCE_PARAMETER_KEYS
        },
    }
    return stable_hash(payload)


def parse_gpu_lanes(snapshot: dict[str, Any]) -> list[ResourceLane]:
    raw_lanes = snapshot.get("gpu_lanes") or snapshot.get("available_gpu") or []
    if isinstance(raw_lanes, dict):
        raw_lanes = raw_lanes.get("lanes") or [raw_lanes]
    lanes = []
    for item in raw_lanes:
        if not isinstance(item, dict):
            continue
        partition = item.get("partition") or item.get("lane")
        if not partition:
            continue
        total = int(item.get("total_gpus") or item.get("total") or 0)
        free = int(item.get("free_gpus") or item.get("free") or 0)
        allocated = int(item.get("allocated_gpus") or max(total - free, 0))
        lanes.append(
            ResourceLane(
                partition=str(partition),
                node=item.get("node") or item.get("hostname"),
                gpu_type=item.get("gpu_type"),
                total_gpus=total,
                free_gpus=free,
                allocated_gpus=allocated,
                free_cpus=int(item.get("free_cpus") or 0),
                node_state=str(item.get("node_state") or "unknown"),
                queue_state=str(item.get("queue_state") or "unknown"),
            )
        )
    return lanes


def cpu_available(snapshot: dict[str, Any]) -> bool:
    cpu = snapshot.get("available_cpu") or snapshot.get("cpu") or {}
    if not isinstance(cpu, dict):
        return False
    return int(cpu.get("free_cpus") or cpu.get("free") or 0) > 0


def requested_gpu_count(params: dict[str, Any], minimum: int) -> int:
    value = params.get("compute_num_gpus")
    if isinstance(value, int) and not isinstance(value, bool):
        return max(value, minimum)
    return minimum


def normalize_status(status: str | None) -> str:
    return str(status or "").lower()


def policy_to_dict(policy: SchedulingPolicy) -> dict[str, Any]:
    data = asdict(policy)
    data["preferred_lanes"] = list(policy.preferred_lanes)
    data["compatible_lanes"] = list(policy.compatible_lanes)
    return data


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def first_or_none(values: tuple[str, ...] | list[str]) -> str | None:
    return values[0] if values else None
