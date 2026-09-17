"""Small, scheduler-aware lane selection policy for CryoSPARC jobs."""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from cryosparc_client import cryosparc_client


def select_available_lane(
    *, preferred_lanes: list[str] | None = None, requires_gpu: bool = True,
    timeout: int = 10,
) -> dict[str, Any]:
    """Choose a lane with scheduler capacity without reserving resources.

    ``sinfo`` reports scheduler-level node states; the scheduler remains the
    authority for exact GPU allocation when the Job is queued.
    """
    if not requires_gpu:
        return {"success": True, "lane": None, "reason": "cpu_job"}
    cs = cryosparc_client()
    targets = list(cs.get_targets())
    allowed = preferred_lanes or _configured_lanes()
    candidates = [t for t in targets if not allowed or t.lane in allowed]
    observations = []
    for target in candidates:
        partition = _partition_from_target(target)
        if not partition:
            continue
        row = _query_partition(partition, timeout=timeout)
        row.update({"lane": target.lane, "hostname": target.hostname, "partition": partition})
        observations.append(row)
    ranked = sorted(observations, key=_capacity_rank, reverse=True)
    chosen = next(
        (
            item for item in ranked
            if item.get("gpu_idle", 0) > 0
            or item.get("idle_nodes", 0)
            or item.get("mix_nodes", 0)
        ),
        None,
    )
    if not chosen:
        return {"success": False, "lane": None, "reason": "no_scheduler_capacity", "observations": observations}
    return {"success": True, "lane": chosen["lane"], "hostname": chosen["hostname"], "partition": chosen["partition"], "observations": observations}


def _capacity_rank(observation: dict[str, Any]) -> tuple[int, int, int, int]:
    """Prefer real free GPUs, then scheduler node capacity as a fallback."""
    gpu_idle = observation.get("gpu_idle")
    if isinstance(gpu_idle, int):
        return (1, gpu_idle, observation.get("idle_nodes", 0), observation.get("mix_nodes", 0))
    return (0, 0, observation.get("idle_nodes", 0), observation.get("mix_nodes", 0))


def _configured_lanes() -> list[str]:
    value = os.getenv("CRYOAGENT_GPU_LANES") or os.getenv("CRYOAGENT_GPU_LANE")
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _partition_from_target(target: Any) -> str | None:
    template = getattr(getattr(target, "config", None), "script_tpl", "") or ""
    match = re.search(r"#SBATCH --partition=([^\s\n]+)", template)
    return match.group(1) if match else None


def _query_partition(partition: str, *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-p", partition, "-o", "%a|%T|%D"],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"success": False, "error": str(exc), "idle_nodes": 0, "mix_nodes": 0}
    idle = mix = 0
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 3 or fields[0].lower() not in {"up", "unknown"}:
            continue
        try:
            count = int(fields[2])
        except ValueError:
            count = 1
        state = fields[1].lower()
        if "idle" in state:
            idle += count
        elif "mix" in state or "alloc" in state:
            mix += count
    inventory = _query_gpu_inventory(partition, timeout=timeout)
    result_data = {
        "success": result.returncode == 0,
        "idle_nodes": idle,
        "mix_nodes": mix,
        "raw": result.stdout[-2000:],
        "error": result.stderr[-1000:],
    }
    result_data.update(inventory)
    return result_data


def _query_gpu_inventory(partition: str, *, timeout: int) -> dict[str, Any]:
    """Read per-node GPU allocation from Slurm when available.

    ``sinfo`` only exposes a coarse node state. ``scontrol`` exposes Gres and
    GresUsed/AllocTRES, which lets us distinguish an idle GPU from a mixed node
    whose GPUs are already occupied.
    """
    try:
        result = subprocess.run(
            ["scontrol", "show", "node", "-o"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"gpu_inventory_ok": False, "gpu_inventory_error": str(exc)}
    if result.returncode != 0:
        return {
            "gpu_inventory_ok": False,
            "gpu_inventory_error": result.stderr[-1000:],
        }

    total = allocated = 0
    matched_nodes = 0
    for line in result.stdout.splitlines():
        fields = dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line))
        partitions = fields.get("Partitions", "").split(",")
        if partition not in partitions:
            continue
        node_total = _gpu_count(fields.get("Gres", ""))
        if node_total is None:
            node_total = _gpu_count(fields.get("CfgTRES", ""))
        node_allocated = _gpu_count(fields.get("GresUsed", ""))
        if node_allocated is None:
            node_allocated = _gpu_count(fields.get("AllocTRES", ""))
        if node_total is None:
            continue
        matched_nodes += 1
        total += node_total
        allocated += node_allocated or 0
    if not matched_nodes:
        return {"gpu_inventory_ok": False, "gpu_inventory_error": "No GPU inventory for partition"}
    return {
        "gpu_inventory_ok": True,
        "gpu_total": total,
        "gpu_allocated": allocated,
        "gpu_idle": max(total - allocated, 0),
    }


def _gpu_count(value: str) -> int | None:
    """Extract a GPU count from Gres, GresUsed, or TRES text."""
    if not value or value.lower() in {"(null)", "none"}:
        return None
    match = re.search(r"(?:^|[,/])gpu(?::[^:,()=]+)?:(\d+)", value)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|[,/])gres/gpu=(\d+)", value)
    return int(match.group(1)) if match else None
