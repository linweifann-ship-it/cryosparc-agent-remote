"""Run ProSHADE symmetry detection on a completed CryoSPARC volume map."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from cryosparc_client import cryosparc_client


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROSHADE_BINARY = str(PACKAGE_ROOT / "external" / "proshade" / "bin" / "proshade")
SUPPORTED_JOB_TYPES = {"homo_abinit", "homo_refine_new", "nonuniform_refine_new", "refine_3D_new"}


def resolve_proshade_binary() -> Path:
    """Return the configured ProSHADE executable without modifying PATH."""
    return Path(os.getenv("CRYOAGENT_PROSHADE_BINARY", DEFAULT_PROSHADE_BINARY)).expanduser()


def analyze_volume_symmetry(
    project_uid: str,
    job_uid: str,
    resolution_A: float = 8.0,
    find_symmetry_center: bool = False,
    timeout_seconds: int = 1800,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Download one volume output and return non-binding ProSHADE evidence."""
    if resolution_A <= 0:
        return failure("invalid_resolution", "resolution_A must be positive.")
    if timeout_seconds <= 0:
        return failure("invalid_timeout", "timeout_seconds must be positive.")

    binary = resolve_proshade_binary()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return failure(
            "proshade_unavailable",
            f"ProSHADE executable is unavailable: {binary}",
            binary=str(binary),
        )

    try:
        cs = cryosparc_client()
        project = cs.find_project(project_uid)
        job = project.find_job(job_uid)
        job_type = str(getattr(job, "type", ""))
        map_source = find_volume_source(job)
        cache_root = Path(output_dir or os.getenv(
            "CRYOAGENT_PROSHADE_CACHE_DIR",
            str(Path.cwd() / "logs" / "proshade"),
        )).expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        local_map = download_volume_map(job, map_source, cache_root, project_uid, job_uid)
    except Exception as exc:
        return failure("volume_download_failed", str(exc), project_uid=project_uid, job_uid=job_uid)

    result = run_proshade(
        local_map,
        binary=binary,
        resolution_A=resolution_A,
        find_symmetry_center=find_symmetry_center,
        timeout_seconds=timeout_seconds,
    )
    result.update({
        "project_uid": project_uid,
        "job_uid": job_uid,
        "job_type": job_type,
        "map_source": map_source,
        "local_map_path": str(local_map),
        "supported_refinement_job": job_type in SUPPORTED_JOB_TYPES,
        "evidence_policy": (
            "Advisory evidence only. Do not overwrite an existing symmetry or force a "
            "refinement; compare a proposed symmetry-constrained branch with C1."
        ),
    })
    return result


def find_volume_source(job: Any) -> str:
    """Prefer the primary volume result, then fall back to non-half MRC files."""
    for output_name in ("volume", "volume_class_0", "volume_map"):
        try:
            dataset = job.load_output(output_name)
            paths = list(dataset.get("blob/path", []))
            if paths:
                return str(paths[0])
        except Exception:
            continue
    files = [str(path) for path in (job.list_files() or [])]
    candidates = [
        path for path in files
        if path.lower().endswith((".mrc", ".map", ".ccp4"))
        and "half" not in Path(path).name.lower()
    ]
    preferred = [path for path in candidates if "volume" in Path(path).name.lower()]
    if preferred:
        return preferred[0]
    if candidates:
        return candidates[0]
    raise ValueError("No non-half volume MRC/MAP output was found for this job.")


def download_volume_map(
    job: Any,
    source_path: str,
    cache_root: Path,
    project_uid: str,
    job_uid: str,
) -> Path:
    """Download a CryoSPARC volume into a deterministic local cache path."""
    suffix = Path(source_path).suffix or ".mrc"
    destination = cache_root / f"{project_uid}_{job_uid}_volume{suffix}"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    _, volume = job.download_mrc(Path(source_path).name)
    try:
        from cryosparc.mrc import write
        write(str(destination), volume)
    except Exception as exc:
        raise RuntimeError(f"Could not write downloaded volume to {destination}: {exc}") from exc
    return destination


def run_proshade(
    map_path: Path,
    binary: Path | None = None,
    resolution_A: float = 8.0,
    find_symmetry_center: bool = False,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Execute ProSHADE CLI and parse its stable human-readable summary."""
    binary = binary or resolve_proshade_binary()
    command = [str(binary), "--symmetry", "--file", str(map_path), "--resolution", str(resolution_A)]
    if find_symmetry_center:
        command.append("--symCentre")
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return failure("proshade_timeout", f"ProSHADE exceeded {timeout_seconds} seconds.", command=command)
    except OSError as exc:
        return failure("proshade_launch_failed", str(exc), command=command)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if completed.returncode != 0:
        return failure(
            "proshade_failed",
            stderr[-4000:] or stdout[-4000:] or f"ProSHADE exited with {completed.returncode}.",
            command=command,
            returncode=completed.returncode,
        )
    parsed = parse_proshade_output(stdout)
    if not parsed["recommended_symmetry"]:
        return failure(
            "proshade_unparsed_output",
            "ProSHADE completed but no recommended symmetry could be parsed.",
            command=command,
            stdout_tail=stdout[-4000:],
        )
    return {
        "success": True,
        "tool": "proshade",
        "command": command,
        "resolution_A": resolution_A,
        "find_symmetry_center": find_symmetry_center,
        "recommended_symmetry": parsed["recommended_symmetry"],
        "symmetry_type": parsed["symmetry_type"],
        "symmetry_fold": parsed["symmetry_fold"],
        "primary_axis": parsed["primary_axis"],
        "axis_height": parsed["axis_height"],
        "average_fsc": parsed["average_fsc"],
        "confidence": confidence_label(parsed["average_fsc"]),
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-2000:],
    }


def parse_proshade_output(text: str) -> dict[str, Any]:
    """Parse the documented recommendation header and its first axis row."""
    match = re.search(r"claims the symmetry to be\s+([CDTOI])-(\d+)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"Detected symmetry:\s*([CDTOI])-(\d+)", text, re.IGNORECASE)
    symmetry_type = match.group(1).upper() if match else None
    fold = int(match.group(2)) if match else None
    row = None
    if match:
        remainder = text[match.end():]
        row_match = re.search(
            r"^\s*[CDTOI]\s+\+?(\d+)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s*$",
            remainder,
            re.MULTILINE | re.IGNORECASE,
        )
        if row_match:
            row = row_match.groups()
    return {
        "recommended_symmetry": f"{symmetry_type}{fold}" if symmetry_type and fold else None,
        "symmetry_type": symmetry_type,
        "symmetry_fold": fold,
        "primary_axis": [float(row[1]), float(row[2]), float(row[3])] if row else None,
        "axis_height": float(row[5]) if row else None,
        "average_fsc": float(row[6]) if row else None,
    }


def confidence_label(average_fsc: float | None) -> str:
    if average_fsc is None:
        return "unscored"
    if average_fsc >= 0.9:
        return "high"
    if average_fsc >= 0.75:
        return "moderate"
    return "low"


def failure(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "tool": "proshade", "error_code": code, "error": message, **extra}
