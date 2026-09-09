"""Optional CryoSift integration for deterministic 2D class scoring."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

from cryosparc_client import cryosparc_client


DEFAULT_EVALUATOR = "/home/lisongyang/cryoagent/external/cryosift/cryosift_runner.py"
DEFAULT_WEIGHTS = "/home/lisongyang/cryoagent/external/cryosift/class_labeling/final_model/final_model_cont.pth"


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser().resolve()


def _job_directory(project_uid: str, job_uid: str) -> Path:
    cs = cryosparc_client()
    job = cs.find_project(project_uid).find_job(job_uid)
    for attr in ("dir", "job_dir", "directory", "path"):
        value = getattr(job, attr, None)
        if value:
            path = Path(str(value)).expanduser().resolve()
            if path.is_dir():
                return path
    raise FileNotFoundError(f"Could not resolve server-side directory for CryoSPARC job {job_uid}.")


def _parse_scores(output_dir: Path, threshold: float) -> tuple[list[int], dict[int, float], str | None]:
    candidates = [output_dir / "score.txt", *sorted(output_dir.glob("score*.txt")), *sorted(output_dir.glob("score*.star"))]
    score_file = next((path for path in candidates if path.is_file()), None)
    if score_file is None:
        return [], {}, None
    scores: dict[int, float] = {}
    for line in score_file.read_text(errors="replace").splitlines():
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        if len(numbers) < 2:
            continue
        try:
            class_id, score = int(float(numbers[0])), float(numbers[1])
        except ValueError:
            continue
        scores[class_id] = score
    selected = [class_id for class_id, score in sorted(scores.items()) if score < threshold]
    return selected, scores, str(score_file)


def evaluate_2d_classes_with_cryosift(
    project_uid: str,
    job_uid: str,
    threshold: float = 3.0,
    output_dir: str | None = None,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    """Run CryoSift on a completed Class 2D job and return normalized scores."""
    evaluator = _env_path("CRYOAGENT_CRYOSIFT_RUNNER", DEFAULT_EVALUATOR)
    weights = _env_path("CRYOAGENT_CRYOSIFT_WEIGHTS", DEFAULT_WEIGHTS)
    python_executable = os.getenv("CRYOAGENT_CRYOSIFT_PYTHON", "/home/lisongyang/.conda/envs/cryosift/bin/python")
    conda_env = os.getenv("CRYOAGENT_CRYOSIFT_CONDA_ENV")
    missing = [str(path) for path in (evaluator, weights) if not path.is_file()]
    if missing:
        return {"success": False, "status": "not_configured", "tool": "cryosift", "project_uid": project_uid, "job_uid": job_uid, "missing_paths": missing, "message": "CryoSift evaluator or model weights are not configured."}
    try:
        classification_dir = _job_directory(project_uid, job_uid)
        result_dir = Path(output_dir).expanduser().resolve() if output_dir else classification_dir / "cryosift_eval"
        result_dir.mkdir(parents=True, exist_ok=True)
        command = []
        if conda_env:
            command.extend([os.getenv("CONDA_EXE", "conda"), "run", "-n", conda_env, python_executable])
        else:
            command.append(python_executable)
        command.extend([str(evaluator), "--input", str(classification_dir), "--output", str(result_dir), "--weights", str(weights)])
        completed = subprocess.run(command, capture_output=True, text=True, timeout=max(timeout_seconds, 1), check=False)
        selected, scores, score_file = _parse_scores(result_dir, threshold)
        if completed.returncode != 0 and not scores:
            return {"success": False, "status": "failed", "tool": "cryosift", "job_uid": job_uid, "returncode": completed.returncode, "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-4000:]}
        return {"success": True, "status": "completed", "tool": "cryosift", "project_uid": project_uid, "job_uid": job_uid, "threshold": threshold, "selected_class_ids": selected, "class_scores": [{"class_id": class_id, "score": score, "selected": class_id in selected} for class_id, score in sorted(scores.items())], "output_dir": str(result_dir), "score_file": score_file, "returncode": completed.returncode}
    except Exception as exc:
        return {"success": False, "status": "error", "tool": "cryosift", "project_uid": project_uid, "job_uid": job_uid, "error_type": type(exc).__name__, "message": str(exc)}
