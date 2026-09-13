"""Preserve observed Inspect Picks evidence across autonomous-loop rounds."""
from typing import Any

THRESHOLD_KEYS = ("ncc_score_thresh", "lpower_thresh_min", "lpower_thresh_max", "curv_thresh", "sinu_thresh", "keep_threshold", "do_auto_cluster")


def build_inspect_picks_observation(job_result: dict[str, Any], applied_parameters: dict[str, Any], visual_context: dict[str, Any] | None) -> dict[str, Any]:
    """Package only evidence already observed before or during this job."""
    visual_context = visual_context or {}
    selected = ((job_result.get("outputs") or {}).get("particles") or {}).get("num_items")
    total = visual_context.get("input_particle_count")
    sheet = visual_context.get("contact_sheet") or {}
    return {
        "source_job_uid": job_result.get("job_uid"), "source_job_type": "inspect_picks_v2",
        "applied_thresholds": {name: applied_parameters[name] for name in THRESHOLD_KEYS if name in applied_parameters},
        "selected_particle_count": selected, "total_input_particle_count": total,
        "retention_fraction": selected / total if isinstance(selected, (int, float)) and isinstance(total, (int, float)) and total else None,
        "score_statistics": visual_context.get("score_statistics"),
        "representative_overlay": {"source_job_uid": (visual_context.get("source") or {}).get("job_uid"), "image_count": visual_context.get("image_count"), "local_path": sheet.get("local_path"), "mime_type": sheet.get("mime_type")},
    }
