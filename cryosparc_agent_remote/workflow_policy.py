"""Conservative cryo-EM workflow stage policy.

This layer ranks and explains candidates; CryoSPARC Registry remains the
source of truth for technical input compatibility.
"""
from typing import Any, Dict, Iterable, List

POLICY_VERSION = "cryoem_standard_v1"

STAGE_ORDER = [
    "not_started",
    "import",
    "motion_correction",
    "ctf_estimation",
    "particle_picking",
    "particle_curation",
    "particle_extraction",
    "classification_2d",
    "particle_selection",
    "initial_model_3d",
    "refinement_3d",
    "validation",
    "completed",
]

JOB_STAGE = {
    "import_movies": "import",
    "import_micrographs": "import",
    "patch_motion_correction_multi": "motion_correction",
    "patch_motion_correction_multi_v2": "motion_correction",
    "patch_ctf_estimation_multi": "ctf_estimation",
    "blob_picker_gpu": "particle_picking",
    "template_picker_gpu": "particle_picking",
    "auto_blob_picker_gpu": "particle_picking",
    "inspect_picks_v2": "particle_curation",
    "curate_exposures_v2": "particle_curation",
    "extract_micrographs_multi": "particle_extraction",
    "extract_micrographs_cpu_parallel": "particle_extraction",
    "class_2D_new": "classification_2d",
    "select_2D": "particle_selection",
    "homo_abinit": "initial_model_3d",
    "class_3D": "initial_model_3d",
    "class_3D_new": "initial_model_3d",
    "homo_refine_new": "refinement_3d",
    "nonuniform_refine_new": "refinement_3d",
    "refine_3D_new": "refinement_3d",
    "validation": "validation",
    "validate": "validation",
}

NEXT_STAGES = {
    "not_started": ("import",),
    "import": ("motion_correction", "ctf_estimation"),
    "motion_correction": ("ctf_estimation",),
    "ctf_estimation": ("particle_picking",),
    "particle_picking": ("particle_curation", "particle_extraction"),
    "particle_curation": ("particle_extraction", "particle_picking"),
    "particle_extraction": ("classification_2d",),
    "classification_2d": ("particle_selection", "initial_model_3d"),
    "particle_selection": ("initial_model_3d", "classification_2d"),
    "initial_model_3d": ("refinement_3d",),
    "refinement_3d": ("validation", "refinement_3d"),
    "validation": ("refinement_3d", "completed"),
    "completed": (),
}


def job_stage(job_type: str) -> str:
    """Map a CryoSPARC job type to the standard workflow stage."""
    return JOB_STAGE.get(job_type, "unknown")


def infer_current_stage(
    nodes: Iterable[Dict[str, Any]],
    current_node_id: Any = None,
) -> Dict[str, Any]:
    """Infer stage from the requested node, or the latest meaningful node."""
    node_list = list(nodes)
    if not node_list:
        return stage_payload("not_started", None, "No CryoSPARC jobs exist yet.")
    if current_node_id is not None:
        selected = next(
            (node for node in node_list
             if node.get("workflow_node_id") == current_node_id
             or node.get("cryosparc_job_uid") == current_node_id),
            None,
        )
        if selected is not None and job_stage(selected.get("job_type", "")) != "unknown":
            stage = job_stage(selected.get("job_type", ""))
            return stage_payload(
                stage,
                selected.get("workflow_node_id"),
                f"Requested current job is {selected.get('job_type')} ({selected.get('status')}).",
            )
    ordered = sorted(node_list, key=lambda node: job_number(node.get("cryosparc_job_uid", "")))
    for node in reversed(ordered):
        stage = job_stage(node.get("job_type", ""))
        if stage != "unknown":
            return stage_payload(
                stage,
                node.get("workflow_node_id"),
                f"Latest recognized job is {node.get('job_type')} ({node.get('status')}).",
            )
    return stage_payload("not_started", None, "No recognized standard cryo-EM stage was found.")


def stage_payload(stage: str, source_node_id: Any, reason: str) -> Dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "current_stage": stage,
        "source_node_id": source_node_id,
        "next_stages": list(NEXT_STAGES.get(stage, ())),
        "reason": reason,
    }


def annotate_candidates(
    candidates: List[Dict[str, Any]],
    current_stage: str,
) -> List[Dict[str, Any]]:
    """Attach advisory stage labels without filtering any candidate."""
    next_stages = set(NEXT_STAGES.get(current_stage, ()))
    annotated = []
    for candidate in candidates:
        stage = job_stage(candidate.get("job_type", ""))
        if stage in next_stages:
            recommendation = "preferred_next_stage"
        elif stage == current_stage:
            recommendation = "same_stage_iteration"
        else:
            recommendation = "outside_primary_transition"
        enriched = dict(candidate)
        enriched["workflow_stage"] = stage
        enriched["workflow_policy_recommendation"] = recommendation
        annotated.append(enriched)
    return annotated


def job_number(uid: str) -> int:
    digits = "".join(char for char in str(uid) if char.isdigit())
    return int(digits) if digits else -1
