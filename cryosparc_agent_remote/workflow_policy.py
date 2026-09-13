"""Conservative cryo-EM workflow stage policy.

This layer ranks and explains candidates; CryoSPARC Registry remains the
source of truth for technical input compatibility.
"""
from typing import Any, Dict, Iterable, List

POLICY_VERSION = "cryoem_standard_v2"

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


def is_manual_picker_candidate(candidate: Dict[str, Any]) -> bool:
    """Recognize manual picker actions from Registry metadata without job-specific IDs."""
    text = " ".join([
        str(candidate.get("job_type") or ""),
        str(candidate.get("description") or ""),
    ]).lower()
    return "manual" in text and "picker" in text


def is_automated_picker_candidate(candidate: Dict[str, Any]) -> bool:
    """Recognize non-manual particle-picking actions across Registry variants."""
    if is_manual_picker_candidate(candidate):
        return False
    metadata = candidate.get("job_spec_metadata") or {}
    category = str(metadata.get("category") or "").lower()
    job_type = str(candidate.get("job_type") or "").lower()
    return "picker" in job_type or category in {"picking", "particle_picking", "deep_picker"}


def apply_autonomous_picker_preference(
    candidates: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Keep manual picking as a fallback while an automated picker is usable.

    This is a workflow-mode rule, not a fixed diameter or Blob Picker rule.
    An automated picker remains selectable even when a tunable scientific
    parameter needs an explicitly labelled heuristic estimate from the Model.
    """
    automated = [
        candidate for candidate in candidates
        if candidate.get("available", True) and is_automated_picker_candidate(candidate)
    ]
    if not automated:
        return candidates, []
    retained = []
    blocked = []
    for candidate in candidates:
        if is_manual_picker_candidate(candidate):
            blocked.append({
                **candidate,
                "available": False,
                "blocked_by": list(candidate.get("blocked_by") or []) + [
                    "Manual Picker is a fallback only while an automated particle picker is available. "
                    "Use an explicitly labelled heuristic estimate for a tunable scientific parameter, "
                    "or request input if no safe estimate is possible."
                ],
            })
        else:
            retained.append(candidate)
    return retained, blocked



def build_decision_guidance(
    dataset_info: Dict[str, Any] | None,
    quality_assessment: Dict[str, Any] | None,
    current_node: Dict[str, Any] | None,
    candidates: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build advisory and enforceable guidance from explicit user goals."""
    dataset_info = dataset_info or {}
    quality_assessment = quality_assessment or {}
    candidate_list = list(candidates)
    target = dataset_info.get("target_resolution_A")
    if target is None:
        target = dataset_info.get("resolution")
    try:
        target = float(target) if target is not None else None
    except (TypeError, ValueError):
        target = None

    observed = quality_assessment.get("resolution_evidence", {}).get("values", {})
    observed_values = []
    for value in observed.values():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            observed_values.append(number)
    current = min(observed_values) if observed_values else None
    refinement_available = any(
        item.get("available")
        and item.get("job_type") in {"homo_refine_new", "nonuniform_refine_new", "refine_3D_new"}
        for item in candidate_list
    )
    target_unmet = (
        target is not None and current is not None and current > target
    )
    inspect_available = any(
        item.get("available") and item.get("job_type") == "inspect_picks_v2"
        for item in candidate_list
    )
    last_action = (current_node or {}).get("job_type")
    inspect_guidance = None
    if inspect_available and last_action in {"blob_picker_gpu", "auto_blob_picker_gpu"}:
        inspect_guidance = {
            "priority": "mandatory",
            "reason": (
                "Inspect Blob Picker locations before extraction so false positives, "
                "missed particles, and diameter/score issues can be reviewed."
            ),
            "model_instruction": (
                "MUST choose inspect_picks_v2 before any extraction or downstream action; "
                "skipping Inspect Picks is not allowed in this run."
            ),
        }
    automated_picker_available = any(
        item.get("available") and is_automated_picker_candidate(item)
        for item in candidate_list
    )
    automated_picker_guidance = None
    if automated_picker_available:
        automated_picker_guidance = {
            "priority": "preferred",
            "model_instruction": (
                "For autonomous single-particle processing, prefer an available automated "
                "particle picker. If a tunable scientific parameter is missing but can be "
                "safely estimated from domain knowledge and current data, provide that estimate "
                "and label it as estimated or assumed in reason/evidence. Do not select a Manual "
                "Picker merely to avoid such an estimate; use Manual Picker only when automation "
                "is unavailable, unsuitable, or explicitly requested by the user."
            ),
        }
    box_sweep_guidance = None
    if last_action == "extract_micrographs_multi" and any(
        item.get("available") and item.get("box_size_trial")
        for item in candidate_list
    ):
        sizes = [
            item.get("default_parameters", {}).get("box_size_pix")
            for item in candidate_list
            if item.get("box_size_trial")
        ]
        box_sweep_guidance = {
            "priority": "mandatory",
            "box_sizes_pix": sizes,
            "model_instruction": (
                "MUST submit all available box-size trial actions in one forward decision; "
                "do not proceed to Class 2D until every trial has completed."
            ),
        }
    reextract_guidance = None
    if last_action == "class_2D_new" and any(
        item.get("available") and item.get("job_type") == "extract_micrographs_multi"
        for item in candidate_list
    ):
        reextract_guidance = {
            "priority": "conditional_fallback",
            "reason": "2D class images can reveal clipping caused by an undersized extraction box.",
            "model_instruction": (
                "If visual 2D evidence shows particles are clipped or truncated, choose "
                "extract_micrographs_multi fallback and increase box_size_pix before rerunning 2D classification."
            ),
        }
    return {
        "target_resolution_A": target,
        "current_resolution_A": current,
        "target_status": (
            "unmet" if target_unmet else
            "met" if target is not None and current is not None else
            "unknown"
        ),
        "must_continue_for_target": bool(target_unmet and refinement_available),
        "refinement_available": refinement_available,
        "inspect_picks": inspect_guidance,
        "automated_picking": automated_picker_guidance,
        "extract_fallback": reextract_guidance,
        "box_size_sweep": box_sweep_guidance,
    }


def job_number(uid: str) -> int:
    digits = "".join(char for char in str(uid) if char.isdigit())
    return int(digits) if digits else -1
