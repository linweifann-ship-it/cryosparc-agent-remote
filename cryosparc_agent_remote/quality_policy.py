"""Evidence-based quality assessment and conservative recovery advice."""
from typing import Any, Dict, List

QUALITY_POLICY_VERSION = "cryoem_quality_v1"


def assess_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """Assess only signals available from the normalized CryoSPARC state."""
    status = node.get("status")
    outputs = node.get("outputs") or {}
    total_items = sum(int(output.get("num_items") or 0) for output in outputs.values())
    available_outputs = [name for name, output in outputs.items() if output.get("available")]
    warnings = list(node.get("run_errors", {}).get("errors_run") or [])
    errors = []
    for key in ("errors_build_inputs", "errors_build_params"):
        value = node.get("run_errors", {}).get(key)
        if value:
            errors.append({"source": key, "value": value})
    observed_metrics = extract_quality_metrics(node)
    resolution_evidence = find_resolution_evidence(node, observed_metrics)

    if status in {"failed", "killed"} or node.get("has_error"):
        quality_status = "failed"
        recommendation = "recover_or_review"
    elif status != "completed":
        quality_status = "not_terminal"
        recommendation = "wait"
    elif total_items == 0 and outputs:
        quality_status = "completed_without_outputs"
        recommendation = "recover_or_review"
    elif node.get("has_warning"):
        quality_status = "completed_with_warnings"
        recommendation = "continue_with_caution"
    else:
        quality_status = "completed_observed"
        recommendation = "continue_if_next_inputs_are_compatible"

    return {
        "policy_version": QUALITY_POLICY_VERSION,
        "quality_status": quality_status,
        "recommendation": recommendation,
        "evidence_level": "observed_state_only",
        "output_item_count": total_items,
        "available_outputs": available_outputs,
        "run_error_count": len(errors),
        "run_warning_count": len(warnings),
        "resolution_evidence": resolution_evidence,
        "observed_metrics": observed_metrics,
        "recovery_options": recovery_options(node, quality_status),
        "limitations": [
            "FSC/resolution values are not inferred unless exposed by CryoSPARC summary data.",
            "Image-based quality requires a separate image artifact extractor.",
        ],
    }


def extract_quality_metrics(node: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a compact set of numeric quality/acquisition values from summaries."""
    selected = {}
    tokens = (
        "fsc", "resolution", "ctf", "defocus", "astig", "num_particles",
        "particle", "pick", "motion", "psize", "accel_kv", "cs_mm",
        "total_dose",
    )
    for output_name, output in (node.get("outputs") or {}).items():
        for field_name in ("summary_values", "latest_summary_stat_values"):
            for key, value in (output.get(field_name) or {}).items():
                key_text = str(key)
                if not any(token in key_text.lower() for token in tokens):
                    continue
                selected[key_text] = {
                    "value": value,
                    "source": f"outputs.{output_name}.{field_name}.{key_text}",
                }
    return selected


def find_resolution_evidence(
    node: Dict[str, Any],
    observed_metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Report resolution evidence and its value when CryoSPARC exposes it."""
    keys = set()
    for output in (node.get("outputs") or {}).values():
        keys.update(output.get("summary_keys") or [])
        keys.update(output.get("latest_summary_stat_keys") or [])
    matches = sorted(key for key in keys if any(token in key.lower() for token in ("fsc", "resolution", "res_")))
    values = {key: item["value"] for key, item in (observed_metrics or {}).items() if key in matches}
    return {
        "available": bool(values),
        "keys": matches,
        "values": values,
        "note": "Values are copied from CryoSPARC summaries; absent values remain unavailable.",
    }


def recovery_options(node: Dict[str, Any], quality_status: str) -> List[str]:
    if quality_status == "failed":
        return ["retry_same_job_with_adjusted_parameters", "request_human_review"]
    if quality_status == "completed_without_outputs":
        return ["inspect_input_and_parameters", "retry_same_job"]
    if quality_status == "completed_with_warnings":
        return ["inspect_warnings", "continue_or_branch"]
    return []


def build_retry_candidate(node: Dict[str, Any]) -> Dict[str, Any]:
    """Create an approval-gated retry candidate for a failed terminal Job."""
    return {
        "action_id": f"retry_{node['cryosparc_job_uid']}",
        "action_type": "forward",
        "workflow_node_id": f"{node['workflow_node_id']}:retry",
        "reference_job_uid": node["cryosparc_job_uid"],
        "reference_status": node["status"],
        "job_type": node["job_type"],
        "description": f"Retry failed {node['job_type']} after reviewing its errors.",
        "execution_mode": "create_job",
        "available": True,
        "blocked_by": [],
        "required_inputs": node.get("inputs") or {},
        "missing_required_inputs": [],
        "parameter_template": {},
        "default_parameters": node.get("key_parameters") or {},
        "approval_required": True,
        "approval_reasons": ["failed_job_retry_requires_human_review"],
        "recovery_policy": "retry_failed_job",
    }
