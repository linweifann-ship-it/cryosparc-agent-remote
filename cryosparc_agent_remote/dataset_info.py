"""Normalization for user-supplied dataset acquisition and input-file facts."""
from typing import Any


ACQUISITION_ALIASES = {
    "psize_A": ("pixel_size_A",),
    "accel_kv": ("accelerating_voltage_kv", "voltage_kV", "voltage_kv"),
    "cs_mm": ("spherical_aberration_mm", "SA_mm"),
    "total_dose_e_per_A2": ("total_exposure_dose_e_per_A2",),
}
INPUT_FILE_ALIASES = {
    "micrograph_blob_paths": ("micrographs_data_path", "micrograph_data_path"),
    "movie_blob_paths": ("movies_data_path", "movie_data_path", "raw_movies"),
    "volume_blob_path": ("volume_data_path",),
}


def normalize_dataset_info(dataset_info: dict[str, Any]) -> dict[str, Any]:
    """Accept user-facing names while preserving CryoSPARC canonical fields."""
    normalized = dict(dataset_info or {})
    facts = normalized.get("dataset_parameter_facts") or {}
    if isinstance(facts, dict):
        for key, value in facts.items():
            if not isinstance(value, (dict, list)):
                normalized.setdefault(key, value)
    nested = normalized.get("acquisition_parameters") or normalized.get(
        "experimental_parameters"
    ) or {}
    if isinstance(nested, dict):
        for key, value in nested.items():
            normalized.setdefault(key, value)
    for canonical, aliases in ACQUISITION_ALIASES.items():
        if normalized.get(canonical) is None:
            for alias in aliases:
                if normalized.get(alias) is not None:
                    normalized[canonical] = normalized[alias]
                    break
    files = normalized.get("available_input_files") or {}
    files = dict(files) if isinstance(files, dict) else {}
    for canonical, aliases in INPUT_FILE_ALIASES.items():
        if files.get(canonical):
            continue
        for alias in aliases:
            if normalized.get(alias):
                files[canonical] = normalized[alias]
                break
    if files:
        normalized["available_input_files"] = files
    return normalized
