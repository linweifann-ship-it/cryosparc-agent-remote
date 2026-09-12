"""Normalization for user-supplied dataset acquisition parameters."""
from typing import Any


ACQUISITION_ALIASES = {
    "psize_A": ("pixel_size_A",),
    "accel_kv": ("accelerating_voltage_kv",),
    "cs_mm": ("spherical_aberration_mm",),
    "total_dose_e_per_A2": ("total_exposure_dose_e_per_A2",),
}


def normalize_dataset_info(dataset_info: dict[str, Any]) -> dict[str, Any]:
    """Accept friendly or CryoSPARC-native acquisition parameter names."""
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
    return normalized
