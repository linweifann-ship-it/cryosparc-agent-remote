# Defines lightweight job metadata used by the generic CryoSPARC job executor.
from copy import deepcopy
from typing import Any


HIGH_GPU_APPROVAL_THRESHOLD = 4
DEFAULT_GPU_LANE = "g8m192_4090_slurm"


# Parameter rules are intentionally small: only expose knobs the model may edit.
JOB_SPECS: dict[str, dict[str, Any]] = {
    "import_movies": {
        "category": "import",
        "requires_gpu": False,
        "requires_approval": False,
        "interactive": False,
        "parameter_template": {
            "blob_paths": {"type": "string", "required": True},
            "gainref_path": {"type": "string"},
            "psize_A": {"type": "number", "minimum": 0},
            "accel_kv": {"type": "number", "minimum": 0},
            "cs_mm": {"type": "number", "minimum": 0},
            "total_dose_e_per_A2": {"type": "number", "minimum": 0},
        },
    },
    "import_micrographs": {
        "category": "import",
        "requires_gpu": False,
        "requires_approval": False,
        "interactive": False,
        "parameter_template": {
            "blob_paths": {"type": "string", "required": True},
            "psize_A": {"type": "number", "minimum": 0},
            "accel_kv": {"type": "number", "minimum": 0},
            "cs_mm": {"type": "number", "minimum": 0},
            "total_dose_e_per_A2": {"type": "number", "minimum": 0},
        },
    },
    "patch_motion_correction_multi": {
        "category": "motion_correction",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["movies"],
        "parameter_template": {
            "compute_num_gpus": {"type": "integer", "minimum": 1, "maximum": 8},
        },
    },
    "patch_ctf_estimation_multi": {
        "category": "ctf",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["exposures"],
        "parameter_template": {
            "compute_num_gpus": {"type": "integer", "minimum": 1, "maximum": 8},
            "df_search_min": {"type": "number", "minimum": 0},
            "df_search_max": {"type": "number", "minimum": 0},
        },
    },
    "curate_exposures_v2": {
        "category": "curation",
        "requires_gpu": False,
        "requires_approval": True,
        "interactive": True,
        "required_inputs": ["exposures"],
        "parameter_template": {},
    },
    "blob_picker_gpu": {
        "category": "picking",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["micrographs"],
        "parameter_template": {
            "diameter": {"type": "number", "minimum": 0},
            "diameter_max": {"type": "number", "minimum": 0},
        },
    },
    "template_picker_gpu": {
        "category": "picking",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["micrographs", "templates"],
        "parameter_template": {
            "diameter": {"type": "number", "minimum": 0},
            "min_distance": {"type": "number", "minimum": 0},
            "use_ctf": {"type": "boolean"},
        },
    },
    "inspect_picks_v2": {
        "category": "inspection",
        "requires_gpu": False,
        "requires_approval": True,
        "interactive": True,
        "required_inputs": ["particles"],
        "parameter_template": {
            "min_score": {"type": "number"},
            "max_score": {"type": "number"},
            "min_power": {"type": "number"},
            "max_power": {"type": "number"},
        },
    },
    "extract_micrographs_multi": {
        "category": "extraction",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["micrographs", "particles"],
        "parameter_template": {
            "compute_num_gpus": {"type": "integer", "minimum": 1, "maximum": 8},
            "box_size_pix": {"type": "integer", "minimum": 32},
        },
    },
    "extract_micrographs_cpu_parallel": {
        "category": "extraction",
        "requires_gpu": False,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["micrographs", "particles"],
        "parameter_template": {
            "box_size_pix": {"type": "integer", "minimum": 32},
            "compute_num_cores": {"type": "integer", "minimum": 1},
        },
    },
    "class_2D_new": {
        "category": "classification",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["particles"],
        "parameter_template": {
            "compute_num_gpus": {"type": "integer", "minimum": 1, "maximum": 8},
            "class2D_K": {"type": "integer", "minimum": 2},
        },
    },
    "select_2D": {
        "category": "selection",
        "requires_gpu": False,
        "requires_approval": True,
        "interactive": True,
        "required_inputs": ["particles"],
        "parameter_template": {
            "selected_templates": {"type": "array"},
            "resolution_better_than": {"type": "number", "minimum": 0},
            "particle_count_above": {"type": "integer", "minimum": 0},
        },
    },
    "homo_abinit": {
        "category": "reconstruction",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["particles"],
        "parameter_template": {},
    },
    "homo_refine_new": {
        "category": "refinement",
        "requires_gpu": True,
        "requires_approval": False,
        "interactive": False,
        "default_lane": DEFAULT_GPU_LANE,
        "required_inputs": ["particles", "volume"],
        "parameter_template": {
            "compute_use_ssd": {"type": "boolean"},
            "refine_symmetry": {"type": "string"},
            "refine_defocus_refine": {"type": "boolean"},
            "refine_ctf_global_refine": {"type": "boolean"},
        },
    },
}


def get_job_spec(job_type: str) -> dict[str, Any]:
    """Return metadata for a job type, using safe defaults for unknown jobs."""
    spec = deepcopy(JOB_SPECS.get(job_type, {}))
    spec.setdefault("category", "unknown")
    spec.setdefault("requires_gpu", False)
    spec.setdefault("requires_approval", True)
    spec.setdefault("interactive", False)
    spec.setdefault("max_auto_gpus", HIGH_GPU_APPROVAL_THRESHOLD)
    spec.setdefault("parameter_template", {})
    spec.setdefault("required_inputs", [])
    return spec


def get_parameter_template(job_type: str) -> dict[str, dict[str, Any]]:
    """Return the editable parameter template for a job type."""
    return deepcopy(get_job_spec(job_type)["parameter_template"])


def list_supported_job_types() -> list[str]:
    """Return job types with explicit local metadata."""
    return sorted(JOB_SPECS)
