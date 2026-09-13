"""Explicit v5 UI contracts for interactive CryoSPARC job types.

An interactive tag only means that the job presents a web UI.  It does not
mean that every job accepts the same completion endpoint or that its state can
be safely supplied by an autonomous runner.
"""
from copy import deepcopy
from typing import Any


_CONTRACTS: dict[str, dict[str, Any]] = {
    "inspect_picks_v2": {
        "autonomous": True,
        "state_action": "get_interactive_info",
        "update_action": "set_thresholds",
        "filament_update_action": "set_filament_thresholds",
        "completion_action": "shutdown_interactive",
    },
    "select_2D": {
        "autonomous": True,
        "completion_action": "finish",
    },
    "manual_picker_v2": {
        "autonomous": False,
        "completion_action": "begin_extract",
        "required_state": ["manual_picks", "box_size_pix"],
        "reason": (
            "Manual Picker completion requires human-created picks and a "
            "human-selected extraction box size."
        ),
    },
    "curate_exposures_v2": {
        "autonomous": False,
        "completion_action": "shutdown_interactive",
        "required_state": ["curation_selection"],
        "reason": "Exposure curation completion requires an explicit curation selection.",
    },
}


def interactive_contract_for(job_type: str) -> dict[str, Any] | None:
    """Return a copy of the documented v5 contract for *job_type*, if known."""
    contract = _CONTRACTS.get(job_type)
    return deepcopy(contract) if contract is not None else None


def autonomous_interactive_contract_available(job_type: str) -> bool:
    """Whether the runner can complete this interactive job without inventing UI state."""
    contract = interactive_contract_for(job_type)
    return bool(contract and contract.get("autonomous"))
