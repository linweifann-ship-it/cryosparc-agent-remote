#!/usr/bin/env python3
"""Minimal live regression for autonomous interactive CryoSPARC job lifecycle."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cryosparc_agent_remote"))

from action_registry import execute_model_decision_payload, get_candidate_actions
from job_result import wait_for_job_result_package
from workflow_state import extract_workflow_state, find_node


TARGETS = {"inspect-picks": "inspect_picks_v2", "select-2d": "select_2D"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--poll-interval-seconds", type=int, default=10)
    return parser.parse_args()


def select_candidate(project_uid, workspace_uid, source_job_uid, target):
    state = extract_workflow_state(project_uid, workspace_uid)
    source = find_node(state, source_job_uid)
    if source is None or source.get("status") != "completed":
        raise ValueError(f"source job {source_job_uid!r} must exist and be completed")
    candidates = get_candidate_actions(project_uid, workspace_uid, source_job_uid)["candidate_actions"]
    job_type = TARGETS[target]
    candidate = next((item for item in candidates if item.get("job_type") == job_type and item.get("available")), None)
    if candidate is None:
        raise ValueError(f"no available {job_type} candidate from {source_job_uid}")
    return candidate, candidates


def candidate_connections(candidate):
    """Materialize Registry input sources into the V2 decision connection shape."""
    connections = {}
    for slot, sources in (candidate.get("required_inputs") or {}).items():
        if not sources:
            raise ValueError(f"mandatory input {slot!r} has no source connection")
        source = sources[0]
        job_uid = source.get("source_job_uid")
        output = source.get("source_output")
        if not job_uid or not output:
            raise ValueError(f"mandatory input {slot!r} has incomplete source connection")
        connections[slot] = {"source_job_uid": job_uid, "source_output": output}
    return connections


def main():
    args = parse_args()
    candidate, candidates = select_candidate(args.project, args.workspace, args.source_job, args.target)
    try:
        connections = candidate_connections(candidate)
    except ValueError as exc:
        print(json.dumps({"result": "FAIL", "stage": "precreate_connections", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"stage": "precreate_connections", "connections": connections}, ensure_ascii=False))
    decision = {
        "schema_version": "1.0", "decision_type": "forward", "reason": "interactive lifecycle regression",
        "confidence": 1.0, "risk_flags": [], "evidence": ["Existing completed source Job selected by regression CLI."],
        "selected_actions": [{
            "action_id": candidate["action_id"], "action_type": candidate["action_type"],
            "workflow_node_id": candidate["workflow_node_id"], "job_type": candidate["job_type"],
            "parameters": {}, "connections": connections,
        }],
    }
    execution = execute_model_decision_payload(
        decision, candidate_actions=candidates, dry_run=False,
        project_uid=args.project, workspace_uid=args.workspace,
    )
    results = execution.get("execution_results") or []
    if not execution.get("success") or len(results) != 1:
        print(json.dumps({"result": "FAIL", "execution": execution}, ensure_ascii=False, indent=2))
        return 1
    job_uid = results[0]["job_uid"]
    package = wait_for_job_result_package(
        args.project, args.workspace, job_uid,
        timeout_seconds=args.timeout_seconds, poll_interval_seconds=args.poll_interval_seconds,
    )
    passed = package.get("ready_for_model") and package.get("status") == "completed"
    print(json.dumps({"result": "PASS" if passed else "FAIL", "source_job_uid": args.source_job,
                      "target": candidate["job_type"], "job_uid": job_uid, "result_package": package},
                     ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
