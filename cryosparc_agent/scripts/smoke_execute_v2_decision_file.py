# Executes or dry-runs a saved V2 model decision against CryoSPARC state.
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from job_result import wait_for_job_result_package
from model_input_builder import build_model_input_payload
from v2_decision_adapter import execute_v2_model_decision_payload


def parse_args() -> argparse.Namespace:
    """Parse workflow context, decision file, and execution options."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--current-node")
    parser.add_argument("--decision-json-file", required=True)
    parser.add_argument("--dataset-json", default="{}")
    parser.add_argument("--known-workflow-dir", action="append", default=[])
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    """Run the V2 adapter and optionally monitor created jobs."""
    args = parse_args()
    decision = json.loads(Path(args.decision_json_file).read_text())
    dataset_info = json.loads(args.dataset_json)
    execution = execute_v2_model_decision_payload(
        decision,
        project_uid=args.project,
        workspace_uid=args.workspace,
        current_node_id=args.current_node,
        dry_run=not args.live,
    )
    result = {
        "success": execution.get("success", False),
        "live": args.live,
        "project_uid": args.project,
        "workspace_uid": args.workspace,
        "current_node": args.current_node,
        "decision": decision,
        "execution": execution,
        "job_results": [],
        "next_model_inputs": [],
    }
    if args.live:
        result["job_results"] = wait_for_created_jobs(args, execution)
        for job_package in result["job_results"]:
            if job_package.get("ready_for_model"):
                result["next_model_inputs"].append(
                    build_model_input_payload(
                        project_uid=args.project,
                        workspace_uid=args.workspace,
                        current_job_uid=job_package["job_uid"],
                        dataset_info=dataset_info,
                        known_workflow_dirs=args.known_workflow_dir or None,
                    )
                )

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def wait_for_created_jobs(
    args: argparse.Namespace,
    execution: dict,
) -> list[dict]:
    """Poll jobs created by live execution until terminal state or timeout."""
    job_results = []
    for execution_result in execution.get("execution_result", {}).get(
        "execution_results",
        [],
    ):
        job_uid = execution_result.get("job_uid")
        if not job_uid:
            job_results.append(execution_result)
            continue
        job_results.append(
            wait_for_job_result_package(
                project_uid=args.project,
                workspace_uid=args.workspace,
                job_uid=job_uid,
                timeout_seconds=args.wait_timeout_seconds,
                poll_interval_seconds=args.poll_interval_seconds,
                include_next_candidates=False,
            )
        )
    return job_results


if __name__ == "__main__":
    main()
