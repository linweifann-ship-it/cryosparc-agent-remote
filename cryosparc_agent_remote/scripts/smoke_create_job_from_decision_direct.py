# Creates a real CryoSPARC job in a target workspace from a saved model decision.
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryosparc_client import cryosparc_client
from job_result import wait_for_job_result_package
from model_input_builder import build_model_input_payload


DEFAULT_LANE = "g8m192_4090_slurm"
DEFAULT_INPUT_BY_JOB_TYPE = {
    "class_2D_new": "particles",
    "extract_micrographs_multi": "particles",
    "select_2D": "particles",
}
DEFAULT_CONNECTIONS_BY_JOB_TYPE = {
    "select_2D": {
        "particles": "particles",
        "templates": "class_averages",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse the model decision and live CryoSPARC execution target."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace")
    parser.add_argument("--workspace-title")
    parser.add_argument("--workspace-desc", default="Created by cryosparc_agent live model closed-loop test.")
    parser.add_argument("--decision-json-file", required=True)
    parser.add_argument("--source-job", required=True)
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--input-name")
    parser.add_argument("--lane", default=DEFAULT_LANE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-queue", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument("--dataset-json", default="{}")
    parser.add_argument("--known-workflow-dir", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    """Create, optionally queue, and optionally monitor a real CryoSPARC job."""
    args = parse_args()
    decision = json.loads(Path(args.decision_json_file).read_text())
    job_type = decision.get("action") or decision.get("job_type")
    if not job_type:
        raise SystemExit("Decision must include action or job_type.")

    connections = build_connections(job_type, args)
    if not connections:
        raise SystemExit(f"No default input mapping for job_type {job_type!r}.")

    cs = cryosparc_client()
    project = cs.find_project(args.project)
    workspace = resolve_workspace(project, args)
    if args.dry_run:
        print(
            json.dumps(
                build_dry_run_result(args, workspace, decision, job_type, connections),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
        return

    job = workspace.create_job(
        job_type,
        connections=connections,
        params=decision.get("parameters") or {},
        title=f"Agent live {job_type} from {args.source_job}",
        desc="Created from a saved model decision by cryosparc_agent.",
    )

    queued = False
    if not args.no_queue:
        queue_job(job, args.lane)
        queued = True

    result: dict[str, Any] = {
        "success": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_uid": args.project,
        "workspace_uid": workspace.uid,
        "workspace_title": workspace.title,
        "decision": decision,
        "created_job": {
            "job_uid": job.uid,
            "job_type": job_type,
            "status": "queued" if queued else job.status,
            "queued": queued,
            "lane": args.lane if queued else None,
            "connections": {
                input_name: {
                    "source_job": source_job,
                    "source_output": source_output,
                }
                for input_name, (source_job, source_output) in connections.items()
            },
            "parameters": decision.get("parameters") or {},
        },
        "job_result": None,
        "next_model_input": None,
    }

    if args.wait_timeout_seconds > 0:
        package = wait_for_job_result_package(
            project_uid=args.project,
            workspace_uid=workspace.uid,
            job_uid=job.uid,
            timeout_seconds=args.wait_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            include_next_candidates=False,
        )
        result["job_result"] = package
        if package.get("ready_for_model"):
            result["next_model_input"] = build_model_input_payload(
                project_uid=args.project,
                workspace_uid=workspace.uid,
                current_job_uid=job.uid,
                dataset_info=json.loads(args.dataset_json),
                known_workflow_dirs=args.known_workflow_dir or None,
            )

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def build_connections(
    job_type: str,
    args: argparse.Namespace,
) -> dict[str, tuple[str, str]]:
    """Build CryoSPARC input connections from the model decision context."""
    if args.input_name:
        return {args.input_name: (args.source_job, args.source_output)}

    connection_template = DEFAULT_CONNECTIONS_BY_JOB_TYPE.get(job_type)
    if connection_template:
        return {
            input_name: (args.source_job, source_output)
            for input_name, source_output in connection_template.items()
        }

    input_name = DEFAULT_INPUT_BY_JOB_TYPE.get(job_type)
    if not input_name:
        return {}
    return {input_name: (args.source_job, args.source_output)}


def build_dry_run_result(
    args: argparse.Namespace,
    workspace: Any,
    decision: dict[str, Any],
    job_type: str,
    connections: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    """Return the job creation plan without creating a CryoSPARC job."""
    return {
        "success": True,
        "dry_run": True,
        "project_uid": args.project,
        "workspace_uid": workspace.uid,
        "workspace_title": workspace.title,
        "decision": decision,
        "planned_job": {
            "job_type": job_type,
            "title": f"Agent live {job_type} from {args.source_job}",
            "connections": {
                input_name: {
                    "source_job": source_job,
                    "source_output": source_output,
                }
                for input_name, (source_job, source_output) in connections.items()
            },
            "parameters": decision.get("parameters") or {},
            "lane": normalized_lane(args.lane),
            "will_queue": not args.no_queue,
            "human_action_required": job_type == "select_2D",
            "instruction": (
                "Open the created Select 2D job in the CryoSPARC UI, choose "
                "good 2D classes, then finish the interactive job."
                if job_type == "select_2D"
                else None
            ),
        },
    }


def queue_job(job: Any, lane: str | None) -> None:
    """Queue a job, allowing interactive CPU jobs to run without a lane."""
    selected_lane = normalized_lane(lane)
    if selected_lane:
        job.queue(lane=selected_lane)
    else:
        job.queue()


def normalized_lane(lane: str | None) -> str | None:
    """Treat empty lane sentinels as no explicit queue lane."""
    if lane is None:
        return None
    stripped = lane.strip()
    if stripped.lower() in {"", "none", "null"}:
        return None
    return stripped


def resolve_workspace(project: Any, args: argparse.Namespace) -> Any:
    """Find the requested workspace or create a new one."""
    if args.workspace:
        return project.find_workspace(args.workspace)
    title = args.workspace_title or f"Agent live test {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return project.create_workspace(title=title, desc=args.workspace_desc)


if __name__ == "__main__":
    main()
