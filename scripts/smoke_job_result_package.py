# Reads a real CryoSPARC job and prints the MCP status/result package.
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from job_result import wait_for_job_result_package


def parse_args() -> argparse.Namespace:
    """Parse the target CryoSPARC job and polling settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument(
        "--no-next-candidates",
        action="store_true",
        help="Do not include next candidate actions in completed result packages.",
    )
    return parser.parse_args()


def main() -> None:
    """Print the package that MCP would store internally or send to the model."""
    args = parse_args()
    package = wait_for_job_result_package(
        project_uid=args.project,
        workspace_uid=args.workspace,
        job_uid=args.job,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        include_next_candidates=not args.no_next_candidates,
    )
    print(json.dumps(package, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
