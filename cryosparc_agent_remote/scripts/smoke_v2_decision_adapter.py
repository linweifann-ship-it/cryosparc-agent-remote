# Runs a V2 decision adapter smoke test against a real CryoSPARC workflow.
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2_decision_adapter import execute_v2_model_decision_payload


def parse_args() -> argparse.Namespace:
    """Parse the workflow context and simulated V2 model decision."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--current-node", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--parameters-json", default="{}")
    parser.add_argument("--live", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Print the adapted internal decision and execution result."""
    args = parse_args()
    decision = {
        "schema_version": "3.0",
        "decision_type": "forward",
        "selected_actions": [
            {
                "job_type": args.action,
                "parameters": json.loads(args.parameters_json),
            }
        ],
    }
    result = execute_v2_model_decision_payload(
        decision,
        project_uid=args.project,
        workspace_uid=args.workspace,
        current_node_id=args.current_node,
        dry_run=not args.live,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
