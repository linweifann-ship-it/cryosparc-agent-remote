# Builds a V2 model input payload from a real CryoSPARC workflow job.
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_xml import load_dataset_info_from_xml
from model_input_builder import build_model_input_payload


def parse_args() -> argparse.Namespace:
    """Parse dataset metadata and target CryoSPARC job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--job")
    parser.add_argument("--empiar-id")
    parser.add_argument("--emdb-id")
    parser.add_argument("--input-type")
    parser.add_argument("--macromolecules-type")
    parser.add_argument("--num-of-maps", type=int)
    parser.add_argument("--abstract")
    parser.add_argument("--resolution", action="append", type=float)
    parser.add_argument("--known-workflow-dir", action="append")
    parser.add_argument("--dataset-xml-file")
    return parser.parse_args()


def main() -> None:
    """Print either V2 model input or MCP-internal status for active jobs."""
    args = parse_args()
    dataset_info = (
        load_dataset_info_from_xml(args.dataset_xml_file)
        if args.dataset_xml_file
        else {}
    )
    cli_dataset_info = {
        "empiar_id": args.empiar_id,
        "emdb_id": args.emdb_id,
        "resolution": args.resolution,
        "input_type": args.input_type,
        "macromolecules_type": args.macromolecules_type,
        "num_of_maps": args.num_of_maps,
        "abstract": args.abstract,
    }
    dataset_info.update(
        {
            key: value
            for key, value in cli_dataset_info.items()
            if value is not None
        }
    )
    payload = build_model_input_payload(
        project_uid=args.project,
        workspace_uid=args.workspace,
        current_job_uid=args.job,
        dataset_info=dataset_info,
        known_workflow_dirs=args.known_workflow_dir,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
