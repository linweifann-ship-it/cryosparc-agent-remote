# Converts EMDB XML metadata into a dataset_info JSON file.
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_xml import load_dataset_info_from_xml


def parse_args() -> argparse.Namespace:
    """Parse XML input and optional JSON output path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-file", required=True)
    parser.add_argument("--output-json-file")
    return parser.parse_args()


def main() -> None:
    """Print and optionally save parsed dataset_info."""
    args = parse_args()
    dataset_info = load_dataset_info_from_xml(args.xml_file)
    text = json.dumps(dataset_info, indent=2, ensure_ascii=False, default=str)
    if args.output_json_file:
        Path(args.output_json_file).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
