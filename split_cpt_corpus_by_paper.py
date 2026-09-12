#!/usr/bin/env python3
"""Split CPT corpus JSONL by paper_id to avoid leakage across chunks."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input_jsonl))

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = record.get("paper_id") or record.get("source_file") or record.get("id")
        grouped[key].append(record)

    keys = list(grouped.keys())
    random.Random(args.seed).shuffle(keys)
    eval_group_count = max(1, round(len(keys) * args.eval_ratio)) if len(keys) > 1 else 0
    eval_keys = set(keys[:eval_group_count])

    train_records: list[dict] = []
    eval_records: list[dict] = []
    for key, bucket in grouped.items():
        if key in eval_keys:
            eval_records.extend(bucket)
        else:
            train_records.extend(bucket)

    if not train_records:
        raise ValueError("Train split is empty after paper grouping.")

    train_path = Path(args.train_jsonl)
    eval_path = Path(args.eval_jsonl)
    write_jsonl(train_path, train_records)
    write_jsonl(eval_path, eval_records)

    print(
        json.dumps(
            {
                "paper_group_count": len(keys),
                "eval_paper_group_count": len(eval_keys),
                "train_record_count": len(train_records),
                "eval_record_count": len(eval_records),
                "train_jsonl": str(train_path.resolve()),
                "eval_jsonl": str(eval_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
