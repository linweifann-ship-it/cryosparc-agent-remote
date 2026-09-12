#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Split SFT JSONL by dataset_id to avoid leakage across workflow steps.')
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--train-jsonl', required=True)
    parser.add_argument('--eval-jsonl', required=True)
    parser.add_argument('--eval-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write('\n')


def main() -> None:
    args = parse_args()
    records = load_jsonl(Path(args.input_jsonl))
    grouped = defaultdict(list)
    for record in records:
        key = record.get('dataset_id') or record.get('workflow_id') or record.get('id')
        grouped[key].append(record)

    keys = list(grouped.keys())
    random.Random(args.seed).shuffle(keys)
    eval_group_count = max(1, round(len(keys) * args.eval_ratio)) if len(keys) > 1 else 0
    eval_keys = set(keys[:eval_group_count])

    train_records = []
    eval_records = []
    for key, bucket in grouped.items():
        if key in eval_keys:
            eval_records.extend(bucket)
        else:
            train_records.extend(bucket)

    if not train_records:
        raise ValueError('Train split is empty after dataset grouping.')

    write_jsonl(Path(args.train_jsonl), train_records)
    write_jsonl(Path(args.eval_jsonl), eval_records)

    print(json.dumps({
        'dataset_group_count': len(keys),
        'eval_dataset_group_count': len(eval_keys),
        'train_record_count': len(train_records),
        'eval_record_count': len(eval_records),
        'train_jsonl': str(Path(args.train_jsonl).resolve()),
        'eval_jsonl': str(Path(args.eval_jsonl).resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
