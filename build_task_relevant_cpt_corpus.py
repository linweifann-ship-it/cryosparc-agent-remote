#!/usr/bin/env python3
"""Merge multiple task-relevant CPT sources into a unified train/eval corpus."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest-json", required=True)
    parser.add_argument("--paper-chunks-jsonl", default=None)
    parser.add_argument("--docs-chunks-jsonl", default=None)
    parser.add_argument("--workflow-chunks-jsonl", default=None)
    parser.add_argument("--dataset-context-jsonl", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bucket-repeat-overrides-json",
        default=None,
        help='Optional JSON like {"procedural_docs": 2, "workflow_examples": 2}.',
    )
    parser.add_argument(
        "--max-records-per-bucket",
        type=int,
        default=None,
        help="Optional cap after repetition, applied independently per bucket.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def bucket_weight_map(source_manifest: dict[str, Any]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in source_manifest.get("recommended_buckets", []):
        bucket_id = item.get("bucket_id")
        weight = item.get("weight")
        if bucket_id is not None and weight is not None:
            weights[str(bucket_id)] = float(weight)
    return weights


def source_info_map(source_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("source_id")): item
        for item in source_manifest.get("sources", [])
        if item.get("source_id")
    }


def normalize_record(
    record: dict[str, Any],
    source_kind: str,
    source_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    text = str(record.get("text") or "").strip()
    if not text:
        raise ValueError("Missing text field in source record.")

    source_id = record.get("source_id")
    if not source_id and source_kind == "research_papers":
        source_id = "filtered_cryoem_papers"
    if not source_id and source_kind == "workflow_examples":
        source_id = "local_workflow_artifacts"

    source_meta = source_lookup.get(str(source_id), {}) if source_id else {}
    bucket = record.get("bucket") or source_meta.get("bucket") or source_kind
    title = (
        record.get("title")
        or record.get("title_from_filename")
        or record.get("source_file")
        or record.get("paper_id")
        or record.get("doc_id")
        or record.get("id")
    )
    group_id = (
        record.get("paper_id")
        or record.get("doc_id")
        or record.get("source_file")
        or record.get("source_id")
        or record.get("id")
    )

    normalized = {
        "id": f"{source_kind}::{record.get('id') or group_id}",
        "group_id": str(group_id),
        "bucket": str(bucket),
        "source_kind": source_kind,
        "source_id": source_id,
        "priority": record.get("priority") or source_meta.get("priority"),
        "type": record.get("type") or source_meta.get("type"),
        "title": title,
        "source_file": record.get("source_file"),
        "source_path": record.get("source_path"),
        "source_url": record.get("source_url"),
        "relative_path": record.get("relative_path"),
        "char_count": record.get("char_count"),
        "chunk_index": record.get("chunk_index"),
        "text": text,
    }
    return normalized


def load_and_normalize_records(
    path_str: str | None,
    source_kind: str,
    source_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not path_str:
        return []
    path = Path(path_str).resolve()
    raw_records = load_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for raw in raw_records:
        try:
            normalized.append(normalize_record(raw, source_kind, source_lookup))
        except Exception:
            continue
    return normalized


def apply_bucket_repeat(
    records: list[dict[str, Any]],
    repeat_overrides: dict[str, int],
    max_records_per_bucket: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["bucket"]].append(record)

    rng = random.Random(seed)
    expanded: list[dict[str, Any]] = []
    for bucket, bucket_records in grouped.items():
        repeat = max(1, int(repeat_overrides.get(bucket, 1)))
        replicated: list[dict[str, Any]] = []
        for rep in range(repeat):
            for record in bucket_records:
                new_record = dict(record)
                if rep > 0:
                    new_record["id"] = f"{record['id']}::rep::{rep}"
                    new_record["repeat_index"] = rep
                else:
                    new_record["repeat_index"] = 0
                replicated.append(new_record)
        if max_records_per_bucket is not None and len(replicated) > max_records_per_bucket:
            rng.shuffle(replicated)
            replicated = replicated[:max_records_per_bucket]
        expanded.extend(replicated)
    rng.shuffle(expanded)
    return expanded


def split_by_group(records: list[dict[str, Any]], eval_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)

    group_keys = list(grouped.keys())
    random.Random(seed).shuffle(group_keys)
    eval_group_count = max(1, math.ceil(len(group_keys) * eval_ratio)) if len(group_keys) > 1 else 0
    eval_keys = set(group_keys[:eval_group_count])

    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    for key, bucket in grouped.items():
        if key in eval_keys:
            eval_records.extend(bucket)
        else:
            train_records.extend(bucket)

    if not train_records:
        raise ValueError("Train split is empty after group-based split.")
    return train_records, eval_records


def summarize_by_bucket(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["bucket"]] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_manifest = load_json(Path(args.source_manifest_json).resolve())
    bucket_weights = bucket_weight_map(source_manifest)
    source_lookup = source_info_map(source_manifest)
    repeat_overrides = load_json(Path(args.bucket_repeat_overrides_json).resolve()) if args.bucket_repeat_overrides_json else {}

    records: list[dict[str, Any]] = []
    records.extend(load_and_normalize_records(args.paper_chunks_jsonl, "research_papers", source_lookup))
    records.extend(load_and_normalize_records(args.docs_chunks_jsonl, "procedural_docs", source_lookup))
    records.extend(load_and_normalize_records(args.workflow_chunks_jsonl, "workflow_examples", source_lookup))
    records.extend(load_and_normalize_records(args.dataset_context_jsonl, "dataset_context", source_lookup))

    if not records:
        raise SystemExit("No input records were loaded.")

    expanded_records = apply_bucket_repeat(
        records=records,
        repeat_overrides=repeat_overrides,
        max_records_per_bucket=args.max_records_per_bucket,
        seed=args.seed,
    )
    train_records, eval_records = split_by_group(expanded_records, args.eval_ratio, args.seed)

    full_path = output_root / "task_relevant_cpt_full.jsonl"
    train_path = output_root / "task_relevant_cpt_train.jsonl"
    eval_path = output_root / "task_relevant_cpt_eval.jsonl"
    manifest_path = output_root / "manifest.json"

    write_jsonl(full_path, expanded_records)
    write_jsonl(train_path, train_records)
    write_jsonl(eval_path, eval_records)

    manifest = {
        "source_manifest_json": str(Path(args.source_manifest_json).resolve()),
        "paper_chunks_jsonl": str(Path(args.paper_chunks_jsonl).resolve()) if args.paper_chunks_jsonl else None,
        "docs_chunks_jsonl": str(Path(args.docs_chunks_jsonl).resolve()) if args.docs_chunks_jsonl else None,
        "workflow_chunks_jsonl": str(Path(args.workflow_chunks_jsonl).resolve()) if args.workflow_chunks_jsonl else None,
        "dataset_context_jsonl": str(Path(args.dataset_context_jsonl).resolve()) if args.dataset_context_jsonl else None,
        "bucket_weights": bucket_weights,
        "bucket_repeat_overrides": repeat_overrides,
        "max_records_per_bucket": args.max_records_per_bucket,
        "input_record_count": len(records),
        "expanded_record_count": len(expanded_records),
        "train_record_count": len(train_records),
        "eval_record_count": len(eval_records),
        "full_bucket_counts": summarize_by_bucket(expanded_records),
        "train_bucket_counts": summarize_by_bucket(train_records),
        "eval_bucket_counts": summarize_by_bucket(eval_records),
        "task_relevant_cpt_full_jsonl": str(full_path),
        "task_relevant_cpt_train_jsonl": str(train_path),
        "task_relevant_cpt_eval_jsonl": str(eval_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
