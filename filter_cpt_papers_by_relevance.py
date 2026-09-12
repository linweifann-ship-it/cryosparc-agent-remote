#!/usr/bin/env python3
"""Filter CPT paper corpus by cryo-EM workflow relevance."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class WeightedPattern:
    label: str
    pattern: re.Pattern[str]
    weight: int


POSITIVE_PATTERNS: tuple[WeightedPattern, ...] = (
    WeightedPattern("cryosparc", re.compile(r"\bcryo\s*sparc\b", re.I), 8),
    WeightedPattern("cryosparc_live", re.compile(r"\bcryo\s*sparc\s+live\b", re.I), 8),
    WeightedPattern("relion", re.compile(r"\brelion(?:[-\s]?\d+(?:\.\d+)*)?\b", re.I), 7),
    WeightedPattern("cryodrgn", re.compile(r"\bcryo\s*drgn\b|\bcryodrgn\b", re.I), 7),
    WeightedPattern("single_particle", re.compile(r"\bsingle[-\s]?particle\b", re.I), 5),
    WeightedPattern("subtomogram", re.compile(r"\bsubtomogram\b|\bsubtomogram averaging\b", re.I), 5),
    WeightedPattern("electron_cryomicroscopy", re.compile(r"\belectron cryo-?microscopy\b", re.I), 5),
    WeightedPattern("cryo_em", re.compile(r"\bcryo-?em\b", re.I), 4),
    WeightedPattern("particle_picking", re.compile(r"\bparticle pick(?:ing|er)?\b", re.I), 5),
    WeightedPattern("template_picker", re.compile(r"\btemplate pick(?:ing|er)?\b", re.I), 5),
    WeightedPattern("motion_correction", re.compile(r"\bmotion correction\b|\bbeam[-\s]induced motion\b", re.I), 4),
    WeightedPattern("ctf", re.compile(r"\bctf\b|\bcontrast transfer function\b", re.I), 4),
    WeightedPattern("2d_classification", re.compile(r"\b2d classification\b", re.I), 4),
    WeightedPattern("3d_classification", re.compile(r"\b3d classification\b", re.I), 4),
    WeightedPattern("ab_initio", re.compile(r"\bab[-\s]initio reconstruction\b", re.I), 4),
    WeightedPattern("heterogeneous_refinement", re.compile(r"\bheterogeneous refinement\b", re.I), 4),
    WeightedPattern("homogeneous_refinement", re.compile(r"\bhomogeneous refinement\b", re.I), 4),
    WeightedPattern("non_uniform_refinement", re.compile(r"\bnon[-\s]uniform refinement\b", re.I), 4),
    WeightedPattern("helical", re.compile(r"\bhelical reconstruction\b|\bhelix refinement\b", re.I), 4),
    WeightedPattern("micrograph", re.compile(r"\bmicrograph(?:s)?\b", re.I), 3),
    WeightedPattern("template_matching", re.compile(r"\btemplate matching\b", re.I), 3),
    WeightedPattern("empiar", re.compile(r"\bempiar[-\s]?\d+\b|\bempiar\b", re.I), 3),
    WeightedPattern("emdb", re.compile(r"\bemdb\b|\bemd[-_ ]?\d+\b", re.I), 3),
)


NEGATIVE_PATTERNS: tuple[WeightedPattern, ...] = (
    WeightedPattern("quantum_chemistry", re.compile(r"\bquantum chemistry\b", re.I), -8),
    WeightedPattern("molecular_dynamics", re.compile(r"\bmolecular dynamics\b", re.I), -6),
    WeightedPattern("temporary_anions", re.compile(r"\btemporary anions?\b", re.I), -8),
    WeightedPattern("density_functional", re.compile(r"\bdensity functional theory\b|\bdft\b", re.I), -6),
    WeightedPattern("spectroscopy", re.compile(r"\bspectroscop(?:y|ic)\b", re.I), -4),
    WeightedPattern("battery", re.compile(r"\bbatter(?:y|ies)\b", re.I), -5),
    WeightedPattern("photocatalysis", re.compile(r"\bphotocatal(?:ysis|yst)\b", re.I), -5),
    WeightedPattern("organic_synthesis", re.compile(r"\borganic synthesis\b", re.I), -5),
    WeightedPattern("polymer", re.compile(r"\bpolymer(?:s)?\b", re.I), -4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--chunks-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--min-paper-score", type=int, default=6)
    parser.add_argument("--min-positive-match-count", type=int, default=1)
    parser.add_argument(
        "--keep-all-chunks-for-kept-papers",
        action="store_true",
        help="Keep every chunk from papers that pass the filter. Default keeps all chunks as well.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=20,
        help="How many kept/dropped paper previews to store in the summary.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def normalize_text(*parts: object) -> str:
    return "\n".join(str(part or "") for part in parts if part).strip()


def collect_matches(text: str, patterns: tuple[WeightedPattern, ...]) -> dict[str, int]:
    hits: dict[str, int] = {}
    for pattern in patterns:
        count = len(pattern.pattern.findall(text))
        if count:
            hits[pattern.label] = count
    return hits


def score_matches(matches: dict[str, int], patterns: tuple[WeightedPattern, ...]) -> int:
    weights = {pattern.label: pattern.weight for pattern in patterns}
    return sum(weights[label] * count for label, count in matches.items())


def summarize_hit_labels(matches: dict[str, int]) -> list[str]:
    return sorted(matches, key=lambda key: (-matches[key], key))


def build_paper_reports(metadata_records: list[dict], chunk_records: list[dict]) -> list[dict]:
    chunks_by_paper: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunk_records:
        paper_id = chunk.get("paper_id") or chunk.get("source_file") or chunk.get("id")
        chunks_by_paper[paper_id].append(chunk)

    reports: list[dict] = []
    for metadata in metadata_records:
        paper_id = metadata.get("paper_id") or metadata.get("source_file")
        paper_chunks = chunks_by_paper.get(paper_id, [])
        chunk_text_preview = "\n".join(str(chunk.get("text", ""))[:1500] for chunk in paper_chunks[:3])
        paper_text = normalize_text(
            metadata.get("title_from_filename"),
            metadata.get("abstract"),
            chunk_text_preview,
        )

        positive_matches = collect_matches(paper_text, POSITIVE_PATTERNS)
        negative_matches = collect_matches(paper_text, NEGATIVE_PATTERNS)
        positive_score = score_matches(positive_matches, POSITIVE_PATTERNS)
        negative_score = score_matches(negative_matches, NEGATIVE_PATTERNS)
        total_score = positive_score + negative_score

        report = {
            "paper_id": paper_id,
            "source_file": metadata.get("source_file"),
            "title_from_filename": metadata.get("title_from_filename"),
            "abstract": metadata.get("abstract"),
            "chunk_count": len(paper_chunks),
            "positive_score": positive_score,
            "negative_score": negative_score,
            "paper_score": total_score,
            "positive_match_count": sum(positive_matches.values()),
            "negative_match_count": sum(negative_matches.values()),
            "positive_hit_labels": summarize_hit_labels(positive_matches),
            "negative_hit_labels": summarize_hit_labels(negative_matches),
            "positive_matches": positive_matches,
            "negative_matches": negative_matches,
        }
        reports.append(report)
    return sorted(reports, key=lambda item: (-item["paper_score"], item["paper_id"] or ""))


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata_jsonl).resolve()
    chunks_path = Path(args.chunks_jsonl).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_records = load_jsonl(metadata_path)
    chunk_records = load_jsonl(chunks_path)
    paper_reports = build_paper_reports(metadata_records, chunk_records)

    kept_paper_ids = {
        report["paper_id"]
        for report in paper_reports
        if report["paper_score"] >= args.min_paper_score
        and report["positive_match_count"] >= args.min_positive_match_count
    }

    filtered_metadata = [record for record in metadata_records if (record.get("paper_id") or record.get("source_file")) in kept_paper_ids]
    filtered_chunks = [record for record in chunk_records if (record.get("paper_id") or record.get("source_file")) in kept_paper_ids]
    dropped_paper_ids = {
        record.get("paper_id") or record.get("source_file")
        for record in metadata_records
        if (record.get("paper_id") or record.get("source_file")) not in kept_paper_ids
    }

    reports_path = output_root / "paper_relevance_report.jsonl"
    kept_metadata_path = output_root / "filtered_papers_metadata.jsonl"
    kept_chunks_path = output_root / "filtered_papers_cpt_chunks.jsonl"
    manifest_path = output_root / "manifest.json"

    write_jsonl(reports_path, paper_reports)
    write_jsonl(kept_metadata_path, filtered_metadata)
    write_jsonl(kept_chunks_path, filtered_chunks)

    kept_preview = [
        {
            "paper_id": report["paper_id"],
            "paper_score": report["paper_score"],
            "positive_hit_labels": report["positive_hit_labels"][:8],
            "negative_hit_labels": report["negative_hit_labels"][:4],
            "title_from_filename": report["title_from_filename"],
        }
        for report in paper_reports
        if report["paper_id"] in kept_paper_ids
    ][: args.preview_limit]
    dropped_preview = [
        {
            "paper_id": report["paper_id"],
            "paper_score": report["paper_score"],
            "positive_hit_labels": report["positive_hit_labels"][:8],
            "negative_hit_labels": report["negative_hit_labels"][:4],
            "title_from_filename": report["title_from_filename"],
        }
        for report in paper_reports
        if report["paper_id"] in dropped_paper_ids
    ][: args.preview_limit]

    manifest = {
        "metadata_jsonl": str(metadata_path),
        "chunks_jsonl": str(chunks_path),
        "output_root": str(output_root),
        "min_paper_score": args.min_paper_score,
        "min_positive_match_count": args.min_positive_match_count,
        "input_paper_count": len(metadata_records),
        "kept_paper_count": len(filtered_metadata),
        "dropped_paper_count": len(metadata_records) - len(filtered_metadata),
        "input_chunk_count": len(chunk_records),
        "kept_chunk_count": len(filtered_chunks),
        "dropped_chunk_count": len(chunk_records) - len(filtered_chunks),
        "paper_relevance_report_jsonl": str(reports_path),
        "filtered_papers_metadata_jsonl": str(kept_metadata_path),
        "filtered_papers_cpt_chunks_jsonl": str(kept_chunks_path),
        "kept_paper_preview": kept_preview,
        "dropped_paper_preview": dropped_preview,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
