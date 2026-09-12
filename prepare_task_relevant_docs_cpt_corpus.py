#!/usr/bin/env python3
"""Prepare CPT-ready corpus from task-relevant local docs/tutorial snapshots."""

from __future__ import annotations

import argparse
import json
import re
from html import unescape
from pathlib import Path
from typing import Any


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".json"}
SECTION_HEADING_RE = re.compile(
    r"^(abstract|introduction|background|methods?|results?|discussion|conclusion|faq|troubleshooting|overview|tutorial)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-root",
        default="/home/lisongyang/cryoagent/task_relevant_docs_raw",
        help="Root directory containing locally saved docs/tutorial snapshots.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/lisongyang/cryoagent/task_relevant_docs_cpt",
        help="Directory for cleaned text and JSONL outputs.",
    )
    parser.add_argument(
        "--source-manifest-json",
        default="/home/lisongyang/cryoagent/task_relevant_cpt_sources.json",
        help="Optional manifest describing source_id -> bucket/priority mappings.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional file limit for smoke tests.",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=2200,
        help="Approximate character budget for each text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=250,
        help="Approximate overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=300,
        help="Minimum characters required to keep a chunk.",
    )
    parser.add_argument(
        "--write-cleaned-text",
        action="store_true",
        help="Also write one cleaned .txt file per source file.",
    )
    return parser.parse_args()


def load_source_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = data.get("sources", [])
    return {str(item.get("source_id")): item for item in sources if item.get("source_id")}


def discover_source_files(root: Path, max_files: int | None) -> list[Path]:
    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    if max_files is not None:
        return paths[:max_files]
    return paths


def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = line.replace("\u200b", "")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.+?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    return text


def strip_html(text: str) -> str:
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.I)
    text = re.sub(r"</div\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(text)


def flatten_json_payload(payload: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(payload, dict):
        title = payload.get("title") or payload.get("name") or payload.get("page_title")
        source_url = payload.get("source_url") or payload.get("url") or payload.get("homepage")
        text = payload.get("text") or payload.get("content") or payload.get("body") or payload.get("markdown")
        if text:
            return str(text), {"embedded_title": title, "embedded_source_url": source_url}
        text_parts: list[str] = []
        for key in ["summary", "description", "abstract", "notes"]:
            value = payload.get(key)
            if value:
                text_parts.append(f"{key}: {value}")
        if not text_parts:
            text_parts.append(json.dumps(payload, ensure_ascii=False, indent=2))
        return "\n\n".join(text_parts), {"embedded_title": title, "embedded_source_url": source_url}
    if isinstance(payload, list):
        return "\n\n".join(str(item) for item in payload), {}
    return str(payload), {}


def extract_raw_text(path: Path) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".md", ".markdown"}:
        return strip_markdown(raw), {}
    if suffix in {".html", ".htm"}:
        return strip_html(raw), {}
    if suffix == ".json":
        payload = json.loads(raw)
        return flatten_json_payload(payload)
    return raw, {}


def clean_text(raw_text: str) -> tuple[str, dict[str, Any]]:
    raw_text = raw_text.replace("\x00", " ")
    raw_text = re.sub(r"[ \t]+", " ", raw_text)
    lines = [normalize_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if SECTION_HEADING_RE.match(line) and current:
            paragraphs.append(" ".join(current).strip())
            current = [line]
            continue
        current.append(line)
        if line.endswith((".", ":", ";")) and len(" ".join(current)) > 280:
            paragraphs.append(" ".join(current).strip())
            current = []
    if current:
        paragraphs.append(" ".join(current).strip())

    cleaned_paragraphs: list[str] = []
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < 35:
            continue
        cleaned_paragraphs.append(para)

    cleaned_text = "\n\n".join(cleaned_paragraphs)
    stats = {
        "raw_line_count": len(lines),
        "cleaned_paragraph_count": len(cleaned_paragraphs),
        "cleaned_char_count": len(cleaned_text),
    }
    return cleaned_text, stats


def chunk_text(
    cleaned_text: str,
    chunk_chars: int,
    chunk_overlap_chars: int,
    min_chunk_chars: int,
) -> list[str]:
    paragraphs = [part.strip() for part in cleaned_text.split("\n\n") if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []

    for para in paragraphs:
        candidate = "\n\n".join(current + [para]).strip()
        if current and len(candidate) > chunk_chars:
            chunk = "\n\n".join(current).strip()
            if len(chunk) >= min_chunk_chars:
                chunks.append(chunk)

            overlap_parts: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                extra = len(prev) + (2 if overlap_parts else 0)
                if overlap_len + extra > chunk_overlap_chars:
                    break
                overlap_parts.insert(0, prev)
                overlap_len += extra
            current = overlap_parts[:]

        if not current and len(para) > chunk_chars:
            start = 0
            while start < len(para):
                end = min(len(para), start + chunk_chars)
                piece = para[start:end].strip()
                if len(piece) >= min_chunk_chars:
                    chunks.append(piece)
                if end >= len(para):
                    break
                start = max(start + 1, end - chunk_overlap_chars)
            continue

        current.append(para)

    if current:
        chunk = "\n\n".join(current).strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)

    return chunks


def infer_source_id(path: Path, docs_root: Path) -> str:
    relative = path.relative_to(docs_root)
    return relative.parts[0] if relative.parts else "unknown_source"


def build_doc_record(
    source_path: Path,
    docs_root: Path,
    source_info: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    relative_path = source_path.relative_to(docs_root)
    source_id = infer_source_id(source_path, docs_root)
    raw_text, embedded_meta = extract_raw_text(source_path)
    cleaned_text, clean_stats = clean_text(raw_text)
    chunks = chunk_text(
        cleaned_text=cleaned_text,
        chunk_chars=args.chunk_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        min_chunk_chars=args.min_chunk_chars,
    )

    title = (
        embedded_meta.get("embedded_title")
        or source_path.stem.replace("_", " ").replace("-", " ").strip()
    )
    source_url = embedded_meta.get("embedded_source_url")
    doc_id = re.sub(r"[^a-z0-9]+", "-", str(relative_path).lower()).strip("-")

    doc_record = {
        "doc_id": doc_id,
        "source_id": source_id,
        "bucket": source_info.get("bucket") if source_info else None,
        "priority": source_info.get("priority") if source_info else None,
        "type": source_info.get("type") if source_info else None,
        "source_file": source_path.name,
        "source_path": str(source_path),
        "relative_path": str(relative_path),
        "title": title,
        "source_url": source_url,
        "chunk_count": len(chunks),
        **clean_stats,
    }

    chunk_records: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        chunk_records.append(
            {
                "id": f"{doc_id}::chunk::{idx:04d}",
                "doc_id": doc_id,
                "source_id": source_id,
                "bucket": source_info.get("bucket") if source_info else None,
                "priority": source_info.get("priority") if source_info else None,
                "type": source_info.get("type") if source_info else None,
                "source_file": source_path.name,
                "relative_path": str(relative_path),
                "title": title,
                "source_url": source_url,
                "chunk_index": idx,
                "char_count": len(chunk),
                "text": chunk,
            }
        )

    return doc_record, chunk_records, cleaned_text


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    docs_root = Path(args.docs_root).resolve()
    output_root = Path(args.output_root).resolve()
    source_manifest_json = Path(args.source_manifest_json).resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    cleaned_text_root = output_root / "cleaned_txt"
    if args.write_cleaned_text:
        cleaned_text_root.mkdir(parents=True, exist_ok=True)

    source_manifest = load_source_manifest(source_manifest_json)
    source_files = discover_source_files(docs_root, args.max_files)
    if not source_files:
        raise SystemExit(f"No supported source files found under {docs_root}")

    doc_records: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []

    for source_path in source_files:
        try:
            source_id = infer_source_id(source_path, docs_root)
            source_info = source_manifest.get(source_id, {})
            doc_record, per_doc_chunks, cleaned_text = build_doc_record(source_path, docs_root, source_info, args)
            doc_records.append(doc_record)
            chunk_records.extend(per_doc_chunks)
            if args.write_cleaned_text:
                txt_path = cleaned_text_root / f"{doc_record['doc_id']}.txt"
                txt_path.write_text(cleaned_text, encoding="utf-8")
        except Exception as exc:
            failed_records.append(
                {
                    "source_file": source_path.name,
                    "source_path": str(source_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    docs_jsonl = output_root / "docs_metadata.jsonl"
    chunks_jsonl = output_root / "docs_cpt_chunks.jsonl"
    failed_jsonl = output_root / "failed_docs.jsonl"
    manifest_json = output_root / "manifest.json"

    write_jsonl(docs_jsonl, doc_records)
    write_jsonl(chunks_jsonl, chunk_records)
    write_jsonl(failed_jsonl, failed_records)

    manifest = {
        "docs_root": str(docs_root),
        "source_manifest_json": str(source_manifest_json) if source_manifest_json.exists() else None,
        "doc_count": len(doc_records),
        "failed_doc_count": len(failed_records),
        "chunk_count": len(chunk_records),
        "docs_metadata_jsonl": str(docs_jsonl),
        "docs_cpt_chunks_jsonl": str(chunks_jsonl),
        "failed_docs_jsonl": str(failed_jsonl),
        "write_cleaned_text": args.write_cleaned_text,
        "cleaned_text_dir": str(cleaned_text_root) if args.write_cleaned_text else None,
        "chunk_chars": args.chunk_chars,
        "chunk_overlap_chars": args.chunk_overlap_chars,
        "min_chunk_chars": args.min_chunk_chars,
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
