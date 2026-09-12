#!/usr/bin/env python3
"""Prepare CPT-ready paper corpus from a flat directory of PDFs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pypdf. Install it with `pip install pypdf`."
    ) from exc

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None


FILENAME_META_RE = re.compile(
    r"^(?P<title>.+?)_(?P<source>arXiv|bioRxiv|medRxiv|doi|PMID|PMC)_(?P<year>\d{4})$",
    re.I,
)
ARXIV_HEADER_RE = re.compile(r"^arXiv:\S+(?:\s+\[[^\]]+\])?\s+\d{1,2}\s+\w+\s+\d{4}$")
LATEXIT_RE = re.compile(r"<latexit\b.*?</latexit>", re.I | re.S)
SECTION_HEADING_RE = re.compile(
    r"^(abstract|introduction|background|methods?|materials? and methods?|results?|discussion|conclusion|references|appendix)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        default="/hdd1/huangjianhua/agent/data/CPT/paper",
        help="Directory containing source PDFs.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/lisongyang/cryoagent/cpt_paper_outputs",
        help="Directory for cleaned text and JSONL outputs.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
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
        default=400,
        help="Minimum characters required to keep a chunk.",
    )
    parser.add_argument(
        "--write-cleaned-text",
        action="store_true",
        help="Also write one cleaned .txt file per PDF.",
    )
    return parser.parse_args()


def extract_pdf_text(pdf_path: Path) -> str:
    if pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                chunks = [(page.extract_text() or "") for page in pdf.pages]
            text = "\n".join(chunks)
            if text.strip():
                return text
        except Exception:
            pass

    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def normalize_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = line.replace("\u200b", "")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def parse_filename_metadata(pdf_path: Path) -> dict[str, Any]:
    stem = pdf_path.stem
    match = FILENAME_META_RE.match(stem)
    if match:
        title = match.group("title").replace("_", " ").strip()
        source = match.group("source")
        year = int(match.group("year"))
    else:
        title = stem.replace("_", " ").strip()
        source = None
        year = None

    paper_id = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return {
        "paper_id": paper_id,
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "title_from_filename": title,
        "source_hint": source,
        "year_hint": year,
    }


def clean_pdf_text(raw_text: str) -> tuple[str, dict[str, Any]]:
    raw_text = raw_text.replace("\x00", " ")
    raw_text = LATEXIT_RE.sub(" ", raw_text)
    raw_text = raw_text.replace("Graphical TOC Entry", " ")
    raw_text = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", raw_text)
    raw_text = re.sub(r"[ \t]+", " ", raw_text)

    lines = [normalize_line(line) for line in raw_text.splitlines()]
    lines = [line for line in lines if line]

    filtered: list[str] = []
    dropped_headers = 0
    dropped_footers = 0
    for line in lines:
        if ARXIV_HEADER_RE.match(line):
            dropped_headers += 1
            continue
        if re.fullmatch(r"\d+", line):
            dropped_footers += 1
            continue
        filtered.append(line)

    paragraphs: list[str] = []
    current: list[str] = []
    for line in filtered:
        if SECTION_HEADING_RE.match(line) and current:
            paragraphs.append(" ".join(current).strip())
            current = [line]
            continue
        current.append(line)
        if line.endswith((".", ":", ";")) and len(" ".join(current)) > 300:
            paragraphs.append(" ".join(current).strip())
            current = []
    if current:
        paragraphs.append(" ".join(current).strip())

    cleaned_paragraphs: list[str] = []
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < 40:
            continue
        cleaned_paragraphs.append(para)

    cleaned_text = "\n\n".join(cleaned_paragraphs)
    stats = {
        "raw_line_count": len(lines),
        "cleaned_paragraph_count": len(cleaned_paragraphs),
        "dropped_header_lines": dropped_headers,
        "dropped_footer_lines": dropped_footers,
        "cleaned_char_count": len(cleaned_text),
    }
    return cleaned_text, stats


def extract_abstract(cleaned_text: str) -> str | None:
    text = cleaned_text.strip()
    lower = text.lower()
    idx = lower.find("abstract")
    if idx < 0:
        return None

    abstract_text = text[idx:]
    split_match = re.search(
        r"\n\n(?:introduction|background|methods?|materials? and methods?)\b",
        abstract_text,
        flags=re.I,
    )
    if split_match:
        abstract_text = abstract_text[: split_match.start()]

    abstract_text = re.sub(r"^abstract[:\s]*", "", abstract_text, flags=re.I).strip()
    abstract_text = re.sub(r"\s+", " ", abstract_text)
    if len(abstract_text) < 80:
        return None
    return abstract_text[:3000]


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
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        sep_len = 2 if current else 0
        if current and current_len + sep_len + para_len > chunk_chars:
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
            current_len = len("\n\n".join(current)) if current else 0

        if not current and para_len > chunk_chars:
            start = 0
            while start < para_len:
                end = min(para_len, start + chunk_chars)
                piece = para[start:end].strip()
                if len(piece) >= min_chunk_chars:
                    chunks.append(piece)
                if end >= para_len:
                    break
                start = max(start + 1, end - chunk_overlap_chars)
            current = []
            current_len = 0
            continue

        current.append(para)
        current_len = len("\n\n".join(current))

    if current:
        chunk = "\n\n".join(current).strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)

    return chunks


def build_paper_record(pdf_path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    metadata = parse_filename_metadata(pdf_path)
    raw_text = extract_pdf_text(pdf_path)
    cleaned_text, clean_stats = clean_pdf_text(raw_text)
    abstract = extract_abstract(cleaned_text)
    chunks = chunk_text(
        cleaned_text=cleaned_text,
        chunk_chars=args.chunk_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
        min_chunk_chars=args.min_chunk_chars,
    )

    paper_record = {
        **metadata,
        **clean_stats,
        "page_count": len(PdfReader(str(pdf_path)).pages),
        "abstract": abstract,
        "chunk_count": len(chunks),
    }

    chunk_records: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        chunk_records.append(
            {
                "id": f"{metadata['paper_id']}::chunk::{idx:04d}",
                "paper_id": metadata["paper_id"],
                "source_file": metadata["source_file"],
                "title_from_filename": metadata["title_from_filename"],
                "source_hint": metadata["source_hint"],
                "year_hint": metadata["year_hint"],
                "chunk_index": idx,
                "char_count": len(chunk),
                "text": chunk,
            }
        )

    return paper_record, chunk_records, cleaned_text


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    paper_root = Path(args.paper_root).resolve()
    output_root = Path(args.output_root).resolve()
    cleaned_text_root = output_root / "cleaned_txt"
    output_root.mkdir(parents=True, exist_ok=True)
    if args.write_cleaned_text:
        cleaned_text_root.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(paper_root.glob("*.pdf"))
    if args.max_papers is not None:
        pdf_paths = pdf_paths[: args.max_papers]
    if not pdf_paths:
        raise SystemExit(f"No PDFs found under {paper_root}")

    paper_records: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        try:
            paper_record, per_paper_chunks, cleaned_text = build_paper_record(pdf_path, args)
            paper_records.append(paper_record)
            chunk_records.extend(per_paper_chunks)
            if args.write_cleaned_text:
                txt_path = cleaned_text_root / f"{paper_record['paper_id']}.txt"
                txt_path.write_text(cleaned_text, encoding="utf-8")
        except Exception as exc:
            failed_records.append(
                {
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    papers_jsonl = output_root / "papers_metadata.jsonl"
    chunks_jsonl = output_root / "papers_cpt_chunks.jsonl"
    failed_jsonl = output_root / "failed_papers.jsonl"
    manifest_json = output_root / "manifest.json"

    write_jsonl(papers_jsonl, paper_records)
    write_jsonl(chunks_jsonl, chunk_records)
    write_jsonl(failed_jsonl, failed_records)

    manifest = {
        "paper_root": str(paper_root),
        "paper_count": len(paper_records),
        "failed_paper_count": len(failed_records),
        "chunk_count": len(chunk_records),
        "papers_metadata_jsonl": str(papers_jsonl),
        "papers_cpt_chunks_jsonl": str(chunks_jsonl),
        "failed_papers_jsonl": str(failed_jsonl),
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
