#!/usr/bin/env python3
"""Prepare SFT data from cryoSPARC event log PDFs.

This script extracts structured metadata and timeline events from exported
cryoSPARC event log PDFs such as:

    P3_J4748_event_log_2026_06_06T12_41_51.644Z.pdf

It writes:
1. Parsed records as JSONL for downstream analysis/custom labeling.
2. SFT-ready chat records as JSONL for supervised fine-tuning.

The parser is intentionally heuristic and tuned for cryoSPARC event log PDFs
exported with a layout similar to the sample above.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


PIPE = "|"
TABLE_BORDER_RE = re.compile(r"^\+-[-+]+\+$")
EVENT_RE = re.compile(
    r">\s*\n\[(?P<timestamp>[^\]]+)\]\n(?P<body>.*?)(?=\n>\s*\n\[|\Z)",
    re.S,
)
KEY_VALUE_RE = re.compile(r"^\s*([^:]+?)\s*:\s*(.+?)\s*$")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class ParsedSectionRow:
    section: str
    cells: list[str]


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
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def cleaned_lines(text: str) -> list[str]:
    return [normalize_line(line) for line in text.splitlines() if normalize_line(line)]


def is_border(line: str) -> bool:
    return bool(TABLE_BORDER_RE.match(line))


def parse_row_block(block_lines: list[str]) -> list[str]:
    flat_tokens: list[str] = []
    for line in block_lines:
        if line == PIPE:
            flat_tokens.append(line)
            continue
        if PIPE in line:
            pieces = line.split(PIPE)
            for idx, piece in enumerate(pieces):
                if idx > 0:
                    flat_tokens.append(PIPE)
                cleaned_piece = piece.strip()
                if cleaned_piece:
                    flat_tokens.append(cleaned_piece)
            continue
        flat_tokens.append(line)

    cells: list[str] = []
    started = False
    current: list[str] = []

    for token in flat_tokens:
        if token == PIPE:
            if started:
                cells.append(" ".join(current).strip())
                current = []
            else:
                started = True
        else:
            current.append(token)

    return cells


def extract_bordered_rows(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    i = 0
    while i < len(lines):
        if is_border(lines[i]):
            j = i + 1
            block: list[str] = []
            while j < len(lines) and not is_border(lines[j]):
                block.append(lines[j])
                j += 1
            if block:
                row = parse_row_block(block)
                if row:
                    rows.append(row)
            i = j
        else:
            i += 1
    return rows


def slice_between(lines: list[str], start: str, end: str | None) -> list[str]:
    try:
        start_idx = lines.index(start) + 1
    except ValueError:
        return []

    if end is None:
        end_idx = len(lines)
    else:
        try:
            end_idx = lines.index(end, start_idx)
        except ValueError:
            end_idx = len(lines)
    return lines[start_idx:end_idx]


def next_non_border_text(lines: list[str], start_idx: int) -> str | None:
    for idx in range(start_idx, len(lines)):
        if lines[idx] != PIPE and not is_border(lines[idx]):
            return lines[idx]
    return None


def find_line(lines: list[str], value: str, start_idx: int = 0) -> int:
    for idx in range(start_idx, len(lines)):
        if lines[idx] == value:
            return idx
    return -1


def find_outputs_start(lines: list[str], start_idx: int) -> int:
    for idx in range(start_idx, len(lines)):
        if lines[idx] != "Outputs":
            continue
        next_text = next_non_border_text(lines, idx + 1)
        if next_text and next_text.lower() == next_text and "." not in next_text:
            return idx
    return -1


def rows_to_mapping(rows: list[list[str]], key_index: int = 0, value_index: int = 1) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in rows:
        if len(row) <= value_index:
            continue
        key = row[key_index].strip()
        value = row[value_index].strip()
        if not key or key in {"DETAIL", "VALUE", "PARAMETER", "DEFAULT", "SPEC", "ADVANCED"}:
            continue
        mapping[key] = value
    return mapping


def parse_parameter_section(lines: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    current_section = "General"
    i = 0

    while i < len(lines):
        line = lines[i]
        if line == "Outputs":
            i += 1
            continue

        if line not in {PIPE} and not is_border(line) and i + 1 < len(lines) and is_border(lines[i + 1]):
            current_section = line
            i += 1
            continue

        if is_border(line):
            j = i + 1
            block: list[str] = []
            while j < len(lines) and not is_border(lines[j]):
                block.append(lines[j])
                j += 1
            if block:
                row = parse_row_block(block)
                if not row and len(block) == 1 and block[0] not in {PIPE, "Outputs"}:
                    current_section = block[0]
                elif len(row) >= 2 and row[0] != "PARAMETER":
                    entry = {
                        "value": row[1],
                        "default": "X" in row[2:] if len(row) > 2 else False,
                        "spec": row[3] == "X" if len(row) > 3 else False,
                        "advanced": row[4] == "X" if len(row) > 4 else False,
                    }
                    result.setdefault(current_section, {})[row[0]] = entry
            i = j
            continue

        i += 1

    return result


def parse_io_section(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_group: str | None = None
    pending_key: str | None = None
    i = 0

    while i < len(lines):
        line = lines[i]

        if (
            line not in {PIPE}
            and not is_border(line)
            and "." not in line
            and line.lower() == line
            and not line.startswith("[")
        ):
            current_group = line
            result.setdefault(current_group, {"dataset": None, "fields": {}})
            pending_key = None
            i += 1
            continue

        if current_group and result[current_group]["dataset"] is None and "." in line and not line.startswith("+"):
            result[current_group]["dataset"] = line
            i += 1
            continue

        if is_border(line):
            j = i + 1
            block: list[str] = []
            while j < len(lines) and not is_border(lines[j]):
                block.append(lines[j])
                j += 1
            if (
                block
                and len(block) == 2
                and block[0].lower() == block[0]
                and "." not in block[0]
                and "." in block[1]
            ):
                current_group = block[0]
                result.setdefault(current_group, {"dataset": None, "fields": {}})
                result[current_group]["dataset"] = block[1]
                pending_key = None
                i = j
                continue

            if block and len(block) == 1 and block[0].lower() == block[0] and "." not in block[0]:
                current_group = block[0]
                result.setdefault(current_group, {"dataset": None, "fields": {}})
                pending_key = None
                i = j
                continue

            if current_group and block:
                row = parse_row_block(block)
                if len(row) == 1:
                    item = row[0]
                    if "." in item:
                        if pending_key is None:
                            if result[current_group]["dataset"] is None:
                                result[current_group]["dataset"] = item
                        else:
                            existing = result[current_group]["fields"].get(pending_key)
                            if existing is None:
                                result[current_group]["fields"][pending_key] = item
                            elif isinstance(existing, list):
                                existing.append(item)
                            else:
                                result[current_group]["fields"][pending_key] = [existing, item]
                    else:
                        pending_key = item
            i = j
            continue

        i += 1

    return result


def parse_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx, match in enumerate(EVENT_RE.finditer(text), start=1):
        body = match.group("body").strip()
        cpu_match = re.search(r"\[CPU:\s*([^\]]+)\]", body)
        avail_match = re.search(r"\[Avail:\s*([^\]]+)\]", body)
        cleaned_body = re.sub(r"\[CPU:\s*[^\]]+\]", "", body)
        cleaned_body = re.sub(r"\[Avail:\s*[^\]]+\]", "", cleaned_body)
        message = " ".join(normalize_line(line) for line in cleaned_body.splitlines() if normalize_line(line))
        message = re.sub(r"\s+", " ", message).strip()
        events.append(
            {
                "index": idx,
                "timestamp": match.group("timestamp").strip(),
                "cpu": cpu_match.group(1).strip() if cpu_match else None,
                "avail": avail_match.group(1).strip() if avail_match else None,
                "message": message,
            }
        )
    return events


def parse_header(lines: list[str], pdf_path: Path) -> dict[str, Any]:
    header = {
        "source_file": pdf_path.name,
        "source_stem": pdf_path.stem,
    }

    if lines:
        first = lines[0]
        match = re.match(r"^(P\d+)\s+(J\d+):\s+(.+)$", first)
        if match:
            header["project_id"] = match.group(1)
            header["job_id"] = match.group(2)
            header["job_type_from_title"] = match.group(3)

    for line in lines[:20]:
        if line.startswith("CryoSPARC Running Version:"):
            header["cryosparc_version"] = line.split(":", 1)[1].strip()
            break

    return header


def coerce_number(value: str) -> int | float | str:
    raw = value.replace(",", "").strip()
    if not raw:
        return value
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return value


def extract_derived_metrics(text: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    patterns = {
        "loaded_micrographs": r"Loaded info for ([\d,]+) micrographs",
        "loaded_particles": r"Loaded info for ([\d,]+) particles",
        "included_particles": r"([\d,]+) particles included",
        "excluded_particles": r"([\d,]+) particles excluded",
        "extracted_particles": r"Completed\. Extracted ([\d,]+) particles",
        "total_time_seconds": r"Job complete\. Total time ([\d.]+)s",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[key] = coerce_number(match.group(1))

    thresholds_match = re.search(
        r"Thresholds changed:\s*NCC\s*:\s*([-\d.]+),\s*Power Min:\s*([-\d.]+),\s*Power Max:\s*([-\d.]+)",
        text.replace("\n", " "),
    )
    if thresholds_match:
        metrics["thresholds"] = {
            "ncc": float(thresholds_match.group(1)),
            "power_min": float(thresholds_match.group(2)),
            "power_max": float(thresholds_match.group(3)),
        }

    coordinates = [
        event["message"].replace("Extracted coordinates on ", "")
        for event in events
        if event["message"].startswith("Extracted coordinates on ")
    ]
    if coordinates:
        metrics["coordinate_examples"] = coordinates[:10]
        metrics["coordinate_example_count"] = len(coordinates)

    interesting_events = []
    keywords = (
        "Started",
        "Loaded info",
        "Calibrating",
        "Thresholds changed",
        "included",
        "Completed.",
        "Failed",
        "Error",
        "Warning",
        "Job complete",
    )
    for event in events:
        if any(keyword in event["message"] for keyword in keywords):
            interesting_events.append(
                {
                    "timestamp": event["timestamp"],
                    "message": event["message"],
                }
            )
    if interesting_events:
        metrics["key_events"] = interesting_events[:30]

    return metrics


def summarize_for_sft(parsed: dict[str, Any]) -> dict[str, Any]:
    job_details = parsed.get("job_details", {})
    project_details = parsed.get("project_details", {})
    derived = parsed.get("derived_metrics", {})

    summary = {
        "project": {
            "id": parsed.get("header", {}).get("project_id") or project_details.get("ID"),
            "title": project_details.get("Title"),
        },
        "job": {
            "id": parsed.get("header", {}).get("job_id") or job_details.get("ID"),
            "title": job_details.get("Title"),
            "type": job_details.get("Job type") or parsed.get("header", {}).get("job_type_from_title"),
            "status": job_details.get("Status"),
            "cryosparc_version": parsed.get("header", {}).get("cryosparc_version"),
            "created_by": job_details.get("Created by user"),
            "timestamps": {
                key: value
                for key, value in job_details.items()
                if key in {"Created", "Queued", "Launched", "Started", "Waiting", "Completed", "Failed"}
            },
        },
        "inputs": parsed.get("inputs", {}),
        "parameters": {
            section: {name: meta["value"] for name, meta in params.items()}
            for section, params in parsed.get("parameters", {}).items()
        },
        "outputs": parsed.get("outputs", {}),
        "derived_metrics": derived,
    }

    return summary


def build_sft_messages(parsed: dict[str, Any]) -> list[dict[str, str]]:
    target = summarize_for_sft(parsed)
    user_payload = {
        "project": target["project"],
        "job": target["job"],
        "inputs": target["inputs"],
        "parameters": target["parameters"],
        "outputs": target["outputs"],
        "derived_metrics": target["derived_metrics"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a cryoSPARC workflow assistant. Read structured cryoSPARC "
                "job data and return valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Normalize the following cryoSPARC job record into valid JSON. "
                "Preserve the structured information under project, job, inputs, "
                "parameters, outputs, and derived_metrics.\n\n"
                f"Record:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(target, ensure_ascii=False, indent=2),
        },
    ]


def parse_pdf_record(pdf_path: Path) -> dict[str, Any]:
    text = extract_pdf_text(pdf_path)
    lines = cleaned_lines(text)

    project_start = find_line(lines, "Project Details")
    job_start = find_line(lines, "Job Details", project_start + 1)
    inputs_start = find_line(lines, "Inputs", job_start + 1)
    parameters_start = find_line(lines, "Parameters", inputs_start + 1)
    outputs_start = find_outputs_start(lines, parameters_start + 1)
    target_start = find_line(lines, "Target", outputs_start + 1)

    project_lines = lines[project_start + 1 : job_start] if project_start >= 0 and job_start >= 0 else []
    job_lines = lines[job_start + 1 : inputs_start] if job_start >= 0 and inputs_start >= 0 else []
    input_lines = lines[inputs_start + 1 : parameters_start] if inputs_start >= 0 and parameters_start >= 0 else []
    parameter_lines = (
        lines[parameters_start + 1 : outputs_start]
        if parameters_start >= 0 and outputs_start >= 0
        else []
    )
    output_lines = lines[outputs_start + 1 : target_start] if outputs_start >= 0 and target_start >= 0 else []

    header = parse_header(lines, pdf_path)
    project_details = rows_to_mapping(extract_bordered_rows(project_lines))
    job_details = rows_to_mapping(extract_bordered_rows(job_lines))
    inputs = parse_io_section(input_lines)
    parameters = parse_parameter_section(parameter_lines)
    outputs = parse_io_section(output_lines)
    events = parse_events(text)
    derived_metrics = extract_derived_metrics(text, events)

    return {
        "header": header,
        "project_details": project_details,
        "job_details": job_details,
        "inputs": inputs,
        "parameters": parameters,
        "outputs": outputs,
        "events": events,
        "derived_metrics": derived_metrics,
        "raw_text": text,
    }


def iter_pdf_paths(inputs: Iterable[str]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        elif any(char in item for char in "*?[]"):
            pdfs.extend(sorted(Path().glob(item)))
        else:
            pdfs.append(path)
    return [pdf for pdf in pdfs if pdf.suffix.lower() == ".pdf" and pdf.exists()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="PDF paths, directories, or globs containing cryoSPARC event log PDFs.",
    )
    parser.add_argument(
        "--parsed-jsonl",
        default="parsed_cryosparc_logs.jsonl",
        help="Output JSONL path for parsed structured records.",
    )
    parser.add_argument(
        "--sft-jsonl",
        default="cryosparc_sft_data.jsonl",
        help="Output JSONL path for SFT chat records.",
    )
    parser.add_argument(
        "--max-log-chars",
        type=int,
        default=20000,
        help="Maximum number of log characters embedded into each SFT prompt.",
    )
    parser.add_argument(
        "--drop-raw-text",
        action="store_true",
        help="Exclude the full extracted PDF text from parsed JSONL records.",
    )
    args = parser.parse_args()

    pdf_paths = iter_pdf_paths(args.inputs)
    if not pdf_paths:
        raise SystemExit("No PDF files found from the provided inputs.")

    parsed_records: list[dict[str, Any]] = []
    sft_records: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        parsed = parse_pdf_record(pdf_path)
        raw_text = parsed["raw_text"]
        if args.drop_raw_text:
            parsed = {**parsed, "raw_text": None}

        parsed_records.append(parsed)
        sft_records.append(
            {
                "id": parsed["header"].get("source_stem", pdf_path.stem),
                "source_file": pdf_path.name,
                "messages": build_sft_messages(parsed),
            }
        )

    write_jsonl(Path(args.parsed_jsonl), parsed_records)
    write_jsonl(Path(args.sft_jsonl), sft_records)

    print(
        json.dumps(
            {
                "pdf_count": len(pdf_paths),
                "parsed_jsonl": str(Path(args.parsed_jsonl).resolve()),
                "sft_jsonl": str(Path(args.sft_jsonl).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
