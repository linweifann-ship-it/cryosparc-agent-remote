#!/usr/bin/env python3
"""Structured extraction helpers for V2 cryoSPARC workflow-state samples."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


RESOURCE_LINE_RE = re.compile(r"^\[\d+\]\s+\[(?P<ts>[^\]]+)\]\s+(?P<msg>.+)$")
WORK_DIR_RE = re.compile(r"Working in directory:\s*(?P<value>.+)$")
LANE_RE = re.compile(r"Running on lane\s+(?P<value>.+)$")
WORKER_RE = re.compile(r"Worker:\s+(?P<value>.+)$")
CPU_RE = re.compile(r"CPU\s*:\s*(?P<value>.+)$")
GPU_RE = re.compile(r"GPU\s*:\s*(?P<value>.+)$")
RAM_RE = re.compile(r"RAM\s*:\s*(?P<value>.+)$")
SSD_RE = re.compile(r"SSD\s*:\s*(?P<value>.+)$")
JOB_COUNT_RE = re.compile(r"process this many micrographs:\s*(?P<count>\d+)", re.I)
IMPORT_COUNT_RE = re.compile(r"Importing\s+(?P<count>\d+)\s+files", re.I)
EXTRACT_COUNT_RE = re.compile(r"Extracted\s+(?P<count>\d+)\s+particles", re.I)


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_iso_seconds(start: str | None, end: str | None) -> float | None:
    start_dt = _parse_iso_datetime(start)
    end_dt = _parse_iso_datetime(end)
    if start_dt is None or end_dt is None:
        return None
    return round((end_dt - start_dt).total_seconds(), 2)




def _normalize_runtime_value(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    if value.startswith('[') and value.endswith(']'):
        inner = value[1:-1].strip()
        if not inner:
            return []
        parts = [part.strip() for part in inner.split(',')]
        result: list[Any] = []
        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                result.append(part)
        return result
    return value


def _summarize_input_groups(job_document: dict[str, Any]) -> list[dict[str, Any]]:
    groups = []
    for group in job_document.get('input_slot_groups') or []:
        count_max = group.get('count_max')
        if isinstance(count_max, float) and count_max == float('inf'):
            count_max = None
        groups.append({
            'name': group.get('name'),
            'title': group.get('title'),
            'count_min': group.get('count_min'),
            'count_max': count_max,
            'slot_names': [slot.get('name') for slot in group.get('slots') or [] if slot.get('name')],
            'connected_job_uids': sorted({conn.get('job_uid') for conn in group.get('connections') or [] if conn.get('job_uid')}),
        })
    return groups

def _parse_structured_log_lines(text: str) -> list[tuple[str | None, str]]:
    lines: list[tuple[str | None, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = RESOURCE_LINE_RE.match(line)
        if match:
            lines.append((match.group("ts"), match.group("msg")))
        else:
            lines.append((None, line))
    return lines


def _safe_scalar_stats(latest_summary_stats: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in latest_summary_stats.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
    return safe


def _image_kind(filename: str) -> str:
    lowered = filename.lower()
    for marker, kind in [
        ("power_spectrum", "power_spectrum"),
        ("diagnostic_plot", "diagnostic_plot"),
        ("raw_data", "raw_data"),
        ("raw_images", "raw_images"),
        ("lowpass_filtered", "lowpass_filtered"),
        ("ncc_calibration", "ncc_calibration"),
        ("imported_volume", "imported_volume"),
        ("class_avg", "class_average"),
    ]:
        if marker in lowered:
            return kind
    return "other"


def _summarize_image_refs(job_document: dict[str, Any]) -> dict[str, Any]:
    event_images: list[dict[str, Any]] = []
    for entry in job_document.get("event_log_entries") or []:
        for image in entry.get("images") or []:
            if image.get("filename"):
                event_images.append(
                    {
                        "filename": image.get("filename"),
                        "path": image.get("path"),
                        "bytes": image.get("bytes"),
                        "kind": _image_kind(image.get("filename", "")),
                    }
                )

    kind_counts: dict[str, int] = {}
    for image in event_images:
        kind = image["kind"]
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    return {
        "ui_tile_images": job_document.get("ui_tile_images") or [],
        "output_group_images": job_document.get("output_group_images") or {},
        "event_images": event_images[:24],
        "event_image_count": len(event_images),
        "event_image_kind_counts": kind_counts,
    }


def _extract_runtime(structured_log_text: str) -> tuple[dict[str, Any], list[str], list[str]]:
    runtime: dict[str, Any] = {}
    evidence: list[str] = []
    warnings: list[str] = []

    for _ts, line in _parse_structured_log_lines(structured_log_text):
        if len(evidence) < 10 and (
            "Job ready to run" in line
            or "Working in directory" in line
            or "Running on lane" in line
            or "process this many micrographs" in line
            or "Importing " in line
            or "Extracted " in line
            or "Completed" in line
        ):
            evidence.append(line[:240])

        if "warning" in line.lower():
            warnings.append(line[:240])

        for pattern, key in [
            (WORK_DIR_RE, "work_dir"),
            (LANE_RE, "lane"),
            (WORKER_RE, "worker_hostname"),
            (CPU_RE, "allocated_cpu"),
            (GPU_RE, "allocated_gpu"),
            (RAM_RE, "allocated_ram"),
            (SSD_RE, "allocated_ssd"),
        ]:
            match = pattern.search(line)
            if match:
                runtime[key] = _normalize_runtime_value(match.group("value").strip())

        for pattern, key in [
            (JOB_COUNT_RE, "micrograph_count_logged"),
            (IMPORT_COUNT_RE, "import_file_count_logged"),
            (EXTRACT_COUNT_RE, "extracted_particle_count_logged"),
        ]:
            match = pattern.search(line)
            if match:
                runtime[key] = int(match.group("count"))

    return runtime, evidence[:10], warnings[:6]


def _summarize_outputs(job_document: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summarized_groups: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}

    for group in job_document.get("output_result_groups") or []:
        group_name = group.get("name") or group.get("type") or "output"
        num_items = group.get("num_items")
        field_names = [item.get("name") for item in group.get("contains") or [] if item.get("name")]
        latest_summary_stats = group.get("latest_summary_stats") or {}
        scalar_stats = _safe_scalar_stats(latest_summary_stats)

        summarized_groups.append(
            {
                "name": group_name,
                "description": group.get("description"),
                "num_items": num_items,
                "field_names": field_names,
                "scalar_stats": scalar_stats,
                "summary_stat_keys": sorted(latest_summary_stats.keys()),
            }
        )

        if isinstance(num_items, int):
            lowered = group_name.lower()
            if "particle" in lowered and "selected" in lowered:
                metrics["selected_particle_count"] = num_items
            elif "particle" in lowered and ("excluded" in lowered or "rejected" in lowered):
                metrics["rejected_particle_count"] = num_items
            elif "particle" in lowered:
                metrics.setdefault("particle_count", num_items)
            elif "micrograph" in lowered or "exposure" in lowered:
                metrics.setdefault("micrograph_count", num_items)
            elif "class" in lowered:
                metrics.setdefault("class_count", num_items)
            elif "volume" in lowered:
                metrics.setdefault("volume_count", num_items)
            elif "mask" in lowered:
                metrics.setdefault("mask_count", num_items)

    return summarized_groups, metrics


def _flatten_parameters(job_document: dict[str, Any]) -> dict[str, Any]:
    params = job_document.get("params_spec") or {}
    flattened: dict[str, Any] = {}
    for key, spec in params.items():
        if isinstance(spec, dict):
            flattened[key] = spec.get("value")
        else:
            flattened[key] = spec
    return flattened


def build_last_node_info(job_document: dict[str, Any]) -> dict[str, Any]:
    structured_log_text = (
        job_document.get("structured_event_log_text")
        or job_document.get("worker_log")
        or job_document.get("event_log_text")
        or ""
    )

    runtime, evidence_text, warning_lines = _extract_runtime(structured_log_text)
    output_groups, output_metrics = _summarize_outputs(job_document)
    image_refs = _summarize_image_refs(job_document)
    parameters_flat = _flatten_parameters(job_document)

    runtime_seconds = parse_iso_seconds(job_document.get("started_at"), job_document.get("completed_at"))
    if runtime_seconds is not None:
        output_metrics["total_runtime_seconds"] = runtime_seconds

    return {
        "job_type": job_document.get("job_type"),
        "job_uid": job_document.get("job_uid"),
        "job_title": job_document.get("title"),
        "project_uid": job_document.get("project_uid"),
        "status": job_document.get("status"),
        "timestamps": {
            "created_at": job_document.get("created_at"),
            "started_at": job_document.get("started_at"),
            "completed_at": job_document.get("completed_at"),
        },
        "inputs": {
            "groups": _summarize_input_groups(job_document),
        },
        "parameters": parameters_flat,
        "outputs": {
            "groups": output_groups,
        },
        "metrics": output_metrics,
        "runtime": runtime,
        "evidence_text": evidence_text,
        "warning_lines": warning_lines,
        "image_refs": image_refs,
    }
