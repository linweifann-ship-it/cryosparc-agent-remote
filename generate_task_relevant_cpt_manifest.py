#!/usr/bin/env python3
"""Generate a curated manifest of task-relevant CPT sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        default="/home/lisongyang/cryoagent/task_relevant_cpt_sources.json",
        help="Path to the generated manifest JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "1.0",
        "purpose": "Curated source manifest for a task-relevant CPT corpus focused on cryo-EM workflow assistance.",
        "recommended_buckets": [
            {
                "bucket_id": "procedural_docs",
                "weight": 0.35,
                "notes": "Highest-priority operational documentation aligned with workflow decisions.",
            },
            {
                "bucket_id": "workflow_examples",
                "weight": 0.30,
                "notes": "Local workflow traces and structured execution artifacts from the real task distribution.",
            },
            {
                "bucket_id": "dataset_context",
                "weight": 0.15,
                "notes": "Archive metadata and dataset descriptions useful for state/context understanding.",
            },
            {
                "bucket_id": "research_papers",
                "weight": 0.20,
                "notes": "Method papers and highly task-relevant cryo-EM literature.",
            },
        ],
        "sources": [
            {
                "source_id": "cryosparc_guide_core",
                "priority": "P0",
                "bucket": "procedural_docs",
                "name": "CryoSPARC Guide",
                "homepage": "https://guide.cryosparc.com/",
                "type": "official_docs",
                "recommended_sections": [
                    "Jobs",
                    "Creating and Running Jobs",
                    "Inspecting Job Data",
                    "Workflows",
                    "Managing Jobs",
                    "Interactive Jobs",
                    "All Job Types in CryoSPARC",
                    "Automated Workflows",
                    "Data Processing Tutorials",
                ],
                "why_useful": "Directly aligned with cryoSPARC job semantics, workflow structure, and task vocabulary.",
                "acquisition_hint": "Mirror or export the relevant guide pages into text or markdown snapshots with URL provenance.",
            },
            {
                "source_id": "cryosparc_live_docs",
                "priority": "P0",
                "bucket": "procedural_docs",
                "name": "CryoSPARC Live Documentation",
                "homepage": "https://guide.cryosparc.com/",
                "type": "official_docs",
                "recommended_sections": [
                    "New Live Session: Start to Finish Guide",
                    "Live Jobs and Session-Level Functions",
                    "Performance Metrics",
                    "FAQs and Troubleshooting",
                ],
                "why_useful": "Covers real-time monitoring, session state, and operational decision points close to agent behavior.",
                "acquisition_hint": "Collect page snapshots plus any available tutorial transcripts or notes.",
            },
            {
                "source_id": "relion_docs",
                "priority": "P1",
                "bucket": "procedural_docs",
                "name": "RELION Documentation",
                "homepage": "https://relion.readthedocs.io/en/latest/",
                "type": "official_docs",
                "recommended_sections": [
                    "Single particle tutorial",
                    "Subtomogram tutorial",
                    "On-the-fly processing",
                    "Reference pages",
                ],
                "why_useful": "Adds general cryo-EM workflow knowledge and supports future multi-tool expansion.",
                "acquisition_hint": "Export tutorial and reference pages as markdown or cleaned text with section metadata.",
            },
            {
                "source_id": "cryodrgn_docs",
                "priority": "P1",
                "bucket": "procedural_docs",
                "name": "cryoDRGN Documentation and Repository",
                "homepage": "https://github.com/ml-struct-bio/cryodrgn",
                "type": "repository_docs",
                "recommended_sections": [
                    "README",
                    "Installation",
                    "Usage examples",
                    "Command reference",
                ],
                "why_useful": "Adds reconstruction-tool concepts relevant to future tool generalization.",
                "acquisition_hint": "Collect README, docs pages, example workflows, and command help text.",
            },
            {
                "source_id": "empiar_tutorials",
                "priority": "P1.5",
                "bucket": "dataset_context",
                "name": "EMPIAR Archive, Talks, and Tutorials",
                "homepage": "https://www.ebi.ac.uk/empiar/",
                "type": "archive_docs",
                "recommended_sections": [
                    "Talks and Tutorials",
                    "Quick tour",
                    "FAQ",
                    "About EMPIAR",
                ],
                "why_useful": "Improves dataset-level understanding, archive conventions, and EMPIAR/EMDB context mapping.",
                "acquisition_hint": "Capture tutorial pages and archive metadata descriptions with source URLs.",
            },
            {
                "source_id": "local_workflow_artifacts",
                "priority": "P0.5",
                "bucket": "workflow_examples",
                "name": "Local workflow and log artifacts",
                "homepage": "local",
                "type": "internal_data",
                "local_paths": [
                    "/hdd1/huangjianhua/agent/data/workflow",
                    "/hdd1/huangjianhua/agent/data/Log",
                    "/hdd1/huangjianhua/agent/data/experiment",
                    "/ssd1/linweifan/cryosparc_agent/reports",
                ],
                "recommended_sections": [
                    "workflow json",
                    "structured log summaries",
                    "MCP execution traces",
                    "live dry-run and live-run reports",
                ],
                "why_useful": "Best match to the actual downstream task and deployment environment.",
                "acquisition_hint": "Convert to normalized text-plus-metadata records for CPT bucket mixing.",
            },
            {
                "source_id": "filtered_cryoem_papers",
                "priority": "P2",
                "bucket": "research_papers",
                "name": "Filtered cryo-EM method papers",
                "homepage": "local",
                "type": "papers",
                "local_paths": [
                    "/hdd1/huangjianhua/agent/data/CPT/paper",
                    "/ssd1/lisongyang/data/cryoagent_cpt_papers",
                ],
                "recommended_sections": [
                    "method papers about cryoSPARC/RELION/cryoDRGN",
                    "parameter sensitivity discussions",
                    "workflow or failure-analysis text",
                ],
                "why_useful": "Can provide additional method and parameter priors once irrelevant papers are removed.",
                "acquisition_hint": "Use the relevance filter script before adding paper chunks to the CPT mix.",
            },
        ],
    }

    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_path), "source_count": len(manifest["sources"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
