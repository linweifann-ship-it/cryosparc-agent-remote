#!/usr/bin/env python3
"""Collect a first batch of official task-relevant docs into local raw snapshots."""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


CRYOSPARC_WANTED_TITLES = {
    "About CryoSPARC™",
    "Jobs",
    "Creating and Running Jobs",
    "Inspecting Job Data",
    "Workflows",
    "Managing Jobs",
    "Interactive Jobs",
    "All Job Types in CryoSPARC",
    "Automated Workflows",
    "Get Started with CryoSPARC: Introductory Tutorial (v4.0+)",
    "Data Processing Tutorials",
    "Tutorial: Patch Motion and Patch CTF",
    "Tutorial: Particle Picking Calibration",
    "Tutorial: Blob Picker Tuner",
    "Tutorial: CTF Refinement",
    "Tutorial: 3D Classification",
    "Tutorial: 3D Variability Analysis (Part One)",
    "Tutorial: 3D Variability Analysis (Part Two)",
    "About CryoSPARC Live",
    "New Live Session: Start to Finish Guide",
    "Live Jobs and Session-Level Functions",
    "Performance Metrics",
    "FAQs and Troubleshooting",
}

CRYODRGN_WANTED_TITLES = {
    "CryoDRGN User Guide",
    "Installation",
    "CryoDRGN EMPIAR-10076 Tutorial",
    "CryoDRGN-AI ab initio reconstruction",
    "Running reconstruction experiments",
    "CryoDRGN-AI ab initio EMPIAR-10076 tutorial",
    "cryodrgn abinit command API",
    "CryoDRGN-ET Subtomogram Analysis",
    "CryoDRGN Conformational Landscape Analysis",
    "Making long trajectories with cryoDRGN graph_traversal",
    "FAQ & Troubleshooting",
    "Helpful Resources",
    "ChimeraX tips",
}

RELION_URLS = {
    "index.html": "https://relion.readthedocs.io/en/latest/",
    "installation.html": "https://relion.readthedocs.io/en/latest/Installation.html",
    "single_particle_tutorial.html": "https://relion.readthedocs.io/en/latest/SPA_tutorial/index.html",
    "subtomogram_tutorial.html": "https://relion.readthedocs.io/en/latest/STA_tutorial/index.html",
    "on_the_fly_processing.html": "https://relion.readthedocs.io/en/latest/Onthefly.html",
    "reference_pages.html": "https://relion.readthedocs.io/en/latest/Reference/index.html",
}

EMPIAR_URLS = {
    "index.html": "https://www.ebi.ac.uk/empiar/",
    "about.html": "https://www.ebi.ac.uk/empiar/about/",
    "faq.html": "https://www.ebi.ac.uk/empiar/faq",
    "tutorials.html": "https://www.ebi.ac.uk/empiar/tutorials/",
    "quick_tour.html": "https://www.ebi.ac.uk/training/online/course/empiar-quick-tour",
}

CRYODRGN_EXTRA_URLS = {
    "github_readme.md": "https://raw.githubusercontent.com/ml-struct-bio/cryodrgn/master/README.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="/home/lisongyang/cryoagent/task_relevant_docs_raw",
        help="Directory where collected docs will be saved.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; CryoAgentDocCollector/1.0)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fetch_gitbook_pages(
    site_index_url: str,
    markdown_base: str,
    wanted_titles: set[str],
    output_dir: Path,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    site_index = json.loads(fetch_text(site_index_url, timeout))
    pages = site_index.get("pages", [])
    collected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    by_title = {page.get("title"): page for page in pages}
    for title in sorted(wanted_titles):
        page = by_title.get(title)
        if not page:
            failures.append({"title": title, "error": "title_not_found_in_site_index"})
            continue
        pathname = page.get("pathname") or "/"
        if pathname == "/" or pathname == "/cryodrgn":
            if markdown_base.endswith("/cryodrgn"):
                url = f"{markdown_base}/cryodrgn-user-guide.md"
            else:
                url = f"{markdown_base}/readme.md"
        else:
            url = f"{markdown_base}{pathname}.md"
        filename = f"{slugify(title)}.md"
        try:
            text = fetch_text(url, timeout)
            save_text(output_dir / filename, text)
            collected.append(
                {
                    "title": title,
                    "source_url": url,
                    "output_path": str((output_dir / filename).resolve()),
                }
            )
        except Exception as exc:
            failures.append({"title": title, "source_url": url, "error": f"{type(exc).__name__}: {exc}"})
    return collected, failures


def fetch_explicit_urls(
    urls: dict[str, str],
    output_dir: Path,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    collected: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for filename, url in urls.items():
        try:
            text = fetch_text(url, timeout)
            save_text(output_dir / filename, text)
            collected.append(
                {
                    "title": filename,
                    "source_url": url,
                    "output_path": str((output_dir / filename).resolve()),
                }
            )
        except Exception as exc:
            failures.append({"title": filename, "source_url": url, "error": f"{type(exc).__name__}: {exc}"})
    return collected, failures


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"collections": {}, "failures": {}}

    cryosparc_dir = output_root / "cryosparc_guide_core"
    cryosparc_collected, cryosparc_failures = fetch_gitbook_pages(
        site_index_url="https://guide.cryosparc.com/~gitbook/site-index",
        markdown_base="https://guide.cryosparc.com",
        wanted_titles=CRYOSPARC_WANTED_TITLES,
        output_dir=cryosparc_dir,
        timeout=args.timeout,
    )
    manifest["collections"]["cryosparc_guide_core"] = cryosparc_collected
    manifest["failures"]["cryosparc_guide_core"] = cryosparc_failures

    cryodrgn_dir = output_root / "cryodrgn_docs"
    cryodrgn_collected, cryodrgn_failures = fetch_gitbook_pages(
        site_index_url="https://ez-lab.gitbook.io/cryodrgn/~gitbook/site-index",
        markdown_base="https://ez-lab.gitbook.io/cryodrgn",
        wanted_titles=CRYODRGN_WANTED_TITLES,
        output_dir=cryodrgn_dir,
        timeout=args.timeout,
    )
    extra_collected, extra_failures = fetch_explicit_urls(CRYODRGN_EXTRA_URLS, cryodrgn_dir, args.timeout)
    manifest["collections"]["cryodrgn_docs"] = cryodrgn_collected + extra_collected
    manifest["failures"]["cryodrgn_docs"] = cryodrgn_failures + extra_failures

    relion_dir = output_root / "relion_docs"
    relion_collected, relion_failures = fetch_explicit_urls(RELION_URLS, relion_dir, args.timeout)
    manifest["collections"]["relion_docs"] = relion_collected
    manifest["failures"]["relion_docs"] = relion_failures

    empiar_dir = output_root / "empiar_tutorials"
    empiar_collected, empiar_failures = fetch_explicit_urls(EMPIAR_URLS, empiar_dir, args.timeout)
    manifest["collections"]["empiar_tutorials"] = empiar_collected
    manifest["failures"]["empiar_tutorials"] = empiar_failures

    manifest["summary"] = {
        source_id: len(items)
        for source_id, items in manifest["collections"].items()
    }
    manifest["failure_summary"] = {
        source_id: len(items)
        for source_id, items in manifest["failures"].items()
    }

    manifest_path = output_root / "collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "manifest": str(manifest_path), "summary": manifest["summary"], "failure_summary": manifest["failure_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
