#!/usr/bin/env python3
"""Collect curated community repo docs and public training materials."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


COMMUNITY_URLS: dict[str, dict[str, str]] = {
    "github_relion_repo": {
        "readme.md": "https://raw.githubusercontent.com/3dem/relion/master/README.md",
    },
    "github_relion_documents": {
        "readme.md": "https://raw.githubusercontent.com/3dem/relion-documents/master/README.md",
    },
    "github_topaz": {
        "readme.md": "https://raw.githubusercontent.com/tbepler/topaz/master/README.md",
    },
    "github_nextpyp": {
        "readme.rst": "https://raw.githubusercontent.com/nextpyp/pyp/master/README.rst",
    },
    "scipion_tutorials": {
        "index.html": "https://scipion-em.github.io/docs/release-3.0.0/index.html",
        "user_documentation.html": "https://scipion-em.github.io/docs/release-3.0.0/docs/user/user-documentation.html",
        "troubleshooting.html": "https://scipion-em.github.io/docs/release-3.0.0/docs/user/troubleshooting.html",
        "how_to_install.html": "https://scipion-em.github.io/docs/release-3.0.0/docs/scipion-modes/how-to-install.html",
        "facilities_overview.html": "https://scipion-em.github.io/docs/release-3.0.0/docs/facilities/facilities-overview.html",
        "spa_facility.html": "https://scipion-em.github.io/docs/release-3.0.0/docs/facilities/SPA.html"
    },
    "teamtomo_site": {
        "index.html": "https://teamtomo.org/",
        "algorithms.html": "https://teamtomo.org/site/algorithms/",
        "io_packages.html": "https://teamtomo.org/site/io_packages/",
        "primitives.html": "https://teamtomo.org/site/primitives/"
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="/home/lisongyang/cryoagent/task_relevant_docs_raw",
        help="Directory where collected community docs will be saved.",
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
        headers={"User-Agent": "Mozilla/5.0 (compatible; CryoAgentCommunityCollector/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"collections": {}, "failures": {}}
    for source_id, file_map in COMMUNITY_URLS.items():
        collected: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        source_dir = output_root / source_id
        for filename, url in file_map.items():
            try:
                text = fetch_text(url, args.timeout)
                save_text(source_dir / filename, text)
                collected.append(
                    {
                        "title": filename,
                        "source_url": url,
                        "output_path": str((source_dir / filename).resolve()),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "title": filename,
                        "source_url": url,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        manifest["collections"][source_id] = collected
        manifest["failures"][source_id] = failures

    manifest["summary"] = {key: len(value) for key, value in manifest["collections"].items()}
    manifest["failure_summary"] = {key: len(value) for key, value in manifest["failures"].items()}

    manifest_path = output_root / "community_collection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "summary": manifest["summary"],
                "failure_summary": manifest["failure_summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
