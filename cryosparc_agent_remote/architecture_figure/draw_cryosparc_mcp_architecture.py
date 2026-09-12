#!/usr/bin/env python3
"""Render the CryoSPARC MCP architecture figure.

The editable figure source is the SVG file in this directory. This script
keeps the SVG as the canonical drawing artifact and exports vector PDF plus a
high-resolution PNG preview for manuscript checks.

Usage:
    python draw_cryosparc_mcp_architecture.py
    python draw_cryosparc_mcp_architecture.py --variant simple
    python draw_cryosparc_mcp_architecture.py --variant full --no-png
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


FIGURES = {
    "full": {
        "stem": "cryosparc_mcp_architecture",
        "description": "Full two-panel architecture figure.",
    },
    "simple": {
        "stem": "cryosparc_mcp_architecture_简版",
        "description": "Simplified two-panel architecture figure.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the CryoSPARC MCP architecture SVG to PDF and PNG."
    )
    parser.add_argument(
        "--variant",
        choices=("full", "simple", "both"),
        default="both",
        help="Which figure variant to export. Default: both.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the SVG source and receiving exported files.",
    )
    parser.add_argument(
        "--png-scale",
        type=float,
        default=2.0,
        help="PNG preview scale relative to the SVG canvas. Default: 2.0.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip vector PDF export.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG preview export.",
    )
    return parser.parse_args()


def require_cairosvg():
    try:
        import cairosvg  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: cairosvg. Install with `python3 -m pip install cairosvg`."
        ) from exc
    return cairosvg


def svg_canvas_size(svg_path: Path) -> tuple[int, int]:
    root = ET.parse(svg_path).getroot()
    width = root.attrib.get("width")
    height = root.attrib.get("height")
    if width is None or height is None:
        raise ValueError(f"SVG is missing width/height attributes: {svg_path}")
    return int(float(width)), int(float(height))


def export_variant(name: str, output_dir: Path, png_scale: float, make_pdf: bool, make_png: bool) -> None:
    cairosvg = require_cairosvg()
    stem = FIGURES[name]["stem"]
    svg_path = output_dir / f"{stem}.svg"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}_preview.png"

    if not svg_path.exists():
        raise FileNotFoundError(f"SVG source not found: {svg_path}")

    # Parse once before export so malformed SVG fails early.
    width, height = svg_canvas_size(svg_path)

    if make_pdf:
        cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
        print(f"Wrote {pdf_path}")

    if make_png:
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=round(width * png_scale),
            output_height=round(height * png_scale),
        )
        print(f"Wrote {png_path}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    variants = ("full", "simple") if args.variant == "both" else (args.variant,)

    if args.no_pdf and args.no_png:
        raise SystemExit("Nothing to export: both --no-pdf and --no-png were set.")

    for variant in variants:
        export_variant(
            name=variant,
            output_dir=output_dir,
            png_scale=args.png_scale,
            make_pdf=not args.no_pdf,
            make_png=not args.no_png,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
