#!/usr/bin/env python3
"""Parse EMDB XML metadata into compact JSON."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


EMD_ID_RE = re.compile(r"EMD[-_]?(\d+)", re.I)


def text_or_none(node: ET.Element | None) -> str | None:
    if node is None or node.text is None:
        return None
    text = node.text.strip()
    return text or None


def find_text(root: ET.Element, path: str) -> str | None:
    return text_or_none(root.find(path))


def findall_text(root: ET.Element, path: str) -> list[str]:
    values = []
    for node in root.findall(path):
        value = text_or_none(node)
        if value is not None:
            values.append(value)
    return values


def maybe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def infer_emd_id(xml_path: Path, root: ET.Element) -> str | None:
    emdb_id = root.attrib.get("emdb_id")
    if emdb_id:
        return emdb_id
    match = EMD_ID_RE.search(xml_path.stem)
    if match:
        return f"EMD-{match.group(1)}"
    return None


def parse_emdb_xml(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()

    reference = {
        "reference_id": infer_emd_id(xml_path, root),
        "xml_version": root.attrib.get("version"),
        "admin": {
            "title": find_text(root, "./admin/title"),
            "status_code": find_text(root, "./admin/current_status/code"),
            "processing_site": find_text(root, "./admin/current_status/processing_site"),
            "deposition_date": find_text(root, "./admin/key_dates/deposition"),
            "map_release_date": find_text(root, "./admin/key_dates/map_release"),
            "authors": findall_text(root, "./admin/authors_list/author"),
            "keywords": find_text(root, "./admin/keywords"),
        },
        "citation": {
            "title": find_text(
                root,
                "./crossreferences/citation_list/primary_citation/journal_citation/title",
            ),
            "journal": find_text(
                root,
                "./crossreferences/citation_list/primary_citation/journal_citation/journal",
            ),
            "year": find_text(
                root,
                "./crossreferences/citation_list/primary_citation/journal_citation/year",
            ),
            "doi": next(
                (
                    text_or_none(node)
                    for node in root.findall(
                        "./crossreferences/citation_list/primary_citation/journal_citation/external_references"
                    )
                    if node.attrib.get("type") == "DOI"
                ),
                None,
            ),
            "pubmed": next(
                (
                    text_or_none(node)
                    for node in root.findall(
                        "./crossreferences/citation_list/primary_citation/journal_citation/external_references"
                    )
                    if node.attrib.get("type") == "PUBMED"
                ),
                None,
            ),
            "related_empiar_links": findall_text(
                root,
                "./crossreferences/auxiliary_link_list/auxiliary_link/link",
            ),
        },
        "sample": {
            "name": find_text(root, "./sample/name"),
            "supramolecule_name": find_text(root, "./sample/supramolecule_list/sample_supramolecule/name"),
            "symmetry": find_text(root, "./sample/supramolecule_list/sample_supramolecule/oligomeric_state"),
            "molecular_weight_mda": maybe_float(
                find_text(root, "./sample/supramolecule_list/sample_supramolecule/molecular_weight/theoretical")
            ),
            "macromolecules": [
                {
                    "name": find_text(node, "./name"),
                    "organism": find_text(node, "./natural_source/organism"),
                    "copies": maybe_float(find_text(node, "./number_of_copies")),
                    "oligomeric_state": find_text(node, "./oligomeric_state"),
                }
                for node in root.findall("./sample/macromolecule_list/protein_or_peptide")
            ],
        },
        "structure_determination": {
            "method": find_text(root, "./structure_determination_list/structure_determination/method"),
            "aggregation_state": find_text(
                root,
                "./structure_determination_list/structure_determination/aggregation_state",
            ),
            "microscopy": {
                "microscope": find_text(
                    root,
                    "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/microscope",
                ),
                "detector": find_text(
                    root,
                    "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/image_recording_list/image_recording/film_or_detector_model",
                ),
                "acceleration_voltage_kv": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/acceleration_voltage",
                    )
                ),
                "nominal_cs_mm": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/nominal_cs",
                    )
                ),
                "defocus_min_um": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/nominal_defocus_min",
                    )
                ),
                "defocus_max_um": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/nominal_defocus_max",
                    )
                ),
                "nominal_magnification": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/nominal_magnification",
                    )
                ),
                "calibrated_magnification": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/calibrated_magnification",
                    )
                ),
                "number_real_images": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/image_recording_list/image_recording/number_real_images",
                    )
                ),
                "dose_e_per_a2": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/microscopy_list/single_particle_microscopy/image_recording_list/image_recording/average_electron_dose_per_image",
                    )
                ),
            },
            "final_reconstruction": {
                "resolution_a": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/singleparticle_processing/final_reconstruction/resolution",
                    )
                ),
                "resolution_method": find_text(
                    root,
                    "./structure_determination_list/structure_determination/singleparticle_processing/final_reconstruction/resolution_method",
                ),
                "software": findall_text(
                    root,
                    "./structure_determination_list/structure_determination/singleparticle_processing/final_reconstruction/software_list/software/name",
                ),
                "number_images_used": maybe_float(
                    find_text(
                        root,
                        "./structure_determination_list/structure_determination/singleparticle_processing/final_reconstruction/number_images_used",
                    )
                ),
            },
        },
        "map": {
            "file": find_text(root, "./map/file"),
            "format": root.find("./map").attrib.get("format") if root.find("./map") is not None else None,
            "dimensions": {
                "col": maybe_float(find_text(root, "./map/dimensions/col")),
                "row": maybe_float(find_text(root, "./map/dimensions/row")),
                "sec": maybe_float(find_text(root, "./map/dimensions/sec")),
            },
            "pixel_spacing_a": {
                "x": maybe_float(find_text(root, "./map/pixel_spacing/x")),
                "y": maybe_float(find_text(root, "./map/pixel_spacing/y")),
                "z": maybe_float(find_text(root, "./map/pixel_spacing/z")),
            },
            "contour_level": maybe_float(find_text(root, "./map/contour_list/contour/level")),
            "annotation_details": find_text(root, "./map/annotation_details"),
        },
    }

    reference["summary"] = {
        "reference_id": reference["reference_id"],
        "title": reference["admin"]["title"],
        "sample_name": reference["sample"]["name"],
        "symmetry": reference["sample"]["symmetry"],
        "resolution_a": reference["structure_determination"]["final_reconstruction"]["resolution_a"],
        "software": reference["structure_determination"]["final_reconstruction"]["software"],
        "microscope": reference["structure_determination"]["microscopy"]["microscope"],
        "detector": reference["structure_determination"]["microscopy"]["detector"],
        "pixel_spacing_a": reference["map"]["pixel_spacing_a"]["x"],
    }

    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path", help="EMDB XML path")
    args = parser.parse_args()
    record = parse_emdb_xml(Path(args.xml_path))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
