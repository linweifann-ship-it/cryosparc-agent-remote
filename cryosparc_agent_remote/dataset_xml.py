# Parses EMDB XML metadata into V2 dataset_info fields.
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def load_dataset_info_from_xml(path: str | Path) -> dict[str, Any]:
    """Extract required user-provided protein/microscope metadata from XML."""
    root = ElementTree.parse(path).getroot()
    return {
        "emdb_id": root.attrib.get("emdb_id"),
        "empiar_id": extract_empiar_id(root),
        "abstract": text_at(root, "admin/title"),
        "macromolecules_type": text_at(
            root,
            "sample/macromolecule_list/protein_or_peptide/name",
        ),
        "input_type": "movies",
        "pixel_size_A": number_at(root, ".//pixel_spacing/x"),
        "accelerating_voltage_kv": number_at(root, ".//acceleration_voltage"),
        "spherical_aberration_mm": number_at(root, ".//nominal_cs"),
        "total_exposure_dose_e_per_A2": number_at(
            root,
            ".//average_electron_dose_per_image",
        ),
        "symmetry": (
            text_at(root, ".//sample_supramolecule/oligomeric_state")
            or text_at(root, ".//map/symmetry/space_group")
        ),
    }


def text_at(root: ElementTree.Element, path: str) -> str | None:
    """Return stripped text for a namespaceless ElementTree path."""
    node = root.find(path)
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None


def number_at(root: ElementTree.Element, path: str) -> float | int | None:
    """Return an XML numeric field as int when possible, otherwise float."""
    text = text_at(root, path)
    if text is None:
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def extract_empiar_id(root: ElementTree.Element) -> str | None:
    """Find an EMPIAR ID from auxiliary links."""
    for node in root.findall(".//auxiliary_link/link"):
        if node.text and "EMPIAR-" in node.text:
            return node.text.strip().rsplit("/", 1)[-1]
    return None
