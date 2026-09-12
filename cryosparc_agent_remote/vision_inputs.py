"""Build compact, class-id-labelled visual inputs from CryoSPARC outputs."""
import base64
import io
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cryosparc_client import cryosparc_client


def build_class_average_visual_context(
    project_uid: str,
    job_uid: str,
    max_classes: int = 50,
    tile_size: int = 192,
    columns: int = 5,
) -> dict[str, Any]:
    """Return a labelled contact sheet and per-class metadata for vision APIs."""
    cs = cryosparc_client()
    project = cs.find_project(project_uid)
    job = project.find_job(job_uid)
    dataset = job.load_output("class_averages")
    paths = list(dataset["blob/path"])
    indices = list(dataset["blob/idx"])
    resolutions = list(dataset.get("blob/res_A", []))
    if not paths:
        raise ValueError(f"Job {job_uid} has no class average images.")

    image_path = str(paths[0])
    filename = Path(image_path).name
    _, stack = job.download_mrc(filename)
    stack = np.asarray(stack)
    if stack.ndim == 2:
        stack = stack[None, ...]
    count = min(len(paths), len(stack), max_classes)
    selected_ids = list(range(count))
    rows = math.ceil(count / columns)
    label_height = 26
    sheet = Image.new(
        "RGB", (columns * tile_size, rows * (tile_size + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for class_id in selected_ids:
        image = normalize_class_image(stack[class_id], tile_size)
        x = (class_id % columns) * tile_size
        y = (class_id // columns) * (tile_size + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.rectangle((x, y, x + tile_size - 1, y + label_height - 1), fill="black")
        draw.text((x + 5, y + 6), f"class_id={class_id}", fill="white", font=font)

    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()
    cache_dir = Path(
        os.getenv(
            "CRYOAGENT_VISION_CACHE_DIR",
            "/home/lisongyang/cryoagent/logs/vision_inputs",
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / f"{project_uid}_{job_uid}_class_averages.png"
    cached_path.write_bytes(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    stats = []
    for class_id in selected_ids:
        item = {"class_id": class_id}
        if class_id < len(resolutions):
            item["resolution_A"] = float(resolutions[class_id])
        stats.append(item)
    return {
        "kind": "class_average",
        "source": {"project_uid": project_uid, "job_uid": job_uid},
        "image_count": count,
        "class_ids": selected_ids,
        "class_statistics": stats,
        "contact_sheet": {
            "mime_type": "image/png",
            "local_path": str(cached_path),
            "data_url": f"data:image/png;base64,{encoded}",
            "columns": columns,
            "tile_size": tile_size,
            "label_format": "class_id=<integer>",
        },
    }


def normalize_class_image(array: Any, size: int) -> Image.Image:
    """Convert one MRC image to a contrast-normalized square PNG tile."""
    image = np.asarray(array, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((image - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    tile = Image.fromarray(scaled, mode="L").convert("RGB")
    return tile.resize((size, size), Image.Resampling.BILINEAR)



def build_pick_inspection_visual_context(
    project_uid: str,
    job_uid: str,
    max_micrographs: int = 8,
    max_picks_per_micrograph: int = 400,
    tile_size: int = 512,
    micrograph_root: str | None = None,
) -> dict[str, Any]:
    """Build a micrograph contact sheet with sampled Blob Picker overlays."""
    cs = cryosparc_client()
    project = cs.find_project(project_uid)
    job = project.find_job(job_uid)
    micrographs = job.load_output("micrographs")
    particles = job.load_output("particles")
    mic_paths = list(micrographs["micrograph_blob/path"])
    mic_uids = [int(value) for value in micrographs["uid"]]
    particle_uids = [int(value) for value in particles["location/micrograph_uid"]]
    x_frac = np.asarray(particles["location/center_x_frac"], dtype=np.float32)
    y_frac = np.asarray(particles["location/center_y_frac"], dtype=np.float32)
    ncc = np.asarray(particles["pick_stats/ncc_score"], dtype=np.float32)
    power = np.asarray(particles.get("pick_stats/power", np.zeros(len(ncc))), dtype=np.float32)
    if not mic_paths:
        raise ValueError(f"Job {job_uid} has no micrograph output.")

    by_uid: dict[int, list[int]] = {uid: [] for uid in mic_uids}
    for index, uid in enumerate(particle_uids):
        if uid in by_uid:
            by_uid[uid].append(index)
    # Cover the whole pick-count range instead of selecting only the densest images.
    ranked_mics = sorted(by_uid, key=lambda uid: len(by_uid[uid]))
    sample_count = max(1, min(max_micrographs, len(ranked_mics)))
    sample_indices = np.linspace(0, len(ranked_mics) - 1, sample_count).round().astype(int)
    chosen_uids = [ranked_mics[index] for index in sorted(set(sample_indices))]
    mic_index = {uid: index for index, uid in enumerate(mic_uids)}
    columns = min(3, len(chosen_uids))
    rows = math.ceil(len(chosen_uids) / columns)
    label_height = 30
    sheet = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    panels = []
    for panel_index, uid in enumerate(chosen_uids):
        row_index = panel_index // columns
        col_index = panel_index % columns
        x0 = col_index * tile_size
        y0 = row_index * (tile_size + label_height)
        source_path = str(mic_paths[mic_index[uid]])
        image = normalize_micrograph_image(
            load_micrograph_image(project, source_path, micrograph_root), tile_size
        )
        sheet.paste(Image.fromarray(image, mode="RGB"), (x0, y0 + label_height))
        picks = by_uid[uid]
        if len(picks) > max_picks_per_micrograph:
            positions = np.linspace(0, len(picks) - 1, max_picks_per_micrograph).astype(int)
            picks = [picks[position] for position in positions]
        for particle_index in picks:
            px = int(float(x_frac[particle_index]) * (tile_size - 1))
            py = int(float(y_frac[particle_index]) * (tile_size - 1))
            radius = 4
            draw.ellipse(
                (x0 + px - radius, y0 + label_height + py - radius,
                 x0 + px + radius, y0 + label_height + py + radius),
                outline=(255, 50, 30), width=2,
            )
        draw.rectangle((x0, y0, x0 + tile_size - 1, y0 + label_height - 1), fill="black")
        draw.text(
            (x0 + 5, y0 + 7),
            f"micrograph_id={panel_index} uid={uid} picks={len(by_uid[uid])}",
            fill="white", font=font,
        )
        scores = ncc[by_uid[uid]]
        panels.append({
            "micrograph_id": panel_index,
            "micrograph_uid": uid,
            "source_path": source_path,
            "pick_count": len(by_uid[uid]),
            "overlay_count": len(picks),
            "ncc_score_mean": float(np.mean(scores)) if len(scores) else None,
            "ncc_score_min": float(np.min(scores)) if len(scores) else None,
            "ncc_score_max": float(np.max(scores)) if len(scores) else None,
        })

    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()
    cache_dir = Path(os.getenv("CRYOAGENT_VISION_CACHE_DIR", "/home/lisongyang/cryoagent/logs/vision_inputs"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / f"{project_uid}_{job_uid}_pick_inspection.png"
    cached_path.write_bytes(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    def summary(values: np.ndarray) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"count": 0}
        q = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
        return {
            "count": int(finite.size),
            "min": float(np.min(finite)),
            "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
            "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
            "p99": float(q[6]), "max": float(np.max(finite)),
        }

    micrograph_flags = list(micrographs.get("micrograph_blob/is_background_subtracted", []))
    input_is_denoised = bool(micrograph_flags) and all(bool(value) for value in micrograph_flags)
    return {
        "kind": "pick_inspection",
        "source": {"project_uid": project_uid, "job_uid": job_uid},
        "image_count": len(panels),
        "input_particle_count": int(len(particle_uids)),
        "input_micrograph_count": int(len(mic_uids)),
        "representative_sampling": "pick_count_quantiles",
        "panels": panels,
        "score_statistics": {"ncc_score": summary(ncc), "power": summary(power)},
        "input_is_denoised": input_is_denoised,
        "auto_cluster_supported": input_is_denoised,
        "auto_cluster_warning": (
            None if input_is_denoised else
            "Input micrographs are not marked denoised; prefer explicit NCC/Power thresholds over auto clustering."
        ),
        "retention_guard": {
            "minimum_fraction": 0.5,
            "minimum_expected_particles": int(len(particle_uids) * 0.5),
            "instruction": (
                "Preserve all high NCC/Power picks (bright, particle-like regions). "
                "Use thresholds only to remove the low-score dark/background tail; "
                "do not set a Power upper bound that clips the bright region."
            )
        },
        "overlay": {"marker": "red circles", "max_picks_per_micrograph": max_picks_per_micrograph},
        "contact_sheet": {
            "mime_type": "image/png",
            "local_path": str(cached_path),
            "data_url": f"data:image/png;base64,{encoded}",
            "columns": columns,
            "tile_size": tile_size,
            "label_format": "micrograph_id=<integer> uid=<integer> picks=<integer>",
        },
    }


def load_micrograph_image(project: Any, source_path: str, micrograph_root: str | None) -> np.ndarray:
    """Load a micrograph from a shared root when possible, otherwise via API."""
    candidates = []
    if micrograph_root:
        candidates.append(Path(micrograph_root) / Path(source_path).name)
    if Path(source_path).is_absolute():
        candidates.append(Path(source_path))
    for path in candidates:
        if path.exists():
            from cryosparc.mrc import read
            _, image = read(str(path))
            image = np.asarray(image)
            return image[0] if image.ndim == 3 else image
    _, image = project.download_mrc(source_path)
    image = np.asarray(image)
    return image[0] if image.ndim == 3 else image


def normalize_micrograph_image(array: Any, size: int) -> np.ndarray:
    """Normalize and resize a large micrograph for visual inspection."""
    image = np.asarray(array, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        scaled = np.zeros(image.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1, 99])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((image - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    tile = Image.fromarray(scaled, mode="L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(tile.convert("RGB"))
