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
            str(Path.cwd() / "logs" / "vision_inputs"),
        )
    ).expanduser()
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


def _vision_cache_dir() -> Path:
    """Resolve the project-local cache directory used for generated PNG evidence."""
    return Path(os.getenv(
        "CRYOAGENT_VISION_CACHE_DIR",
        str(Path.cwd() / "logs" / "vision_inputs"),
    )).expanduser()


def _png_artifact(image: Image.Image, cached_path: Path) -> dict[str, Any]:
    """Persist an image and return the compact artifact contract used by prompts."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()
    cached_path.write_bytes(image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return {
        "mime_type": "image/png",
        "local_path": str(cached_path),
        "data_url": f"data:image/png;base64,{encoded}",
    }


def _summary(values: np.ndarray) -> dict[str, Any]:
    """Return observed distribution statistics without proposing any cutoff."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {
        "count": int(finite.size),
        "nonfinite_count": int(values.size - finite.size),
    }
    if finite.size == 0:
        return result
    q = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    result.update({
        "min": float(np.min(finite)),
        "p01": float(q[0]), "p05": float(q[1]), "p25": float(q[2]),
        "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
        "p99": float(q[6]), "max": float(np.max(finite)),
    })
    return result


PICK_STAT_PERCENTILES = (0.5, 1, 5, 25, 50, 75, 90, 95, 97.5, 99, 99.5)


def _pick_quantiles(values: np.ndarray) -> dict[str, float | None]:
    """Return the complete fixed percentile table used as Model evidence."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {f"P{percentile:g}": None for percentile in PICK_STAT_PERCENTILES}
    quantiles = np.percentile(finite, PICK_STAT_PERCENTILES)
    return {
        f"P{percentile:g}": float(value)
        for percentile, value in zip(PICK_STAT_PERCENTILES, quantiles)
    }


def _power_tail_bins(power_scores: np.ndarray) -> list[dict[str, Any]]:
    """Describe fixed percentile tail intervals without treating any as a cutoff."""
    power_scores = np.asarray(power_scores, dtype=np.float64)
    finite = power_scores[np.isfinite(power_scores)]
    if finite.size == 0:
        return []
    percentile_values = _pick_quantiles(finite)
    definitions = (
        ("P95-P97.5", "P95", "P97.5"),
        ("P97.5-P99", "P97.5", "P99"),
        ("P99-P99.5", "P99", "P99.5"),
        (">P99.5", "P99.5", None),
    )
    records = []
    for label, lower_key, upper_key in definitions:
        lower = percentile_values[lower_key]
        upper = percentile_values.get(upper_key) if upper_key else None
        mask = np.isfinite(power_scores) & (power_scores >= lower)
        if upper is not None:
            mask &= power_scores < upper
        records.append({
            "label": label,
            "lower_percentile": lower_key,
            "upper_percentile": upper_key,
            "lower_bound": lower,
            "upper_bound": upper,
            "particle_count": int(np.count_nonzero(mask)),
        })
    return records


def _candidate_upper_thresholds(power_scores: np.ndarray) -> list[dict[str, Any]]:
    """Report percentile-derived upper-bound sensitivity counts, never a recommendation."""
    power_scores = np.asarray(power_scores, dtype=np.float64)
    finite = power_scores[np.isfinite(power_scores)]
    if finite.size == 0:
        return []
    candidates = []
    for percentile in (90, 95, 97.5, 99, 99.5):
        threshold = float(np.percentile(finite, percentile))
        retained = int(np.count_nonzero(finite < threshold))
        removed = int(finite.size - retained)
        candidates.append({
            "derived_from_power_percentile": f"P{percentile:g}",
            "lpower_thresh_max": threshold,
            "comparison": "power < lpower_thresh_max",
            "population": "finite_power_particles",
            "population_count": int(finite.size),
            "retained_count": retained,
            "retained_fraction": retained / finite.size,
            "removed_high_power_count": removed,
            "removed_high_power_fraction": removed / finite.size,
            "threshold_recommendation": None,
        })
    return candidates


def build_structured_pick_statistics(
    particle_count: int,
    ncc_scores: np.ndarray,
    power_scores: np.ndarray,
    histogram_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build compact full-population evidence for an Inspect Picks Model input."""
    ncc_scores = np.asarray(ncc_scores, dtype=np.float64)
    power_scores = np.asarray(power_scores, dtype=np.float64)
    valid_pairs = np.isfinite(ncc_scores) & np.isfinite(power_scores)
    finite_power_count = int(np.count_nonzero(np.isfinite(power_scores)))
    return {
        "particle_count": int(particle_count),
        "valid_ncc_power_pair_count": int(np.count_nonzero(valid_pairs)),
        "ncc_quantiles": _pick_quantiles(ncc_scores),
        "power_quantiles": _pick_quantiles(power_scores),
        "histogram": {
            "ncc_power": histogram_metadata,
            "power_tail_bins": _power_tail_bins(power_scores),
            "finite_power_count": finite_power_count,
            "nonfinite_power_count": int(power_scores.size - finite_power_count),
        },
        "candidate_upper_thresholds": _candidate_upper_thresholds(power_scores),
        "threshold_recommendation": None,
    }


def _select_diverse_tail_indices(
    candidate_indices: np.ndarray,
    particle_uids: Any,
    maximum: int,
) -> list[int]:
    """Select tail examples across micrographs before taking a second pick from any one."""
    grouped: dict[int, list[int]] = {}
    for index in candidate_indices.tolist():
        grouped.setdefault(int(particle_uids[index]), []).append(int(index))
    if not grouped or maximum <= 0:
        return []
    uids = sorted(grouped)
    if len(uids) > maximum:
        sampled_positions = np.linspace(0, len(uids) - 1, maximum).round().astype(int)
        uids = [uids[position] for position in sorted(set(sampled_positions))]
    selected: list[int] = []
    cursor = 0
    while len(selected) < maximum:
        added = False
        for uid in uids:
            picks = grouped[uid]
            if cursor < len(picks):
                selected.append(picks[cursor])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        cursor += 1
    return selected


def select_high_power_tail_particles(
    power_scores: np.ndarray,
    particle_uids: Any,
    samples_per_bin: int = 6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select approximately 24 high-Power particle crops with micrograph diversity."""
    power_scores = np.asarray(power_scores, dtype=np.float64)
    # CryoSPARC UIDs are unsigned 64-bit values. Keep Python ints: converting a
    # mixed high-bit UID collection with np.asarray can silently produce float64
    # and lose the bits required to locate the source micrograph.
    particle_uids = [int(value) for value in particle_uids]
    if power_scores.size != len(particle_uids):
        raise ValueError("Power Score and micrograph UID arrays must have matching lengths.")
    tail_bins = _power_tail_bins(power_scores)
    selected: list[dict[str, Any]] = []
    for tail_bin in tail_bins:
        lower = tail_bin["lower_bound"]
        upper = tail_bin["upper_bound"]
        mask = np.isfinite(power_scores) & (power_scores >= lower)
        if upper is not None:
            mask &= power_scores < upper
        indices = np.flatnonzero(mask)
        chosen = _select_diverse_tail_indices(indices, particle_uids, samples_per_bin)
        selected.extend({
            "particle_index": index,
            "tail_bin": tail_bin["label"],
        } for index in chosen)
        tail_bin["selected_particle_count"] = len(chosen)
        tail_bin["selected_micrograph_count"] = len({int(particle_uids[index]) for index in chosen})
    return selected, tail_bins


def _display_bounds(values: np.ndarray) -> tuple[float, float]:
    """Choose a robust chart viewport while retaining the full range in metadata."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        center = float(low)
        padding = max(abs(center) * 0.05, 1.0)
        return center - padding, center + padding
    padding = (high - low) * 0.03
    return float(low - padding), float(high + padding)


def _font(size: int) -> ImageFont.ImageFont:
    """Use a readable scalable font where Pillow provides one, with a safe fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _density_color(value: float) -> tuple[int, int, int]:
    """Map normalized log-density to a compact perceptually ordered colour ramp."""
    stops = [
        (68, 1, 84), (59, 82, 139), (33, 145, 140),
        (94, 201, 98), (253, 231, 37),
    ]
    value = min(1.0, max(0.0, value))
    scaled = value * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    fraction = scaled - index
    start, end = stops[index], stops[index + 1]
    return tuple(round(start[channel] + (end[channel] - start[channel]) * fraction)
                 for channel in range(3))


def build_pick_qc_dashboard(
    pick_counts: np.ndarray,
    ncc_scores: np.ndarray,
    power_scores: np.ndarray,
    width: int = 768,
    height: int = 960,
    bins: int = 48,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render Exposure Plot and NCC × Power density evidence without a threshold heuristic."""
    pick_counts = np.asarray(pick_counts, dtype=np.float64)
    ncc_scores = np.asarray(ncc_scores, dtype=np.float64)
    power_scores = np.asarray(power_scores, dtype=np.float64)
    if ncc_scores.size != power_scores.size:
        raise ValueError("NCC Score and Power Score arrays must have matching lengths.")

    dashboard = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(dashboard)
    title_font = _font(22)
    label_font = _font(16)
    small_font = _font(13)
    muted = (75, 85, 99)
    axis = (45, 55, 72)
    grid = (220, 225, 232)

    # Exposure Plot: each point is one input micrograph in source order.
    exposure_bottom = min(400, round(height * 0.42))
    exposure_bounds = (86, 74, width - 34, exposure_bottom)
    left, top, right, bottom = exposure_bounds
    draw.text((left, 28), "Exposure Plot", fill=axis, font=title_font)
    draw.text(
        (left, 54), "Number of picked particles by micrograph index (source order)",
        fill=muted, font=small_font,
    )
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = round(bottom - (bottom - top) * fraction)
        draw.line((left, y, right, y), fill=grid, width=1)
    draw.line((left, top, left, bottom), fill=axis, width=2)
    draw.line((left, bottom, right, bottom), fill=axis, width=2)
    finite_counts = pick_counts[np.isfinite(pick_counts)]
    if finite_counts.size:
        count_min = float(np.min(finite_counts))
        count_max = float(np.max(finite_counts))
        count_span = max(count_max - count_min, 1.0)
        count_low = max(0.0, count_min - count_span * 0.06)
        count_high = count_max + count_span * 0.06
        for fraction in (0.0, 0.5, 1.0):
            y = bottom - (bottom - top) * fraction
            value = count_low + (count_high - count_low) * fraction
            draw.text((10, y - 8), f"{value:.0f}", fill=muted, font=small_font)
        if len(pick_counts) == 1:
            x_positions = np.asarray([(left + right) / 2])
        else:
            x_positions = np.linspace(left, right, len(pick_counts))
        points = []
        for x, value in zip(x_positions, pick_counts):
            if not np.isfinite(value):
                continue
            y = bottom - (value - count_low) / (count_high - count_low) * (bottom - top)
            points.append((round(float(x)), round(float(y))))
        if len(points) > 1:
            draw.line(points, fill=(75, 95, 235), width=2)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(68, 82, 240))
    else:
        draw.text((left + 12, top + 12), "No finite pick counts", fill=muted, font=label_font)
    draw.text((left, bottom + 12), "Micrograph index", fill=axis, font=label_font)
    draw.text((left, bottom + 34), "Observed counts only; no automatic exposure exclusion.",
              fill=muted, font=small_font)

    # Power Histogram: a two-dimensional observed density of the two Picker scores.
    chart_top = max(exposure_bottom + 70, round(height * 0.54))
    chart_bounds = (86, chart_top + 52, width - 76, height - max(84, round(height * 0.09)))
    left, top, right, bottom = chart_bounds
    draw.text((left, chart_top), "Power Histogram", fill=axis, font=title_font)
    pair_mask = np.isfinite(ncc_scores) & np.isfinite(power_scores)
    ncc_pairs = ncc_scores[pair_mask]
    power_pairs = power_scores[pair_mask]
    ncc_low, ncc_high = _display_bounds(ncc_pairs)
    power_low, power_high = _display_bounds(power_pairs)
    histogram, _, _ = np.histogram2d(
        ncc_pairs,
        power_pairs,
        bins=(bins, bins),
        range=((ncc_low, ncc_high), (power_low, power_high)),
    )
    display_mask = (
        (ncc_pairs >= ncc_low) & (ncc_pairs <= ncc_high) &
        (power_pairs >= power_low) & (power_pairs <= power_high)
    )
    outside_count = int(ncc_pairs.size - np.count_nonzero(display_mask))
    draw.text(
        (left, chart_top + 27),
        "NCC Score (x) × Power Score (y); viewport is robust, tails are counted below.",
        fill=muted, font=small_font,
    )
    max_density = float(np.max(histogram)) if histogram.size else 0.0
    cell_width = (right - left) / bins
    cell_height = (bottom - top) / bins
    for x_index in range(bins):
        for y_index in range(bins):
            count = histogram[x_index, y_index]
            if count <= 0:
                continue
            normalised = math.log1p(float(count)) / math.log1p(max_density)
            x0 = round(left + x_index * cell_width)
            x1 = round(left + (x_index + 1) * cell_width)
            y1 = round(bottom - y_index * cell_height)
            y0 = round(bottom - (y_index + 1) * cell_height)
            draw.rectangle((x0, y0, x1, y1), fill=_density_color(normalised))
    draw.rectangle((left, top, right, bottom), outline=axis, width=2)
    draw.text((10, top - 20), "Power Score ↑", fill=axis, font=label_font)
    for fraction in (0.0, 0.5, 1.0):
        x = left + (right - left) * fraction
        y = bottom - (bottom - top) * fraction
        ncc_value = ncc_low + (ncc_high - ncc_low) * fraction
        power_value = power_low + (power_high - power_low) * fraction
        draw.text((x - 18, bottom + 8), f"{ncc_value:.2g}", fill=muted, font=small_font)
        draw.text((10, y - 8), f"{power_value:.3g}", fill=muted, font=small_font)
    draw.text((left, bottom + 32), "NCC Score", fill=axis, font=label_font)
    draw.text(
        (left, bottom + 53),
        f"Finite pairs: {int(ncc_pairs.size)}; outside displayed viewport: {outside_count}; "
        "displayed bounds are not threshold recommendations.",
        fill=muted, font=small_font,
    )

    return dashboard, {
        "exposure_plot": {
            "x_axis": "micrograph_index_source_order",
            "y_axis": "number_of_picked_particles",
            "micrograph_count": int(pick_counts.size),
            "pick_count_statistics": _summary(pick_counts),
        },
        "power_histogram": {
            "x_axis": "ncc_score",
            "y_axis": "power_score",
            "finite_pair_count": int(ncc_pairs.size),
            "nonfinite_pair_count": int(ncc_scores.size - ncc_pairs.size),
            "bin_count": [bins, bins],
            "display_bounds": {
                "ncc_score": {"min": ncc_low, "max": ncc_high},
                "power_score": {"min": power_low, "max": power_high},
            },
            "outside_display_bounds_count": outside_count,
            "threshold_recommendation": None,
        },
    }


def _extract_particle_crop(
    image: np.ndarray,
    x_fraction: float,
    y_fraction: float,
    crop_size: int,
) -> np.ndarray:
    """Extract a fixed-size crop centred on one particle, padding edges with its median."""
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape[-2:]
    finite = image[np.isfinite(image)]
    fill_value = float(np.median(finite)) if finite.size else 0.0
    crop = np.full((crop_size, crop_size), fill_value, dtype=np.float32)
    center_x = round(float(x_fraction) * (width - 1))
    center_y = round(float(y_fraction) * (height - 1))
    half = crop_size // 2
    source_left = max(0, center_x - half)
    source_top = max(0, center_y - half)
    source_right = min(width, source_left + crop_size)
    source_bottom = min(height, source_top + crop_size)
    destination_left = max(0, half - center_x)
    destination_top = max(0, half - center_y)
    destination_right = destination_left + (source_right - source_left)
    destination_bottom = destination_top + (source_bottom - source_top)
    crop[destination_top:destination_bottom, destination_left:destination_right] = (
        image[source_top:source_bottom, source_left:source_right]
    )
    return crop


def build_high_power_targeted_inspection(
    project: Any,
    project_uid: str,
    job_uid: str,
    mic_paths: list[Any],
    mic_uids: list[int],
    particle_uids: np.ndarray,
    x_fraction: np.ndarray,
    y_fraction: np.ndarray,
    ncc_scores: np.ndarray,
    power_scores: np.ndarray,
    cache_dir: Path,
    micrograph_root: str | None,
    samples_per_bin: int = 6,
    tile_size: int = 192,
    columns: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render local crops for high-Power tail examples without selecting a threshold."""
    selected, tail_bins = select_high_power_tail_particles(
        power_scores,
        particle_uids,
        samples_per_bin=samples_per_bin,
    )
    mic_index = {uid: index for index, uid in enumerate(mic_uids)}
    loaded_images: dict[int, np.ndarray] = {}
    rendered: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for item in selected:
        particle_index = item["particle_index"]
        micrograph_uid = int(particle_uids[particle_index])
        if micrograph_uid not in mic_index:
            failures.append({"particle_index": particle_index, "error": "unknown_micrograph_uid"})
            continue
        try:
            if micrograph_uid not in loaded_images:
                source_path = str(mic_paths[mic_index[micrograph_uid]])
                loaded_images[micrograph_uid] = load_micrograph_image(
                    project,
                    source_path,
                    micrograph_root,
                )
            image = loaded_images[micrograph_uid]
            crop_size = min(768, max(256, min(image.shape[-2:]) // 8))
            crop = _extract_particle_crop(
                image,
                x_fraction[particle_index],
                y_fraction[particle_index],
                crop_size,
            )
            rendered.append({
                **item,
                "micrograph_uid": micrograph_uid,
                "crop": normalize_micrograph_image(crop, tile_size),
                "crop_size_px": crop_size,
                "power": float(power_scores[particle_index]),
                "ncc_score": float(ncc_scores[particle_index]),
            })
        except Exception as exc:  # Keep other tail evidence available if one MRC is unreadable.
            failures.append({"particle_index": particle_index, "error": str(exc)})

    metadata = {
        "selection_strategy": "power_percentile_tail_bins_then_micrograph_diverse_round_robin",
        "samples_per_tail_bin": samples_per_bin,
        "requested_particle_count": len(selected),
        "rendered_particle_count": len(rendered),
        "tail_bins": tail_bins,
        "unavailable_particle_count": len(failures),
        "unavailable_examples": failures[:5],
        "threshold_recommendation": None,
    }
    if not rendered:
        return {"error": "No high-Power tail crops could be rendered."}, metadata

    rows = math.ceil(len(rendered) / columns)
    label_height = 44
    sheet = Image.new(
        "RGB",
        (columns * tile_size, rows * (tile_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for panel_index, item in enumerate(rendered):
        row_index = panel_index // columns
        column_index = panel_index % columns
        x0 = column_index * tile_size
        y0 = row_index * (tile_size + label_height)
        sheet.paste(Image.fromarray(item["crop"], mode="RGB"), (x0, y0 + label_height))
        center = tile_size // 2
        radius = max(8, tile_size // 9)
        draw.ellipse(
            (x0 + center - radius, y0 + label_height + center - radius,
             x0 + center + radius, y0 + label_height + center + radius),
            outline=(255, 50, 30),
            width=2,
        )
        draw.rectangle((x0, y0, x0 + tile_size - 1, y0 + label_height - 1), fill="black")
        draw.text((x0 + 4, y0 + 3), item["tail_bin"], fill="white", font=font)
        draw.text(
            (x0 + 4, y0 + 19),
            f"Power={item['power']:.1f} NCC={item['ncc_score']:.3g}",
            fill="white",
            font=font,
        )

    artifact = _png_artifact(
        sheet,
        cache_dir / f"{project_uid}_{job_uid}_high_power_targeted_inspection.png",
    )
    artifact.update({
        "columns": columns,
        "tile_size": tile_size,
        "label_format": "tail_bin; Power=<number> NCC=<number>",
        "target_marker": "red circle at crop centre",
    })
    return artifact, metadata



def build_pick_inspection_visual_context(
    project_uid: str,
    job_uid: str,
    max_micrographs: int = 9,
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
    cache_dir = _vision_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    pick_counts = np.asarray([len(by_uid[uid]) for uid in mic_uids], dtype=np.float64)
    dashboard, dashboard_metadata = build_pick_qc_dashboard(pick_counts, ncc, power)
    dashboard_artifact = _png_artifact(
        dashboard,
        cache_dir / f"{project_uid}_{job_uid}_pick_qc_dashboard.png",
    )
    dashboard_artifact.update({
        "panels": ["exposure_plot", "power_histogram"],
        "layout": "top=exposure_plot,bottom=power_histogram",
    })
    structured_pick_statistics = build_structured_pick_statistics(
        particle_count=len(particle_uids),
        ncc_scores=ncc,
        power_scores=power,
        histogram_metadata=dashboard_metadata["power_histogram"],
    )
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

    cached_path = cache_dir / f"{project_uid}_{job_uid}_pick_inspection.png"
    contact_sheet_artifact = _png_artifact(sheet, cached_path)
    contact_sheet_artifact.update({
        "columns": columns,
        "tile_size": tile_size,
        "label_format": "micrograph_id=<integer> uid=<integer> picks=<integer>",
    })
    high_power_artifact, high_power_metadata = build_high_power_targeted_inspection(
        project=project,
        project_uid=project_uid,
        job_uid=job_uid,
        mic_paths=mic_paths,
        mic_uids=mic_uids,
        particle_uids=particle_uids,
        x_fraction=x_frac,
        y_fraction=y_frac,
        ncc_scores=ncc,
        power_scores=power,
        cache_dir=cache_dir,
        micrograph_root=micrograph_root,
    )

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
        "score_statistics": {"ncc_score": _summary(ncc), "power": _summary(power)},
        "structured_pick_statistics": structured_pick_statistics,
        "exposure_plot": dashboard_metadata["exposure_plot"],
        "power_histogram": dashboard_metadata["power_histogram"],
        "high_power_targeted_inspection": high_power_artifact,
        "high_power_targeted_inspection_metadata": high_power_metadata,
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
                "Use thresholds to remove the low-score dark/background tail, and allow "
                "upper thresholds when the visual evidence identifies high-score artifacts."
            )
        },
        "overlay": {"marker": "red circles", "max_picks_per_micrograph": max_picks_per_micrograph},
        "pick_qc_dashboard": dashboard_artifact,
        "contact_sheet": contact_sheet_artifact,
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
