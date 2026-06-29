from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label, regionprops


# -----------------------------
# User-editable configuration
# -----------------------------
MASK_TIFF_PATH = r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium_data\Xen24_3782_aki-selected\aki-protein-roi-masks.tif"
OUTPUT_CSV_PATH = None  # None -> <mask_stem>_instance_bboxes.csv
PAD_X = 10
PAD_Y = 10
BACKGROUND_VALUE = 0
SAVE_DEBUG_BBOX_PLOT = True
DEBUG_PLOT_PATH = None  # None -> <mask_stem>_instance_bboxes_debug.png
DEBUG_PLOT_MAX_DIM_PX = 3000
DEBUG_BBOX_THICKNESS_PX = 4
DEBUG_BBOX_COLOR_RGB = (0, 255, 0)


def _safe_ratio_to_float(value):
    """Convert TIFF rational-like values to float."""
    if value is None:
        return None
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            if value.denominator == 0:
                return None
            return float(value.numerator) / float(value.denominator)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            num, den = value
            if den == 0:
                return None
            return float(num) / float(den)
        return float(value)
    except Exception:
        return None


def get_pixel_size_from_tiff(tiff_path):
    """
    Returns:
        (pixel_size_x, pixel_size_y, unit)
    where pixel sizes are in "unit per pixel".
    """
    with tifffile.TiffFile(tiff_path) as tif:
        page = tif.pages[0]

        xres_tag = page.tags.get("XResolution")
        yres_tag = page.tags.get("YResolution")
        res_unit_tag = page.tags.get("ResolutionUnit")

        x_pixels_per_unit = _safe_ratio_to_float(xres_tag.value if xres_tag else None)
        y_pixels_per_unit = _safe_ratio_to_float(yres_tag.value if yres_tag else None)

        # TIFF resolution unit spec: 1=None, 2=inch, 3=centimeter
        tiff_unit_map = {1: "none", 2: "inch", 3: "centimeter"}
        tiff_unit = tiff_unit_map.get(int(res_unit_tag.value), "unknown") if res_unit_tag else "unknown"

        ij_meta = tif.imagej_metadata or {}
        unit = ij_meta.get("unit", tiff_unit)

    pixel_size_x = (1.0 / x_pixels_per_unit) if x_pixels_per_unit and x_pixels_per_unit > 0 else np.nan
    pixel_size_y = (1.0 / y_pixels_per_unit) if y_pixels_per_unit and y_pixels_per_unit > 0 else np.nan

    return pixel_size_x, pixel_size_y, unit


def extract_instance_bboxes(mask, pad_x=10, pad_y=10, background_value=0):
    """
    Find instance bounding boxes in an 8-bit mask.

    If a label value appears in disconnected islands, each connected component
    is treated as a separate instance.
    """
    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, got shape {mask.shape}. "
            "Provide a 2D mask image."
        )

    rows = []
    instance_id = 0

    # Process each non-background value separately.
    values = np.unique(mask)
    values = values[values != background_value]

    for mask_value in values:
        binary = mask == mask_value
        cc = label(binary, connectivity=2)

        for component_index, region in enumerate(regionprops(cc), start=1):
            if region.area <= 10000:
                continue

            instance_id += 1
            min_row, min_col, max_row_exclusive, max_col_exclusive = region.bbox

            # Convert to inclusive pixel max for easier downstream use.
            max_row = max_row_exclusive - 1
            max_col = max_col_exclusive - 1

            width = max_col - min_col + 1
            height = max_row - min_row + 1

            # Remove XY padding used when mask was generated from parquet vertices.
            min_x_unpadded = min_col - pad_x
            min_y_unpadded = min_row - pad_y
            max_x_unpadded = max_col - pad_x
            max_y_unpadded = max_row - pad_y

            rows.append(
                {
                    "instance_id": instance_id,
                    "area_px": int(region.area),
                    "bbox_width_px": int(width),
                    "bbox_height_px": int(height),
                    "bbox_min_x_px": int(min_x_unpadded),
                    "bbox_max_x_px": int(max_x_unpadded),
                    "bbox_min_y_px": int(min_y_unpadded),
                    "bbox_max_y_px": int(max_y_unpadded),
                }
            )

    return pd.DataFrame(rows)


def _draw_thick_bbox(canvas, min_x, min_y, max_x, max_y, thickness, color):
    """Draw a thick rectangle border on an HxWx3 uint8 canvas."""
    h, w = canvas.shape[:2]

    min_x = max(0, min(w - 1, int(min_x)))
    max_x = max(0, min(w - 1, int(max_x)))
    min_y = max(0, min(h - 1, int(min_y)))
    max_y = max(0, min(h - 1, int(max_y)))

    if max_x < min_x or max_y < min_y:
        return

    t = max(1, int(thickness))

    # Top and bottom edges
    y0 = max(0, min_y)
    y1 = min(h, min_y + t)
    y2 = max(0, max_y - t + 1)
    y3 = min(h, max_y + 1)
    canvas[y0:y1, min_x:max_x + 1] = color
    canvas[y2:y3, min_x:max_x + 1] = color

    # Left and right edges
    x0 = max(0, min_x)
    x1 = min(w, min_x + t)
    x2 = max(0, max_x - t + 1)
    x3 = min(w, max_x + 1)
    canvas[min_y:max_y + 1, x0:x1] = color
    canvas[min_y:max_y + 1, x2:x3] = color


def save_debug_bbox_plot(
    bboxes,
    image_height_px,
    image_width_px,
    output_path,
    max_dim_px=3000,
    thickness_px=4,
    color_rgb=(0, 255, 0),
):
    """Save a downscaled bbox debug plot on a black background."""
    h = int(image_height_px)
    w = int(image_width_px)

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size for plotting: h={h}, w={w}")

    scale = max(h, w) / float(max_dim_px) if max(h, w) > max_dim_px else 1.0
    ds_h = max(1, int(np.ceil(h / scale)))
    ds_w = max(1, int(np.ceil(w / scale)))

    canvas = np.zeros((ds_h, ds_w, 3), dtype=np.uint8)

    for row in bboxes.itertuples(index=False):
        min_x = int(np.floor(row.bbox_min_x_px / scale))
        max_x = int(np.floor(row.bbox_max_x_px / scale))
        min_y = int(np.floor(row.bbox_min_y_px / scale))
        max_y = int(np.floor(row.bbox_max_y_px / scale))

        _draw_thick_bbox(
            canvas=canvas,
            min_x=min_x,
            min_y=min_y,
            max_x=max_x,
            max_y=max_y,
            thickness=thickness_px,
            color=color_rgb,
        )

    plt.imsave(output_path, canvas)
    return scale, ds_h, ds_w


def main():
    mask_path = Path(MASK_TIFF_PATH)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask TIFF not found: {mask_path}")

    if OUTPUT_CSV_PATH is None:
        output_csv = mask_path.with_name(f"{mask_path.stem}_instance_bboxes.csv")
    else:
        output_csv = Path(OUTPUT_CSV_PATH)

    # Memory-map to handle large TIFFs more efficiently.
    mask = tifffile.memmap(mask_path)

    bboxes = extract_instance_bboxes(
        mask=mask,
        pad_x=PAD_X,
        pad_y=PAD_Y,
        background_value=BACKGROUND_VALUE,
    )

    pixel_size_x, pixel_size_y, unit = get_pixel_size_from_tiff(mask_path)

    pixel_size_x = 0.2124023
    pixel_size_y = 0.2124023

    bboxes["pixel_size_x"] = pixel_size_x
    bboxes["pixel_size_y"] = pixel_size_y
    bboxes["spatial_unit"] = unit


    # Physical coordinates are derived after padding removal.
    bboxes["bbox_min_x_coord"] = bboxes["bbox_min_x_px"] * pixel_size_x
    bboxes["bbox_max_x_coord"] = bboxes["bbox_max_x_px"] * pixel_size_x
    bboxes["bbox_min_y_coord"] = bboxes["bbox_min_y_px"] * pixel_size_y
    bboxes["bbox_max_y_coord"] = bboxes["bbox_max_y_px"] * pixel_size_y

    bboxes.to_csv(output_csv, index=False)

    if SAVE_DEBUG_BBOX_PLOT:
        if DEBUG_PLOT_PATH is None:
            debug_plot_path = mask_path.with_name(f"{mask_path.stem}_instance_bboxes_debug.png")
        else:
            debug_plot_path = Path(DEBUG_PLOT_PATH)

        # Plot in the unpadded coordinate system used in bbox_min/max_x/y_px columns.
        unpadded_h = int(mask.shape[0]) - (2 * PAD_Y)
        unpadded_w = int(mask.shape[1]) - (2 * PAD_X)

        scale, ds_h, ds_w = save_debug_bbox_plot(
            bboxes=bboxes,
            image_height_px=unpadded_h,
            image_width_px=unpadded_w,
            output_path=debug_plot_path,
            max_dim_px=DEBUG_PLOT_MAX_DIM_PX,
            thickness_px=DEBUG_BBOX_THICKNESS_PX,
            color_rgb=DEBUG_BBOX_COLOR_RGB,
        )

    print(f"Saved {len(bboxes)} bounding boxes to: {output_csv}")
    print(f"Pixel size: x={pixel_size_x}, y={pixel_size_y}, unit={unit}")
    if SAVE_DEBUG_BBOX_PLOT:
        print(f"Saved debug bbox plot to: {debug_plot_path}")
        print(f"Debug plot scale: 1/{scale:.3f}, size: {ds_w}x{ds_h}")


if __name__ == "__main__":
    main()
