import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


# --------------------
# Inline configuration
# --------------------
# FOLDER_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI1X left side\processed-tifs"
# OUTPUT_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI1X left side\aki-roi1-HSI.tif"
# FOLDER_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI2X bottom center\processed-tifs"
# OUTPUT_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-13-25 Xenium AKI 3782 K2200093_5\ROI2X bottom center\aki-roi2-HSI.tif"
# FOLDER_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-14-25 Xenium Ref KPMP_038 S-2211-0003960\ROI1X glomeruli top left\processed-tifs"
# OUTPUT_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-14-25 Xenium Ref KPMP_038 S-2211-0003960\ROI1X glomeruli top left\ref-roi1-HSI.tif"
# FOLDER_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-14-25 Xenium Ref KPMP_038 S-2211-0003960\ROI2X glomeruli middle\processed-tifs"
# OUTPUT_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-14-25 Xenium Ref KPMP_038 S-2211-0003960\ROI2X glomeruli middle\ref-roi2-HSI.tif"
# FOLDER_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-15-25 Xenium CKD 3990\ROI1 glomeruli bottom right\processed-tifs"
# OUTPUT_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-15-25 Xenium CKD 3990\ROI1 glomeruli bottom right\ckd-roi1-HSI.tif"
FOLDER_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-15-25 Xenium CKD 3990\ROI2 glomeruli middle\processed-tifs"
OUTPUT_PATH = r"E:\Data\Jorge\01_HumanKidney_HuBMAP\05-15-25 Xenium CKD 3990\ROI2 glomeruli middle\ckd-roi2-HSI.tif"
PREVIEW_PATH = None
OVERLAP_OVERRIDE_PERCENT = 5.0

# GRID_ROWS = 4
# GRID_COLS = 9
# TARGET_OUTPUT_SHAPE = (1973, 4408)  # (height, width) or None to keep original canvas size

# GRID_ROWS = 5
# GRID_COLS = 7
# TARGET_OUTPUT_SHAPE = (2460, 3434)  # (height, width) or None to keep original canvas size

# GRID_ROWS = 7
# GRID_COLS = 5
# TARGET_OUTPUT_SHAPE = (3434, 2460)  # (height, width) or None to keep original canvas size

# GRID_ROWS = 6
# GRID_COLS = 6
# TARGET_OUTPUT_SHAPE = (2947, 2947)  # (height, width) or None to keep original canvas size

# GRID_ROWS = 4
# GRID_COLS = 9
# TARGET_OUTPUT_SHAPE = (1973, 4408)  # (height, width) or None to keep original canvas size

# GRID_ROWS = 4
# GRID_COLS = 9
# TARGET_OUTPUT_SHAPE = (1973, 4408)  # (height, width) or None to keep original canvas size

GRID_ROWS = 9
GRID_COLS = 4
TARGET_OUTPUT_SHAPE = (4408, 1973)  # (height, width) or None to keep original canvas size

RESIZE_INTERPOLATION = "bicubic"



def natural_key(value):
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def discover_tiff_files(folder_path):
    folder_path = Path(folder_path)
    paths = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        paths.extend(folder_path.glob(pattern))

    unique_paths = {}
    for path in paths:
        normalized = os.path.normcase(os.path.normpath(str(path)))
        if "stitch" in path.name.lower():
            continue
        unique_paths[normalized] = path

    return sorted(unique_paths.values(), key=lambda path: natural_key(path.name))


def _load_tile(path):
    tile = tifffile.memmap(path, mode="r")
    if tile.ndim == 2:
        return tile, tile.shape
    if tile.ndim != 3:
        raise ValueError(f"Expected 2D or 3D TIFF tile, got shape {tile.shape} from {path}")
    return tile, tile.shape[-2:]


def _get_resample_filter():
    interpolation = RESIZE_INTERPOLATION.lower()
    if interpolation == "nearest":
        return Image.Resampling.NEAREST
    if interpolation == "bicubic":
        return Image.Resampling.BICUBIC
    return Image.Resampling.BILINEAR


def _resize_2d_array(array_2d, target_height, target_width):
    if array_2d.shape == (target_height, target_width):
        return np.asarray(array_2d)

    resized = Image.fromarray(np.asarray(array_2d, dtype=np.float32), mode="F").resize(
        (target_width, target_height),
        resample=_get_resample_filter(),
    )
    return np.asarray(resized, dtype=np.float32)


def _cast_to_dtype(array, dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(array), info.min, info.max).astype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return (array > 0).astype(dtype)
    return np.asarray(array, dtype=dtype)


def _build_feather_weight(
    tile_height,
    tile_width,
    overlap_fraction,
    has_left_neighbor,
    has_right_neighbor,
    has_top_neighbor,
    has_bottom_neighbor,
):
    if overlap_fraction <= 0:
        return np.ones((tile_height, tile_width), dtype=np.float32)

    blend_x = max(1, int(round(tile_width * overlap_fraction)))
    blend_y = max(1, int(round(tile_height * overlap_fraction)))
    blend_x = min(blend_x, tile_width)
    blend_y = min(blend_y, tile_height)

    x_weight = np.ones(tile_width, dtype=np.float32)
    y_weight = np.ones(tile_height, dtype=np.float32)

    if has_left_neighbor:
        x_weight[:blend_x] = np.minimum(x_weight[:blend_x], np.linspace(0.0, 1.0, blend_x, dtype=np.float32))
    if has_right_neighbor:
        x_weight[-blend_x:] = np.minimum(x_weight[-blend_x:], np.linspace(1.0, 0.0, blend_x, dtype=np.float32))

    if has_top_neighbor:
        y_weight[:blend_y] = np.minimum(y_weight[:blend_y], np.linspace(0.0, 1.0, blend_y, dtype=np.float32))
    if has_bottom_neighbor:
        y_weight[-blend_y:] = np.minimum(y_weight[-blend_y:], np.linspace(1.0, 0.0, blend_y, dtype=np.float32))

    return (y_weight[:, None] * x_weight[None, :]).astype(np.float32)


def _snake_order(tile_paths, rows, cols):
    ordered = list(tile_paths)
    if len(ordered) != rows * cols:
        raise ValueError(f"Snake grid requires exactly {rows * cols} tiles, got {len(ordered)}")

    grid = []
    for row_index in range(rows):
        row_tiles = ordered[row_index * cols : (row_index + 1) * cols]
        if row_index % 2 == 1:
            row_tiles = list(reversed(row_tiles))
        grid.append(row_tiles)
    return grid


def _compute_snake_canvas(tile_shapes, rows, cols, overlap_percent):
    if not tile_shapes:
        raise ValueError("No tiles were found in the input folder")

    first_h, first_w = tile_shapes[0]
    if any(shape != (first_h, first_w) for shape in tile_shapes):
        raise ValueError(f"All tiles must have the same spatial shape; found {sorted(set(tile_shapes))}")

    overlap_fraction = (overlap_percent or 0.0) / 100.0
    step_x = max(1, int(round(first_w * (1.0 - overlap_fraction))))
    step_y = max(1, int(round(first_h * (1.0 - overlap_fraction))))
    canvas_width = first_w + (cols - 1) * step_x
    canvas_height = first_h + (rows - 1) * step_y
    return canvas_height, canvas_width, step_x, step_y


def stitch_omp2info_tifs(folder_path, output_path=None, preview_path=None):
    folder_path = Path(folder_path)
    tile_paths = discover_tiff_files(folder_path)
    if not tile_paths:
        raise FileNotFoundError(f"No TIFF files found in {folder_path}")

    tile_grid = _snake_order(tile_paths, GRID_ROWS, GRID_COLS)
    ordered_tile_paths = [path for row_tiles in tile_grid for path in row_tiles]

    tile_arrays = []
    tile_shapes = []
    for path in ordered_tile_paths:
        array, spatial_shape = _load_tile(path)
        tile_arrays.append(array)
        tile_shapes.append(spatial_shape)

    reference_array = tile_arrays[0]
    if reference_array.ndim not in {2, 3}:
        raise ValueError(f"Expected each TIFF tile to be 2D or 3D, got shape {reference_array.shape}.")

    if len(reference_array.shape[:-2]) > 1:
        raise ValueError(f"Expected a single leading stack axis or a 2D tile, got shape {reference_array.shape}.")

    canvas_height, canvas_width, step_x, step_y = _compute_snake_canvas(
        tile_shapes,
        GRID_ROWS,
        GRID_COLS,
        OVERLAP_OVERRIDE_PERCENT,
    )
    target_height, target_width = TARGET_OUTPUT_SHAPE if TARGET_OUTPUT_SHAPE is not None else (canvas_height, canvas_width)
    output_shape = reference_array.shape[:-2] + (target_height, target_width)

    if output_path is None:
        output_path = folder_path / f"{folder_path.name}_snake_mosaic.tif"
    else:
        output_path = Path(output_path)

    blend_sum = np.zeros(reference_array.shape[:-2] + (canvas_height, canvas_width), dtype=np.float32)
    blend_weight = np.zeros_like(blend_sum)

    placements = []
    overlap_fraction = (OVERLAP_OVERRIDE_PERCENT or 0.0) / 100.0

    for row_index, row_tiles in enumerate(tile_grid):
        row_start = row_index * GRID_COLS
        row_end = row_start + GRID_COLS
        for col_index, (tile_path, tile_array) in enumerate(zip(row_tiles, tile_arrays[row_start:row_end])):
            tile_height, tile_width = tile_array.shape[-2], tile_array.shape[-1]

            spatial_col = col_index
            has_left_neighbor = spatial_col > 0
            has_right_neighbor = spatial_col < GRID_COLS - 1

            feather_weight = _build_feather_weight(
                tile_height,
                tile_width,
                overlap_fraction,
                has_left_neighbor=has_left_neighbor,
                has_right_neighbor=has_right_neighbor,
                has_top_neighbor=row_index > 0,
                has_bottom_neighbor=row_index < GRID_ROWS - 1,
            )

            x0 = spatial_col * step_x
            y0 = row_index * step_y
            x1 = x0 + tile_width
            y1 = y0 + tile_height

            if tile_array.ndim == 2:
                weighted_tile = np.asarray(tile_array, dtype=np.float32) * feather_weight
                blend_sum[y0:y1, x0:x1] += weighted_tile
                blend_weight[y0:y1, x0:x1] += feather_weight
            else:
                weighted_tile = np.asarray(tile_array, dtype=np.float32) * feather_weight[None, :, :]
                blend_sum[:, y0:y1, x0:x1] += weighted_tile
                blend_weight[:, y0:y1, x0:x1] += feather_weight[None, :, :]

            placements.append(
                {
                    "tile_path": str(tile_path),
                    "row_index": row_index,
                    "col_index": spatial_col,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )

    if reference_array.ndim == 2:
        blended = np.divide(
            blend_sum,
            blend_weight,
            out=np.zeros_like(blend_sum, dtype=np.float32),
            where=blend_weight > 0,
        )
        final_preview = _resize_2d_array(blended, target_height, target_width)
        mosaic_data = _cast_to_dtype(final_preview, reference_array.dtype)
    else:
        mosaic_data = np.zeros(output_shape, dtype=reference_array.dtype)
        for plane_index in range(reference_array.shape[0]):
            blended = np.divide(
                blend_sum[plane_index],
                blend_weight[plane_index],
                out=np.zeros((canvas_height, canvas_width), dtype=np.float32),
                where=blend_weight[plane_index] > 0,
            )
            resized = _resize_2d_array(blended, target_height, target_width)
            mosaic_data[plane_index] = _cast_to_dtype(resized, reference_array.dtype)

    tifffile.imwrite(
        output_path,
        mosaic_data,
        imagej=True,
        metadata={"axes": "ZYX" if reference_array.ndim == 3 else "YX"},
    )

    if preview_path is not None:
        preview_path = Path(preview_path)
        if mosaic_data.ndim == 2:
            preview = np.asarray(mosaic_data)
        else:
            preview = np.max(np.asarray(mosaic_data), axis=0)
        tifffile.imwrite(preview_path, preview, metadata={"axes": "YX"}, imagej=True)

    return {
        "tile_paths": ordered_tile_paths,
        "output_path": output_path,
        "placements": placements,
        "canvas_shape": output_shape,
        "blend_mode": "linear_feather",
        "target_shape": (target_height, target_width),
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
    }


def main():
    result = stitch_omp2info_tifs(
        FOLDER_PATH,
        output_path=OUTPUT_PATH,
        preview_path=PREVIEW_PATH,
    )
    print(f"Stitched output: {result['output_path']}")
    print(f"Canvas shape: {result['canvas_shape']}")
    print(f"Tiles placed: {len(result['placements'])}")


if __name__ == "__main__":
    main()
