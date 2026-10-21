#!/usr/bin/env python3
"""
split_roi_mosaic_subsections.py
===============================
Splits transformed/coregistered ROI cell tables (e.g. *_cells_gui_rois.parquet,
*_transformed_rois.parquet, or vertex-level tables) into smaller subsections / tiles
matching an N x M image mosaic (e.g. stitched by stitch_omp2info_tifs.py).

Features:
- Automatic grid configuration discovery from unstitched TIFF tile directories
  (matching snake or raster ordering and tile spatial dimensions).
- Manual grid parameter specification (grid rows/cols, tile shape, overlap %).
- Overlap handling modes: 'intersect' (cells in overlapping margins are included
  in both adjacent sub-ROIs), 'centroid' (assigned uniquely by centroid), 'strict'.
- Multi-coordinate reference support: generates Tile-Local coordinates [0, tile_w/h]
  for direct per-tile inspection and HSI processing, while preserving Canvas-Global
  coordinates in auxiliary columns.
- Outputs individual sub-ROI Parquet/CSV files, a JSON manifest of all tile bounding
  boxes and cell counts, and optional per-tile labeled mask TIFFs.
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
try:
    import shapely.geometry as sg
    from shapely import wkt as shapely_wkt
except ImportError:
    sg = None
    shapely_wkt = None
import tifffile

try:
    import cv2
except ImportError:
    cv2 = None


def natural_key(value: Any) -> List[Union[int, str]]:
    """Sort strings with embedded numbers naturally (e.g. tile_2 before tile_10)."""
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def parse_wkt_polygon(wkt_str: str) -> np.ndarray:
    """Parse POLYGON ((x y, x y, ...)) into Nx2 numpy array without requiring shapely."""
    if shapely_wkt is not None:
        try:
            geom = shapely_wkt.loads(wkt_str)
            return np.array(geom.exterior.coords, dtype=np.float64)
        except Exception:
            pass
    # Regex fallback
    m = re.search(r"\(\((.*?)\)\)", wkt_str)
    if not m:
        m = re.search(r"\((.*?)\)", wkt_str)
    if not m:
        raise ValueError(f"Cannot parse WKT polygon: {wkt_str[:50]}...")
    coords_text = m.group(1).strip()
    points = []
    for pair in coords_text.split(","):
        parts = pair.strip().split()
        if len(parts) >= 2:
            points.append([float(parts[0]), float(parts[1])])
    return np.array(points, dtype=np.float64)


def compute_polygon_area(pts: np.ndarray) -> float:
    """Compute polygon area using Shoelace formula."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def clip_polygon_to_box(pts: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """
    Clips a 2D polygon against an axis-aligned bounding box [x0, y0, x1, y1]
    using the Sutherland-Hodgman polygon clipping algorithm.
    """
    if len(pts) < 3:
        return np.clip(pts, [x0, y0], [x1, y1])

    poly = pts[:-1].tolist() if (len(pts) > 2 and np.allclose(pts[0], pts[-1])) else pts.tolist()

    def clip_edge(poly_pts, edge):
        if not poly_pts:
            return []
        clipped = []
        for i in range(len(poly_pts)):
            p1 = poly_pts[i]
            p2 = poly_pts[(i + 1) % len(poly_pts)]

            if edge == "left":
                inside1 = p1[0] >= x0
                inside2 = p2[0] >= x0
                if inside1 and inside2:
                    clipped.append(p2)
                elif inside1 and not inside2:
                    dy = p2[1] - p1[1]
                    dx = p2[0] - p1[0]
                    y = p1[1] + dy * (x0 - p1[0]) / dx if dx != 0 else p1[1]
                    clipped.append([x0, y])
                elif not inside1 and inside2:
                    dy = p2[1] - p1[1]
                    dx = p2[0] - p1[0]
                    y = p1[1] + dy * (x0 - p1[0]) / dx if dx != 0 else p1[1]
                    clipped.append([x0, y])
                    clipped.append(p2)

            elif edge == "right":
                inside1 = p1[0] <= x1
                inside2 = p2[0] <= x1
                if inside1 and inside2:
                    clipped.append(p2)
                elif inside1 and not inside2:
                    dy = p2[1] - p1[1]
                    dx = p2[0] - p1[0]
                    y = p1[1] + dy * (x1 - p1[0]) / dx if dx != 0 else p1[1]
                    clipped.append([x1, y])
                elif not inside1 and inside2:
                    dy = p2[1] - p1[1]
                    dx = p2[0] - p1[0]
                    y = p1[1] + dy * (x1 - p1[0]) / dx if dx != 0 else p1[1]
                    clipped.append([x1, y])
                    clipped.append(p2)

            elif edge == "top":
                inside1 = p1[1] >= y0
                inside2 = p2[1] >= y0
                if inside1 and inside2:
                    clipped.append(p2)
                elif inside1 and not inside2:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    x = p1[0] + dx * (y0 - p1[1]) / dy if dy != 0 else p1[0]
                    clipped.append([x, y0])
                elif not inside1 and inside2:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    x = p1[0] + dx * (y0 - p1[1]) / dy if dy != 0 else p1[0]
                    clipped.append([x, y0])
                    clipped.append(p2)

            elif edge == "bottom":
                inside1 = p1[1] <= y1
                inside2 = p2[1] <= y1
                if inside1 and inside2:
                    clipped.append(p2)
                elif inside1 and not inside2:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    x = p1[0] + dx * (y1 - p1[1]) / dy if dy != 0 else p1[0]
                    clipped.append([x, y1])
                elif not inside1 and inside2:
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    x = p1[0] + dx * (y1 - p1[1]) / dy if dy != 0 else p1[0]
                    clipped.append([x, y1])
                    clipped.append(p2)

        return clipped

    res = poly
    for e in ["left", "right", "top", "bottom"]:
        res = clip_edge(res, e)

    if len(res) >= 3:
        return np.array(res, dtype=np.float64)
    return np.clip(pts, [x0, y0], [x1, y1])


def discover_tiff_tiles(folder_path: Union[str, Path]) -> List[Path]:
    """Discover and naturally sort all tile image files in folder excluding existing stitched files."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Tile directory not found: {folder_path}")

    patterns = ("*.tif", "*.tiff", "*.TIF", "*.TIFF", "*.png", "*.PNG", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
    paths = []
    for pattern in patterns:
        paths.extend(folder_path.glob(pattern))

    # If top-level has no images, check direct subdirectories (e.g. processed-tifs, tiles)
    if not paths:
        for sub in folder_path.iterdir():
            if sub.is_dir() and not sub.name.startswith("."):
                for pattern in patterns:
                    paths.extend(sub.glob(pattern))
                if paths:
                    break

    unique_paths = {}
    for p in paths:
        norm = os.path.normcase(os.path.normpath(str(p)))
        p_name_lower = p.name.lower()
        if any(keyword in p_name_lower for keyword in ("stitch", "mosaic", "preview", "mask", "transformed", ".tmp")):
            continue
        unique_paths[norm] = p

    return sorted(unique_paths.values(), key=lambda p: natural_key(p.name))


def get_tile_spatial_shape(tile_path: Path) -> Tuple[int, int]:
    """Read spatial (height, width) of a 2D or 3D TIFF or image tile without full memory load."""
    ext = tile_path.suffix.lower()
    if ext in (".tif", ".tiff"):
        with tifffile.TiffFile(tile_path) as tif:
            shape = tif.series[0].shape
            if len(shape) == 2:
                return int(shape[0]), int(shape[1])
            elif len(shape) >= 3:
                return int(shape[-2]), int(shape[-1])
            else:
                raise ValueError(f"Unsupported TIFF shape {shape} in {tile_path}")
    else:
        from PIL import Image as PILImage
        with PILImage.open(tile_path) as img:
            w, h = img.size
            return int(h), int(w)


def build_mosaic_grid_layout(
    rows: int,
    cols: int,
    tile_height: int,
    tile_width: int,
    overlap_percent: float = 5.0,
    order: str = "snake",
    tile_paths: Optional[List[Path]] = None,
    target_canvas_shape: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute spatial placement and bounding box [x0, y0, x1, y1] for each tile in the mosaic.
    Supports snake ordering matching stitch_omp2info_tifs.py.
    """
    overlap_fraction = (overlap_percent or 0.0) / 100.0
    step_x = max(1, int(round(tile_width * (1.0 - overlap_fraction))))
    step_y = max(1, int(round(tile_height * (1.0 - overlap_fraction))))

    raw_canvas_w = tile_width + (cols - 1) * step_x
    raw_canvas_h = tile_height + (rows - 1) * step_y

    # Canvas scale factor if target canvas was resized
    scale_x = 1.0
    scale_y = 1.0
    if target_canvas_shape is not None:
        target_h, target_w = target_canvas_shape
        scale_x = target_w / raw_canvas_w
        scale_y = target_h / raw_canvas_h

    # Organize tile paths in snake or raster grid if provided
    tile_path_grid: List[List[Optional[Path]]] = []
    if tile_paths:
        num_tiles_needed = rows * cols
        if len(tile_paths) < num_tiles_needed:
            print(f"Warning: Grid has {num_tiles_needed} positions ({rows}x{cols}) but only {len(tile_paths)} tiles discovered. Missing positions will be unassigned.")
            padded_tiles = list(tile_paths) + [None] * (num_tiles_needed - len(tile_paths))
        elif len(tile_paths) > num_tiles_needed:
            print(f"Note: Found {len(tile_paths)} tiles in tile directory, using the first {num_tiles_needed} for the {rows}x{cols} grid.")
            padded_tiles = list(tile_paths[:num_tiles_needed])
        else:
            padded_tiles = list(tile_paths)

        for r in range(rows):
            row_tiles = padded_tiles[r * cols : (r + 1) * cols]
            if order == "snake" and r % 2 == 1:
                row_tiles = list(reversed(row_tiles))
            tile_path_grid.append(row_tiles)
    else:
        for r in range(rows):
            tile_path_grid.append([None] * cols)

    tiles_info = []
    tile_idx = 0

    for r in range(rows):
        for c in range(cols):
            tile_idx += 1
            raw_x0 = c * step_x
            raw_y0 = r * step_y
            raw_x1 = raw_x0 + tile_width
            raw_y1 = raw_y0 + tile_height

            canvas_x0 = raw_x0 * scale_x
            canvas_y0 = raw_y0 * scale_y
            canvas_x1 = raw_x1 * scale_x
            canvas_y1 = raw_y1 * scale_y

            tile_file = tile_path_grid[r][c]
            tile_name = tile_file.stem if tile_file else f"tile_r{r:02d}_c{c:02d}"

            tiles_info.append({
                "tile_index": tile_idx,
                "row": r,
                "col": c,
                "tile_name": tile_name,
                "tile_file": str(tile_file.resolve()) if tile_file else "",
                "tile_filename": tile_file.name if tile_file else "",
                "tile_height": tile_height,
                "tile_width": tile_width,
                "raw_bounds": {"x0": raw_x0, "y0": raw_y0, "x1": raw_x1, "y1": raw_y1},
                "canvas_bounds": {"x0": canvas_x0, "y0": canvas_y0, "x1": canvas_x1, "y1": canvas_y1},
                "scale_x": scale_x,
                "scale_y": scale_y,
                "step_x": step_x,
                "step_y": step_y,
            })

    return tiles_info


class CellPolygon:
    """Helper representing a cell with its geometry and metadata."""
    def __init__(
        self,
        cell_id: str,
        label: str,
        vertices: np.ndarray,
        raw_record: Optional[Dict[str, Any]] = None,
    ):
        self.cell_id = str(cell_id)
        self.label = str(label)
        self.vertices = np.asarray(vertices, dtype=np.float64)  # Nx2 [ [x, y], ... ]
        self.raw_record = raw_record or {}

        # If already closed, unclose for raw vertex operations
        if len(self.vertices) > 2 and np.allclose(self.vertices[0], self.vertices[-1]):
            self.vertices_unique = self.vertices[:-1]
        else:
            self.vertices_unique = self.vertices

        if len(self.vertices) >= 2 and not np.allclose(self.vertices[0], self.vertices[-1]):
            self.vertices_closed = np.vstack([self.vertices, self.vertices[0]])
        else:
            self.vertices_closed = self.vertices

        self.cx = float(np.mean(self.vertices_unique[:, 0])) if len(self.vertices_unique) > 0 else 0.0
        self.cy = float(np.mean(self.vertices_unique[:, 1])) if len(self.vertices_unique) > 0 else 0.0
        self.min_x = float(np.min(self.vertices_unique[:, 0])) if len(self.vertices_unique) > 0 else 0.0
        self.max_x = float(np.max(self.vertices_unique[:, 0])) if len(self.vertices_unique) > 0 else 0.0
        self.min_y = float(np.min(self.vertices_unique[:, 1])) if len(self.vertices_unique) > 0 else 0.0
        self.max_y = float(np.max(self.vertices_unique[:, 1])) if len(self.vertices_unique) > 0 else 0.0
        self.area = compute_polygon_area(self.vertices_unique)


def load_cells_from_table(input_path: Union[str, Path]) -> List[CellPolygon]:
    """
    Loads cells from either GUI ROI table (with Vertices_JSON/Geometry_WKT)
    or per-vertex table (with cell_id, x, y, label).
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = input_path.suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    cells: List[CellPolygon] = []

    # Case 1: GUI Table Format (contains Vertices_JSON or Geometry_WKT)
    if "Vertices_JSON" in df.columns:
        cell_id_col = "UID" if "UID" in df.columns else ("cell_id" if "cell_id" in df.columns else df.columns[0])
        label_col = "Label" if "Label" in df.columns else ("group" if "group" in df.columns else "cell_type")

        for _, row in df.iterrows():
            cid = str(row[cell_id_col])
            lbl = str(row.get(label_col, "cell"))
            pts_data = json.loads(row["Vertices_JSON"])
            pts = np.array(pts_data, dtype=np.float64)
            cells.append(CellPolygon(cid, lbl, pts, raw_record=dict(row)))
        return cells

    if "Geometry_WKT" in df.columns:
        cell_id_col = "UID" if "UID" in df.columns else ("cell_id" if "cell_id" in df.columns else df.columns[0])
        label_col = "Label" if "Label" in df.columns else ("group" if "group" in df.columns else "cell_type")

        for _, row in df.iterrows():
            cid = str(row[cell_id_col])
            lbl = str(row.get(label_col, "cell"))
            pts = parse_wkt_polygon(str(row["Geometry_WKT"]))
            cells.append(CellPolygon(cid, lbl, pts, raw_record=dict(row)))
        return cells

    # Case 2: Per-vertex format (cell_id, x, y, label)
    x_col = next((c for c in ["x_roi", "x", "vertex_x", "Centroid_X"] if c in df.columns), None)
    y_col = next((c for c in ["y_roi", "y", "vertex_y", "Centroid_Y"] if c in df.columns), None)
    cid_col = next((c for c in ["cell_id", "UID", "cell_ID", "id"] if c in df.columns), df.columns[0])
    lbl_col = next((c for c in ["Label", "group", "cell_type", "levelKID_refined"] if c in df.columns), None)

    if x_col is None or y_col is None:
        raise ValueError(f"Cannot identify coordinate columns in {input_path}. Available: {list(df.columns)}")

    c_ids = df[cid_col].astype(str).values
    lbls = df[lbl_col].astype(str).values if lbl_col else np.full(len(df), "cell")
    vx = df[x_col].values
    vy = df[y_col].values

    change_idx = np.where(c_ids[:-1] != c_ids[1:])[0] + 1
    splits_xy = np.split(np.column_stack((vx, vy)), change_idx)
    splits_c = np.split(c_ids, change_idx)
    splits_l = np.split(lbls, change_idx)

    for pts, c_arr, l_arr in zip(splits_xy, splits_c, splits_l):
        cells.append(CellPolygon(str(c_arr[0]), str(l_arr[0]), pts))

    return cells


def split_roi_table_into_mosaic_tiles(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    tile_dir: Optional[Union[str, Path]] = None,
    grid_rows: Optional[int] = None,
    grid_cols: Optional[int] = None,
    tile_height: Optional[int] = None,
    tile_width: Optional[int] = None,
    overlap_percent: float = 5.0,
    order: str = "snake",
    stitched_image: Optional[Union[str, Path]] = None,
    target_canvas_shape: Optional[Tuple[int, int]] = None,
    coord_mode: str = "both",  # "local", "global", or "both"
    crop_mode: str = "intersect",  # "intersect", "centroid", "strict"
    clip_vertices: bool = False,
    sample_id: str = "1",
    output_prefix: Optional[str] = None,
    cohesive_name: Optional[str] = None,
    save_cohesive: bool = True,
    save_individual: bool = False,
    export_format: str = "both",  # "parquet", "csv", "both"
    generate_masks: bool = False,
    pixel_size_um: float = 1.0,
    metadata_dict: Optional[str] = None,
    metadata_list: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main function to split an ROI cell table into mosaic subsections and produce
    a cohesive .parquet / .csv table with tile image names from tile_dir.
    """
    t0 = time.time()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_prefix is None:
        output_prefix = input_path.stem.replace("_cells_gui_rois", "").replace("_transformed_rois", "")

    # 1. Discover or validate grid setup
    tile_paths: Optional[List[Path]] = None
    if tile_dir:
        tile_dir_path = Path(tile_dir)
        tile_paths = discover_tiff_tiles(tile_dir_path)
        print(f"Discovered {len(tile_paths)} raw tiles in {tile_dir_path}")
        if len(tile_paths) > 0:
            first_th, first_tw = get_tile_spatial_shape(tile_paths[0])
            if tile_height is None:
                tile_height = first_th
            if tile_width is None:
                tile_width = first_tw

    if tile_height is None or tile_width is None:
        tile_height, tile_width = 1024, 1024
        print(f"Defaulting tile size to {tile_width}x{tile_height} px.")

    # 2. Load all cells from input table first to know coordinate bounds
    print(f"Loading cell boundaries from {input_path}...")
    cells = load_cells_from_table(input_path)
    print(f"Loaded {len(cells):,} total cells.")

    if not cells:
        raise ValueError(f"No cells found in input table {input_path}")

    max_cell_x = max(c.max_x for c in cells)
    max_cell_y = max(c.max_y for c in cells)

    if grid_rows is None or grid_cols is None:
        if tile_paths and len(tile_paths) > 0:
            total = len(tile_paths)
            factors = [(r, total // r) for r in range(1, total + 1) if total % r == 0]
            aspect_target = (max_cell_y / max_cell_x) if max_cell_x > 0 else 1.0
            best_pair = min(factors, key=lambda pair: abs((pair[0] / pair[1]) * (tile_width / tile_height) - aspect_target))
            grid_rows, grid_cols = best_pair
            print(f"Auto-inferred grid layout from {total} tiles and canvas aspect ratio: {grid_rows} rows x {grid_cols} cols")
        else:
            raise ValueError("Must provide either --tile-dir or both --grid-rows and --grid-cols.")

    # Canvas dimensions & optional stitched reference
    if stitched_image and os.path.exists(stitched_image):
        canvas_h, canvas_w = get_tile_spatial_shape(Path(stitched_image))
        target_canvas_shape = (canvas_h, canvas_w)
        print(f"Canvas shape read from stitched image: {canvas_w} x {canvas_h} px ({stitched_image})")
    elif target_canvas_shape is None and tile_dir:
        cand_stitched = []
        for p in Path(tile_dir).glob("*.tif*"):
            if "stitch" in p.name.lower() or "mosaic" in p.name.lower():
                cand_stitched.append(p)
        if not cand_stitched and input_path.parent.exists():
            for p in input_path.parent.glob("*.tif*"):
                if "stitch" in p.name.lower() or "mosaic" in p.name.lower() or p.stem == output_prefix:
                    cand_stitched.append(p)
        if cand_stitched:
            try:
                canvas_h, canvas_w = get_tile_spatial_shape(cand_stitched[0])
                target_canvas_shape = (canvas_h, canvas_w)
                print(f"Auto-discovered stitched mosaic image: {cand_stitched[0].name} ({canvas_w} x {canvas_h} px)")
            except Exception:
                pass

    print(f"Mosaic Grid: {grid_rows} rows x {grid_cols} cols ({grid_rows * grid_cols} tiles), overlap={overlap_percent}%, order='{order}'")
    print(f"Tile dimensions: {tile_width} x {tile_height} px")

    # 3. Build grid bounding boxes
    tiles_layout = build_mosaic_grid_layout(
        rows=grid_rows,
        cols=grid_cols,
        tile_height=tile_height,
        tile_width=tile_width,
        overlap_percent=overlap_percent,
        order=order,
        tile_paths=tile_paths,
        target_canvas_shape=target_canvas_shape,
    )

    # 4. Build global label mapping across all cells in input dataset
    global_unique_labels = sorted(list(set(c.label for c in cells if c.label)))
    label_to_id = {}
    if metadata_dict:
        if os.path.exists(metadata_dict):
            with open(metadata_dict, "r") as f:
                raw_dict = json.load(f)
        else:
            raw_dict = json.loads(metadata_dict)
        label_to_id = {str(k): int(v) for k, v in raw_dict.items()}
    elif metadata_list:
        if os.path.exists(metadata_list):
            with open(metadata_list, "r") as f:
                content = f.read().strip()
                items = json.loads(content) if content.startswith("[") else [l.strip() for l in content.splitlines() if l.strip()]
        elif metadata_list.strip().startswith("["):
            items = json.loads(metadata_list)
        else:
            items = [x.strip() for x in metadata_list.split(",") if x.strip()]
        label_to_id = {str(lbl): idx for idx, lbl in enumerate(items, start=1)}
    else:
        label_to_id = {lbl: idx for idx, lbl in enumerate(global_unique_labels, start=1)}

    id_to_label = {v: k for k, v in label_to_id.items()}
    print(f"Dataset Label Mapping: {len(label_to_id)} unique cell classes registered.")

    # 5. Partition cells into each tile subsection
    manifest = {
        "source_file": str(input_path),
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "tile_height": tile_height,
        "tile_width": tile_width,
        "overlap_percent": overlap_percent,
        "order": order,
        "coord_mode": coord_mode,
        "crop_mode": crop_mode,
        "total_source_cells": len(cells),
        "save_cohesive": save_cohesive,
        "save_individual": save_individual,
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "tiles": [],
    }

    tile_summary = []
    all_records: List[Dict[str, Any]] = []

    for tile in tiles_layout:
        bx0 = tile["canvas_bounds"]["x0"]
        by0 = tile["canvas_bounds"]["y0"]
        bx1 = tile["canvas_bounds"]["x1"]
        by1 = tile["canvas_bounds"]["y1"]
        t_w = tile["tile_width"]
        t_h = tile["tile_height"]
        t_name = tile["tile_name"]
        scale_x = tile["scale_x"]
        scale_y = tile["scale_y"]

        matched_cells: List[Tuple[CellPolygon, np.ndarray, np.ndarray, float, float, float, float]] = []

        for cell in cells:
            # Check overlap / inclusion
            if crop_mode == "centroid":
                if not (bx0 <= cell.cx < bx1 and by0 <= cell.cy < by1):
                    continue
            elif crop_mode == "strict":
                if not (bx0 <= cell.min_x and cell.max_x < bx1 and by0 <= cell.min_y and cell.max_y < by1):
                    continue
            else:  # "intersect" (default)
                if cell.max_x < bx0 or cell.min_x >= bx1 or cell.max_y < by0 or cell.min_y >= by1:
                    continue

            # Compute canvas coordinates and local tile coordinates
            canvas_pts = cell.vertices.copy()
            local_pts = canvas_pts.copy()

            # Translate to local tile origin and remove any canvas scaling
            local_pts[:, 0] = (local_pts[:, 0] - bx0) / scale_x
            local_pts[:, 1] = (local_pts[:, 1] - by0) / scale_y

            if clip_vertices:
                local_pts = clip_polygon_to_box(local_pts, 0.0, 0.0, float(t_w), float(t_h))
                canvas_pts = clip_polygon_to_box(canvas_pts, bx0, by0, bx1, by1)

            # Ensure minimum polygon area and vertex count
            if len(local_pts) < 3 or len(canvas_pts) < 3:
                continue
            area_loc = compute_polygon_area(local_pts)
            if area_loc < 0.5:
                continue

            loc_u = local_pts[:-1] if (len(local_pts) > 2 and np.allclose(local_pts[0], local_pts[-1])) else local_pts
            can_u = canvas_pts[:-1] if (len(canvas_pts) > 2 and np.allclose(canvas_pts[0], canvas_pts[-1])) else canvas_pts

            cx_local = float(np.mean(loc_u[:, 0])) if len(loc_u) > 0 else 0.0
            cy_local = float(np.mean(loc_u[:, 1])) if len(loc_u) > 0 else 0.0
            cx_canvas = float(np.mean(can_u[:, 0])) if len(can_u) > 0 else 0.0
            cy_canvas = float(np.mean(can_u[:, 1])) if len(can_u) > 0 else 0.0

            matched_cells.append((cell, local_pts, canvas_pts, cx_local, cy_local, cx_canvas, cy_canvas))

        tile_cell_count = len(matched_cells)
        tile_manifest_entry = {
            **tile,
            "cell_count": tile_cell_count,
            "output_files": {},
        }

        if tile_cell_count == 0:
            print(f"[{tile['tile_index']}/{len(tiles_layout)}] {t_name}: 0 cells")
            manifest["tiles"].append(tile_manifest_entry)
            continue

        # Format records for this tile
        records = []
        tile_label_counters: Dict[str, int] = {}

        for cell_idx_in_tile, (cell_item, local_pts, canvas_pts, cxl, cyl, cxc, cyc) in enumerate(matched_cells, start=1):
            lbl = cell_item.label
            tile_label_counters[lbl] = tile_label_counters.get(lbl, 0) + 1
            roi_inst = tile_label_counters[lbl]

            # Primary coordinates selection based on coord_mode
            if coord_mode == "global":
                primary_pts = canvas_pts
                primary_cx, primary_cy = cxc, cyc
            else:  # "local" or "both"
                primary_pts = local_pts
                primary_cx, primary_cy = cxl, cyl

            if len(primary_pts) >= 2 and not np.allclose(primary_pts[0], primary_pts[-1]):
                primary_pts_closed = np.vstack([primary_pts, primary_pts[0]])
            else:
                primary_pts_closed = primary_pts

            if len(local_pts) >= 2 and not np.allclose(local_pts[0], local_pts[-1]):
                local_pts_closed = np.vstack([local_pts, local_pts[0]])
            else:
                local_pts_closed = local_pts

            if len(canvas_pts) >= 2 and not np.allclose(canvas_pts[0], canvas_pts[-1]):
                canvas_pts_closed = np.vstack([canvas_pts, canvas_pts[0]])
            else:
                canvas_pts_closed = canvas_pts

            wkt_primary = "POLYGON ((" + ", ".join(f"{x:.3f} {y:.3f}" for x, y in primary_pts_closed) + "))"
            wkt_local = "POLYGON ((" + ", ".join(f"{x:.3f} {y:.3f}" for x, y in local_pts_closed) + "))"
            wkt_canvas = "POLYGON ((" + ", ".join(f"{x:.3f} {y:.3f}" for x, y in canvas_pts_closed) + "))"

            pts_json_primary = json.dumps(primary_pts_closed.round(3).tolist())
            pts_json_local = json.dumps(local_pts_closed.round(3).tolist())
            pts_json_canvas = json.dumps(canvas_pts_closed.round(3).tolist())

            area = compute_polygon_area(primary_pts)

            # Preserve any existing columns from source table
            rec = dict(cell_item.raw_record) if cell_item.raw_record else {}
            rec.update({
                "Sample_ID": sample_id,
                "ROI_Number": len(all_records) + 1,
                "ROI_Number_tile": cell_idx_in_tile,
                "ROI_instance": roi_inst,
                "Label": lbl,
                "Vertex_Count": len(primary_pts_closed),
                "Geometry_WKT": wkt_primary,
                "Vertices_JSON": pts_json_primary,
                "Centroid_X": round(primary_cx, 3),
                "Centroid_Y": round(primary_cy, 3),
                "Area_px": round(area, 2),
                "UID": cell_item.cell_id,
                "cell_id": cell_item.cell_id,
                "Tile_Row": tile["row"],
                "Tile_Col": tile["col"],
                "Tile_Index": tile["tile_index"],
                "Tile_Filename": tile.get("tile_filename", ""),
                "Image_Name": os.path.splitext(t_name)[0],
                "Image_Path": tile["tile_file"],
                # Auxiliary dual-coordinate system columns
                "Centroid_X_local": round(cxl, 3),
                "Centroid_Y_local": round(cyl, 3),
                "Centroid_X_canvas": round(cxc, 3),
                "Centroid_Y_canvas": round(cyc, 3),
                "Geometry_WKT_local": wkt_local,
                "Geometry_WKT_canvas": wkt_canvas,
                "Vertices_JSON_local": pts_json_local,
                "Vertices_JSON_canvas": pts_json_canvas,
            })
            records.append(rec)
            all_records.append(rec)

        df_tile = pd.DataFrame(records)
        tile_base = output_dir / f"{output_prefix}_{t_name}"

        # Export individual Parquet / CSV only if save_individual is requested
        if save_individual:
            if export_format in ("parquet", "both"):
                pq_out = f"{tile_base}_rois.parquet"
                df_tile.to_parquet(pq_out, index=False)
                tile_manifest_entry["output_files"]["parquet"] = str(pq_out)

            if export_format in ("csv", "both"):
                csv_out = f"{tile_base}_rois.csv"
                df_tile.to_csv(csv_out, index=False)
                tile_manifest_entry["output_files"]["csv"] = str(csv_out)

        # Optional: Generate per-tile labeled mask TIFF
        if generate_masks:
            mask_out = f"{tile_base}_mask.tif"
            mask_arr = np.zeros((t_h, t_w), dtype=np.uint16)

            for cell_item, local_pts, _, _, _, _, _ in matched_cells:
                lbl = cell_item.label
                val = label_to_id.get(lbl, 0)
                if val > 0 and len(local_pts) >= 3:
                    pts_round = np.round(local_pts).astype(np.int32)
                    if cv2 is not None:
                        cv2.fillPoly(mask_arr, [pts_round], int(val))
                    else:
                        from PIL import Image as PILImage, ImageDraw as PILImageDraw
                        img_pil = PILImage.fromarray(mask_arr)
                        draw_pil = PILImageDraw.Draw(img_pil)
                        poly_tuples = [(float(p[0]), float(p[1])) for p in pts_round]
                        draw_pil.polygon(poly_tuples, fill=int(val))
                        mask_arr = np.array(img_pil)

            res_val = 1.0 / pixel_size_um if pixel_size_um > 0 else 1.0
            tifffile.imwrite(
                mask_out,
                mask_arr,
                imagej=True,
                resolution=(res_val, res_val),
                resolutionunit=1,
                metadata={"axes": "YX", "unit": "micron", "spacing": pixel_size_um},
            )
            tile_manifest_entry["output_files"]["mask_tif"] = str(mask_out)

        manifest["tiles"].append(tile_manifest_entry)
        tile_summary.append((t_name, tile_cell_count))
        print(f"[{tile['tile_index']}/{len(tiles_layout)}] {t_name}: {tile_cell_count:,} cells processed.")

    # 6. Save Cohesive Output Parquet / CSV across all tiles
    cohesive_files: Dict[str, str] = {}
    if save_cohesive and all_records:
        df_cohesive = pd.DataFrame(all_records)

        # Re-index global sequential ROI_Number and Region_Index
        df_cohesive["ROI_Number"] = np.arange(1, len(df_cohesive) + 1)
        # df_cohesive["Region_Index"] = np.arange(1, len(df_cohesive) + 1)

        # Clean column ordering with key identifiers first
        primary_cols = [
            "Sample_ID",
            "ROI_Number",
            "ROI_Number_tile",
            "ROI_instance",
            "Label",
            "Vertex_Count",
            "Geometry_WKT",
            "Vertices_JSON",
            "Centroid_X",
            "Centroid_Y",
            "Area_px",
            "UID",
            "cell_id",
            "Image_Name",
            "Image_Path",
            "Tile_Row",
            "Tile_Col",
            "Tile_Index",
            "Tile_Filename",
            "Centroid_X_local",
            "Centroid_Y_local",
            "Centroid_X_canvas",
            "Centroid_Y_canvas",
            "Geometry_WKT_local",
            "Geometry_WKT_canvas",
            "Vertices_JSON_local",
            "Vertices_JSON_canvas",
        ]
        ordered = [c for c in primary_cols if c in df_cohesive.columns]
        extra = [c for c in df_cohesive.columns if c not in ordered]
        df_cohesive = df_cohesive[ordered + extra]

        c_name = cohesive_name or f"{output_prefix}_cohesive_rois"
        cohesive_base = output_dir / c_name

        if export_format in ("parquet", "both"):
            cohesive_pq = f"{cohesive_base}.parquet"
            df_cohesive.to_parquet(cohesive_pq, index=False)
            sz_mb = os.path.getsize(cohesive_pq) / (1024 * 1024)
            print(f"Saved cohesive Parquet : {cohesive_pq} ({sz_mb:.2f} MB, {len(df_cohesive):,} records)")
            cohesive_files["parquet"] = str(cohesive_pq)

        if export_format in ("csv", "both"):
            cohesive_csv = f"{cohesive_base}.csv"
            df_cohesive.to_csv(cohesive_csv, index=False)
            sz_mb = os.path.getsize(cohesive_csv) / (1024 * 1024)
            print(f"Saved cohesive CSV     : {cohesive_csv} ({sz_mb:.2f} MB, {len(df_cohesive):,} records)")
            cohesive_files["csv"] = str(cohesive_csv)

    manifest["cohesive_output_files"] = cohesive_files
    manifest["total_subsections_cells"] = len(all_records)

    # 7. Save summary manifest and legend
    manifest_path = output_dir / f"{output_prefix}_mosaic_split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    legend_path = output_dir / f"{output_prefix}_mosaic_split_legend.json"
    with open(legend_path, "w") as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": {str(k): v for k, v in id_to_label.items()}
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Mosaic Subsections Processing Finished in {time.time() - t0:.2f}s")
    print(f"Total tiles processed: {len(tiles_layout)}")
    print(f"Total cohesive records: {len(all_records):,}")
    if cohesive_files:
        for fmt, p in cohesive_files.items():
            print(f"Cohesive {fmt.upper()} output: {p}")
    print(f"Manifest written to  : {manifest_path}")
    print(f"Legend written to    : {legend_path}")
    print("=" * 60)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Split coregistered ROI cell tables into smaller mosaic tile subsections with overlap support."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input GUI ROI Parquet/CSV file")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory to save tile sub-ROIs")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output filenames (default: input file stem)")
    parser.add_argument("--tile-dir", "-d", default=None, help="Directory of raw unstitched TIFF tiles (for auto dimension & name discovery)")
    parser.add_argument("--grid-rows", "-r", type=int, default=None, help="Number of mosaic grid rows")
    parser.add_argument("--grid-cols", "-c", type=int, default=None, help="Number of mosaic grid columns")
    parser.add_argument("--tile-height", type=int, default=None, help="Individual tile height in pixels (e.g. 1024)")
    parser.add_argument("--tile-width", type=int, default=None, help="Individual tile width in pixels (e.g. 1024)")
    parser.add_argument("--overlap-percent", type=float, default=5.0, help="Overlap percentage between adjacent tiles (e.g. 5.0)")
    parser.add_argument("--order", choices=["snake", "raster"], default="snake", help="Tile acquisition order (default: snake)")
    parser.add_argument("--coord-mode", choices=["local", "global", "both"], default="both", help="Primary coordinate system for Vertices_JSON/WKT (default: both columns included, primary in local)")
    parser.add_argument("--crop-mode", choices=["intersect", "centroid", "strict"], default="intersect", help="Tile boundary assignment mode (default: intersect)")
    parser.add_argument("--clip-vertices", action="store_true", default=False, help="Clip polygon vertices to exact tile bounding box [0, W] x [0, H]")
    parser.add_argument("--sample-id", default="1", help="Sample ID for metadata")
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="both", help="Output file format (default: both)")
    parser.add_argument("--cohesive-name", default=None, help="Base filename for cohesive output (default: {output_prefix}_cohesive_rois)")
    parser.add_argument("--save-individual", action="store_true", default=False, help="Also save individual parquet/csv files for each tile (default: False)")
    parser.add_argument("--no-cohesive", action="store_true", default=False, help="Disable cohesive output export")
    parser.add_argument("--stitched-image", default=None, help="Path to stitched mosaic TIFF image (for reading exact canvas shape)")
    parser.add_argument("--target-canvas-shape", default=None, help="Target canvas shape as 'height,width' (e.g. '2460,3434')")
    parser.add_argument("--generate-masks", action="store_true", default=False, help="Generate per-tile labeled mask TIFFs")
    parser.add_argument("--pixel-size", type=float, default=1.0, help="Pixel size in microns for mask metadata")
    parser.add_argument("--metadata-dict", default=None, help="JSON file or string with {label: integer_id} mappings")
    parser.add_argument("--metadata-list", default=None, help="JSON file, text file, or list of ordered label names")

    args = parser.parse_args()

    target_canvas_shape = None
    if args.target_canvas_shape:
        parts = [int(p.strip()) for p in args.target_canvas_shape.split(",") if p.strip()]
        if len(parts) == 2:
            target_canvas_shape = (parts[0], parts[1])

    split_roi_table_into_mosaic_tiles(
        input_path=args.input,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        tile_dir=args.tile_dir,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        tile_height=args.tile_height,
        tile_width=args.tile_width,
        overlap_percent=args.overlap_percent,
        order=args.order,
        stitched_image=args.stitched_image,
        target_canvas_shape=target_canvas_shape,
        coord_mode=args.coord_mode,
        crop_mode=args.crop_mode,
        clip_vertices=args.clip_vertices,
        sample_id=args.sample_id,
        cohesive_name=args.cohesive_name,
        save_cohesive=not args.no_cohesive,
        save_individual=args.save_individual,
        export_format=args.format,
        generate_masks=args.generate_masks,
        pixel_size_um=args.pixel_size,
        metadata_dict=args.metadata_dict,
        metadata_list=args.metadata_list,
    )


if __name__ == "__main__":
    main()
