#!/usr/bin/env python3
"""
roi_converter.py
================
Standardized, high-speed converter to generate GUI-compatible ROI Parquet/CSV tables
for both multimodal_gui and hsi_machine_learning pipelines.

Key Principles:
1. Canonical, non-redundant schema:
   - Uses `FOV_Number` (and legacy alias `ROI_Number`) to distinguish multiple sub-regions/FOVs per Sample_ID.
   - Retains `cell_id` for downstream Xenium/single-cell compatibility.
   - Removes redundant columns (e.g. Fluorescence_Label, duplicate grouping names, Region_Index).
   - Automatically extracts and saves `pixel_size_um`, `Image_Width`, `Image_Height` if an attached image is provided.
2. Two Standard Modes:
   - `from-mask`: High-speed multi-slice / multi-channel OME-TIFF mask -> GUI ROIs using OpenCV contours.
   - `from-cells`: Direct single-step projection of raw per-vertex tabular segmentation + metadata -> GUI ROIs
                   (eliminates slow intermediate 300k+ row per-vertex parquet files).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import tifffile
import cv2


# ==============================================================================
# 1. IMAGE METADATA HELPERS
# ==============================================================================

def extract_image_metadata(image_path: Optional[str]) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Extracts pixel_size_um, width, height from TIFF, OME-TIFF, or OIR image files.
    Returns (pixel_size_um, width, height).
    """
    if not image_path or not os.path.exists(image_path):
        return None, None, None

    pixel_size_um = None
    width = None
    height = None

    safe_path = os.path.abspath(image_path)
    if os.name == 'nt' and not safe_path.startswith('\\\\?\\'):
        safe_path = '\\\\?\\' + safe_path

    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".oir":
        try:
            import oirfile
            with oirfile.OirFile(safe_path) as oir:
                coord_scales = getattr(oir, "coord_scales", {}) or {}
                if "X" in coord_scales:
                    try:
                        pixel_size_um = float(coord_scales["X"])
                    except (TypeError, ValueError):
                        pass
                shape = oir.shape
                if len(shape) >= 2:
                    height = int(shape[-2])
                    width = int(shape[-1])
            return pixel_size_um, width, height
        except Exception as e:
            print(f"[roi_converter] Note: Could not read OIR metadata from {image_path}: {e}")

    try:
        with open(safe_path, 'rb') as f:
            with tifffile.TiffFile(f) as tif:
                page = tif.pages[0]
                height = int(page.shape[0])
                width = int(page.shape[1])

                if tif.is_imagej and page.resolution and page.resolution[1] != 0:
                    pixel_size_um = float(1.0 / page.resolution[1])
                elif page.resolution and page.resolution[0] not in (0, None):
                    res_x = page.resolution[0]
                    res_val = res_x[0] / res_x[1] if isinstance(res_x, (tuple, list)) else float(res_x)
                    unit = getattr(page, "resolutionunit", 2)
                    if res_val > 0:
                        if unit == 2:  # inch -> um
                            pixel_size_um = float(25400.0 / res_val)
                        elif unit == 3:  # cm -> um
                            pixel_size_um = float(10000.0 / res_val)
                        else:
                            pixel_size_um = float(1.0 / res_val)
    except Exception as e:
        print(f"[roi_converter] Note: Could not read TIFF metadata from {image_path}: {e}")

    return pixel_size_um, width, height


def compute_polygon_area(pts: np.ndarray) -> float:
    """Compute polygon area using Shoelace formula."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


# ==============================================================================
# 2. PARADIGM A: MASK TO GUI ROIS
# ==============================================================================

def mask_to_gui_rois(
    mask_path: str,
    output_path: str,
    sample_id: str = "Sample_1",
    fov_number: Union[str, int] = "1",
    image_path: Optional[str] = None,
    channel_labels: Optional[Union[Dict[int, str], List[str]]] = None,
    min_instance_size: int = 10,
    approx_epsilon: float = 0.8,
    export_csv: bool = True
) -> pd.DataFrame:
    """
    Extracts closed polygon contours from multi-slice binary mask OME-TIFF directly into
    a standard GUI-compatible Parquet (and optional CSV).
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Extract target image metadata if provided
    img_pixel_size, img_w, img_h = extract_image_metadata(image_path)
    image_name = os.path.basename(image_path) if image_path else os.path.basename(mask_path)

    records = []
    roi_counter = 0

    with tifffile.TiffFile(mask_path) as tif:
        num_pages = len(tif.pages)
        # Determine 2D vs 3D / multi-channel
        first_shape = tif.pages[0].shape
        mask_h, mask_w = first_shape[-2], first_shape[-1]
        
        # Use image dimensions if available, otherwise fall back to mask canvas dimensions
        final_w = img_w if img_w is not None else mask_w
        final_h = img_h if img_h is not None else mask_h

        # Resolve channel labels
        label_map = {}
        if isinstance(channel_labels, list):
            label_map = {i: lbl for i, lbl in enumerate(channel_labels)}
        elif isinstance(channel_labels, dict):
            label_map = channel_labels

        for page_idx, page in enumerate(tif.pages):
            data = page.asarray()
            # If page is 3D (C, Y, X), iterate channels
            if data.ndim == 3:
                slices = [(c, data[c]) for c in range(data.shape[0])]
            else:
                slices = [(page_idx, data)]

            for ch_idx, sl in slices:
                label_name = label_map.get(ch_idx, f"Class_{ch_idx}")
                bin_mask = (sl > 0).astype(np.uint8)
                if not np.any(bin_mask):
                    continue

                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                ch_inst = 0

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_instance_size:
                        continue

                    approx = cv2.approxPolyDP(cnt, approx_epsilon, True)
                    pts = approx.reshape(-1, 2)
                    if len(pts) < 3:
                        continue

                    # Spatial moments centroid
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = float(M['m10'] / M['m00'])
                        cy = float(M['m01'] / M['m00'])
                    else:
                        cx = float(pts[:, 0].mean())
                        cy = float(pts[:, 1].mean())

                    # Close polygon
                    if not np.allclose(pts[0], pts[-1]):
                        pts_closed = np.vstack([pts, pts[0]])
                    else:
                        pts_closed = pts

                    roi_counter += 1
                    ch_inst += 1

                    wkt = "POLYGON ((" + ", ".join(f"{x:.1f} {y:.1f}" for x, y in pts_closed) + "))"

                    rec = {
                        "UID": f"{sample_id}_{fov_number}_{label_name}_{ch_inst}",
                        "Sample_ID": str(sample_id),
                        "FOV_Number": fov_number,
                        "ROI_Number": fov_number,  # Legacy alias
                        "ROI_instance": ch_inst,
                        "Label": label_name,
                        "Vertices_JSON": json.dumps(pts_closed.tolist()),
                        "Centroid_X": round(cx, 2),
                        "Centroid_Y": round(cy, 2),
                        "Area_px": int(round(area)),
                        "Geometry_WKT": wkt,
                        "Image_Name": image_name,
                        "Image_Width": final_w,
                        "Image_Height": final_h,
                    }
                    if img_pixel_size is not None:
                        rec["pixel_size_um"] = round(float(img_pixel_size), 4)

                    records.append(rec)

    df_rois = pd.DataFrame(records)
    out_base = os.path.splitext(output_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    df_rois.to_parquet(f"{out_base}.parquet", index=False)
    if export_csv:
        df_rois.to_csv(f"{out_base}.csv", index=False)

    print(f"[roi_converter] Extracted {len(df_rois):,} ROIs from mask -> {out_base}.parquet")
    return df_rois


# ==============================================================================
# 3. PARADIGM B: TABULAR CELL VERTICES + METADATA TO GUI ROIS (SINGLE STEP)
# ==============================================================================

def cells_to_gui_rois(
    vertices_path: str,
    output_path: str,
    metadata_path: Optional[str] = None,
    i2r_json_path: Optional[str] = None,
    image_path: Optional[str] = None,
    sample_id: str = "Sample_1",
    fov_number: Union[str, int] = "1",
    label_col: Optional[str] = None,
    cell_col: Optional[str] = None,
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    flip_y: bool = True,
    flip_y_height: Optional[float] = None,
    min_vertices: int = 3,
    use_inverse: bool = False,
    clip_to_canvas: bool = True,
    export_csv: bool = False
) -> pd.DataFrame:
    """
    Direct single-pass conversion of raw tabular cell vertices + cell metadata into
    a standard GUI-compatible Parquet table.
    
    Eliminates intermediate 300k+ row per-vertex parquet files by transforming,
    grouping, and serializing directly into the final per-cell table.
    """
    if not os.path.exists(vertices_path):
        raise FileNotFoundError(f"Vertices file not found: {vertices_path}")

    # Load raw vertices
    print(f"[roi_converter] Loading cell vertices from {vertices_path}...")
    if vertices_path.lower().endswith((".parquet", ".pq")):
        df_v = pd.read_parquet(vertices_path)
    else:
        df_v = pd.read_csv(vertices_path)

    # Detect vertex column names
    col_map = {c.lower(): c for c in df_v.columns}
    resolved_cell = cell_col or col_map.get("cell_id") or col_map.get("cell") or col_map.get("id")
    resolved_x = x_col or col_map.get("vertex_x") or col_map.get("x") or col_map.get("x_roi")
    resolved_y = y_col or col_map.get("vertex_y") or col_map.get("y") or col_map.get("y_roi")

    if not (resolved_cell and resolved_x and resolved_y):
        raise ValueError(f"Could not identify cell_id, x, y columns in vertices file: {df_v.columns.tolist()}")

    # Merge cell metadata if provided
    meta_df = None
    if metadata_path and os.path.exists(metadata_path):
        print(f"[roi_converter] Loading metadata from {metadata_path}...")
        meta_df = pd.read_parquet(metadata_path) if metadata_path.lower().endswith((".parquet", ".pq")) else pd.read_csv(metadata_path)
        m_col_map = {c.lower(): c for c in meta_df.columns}
        m_cell = m_col_map.get("cell_id") or m_col_map.get("cell") or m_col_map.get("id")
        if m_cell:
            meta_df = meta_df.drop_duplicates(subset=[m_cell])
            # Ensure cell_id string matching
            df_v[resolved_cell] = df_v[resolved_cell].astype(str)
            meta_df[m_cell] = meta_df[m_cell].astype(str)
            df_v = df_v.merge(meta_df, left_on=resolved_cell, right_on=m_cell, how="left")

    # Detect label column
    resolved_label = label_col
    if not resolved_label:
        candidates = ["cell_type", "cluster", "annotation", "levelkid_refined", "group", "class", "label"]
        for cand in candidates:
            if cand in [c.lower() for c in df_v.columns]:
                resolved_label = next(c for c in df_v.columns if c.lower() == cand)
                break
    if not resolved_label:
        df_v["_default_label"] = "Cell"
        resolved_label = "_default_label"

    # Extract target image metadata if available
    img_pixel_size, img_w, img_h = extract_image_metadata(image_path)
    image_name = os.path.basename(image_path) if image_path else ""

    # Parse and apply transformation matrix if provided
    pts_raw = df_v[[resolved_x, resolved_y]].values.astype(np.float64)
    canvas_w = img_w
    canvas_h = img_h

    if i2r_json_path and os.path.exists(i2r_json_path):
        print(f"[roi_converter] Applying registration transform from {i2r_json_path}...")
        with open(i2r_json_path, 'r', encoding='utf-8') as f:
            t_data = json.load(f)

        if use_inverse and 'matrix_xy_px_inv' in t_data:
            mat = np.array(t_data['matrix_xy_px_inv'], dtype=np.float64)
        elif 'matrix_xy_px' in t_data:
            mat = np.array(t_data['matrix_xy_px'], dtype=np.float64)
            if use_inverse:
                mat = np.linalg.inv(mat)
        else:
            raise ValueError(f"No suitable matrix_xy_px found in {i2r_json_path}")

        # Homogeneous transformation
        homo = np.hstack([pts_raw, np.ones((len(pts_raw), 1), dtype=np.float64)])
        pts_trans = (homo @ mat.T)[:, :2]
        pts_raw = pts_trans

        # Canvas bounds from i2r if not from image
        if canvas_w is None or canvas_h is None:
            fixed = t_data.get('fixed_paths', [{}])[0]
            if 'image_shape' in fixed:
                canvas_h = int(fixed['image_shape'][0])
                canvas_w = int(fixed['image_shape'][1])

    # Vertical flip if specified (e.g., matching SRS coordinate space)
    if flip_y:
        h_flip = flip_y_height if flip_y_height is not None else (canvas_h if canvas_h else float(np.max(pts_raw[:, 1])))
        pts_raw[:, 1] = h_flip - pts_raw[:, 1]

    # Group contiguous cell boundaries
    cell_ids = df_v[resolved_cell].astype(str).values
    labels = df_v[resolved_label].astype(str).values

    change_idx = np.where(cell_ids[:-1] != cell_ids[1:])[0] + 1
    splits_pts = np.split(pts_raw, change_idx)
    splits_cid = np.split(cell_ids, change_idx)
    splits_lbl = np.split(labels, change_idx)

    # Optional extra metadata columns to preserve per cell
    skip_cols = {resolved_cell, resolved_x, resolved_y, resolved_label, "_default_label"}
    extra_cols = [c for c in df_v.columns if c not in skip_cols and not c.startswith("_")]
    first_row_indices = np.insert(change_idx, 0, 0)
    extra_metadata_sample = df_v.iloc[first_row_indices][extra_cols].to_dict(orient="records") if extra_cols else []

    records = []
    label_instance_counts = {}

    for idx, (pts, c_arr, l_arr) in enumerate(zip(splits_pts, splits_cid, splits_lbl)):
        if len(pts) < min_vertices:
            continue

        cid = str(c_arr[0])
        lbl = str(l_arr[0])
        label_instance_counts[lbl] = label_instance_counts.get(lbl, 0) + 1
        inst_num = label_instance_counts[lbl]

        # Enforce polygon closure
        if not np.allclose(pts[0], pts[-1]):
            pts_closed = np.vstack([pts, pts[0]])
        else:
            pts_closed = pts

        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))

        # Filter cells completely out of canvas if clip_to_canvas is enabled
        if clip_to_canvas and canvas_w is not None and canvas_h is not None:
            if cx < 0 or cx >= canvas_w or cy < 0 or cy >= canvas_h:
                continue

        area = compute_polygon_area(pts)
        pts_list = pts_closed.round(2).tolist()
        wkt = "POLYGON ((" + ", ".join(f"{x:.2f} {y:.2f}" for x, y in pts_closed) + "))"

        rec = {
            "UID": f"{sample_id}_{fov_number}_{lbl}_{inst_num}",
            "Sample_ID": str(sample_id),
            "FOV_Number": fov_number,
            "ROI_Number": fov_number,  # Legacy alias
            "ROI_instance": inst_num,
            "Label": lbl,
            "cell_id": cid,  # Retained for downstream single-cell legacy compatibility
            "Vertices_JSON": json.dumps(pts_list),
            "Centroid_X": round(cx, 2),
            "Centroid_Y": round(cy, 2),
            "Area_px": round(area, 2),
            "Geometry_WKT": wkt,
            "Image_Name": image_name,
        }
        if canvas_w is not None:
            rec["Image_Width"] = canvas_w
        if canvas_h is not None:
            rec["Image_Height"] = canvas_h
        if img_pixel_size is not None:
            rec["pixel_size_um"] = round(float(img_pixel_size), 4)

        # Merge preserved cell metadata
        if extra_metadata_sample and idx < len(extra_metadata_sample):
            for k, v in extra_metadata_sample[idx].items():
                if k not in rec:
                    rec[k] = v

        records.append(rec)

    df_out = pd.DataFrame(records)
    out_base = os.path.splitext(output_path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    df_out.to_parquet(f"{out_base}.parquet", index=False)
    if export_csv:
        df_out.to_csv(f"{out_base}.csv", index=False)

    print(f"[roi_converter] Generated {len(df_out):,} aggregated ROIs -> {out_base}.parquet")
    return df_out


# ==============================================================================
# 4. CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standardized GUI ROI Parquet generator for masks and cell tables."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: from-mask
    p_mask = subparsers.add_parser("from-mask", help="Convert multi-slice mask TIFF to GUI ROIs")
    p_mask.add_argument("--mask", required=True, help="Input multi-slice OME-TIFF mask")
    p_mask.add_argument("--output", required=True, help="Output .parquet file path")
    p_mask.add_argument("--sample-id", default="Sample_1", help="Sample ID")
    p_mask.add_argument("--fov-number", default="1", help="FOV / sub-region identifier (e.g. 1 or roi1)")
    p_mask.add_argument("--image", default=None, help="Attached reference image (to extract pixel_size_um, W, H)")
    p_mask.add_argument("--min-size", type=int, default=10, help="Minimum pixel area for instance")
    p_mask.add_argument("--approx-eps", type=float, default=0.8, help="OpenCV approxPolyDP epsilon")
    p_mask.add_argument("--csv", action="store_true", help="Also export CSV")

    # Subcommand: from-cells
    p_cells = subparsers.add_parser("from-cells", help="Convert raw per-vertex cell table + metadata to GUI ROIs")
    p_cells.add_argument("--vertices", required=True, help="Cell boundary vertices (parquet or csv)")
    p_cells.add_argument("--output", required=True, help="Output .parquet file path")
    p_cells.add_argument("--metadata", default=None, help="Cell metadata table (parquet or csv)")
    p_cells.add_argument("--i2r", default=None, help="Registration .i2r.json matrix file")
    p_cells.add_argument("--image", default=None, help="Attached canvas image")
    p_cells.add_argument("--sample-id", default="Sample_1", help="Sample ID")
    p_cells.add_argument("--fov-number", default="1", help="FOV / sub-region identifier (e.g. 1 or roi1)")
    p_cells.add_argument("--label-col", default=None, help="Column name to use for Label")
    p_cells.add_argument("--cell-col", default=None, help="Column name for cell_id")
    p_cells.add_argument("--x-col", default=None, help="Column name for x coordinate")
    p_cells.add_argument("--y-col", default=None, help="Column name for y coordinate")
    p_cells.add_argument("--no-flip-y", dest="flip_y", action="store_false", help="Disable vertical y flip")
    p_cells.add_argument("--inverse", action="store_true", help="Use inverse transform matrix")
    p_cells.add_argument("--csv", action="store_true", help="Also export CSV")

    args = parser.parse_args()

    if args.subcommand == "from-mask":
        mask_to_gui_rois(
            mask_path=args.mask,
            output_path=args.output,
            sample_id=args.sample_id,
            fov_number=args.fov_number,
            image_path=args.image,
            min_instance_size=args.min_size,
            approx_epsilon=args.approx_eps,
            export_csv=args.csv
        )
    elif args.subcommand == "from-cells":
        cells_to_gui_rois(
            vertices_path=args.vertices,
            output_path=args.output,
            metadata_path=args.metadata,
            i2r_json_path=args.i2r,
            image_path=args.image,
            sample_id=args.sample_id,
            fov_number=args.fov_number,
            label_col=args.label_col,
            cell_col=args.cell_col,
            x_col=args.x_col,
            y_col=args.y_col,
            flip_y=args.flip_y,
            use_inverse=args.inverse,
            export_csv=args.csv
        )


if __name__ == "__main__":
    main()
