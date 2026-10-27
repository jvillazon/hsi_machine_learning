#!/usr/bin/env python3
"""
transform_and_crop_rois.py
==========================
Step 3 of Xenium / Spatial ROI Pipeline:
Applies affine coordinate transformations from a transform JSON file (e.g. *.i2r.json from
SimpleITK/elastix image-to-registration) using `core.roi_coordinate_transformer`.

Crops and filters cells to the target ROI bounds, flips the Y axis (to match microscope orientation),
and exports the transformed polygons in the exact format compatible with `1_regions.csv`
and `multimodal_napari_gui/run_app.py` (load_napari_rois).

Optionally generates the transformed ROI labeled mask TIFF.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
import numpy as np
import pandas as pd
import tifffile

try:
    import cv2
except ImportError:
    cv2 = None

# Ensure hsi_machine_learning root is in sys.path to import core
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from core.roi_coordinate_transformer import (
        TransformationLoader,
        CoordinateTransformer,
        CellDataMapper,
        apply_roi_transform_from_json,
    )
except ImportError:
    TransformationLoader = None
    CoordinateTransformer = None


def compute_polygon_area(pts: np.ndarray) -> float:
    """Compute 2D polygon area using the Shoelace formula."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))





def load_dataframe(path: str) -> pd.DataFrame:
    """Load DataFrame from Parquet or CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_transform_metadata(transform_json_path: str, use_inverse: bool = False) -> Tuple[np.ndarray, Dict[str, int], float]:
    """
    Extract affine matrix, ROI bounds, and pixel size from transform JSON.
    Supports both forward and inverse transformations (using matrix_xy_px_inv or np.linalg.inv),
    and moving/fixed image bounds matching core.roi_coordinate_transformer.
    """
    with open(transform_json_path, "r") as f:
        data = json.load(f)

    # 1. Extract 3x3 matrix (forward or inverse)
    if use_inverse:
        if "matrix_xy_px_inv" in data:
            matrix = np.array(data["matrix_xy_px_inv"], dtype=np.float64)
            print("Loaded inverse matrix from 'matrix_xy_px_inv'.")
        elif "matrix_inv" in data:
            matrix = np.array(data["matrix_inv"], dtype=np.float64)
            print("Loaded inverse matrix from 'matrix_inv'.")
        elif "matrix_xy_px" in data:
            matrix = np.linalg.inv(np.array(data["matrix_xy_px"], dtype=np.float64))
            print("Calculated inverse matrix via np.linalg.inv(matrix_xy_px).")
        elif "matrix" in data:
            matrix = np.linalg.inv(np.array(data["matrix"], dtype=np.float64))
            print("Calculated inverse matrix via np.linalg.inv(matrix).")
        elif "affine_matrix" in data:
            matrix = np.linalg.inv(np.array(data["affine_matrix"], dtype=np.float64))
            print("Calculated inverse matrix via np.linalg.inv(affine_matrix).")
        else:
            raise ValueError(f"No affine matrix found to invert in {transform_json_path}")
    else:
        if "matrix_xy_px" in data:
            matrix = np.array(data["matrix_xy_px"], dtype=np.float64)
        elif "matrix" in data:
            matrix = np.array(data["matrix"], dtype=np.float64)
        elif "affine_matrix" in data:
            matrix = np.array(data["affine_matrix"], dtype=np.float64)
        else:
            raise ValueError(f"No affine matrix ('matrix_xy_px') found in {transform_json_path}")

    # 2. Extract bounds (moving bounds for inverse, fixed bounds for forward)
    if use_inverse:
        if "moving_paths" in data and len(data["moving_paths"]) > 0:
            shape = data["moving_paths"][0].get("image_shape", [1024, 1024])
            bounds = {"height": int(shape[0]), "width": int(shape[1])}
            pixel_size = float(data["moving_paths"][0].get("pixel_size_um", 1.0))
        elif "moving_bounds" in data:
            bounds = {"height": int(data["moving_bounds"]["height"]), "width": int(data["moving_bounds"]["width"])}
            pixel_size = float(data.get("pixel_size_um", 1.0))
        elif "fixed_paths" in data and len(data["fixed_paths"]) > 0:
            shape = data["fixed_paths"][0].get("image_shape", [1024, 1024])
            bounds = {"height": int(shape[0]), "width": int(shape[1])}
            pixel_size = float(data["fixed_paths"][0].get("pixel_size_um", 1.0))
        else:
            bounds = {"height": 1024, "width": 1024}
            pixel_size = 1.0
    else:
        if "fixed_paths" in data and len(data["fixed_paths"]) > 0:
            shape = data["fixed_paths"][0].get("image_shape", [1024, 1024])
            bounds = {"height": int(shape[0]), "width": int(shape[1])}
            pixel_size = float(data["fixed_paths"][0].get("pixel_size_um", 1.0))
        elif "roi_bounds" in data:
            bounds = {"height": int(data["roi_bounds"]["height"]), "width": int(data["roi_bounds"]["width"])}
            pixel_size = float(data.get("pixel_size_um", 1.0))
        elif "image_shape" in data:
            bounds = {"height": int(data["image_shape"][0]), "width": int(data["image_shape"][1])}
            pixel_size = float(data.get("pixel_size_um", 1.0))
        else:
            bounds = {"height": 1024, "width": 1024}
            pixel_size = 1.0

    return matrix, bounds, pixel_size


def auto_detect_alignment_offset(
    input_path: str,
    transform_json: str,
    reference_mask: str = "",
    df_raw: Optional[pd.DataFrame] = None,
    active_x: str = "vertex_x",
    active_y: str = "vertex_y",
) -> Tuple[float, float]:
    """
    Auto-detect the (X, Y) pixel offset between raw slide coordinates
    and the cropped/registered canvas recorded in transform_json.
    """
    try:
        with open(transform_json, "r") as f:
            t_data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read transform JSON for offset detection: {e}")
        return 0.0, 0.0

    fixed_paths = t_data.get("fixed_paths", [])
    fixed_path = fixed_paths[0].get("path", "") if len(fixed_paths) > 0 else ""

    # 1. Try image-based correlation if fixed image and reference mask exist
    mask_to_use = reference_mask
    if not mask_to_use or not os.path.exists(mask_to_use):
        inp_dir = os.path.dirname(os.path.abspath(input_path))
        inp_base = os.path.basename(input_path).lower()
        candidates = [
            os.path.join(inp_dir, f) for f in os.listdir(inp_dir)
            if f.lower().endswith((".tif", ".tiff")) and "mask" in f.lower()
        ]
        # Prefer candidate that shares the sample prefix and encompasses the raw coordinates
        stem_parts = inp_base.split("_")
        sample_prefix = "_".join(stem_parts[:3]) if len(stem_parts) >= 3 else stem_parts[0]
        raw_max_x = float(df_raw[active_x].max()) if df_raw is not None and active_x in df_raw.columns else 0.0
        raw_max_y = float(df_raw[active_y].max()) if df_raw is not None and active_y in df_raw.columns else 0.0
        scored = []
        for c in candidates:
            c_base = os.path.basename(c).lower()
            try:
                with tifffile.TiffFile(c) as tif:
                    m_shape = tif.series[0].shape
                if raw_max_x > 1000 and (m_shape[1] < raw_max_x or m_shape[0] < raw_max_y):
                    continue
            except Exception:
                pass
            score = 10 if sample_prefix in c_base else 0
            score += sum(1 for p in stem_parts if p in c_base)
            scored.append((score, c))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            mask_to_use = scored[0][1]

    if os.path.exists(fixed_path) and mask_to_use and os.path.exists(mask_to_use):
        try:
            print(f"Auto-detecting alignment offset between:")
            print(f"  Reference Mask: {mask_to_use}")
            print(f"  Fixed Canvas  : {fixed_path}")
            from scipy.signal import fftconvolve

            mask_img = tifffile.imread(mask_to_use)
            fixed_img = tifffile.imread(fixed_path)

            if mask_img.ndim > 2:
                mask_img = mask_img[0]
            if fixed_img.ndim > 2:
                fixed_img = fixed_img[0]

            ds = 8
            m_small = (mask_img[::ds, ::ds] > 0).astype(np.float32)
            f_small = (fixed_img[::ds, ::ds] > 0).astype(np.float32)

            if m_small.shape[0] >= f_small.shape[0] and m_small.shape[1] >= f_small.shape[1]:
                corr = fftconvolve(m_small, f_small[::-1, ::-1], mode="valid")
                best_y, best_x = np.unravel_index(np.argmax(corr), corr.shape)
                y_coarse = int(best_y * ds)
                x_coarse = int(best_x * ds)

                best_score = -1
                best_yx = (y_coarse, x_coarse)
                f_bool = fixed_img > 0
                max_y_search = min(mask_img.shape[0] - fixed_img.shape[0], y_coarse + 12)
                max_x_search = min(mask_img.shape[1] - fixed_img.shape[1], x_coarse + 12)
                for cy in range(max(0, y_coarse - 12), max_y_search + 1):
                    for cx in range(max(0, x_coarse - 12), max_x_search + 1):
                        sc = np.sum((mask_img[cy : cy + fixed_img.shape[0], cx : cx + fixed_img.shape[1]] > 0) & f_bool)
                        if sc > best_score:
                            best_score = sc
                            best_yx = (cy, cx)

                print(f"Auto-detected canvas offset: X = {best_yx[1]}, Y = {best_yx[0]}")
                return float(best_yx[1]), float(best_yx[0])
        except Exception as e:
            print(f"Note: Correlation offset detection skipped ({e}). Falling back to coordinate bounds.")

    # 2. Fallback: coordinate bounds offset (min - padding)
    if df_raw is not None and active_x in df_raw.columns and active_y in df_raw.columns:
        x_min = float(df_raw[active_x].min())
        y_min = float(df_raw[active_y].min())
        if x_min > 500:
            print(f"Estimated offset from coordinate bounds: X = {x_min - 10:.2f}, Y = {y_min - 10:.2f}")
            return float(x_min - 10), float(y_min - 10)

    return 0.0, 0.0


def transform_and_crop_rois(
    input_path: str,
    transform_json: str,
    output_prefix: str,
    cell_col: str = "cell_id",
    label_col: str = "group",
    x_col: str = "x",
    y_col: str = "y",
    offset_x: Union[float, str] = "auto",
    offset_y: Union[float, str] = "auto",
    reference_mask: str = "",
    flip_y: bool = True,
    use_inverse: bool = False,
    crop_mode: str = "intersect",  # 'intersect', 'strict', 'none'
    sample_id: str = "1",
    roi_name: str = "",
    image_path: str = "",
    generate_mask: bool = True,
    export_format: str = "both",  # 'parquet', 'csv', 'both'
    split_mosaic: bool = False,
    split_tile_dir: Optional[str] = None,
    split_grid_rows: Optional[int] = None,
    split_grid_cols: Optional[int] = None,
    split_tile_height: Optional[int] = None,
    split_tile_width: Optional[int] = None,
    split_overlap_percent: float = 5.0,
    split_order: str = "snake",
    split_coord_mode: str = "both",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply transform JSON to cell coordinates, crop to ROI bounds, and export
    GUI-compatible Parquet/CSV and optional mask TIFF. Supports inverse transforms.
    """
    t0 = time.time()
    print("=" * 70)
    print("STEP 3: TRANSFORM & CROP CELLS TO TARGET ROI")
    print("=" * 70)
    print(f"Input Cells    : {input_path}")
    print(f"Transform JSON : {transform_json}")
    print(f"Output Prefix  : {output_prefix}")
    print(f"Use Inverse    : {use_inverse}")
    print(f"Offset X/Y     : {offset_x} / {offset_y}")

    matrix, bounds, pixel_size = parse_transform_metadata(transform_json, use_inverse=use_inverse)
    roi_h, roi_w = bounds["height"], bounds["width"]
    print(f"Target ROI Dimensions : {roi_w} x {roi_h} px (pixel size: {pixel_size} um)")

    # 1. Load input table
    df_raw = load_dataframe(input_path)

    # If input is aggregated ROI table (has Vertices_JSON), unpack to per-vertex
    if "Vertices_JSON" in df_raw.columns:
        print(f"Unpacking {len(df_raw):,} aggregated ROIs into vertex coordinates...")
        vertex_rows = []
        c_lbl_col = "Label" if "Label" in df_raw.columns else ("Fluorescence_Label" if "Fluorescence_Label" in df_raw.columns else label_col)
        c_id_col = "cell_id" if "cell_id" in df_raw.columns else ("UID" if "UID" in df_raw.columns else cell_col)

        for _, r in df_raw.iterrows():
            cid = str(r[c_id_col])
            lbl = str(r[c_lbl_col])
            v_raw = r["Vertices_JSON"]
            pts = json.loads(v_raw) if isinstance(v_raw, str) else v_raw
            for pt in pts:
                vertex_rows.append({"cell_id": cid, "group": lbl, "x": float(pt[0]), "y": float(pt[1])})
        df_vertices = pd.DataFrame(vertex_rows)
        active_x = "x"
        active_y = "y"
        active_cell = "cell_id"
        active_label = "group"
    else:
        df_vertices = df_raw.copy()
        # Resolve columns
        active_x = x_col if x_col in df_vertices.columns else ("vertex_x" if "vertex_x" in df_vertices.columns else "x")
        active_y = y_col if y_col in df_vertices.columns else ("vertex_y" if "vertex_y" in df_vertices.columns else "y")
        active_cell = cell_col if cell_col in df_vertices.columns else ("cell_ID" if "cell_ID" in df_vertices.columns else "cell_id")
        active_label = label_col if label_col in df_vertices.columns else ("Label" if "Label" in df_vertices.columns else "group")

    # 2. Transform coordinates (affine 3x3 matrix)
    print("Applying affine transformation to coordinates...")
    coords = df_vertices[[active_x, active_y]].values.astype(np.float64)

    # Determine and apply coordinate offset
    raw_min_x = float(coords[:, 0].min())
    raw_min_y = float(coords[:, 1].min())

    if str(offset_x).lower() == "auto" or str(offset_y).lower() == "auto":
        if raw_min_x > roi_w or raw_min_y > roi_h:
            print(f"Input coordinates (min_x={raw_min_x:.1f}, min_y={raw_min_y:.1f}) exceed ROI bounds ({roi_w}x{roi_h}).")
            auto_x, auto_y = auto_detect_alignment_offset(
                input_path=input_path,
                transform_json=transform_json,
                reference_mask=reference_mask,
                df_raw=df_vertices,
                active_x=active_x,
                active_y=active_y,
            )
            parsed_off_x = auto_x if str(offset_x).lower() == "auto" else float(offset_x)
            parsed_off_y = auto_y if str(offset_y).lower() == "auto" else float(offset_y)
        else:
            parsed_off_x = 0.0 if str(offset_x).lower() == "auto" else float(offset_x)
            parsed_off_y = 0.0 if str(offset_y).lower() == "auto" else float(offset_y)
    else:
        parsed_off_x = float(offset_x)
        parsed_off_y = float(offset_y)

    if parsed_off_x != 0.0 or parsed_off_y != 0.0:
        print(f"Applying coordinate offset: X -= {parsed_off_x:.2f}, Y -= {parsed_off_y:.2f}")
        coords[:, 0] -= parsed_off_x
        coords[:, 1] -= parsed_off_y

    n_pts = len(coords)
    homogeneous = np.hstack([coords, np.ones((n_pts, 1), dtype=np.float64)])
    transformed_homo = (matrix @ homogeneous.T).T
    # Perspective divide by w (for projective/homography transforms; w=1.0 for affine)
    w_vec = transformed_homo[:, 2:3]
    w_safe = np.where(np.abs(w_vec) > 1e-12, w_vec, 1.0)
    transformed_coords = transformed_homo[:, :2] / w_safe

    df_vertices["x_roi"] = transformed_coords[:, 0]
    df_vertices["y_roi"] = transformed_coords[:, 1]

    # Flip Y axis if requested
    if flip_y:
        print("Inverting Y axis to match target coordinate system orientation...")
        df_vertices["y_roi"] = float(roi_h) - df_vertices["y_roi"]

    # 3. Filter and crop cells to ROI bounds using geometric polygon clipping
    print(f"Filtering and cropping cells to ROI bounds ({roi_w} x {roi_h}) with mode '{crop_mode}'...")
    c_ids = df_vertices[active_cell].astype(str).values
    lbls = df_vertices[active_label].astype(str).values
    vx = df_vertices["x_roi"].values
    vy = df_vertices["y_roi"].values

    change_idx = np.where(c_ids[:-1] != c_ids[1:])[0] + 1
    splits_xy = np.split(np.column_stack((vx, vy)), change_idx)
    splits_c = np.split(c_ids, change_idx)
    splits_l = np.split(lbls, change_idx)

    records = []
    label_counters = {}
    roi_id_str = roi_name if roi_name else Path(transform_json).stem
    valid_polys_for_mask = []

    for pts, c_arr, l_arr in zip(splits_xy, splits_c, splits_l):
        c_id = str(c_arr[0])
        lbl = str(l_arr[0])

        # 1. Un-loop if first and last vertices are identical, preventing duplicate zero-length edges
        pts_u = pts[:-1] if (len(pts) > 2 and np.allclose(pts[0], pts[-1])) else pts
        if len(pts_u) < 3:
            continue

        # 2. Filter / crop according to crop_mode
        if crop_mode == "strict":
            if (
                pts_u[:, 0].min() < 0.0
                or pts_u[:, 0].max() >= roi_w
                or pts_u[:, 1].min() < 0.0
                or pts_u[:, 1].max() >= roi_h
            ):
                continue
            pts_use = pts_u
        elif crop_mode == "intersect":
            # Keep cells with at least one vertex inside ROI bounds without clipping/clamping vertices
            if (
                pts_u[:, 0].max() <= 0.0
                or pts_u[:, 0].min() >= roi_w
                or pts_u[:, 1].max() <= 0.0
                or pts_u[:, 1].min() >= roi_h
            ):
                continue
            pts_use = pts_u
        elif crop_mode == "clip":
            # Sutherland-Hodgman box clipping against canvas
            from core.roi_transform import clip_polygon_to_box
            pts_clipped = clip_polygon_to_box(pts_u.tolist(), 0.0, 0.0, float(roi_w), float(roi_h))
            if len(pts_clipped) < 3:
                continue
            pts_use = np.array(pts_clipped, dtype=np.float64)
        else:  # 'none'
            pts_use = pts_u

        # 3. Safeguard against degenerate polygons
        area = compute_polygon_area(pts_use)
        if area < 0.5 or len(pts_use) < 3:
            continue

        label_counters[lbl] = label_counters.get(lbl, 0) + 1
        roi_inst = label_counters[lbl]
        idx = len(records) + 1

        pts_closed = np.vstack([pts_use, pts_use[0]])
        pts_list = pts_closed.round(3).tolist()
        wkt = "POLYGON ((" + ", ".join(f"{x:.3f} {y:.3f}" for x, y in pts_closed) + "))"
        cx = float(np.mean(pts_use[:, 0]))
        cy = float(np.mean(pts_use[:, 1]))

        records.append({
            "Sample_ID": sample_id,
            "ROI_Number": idx,
            "ROI_instance": roi_inst,
            "Region_Index": idx,
            "Label": lbl,
            "Fluorescence_Label": lbl,
            "Vertex_Count": len(pts_closed),
            "Geometry_WKT": wkt,
            "Vertices_JSON": json.dumps(pts_list),
            "Centroid_X": round(cx, 3),
            "Centroid_Y": round(cy, 3),
            "Area_px": round(area, 2),
            "UID": c_id,
            "cell_id": c_id,
            "Image_Name": roi_id_str,
            "Image_Path": image_path,
        })
        valid_polys_for_mask.append((lbl, pts_use))

    df_gui_rois = pd.DataFrame(records)
    total_cells_transformed = len(df_gui_rois)
    print(f"Retained {total_cells_transformed:,} valid 2D cells inside target ROI.")

    # 5. Export formatted files
    out_dir = os.path.dirname(os.path.abspath(output_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    saved_files = {}
    base = output_prefix
    if export_format in ("parquet", "both"):
        pq_path = f"{base}_transformed_rois.parquet"
        df_gui_rois.to_parquet(pq_path, index=False)
        print(f"Saved transformed ROI Parquet: {pq_path} ({os.path.getsize(pq_path)/1024/1024:.2f} MB)")
        saved_files["parquet"] = pq_path

    if export_format in ("csv", "both"):
        csv_path = f"{base}_transformed_rois.csv"
        df_gui_rois.to_csv(csv_path, index=False)
        print(f"Saved transformed ROI CSV    : {csv_path} ({os.path.getsize(csv_path)/1024/1024:.2f} MB)")
        saved_files["csv"] = csv_path

    # 6. Optional: Generate transformed ROI labeled mask TIFF
    if generate_mask:
        print(f"Generating transformed ROI labeled mask TIFF ({roi_w} x {roi_h})...")
        unique_lbls = sorted(df_gui_rois["Label"].unique().tolist())
        label_to_id = {lbl: i for i, lbl in enumerate(unique_lbls, start=1)}

        mask_tif_path = f"{base}_transformed_mask.tif"
        mask_arr = np.zeros((roi_h, roi_w), dtype=np.uint16)

        for lbl, pts_valid in valid_polys_for_mask:
            val = label_to_id.get(lbl, 0)
            if val > 0 and len(pts_valid) >= 3:
                pts_round = np.round(pts_valid).astype(np.int32)
                if cv2 is not None:
                    cv2.fillPoly(mask_arr, [pts_round], val)

        res_val = 1.0 / pixel_size if pixel_size > 0 else 1.0
        tifffile.imwrite(
            mask_tif_path,
            mask_arr,
            imagej=True,
            resolution=(res_val, res_val),
            resolutionunit=1,
            metadata={
                "axes": "YX",
                "unit": "micron",
                "spacing": pixel_size,
            },
        )
        print(f"Saved transformed mask TIFF: {mask_tif_path}")
        saved_files["mask_tif"] = mask_tif_path

        # Also save legend
        legend_path = f"{base}_transformed_legend.json"
        with open(legend_path, "w") as f:
            json.dump({
                "label_to_id": label_to_id,
                "id_to_label": {str(v): k for k, v in label_to_id.items()},
                "roi_bounds": bounds,
            }, f, indent=2)
        saved_files["legend"] = legend_path

    # 7. Optional: Split into mosaic subsections
    if split_mosaic:
        print("\nSplitting transformed ROI into mosaic subsections...")
        from task_specific.Xenium_profile.split_roi_mosaic_subsections import split_roi_table_into_mosaic_tiles

        split_out_dir = Path(output_prefix).parent / f"{Path(output_prefix).stem}_subsections"
        source_pq = saved_files.get("parquet") or (f"{base}_transformed_rois.parquet")

        split_manifest = split_roi_table_into_mosaic_tiles(
            input_path=source_pq,
            output_dir=split_out_dir,
            output_prefix=Path(output_prefix).stem,
            tile_dir=split_tile_dir,
            grid_rows=split_grid_rows,
            grid_cols=split_grid_cols,
            tile_height=split_tile_height,
            tile_width=split_tile_width,
            overlap_percent=split_overlap_percent,
            order=split_order,
            coord_mode=split_coord_mode,
            crop_mode=crop_mode,
            sample_id=sample_id,
            export_format=export_format,
            generate_masks=generate_mask,
            pixel_size_um=pixel_size,
        )
        saved_files["split_manifest"] = split_manifest
        if "cohesive_output_files" in split_manifest:
            for fmt, path in split_manifest["cohesive_output_files"].items():
                saved_files[f"cohesive_{fmt}"] = path

    # 8. Export run configuration JSON
    cfg_suffix = "transform_config.json" if (output_prefix.endswith("-") or output_prefix.endswith("_")) else "_transform_config.json"
    cfg_path = f"{output_prefix}{cfg_suffix}"
    config_record = {
        "input_path": str(input_path),
        "transform_json": str(transform_json),
        "output_prefix": str(output_prefix),
        "cell_col": cell_col,
        "label_col": label_col,
        "x_col": x_col,
        "y_col": y_col,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "reference_mask": reference_mask,
        "flip_y": flip_y,
        "use_inverse": use_inverse,
        "crop_mode": crop_mode,
        "sample_id": sample_id,
        "roi_name": roi_name,
        "image_path": image_path,
        "generate_mask": generate_mask,
        "export_format": export_format,
        "roi_bounds": bounds,
        "pixel_size_um": pixel_size,
        "total_cells_retained": total_cells_transformed,
        "output_files": saved_files,
    }
    with open(cfg_path, "w") as f:
        json.dump(config_record, f, indent=2)
    print(f"Saved run configuration JSON : {cfg_path}")
    saved_files["transform_config"] = cfg_path

    print(f"Step 3 completed in {time.time() - t0:.2f}s.")
    return df_gui_rois, saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Transform and crop cell boundaries to target ROI based on transform JSON."
    )
    parser.add_argument("--input", "-i", required=True, help="Input cell file (merged vertices or labeled regions)")
    parser.add_argument("--transform-json", "-t", required=True, help="Path to transform JSON (*.i2r.json)")
    parser.add_argument("--output-prefix", "-o", required=True, help="Output prefix for saved files")
    parser.add_argument("--cell-col", default="cell_id", help="Cell ID column")
    parser.add_argument("--label-col", default="group", help="Label column")
    parser.add_argument("--x-col", default="x", help="X coordinate column")
    parser.add_argument("--y-col", default="y", help="Y coordinate column")
    parser.add_argument("--offset-x", default="auto", help="X coordinate offset to subtract before transform (float or 'auto')")
    parser.add_argument("--offset-y", default="auto", help="Y coordinate offset to subtract before transform (float or 'auto')")
    parser.add_argument("--reference-mask", default="", help="Path to reference mask for auto-detecting alignment offset")
    parser.add_argument("--flip-y", action="store_true", default=True, help="Invert Y axis for coordinate orientation")
    parser.add_argument("--no-flip-y", dest="flip_y", action="store_false", help="Do not invert Y axis")
    parser.add_argument("--crop-mode", choices=["intersect", "strict", "none", "clip"], default="intersect", help="ROI cropping mode ('none', 'intersect', 'clip', 'strict')")
    parser.add_argument("--use-inverse", action="store_true", default=False, help="Apply the inverse transformation matrix (e.g. matrix_xy_px_inv or np.linalg.inv)")
    parser.add_argument("--sample-id", default="1", help="Sample ID for GUI table")
    parser.add_argument("--roi-name", default="", help="ROI / Image name for GUI table")
    parser.add_argument("--image-path", default="", help="Image path for GUI table")
    parser.add_argument("--generate-mask", action="store_true", default=True, help="Also generate transformed ROI mask TIFF")
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="both", help="Export format")

    # Optional mosaic tile splitting arguments
    parser.add_argument("--split-mosaic", action="store_true", default=False, help="Automatically split output into mosaic subsections")
    parser.add_argument("--split-tile-dir", default=None, help="Directory containing unstitched TIFF tiles for automatic discovery")
    parser.add_argument("--split-grid-rows", type=int, default=None, help="Mosaic grid rows")
    parser.add_argument("--split-grid-cols", type=int, default=None, help="Mosaic grid columns")
    parser.add_argument("--split-tile-height", type=int, default=None, help="Tile height in pixels")
    parser.add_argument("--split-tile-width", type=int, default=None, help="Tile width in pixels")
    parser.add_argument("--split-overlap-percent", type=float, default=5.0, help="Overlap percentage between tiles")
    parser.add_argument("--split-order", choices=["snake", "raster"], default="snake", help="Tile acquisition order")
    parser.add_argument("--split-coord-mode", choices=["local", "global", "both"], default="both", help="Coordinate mode for split tables")

    args = parser.parse_args()

    split_mosaic = args.split_mosaic or bool(args.split_tile_dir) or bool(args.split_grid_rows and args.split_grid_cols)

    transform_and_crop_rois(
        input_path=args.input,
        transform_json=args.transform_json,
        output_prefix=args.output_prefix,
        cell_col=args.cell_col,
        label_col=args.label_col,
        x_col=args.x_col,
        y_col=args.y_col,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        reference_mask=args.reference_mask,
        flip_y=args.flip_y,
        use_inverse=args.use_inverse,
        crop_mode=args.crop_mode,
        sample_id=args.sample_id,
        roi_name=args.roi_name,
        image_path=args.image_path,
        generate_mask=args.generate_mask,
        export_format=args.format,
        split_mosaic=split_mosaic,
        split_tile_dir=args.split_tile_dir,
        split_grid_rows=args.split_grid_rows,
        split_grid_cols=args.split_grid_cols,
        split_tile_height=args.split_tile_height,
        split_tile_width=args.split_tile_width,
        split_overlap_percent=args.split_overlap_percent,
        split_order=args.split_order,
        split_coord_mode=args.split_coord_mode,
    )


if __name__ == "__main__":
    main()

