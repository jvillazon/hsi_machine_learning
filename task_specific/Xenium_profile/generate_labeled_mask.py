#!/usr/bin/env python3
"""
generate_labeled_mask.py
========================
Step 2 of Xenium / Spatial ROI Pipeline:
Generates labeled integer mask TIFF images from cell boundary polygons and metadata labels,
using a metadata dictionary or list (as in processing_mask.ipynb and read_parquet.ipynb).

Features:
  - Supports user-specified metadata dictionary ({label: integer_id}) or list ([label1, label2, ...]).
  - If no dictionary is provided, automatically enumerates unique labels (1..N) and saves a legend JSON.
  - Rasterizes polygons with ultra-fast cv2.fillPoly.
  - Supports 2D integer labeled mask (uint16/uint32), 2D instance mask, 3D binary mask stack,
    per-label split binary TIFFs, and RGB visualization.
  - ImageJ-compatible TIFF metadata (axes, pixel resolution, unit).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
import numpy as np
import pandas as pd
import tifffile

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage import measure
except ImportError:
    measure = None


def load_dataframe(path: str) -> pd.DataFrame:
    """Load DataFrame from Parquet or CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_metadata_mapping(
    dict_arg: Optional[str] = None,
    list_arg: Optional[str] = None,
    unique_labels: Optional[List[str]] = None,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label-to-int and int-to-label mappings from dict, list, or unique labels.
    """
    label_to_id: Dict[str, int] = {}

    if dict_arg:
        # Check if dict_arg is a JSON file or JSON string
        if os.path.exists(dict_arg):
            with open(dict_arg, "r") as f:
                raw_dict = json.load(f)
        else:
            raw_dict = json.loads(dict_arg)
        label_to_id = {str(k): int(v) for k, v in raw_dict.items()}

    elif list_arg:
        # Check if list_arg is a file, JSON, or comma-separated
        items = []
        if os.path.exists(list_arg):
            with open(list_arg, "r") as f:
                content = f.read().strip()
                if content.startswith("["):
                    items = json.loads(content)
                else:
                    items = [line.strip() for line in content.splitlines() if line.strip()]
        elif list_arg.strip().startswith("["):
            items = json.loads(list_arg)
        else:
            items = [x.strip() for x in list_arg.split(",") if x.strip()]

        label_to_id = {str(label): idx for idx, label in enumerate(items, start=1)}

    elif unique_labels is not None:
        # Auto-enumerate sorted unique labels
        sorted_labels = sorted([str(l) for l in unique_labels if pd.notna(l)])
        label_to_id = {label: idx for idx, label in enumerate(sorted_labels, start=1)}

    id_to_label = {v: k for k, v in label_to_id.items()}
    return label_to_id, id_to_label


def generate_labeled_mask(
    input_path: str,
    output_path: str,
    metadata_dict_arg: Optional[str] = None,
    metadata_list_arg: Optional[str] = None,
    cell_col: str = "cell_id",
    label_col: str = "group",
    x_col: str = "x",
    y_col: str = "y",
    height: Optional[int] = None,
    width: Optional[int] = None,
    padding: int = 10,
    pixel_size_um: float = 1.0,
    mask_type: str = "labeled",  # 'labeled', 'instance', 'stack', 'split', 'all'
    min_size: int = 0,
    default_bg_val: int = 0,
) -> Dict[str, str]:
    """
    Generate labeled mask TIFFs from cell boundary data and label metadata.
    """
    t0 = time.time()
    print("=" * 70)
    print("STEP 2: GENERATING LABELED MASK TIFF")
    print("=" * 70)
    print(f"Input file   : {input_path}")
    print(f"Output path  : {output_path}")

    # 1. Load input table
    df = load_dataframe(input_path)

    # Detect if input is aggregated regions (has Vertices_JSON) or per-vertex
    has_vertices_json = "Vertices_JSON" in df.columns
    if has_vertices_json:
        print(f"Detected aggregated ROI format ({len(df):,} cell rows).")
        # Extract cell polygons from Vertices_JSON
        c_label_col = "Label" if "Label" in df.columns else ("Fluorescence_Label" if "Fluorescence_Label" in df.columns else label_col)
    else:
        print(f"Detected per-vertex coordinate format ({len(df):,} vertex rows).")
        # Resolve coordinate columns
        for cand in [x_col, "vertex_x", "x_roi", "x"]:
            if cand in df.columns:
                x_col = cand
                break
        for cand in [y_col, "vertex_y", "y_roi", "y"]:
            if cand in df.columns:
                y_col = cand
                break
        for cand in [label_col, "Label", "group", "cell_type", "levelKID_refined"]:
            if cand in df.columns:
                c_label_col = cand
                break
        for cand in [cell_col, "cell_ID", "cell_id", "UID"]:
            if cand in df.columns:
                cell_col = cand
                break

    # 2. Build metadata mapping
    all_labels = df[c_label_col].dropna().unique().tolist()
    label_to_id, id_to_label = parse_metadata_mapping(
        dict_arg=metadata_dict_arg,
        list_arg=metadata_list_arg,
        unique_labels=all_labels,
    )
    print(f"Mapped {len(label_to_id)} labels to integer IDs.")

    # 3. Calculate resolution / pixel size using min_x and min_y (processing_mask.ipynb method)
    if has_vertices_json:
        all_xs, all_ys = [], []
        for v_raw in df["Vertices_JSON"]:
            pts = json.loads(v_raw) if isinstance(v_raw, str) else v_raw
            if pts:
                arr = np.array(pts)
                all_xs.extend(arr[:, 0])
                all_ys.extend(arr[:, 1])
        raw_x_arr = np.array(all_xs, dtype=np.float64)
        raw_y_arr = np.array(all_ys, dtype=np.float64)
    else:
        raw_x_arr = df[x_col].values.astype(np.float64)
        raw_y_arr = df[y_col].values.astype(np.float64)

    diff_x = np.diff(raw_x_arr)
    diff_y = np.diff(raw_y_arr)
    diff_x_pos = diff_x[diff_x > 0]
    diff_y_pos = diff_y[diff_y > 0]
    calc_min_x = float(np.min(diff_x_pos)) if len(diff_x_pos) > 0 else 1.0
    calc_min_y = float(np.min(diff_y_pos)) if len(diff_y_pos) > 0 else 1.0

    # Determine final pixel size and resolution
    if pixel_size_um is not None and str(pixel_size_um).lower() not in ("auto", "none", "0", "0.0"):
        try:
            val = float(pixel_size_um)
            res_x, res_y= val, val
            # res_x, res_y = 1.0 / val, 1.0 / val
            print(f"Using specified pixel size: {val} um (res: {res_x:.4f} px/um)")
        except ValueError:
            res_x, res_y= calc_min_x, calc_min_y
            # res_x, res_y = 1.0 / calc_min_x, 1.0 / calc_min_y
            print(f"Calculated pixel size from processing_mask.ipynb method:")
            print(f"  min_x = {calc_min_x:.6f} um -> res_x = {res_x:.4f} px/um")
            print(f"  min_y = {calc_min_y:.6f} um -> res_y = {res_y:.4f} px/um")
    else:
        res_x, res_y= calc_min_x, calc_min_y
        # res_x, res_y = 1.0 / calc_min_x, 1.0 / calc_min_y
        print(f"Calculated pixel size from processing_mask.ipynb method:")
        print(f"  min_x = {calc_min_x:.6f} um -> res_x = {res_x:.4f} px/um")
        print(f"  min_y = {calc_min_y:.6f} um -> res_y = {res_y:.4f} px/um")

    # 4. Determine canvas dimensions
    max_x = int(np.ceil(raw_x_arr.max())) if len(raw_x_arr) > 0 else 1000
    max_y = int(np.ceil(raw_y_arr.max())) if len(raw_y_arr) > 0 else 1000

    canvas_w = width if width is not None else (max_x + padding * 2)
    canvas_h = height if height is not None else (max_y + padding * 2)
    print(f"Mask canvas dimensions: height = {canvas_h}, width = {canvas_w}")

    # Output directory
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    base, ext = os.path.splitext(output_path)

    # Save legend JSON
    legend_path = f"{base}_label_legend.json"
    with open(legend_path, "w") as f:
        json.dump({
            "label_to_id": label_to_id,
            "id_to_label": {str(k): v for k, v in id_to_label.items()},
            "resolution": {"min_x_um": res_x, "min_y_um": res_y, "res_x": res_x, "res_y": res_y}
        }, f, indent=2)
    print(f"Saved label legend mapping to: {legend_path}")

    # 5. Prepare cell polygon list: list of (pts_array, label_str, cell_id)
    cell_polys = []
    if has_vertices_json:
        for idx, row in df.iterrows():
            v_raw = row["Vertices_JSON"]
            pts = json.loads(v_raw) if isinstance(v_raw, str) else v_raw
            if len(pts) >= 3:
                pts_arr = np.array(pts, dtype=np.int32)
                lbl = str(row[c_label_col])
                cid = str(row.get(cell_col, idx))
                cell_polys.append((pts_arr, lbl, cid))
    else:
        # Group per-vertex rows
        c_ids = df[cell_col].values
        lbls = df[c_label_col].astype(str).values
        vx = df[x_col].values
        vy = df[y_col].values

        change_idx = np.where(c_ids[:-1] != c_ids[1:])[0] + 1
        splits_xy = np.split(np.column_stack((vx, vy)), change_idx)
        splits_c = np.split(c_ids, change_idx)
        splits_l = np.split(lbls, change_idx)

        for pts, c_arr, l_arr in zip(splits_xy, splits_c, splits_l):
            if len(pts) >= 3:
                pts_arr = np.round(pts).astype(np.int32)
                cell_polys.append((pts_arr, str(l_arr[0]), str(c_arr[0])))

    print(f"Ready to rasterize {len(cell_polys):,} cell polygons.")
    saved_outputs = {"legend": legend_path}

    tiff_meta = {
        "axes": "YX",
        "unit": "micron",
        "spacing": 1,
    }

    # 6. Generate Binary Mask (completely binary 0 / 255)
    if mask_type in ("binary", "all"):
        bin_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        for pts, _, _ in cell_polys:
            if cv2 is not None:
                cv2.fillPoly(bin_mask, [pts], 255)
            else:
                from skimage.draw import polygon as draw_poly
                rr, cc = draw_poly(pts[:, 1], pts[:, 0], shape=bin_mask.shape)
                bin_mask[rr, cc] = 255

        # If mask_type is exclusively binary, save directly to output_path if it ends in .tif
        bin_tif = output_path if (mask_type == "binary" and output_path.endswith((".tif", ".tiff"))) else f"{base}_binary_mask.tif"
        tifffile.imwrite(
            bin_tif,
            bin_mask,
            imagej=True,
            resolution=(res_x, res_y),
            resolutionunit=1,
            metadata=tiff_meta,
        )
        print(f"Saved Binary Mask: {bin_tif} (uint8, values: 0 / 255, res: {res_x:.4f} px/um)")
        saved_outputs["binary_mask"] = bin_tif

    # 7. Generate 2D Labeled Mask (pixel value = label integer ID)
    if mask_type in ("labeled", "all"):
        max_id = max(label_to_id.values()) if label_to_id else 1
        dtype = np.uint16 if max_id < 65535 else np.uint32
        mask_2d = np.full((canvas_h, canvas_w), default_bg_val, dtype=dtype)

        for pts, lbl, _ in cell_polys:
            val = label_to_id.get(lbl, default_bg_val)
            if val != default_bg_val:
                if cv2 is not None:
                    cv2.fillPoly(mask_2d, [pts], int(val))
                else:
                    from skimage.draw import polygon as draw_poly
                    rr, cc = draw_poly(pts[:, 1], pts[:, 0], shape=mask_2d.shape)
                    mask_2d[rr, cc] = val

        labeled_tif = output_path if (mask_type == "labeled" and output_path.endswith((".tif", ".tiff"))) else f"{base}_labeled_mask.tif"
        use_ij = (dtype != np.uint32)
        tifffile.imwrite(
            labeled_tif,
            mask_2d,
            imagej=use_ij,
            resolution=(res_x, res_y),
            resolutionunit=1,
            metadata=tiff_meta,
        )
        print(f"Saved 2D Labeled Mask: {labeled_tif} (dtype={dtype.__name__})")
        saved_outputs["labeled_mask"] = labeled_tif

    # 8. Generate 2D Instance Mask (pixel value = cell instance 1..N)
    if mask_type in ("instance", "all"):
        n_cells = len(cell_polys)
        # ImageJ only supports uint8, uint16, float32. Use uint16 if <= 65535 to avoid ValueError 'I'
        inst_dtype = np.uint16 if n_cells <= 65535 else np.uint32
        use_ij = (inst_dtype != np.uint32)

        inst_mask = np.zeros((canvas_h, canvas_w), dtype=inst_dtype)
        for idx, (pts, _, _) in enumerate(cell_polys, start=1):
            cell_val = idx if inst_dtype == np.uint32 else min(idx, 65535)
            if cv2 is not None:
                cv2.fillPoly(inst_mask, [pts], cell_val)
            else:
                from skimage.draw import polygon as draw_poly
                rr, cc = draw_poly(pts[:, 1], pts[:, 0], shape=inst_mask.shape)
                inst_mask[rr, cc] = cell_val

        inst_tif = output_path if (mask_type == "instance" and output_path.endswith((".tif", ".tiff"))) else f"{base}_instance_mask.tif"
        tifffile.imwrite(
            inst_tif,
            inst_mask,
            imagej=use_ij,
            resolution=(res_x, res_y),
            resolutionunit=1,
            metadata=tiff_meta,
        )
        print(f"Saved 2D Instance Mask: {inst_tif} ({n_cells:,} instances, dtype={inst_dtype.__name__})")
        saved_outputs["instance_mask"] = inst_tif

    # 9. Generate Multi-channel Stack or Split TIFFs
    if mask_type in ("stack", "split", "all"):
        sorted_id_items = sorted(id_to_label.items())
        n_channels = len(sorted_id_items)

        if mask_type in ("stack", "all"):
            stack_3d = np.zeros((n_channels, canvas_h, canvas_w), dtype=np.uint8)
            channel_labels = []

            for c_idx, (lid, lname) in enumerate(sorted_id_items):
                channel_labels.append(lname)
                c_mask = stack_3d[c_idx]
                for pts, lbl, _ in cell_polys:
                    if lbl == lname and cv2 is not None:
                        cv2.fillPoly(c_mask, [pts], 255)

            stack_tif = f"{base}_mask_stack.tif"
            tifffile.imwrite(
                stack_tif,
                stack_3d,
                imagej=True,
                resolution=(res_val, res_val),
                resolutionunit=1,
                metadata={
                    "axes": "CYX",
                    "Labels": channel_labels,
                    "unit": "micron",
                    "spacing": pixel_size_um,
                },
            )
            print(f"Saved 3D Mask Stack: {stack_tif} (channels={n_channels})")
            saved_outputs["mask_stack"] = stack_tif

        if mask_type in ("split", "all"):
            split_dir = f"{base}_split_masks"
            os.makedirs(split_dir, exist_ok=True)
            print(f"Writing per-label binary masks into: {split_dir}/")

            for lid, lname in sorted_id_items:
                clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in lname)
                bin_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                for pts, lbl, _ in cell_polys:
                    if lbl == lname and cv2 is not None:
                        cv2.fillPoly(bin_mask, [pts], 255)

                if not np.any(bin_mask):
                    continue

                if min_size > 0 and measure is not None:
                    lbl_arr = measure.label(bin_mask, connectivity=2)
                    props = measure.regionprops(lbl_arr)
                    filtered = np.zeros_like(bin_mask)
                    for r in props:
                        if r.area >= min_size:
                            filtered[lbl_arr == r.label] = 255
                    bin_mask = filtered

                fname = os.path.join(split_dir, f"Xenium-{clean_name}-mask.tif")
                tifffile.imwrite(
                    fname,
                    bin_mask,
                    imagej=True,
                    resolution=(res_x, res_y),
                    resolutionunit=1,
                    metadata=tiff_meta,
                )
            saved_outputs["split_dir"] = split_dir

    print(f"Mask generation completed in {time.time() - t0:.2f}s.")
    return saved_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Generate integer-labeled or binary mask TIFF from cell polygons and metadata."
    )
    parser.add_argument("--input", "-i", required=True, help="Input file (merged vertices or labeled regions parquet/csv)")
    parser.add_argument("--output", "-o", required=True, help="Output mask TIFF path (e.g. output_mask.tif)")
    parser.add_argument("--metadata-dict", help="JSON file or string mapping label names to integer values")
    parser.add_argument("--metadata-list", help="Text file, JSON list, or comma-separated list of label names")
    parser.add_argument("--cell-col", default="cell_id", help="Cell ID column name")
    parser.add_argument("--label-col", default="group", help="Label column name")
    parser.add_argument("--x-col", default="x", help="X coordinate column name")
    parser.add_argument("--y-col", default="y", help="Y coordinate column name")
    parser.add_argument("--height", type=int, help="Optional canvas height in pixels")
    parser.add_argument("--width", type=int, help="Optional canvas width in pixels")
    parser.add_argument("--pad", type=int, default=10, help="Padding in pixels if auto-calculating canvas size")
    parser.add_argument("--pixel-size", default="auto", help="Pixel size in microns (default: auto, calculated as min_x/min_y from processing_mask.ipynb)")
    parser.add_argument("--mask-type", choices=["binary", "labeled", "instance", "stack", "split", "all"], default="binary", help="Mask output type")
    parser.add_argument("--min-size", type=int, default=0, help="Minimum connected component area in pixels for split masks")

    args = parser.parse_args()

    generate_labeled_mask(
        input_path=args.input,
        output_path=args.output,
        metadata_dict_arg=args.metadata_dict,
        metadata_list_arg=args.metadata_list,
        cell_col=args.cell_col,
        label_col=args.label_col,
        x_col=args.x_col,
        y_col=args.y_col,
        height=args.height,
        width=args.width,
        padding=args.pad,
        pixel_size_um=args.pixel_size,
        mask_type=args.mask_type,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()
