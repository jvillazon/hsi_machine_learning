#!/usr/bin/env python3
"""
merge_boundaries_and_labels.py
==============================
Step 1 of Xenium / Spatial ROI Pipeline:
Merges cell boundary polygon coordinates (Parquet or CSV) with cell metadata/labels (CSV or Parquet).

Outputs:
  - Per-vertex merged table (columns: cell_id, x, y, label/group, etc.) for mask generation and transforms.
  - Aggregated polygon ROI table (Vertices_JSON, Geometry_WKT, Centroid_X, Centroid_Y, Area_px, Label, etc.)
    ready for multimodal_napari_gui import (1_regions.csv compatible).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd


def compute_polygon_area(pts: np.ndarray) -> float:
    """Compute 2D polygon area using the Shoelace formula."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def load_dataframe(path: str) -> pd.DataFrame:
    """Load DataFrame from Parquet or CSV based on file extension."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    elif ext in (".csv", ".txt"):
        return pd.read_csv(path)
    else:
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)


def resolve_column(df: pd.DataFrame, requested_col: str, candidates: List[str], file_desc: str) -> str:
    """Resolve column name with fallback candidates."""
    if requested_col in df.columns:
        return requested_col
    for c in candidates:
        if c in df.columns:
            print(f"[{file_desc}] Column '{requested_col}' not found, using '{c}' instead.")
            return c
    raise KeyError(
        f"[{file_desc}] None of { [requested_col] + candidates } found. Available columns: {list(df.columns)}"
    )


def merge_boundaries_and_labels(
    boundaries_path: str,
    labels_path: str,
    output_prefix: str,
    boundary_cell_col: str = "cell_id",
    boundary_x_col: str = "vertex_x",
    boundary_y_col: str = "vertex_y",
    label_cell_col: str = "cell_id",
    label_group_col: str = "group",
    unlabeled_handling: str = "filter",  # 'filter' or 'keep'
    default_unlabeled: str = "Unassigned",
    sample_id: str = "1",
    image_name: str = "",
    image_path: str = "",
    save_mode: str = "both",  # 'vertices', 'regions', 'both'
    export_format: str = "both",  # 'parquet', 'csv', 'both'
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Merge boundary coordinates with cell labels and export per-vertex and/or aggregated ROI tables.
    """
    t0 = time.time()
    print("=" * 70)
    print("STEP 1: MERGING BOUNDARIES AND LABELS")
    print("=" * 70)
    print(f"Boundaries : {boundaries_path}")
    print(f"Labels     : {labels_path}")

    # 1. Load files
    df_bounds = load_dataframe(boundaries_path)
    df_labels = load_dataframe(labels_path)
    print(f"Loaded {len(df_bounds):,} boundary rows and {len(df_labels):,} label rows.")

    # 2. Resolve columns
    b_cell = resolve_column(df_bounds, boundary_cell_col, ["cell_id", "cell_ID", "Cell_ID", "id"], "Boundaries")
    b_x = resolve_column(df_bounds, boundary_x_col, ["vertex_x", "x", "X", "vertexX"], "Boundaries")
    b_y = resolve_column(df_bounds, boundary_y_col, ["vertex_y", "y", "Y", "vertexY"], "Boundaries")

    l_cell = resolve_column(df_labels, label_cell_col, ["cell_id", "cell_ID", "Cell_ID", "id"], "Labels")
    l_grp = resolve_column(df_labels, label_group_col, ["group", "Label", "label", "cell_type", "levelKID_refined", "cluster"], "Labels")

    # Standardize label DataFrame
    df_lbl_clean = df_labels[[l_cell, l_grp]].drop_duplicates(subset=[l_cell]).copy()
    df_lbl_clean.rename(columns={l_cell: "cell_id", l_grp: "group"}, inplace=True)
    df_lbl_clean["cell_id"] = df_lbl_clean["cell_id"].astype(str)

    # Standardize boundary DataFrame
    df_bounds_clean = df_bounds.copy()
    df_bounds_clean["cell_id"] = df_bounds_clean[b_cell].astype(str)
    if b_x != "x":
        df_bounds_clean["x"] = df_bounds_clean[b_x].astype(np.float64)
    if b_y != "y":
        df_bounds_clean["y"] = df_bounds_clean[b_y].astype(np.float64)

    # 3. Merge
    how_merge = "inner" if unlabeled_handling == "filter" else "left"
    df_merged_vertices = df_bounds_clean.merge(df_lbl_clean, on="cell_id", how=how_merge)

    if unlabeled_handling != "filter":
        df_merged_vertices["group"] = df_merged_vertices["group"].fillna(default_unlabeled)

    n_input = df_bounds_clean["cell_id"].nunique()
    n_matched = df_merged_vertices["cell_id"].nunique()
    print(f"Matched {n_matched:,} unique cells out of {n_input:,} total cells.")

    if df_merged_vertices.empty:
        raise ValueError("No matching cell IDs found between boundary file and label file!")

    out_dir = os.path.dirname(os.path.abspath(output_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # 4. Save per-vertex table if requested
    if save_mode in ("vertices", "both"):
        v_base = f"{output_prefix}_merged_vertices"
        if export_format in ("parquet", "both"):
            df_merged_vertices.to_parquet(f"{v_base}.parquet", index=False)
            print(f"Saved merged vertices Parquet: {v_base}.parquet")
        if export_format in ("csv", "both"):
            df_merged_vertices.to_csv(f"{v_base}.csv", index=False)
            print(f"Saved merged vertices CSV    : {v_base}.csv")

    df_regions = None
    # 5. Build and save aggregated GUI-compatible ROI table if requested
    if save_mode in ("regions", "both"):
        print("\nAggregating vertices into GUI-compatible polygon ROIs...")
        c_ids = df_merged_vertices["cell_id"].values
        lbls = df_merged_vertices["group"].astype(str).values
        vx = df_merged_vertices["x"].values
        vy = df_merged_vertices["y"].values

        change_idx = np.where(c_ids[:-1] != c_ids[1:])[0] + 1
        splits_xy = np.split(np.column_stack((vx, vy)), change_idx)
        splits_c = np.split(c_ids, change_idx)
        splits_l = np.split(lbls, change_idx)

        records = []
        label_counters = {}

        for idx, (pts, c_arr, l_arr) in enumerate(zip(splits_xy, splits_c, splits_l), start=1):
            c_id = str(c_arr[0])
            lbl = str(l_arr[0])

            label_counters[lbl] = label_counters.get(lbl, 0) + 1
            roi_inst = label_counters[lbl]

            # Ensure polygon is closed for WKT
            if len(pts) >= 2 and not np.allclose(pts[0], pts[-1]):
                pts_closed = np.vstack([pts, pts[0]])
            else:
                pts_closed = pts

            pts_list = pts_closed.round(4).tolist()
            wkt = "POLYGON ((" + ", ".join(f"{x:.4f} {y:.4f}" for x, y in pts_closed) + "))"
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            area = compute_polygon_area(pts)

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
                "Centroid_X": round(cx, 4),
                "Centroid_Y": round(cy, 4),
                "Area_px": round(area, 2),
                "UID": c_id,
                "cell_id": c_id,
                "Image_Name": image_name,
                "Image_Path": image_path,
            })

        df_regions = pd.DataFrame(records)
        r_base = f"{output_prefix}_labeled_regions"
        if export_format in ("parquet", "both"):
            df_regions.to_parquet(f"{r_base}.parquet", index=False)
            print(f"Saved GUI-compatible Parquet: {r_base}.parquet ({len(df_regions):,} ROIs)")
        if export_format in ("csv", "both"):
            df_regions.to_csv(f"{r_base}.csv", index=False)
            print(f"Saved GUI-compatible CSV    : {r_base}.csv ({len(df_regions):,} ROIs)")

    print(f"Done in {time.time() - t0:.2f}s.")
    return df_merged_vertices, df_regions


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Merge cell boundary coordinates with cell metadata/labels."
    )
    parser.add_argument("--boundaries", "-b", required=True, help="Path to boundaries file (.parquet or .csv)")
    parser.add_argument("--labels", "-l", required=True, help="Path to labels file (.csv or .parquet)")
    parser.add_argument("--output-prefix", "-o", required=True, help="Prefix for output files (without extension)")
    parser.add_argument("--boundary-cell-col", default="cell_id", help="Cell ID column in boundary file")
    parser.add_argument("--boundary-x-col", default="vertex_x", help="X coordinate column in boundary file")
    parser.add_argument("--boundary-y-col", default="vertex_y", help="Y coordinate column in boundary file")
    parser.add_argument("--label-cell-col", default="cell_id", help="Cell ID column in label file")
    parser.add_argument("--label-group-col", default="group", help="Label/group column in label file")
    parser.add_argument("--unlabeled-handling", choices=["filter", "keep"], default="filter", help="Filter or keep unlabeled cells")
    parser.add_argument("--default-unlabeled", default="Unassigned", help="Default label for unlabeled cells if keep")
    parser.add_argument("--sample-id", default="1", help="Sample_ID string")
    parser.add_argument("--image-name", default="", help="Image_Name string")
    parser.add_argument("--image-path", default="", help="Image_Path string")
    parser.add_argument("--mode", choices=["vertices", "regions", "both"], default="both", help="What tables to save")
    parser.add_argument("--format", choices=["parquet", "csv", "both"], default="both", help="Export file format")

    args = parser.parse_args()

    merge_boundaries_and_labels(
        boundaries_path=args.boundaries,
        labels_path=args.labels,
        output_prefix=args.output_prefix,
        boundary_cell_col=args.boundary_cell_col,
        boundary_x_col=args.boundary_x_col,
        boundary_y_col=args.boundary_y_col,
        label_cell_col=args.label_cell_col,
        label_group_col=args.label_group_col,
        unlabeled_handling=args.unlabeled_handling,
        default_unlabeled=args.default_unlabeled,
        sample_id=args.sample_id,
        image_name=args.image_name,
        image_path=args.image_path,
        save_mode=args.mode,
        export_format=args.format,
    )


if __name__ == "__main__":
    main()
