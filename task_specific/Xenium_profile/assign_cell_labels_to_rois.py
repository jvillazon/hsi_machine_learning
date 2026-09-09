#!/usr/bin/env python3
"""
assign_cell_labels_to_rois.py
=============================
Takes cell boundary coordinates (from Parquet or CSV, e.g., Xenium cell_boundaries.parquet),
merges them with cell type / cluster annotations (from CSV or Parquet, e.g., cell_labels.csv),
and formats each cell into a polygon ROI table compatible with `1_regions.csv` and
the `multimodal_napari_gui` application (`run_app.py` / `core.roi_manager`).

Compatible with:
- multimodal_napari_gui/widgets/segmentation_widget (Import ROI File)
- multimodal_napari_gui/widgets/ratiometric_widget (Select ROI File)
- multimodal_napari_gui/widgets/srs_widget (Select ROI File)
- napari Shapes layer visualization
"""

import os
import sys
import json
import time
import argparse
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
        # Try parquet first, then csv
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_csv(path)


def resolve_column(df: pd.DataFrame, requested_col: str, candidates: List[str], file_desc: str) -> str:
    """Resolve column name with fallback candidates and clear error messages."""
    if requested_col in df.columns:
        return requested_col
    for c in candidates:
        if c in df.columns:
            print(f"[{file_desc}] Column '{requested_col}' not found, using '{c}' instead.")
            return c
    raise KeyError(
        f"[{file_desc}] None of the expected columns { [requested_col] + candidates } were found. "
        f"Available columns: {list(df.columns)}"
    )


def process_and_assign_labels(
    boundaries_path: str,
    labels_path: str,
    output_path: str,
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
    scale_factor: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    export_format: str = "auto",  # 'auto', 'parquet', 'csv', 'both'
) -> Tuple[pd.DataFrame, str]:
    """
    Core function to process boundaries and labels, returning formatted DataFrame and saved paths.
    """
    t_start = time.time()
    print("=" * 70)
    print("STARTING CELL BOUNDARY & LABEL FORMATTER")
    print("=" * 70)
    print(f"Boundaries File : {boundaries_path}")
    print(f"Labels File     : {labels_path}")
    print(f"Output Target   : {output_path}")

    # 1. Load data
    print("\n[1/5] Loading files...")
    df_bounds = load_dataframe(boundaries_path)
    print(f"  Boundaries: {len(df_bounds):,} rows loaded.")
    df_labels = load_dataframe(labels_path)
    print(f"  Labels    : {len(df_labels):,} rows loaded.")

    # 2. Resolve columns
    b_cell_col = resolve_column(df_bounds, boundary_cell_col, ["cell_id", "cell_ID", "Cell_ID", "id"], "Boundaries")
    b_x_col = resolve_column(df_bounds, boundary_x_col, ["vertex_x", "x", "X", "vertexX", "x_coord"], "Boundaries")
    b_y_col = resolve_column(df_bounds, boundary_y_col, ["vertex_y", "y", "Y", "vertexY", "y_coord"], "Boundaries")

    l_cell_col = resolve_column(df_labels, label_cell_col, ["cell_id", "cell_ID", "Cell_ID", "id"], "Labels")
    l_group_col = resolve_column(df_labels, label_group_col, ["group", "Label", "label", "cell_type", "cluster", "Cluster"], "Labels")

    # Standardize label DataFrame
    df_labels_clean = df_labels[[l_cell_col, l_group_col]].drop_duplicates(subset=[l_cell_col]).copy()
    df_labels_clean.rename(columns={l_cell_col: "_match_cell_id", l_group_col: "_assigned_label"}, inplace=True)
    df_labels_clean["_match_cell_id"] = df_labels_clean["_match_cell_id"].astype(str)

    # 3. Merge boundaries with labels
    print("\n[2/5] Joining boundaries with labels...")
    df_bounds["_match_cell_id"] = df_bounds[b_cell_col].astype(str)

    merge_how = "inner" if unlabeled_handling == "filter" else "left"
    df_merged = df_bounds.merge(df_labels_clean, on="_match_cell_id", how=merge_how)

    if unlabeled_handling != "filter":
        df_merged["_assigned_label"] = df_merged["_assigned_label"].fillna(default_unlabeled)

    total_input_cells = df_bounds["_match_cell_id"].nunique()
    matched_cells = df_merged["_match_cell_id"].nunique()
    print(f"  Unique cells in boundary file : {total_input_cells:,}")
    print(f"  Unique cells matched to label : {matched_cells:,}")
    if unlabeled_handling == "filter":
        print(f"  Filtered out {total_input_cells - matched_cells:,} unlabeled cells.")
    else:
        print(f"  Retained all cells (unlabeled filled with '{default_unlabeled}').")

    if df_merged.empty:
        raise ValueError("No matching cell IDs found between boundary file and label file!")

    # 4. Group vertices into polygons
    print("\n[3/5] Grouping vertices into polygon ROIs and computing geometries...")
    c_ids = df_merged["_match_cell_id"].values
    lbls = df_merged["_assigned_label"].astype(str).values
    vx = (df_merged[b_x_col].values.astype(np.float64) * scale_factor) + offset_x
    vy = (df_merged[b_y_col].values.astype(np.float64) * scale_factor) + offset_y

    # Fast contiguous chunk splitting
    change_idx = np.where(c_ids[:-1] != c_ids[1:])[0] + 1
    splits_xy = np.split(np.column_stack((vx, vy)), change_idx)
    splits_c = np.split(c_ids, change_idx)
    splits_l = np.split(lbls, change_idx)

    records = []
    label_instance_counters = {}

    for idx, (pts, c_arr, l_arr) in enumerate(zip(splits_xy, splits_c, splits_l), start=1):
        c_id = str(c_arr[0])
        lbl = str(l_arr[0])

        label_instance_counters[lbl] = label_instance_counters.get(lbl, 0) + 1
        roi_inst = label_instance_counters[lbl]

        # Polygon closure check
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

    # Standard column ordering matching 1_regions.csv and multimodal_napari_gui
    df_out = pd.DataFrame(records)
    print(f"  Successfully constructed {len(df_out):,} cell polygon ROIs.")

    # 5. Export to disk
    print("\n[4/5] Saving formatted ROI file...")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(output_path)
    if export_format == "auto":
        save_pq = (ext.lower() == ".parquet")
        save_csv = not save_pq
    elif export_format == "parquet":
        save_pq = True
        save_csv = False
        if ext.lower() != ".parquet":
            output_path = base + ".parquet"
    elif export_format == "csv":
        save_pq = False
        save_csv = True
        if ext.lower() != ".csv":
            output_path = base + ".csv"
    elif export_format == "both":
        save_pq = True
        save_csv = True
    else:
        save_pq = True
        save_csv = False

    saved_files = []
    if save_pq:
        pq_path = output_path if output_path.endswith(".parquet") else base + ".parquet"
        df_out.to_parquet(pq_path, index=False)
        sz_mb = os.path.getsize(pq_path) / (1024 * 1024)
        print(f"  Saved Parquet: {pq_path} ({sz_mb:.2f} MB)")
        saved_files.append(pq_path)

    if save_csv:
        csv_path = output_path if output_path.endswith(".csv") else base + ".csv"
        df_out.to_csv(csv_path, index=False)
        sz_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"  Saved CSV    : {csv_path} ({sz_mb:.2f} MB)")
        saved_files.append(csv_path)

    # 6. Summary stats
    print("\n[5/5] Top 10 Cell Groups Exported:")
    top_counts = df_out["Label"].value_counts().head(10)
    for grp_name, count in top_counts.items():
        print(f"    {grp_name:<20}: {count:>6,} cells")

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"DONE in {elapsed:.2f}s! Compatible with multimodal_napari_gui ROI import.")
    print("=" * 70)

    return df_out, ", ".join(saved_files)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract cell polygons from boundary files (e.g., cell_boundaries.parquet), "
            "assign group/cell-type labels from metadata (e.g., cell_labels.csv), and "
            "format for multimodal_napari_gui (1_regions.csv / napari_rois.parquet format)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # File paths
    parser.add_argument(
        "--boundaries", "-b",
        default="/Users/jorgevillazon/Downloads/metadata/xen65_KPMP057_ABC/cell_boundaries.parquet",
        help="Path to cell boundaries file (.parquet or .csv)",
    )
    parser.add_argument(
        "--labels", "-l",
        default="/Users/jorgevillazon/Downloads/metadata/xen65_KPMP057_ABC/xen65_KPMP057_C_cell_labels_KPMP3_SRS_mincounts20_allgenes_taccofix.csv",
        help="Path to cell labels file (.csv or .parquet)",
    )
    parser.add_argument(
        "--output", "-o",
        default="/Users/jorgevillazon/Downloads/metadata/xen65_KPMP057_ABC/xen65_KPMP057_C_labeled_regions.parquet",
        help="Output file path (.parquet or .csv)",
    )

    # Column mappings
    parser.add_argument(
        "--boundary-cell-col",
        default="cell_id",
        help="Column name for cell ID in the boundary file",
    )
    parser.add_argument(
        "--boundary-x-col",
        default="vertex_x",
        help="Column name for X coordinate in the boundary file",
    )
    parser.add_argument(
        "--boundary-y-col",
        default="vertex_y",
        help="Column name for Y coordinate in the boundary file",
    )
    parser.add_argument(
        "--label-cell-col",
        default="cell_id",
        help="Column name for cell ID in the label file",
    )
    parser.add_argument(
        "--label-group-col",
        default="group",
        help="Column name for cell group / cell type in the label file",
    )

    # Unlabeled handling
    parser.add_argument(
        "--unlabeled-handling",
        choices=["filter", "keep"],
        default="filter",
        help="'filter' to export only cells present in the labels file; 'keep' to include all cells",
    )
    parser.add_argument(
        "--default-unlabeled",
        default="Unassigned",
        help="Default label string when --unlabeled-handling is 'keep' and a cell lacks a label",
    )

    # Metadata & coordinates
    parser.add_argument(
        "--sample-id",
        default="xen65_KPMP057",
        help="Sample_ID string to populate in output",
    )
    parser.add_argument(
        "--image-name",
        default="",
        help="Optional Image_Name to associate with ROIs",
    )
    parser.add_argument(
        "--image-path",
        default="",
        help="Optional Image_Path to associate with ROIs",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Multiplicative coordinate scaling factor (e.g. for registration)",
    )
    parser.add_argument(
        "--offset-x",
        type=float,
        default=0.0,
        help="Additive X coordinate shift",
    )
    parser.add_argument(
        "--offset-y",
        type=float,
        default=0.0,
        help="Additive Y coordinate shift",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "parquet", "csv", "both"],
        default="both",
        help="Export format: 'parquet', 'csv', 'both', or 'auto' (inferred from output filename)",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    process_and_assign_labels(
        boundaries_path=args.boundaries,
        labels_path=args.labels,
        output_path=args.output,
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
        scale_factor=args.scale_factor,
        offset_x=args.offset_x,
        offset_y=args.offset_y,
        export_format=args.format,
    )


if __name__ == "__main__":
    main()
