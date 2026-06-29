"""
Compute per-instance distance to nearest thick collagen peak.

Given:
1) An instance mask image/stack (2D or 3D), where nonzero pixels are cell masks
2) A collagen thickness TIFF (2D or 3D)

This script performs a post hoc peak-focused analysis of thickness:
- Split peaks into low/high intensity classes (e.g., high = top 5% of peaks)
- Compute distance from each cell instance boundary to nearest peak
- Compute distance from each cell instance boundary to nearest high-intensity peak
- Report summaries and plots in microns using a configurable microns-per-pixel factor

Outputs:
- per_instance_peak_distance.csv
- per_slice_peak_distance_summary.csv
- peak_detection_summary.csv
- mean_boundary_distance_by_slice.png
- per_slice_mean_distance_high_vs_low.csv
- mean_distance_high_vs_low_by_slice.png

Usage:
- Edit paths in main(), then run:
    python task_specific/collagen/thickness_instance_distance_distribution.py
"""

import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi


def _plot_boxplot_with_significance_brackets(
    plot_data,
    x_col,
    y_col,
    hue_col,
    order,
    hue_order,
    palette,
    p_df,
    legend_title,
    plot_title,
    y_label,
    output_dir,
    output_filename,
):
    """
    Create a publication-quality boxplot with significance brackets above hue groups.
    
    Parameters
    ----------
    plot_data : pd.DataFrame
        Data for plotting
    x_col : str
        Column name for x-axis (e.g., "slice_label")
    y_col : str
        Column name for y-axis (e.g., "distance_boundary_to_nearest_peak_microns")
    hue_col : str
        Column name for hue/color grouping (e.g., "nearest_peak_class" or "coherence_label")
    order : list
        Order of x-axis values (e.g., list of slice labels)
    hue_order : list
        Order of hue values (e.g., ["Low", "High"])
    palette : dict
        Color palette mapping hue values to colors
    p_df : pd.DataFrame
        DataFrame with significance data; must have 'slice_index' and 'significance' columns
    legend_title : str
        Title for the legend
    plot_title : str
        Title for the plot
    y_label : str
        Label for y-axis
    output_dir : str
        Directory to save the output PNG
    output_filename : str
        Filename for the output PNG
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Create boxplot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.boxplot(
        data=plot_data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        order=order,
        hue_order=hue_order,
        palette=palette,
        showfliers=False,
        width=0.5,
        ax=ax,
        linewidth=2
    )

    # Thicken all boxplot outlines
    for artist in ax.artists:
        artist.set_linewidth(2)
    for line in ax.lines:
        line.set_linewidth(2)

    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title=legend_title,
        fontsize=10,
        title_fontsize=11,
        loc='upper left',
        bbox_to_anchor=(1.02, 1)
    )

    # Calculate y-axis range for significance brackets
    x = np.arange(len(order), dtype=np.float64)
    half_offset = 0.15
    y_all = plot_data[y_col].to_numpy(dtype=np.float64)
    finite_vals = y_all[np.isfinite(y_all)]
    # ignore outliers for bracket placement, but still allow for some margin above the top 98th percentile
    iqr = np.percentile(finite_vals, 75) - np.percentile(finite_vals, 25) if finite_vals.size > 0 else 0.0
    finite_vals = finite_vals[finite_vals <= np.percentile(finite_vals, 75) + 1.5 * iqr]
    
    if finite_vals.size > 0:
        y_min = float(np.nanmin(finite_vals))
        y_top_display = float(np.nanmax(finite_vals))
    else:
        y_min = 0.0
        y_top_display = 1.0
    
    y_range = max(1.0, y_top_display - y_min)
    gap = 0.4 * y_range
    bracket_h = 0.08 * y_range
    max_annot_y = y_top_display

    # Add significance brackets
    for i, x_label in enumerate(order):
        # Get slice_index from plot data
        slice_rows = plot_data[plot_data[x_col] == x_label]['slice_index'].unique()
        if len(slice_rows) == 0:
            continue
        
        slice_idx = int(slice_rows[0])
        sig_row = p_df[p_df['slice_index'] == slice_idx]
        if sig_row.empty:
            continue

        stars = str(sig_row['significance'].iloc[0])
        if stars in ["ns", "n/a"]:
            continue
        y = y_top_display + gap
        
        ax.plot(
            [x[i] - half_offset, x[i] - half_offset, x[i] + half_offset, x[i] + half_offset],
            [y, y + bracket_h, y + bracket_h, y],
            color="black",
            linewidth=2
        )
        ax.text(x[i], y + bracket_h + 0.02 * y_range, stars, ha="center", va="bottom", fontsize=10)
        max_annot_y = max(max_annot_y, y + bracket_h + 0.04 * y_range)

    # Set axis properties
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=0, ha="center", fontsize=11)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Cell Type", fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=11)

    # Add margin to prevent clipping
    margin = 0.08 * y_range
    ax.set_ylim(bottom=0, top=max_annot_y + margin)

    # Publication-quality styling
    ax.grid(True, alpha=0.15, axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_facecolor('white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1)

    plt.tight_layout()
    out_png = os.path.join(output_dir, output_filename)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return out_png


def _ensure_stack(arr):
    """Ensure array shape is (Z, Y, X)."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def _align_stack_depth(mask_stack, thickness_stack):
    """Match Z-depth if one stack has singleton depth."""
    mz = int(mask_stack.shape[0])
    tz = int(thickness_stack.shape[0])
    if mz == tz:
        return mask_stack, thickness_stack
    if mz > 1 and tz == 1:
        return mask_stack, np.repeat(thickness_stack, mz, axis=0)
    if tz > 1 and mz == 1:
        return np.repeat(mask_stack, tz, axis=0), thickness_stack
    raise ValueError(
        "Mask and thickness stack depths are incompatible. "
        f"Got mask={mask_stack.shape}, thickness={thickness_stack.shape}."
    )


def _validate_shapes(mask_stack, thickness_stack):
    if mask_stack.shape[1:] != thickness_stack.shape[1:]:
        raise ValueError(
            "Mask and thickness must have the same spatial shape after stack normalization. "
            f"Got mask={mask_stack.shape}, thickness={thickness_stack.shape}."
        )


def _instance_labels_for_slice(mask_slice):
    labeled, n = ndi.label(mask_slice > 0)
    if n == 0:
        return None, np.array([], dtype=np.int64)
    ids = np.arange(1, n + 1, dtype=np.int64)
    return labeled, ids


def _detect_thick_peaks(
    thickness_slice,
    high_percentile=95.0,
    low_percentile=90.0,
):
    """
    Detect local high-intensity thickness peaks in one slice.

    Returns:
    - all_peak_mask: bool[Y, X] union of disconnected low peaks and high peaks
    - high_peak_mask: bool[Y, X]
    - threshold: float threshold on intensities for high peaks
    """
    t = np.asarray(thickness_slice, dtype=np.float32)
    valid = np.isfinite(t) & (t > 0)
    if not np.any(valid):
        empty = np.zeros_like(t, dtype=bool)
        return empty, empty, t, np.nan



    threshold = float(np.percentile(t[valid], float(high_percentile)))
    high_peak_mask = valid & (t >= threshold)

    # Build a mask of valid regions disconnected from high peaks.
    labeled_valid, n_valid = ndi.label(valid)
    disconnected_mask = np.zeros_like(valid, dtype=bool)
    if n_valid > 0:
        connected_labels = np.unique(labeled_valid[high_peak_mask])
        connected_labels = connected_labels[connected_labels > 0]
        connected_to_high = np.isin(labeled_valid, connected_labels)
        disconnected_mask = valid & ~connected_to_high

    # In disconnected regions, keep local maxima as thin-fiber candidates.  
    if np.any(disconnected_mask):
        low_threshold = float(np.percentile(t[valid & disconnected_mask], float(low_percentile)))
        low_peak_mask = disconnected_mask & (t >= low_threshold)
    else:
        low_peak_mask = np.zeros_like(valid, dtype=bool)


    all_peak_mask = high_peak_mask | low_peak_mask

    return all_peak_mask, high_peak_mask, threshold


def _distance_map_to_peak_mask(peak_mask):
    if not np.any(peak_mask):
        return None
    return ndi.distance_transform_edt(~peak_mask)


def _instance_boundary_mask(inst_mask):
    eroded = ndi.binary_erosion(inst_mask)
    boundary = inst_mask & ~eroded
    if not np.any(boundary):
        boundary = inst_mask
    return boundary


def analyze_instance_distance_to_peaks(
    mask_tif,
    thickness_tif,
    output_dir,
    high_percentile=95.0,
    low_percentile=50.0,
    microns_per_pixel=0.65,
):
    os.makedirs(output_dir, exist_ok=True)

    mask_stack = _ensure_stack(tifffile.imread(mask_tif))
    thickness_stack = _ensure_stack(tifffile.imread(thickness_tif)).astype(np.float32)
    mask_stack, thickness_stack = _align_stack_depth(mask_stack, thickness_stack)
    _validate_shapes(mask_stack, thickness_stack)

    instance_rows = []
    slice_rows = []
    peak_rows = []

    for z in range(mask_stack.shape[0]):
        mask_slice = mask_stack[z]
        thickness_slice = thickness_stack[z]

        label_map, instance_ids = _instance_labels_for_slice(mask_slice)
        if instance_ids.size == 0:
            print(f"Slice {z}: no instances found")
            continue

        all_peak_mask, high_peak_mask, peak_threshold = _detect_thick_peaks(
            thickness_slice,
            high_percentile=high_percentile,
            low_percentile=low_percentile,
        )
        peak_dist_map_all, peak_nn_idx = ndi.distance_transform_edt(~all_peak_mask, return_indices=True)
        peak_dist_map_high = _distance_map_to_peak_mask(high_peak_mask)
        if peak_dist_map_high is None:
            print(f"Slice {z}: no peaks found")
            continue

        peak_coords = np.argwhere(all_peak_mask)
        high_peak_coords = np.argwhere(high_peak_mask)
        peak_rows.append(
            {
                "slice_index": int(z),
                "n_instances": int(instance_ids.size),
                "n_peaks": int(peak_coords.shape[0]),
                "n_high_peaks": int(high_peak_coords.shape[0]),
                "peak_threshold": float(peak_threshold),
                "high_percentile": float(high_percentile),
                "low_percentile": float(low_percentile),
            }
        )

        observed_dists_high = []
        observed_dists_all = []

        for instance_id in instance_ids:
            inst_mask = label_map == instance_id
            if not np.any(inst_mask):
                continue

            boundary_mask = _instance_boundary_mask(inst_mask)
            by, bx = np.where(boundary_mask)
            if by.size == 0:
                continue

            d_all = peak_dist_map_all[by, bx]
            d_high = peak_dist_map_high[by, bx]
            bmin_all_idx = int(np.argmin(d_all))
            bmin_high_idx = int(np.argmin(d_high))

            dist_boundary_all = float(d_all[bmin_all_idx])
            dist_boundary_high = float(d_high[bmin_high_idx])

            py = int(peak_nn_idx[0, by[bmin_all_idx], bx[bmin_all_idx]])
            px = int(peak_nn_idx[1, by[bmin_all_idx], bx[bmin_all_idx]])
            nearest_peak_intensity_raw = float(thickness_slice[py, px])
            nearest_peak_class = (
                "high" if nearest_peak_intensity_raw >= float(peak_threshold) else "low"
            )

            observed_dists_high.append(dist_boundary_high)
            observed_dists_all.append(dist_boundary_all)
            instance_rows.append(
                {
                    "slice_index": int(z),
                    "instance_id": int(instance_id),
                    "instance_area_pixels": int(np.sum(inst_mask)),
                    "distance_boundary_to_nearest_peak_px": dist_boundary_all,
                    "distance_boundary_to_nearest_peak_microns": float(dist_boundary_all * microns_per_pixel),
                    "distance_boundary_to_nearest_high_peak_px": dist_boundary_high,
                    "distance_boundary_to_nearest_high_peak_microns": float(dist_boundary_high * microns_per_pixel),
                    "nearest_peak_y": int(py),
                    "nearest_peak_x": int(px),
                    "nearest_peak_intensity_raw": nearest_peak_intensity_raw,
                    "nearest_peak_class": nearest_peak_class,
                    "peak_threshold": float(peak_threshold),
                }
            )

        if len(observed_dists_high) == 0:
            continue

        observed_dists_high = np.asarray(observed_dists_high, dtype=np.float64)
        observed_dists_all = np.asarray(observed_dists_all, dtype=np.float64)
        n_obs = int(observed_dists_high.size)

        slice_rows.append(
            {
                "slice_index": int(z),
                "n_instances": int(n_obs),
                "n_peaks": int(peak_coords.shape[0]),
                "n_high_peaks": int(high_peak_coords.shape[0]),
                "mean_boundary_distance_to_nearest_peak_px": float(np.mean(observed_dists_all)),
                "mean_boundary_distance_to_nearest_peak_microns": float(np.mean(observed_dists_all) * microns_per_pixel),
                "median_boundary_distance_to_nearest_peak_px": float(np.median(observed_dists_all)),
                "median_boundary_distance_to_nearest_peak_microns": float(np.median(observed_dists_all) * microns_per_pixel),
                "std_boundary_distance_to_nearest_peak_px": float(np.std(observed_dists_all)),
                "std_boundary_distance_to_nearest_peak_microns": float(np.std(observed_dists_all) * microns_per_pixel),
                "mean_boundary_distance_to_nearest_high_peak_px": float(np.mean(observed_dists_high)),
                "mean_boundary_distance_to_nearest_high_peak_microns": float(np.mean(observed_dists_high) * microns_per_pixel),
                "median_boundary_distance_to_nearest_high_peak_px": float(np.median(observed_dists_high)),
                "median_boundary_distance_to_nearest_high_peak_microns": float(np.median(observed_dists_high) * microns_per_pixel),
                "std_boundary_distance_to_nearest_high_peak_px": float(np.std(observed_dists_high)),
                "std_boundary_distance_to_nearest_high_peak_microns": float(np.std(observed_dists_high) * microns_per_pixel),
                "peak_threshold": float(peak_threshold),
            }
        )

        print(
            f"Slice {z}: instances={n_obs}, peaks={peak_coords.shape[0]}, "
            f"mean_to_high={np.mean(observed_dists_high):.3f}"
        )

    if len(instance_rows) == 0:
        raise RuntimeError("No per-instance rows generated. Check mask and thickness content.")

    per_instance_path = os.path.join(output_dir, "per_instance_peak_distance.csv")
    per_slice_path = os.path.join(output_dir, "per_slice_peak_distance_summary.csv")
    peak_summary_path = os.path.join(output_dir, "peak_detection_summary.csv")

    pd.DataFrame(instance_rows).sort_values(["slice_index", "instance_id"]).to_csv(
        per_instance_path,
        index=False,
    )
    pd.DataFrame(slice_rows).sort_values("slice_index").to_csv(per_slice_path, index=False)
    pd.DataFrame(peak_rows).sort_values("slice_index").to_csv(peak_summary_path, index=False)

    print("\nSaved outputs:")
    print(f"  {per_instance_path}")
    print(f"  {per_slice_path}")
    print(f"  {peak_summary_path}")


def plot_peak_distance_outputs(per_instance_csv, output_dir, slice_idx_to_id, microns_per_pixel):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    inst = pd.read_csv(per_instance_csv)
    inst = inst.copy()
    inst["distance_boundary_to_nearest_high_peak_microns"] = (
        inst["distance_boundary_to_nearest_high_peak_px"] * float(microns_per_pixel)
    )

    inst["slice_label"] = inst["slice_index"].map(
        lambda idx: slice_idx_to_id.get(int(idx), f"Slice {int(idx)}")
    )
    order = [
        slice_idx_to_id.get(int(i), f"Slice {int(i)}")
        for i in sorted(inst["slice_index"].unique())
    ]

    # Remove if instance number <=2

    # for slice_idx in sorted(inst["slice_index"].unique()):
    #     if len(inst[slice_idx]) >= 2:


    plt.figure(figsize=(10, 5.5))
    sns.boxplot(
        data=inst,
        x="slice_label",
        y="distance_boundary_to_nearest_high_peak_microns",
        order=order,
        showfliers=False,
        color="#4C78A8",
    )
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.title("Boundary Distance to Nearest High-Thickness Collagen Region", fontsize=13, fontweight='bold')
    plt.xlabel("Slice", fontsize=12, fontweight='bold')
    plt.ylabel("Boundary Distance (µm)", fontsize=12, fontweight='bold')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.set_facecolor('#f8f8f8')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_boundary_distance_by_slice.png"), dpi=300)
    plt.close()


def summarize_high_vs_low_distance(per_instance_csv, output_dir, slice_idx_to_id, microns_per_pixel):
    from scipy.stats import mannwhitneyu

    os.makedirs(output_dir, exist_ok=True)
    inst = pd.read_csv(per_instance_csv)
    inst = inst.copy()
    inst["distance_boundary_to_nearest_peak_microns"] = (
        inst["distance_boundary_to_nearest_peak_px"] * float(microns_per_pixel)
    )

    summary = (
        inst.groupby(["slice_index", "nearest_peak_class"], dropna=False)
        .agg(
            n_instances=("instance_id", "size"),
            mean_distance_px=("distance_boundary_to_nearest_peak_px", "mean"),
            median_distance_px=("distance_boundary_to_nearest_peak_px", "median"),
            std_distance_px=("distance_boundary_to_nearest_peak_px", "std"),
            mean_distance_microns=("distance_boundary_to_nearest_peak_microns", "mean"),
            median_distance_microns=("distance_boundary_to_nearest_peak_microns", "median"),
            std_distance_microns=("distance_boundary_to_nearest_peak_microns", "std"),
        )
        .reset_index()
    )

    summary["std_distance_px"] = summary["std_distance_px"].fillna(0.0)
    summary["std_distance_microns"] = summary["std_distance_microns"].fillna(0.0)

    summary["slice_label"] = summary["slice_index"].map(
        lambda idx: slice_idx_to_id.get(int(idx), f"Slice {int(idx)}")
    )

    def _p_to_stars(p_value):
        if not np.isfinite(p_value):
            return "n/a"
        if p_value <= 0.001:
            return "***"
        if p_value <= 0.01:
            return "**"
        if p_value <= 0.05:
            return "*"
        return "ns"


    p_rows = []
    for slice_idx in sorted(inst["slice_index"].unique()):
        low_vals = inst[
            (inst["slice_index"] == slice_idx) & (inst["nearest_peak_class"] == "low")
        ]["distance_boundary_to_nearest_peak_microns"].to_numpy(dtype=np.float64)
        high_vals = inst[
            (inst["slice_index"] == slice_idx) & (inst["nearest_peak_class"] == "high")
        ]["distance_boundary_to_nearest_peak_microns"].to_numpy(dtype=np.float64)

        if low_vals.size > 0 and high_vals.size > 0:
            _, p_value = mannwhitneyu(low_vals, high_vals, alternative="two-sided")
            p_value = float(p_value)
        else:
            p_value = np.nan

        p_rows.append(
            {
                "slice_index": int(slice_idx),
                "p_value_low_vs_high": p_value,
                "significance": _p_to_stars(p_value),
            }
        )

    p_df = pd.DataFrame(p_rows)
    summary = summary.merge(p_df, on="slice_index", how="left")

    out_csv = os.path.join(output_dir, "per_slice_mean_distance_high_vs_low.csv")
    summary.sort_values(["slice_index", "nearest_peak_class"]).to_csv(out_csv, index=False)

    order = [
        slice_idx_to_id.get(int(i), f"Slice {int(i)}")
        for i in sorted(summary["slice_index"].unique())
    ]
    plot_inst = inst.copy()
    plot_inst["slice_label"] = plot_inst["slice_index"].map(
        lambda idx: slice_idx_to_id.get(int(idx), f"Slice {int(idx)}")
    )

    # Use shared boxplot function with significance brackets
    out_png = _plot_boxplot_with_significance_brackets(
        plot_data=plot_inst,
        x_col="slice_label",
        y_col="distance_boundary_to_nearest_peak_microns",
        hue_col="nearest_peak_class",
        order=order,
        hue_order=["low", "high"],
        palette={"low": "#72B7B2", "high": "#E45756"},
        p_df=p_df,
        legend_title="Fiber Thickness",
        plot_title="Distance per Cell: Fiber Thickness",
        y_label="Distance (µm)",
        output_dir=output_dir,
        output_filename="mean_distance_high_vs_low_by_slice.png"
    )

    print("\nSaved high-vs-low summary outputs:")
    print(f"  {out_csv}")
    print(f"  {out_png}")


def main():
    # ---------------------------------------------------------------------
    # Update these paths directly before running.
    # ---------------------------------------------------------------------
    mask_tif = r"D:\integrated_pipeline\HSI_data\journal_figures\K198_3-immune_cells-1.tif"
    thickness_tif = r"D:\integrated_pipeline\HSI_data\journal_figures\collagen_output\thickness_map.tif"
    output_dir = os.path.join(os.path.dirname(thickness_tif),"distance_to_thickness")
    # Peak definition parameters (post hoc transform behavior).
    high_percentile = 95.0
    low_percentile = 90.0

    # slice_idx_to_id = {
    #     0: "ATL",
    #     1: "CNT",
    #     2: "DCT",
    #     3: "DTL",
    #     4: "IC",
    #     5: "POD",
    #     6: "PT",
    #     7: "TAL",
    #     8: "altATL",
    #     9: "altCNT",
    #     10: "altDCT",
    #     12: "altDTL",
    #     13: "altIC",
    #     14: "altPT",
    #     15: "altTAL"
    # }

    slice_idx_to_id = {
        0: "Macrophages",
        1: "T Cells (CD4+)",
        2: "T Cells (CD8+)",
        3: "Immune Cells (unlabeled)"
    }

    microns_per_pixel = 0.65

    analyze_instance_distance_to_peaks(
        mask_tif=mask_tif,
        thickness_tif=thickness_tif,
        output_dir=output_dir,
        high_percentile=high_percentile,
        low_percentile=low_percentile,
        microns_per_pixel=microns_per_pixel,
    )

    plot_peak_distance_outputs(
        per_instance_csv=os.path.join(output_dir, "per_instance_peak_distance.csv"),
        output_dir=output_dir,
        slice_idx_to_id=slice_idx_to_id,
        microns_per_pixel=microns_per_pixel,
    )

    summarize_high_vs_low_distance(
        per_instance_csv=os.path.join(output_dir, "per_instance_peak_distance.csv"),
        output_dir=output_dir,
        slice_idx_to_id=slice_idx_to_id,
        microns_per_pixel=microns_per_pixel,
    )


if __name__ == "__main__":
    main()
