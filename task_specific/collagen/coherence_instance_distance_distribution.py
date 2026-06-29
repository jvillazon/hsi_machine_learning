"""
Compute per-instance boundary distance to low/high coherence regions.

Given:
1) An instance mask image/stack (2D or 3D), where nonzero pixels are cell masks
2) A coherence TIFF (2D or 3D)

This script:
- Detects all valid coherence pixels (finite and > 0)
- Splits coherence into low/high classes using the 50th percentile (median)
- Computes distance from each cell instance boundary to nearest low and high coherence pixels
- Summarizes distances by slice and class, and plots low vs high distributions

Outputs:
- per_instance_coherence_distance.csv
- per_slice_coherence_distance_summary.csv
- coherence_thresholds_used.csv
- mean_boundary_distance_to_high_coherence_by_slice.png
- per_slice_mean_distance_low_vs_high_coherence.csv
- mean_distance_low_vs_high_coherence_by_slice.png

Usage:
- Edit paths in main(), then run:
    python task_specific/collagen/coherence_instance_distance_distribution.py
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


def _align_stack_depth(mask_stack, coherence_stack):
    """Match Z-depth if one stack has singleton depth."""
    mz = int(mask_stack.shape[0])
    cz = int(coherence_stack.shape[0])
    if mz == cz:
        return mask_stack, coherence_stack
    if mz > 1 and cz == 1:
        return mask_stack, np.repeat(coherence_stack, mz, axis=0)
    if cz > 1 and mz == 1:
        return np.repeat(mask_stack, cz, axis=0), coherence_stack
    raise ValueError(
        "Mask and coherence stack depths are incompatible. "
        f"Got mask={mask_stack.shape}, coherence={coherence_stack.shape}."
    )


def _validate_shapes(mask_stack, coherence_stack):
    if mask_stack.shape[1:] != coherence_stack.shape[1:]:
        raise ValueError(
            "Mask and coherence must have the same spatial shape after stack normalization. "
            f"Got mask={mask_stack.shape}, coherence={coherence_stack.shape}."
        )


def _instance_labels_for_slice(mask_slice):
    labeled, n = ndi.label(mask_slice > 0)
    if n == 0:
        return None, np.array([], dtype=np.int64)
    ids = np.arange(1, n + 1, dtype=np.int64)
    return labeled, ids


def _instance_boundary_mask(inst_mask):
    eroded = ndi.binary_erosion(inst_mask)
    boundary = inst_mask & ~eroded
    if not np.any(boundary):
        boundary = inst_mask
    return boundary


def _coherence_classes(coherence_stack):
    valid = np.isfinite(coherence_stack) & (coherence_stack > 0)
    if not np.any(valid):
        raise ValueError("Coherence image contains no nonzero finite values.")

    values = coherence_stack[valid]
    # threshold = float(np.quantile(values, 0.5))
    threshold = 0.5 * (float(np.min(values)) + float(np.max(values)))
    low_mask = valid & (coherence_stack <= threshold)
    high_mask = valid & (coherence_stack > threshold)

    return {
        "valid": valid,
        "low": low_mask,
        "high": high_mask,
    }, {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "t50": threshold,
        "valid_pixel_count": int(values.size),
        "low_pixel_count": int(np.sum(low_mask)),
        "high_pixel_count": int(np.sum(high_mask)),
    }


def analyze_instance_distance_to_coherence(
    mask_tif,
    coherence_tif,
    output_dir,
    microns_per_pixel=0.65,
):
    os.makedirs(output_dir, exist_ok=True)

    mask_stack = _ensure_stack(tifffile.imread(mask_tif))
    coherence_stack = _ensure_stack(tifffile.imread(coherence_tif)).astype(np.float32)
    mask_stack, coherence_stack = _align_stack_depth(mask_stack, coherence_stack)
    _validate_shapes(mask_stack, coherence_stack)

    class_masks, thresholds = _coherence_classes(coherence_stack)

    print("Coherence thresholds:")
    print(f"  min={thresholds['min']:.6f}")
    print(f"  t50={thresholds['t50']:.6f}")
    print(f"  max={thresholds['max']:.6f}")

    instance_rows = []
    slice_rows = []

    for z in range(mask_stack.shape[0]):
        mask_slice = mask_stack[z]
        label_map, instance_ids = _instance_labels_for_slice(mask_slice)
        if instance_ids.size == 0:
            print(f"Slice {z}: no instances found")
            continue

        valid_slice = class_masks["valid"][z]
        low_slice = class_masks["low"][z]
        high_slice = class_masks["high"][z]

        if not np.any(valid_slice):
            print(f"Slice {z}: no valid coherence pixels")
            continue

        dist_to_valid, nn_valid = ndi.distance_transform_edt(~valid_slice, return_indices=True)
        dist_to_low = ndi.distance_transform_edt(~low_slice) if np.any(low_slice) else None
        dist_to_high = ndi.distance_transform_edt(~high_slice) if np.any(high_slice) else None

        observed_low = []
        observed_high = []

        for instance_id in instance_ids:
            inst_mask = label_map == instance_id
            if not np.any(inst_mask):
                continue

            boundary_mask = _instance_boundary_mask(inst_mask)
            by, bx = np.where(boundary_mask)
            if by.size == 0:
                continue

            if dist_to_low is None:
                dist_low = np.nan
            else:
                dist_low = float(np.min(dist_to_low[by, bx]))

            if dist_to_high is None:
                dist_high = np.nan
            else:
                dist_high = float(np.min(dist_to_high[by, bx]))

            bmin_valid_idx = int(np.argmin(dist_to_valid[by, bx]))
            py = int(nn_valid[0, by[bmin_valid_idx], bx[bmin_valid_idx]])
            px = int(nn_valid[1, by[bmin_valid_idx], bx[bmin_valid_idx]])
            nearest_coherence = float(coherence_stack[z, py, px])
            nearest_class = "high" if nearest_coherence > thresholds["t50"] else "low"

            if np.isfinite(dist_low):
                observed_low.append(dist_low)
            if np.isfinite(dist_high):
                observed_high.append(dist_high)

            instance_rows.append(
                {
                    "slice_index": int(z),
                    "instance_id": int(instance_id),
                    "instance_area_pixels": int(np.sum(inst_mask)),
                    "distance_boundary_to_nearest_low_coherence_px": dist_low,
                    "distance_boundary_to_nearest_low_coherence_microns": float(dist_low * microns_per_pixel) if np.isfinite(dist_low) else np.nan,
                    "distance_boundary_to_nearest_high_coherence_px": dist_high,
                    "distance_boundary_to_nearest_high_coherence_microns": float(dist_high * microns_per_pixel) if np.isfinite(dist_high) else np.nan,
                    "nearest_coherence_y": int(py),
                    "nearest_coherence_x": int(px),
                    "nearest_coherence_value": nearest_coherence,
                    "nearest_coherence_class": nearest_class,
                    "coherence_t50_threshold": float(thresholds["t50"]),
                }
            )

        if len(observed_low) == 0 and len(observed_high) == 0:
            continue

        low_arr = np.asarray(observed_low, dtype=np.float64)
        high_arr = np.asarray(observed_high, dtype=np.float64)

        slice_rows.append(
            {
                "slice_index": int(z),
                "n_instances": int(len(instance_ids)),
                "mean_boundary_distance_to_nearest_low_coherence_px": float(np.mean(low_arr)) if low_arr.size > 0 else np.nan,
                "mean_boundary_distance_to_nearest_low_coherence_microns": float(np.mean(low_arr) * microns_per_pixel) if low_arr.size > 0 else np.nan,
                "median_boundary_distance_to_nearest_low_coherence_px": float(np.median(low_arr)) if low_arr.size > 0 else np.nan,
                "median_boundary_distance_to_nearest_low_coherence_microns": float(np.median(low_arr) * microns_per_pixel) if low_arr.size > 0 else np.nan,
                "std_boundary_distance_to_nearest_low_coherence_px": float(np.std(low_arr)) if low_arr.size > 0 else np.nan,
                "std_boundary_distance_to_nearest_low_coherence_microns": float(np.std(low_arr) * microns_per_pixel) if low_arr.size > 0 else np.nan,
                "mean_boundary_distance_to_nearest_high_coherence_px": float(np.mean(high_arr)) if high_arr.size > 0 else np.nan,
                "mean_boundary_distance_to_nearest_high_coherence_microns": float(np.mean(high_arr) * microns_per_pixel) if high_arr.size > 0 else np.nan,
                "median_boundary_distance_to_nearest_high_coherence_px": float(np.median(high_arr)) if high_arr.size > 0 else np.nan,
                "median_boundary_distance_to_nearest_high_coherence_microns": float(np.median(high_arr) * microns_per_pixel) if high_arr.size > 0 else np.nan,
                "std_boundary_distance_to_nearest_high_coherence_px": float(np.std(high_arr)) if high_arr.size > 0 else np.nan,
                "std_boundary_distance_to_nearest_high_coherence_microns": float(np.std(high_arr) * microns_per_pixel) if high_arr.size > 0 else np.nan,
                "coherence_t50_threshold": float(thresholds["t50"]),
            }
        )

        msg_high = float(np.mean(high_arr)) if high_arr.size > 0 else np.nan
        print(
            f"Slice {z}: instances={len(instance_ids)}, mean_to_high={msg_high:.3f}"
        )

    if len(instance_rows) == 0:
        raise RuntimeError("No per-instance rows generated. Check mask and coherence content.")

    per_instance_path = os.path.join(output_dir, "per_instance_coherence_distance.csv")
    per_slice_path = os.path.join(output_dir, "per_slice_coherence_distance_summary.csv")
    threshold_path = os.path.join(output_dir, "coherence_thresholds_used.csv")

    pd.DataFrame(instance_rows).sort_values(["slice_index", "instance_id"]).to_csv(
        per_instance_path,
        index=False,
    )
    pd.DataFrame(slice_rows).sort_values("slice_index").to_csv(per_slice_path, index=False)
    pd.DataFrame([thresholds]).to_csv(threshold_path, index=False)

    print("\nSaved outputs:")
    print(f"  {per_instance_path}")
    print(f"  {per_slice_path}")
    print(f"  {threshold_path}")

    return thresholds


def plot_boundary_distance_to_high_coherence(per_instance_csv, output_dir, slice_idx_to_id):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    inst = pd.read_csv(per_instance_csv)
    inst = inst.copy()
    inst["slice_label"] = inst["slice_index"].map(
        lambda idx: slice_idx_to_id.get(int(idx), f"Slice {int(idx)}")
    )
    order = [
        slice_idx_to_id.get(int(i), f"Slice {int(i)}")
        for i in sorted(inst["slice_index"].unique())
    ]

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=inst,
        x="slice_label",
        y="distance_boundary_to_nearest_high_coherence_microns",
        order=order,
        showfliers=False,
        color="#4C78A8",
    )
    plt.xticks(rotation=0, ha="center", fontsize=10)
    plt.title("Boundary Distance to Nearest High-Coherence Region", fontsize=14, fontweight='bold')
    plt.xlabel("Slice", fontsize=12, fontweight='bold')
    plt.ylabel("Boundary Distance to Nearest High Coherence (microns)", fontsize=12, fontweight='bold')
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    ax.set_facecolor('#f8f8f8')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mean_boundary_distance_to_high_coherence_by_slice.png"), dpi=300)
    plt.close()


def summarize_low_vs_high_coherence_distance(
    per_instance_csv,
    output_dir,
    slice_idx_to_id,
    threshold_t50,
    microns_per_pixel=0.65,
):
    from scipy.stats import mannwhitneyu

    os.makedirs(output_dir, exist_ok=True)
    inst = pd.read_csv(per_instance_csv)
    inst = inst.copy()
    
    # Recalculate microns from pixels to ensure consistent conversion factor
    inst["distance_boundary_to_nearest_low_coherence_microns"] = (
        inst["distance_boundary_to_nearest_low_coherence_px"] * float(microns_per_pixel)
    )
    inst["distance_boundary_to_nearest_high_coherence_microns"] = (
        inst["distance_boundary_to_nearest_high_coherence_px"] * float(microns_per_pixel)
    )

    long_df = pd.concat(
        [
            inst[["slice_index", "instance_id", "distance_boundary_to_nearest_low_coherence_microns"]]
            .rename(columns={"distance_boundary_to_nearest_low_coherence_microns": "distance_microns"})
            .assign(coherence_class="low"),
            inst[["slice_index", "instance_id", "distance_boundary_to_nearest_high_coherence_microns"]]
            .rename(columns={"distance_boundary_to_nearest_high_coherence_microns": "distance_microns"})
            .assign(coherence_class="high"),
        ],
        ignore_index=True,
    )
    long_df = long_df[np.isfinite(long_df["distance_microns"])].copy()

    summary = (
        long_df.groupby(["slice_index", "coherence_class"], dropna=False)
        .agg(
            n_instances=("instance_id", "size"),
            mean_distance_microns=("distance_microns", "mean"),
            median_distance_microns=("distance_microns", "median"),
            std_distance_microns=("distance_microns", "std"),
        )
        .reset_index()
    )
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
    for slice_idx in sorted(long_df["slice_index"].unique()):
        low_vals = long_df[
            (long_df["slice_index"] == slice_idx) & (long_df["coherence_class"] == "low")
        ]["distance_microns"].to_numpy(dtype=np.float64)
        high_vals = long_df[
            (long_df["slice_index"] == slice_idx) & (long_df["coherence_class"] == "high")
        ]["distance_microns"].to_numpy(dtype=np.float64)

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

    out_csv = os.path.join(output_dir, "per_slice_mean_distance_low_vs_high_coherence.csv")
    summary.sort_values(["slice_index", "coherence_class"]).to_csv(out_csv, index=False)

    order = [
        slice_idx_to_id.get(int(i), f"Slice {int(i)}")
        for i in sorted(summary["slice_index"].unique())
    ]
    plot_df = long_df.copy()
    plot_df["slice_label"] = plot_df["slice_index"].map(
        lambda idx: slice_idx_to_id.get(int(idx), f"Slice {int(idx)}")
    )

    # Create hue labels for legend
    low_label = "Low Coherence <= mean"
    high_label = "High Coherence > mean"
    label_map = {"low": low_label, "high": high_label}
    plot_df["coherence_label"] = plot_df["coherence_class"].map(label_map)
    hue_order = [low_label, high_label]
    palette = {
        low_label: "#72B7B2",
        high_label: "#E45756",
    }

    # Use shared boxplot function with significance brackets
    out_png = _plot_boxplot_with_significance_brackets(
        plot_data=plot_df,
        x_col="slice_label",
        y_col="distance_microns",
        hue_col="coherence_label",
        order=order,
        hue_order=hue_order,
        palette=palette,
        p_df=p_df,
        legend_title="Coherence Class",
        plot_title="Distance per Cell: Collagen Coherence",
        y_label="Distance (µm)",
        output_dir=output_dir,
        output_filename="mean_distance_low_vs_high_coherence_by_slice.png"
    )

    print("\nSaved low-vs-high coherence summary outputs:")
    print(f"  {out_csv}")
    print(f"  {out_png}")


def main():
    # ---------------------------------------------------------------------
    # Update these paths directly before running.
    # ---------------------------------------------------------------------
    mask_tif = r"D:\integrated_pipeline\HSI_data\journal_figures\K198_3-immune_cells-1.tif"
    coherence_tif = r"D:\integrated_pipeline\HSI_data\journal_figures\collagen_output\coherence_heatmap.tif"
    output_dir = r"D:\integrated_pipeline\HSI_data\journal_figures\collagen_output\distance_to_coherence"

    slice_idx_to_id = {
        0: "Macrophages",
        1: "T cells (CD4+)",
        2: "T cells (CD8+)",
        3: "Immune cells",
    }

    microns_per_pixel = 0.65

    thresholds = analyze_instance_distance_to_coherence(
        mask_tif=mask_tif,
        coherence_tif=coherence_tif,
        output_dir=output_dir,
        microns_per_pixel=microns_per_pixel,
    )

    plot_boundary_distance_to_high_coherence(
        per_instance_csv=os.path.join(output_dir, "per_instance_coherence_distance.csv"),
        output_dir=output_dir,
        slice_idx_to_id=slice_idx_to_id,
    )

    summarize_low_vs_high_coherence_distance(
        per_instance_csv=os.path.join(output_dir, "per_instance_coherence_distance.csv"),
        output_dir=output_dir,
        slice_idx_to_id=slice_idx_to_id,
        threshold_t50=float(thresholds["t50"]),
        microns_per_pixel=microns_per_pixel,
    )


if __name__ == "__main__":
    main()
