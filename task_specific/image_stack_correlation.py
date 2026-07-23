#!/usr/bin/env python3
"""
Image Stack Pixel Correlation Plotter
=====================================
This script analyzes the pixel-wise correlation between two 3D image stacks (TIF format),
where each slice represents a specific molecule defined by a dictionary mapping.

It filters out zero-valued pixels (background), calculates correlation statistics
(Pearson correlation), downsamples pixels for plotting to prevent performance issues,
and plots the correlation with a prominent marker representing the mean of all non-zero
pixels in that slice.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from scipy.stats import pearsonr


# =========================================================================
# CONFIGURATION
# =========================================================================
CONFIG = {
    # Hardcode the paths to your 3D image stacks here
    'image_a_path': r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\Normalized_xenium-HSI_sam_outputs_molecules\aki-roi1-HSI\Result of aki-roi1-HSI_sam_similarity.tif",
    'image_b_path': r"D:\ADATA Backup\HuBMAP\HuBMAP Xenium\Xenium HSI\Normalized_xenium-HSI_sam_outputs_molecules\aki-roi1_qc-HSI\Result of aki-roi1_qc-HSI_sam_similarity.tif",
    
    # Hardcode the molecule mapping dictionary here
    # (Slice index as integer -> Molecule name string)
    'molecule_mapping': {
"0":	"16:0 CDP DG",
"1":	"18:2 Cardiolipin",
"2":	"18:0 LPA",
"3":	"18:1 Cholesterol Ester",
"4":	"18:1 LPA",
"5":	"C12 Carnitine",
"6":	"C16 Carnitine",
"7":	"C18 Carnitine (d9-cis)",
"8":	"C8 L-Carnitine",
"9":	"Cer 18:1-12:0",
"10":	"Cer 18:1-18:0",
"11":	"Cer 18:1-18:1",
"12":	"Cer 18:1-22:0",
"13":	"Cer 18:1-24:0",
"14":	"Cer 18:1-24:1",
"15":	"Cholesterol",
"16":	"DAG 16:0",
"17":	"DOPC",
"18":	"DOPE",
"19":	"DSPC",
"20":	"DHA",
"21":	"Glucosylceramide",
"22":	"Background",
"23":	"PC Mix",
"24":	"PE Mix",
"25":	"PG Mix",
"26":	"PI Mix",
"27":	"PS Mix",
"28":	"Palmitic Acid",
"29":	"Sphingosine Mix",
"30":	"Stearic Acid",
"31":	"TAG 16:0",
"32":	"TAG Mix",
"33":	"TPA"

    },
    
    # Execution options
    'output_plot_path': "correlation_plot.png",
    'output_csv_path': "correlation_stats.csv",
    'num_samples': 10,        # Number of pixels to sample per slice for plotting
    'zero_handling': 'any',    # 'any' (exclude if 0 in either image), 'all' (exclude if 0 in both), 'none'
    'show_regression': True,
    'plot_title': None         # Custom title for plot
}


def parse_mapping(mapping_str):
    """
    Parses the molecule mapping from a JSON string or a file path.
    Supported file formats: JSON, or CSV/TXT with 'key,value' format.
    """
    if not mapping_str:
        return {}
    
    # Check if it looks like inline JSON
    trimmed = mapping_str.strip()
    if trimmed.startswith('{') and trimmed.endswith('}'):
        try:
            return json.loads(trimmed)
        except Exception as e:
            print(f"Error parsing inline JSON mapping: {e}")
            sys.exit(1)
            
    # Otherwise, treat as file path
    if not os.path.exists(mapping_str):
        print(f"Error: Mapping file/string '{mapping_str}' not found or invalid JSON.")
        sys.exit(1)
        
    try:
        if mapping_str.endswith('.json'):
            with open(mapping_str, 'r') as f:
                return json.load(f)
        else:
            # Try to parse as CSV/TXT with key-value pairs
            mapping = {}
            with open(mapping_str, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    # Split on first comma or colon
                    for sep in [',', ':']:
                        if sep in line:
                            parts = line.split(sep, 1)
                            key = parts[0].strip()
                            val = parts[1].strip()
                            # Strip quotes if present
                            key = key.strip('"\'')
                            val = val.strip('"\'')
                            mapping[key] = val
                            break
            if not mapping:
                raise ValueError("No key-value pairs detected.")
            return mapping
    except Exception as e:
        print(f"Error reading mapping file '{mapping_str}': {e}")
        sys.exit(1)


def analyze_and_plot_correlations(
    image_a_path,
    image_b_path,
    mapping,
    output_plot_path,
    output_csv_path=None,
    max_samples=100,
    zero_handling='any',
    show_regression=True,
    title=None
):
    print(f"Loading Image A: {image_a_path}")
    img_a = tifffile.imread(image_a_path)
    
    print(f"Loading Image B: {image_b_path}")
    img_b = tifffile.imread(image_b_path)
    
    if img_a.shape != img_b.shape:
        print(f"Error: Image shapes do not match: A={img_a.shape}, B={img_b.shape}")
        sys.exit(1)
        
    if img_a.ndim != 3:
        print(f"Error: Images must be 3D stacks (Slices, Height, Width). Found dimensions: {img_a.ndim}")
        sys.exit(1)
        
    num_slices, height, width = img_a.shape
    print(f"Image dimensions: {num_slices} slices, {height}x{width} pixels")
    
    # Process slice by slice
    slice_data = []
    plot_points = []
    accumulated_a = []
    accumulated_b = []
    
    # Set seed for reproducible downsampling
    rng = np.random.default_rng(42)
    
    for s in range(num_slices):
        slice_a = img_a[s]
        slice_b = img_b[s]
        
        # Get molecule name from mapping (keys can be string or int)
        mol_name = mapping.get(str(s), mapping.get(s, f"Slice {s}"))
        
        # Define mask to ignore zeros and values exactly equal to 0.5
        if zero_handling == 'any':
            # Exclude pixel if it is 0 or 0.5 in either image
            valid_mask = (slice_a != 0.0) & (slice_a != 0.5) & (slice_b != 0.0) & (slice_b != 0.5)
        elif zero_handling == 'all':
            # Exclude pixel if it is (0 or 0.5) in both images
            is_invalid_a = (slice_a == 0.0) | (slice_a == 0.5)
            is_invalid_b = (slice_b == 0.0) | (slice_b == 0.5)
            valid_mask = ~(is_invalid_a & is_invalid_b)
        else:
            # Exclude pixel if it is exactly 0.5 in either image
            valid_mask = (slice_a != 0.5) & (slice_b != 0.5)
            
        vals_a = slice_a[valid_mask]
        vals_b = slice_b[valid_mask]
        
        num_valid = len(vals_a)
        
        if num_valid < 2:
            print(f"Warning: Slice {s} ({mol_name}) has fewer than 2 valid non-zero pixels. Skipping.")
            continue
            
        # Accumulate all valid pixels for overall correlation
        accumulated_a.append(vals_a)
        accumulated_b.append(vals_b)
            
        # Calculate statistics on the entire population of valid pixels
        mean_a = np.mean(vals_a)
        mean_b = np.mean(vals_b)
        std_a = np.std(vals_a)
        std_b = np.std(vals_b)
        
        # Pearson correlation
        try:
            r_val, p_val = pearsonr(vals_a, vals_b)
        except Exception as e:
            print(f"Warning: Could not compute correlation for Slice {s} ({mol_name}): {e}")
            r_val, p_val = np.nan, np.nan
            
        slice_data.append({
            'slice_index': s,
            'molecule': mol_name,
            'valid_pixels': num_valid,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'std_a': std_a,
            'std_b': std_b,
            'pearson_r': r_val,
            'p_value': p_val
        })
        
        # Downsample for visualization
        if num_valid <= max_samples:
            sampled_a = vals_a
            sampled_b = vals_b
        else:
            indices = rng.choice(num_valid, size=max_samples, replace=False)
            sampled_a = vals_a[indices]
            sampled_b = vals_b[indices]
            
        for va, vb in zip(sampled_a, sampled_b):
            plot_points.append({
                'slice_index': s,
                'molecule': mol_name,
                'val_a': va,
                'val_b': vb,
                'is_mean': False
            })
            
        # Add the mean point as a special record in plot points if needed,
        # or handle it separately during plotting (better to keep distinct)
        print(f"Processed Slice {s} ({mol_name}): {num_valid} valid pixels, r = {r_val:.4f}, mean_a = {mean_a:.2f}, mean_b = {mean_b:.2f}")

    df_stats = pd.DataFrame(slice_data)
    df_points = pd.DataFrame(plot_points)
    
    if df_stats.empty:
        print("Error: No slices with valid pixels to analyze.")
        sys.exit(1)
        
    # Save statistics CSV if requested
    if output_csv_path:
        df_stats.to_csv(output_csv_path, index=False)
        print(f"Saved correlation statistics to {output_csv_path}")
        
    # Plotting configuration
    sns.set_theme(style="ticks")
    num_valid_slices = len(df_stats)
    
    # Calculate overall correlation and fit across all valid pixels in the images (un-downsampled)
    all_valid_a = np.concatenate(accumulated_a)
    all_valid_b = np.concatenate(accumulated_b)
    total_valid = len(all_valid_a)
    
    overall_r, overall_p = np.nan, np.nan
    if total_valid > 1:
        try:
            overall_r, overall_p = pearsonr(all_valid_a, all_valid_b)
            print(f"Overall correlation across all valid pixels (N={total_valid:,}): r = {overall_r:.4f}, p = {overall_p:.4e}")
        except Exception as e:
            print(f"Warning: Could not compute correlation across all valid pixels: {e}")
            
    plt.figure(figsize=(11, 8.5))
    palette = sns.color_palette("husl", num_valid_slices)
    
    for idx, row in df_stats.iterrows():
        s_idx = row['slice_index']
        mol_name = row['molecule']
        m_a = row['mean_a']
        m_b = row['mean_b']
        color = palette[idx]
        
        pts = df_points[df_points['slice_index'] == s_idx]
        
        # Plot sampled points with transparent color (to show density/spread)
        plt.scatter(
            pts['val_a'], pts['val_b'],
            alpha=0.15, s=15, color=color, edgecolors='none',
            label=None  # Avoid adding 34 separate entries for raw pixels to the legend
        )
        
        # Plot the mean point with a matching solid color and a prominent outline
        plt.scatter(
            [m_a], [m_b],
            marker='D', color=color, s=120, edgecolors='black', linewidths=1.2,
            zorder=10, label=mol_name
        )
        
    # Calculate pixel-wise differences
    diffs = all_valid_b - all_valid_a
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    rmse = np.sqrt(np.mean(diffs**2))
    
    # Calculate percentage of pixels within mean_diff +/- 2 * std_diff
    within_bounds = (diffs >= mean_diff - 2 * std_diff) & (diffs <= mean_diff + 2 * std_diff)
    pct_within = np.mean(within_bounds) * 100.0
    
    print(f"Comparison: Mean Diff = {mean_diff:.5f}, SD of Diff = {std_diff:.5f}, RMSE = {rmse:.5f}")
    print(f"Percentage of pixels within Mean ± 2 SD: {pct_within:.2f}%")
    
    # Determine x range from displayed points for visual plotting limits
    all_sampled_a = df_points['val_a'].values
    x_min, x_max = all_sampled_a.min(), all_sampled_a.max()
    padding = (x_max - x_min) * 0.05 if x_max != x_min else 0.1
    x_vals = np.linspace(x_min - padding, x_max + padding, 100)
    
    # 1. Plot the identity line (y = x)
    plt.plot(
        x_vals, x_vals,
        color='red', linestyle=':', linewidth=2.0, zorder=5,
        label=None  # Removed from legend
    )
    
    # 2. Plot the shaded standard deviation band around the identity line
    plt.fill_between(
        x_vals, x_vals - std_diff, x_vals + std_diff,
        color='gray', alpha=0.15, zorder=4,
        label=None  # Removed from legend
    )
    # Add a bounded dashed box for values below x, y = 0.7
    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()
    x_start = min(0.0, xlim[0])
    y_start = min(0.0, ylim[0])
    
    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (x_start, y_start), 0.7 - x_start, 0.7 - y_start,
        linewidth=1.5, edgecolor='gray', facecolor='none', linestyle='--',
        zorder=2, label=None  # Removed from legend
    )
    plt.gca().add_patch(rect)
    
    # Restore original limits to keep plot zoomed to data points
    plt.gca().set_xlim(xlim)
    plt.gca().set_ylim(ylim)

        
    plot_title = title or (
        f"Pixel-wise Correlation & Identity Check\n"
        f"Mean Diff: {mean_diff:.4f} ± {std_diff:.4f} ({pct_within:.2f}% within ±2 SD)"
    )
    
    plt.title(plot_title, fontsize=24, fontweight='bold')
    plt.xlabel("First Acq. ROI Analyte Similarity", fontsize=20, fontweight='bold')
    plt.ylabel("QC ROI Analyte Similarity", fontsize=20, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Despine the plot (remove top and right spines)
    sns.despine()
    
    # Make tick marks longer and format tick labels
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().tick_params(direction='out', length=8, width=1.5, labelsize=16)
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    # Put legend outside to avoid overlap with points (divided into two columns)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0., fontsize=10, ncol=2, markerscale=0.7)
    
    # Save plot with tight bbox
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation plot to {output_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate and plot pixel-wise correlation between two 3D TIF image stacks."
    )
    parser.add_argument("-a", "--image-a", default=None, help="Path to first 3D TIF image stack")
    parser.add_argument("-b", "--image-b", default=None, help="Path to second 3D TIF image stack")
    parser.add_argument(
        "-m", "--mapping", default=None,
        help="JSON string or path to JSON/TXT file mapping slices to molecules"
    )
    parser.add_argument("-o", "--output-plot", default=CONFIG['output_plot_path'], help="Path to save output plot")
    parser.add_argument("--output-csv", default=CONFIG['output_csv_path'], help="Path to save correlation statistics CSV")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=CONFIG['num_samples'],
        help="Number of random pixels to sample per slice for the scatter plot (default: 100)"
    )
    parser.add_argument(
        "--zero-handling", choices=["any", "all", "none"], default=CONFIG['zero_handling'],
        help="How to handle zero pixels: 'any' (ignore if zero in either image), 'all' (ignore if zero in both), 'none' (default: any)"
    )
    parser.add_argument("--no-regression", action="store_true", help="Disable plotting of regression lines")
    parser.add_argument("--title", default=CONFIG['plot_title'], help="Custom title for plot")

    args = parser.parse_args()
    
    # Determine image paths (CLI overrides CONFIG)
    image_a = args.image_a or CONFIG['image_a_path']
    image_b = args.image_b or CONFIG['image_b_path']
    
    if not image_a or not image_b:
        print("Error: Image paths must be specified in the CONFIG block at the top of the file or via command-line arguments (-a and -b).")
        sys.exit(1)
        
    # Determine mapping (CLI overrides CONFIG)
    if args.mapping:
        mapping = parse_mapping(args.mapping)
        print(f"Loaded mapping from command line with {len(mapping)} molecule definitions.")
    else:
        mapping = CONFIG['molecule_mapping']
        if not mapping:
            print("Error: Molecule mapping must be specified in the CONFIG block at the top of the file or via command-line mapping (-m).")
            sys.exit(1)
        # Ensure mapping keys are accessible as strings/ints
        # (CONFIG might define them as ints, but string lookup is also supported)
        print(f"Using hardcoded molecule mapping from CONFIG with {len(mapping)} definitions.")
        
    # Run analysis
    analyze_and_plot_correlations(
        image_a_path=image_a,
        image_b_path=image_b,
        mapping=mapping,
        output_plot_path=args.output_plot,
        output_csv_path=args.output_csv,
        max_samples=args.num_samples,
        zero_handling=args.zero_handling,
        show_regression=not args.no_regression,
        title=args.title
    )


if __name__ == "__main__":
    main()
