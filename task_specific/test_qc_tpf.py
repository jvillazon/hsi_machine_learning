#!/usr/bin/env python3
"""
========================================================================================
Two-Photon Autofluorescence (TPAF) Quality Control (QC) & Redox Ratio Suite
========================================================================================
Automated QC and Optical Redox Ratio analysis for 2-channel TPAF images (.oir / .tif):
1. Ingests Olympus FluoView .oir files (via oirfile) and TIFF files.
2. Standardizes layout to (2, H, W):
   - Channel 0: FAD (Flavin Adenine Dinucleotide)
   - Channel 1: NADH (Nicotinamide Adenine Dinucleotide)
3. Computes automated tissue masking (Otsu/Triangle on combined intensity + morphological closing/dilation).
4. Calculates Signal-to-Noise Ratio (SNR) comparing masked tissue to unmasked background.
5. Quantifies Optical Redox Ratio: FAD / (FAD + NADH) strictly on valid tissue pixels,
   preventing negative ratios or division-by-zero artifacts.
6. Runs non-parametric Kruskal-Wallis omnibus & Dunn's post-hoc tests with BH-FDR correction
   and Cliff's Delta effect size across experimental conditions.
7. Exports publication-ready standalone PNG plots, mask previews, dashboard, and CSV metrics.
========================================================================================
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_closing, binary_dilation

# Try importing oirfile (available in Python 3.12+ environments such as gui_analysis_312)
try:
    import oirfile
    HAS_OIRFILE = True
except ImportError:
    HAS_OIRFILE = False

# Try importing skimage thresholding
try:
    from skimage.filters import threshold_otsu, threshold_triangle, threshold_li, threshold_yen
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Add repository root to path for imports from core/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.nonparametric_stats import (
    run_kruskal_wallis,
    run_dunns_posthoc_fdr,
    p_to_stars,
    format_p_value,
)

# ============================================================================
# CONFIGURATION -- Adjust directory mappings and parameters
# ============================================================================

# Dictionary of condition labels -> directory path containing .oir / .tif images
DATA_DIRS: Dict[str, str] = {
    "Round 1 - Ref": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 1 - Ref",
    "Round 2 - Ref": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 2 - Ref",
    "Round 1 - AKI": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 1 - AKI",
    "Round 2 - AKI": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 2 - AKI",
    "Round 1 - CKD": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 1 - CKD",
    "Round 2 - CKD": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 2 - CKD",
}

# Channel identities
CH0_NAME: str = "FAD"
CH1_NAME: str = "NADH"

# Filename pattern filter to isolate 2-channel TPAF images (e.g. 'tpf' to exclude SRS/HSI tiles)
FILENAME_FILTER: str = "tpf"

# ============================================================================
# TISSUE MASKING PARAMETERS (Customize these to control segmentation)
# ============================================================================
# Method options:
#   "percentile" : Keeps top (100 - PERCENTILE_CUTOFF)% brightest pixels.
#                  RECOMMENDED when the tissue fills 70-90% of the image FOV.
#   "fixed"      : Fixed absolute intensity threshold in counts (FIXED_THRESHOLD).
#   "li"         : Minimum cross-entropy threshold (skimage.filters.threshold_li).
#   "otsu"       : Classic bimodal Otsu threshold (skimage.filters.threshold_otsu).
#   "triangle"   : Geometric triangle threshold (skimage.filters.threshold_triangle).
#   "yen"        : Yen maximum correlation threshold (skimage.filters.threshold_yen).
MASK_METHOD: str = "percentile"

# Percentile cutoff when MASK_METHOD = "percentile"
# (e.g., 15.0 means the darkest 15% pixels are considered background, top 85% is tissue)
PERCENTILE_CUTOFF: float = 20.0

# Fixed intensity cutoff when MASK_METHOD = "fixed" (raw counts)
FIXED_THRESHOLD: float = 450.0

# Pre-smoothing Gaussian filter sigma to suppress PMT Poisson shot noise (set 0.0 to disable)
GAUSSIAN_SIGMA: float = 1.0

# Morphological dilation radius in pixels (expands tissue mask outward to capture margins)
DILATION_PX: int = 2

# Minimum connected object size (px)
MIN_OBJECT_SIZE: int = 50

# Output folder path
SAVE_FOLDER: str = r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\presentation_images_tpf-780"

# Output individual binary masks into subfolders matching each data_dir basename
SAVE_INDIVIDUAL_MASKS: bool = True
MASK_FORMAT: str = "tif"          # "png" | "tif" | "both" (PNG for direct preview, TIF for ImageJ/Fiji)

# Output masked Optical Redox Ratio as 32-bit floating point TIFF into each condition subfolder
SAVE_REDOX_RATIO_TIF: bool = True


# ============================================================================
# Image Ingestion & Layout Standardization
# ============================================================================

def load_tpaf_image(file_path: str) -> Tuple[np.ndarray, str]:
    """
    Loads a 2-channel TPAF image from either .oir or .tif/.tiff format.
    Standardizes output shape to (2, H, W) float32.

    Returns:
        (image_stack, format_string)
    """
    path_obj = Path(file_path)
    suffix = path_obj.suffix.lower()

    if suffix == ".oir":
        if not HAS_OIRFILE:
            raise ImportError(
                "The 'oirfile' library is required to read .oir files. "
                "Please run this script with the Python 3.12 environment (e.g., gui_analysis_312)."
            )
        oir = oirfile.OirFile(str(path_obj))
        raw = oir.asarray().astype(np.float32)
        fmt = "OIR"
    elif suffix in (".tif", ".tiff"):
        raw = tifffile.imread(str(path_obj)).astype(np.float32)
        fmt = "TIFF"
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    raw = np.squeeze(raw)

    # Standardize orientation to (2, H, W)
    if raw.ndim == 2:
        # Single channel image: cannot compute Redox Ratio
        raise ValueError(f"Image {path_obj.name} has only 1 channel; 2 channels (FAD, NADH) required.")
    elif raw.ndim == 3:
        if raw.shape[0] == 2:
            stack = raw
        elif raw.shape[-1] == 2:
            # (H, W, 2) -> (2, H, W)
            stack = np.transpose(raw, (2, 0, 1))
        elif raw.shape[1] == 2:
            # (Z, 2, H) etc.
            stack = np.transpose(raw, (1, 0, 2))
        else:
            # Fallback: take first 2 channels
            stack = raw[:2, :, :]
    elif raw.ndim == 4:
        # e.g., (1, 2, H, W) or (Z, C, H, W)
        reshaped = raw.reshape(-1, raw.shape[-2], raw.shape[-1])
        stack = reshaped[:2, :, :]
    else:
        raise ValueError(f"Unexpected array dimensions for {path_obj.name}: shape {raw.shape}")

    return stack.astype(np.float32), fmt


# ============================================================================
# Tissue Segmentation & Masking
# ============================================================================

def compute_tissue_mask(
    fad: np.ndarray,
    nadh: np.ndarray,
    method: str = MASK_METHOD,
    percentile_cutoff: float = PERCENTILE_CUTOFF,
    fixed_thresh: float = FIXED_THRESHOLD,
    sigma: float = GAUSSIAN_SIGMA,
    dilation_px: int = DILATION_PX,
) -> Tuple[np.ndarray, float]:
    """
    Segments biological tissue from background slide/lumen regions based on combined
    autofluorescence intensity (FAD + NADH).

    Methods:
      - 'percentile': Segments tissue as pixels above the bottom X% percentile (e.g. 15th percentile).
                      Ideal when tissue fills 70-95% of the FOV and Otsu would erroneously cut tissue.
      - 'fixed': Uses an absolute threshold in camera counts (e.g., 450.0).
      - 'li': Minimum cross-entropy thresholding (skimage).
      - 'otsu': Classic Otsu bimodal thresholding.
      - 'triangle': Triangle thresholding.
      - 'yen': Yen thresholding.

    Returns:
        (tissue_mask_boolean, threshold_value)
    """
    total = fad + nadh

    # Pre-smooth to suppress high-frequency Poisson/shot noise
    smoothed = gaussian_filter(total, sigma=sigma) if sigma > 0 else total

    method_lower = (method or "percentile").lower().strip()
    thresh = 0.0

    if method_lower == "percentile":
        thresh = float(np.percentile(smoothed, percentile_cutoff))
    elif method_lower in ["fixed", "absolute"]:
        thresh = float(fixed_thresh)
    elif method_lower == "triangle" and HAS_SKIMAGE:
        thresh = float(threshold_triangle(smoothed))
    elif method_lower == "li" and HAS_SKIMAGE:
        thresh = float(threshold_li(smoothed))
    elif method_lower == "yen" and HAS_SKIMAGE:
        thresh = float(threshold_yen(smoothed))
    elif method_lower == "otsu":
        if HAS_SKIMAGE:
            thresh = float(threshold_otsu(smoothed))
        else:
            hist, bin_edges = np.histogram(smoothed.ravel(), bins=256)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2, 1))[::-1]
            variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            thresh = float(bin_centers[int(np.argmax(variance))])
    else:
        # Fallback to percentile
        thresh = float(np.percentile(smoothed, percentile_cutoff))

    # Initial binary threshold
    mask = smoothed > thresh

    # Morphological closing to seal small intra-cellular / membrane gaps
    if np.any(mask):
        mask = binary_closing(mask, structure=np.ones((3, 3)))
        if dilation_px > 0:
            mask = binary_dilation(mask, iterations=dilation_px)

    return mask.astype(bool), thresh


# ============================================================================
# Single-Image Analysis (SNR & Optical Redox Ratio)
# ============================================================================

def analyze_single_tpaf_image(
    stack: np.ndarray,
    image_name: str,
    condition: str,
    mask_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Analyzes a 2-channel TPAF image:
    - Generates tissue mask
    - Computes SNR (tissue vs background) for Total, FAD, and NADH
    - Computes Optical Redox Ratio: FAD / (FAD + NADH) strictly on tissue pixels

    Returns:
        (metrics_dict, tissue_mask, redox_ratio_map)
    """
    fad = stack[0, :, :]
    nadh = stack[1, :, :]
    h, w = fad.shape
    total = fad + nadh

    # 1. Compute Tissue Mask using configured parameters
    tissue_mask, thresh = compute_tissue_mask(fad, nadh, **(mask_kwargs or {}))
    bg_mask = ~tissue_mask

    n_tissue = int(np.sum(tissue_mask))
    n_total = tissue_mask.size
    tissue_coverage_pct = float(100.0 * n_tissue / max(n_total, 1))

    # 2. Foreground (Tissue) and Background Statistics
    if n_tissue > 0:
        fad_tissue = fad[tissue_mask]
        nadh_tissue = nadh[tissue_mask]
        total_tissue = total[tissue_mask]

        fad_mean_tissue = float(np.mean(fad_tissue))
        fad_std_tissue = float(np.std(fad_tissue))
        nadh_mean_tissue = float(np.mean(nadh_tissue))
        nadh_std_tissue = float(np.std(nadh_tissue))
        total_mean_tissue = float(np.mean(total_tissue))
        total_std_tissue = float(np.std(total_tissue))
    else:
        fad_mean_tissue = nadh_mean_tissue = total_mean_tissue = np.nan
        fad_std_tissue = nadh_std_tissue = total_std_tissue = np.nan

    n_bg = int(np.sum(bg_mask))
    if n_bg > 0:
        fad_bg = fad[bg_mask]
        nadh_bg = nadh[bg_mask]
        total_bg = total[bg_mask]

        fad_mean_bg = float(np.mean(fad_bg))
        fad_std_bg = float(np.std(fad_bg))
        nadh_mean_bg = float(np.mean(nadh_bg))
        nadh_std_bg = float(np.std(nadh_bg))
        total_mean_bg = float(np.mean(total_bg))
        total_std_bg = float(np.std(total_bg))
    else:
        fad_mean_bg = nadh_mean_bg = total_mean_bg = np.nan
        fad_std_bg = nadh_std_bg = total_std_bg = np.nan

    # 3. SNR: Comparing Masked Tissue to Unmasked Background
    if n_tissue > 0 and n_bg > 0 and total_std_bg > 1e-6:
        snr_total = float((total_mean_tissue - total_mean_bg) / total_std_bg)
        snr_fad = float((fad_mean_tissue - fad_mean_bg) / max(fad_std_bg, 1e-6))
        snr_nadh = float((nadh_mean_tissue - nadh_mean_bg) / max(nadh_std_bg, 1e-6))
        contrast_ratio = float(total_mean_tissue / max(total_mean_bg, 1e-6))
    else:
        snr_total = snr_fad = snr_nadh = contrast_ratio = np.nan

    # 4. Optical Redox Ratio: FAD / (FAD + NADH) strictly on tissue
    # Guarding that both signals are strictly positive inside the tissue mask
    valid_tissue = tissue_mask & (fad > 0) & (nadh > 0) & (total > 1e-6)

    if np.any(valid_tissue):
        fad_valid = fad[valid_tissue]
        nadh_valid = nadh[valid_tissue]
        tot_valid = fad_valid + nadh_valid

        # Pixel-wise distribution strictly in (0, 1)
        rr_pixels = fad_valid / tot_valid
        rr_mean = float(np.mean(rr_pixels))
        rr_median = float(np.median(rr_pixels))
        rr_std = float(np.std(rr_pixels))
        rr_iqr = float(np.percentile(rr_pixels, 75) - np.percentile(rr_pixels, 25))

        # Pure ratio FAD / NADH
        pure_ratio_mean = float(np.mean(fad_valid / np.maximum(nadh_valid, 1e-6)))
        pure_ratio_median = float(np.median(fad_valid / np.maximum(nadh_valid, 1e-6)))
    else:
        rr_mean = rr_median = rr_std = rr_iqr = np.nan
        pure_ratio_mean = pure_ratio_median = np.nan

    # Whole-tissue mean level ratio
    if n_tissue > 0 and (fad_mean_tissue + nadh_mean_tissue) > 1e-6:
        rr_from_means = float(fad_mean_tissue / (fad_mean_tissue + nadh_mean_tissue))
    else:
        rr_from_means = np.nan

    record: Dict[str, Any] = {
        "condition": condition,
        "image_name": image_name,
        "height": h,
        "width": w,
        "tissue_coverage_pct": tissue_coverage_pct,
        "Tissue_Coverage_Pct": tissue_coverage_pct,
        "threshold_value": thresh,
        # Optical Redox Ratio
        "Redox_Ratio_Mean": rr_mean,
        "Redox_Ratio_Median": rr_median,
        "Redox_Ratio_STD": rr_std,
        "Redox_Ratio_IQR": rr_iqr,
        "Redox_Ratio_From_Means": rr_from_means,
        "FAD_over_NADH_Mean": pure_ratio_mean,
        "FAD_over_NADH_Median": pure_ratio_median,
        # SNR Metrics
        "SNR_Total": snr_total,
        "SNR_FAD": snr_fad,
        "SNR_NADH": snr_nadh,
        "Contrast_Ratio": contrast_ratio,
        # Raw Intensities
        "FAD_Mean_Tissue": fad_mean_tissue,
        "NADH_Mean_Tissue": nadh_mean_tissue,
        "Total_Mean_Tissue": total_mean_tissue,
        "FAD_Mean_Background": fad_mean_bg,
        "NADH_Mean_Background": nadh_mean_bg,
        "Total_Mean_Background": total_mean_bg,
        "Total_STD_Background": total_std_bg,
    }

    # 2D Spatial Masked Optical Redox Ratio map (32-bit float, 0.0 outside mask)
    redox_map = np.zeros((h, w), dtype=np.float32)
    if np.any(valid_tissue):
        redox_map[valid_tissue] = (fad[valid_tissue] / total[valid_tissue]).astype(np.float32)

    return record, tissue_mask, redox_map


# ============================================================================
# Directory Processing Loop
# ============================================================================

def process_tpaf_directories(
    data_dirs: Dict[str, str],
    filename_filter: str = FILENAME_FILTER,
    mask_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    save_masks: bool = SAVE_INDIVIDUAL_MASKS,
    mask_format: str = MASK_FORMAT,
    save_redox_tif: bool = SAVE_REDOX_RATIO_TIF,
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]]]:
    """
    Iterates through all condition directories, processes every .oir/.tif file
    matching filename_filter (e.g. 'tpf'), and returns metrics DataFrame and
    sample previews for plotting.

    If save_masks is True and output_dir is provided, saves individual binary
    masks into a subfolder matching the basename of each data_dir (e.g. output_dir / 'Round 1 - Ref').

    If save_redox_tif is True, also saves the quantitative masked Optical Redox Ratio
    as a 32-bit floating point TIFF into each condition subfolder.
    """
    records: List[Dict[str, Any]] = []
    condition_previews: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {}

    for label, dir_path in data_dirs.items():
        p = Path(dir_path)
        if not p.is_dir():
            print(f"[Warning] Directory not found for '{label}': {dir_path}. Skipping.")
            continue

        # Look for both .oir and .tif files matching filter substring
        raw_candidates = list(p.glob("*.oir")) + list(p.glob("*.tif")) + list(p.glob("*.tiff"))
        if filename_filter:
            filt = filename_filter.lower()
            all_files = [f for f in raw_candidates if (filt in f.name.lower() or "tpaf" in f.name.lower())]
        else:
            all_files = raw_candidates

        all_files = sorted(all_files, key=lambda x: x.name)
        oir_count = sum(1 for f in all_files if f.suffix.lower() == ".oir")
        tif_count = sum(1 for f in all_files if f.suffix.lower() in [".tif", ".tiff"])

        if not all_files:
            print(f"[Warning] No matching TPAF images (filter='{filename_filter}') found in '{label}' ({dir_path}).")
            continue

        # Subdirectory named with the exact basename of data_dir
        dir_basename = p.name
        mask_out_dir = (output_dir / dir_basename) if (output_dir is not None and (save_masks or save_redox_tif)) else None
        if mask_out_dir is not None:
            mask_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing '{label}' ({len(all_files)} images: {oir_count} .oir, {tif_count} .tif)...")
        if mask_out_dir is not None:
            print(f"  [Output Subfolder] Saving masks & redox maps to: {mask_out_dir}")
        condition_previews[label] = []

        for img_file in all_files:
            try:
                stack, fmt = load_tpaf_image(str(img_file))
                res, mask, redox_map = analyze_single_tpaf_image(stack, img_file.name, label, mask_kwargs=mask_kwargs)
                records.append(res)

                # Save individual binary mask and masked redox ratio TIFF to condition subfolder
                if mask_out_dir is not None:
                    if save_masks:
                        mask_u8 = (mask.astype(np.uint8) * 255)
                        fmt_lower = (mask_format or "tif").lower()
                        if fmt_lower in ["png", "both"]:
                            mask_png_path = mask_out_dir / f"{img_file.stem}_mask.png"
                            Image.fromarray(mask_u8).save(mask_png_path)
                        if fmt_lower in ["tif", "tiff", "both"]:
                            mask_tif_path = mask_out_dir / f"{img_file.stem}_mask.tif"
                            tifffile.imwrite(str(mask_tif_path), mask_u8, compression="zlib")

                    # Save masked Optical Redox Ratio as 32-bit floating point TIFF
                    if save_redox_tif:
                        redox_tif_path = mask_out_dir / f"{img_file.stem}_redox_ratio_masked.tif"
                        tifffile.imwrite(str(redox_tif_path), redox_map.astype(np.float32), compression="zlib")

                # Keep up to 2 sample previews per condition for visualization
                if len(condition_previews[label]) < 2:
                    condition_previews[label].append((img_file.name, stack, mask))

                print(
                    f"  + [{fmt}] {img_file.name} | "
                    f"Tissue={res['tissue_coverage_pct']:.1f}% | "
                    f"SNR={res['SNR_Total']:.2f} | "
                    f"Redox Ratio={res['Redox_Ratio_Mean']:.4f} (med: {res['Redox_Ratio_Median']:.4f})"
                )
            except Exception as e:
                print(f"  [Error] Failed to process {img_file.name}: {e}")

    df_metrics = pd.DataFrame(records)
    return df_metrics, condition_previews


# ============================================================================
# Statistical Analysis (Kruskal-Wallis + Post-hoc Dunn's)
# ============================================================================

def run_group_statistics(
    df: pd.DataFrame,
    metric_cols: List[str],
    group_col: str = "condition",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """Computes Kruskal-Wallis omnibus test and post-hoc Dunn's with BH FDR."""
    kw_results: Dict[str, Dict[str, Any]] = {}
    dunns_results: Dict[str, pd.DataFrame] = {}

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        kw = run_kruskal_wallis(df, value_col=metric, group_col=group_col)
        if kw is not None:
            kw_results[metric] = kw

        dunns = run_dunns_posthoc_fdr(df, value_col=metric, group_col=group_col, alpha=0.05)
        if dunns is not None and not dunns.empty:
            dunns_results[metric] = dunns

    return kw_results, dunns_results


# ============================================================================
# Visualization & Plotting Suite
# ============================================================================

def annotate_posthoc_bars(
    ax: plt.Axes,
    dunns_df: pd.DataFrame,
    conditions: List[str],
    all_values: np.ndarray,
):
    """Draws significance comparison brackets with p-value asterisks."""
    valid_vals = all_values[np.isfinite(all_values)]
    y_max = np.nanmax(valid_vals) if len(valid_vals) > 0 else 1.0
    y_min = np.nanmin(valid_vals) if len(valid_vals) > 0 else 0.0
    span = max(y_max - y_min, 1e-3)
    y_start = y_max + span * 0.08
    h_step = span * 0.08

    cond_to_idx = {c: i for i, c in enumerate(conditions)}
    drawn = 0

    for _, row in dunns_df.iterrows():
        g1 = row.get("A", row.get("group1"))
        g2 = row.get("B", row.get("group2"))
        if g1 not in cond_to_idx or g2 not in cond_to_idx:
            continue
        idx1, idx2 = sorted([cond_to_idx[g1], cond_to_idx[g2]])
        q_val = row.get("q_fdr_bh", row.get("pval", 1.0))
        stars = row.get("significance", p_to_stars(q_val))
        if stars == "ns" and len(dunns_df) > 3:
            continue

        bar_y = y_start + drawn * h_step
        ax.plot([idx1, idx1, idx2, idx2], [bar_y - h_step * 0.2, bar_y, bar_y, bar_y - h_step * 0.2], color="black", lw=1.2)
        ax.text((idx1 + idx2) * 0.5, bar_y, stars, ha="center", va="bottom", fontsize=8, fontweight="bold")
        drawn += 1

    if drawn > 0:
        ax.set_ylim(top=y_start + drawn * h_step + h_step * 0.5)


def render_metric_panel(
    ax: plt.Axes,
    df_metrics: pd.DataFrame,
    metric_key: str,
    title_str: str,
    ylabel_str: str,
    conditions: List[str],
    cond_color_map: Dict[str, Any],
    kw_results: Dict[str, Dict[str, Any]],
    dunns_results: Dict[str, pd.DataFrame],
):
    """Renders a single QC metric panel with scatter points, mean +/- SEM, and stats."""
    sub_df = df_metrics.dropna(subset=[metric_key, "condition"])
    if sub_df.empty:
        ax.text(0.5, 0.5, f"No valid data for {metric_key}", ha="center", va="center")
        return

    x_coords = np.arange(len(conditions))
    group_means = []
    group_sems = []

    np.random.seed(42)  # Consistent horizontal jitter

    for i, cond in enumerate(conditions):
        vals = sub_df.loc[sub_df["condition"] == cond, metric_key].values
        vals = vals[np.isfinite(vals)]
        color = cond_color_map.get(cond, "tab:blue")

        if len(vals) > 0:
            m = float(np.mean(vals))
            sem = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            group_means.append(m)
            group_sems.append(sem)

            jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                i + jitter,
                vals,
                color=color,
                edgecolor="black",
                s=60,
                alpha=0.8,
                zorder=4,
            )
        else:
            group_means.append(np.nan)
            group_sems.append(np.nan)

    ax.errorbar(
        x_coords,
        group_means,
        yerr=group_sems,
        fmt="D",
        color="black",
        ecolor="black",
        elinewidth=2,
        capsize=5,
        capthick=2,
        markersize=7,
        zorder=5,
        label="Condition Mean +/- SEM",
    )

    kw_info = kw_results.get(metric_key)
    kw_subtitle = ""
    if kw_info:
        kw_subtitle = f"\nKruskal-Wallis: H={kw_info['H_stat']:.2f}, p={format_p_value(kw_info['p_unc'])} ({p_to_stars(kw_info['p_unc'])})"

    ax.set_title(f"{title_str}{kw_subtitle}", fontsize=10, fontweight="bold")
    ax.set_xticks(x_coords)
    ax.set_xticklabels(conditions, fontsize=9, rotation=15 if len(conditions) > 3 else 0)
    ax.set_ylabel(ylabel_str, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    dunns_df = dunns_results.get(metric_key)
    if dunns_df is not None and not dunns_df.empty:
        annotate_posthoc_bars(ax, dunns_df, conditions, sub_df[metric_key].values)


def robust_savefig(fig: plt.Figure, output_path: Path, dpi: int = 180, **kwargs) -> Path:
    """
    Saves figure robustly, handling Windows/OneDrive file locks by falling back to
    an alternate file name if the target is locked by an image viewer or cloud sync.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", **kwargs)
        return output_path
    except (OSError, PermissionError):
        alt_path = output_path.with_name(f"{output_path.stem}_new{output_path.suffix}")
        print(f"  [Notice] File {output_path.name} was locked by viewer/cloud sync. Saved as {alt_path.name}")
        fig.savefig(str(alt_path), dpi=dpi, bbox_inches="tight", **kwargs)
        return alt_path


def save_representative_mask_previews(
    condition_previews: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]],
    output_path: Path,
) -> Path:
    """
    Saves a preview grid showing the 2-channel TPAF composite (FAD=Green, NADH=Red)
    along with the cyan boundary of the automated tissue mask across all conditions.
    """
    flat_items = []
    for cond, previews in condition_previews.items():
        for name, stack, mask in previews:
            flat_items.append((cond, name, stack, mask))

    if not flat_items:
        return output_path

    n_samples = min(10, len(flat_items))
    n_cols = min(5, n_samples)
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.4 * n_rows), constrained_layout=True)
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_samples):
        cond, name, stack, mask = flat_items[i]
        ax = axes[i]

        fad = stack[0].astype(np.float32)
        nadh = stack[1].astype(np.float32)

        # Normalize to 99.5 percentile for rich visual contrast
        fad_norm = np.clip(fad / max(np.percentile(fad, 99.5), 1e-6), 0, 1)
        nadh_norm = np.clip(nadh / max(np.percentile(nadh, 99.5), 1e-6), 0, 1)

        # RGB composite: Red = NADH, Green = FAD, Blue = 0
        rgb = np.zeros((fad.shape[0], fad.shape[1], 3), dtype=np.float32)
        rgb[:, :, 0] = nadh_norm  # Red = NADH
        rgb[:, :, 1] = fad_norm   # Green = FAD

        ax.imshow(rgb)
        ax.contour(mask, levels=[0.5], colors=["cyan"], linewidths=[1.3])
        cov = 100.0 * np.sum(mask) / mask.size
        ax.set_title(f"{cond}\n{name}\n(Tissue: {cov:.1f}%)", fontsize=9, fontweight="bold")
        ax.axis("off")

    for j in range(n_samples, len(axes)):
        axes[j].axis("off")

    final_saved = robust_savefig(fig, output_path, dpi=180)
    plt.close(fig)
    return final_saved


def save_tissue_masking_workflow(
    sample_stack: np.ndarray,
    sample_name: str,
    condition: str,
    output_path: Path,
    mask_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Generates a detailed 8-panel diagnostic figure explaining every step of the
    tissue masking pipeline:
    1. Raw FAD Channel
    2. Raw NADH Channel
    3. Total Signal (FAD + NADH) with Gaussian pre-smoothing
    4. Intensity Histogram & Threshold cutoff line (tau)
    5. Raw thresholded binary mask (I_smoothed > tau)
    6. Morphological cleanup (closing + dilation)
    7. Dual-channel composite overlay with cyan tissue boundary
    8. Quantified spatial Optical Redox Ratio map (masked)
    """
    fad = sample_stack[0].astype(np.float32)
    nadh = sample_stack[1].astype(np.float32)
    total = fad + nadh

    kw = (mask_kwargs or {}).copy()
    sigma = kw.get("sigma", GAUSSIAN_SIGMA)
    method_name = kw.get("method", MASK_METHOD).upper()
    smoothed = gaussian_filter(total, sigma=sigma) if sigma > 0 else total

    # Compute threshold and final mask using identical segmentation routine
    mask_final, thresh = compute_tissue_mask(fad, nadh, **kw)
    mask_raw = smoothed > thresh

    # Redox ratio map
    rr_map = np.full_like(fad, np.nan)
    valid = mask_final & (total > 0)
    rr_map[valid] = fad[valid] / total[valid]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    # 1. Raw FAD
    im0 = axes[0, 0].imshow(fad, cmap="Greens_r", vmin=np.percentile(fad, 1), vmax=np.percentile(fad, 99))
    axes[0, 0].set_title(f"1. {CH0_NAME} Channel\n(Raw Autofluorescence)", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Counts")

    # 2. Raw NADH
    im1 = axes[0, 1].imshow(nadh, cmap="Reds_r", vmin=np.percentile(nadh, 1), vmax=np.percentile(nadh, 99))
    axes[0, 1].set_title(f"2. {CH1_NAME} Channel\n(Raw Autofluorescence)", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Counts")

    # 3. Total + Smoothing
    im2 = axes[0, 2].imshow(smoothed, cmap="viridis", vmin=np.percentile(smoothed, 1), vmax=np.percentile(smoothed, 99))
    axes[0, 2].set_title(f"3. Total Signal ($I_{{total}}$)\nGaussian Filter ($\\sigma={sigma}$)", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04, label="Smoothed Counts")

    # 4. Histogram & Threshold
    pixels = smoothed.ravel()
    axes[0, 3].hist(pixels, bins=120, color="#4C72B0", alpha=0.75, edgecolor="none", density=True)
    axes[0, 3].axvline(thresh, color="crimson", linestyle="--", linewidth=2.0, label=f"Threshold $\\tau={thresh:.1f}$ ({method_name})")
    axes[0, 3].set_title(f"4. Pixel Intensity Distribution\n& {method_name} Threshold Cutoff", fontsize=10, fontweight="bold")
    axes[0, 3].set_xlabel("Smoothed Intensity ($I_{total}$)", fontsize=9)
    axes[0, 3].set_ylabel("Density", fontsize=9)
    axes[0, 3].set_yscale("log")
    axes[0, 3].legend(loc="upper right", fontsize=8)
    axes[0, 3].grid(True, linestyle=":", alpha=0.5)

    # 5. Raw Binary Mask
    axes[1, 0].imshow(mask_raw, cmap="gray")
    cov_raw = 100.0 * np.sum(mask_raw) / mask_raw.size
    axes[1, 0].set_title(f"5. Raw Binary Threshold\n($I_{{smoothed}} > {thresh:.1f}$, Area={cov_raw:.1f}%)", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    # 6. Morphological Refinement
    axes[1, 1].imshow(mask_final, cmap="gray")
    cov = 100.0 * np.sum(mask_final) / mask_final.size
    dilation_val = kw.get("dilation_px", DILATION_PX)
    axes[1, 1].set_title(f"6. Final Refined Mask\n(Closing + {dilation_val}px Dilation, Area={cov:.1f}%)", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    # 7. Dual-Channel Overlay
    fad_norm = np.clip(fad / max(np.percentile(fad, 99.5), 1e-6), 0, 1)
    nadh_norm = np.clip(nadh / max(np.percentile(nadh, 99.5), 1e-6), 0, 1)
    rgb = np.zeros((fad.shape[0], fad.shape[1], 3), dtype=np.float32)
    rgb[:, :, 0] = nadh_norm
    rgb[:, :, 1] = fad_norm
    axes[1, 2].imshow(rgb)
    axes[1, 2].contour(mask_final, levels=[0.5], colors=["cyan"], linewidths=[1.5])
    axes[1, 2].set_title("7. Tissue Mask Boundary\n(Cyan Contour on Dual-Ch Composite)", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    # 8. Masked Optical Redox Ratio Map
    cmap_rr = plt.cm.plasma.copy()
    cmap_rr.set_bad(color="#222222")
    im7 = axes[1, 3].imshow(rr_map, cmap=cmap_rr, vmin=0.2, vmax=0.8)
    mean_rr = float(np.nanmean(rr_map)) if np.any(~np.isnan(rr_map)) else 0.0
    axes[1, 3].set_title(f"8. Masked Optical Redox Ratio\n[$FAD / (FAD+NADH)$, Mean={mean_rr:.3f}]", fontsize=10, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04, label="Redox Ratio")

    fig.suptitle(f"Automated Tissue Masking & Optical Redox Quantification Pipeline\nSample: {sample_name} | Condition: {condition}", fontsize=13, fontweight="bold")
    final_saved = robust_savefig(fig, output_path, dpi=180)
    plt.close(fig)
    return final_saved


def save_all_tpaf_plots(
    df_metrics: pd.DataFrame,
    kw_results: Dict[str, Dict[str, Any]],
    dunns_results: Dict[str, pd.DataFrame],
    condition_previews: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]],
    output_dir: Path,
    show_plot: bool = False,
    mask_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """
    Generates standalone PNGs for each QC metric and the combined dashboard.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    conditions = sorted(df_metrics["condition"].unique().tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(conditions), 3)))
    cond_color_map = {cond: colors[i % len(colors)] for i, cond in enumerate(conditions)}

    metric_configs = [
        ("Redox_Ratio_Mean", "Optical Redox Ratio: FAD / (FAD + NADH)", "Mean Tissue Redox Ratio"),
        ("Redox_Ratio_Median", "Median Optical Redox Ratio", "Median Tissue Redox Ratio"),
        ("SNR_Total", "Total TPAF SNR (Tissue vs Background)", "SNR ((Tissue - BG) / STD_BG)"),
        ("SNR_FAD", "FAD Channel SNR", "SNR FAD"),
        ("SNR_NADH", "NADH Channel SNR", "SNR NADH"),
        ("Tissue_Coverage_Pct", "Tissue Coverage Area Fraction", "Tissue Area (%)"),
        ("FAD_Mean_Tissue", "FAD Mean Tissue Autofluorescence", "Mean Counts (a.u.)"),
        ("NADH_Mean_Tissue", "NADH Mean Tissue Autofluorescence", "Mean Counts (a.u.)"),
    ]

    # 1. Save Individual PNGs for each metric
    for m_key, title, ylabel in metric_configs:
        if m_key not in df_metrics.columns:
            continue
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", m_key).strip("_")
        fig_m, ax_m = plt.subplots(figsize=(6.5, 5.2), dpi=200)
        render_metric_panel(
            ax=ax_m,
            df_metrics=df_metrics,
            metric_key=m_key,
            title_str=title,
            ylabel_str=ylabel,
            conditions=conditions,
            cond_color_map=cond_color_map,
            kw_results=kw_results,
            dunns_results=dunns_results,
        )
        p_path = output_dir / f"qc_tpf_{clean_name}.png"
        p_path = robust_savefig(fig_m, p_path, dpi=200)
        plt.close(fig_m)
        saved_files.append(p_path)

    # 2. Save Representative Mask Previews
    mask_preview_path = output_dir / "qc_tpf_representative_masks.png"
    mask_preview_path = save_representative_mask_previews(condition_previews, mask_preview_path)
    if mask_preview_path.exists():
        saved_files.append(mask_preview_path)

    # 3. Save Detailed Tissue Masking Step-by-Step Workflow Explanation
    workflow_path = output_dir / "qc_tpf_tissue_masking_workflow.png"
    # Find a sample with representative tissue coverage (between 10% and 80%)
    chosen_sample = None
    for cond, previews in condition_previews.items():
        for name, stack, mask in previews:
            cov = float(np.mean(mask))
            if 0.10 <= cov <= 0.80:
                chosen_sample = (cond, name, stack)
                break
        if chosen_sample is not None:
            break
    if chosen_sample is None:
        for cond, previews in condition_previews.items():
            if previews:
                chosen_sample = (cond, previews[0][0], previews[0][1])
                break

    if chosen_sample is not None:
        workflow_path = save_tissue_masking_workflow(
            sample_stack=chosen_sample[2],
            sample_name=chosen_sample[1],
            condition=chosen_sample[0],
            output_path=workflow_path,
            mask_kwargs=mask_kwargs,
        )
        if workflow_path.exists():
            saved_files.append(workflow_path)

    # 4. Save Combined Multi-Panel Dashboard
    lead_dashboard_metrics = [
        ("Redox_Ratio_Mean", "Optical Redox Ratio (FAD / (FAD + NADH))", "Mean Ratio"),
        ("SNR_Total", "Signal-to-Noise Ratio (Tissue vs. BG)", "SNR Total"),
        ("Tissue_Coverage_Pct", "Tissue Area Coverage", "Coverage (%)"),
        ("FAD_Mean_Tissue", "FAD Tissue Intensity", "Counts (a.u.)"),
        ("NADH_Mean_Tissue", "NADH Tissue Intensity", "Counts (a.u.)"),
        ("Contrast_Ratio", "Tissue-to-Background Contrast", "Ratio (Tissue / BG)"),
    ]

    total_dash_panels = len(lead_dashboard_metrics)
    n_cols = min(3, total_dash_panels)
    n_rows = int(np.ceil(total_dash_panels / n_cols))

    fig_dash, axes_dash = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.8 * n_rows), constrained_layout=True)
    axes_dash = axes_dash.flatten()

    for idx, (m_key, title, ylabel) in enumerate(lead_dashboard_metrics):
        render_metric_panel(
            ax=axes_dash[idx],
            df_metrics=df_metrics,
            metric_key=m_key,
            title_str=title,
            ylabel_str=ylabel,
            conditions=conditions,
            cond_color_map=cond_color_map,
            kw_results=kw_results,
            dunns_results=dunns_results,
        )

    for j in range(total_dash_panels, len(axes_dash)):
        axes_dash[j].axis("off")

    dash_path = output_dir / "qc_tpf_multicondition_dashboard.png"
    dash_path = robust_savefig(fig_dash, dash_path, dpi=200)
    saved_files.append(dash_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig_dash)

    return saved_files


# ============================================================================
# Synthetic Demonstration Generator (for testing without live files)
# ============================================================================

def generate_mock_tpaf_datasets(base_dir: Path) -> Dict[str, str]:
    """Generates synthetic 2-channel TPAF TIFF stacks for demo mode."""
    mock_dirs: Dict[str, str] = {}
    conditions = ["Round 1 - AKI", "Round 1 - Ref", "Round 2 - AKI"]

    print(f"\n[Demo Mode] Generating synthetic 2-channel TPAF images in: {base_dir}")
    h, w = 512, 512
    y, x = np.ogrid[:h, :w]

    for cond_idx, cond in enumerate(conditions):
        cdir = base_dir / cond.replace(" ", "_")
        cdir.mkdir(parents=True, exist_ok=True)
        mock_dirs[cond] = str(cdir)

        for img_i in range(3):
            # Create synthetic tissue blob (kidney tubules / tissue margin)
            center_y, center_x = h // 2 + np.random.randint(-40, 40), w // 2 + np.random.randint(-40, 40)
            dist_sq = ((y - center_y) / 160.0) ** 2 + ((x - center_x) / 190.0) ** 2
            true_tissue = dist_sq < 1.0

            # Background: dark camera offset (~100 counts) + noise
            fad = np.random.normal(120.0, 15.0, size=(h, w)).astype(np.float32)
            nadh = np.random.normal(110.0, 15.0, size=(h, w)).astype(np.float32)

            # Foreground tissue signals: condition-specific Redox Ratio shift
            fad_signal = 400.0 + cond_idx * 70.0
            nadh_signal = 500.0 - cond_idx * 50.0

            fad[true_tissue] += np.random.normal(fad_signal, 40.0, size=np.sum(true_tissue))
            nadh[true_tissue] += np.random.normal(nadh_signal, 45.0, size=np.sum(true_tissue))

            stack = np.stack([np.maximum(0, fad), np.maximum(0, nadh)], axis=0).astype(np.float32)
            tif_name = f"sample_tpaf_{img_i + 1:02d}.tif"
            tifffile.imwrite(str(cdir / tif_name), stack)

    return mock_dirs


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Two-Photon Autofluorescence (TPAF) Quality Control & Optical Redox Ratio Suite"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic 2-channel TPAF TIFF files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=SAVE_FOLDER,
        help="Directory to save QC figures and summary tables (defaults to SAVE_FOLDER).",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=FILENAME_FILTER,
        help="Substring filter to select TPAF files (default: 'tpf'). Pass empty string '' to process all images.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=MASK_METHOD,
        choices=["percentile", "otsu", "triangle", "li", "yen", "fixed"],
        help=f"Tissue segmentation method (default: '{MASK_METHOD}'). Use 'percentile' or 'li' for high-coverage slices.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=PERCENTILE_CUTOFF,
        help=f"Percentile threshold when --method percentile is used (default: {PERCENTILE_CUTOFF}). Bottom X%% is background.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=FIXED_THRESHOLD,
        help=f"Absolute intensity threshold in counts when --method fixed is used (default: {FIXED_THRESHOLD}).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=GAUSSIAN_SIGMA,
        help=f"Gaussian smoothing sigma (default: {GAUSSIAN_SIGMA}). Set 0 to disable.",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=DILATION_PX,
        help=f"Morphological dilation radius in pixels (default: {DILATION_PX}).",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        default=SAVE_INDIVIDUAL_MASKS,
        help="Export individual binary tissue masks into subfolders named after each data_dir basename (default: True).",
    )
    parser.add_argument(
        "--no-save-masks",
        dest="save_masks",
        action="store_false",
        help="Do not export individual binary mask files.",
    )
    parser.add_argument(
        "--mask-format",
        type=str,
        default=MASK_FORMAT,
        choices=["png", "tif", "both"],
        help=f"Image format for saved binary masks: 'png', 'tif', or 'both' (default: '{MASK_FORMAT}').",
    )
    parser.add_argument(
        "--save-redox-tif",
        action="store_true",
        default=SAVE_REDOX_RATIO_TIF,
        help="Export masked Optical Redox Ratio as 32-bit float TIFF into each condition folder (default: True).",
    )
    parser.add_argument(
        "--no-save-redox-tif",
        dest="save_redox_tif",
        action="store_false",
        help="Do not export masked Optical Redox Ratio TIFF files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive matplotlib windows (useful for headless / automated runs).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine directories to process
    active_dirs = DATA_DIRS.copy()
    if args.demo or not active_dirs:
        if not active_dirs:
            print("Notice: No paths configured in DATA_DIRS dictionary. Launching in DEMO mode.")
        demo_dir = out_dir / "demo_tpaf_stacks"
        active_dirs = generate_mock_tpaf_datasets(demo_dir)

    mask_kwargs = {
        "method": args.method,
        "percentile_cutoff": args.percentile,
        "fixed_thresh": args.fixed_threshold,
        "sigma": args.sigma,
        "dilation_px": args.dilation,
    }

    print("=" * 80)
    print("TWO-PHOTON AUTOFLUORESCENCE (TPAF) QUALITY CONTROL & STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Output Directory      : {out_dir.resolve()}")
    print(f"Filename Filter       : '{args.filter}'")
    print(f"Mask Subdirectory Save: {args.save_masks} (format: {args.mask_format})")
    print(f"Redox Ratio TIFF Save : {args.save_redox_tif}")
    print(f"Channel 0 (Ch0)       : {CH0_NAME}")
    print(f"Channel 1 (Ch1)       : {CH1_NAME}")
    print(f"Metric Evaluated      : Optical Redox Ratio = {CH0_NAME} / ({CH0_NAME} + {CH1_NAME})")
    print(f"Tissue Masking Method : {args.method.upper()} (percentile={args.percentile}%, fixed_thresh={args.fixed_threshold}, sigma={args.sigma}, dilation={args.dilation}px)")
    print(f"Conditions Selected   : {list(active_dirs.keys())}")
    print("=" * 80)

    # 1. Process directories
    df_metrics, condition_previews = process_tpaf_directories(
        active_dirs,
        filename_filter=args.filter,
        mask_kwargs=mask_kwargs,
        output_dir=out_dir,
        save_masks=args.save_masks,
        mask_format=args.mask_format,
        save_redox_tif=args.save_redox_tif,
    )

    if df_metrics.empty:
        print("\n[Error] No valid images were processed. Please check your DATA_DIRS paths.")
        sys.exit(1)

    # Export per-image metrics CSV
    metrics_csv_path = out_dir / "qc_tpf_individual_image_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"\nExported per-image QC metrics to: {metrics_csv_path}")

    # 2. Run Non-Parametric Statistics
    test_metrics = [
        "Redox_Ratio_Mean",
        "Redox_Ratio_Median",
        "SNR_Total",
        "SNR_FAD",
        "SNR_NADH",
        "Tissue_Coverage_Pct",
        "FAD_Mean_Tissue",
        "NADH_Mean_Tissue",
    ]
    kw_results, dunns_results = run_group_statistics(df_metrics, metric_cols=test_metrics, group_col="condition")

    # 3. Print Statistical Summary Tables
    print("\n" + "=" * 80)
    print("STATISTICAL TESTING SUMMARY (Kruskal-Wallis Omnibus & Post-hoc Dunn's)")
    print("=" * 80)
    for m in test_metrics:
        kw = kw_results.get(m)
        if kw:
            print(f"\n>> Metric: {m}")
            print(
                f"   Kruskal-Wallis: H = {kw['H_stat']:.3f}, df = {kw['df']}, "
                f"p = {format_p_value(kw['p_unc'])} ({p_to_stars(kw['p_unc'])}), "
                f"eta^2_H = {kw['eta_sq_h']:.3f}"
            )

        dunns = dunns_results.get(m)
        if dunns is not None and not dunns.empty:
            print("   Dunn's Post-Hoc Pairwise Comparisons (BH FDR corrected):")
            for _, r in dunns.iterrows():
                g1 = r.get("A", r.get("group1"))
                g2 = r.get("B", r.get("group2"))
                q_val = r.get("q_fdr_bh", r.get("pval", 1.0))
                stars = r.get("significance", p_to_stars(q_val))
                delta = r.get("cliffs_delta", np.nan)
                mag = r.get("effect_size_magnitude", r.get("cliffs_magnitude", "N/A"))
                print(
                    f"     * {g1} vs {g2}: "
                    f"q_FDR = {format_p_value(q_val)} ({stars}), "
                    f"Cliff's delta = {delta:.3f} [{mag}]"
                )
            dunns_csv_path = out_dir / f"qc_tpf_dunns_posthoc_{m}.csv"
            dunns.to_csv(dunns_csv_path, index=False)

    # 4. Generate & Save Figures
    print("\n" + "=" * 80)
    print("GENERATING & SAVING QC FIGURES")
    print("=" * 80)
    saved_plots = save_all_tpaf_plots(
        df_metrics=df_metrics,
        kw_results=kw_results,
        dunns_results=dunns_results,
        condition_previews=condition_previews,
        output_dir=out_dir,
        show_plot=not args.no_show,
        mask_kwargs=mask_kwargs,
    )
    for p in saved_plots:
        print(f"  + Saved figure: {p.name} -> {p}")

    print("\nTPAF QC Analysis successfully completed!")


if __name__ == "__main__":
    main()
