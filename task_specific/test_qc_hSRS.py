#!/usr/bin/env python3
"""
========================================================================================
Hyperspectral SRS (hSRS) Multi-Condition Quality Control (QC) & Statistics Suite
========================================================================================
Performs automated QC across multiple acquisition directories/conditions:
1. Iterates through 3D hyperspectral .tif/.tiff images in each condition directory.
2. Inverts channel/spectral axis (flip to ascending 2700 -> 3100 cm^-1 order).
3. Computes spectral mean and standard deviation profiles.
4. Calculates peak ratios (e.g., 2930/2850 cm^-1, 2880/2850 cm^-1) as unique data points
   with spatial distribution statistics (mean and std per image).
5. Computes Signal-to-Noise Ratio (SNR) using the silent baseline (< 2800 cm^-1)
   and the maximum resonant peak.
6. Overlays mean spectra +/- shaded STD ribbon across conditions in a single comparison plot.
7. Plots condition averages with jittered individual image scatter points for each ratio & SNR.
8. Evaluates statistical significance via Kruskal-Wallis omnibus and Dunn's post-hoc
   with Benjamini-Hochberg (BH) FDR correction and Cliff's Delta effect size.
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
from scipy.signal import find_peaks

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
# CONFIGURATION -- Adjust directory mappings and spectral parameters
# ============================================================================

# Dictionary of condition labels -> directory path containing .tif stacks
DATA_DIRS: Dict[str, str] = {
    # "Round1 Ref": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 1 - Ref",
    # "Round2 Ref": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 2 - Ref",
    # "Round1 AKI": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 1 - AKI",
    # "Round2 AKI": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 2 - AKI",
    # "Round1 CKD": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 1 - CKD",
    # "Round2 CKD": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 2 - CKD",
    # "Round3 AKI": r"C:\Users\jvill\OneDrive - UC San Diego\Xenium-SRS\202610 Updates\20261022 comparisons\raw_tile_data - Copy\Round 3 - AKI",
    "8us": r"C:\Users\jvill\Downloads\8us",
    "4us": r"C:\Users\jvill\Downloads\4us",
}

# Spectral range (cm^-1)
WN_START: float = 2700.0
WN_END: float = 3100.0

# Silent / non-resonant baseline region cutoff (< 2800 cm^-1)
SILENT_CUTOFF_WN: float = 2800.0

# Target peak pairs for ratio computation: (numerator_wn, denominator_wn, label)
# Formulas with '+' (e.g. A/(A+B)) automatically compute normalized sum ratios
PEAK_RATIO_TARGETS: List[Tuple[float, float, str]] = [
    (2850.0, 2930.0, "2850/(2930 + 2850) (CH2/(CH3+CH2))"),
    (3015.0, 2880.0, "3015/(3015 + 2880) (Lipid Unsaturation)"),
]

# Local search window for peak maximum (cm^-1)
PEAK_SEARCH_WINDOW_CM: float = 12.0

# ----------------------------------------------------------------------------
# Baseline & Peak Fiducial Alignment (from multimodal_gui/headless/srs_headless)
# ----------------------------------------------------------------------------
ENABLE_BASELINE_SUBTRACTION: bool = False
BASELINE_METHOD: str = "gaussian_water"  # "gaussian_water" | "none"
SILENT_CH_COUNT: int = 15                # Number of silent baseline channels (< 2800 cm^-1)
TAIL_CHANNELS: int = 13                  # Channels at high-wavenumber water tail (> 3015 cm^-1)
CLIP_NON_NEGATIVE: bool = False

ENABLE_FIDUCIAL_PEAK_ALIGNMENT: bool = True
MAX_SHIFT_CHANNELS: int = 6              # Maximum shift search window in channels (+/-)
FIDUCIAL_TARGET_WNS: Tuple[float, ...] = (2850.0, 2880.0, 2930.0)

# Output folder path: set your chosen directory path here!
# All individual PNG plots, dashboard figure, and CSV summary tables will be saved here.
SAVE_FOLDER: str = r"C:\Users\jvill\Downloads\4us"


# ============================================================================
# Processing Functions
# ============================================================================

def find_wavenumber_index(wavenumbers: np.ndarray, target_wn: float) -> int:
    """Finds the index of the closest wavenumber."""
    return int(np.argmin(np.abs(wavenumbers - target_wn)))


def get_peak_channel(
    spectrum: np.ndarray,
    wavenumbers: np.ndarray,
    target_wn: float,
    window_cm: float = PEAK_SEARCH_WINDOW_CM,
) -> int:
    """
    Finds the channel index of the maximum intensity within target_wn +/- window_cm.
    """
    mask = (wavenumbers >= target_wn - window_cm) & (wavenumbers <= target_wn + window_cm)
    if not np.any(mask):
        return find_wavenumber_index(wavenumbers, target_wn)
    indices = np.where(mask)[0]
    local_max = indices[np.argmax(spectrum[indices])]
    return int(local_max)


def apply_shift_1d(arr: np.ndarray, shift: int) -> np.ndarray:
    """Shift 1D array along channel axis with edge-replication padding."""
    if shift == 0:
        return arr.copy()
    n_ch = len(arr)
    shifted = np.empty_like(arr)
    if 0 < shift < n_ch:
        shifted[shift:] = arr[:-shift]
        shifted[:shift] = arr[0]
    elif -n_ch < shift < 0:
        shifted[:shift] = arr[-shift:]
        shifted[shift:] = arr[-1]
    else:
        shifted[:] = arr[0] if shift >= n_ch else arr[-1]
    return shifted


def apply_shift_matrix(ref_matrix: np.ndarray, shift: int) -> np.ndarray:
    """
    Shift 2D matrix (num_items, num_channels) along channel axis with edge-replication padding.
    Vectorized across all rows for high performance.
    """
    if ref_matrix is None or ref_matrix.size == 0 or shift == 0:
        return ref_matrix.copy() if ref_matrix is not None else np.zeros((0, 0), dtype=np.float32)

    n_rows, n_ch = ref_matrix.shape
    if n_ch == 0:
        return ref_matrix.copy()

    shifted = np.empty_like(ref_matrix)
    if 0 < shift < n_ch:
        shifted[:, shift:] = ref_matrix[:, :-shift]
        shifted[:, :shift] = ref_matrix[:, 0:1]
    elif -n_ch < shift < 0:
        shifted[:, :shift] = ref_matrix[:, -shift:]
        shifted[:, shift:] = ref_matrix[:, -1:]
    elif shift >= n_ch:
        shifted[:] = ref_matrix[:, 0:1]
    elif shift <= -n_ch:
        shifted[:] = ref_matrix[:, -1:]
    else:
        shifted[:] = ref_matrix
    return shifted


def estimate_fiducial_peak_shift(
    image_mean: np.ndarray,
    wn_start: float = WN_START,
    wn_end: float = WN_END,
    max_shift_ch: int = MAX_SHIFT_CHANNELS,
    target_wns: Tuple[float, ...] = FIDUCIAL_TARGET_WNS,
) -> Tuple[int, float, List[int], List[float], List[int]]:
    """
    Estimate channel shift by detecting biological Raman peaks in active C-H
    stretch region (~2815 - 2975 cm^-1) and aligning them to invariant biological landmarks:
      - 2850 cm^-1 (Lipid CH2 symmetric stretch)
      - 2880 cm^-1 (Saturated lipid Fermi resonance)
      - 2930 cm^-1 (Protein CH3 symmetric stretch)
    """
    n_ch = len(image_mean)
    if n_ch <= 1:
        return 0, 1.0, [0], [1.0], []

    wn_span = float(wn_end - wn_start)
    if wn_span == 0:
        return 0, 1.0, [0], [1.0], []

    wn_step = wn_span / (n_ch - 1)
    ch_axis = np.arange(n_ch)
    wn_axis = wn_start + ch_axis * wn_step

    min_bio = 2815.0
    max_bio = 2975.0
    bio_mask = (wn_axis >= min_bio) & (wn_axis <= max_bio)
    if not np.any(bio_mask):
        return 0, 1.0, [0], [1.0], []

    bio_min = np.min(image_mean[bio_mask])
    bio_max = np.max(image_mean[bio_mask])
    bio_range = max(float(bio_max - bio_min), 1e-6)
    norm_spec = np.zeros_like(image_mean)
    norm_spec[bio_mask] = (image_mean[bio_mask] - bio_min) / bio_range

    peaks, _ = find_peaks(norm_spec, height=0.08, prominence=0.01, distance=2)
    if len(peaks) == 0:
        peaks, _ = find_peaks(norm_spec, height=0.05, prominence=0.005, distance=2)

    peaks = [int(p) for p in peaks if bio_mask[p]]
    if len(peaks) == 0:
        return 0, 1.0, [0], [1.0], []

    shift_range = list(range(-max_shift_ch, max_shift_ch + 1))
    max_cap_wn = 16.0

    costs = []
    for s in shift_range:
        matched_errors = []
        matched_fids = 0
        for f in target_wns:
            if not (min(wn_start, wn_end) <= f <= max(wn_start, wn_end)):
                continue
            best_d = float("inf")
            for p in peaks:
                sp = p + s
                if 0 <= sp < n_ch:
                    p_wn = wn_axis[sp]
                    d = abs(p_wn - f)
                    if d < best_d and d <= max_cap_wn:
                        best_d = d
            if best_d < float("inf"):
                matched_errors.append(best_d)
                matched_fids += 1

        if matched_fids == 0:
            cost = 999.0
        else:
            shift_penalty = 0.40 * abs(s)
            cost = float(np.mean(matched_errors)) + (len(target_wns) - matched_fids) * 15.0 + shift_penalty
            if s == 0 and float(np.mean(matched_errors)) <= 4.0:
                cost -= 1.5
        costs.append(cost)

    best_idx = int(np.argmin(costs))
    if costs[best_idx] >= 900.0:
        best_shift = 0
        best_score = 0.5
        scores = [0.5] * len(shift_range)
    else:
        best_shift = shift_range[best_idx]
        scores = [float(max(0.0, 1.0 - c / 30.0)) for c in costs]
        best_score = scores[best_idx]

    return best_shift, best_score, shift_range, scores, peaks


def compute_gaussian_water_baseline_vectorized(
    chunk: np.ndarray,
    ch_start: int = SILENT_CH_COUNT,
    tail_channels: int = TAIL_CHANNELS,
    wn_start: float = WN_START,
    wn_end: float = WN_END,
    water_center_wn: float = 3400.0,
    water_sigma_wn: float = 220.0,
) -> np.ndarray:
    """
    Vectorized closed-form physical Gaussian water edge baseline correction.
    Models B(nu) = b0 + A * exp( - (nu - mu)^2 / (2 * sigma^2) )
    - b0 is determined directly from silent baseline (< 2800 cm^-1).
    - A is estimated via non-negative OLS over high-wavenumber water tail (> 3015 cm^-1).
    """
    N, C = chunk.shape
    wn = np.linspace(wn_start, wn_end, C, dtype=np.float32)
    gauss_template = np.exp(-0.5 * ((wn - water_center_wn) / water_sigma_wn) ** 2).astype(np.float32)

    k_silent = max(1, min(ch_start, C)) if ch_start > 0 else 0
    k_tail = max(1, min(tail_channels, C))

    if wn[0] > wn[-1]:
        # High-to-low (e.g. 3100 -> 2700 cm^-1)
        b0 = np.mean(chunk[:, -k_silent:], axis=1, keepdims=True)
        g_tail = gauss_template[:k_tail]
        g_tail_sq = np.sum(g_tail ** 2)
        y_tail_residual = chunk[:, :k_tail] - b0
        A = np.maximum(0.0, np.sum(y_tail_residual * g_tail[np.newaxis, :], axis=1, keepdims=True) / max(g_tail_sq, 1e-6))
    else:
        # Low-to-high (e.g. 2700 -> 3100 cm^-1)
        b0 = np.mean(chunk[:, :k_silent], axis=1, keepdims=True)
        g_tail = gauss_template[-k_tail:]
        g_tail_sq = np.sum(g_tail ** 2)
        y_tail_residual = chunk[:, -k_tail:] - b0
        A = np.maximum(0.0, np.sum(y_tail_residual * g_tail[np.newaxis, :], axis=1, keepdims=True) / max(g_tail_sq, 1e-6))

    baselines = b0 + A * gauss_template[np.newaxis, :]
    return baselines.astype(np.float32)


def load_and_preprocess_stack(
    tif_path: str,
    enable_fiducial_shift: bool = ENABLE_FIDUCIAL_PEAK_ALIGNMENT,
    enable_baseline_sub: bool = ENABLE_BASELINE_SUBTRACTION,
) -> Tuple[np.ndarray, np.ndarray, int, float, np.ndarray]:
    """
    Loads a 3D hyperspectral TIFF, standardizes layout to (C, H, W), flips spectral axis
    (ascending 2700 -> 3100 cm^-1), and applies fiducial peak shift matching and
    physical Gaussian water baseline correction.

    Returns:
        (proc_stack, wavenumbers, shift_channels, shift_score, baseline_mean_spec)
    """
    raw_stack = tifffile.imread(tif_path)
    raw_stack = np.squeeze(raw_stack)

    # Standardize orientation to (C, H, W)
    if raw_stack.ndim == 3:
        if raw_stack.shape[0] > raw_stack.shape[2] and raw_stack.shape[1] > raw_stack.shape[2]:
            # (H, W, C) -> transpose to (C, H, W)
            raw_stack = np.transpose(raw_stack, (2, 0, 1))
    elif raw_stack.ndim == 4:
        raw_stack = raw_stack.reshape(-1, raw_stack.shape[-2], raw_stack.shape[-1])
    else:
        raise ValueError(f"Unexpected image dimensions for {tif_path}: shape {raw_stack.shape}")

    # Flip channel direction to ascending 2700 -> 3100 cm^-1 order
    stack_flipped = np.flip(raw_stack.astype(np.float32), axis=0)
    c_in, h, w = stack_flipped.shape
    wavenumbers = np.linspace(WN_START, WN_END, c_in, dtype=np.float32)

    # 1. Fiducial Peak Shift Matching
    shift_ch = 0
    shift_score = 1.0
    if enable_fiducial_shift:
        raw_mean = np.mean(stack_flipped, axis=(1, 2))
        shift_ch, shift_score, _, _, _ = estimate_fiducial_peak_shift(
            raw_mean,
            wn_start=WN_START,
            wn_end=WN_END,
            max_shift_ch=MAX_SHIFT_CHANNELS,
            target_wns=FIDUCIAL_TARGET_WNS,
        )
        if shift_ch != 0:
            flat_2d = stack_flipped.reshape(c_in, -1).T  # (H*W, C)
            shifted_2d = apply_shift_matrix(flat_2d, shift_ch)
            stack_flipped = shifted_2d.T.reshape(c_in, h, w)

    # 2. Physical Gaussian Water Baseline Subtraction
    baseline_mean_spec = np.zeros(c_in, dtype=np.float32)
    if enable_baseline_sub and BASELINE_METHOD == "gaussian_water":
        pixels = stack_flipped.reshape(c_in, -1).T  # (H*W, C)
        baselines = compute_gaussian_water_baseline_vectorized(
            pixels,
            ch_start=SILENT_CH_COUNT,
            tail_channels=TAIL_CHANNELS,
            wn_start=WN_START,
            wn_end=WN_END,
        )
        baseline_mean_spec = np.mean(baselines, axis=0)
        corrected_pixels = pixels - baselines
        if CLIP_NON_NEGATIVE:
            corrected_pixels = np.maximum(corrected_pixels, 0.0)
        stack_flipped = corrected_pixels.T.reshape(c_in, h, w)

    # 3. Subtract silent baseline
    stack_flipped = stack_flipped - np.mean(stack_flipped[:SILENT_CH_COUNT], axis=0, keepdims=True)

    return stack_flipped, wavenumbers, shift_ch, shift_score, baseline_mean_spec


def analyze_single_image(
    stack: np.ndarray,
    wavenumbers: np.ndarray,
    image_name: str,
    condition: str,
    shift_ch: int = 0,
    shift_score: float = 1.0,
) -> Dict[str, Any]:
    """
    Computes spectral statistics, silent region noise, SNR, and peak ratios for one image.
    """
    c_in, h, w = stack.shape

    # 1. Mean and STD spectrum across spatial pixels (H, W) for each spectral channel
    mean_spectrum = np.mean(stack, axis=(1, 2))  # shape (C,)
    std_spectrum = np.std(stack, axis=(1, 2))   # shape (C,)

    # 2. Silent region (< 2800 cm^-1) vs resonant region (>= 2800 cm^-1)
    silent_mask = wavenumbers < SILENT_CUTOFF_WN
    resonant_mask = wavenumbers >= SILENT_CUTOFF_WN

    if not np.any(silent_mask):
        silent_mask = np.zeros_like(wavenumbers, dtype=bool)
        silent_mask[: max(1, int(c_in * 0.15))] = True
        resonant_mask = ~silent_mask

    # Silent baseline statistics
    silent_voxels = stack[silent_mask, :, :]
    silent_mean = float(np.mean(silent_voxels))
    silent_std = float(np.std(silent_voxels))
    if silent_std == 0:
        silent_std = 1e-6

    # Max peak in the resonant CH-region (>= 2800 cm^-1)
    res_indices = np.where(resonant_mask)[0]
    max_peak_ch = int(res_indices[np.argmax(mean_spectrum[res_indices])])
    max_peak_wn = float(wavenumbers[max_peak_ch])
    max_peak_mean = float(mean_spectrum[max_peak_ch])
    max_peak_std = float(std_spectrum[max_peak_ch])

    # SNR: (Peak Intensity - Silent Baseline) / Silent STD
    snr_intensity = (max_peak_mean - silent_mean) / silent_std
    snr_std_ratio = max_peak_std / silent_std

    record: Dict[str, Any] = {
        "condition": condition,
        "image_name": image_name,
        "fiducial_shift_ch": shift_ch,
        "fiducial_score": shift_score,
        "c_channels": c_in,
        "height": h,
        "width": w,
        "silent_mean": silent_mean,
        "silent_std": silent_std,
        "max_peak_wn": max_peak_wn,
        "max_peak_mean": max_peak_mean,
        "max_peak_std": max_peak_std,
        "SNR": snr_intensity,
        "SNR_std_ratio": snr_std_ratio,
        "_mean_spectrum": mean_spectrum,
        "_std_spectrum": std_spectrum,
    }

    # 3. Peak Ratios
    # Automatically handles normalized sum: A / (A + B) if '+' is in label/formula,
    # otherwise standard ratio: A / B.
    for num_wn, den_wn, ratio_name in PEAK_RATIO_TARGETS:
        ch_num = get_peak_channel(mean_spectrum, wavenumbers, num_wn)
        ch_den = get_peak_channel(mean_spectrum, wavenumbers, den_wn)

        val_num = float(np.clip(mean_spectrum[ch_num], 0, None))
        val_den = float(np.clip(mean_spectrum[ch_den], 0, None))

        slice_num = stack[ch_num, :, :].ravel()
        slice_den = stack[ch_den, :, :].ravel()

        is_sum_ratio = ("+" in ratio_name) or ("/( " in ratio_name) or ("/(" in ratio_name)

        if is_sum_ratio:
            # Normalized sum ratio: I_num / (I_num + I_den)
            denom_mean = val_num + val_den
            ratio_from_means = float(val_num / max(denom_mean, 1e-8)) if denom_mean > 1e-8 else np.nan

            denom_px = slice_num + slice_den
            valid_px = denom_px > 1e-6
            if np.any(valid_px):
                px_ratios = slice_num[valid_px] / denom_px[valid_px]
                px_ratios = px_ratios[px_ratios > 1e-6] # get non_zero values
                ratio_mean_px = float(np.nanmean(px_ratios))
                ratio_std_px = float(np.nanstd(px_ratios))
            else:
                ratio_mean_px = ratio_from_means
                ratio_std_px = 0.0
        else:
            # Standard ratio: I_num / I_den
            ratio_from_means = float(val_num / max(val_den, 1e-8)) if val_den > 1e-8 else np.nan

            valid_px = slice_den > 1e-6
            if np.any(valid_px):
                px_ratios = slice_num[valid_px] / slice_den[valid_px]
                px_ratios = px_ratios[px_ratios > 1e-6] # get non_zero values
                ratio_mean_px = float(np.nanmean(px_ratios))
                ratio_std_px = float(np.nanstd(px_ratios))
            else:
                ratio_mean_px = ratio_from_means
                ratio_std_px = 0.0

        record[ratio_name] = ratio_from_means
        record[f"{ratio_name}_px_mean"] = ratio_mean_px
        record[f"{ratio_name}_px_std"] = ratio_std_px

    return record


def process_directories(
    data_dirs: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Iterates through all directories, processes every .tif/.tiff file,
    and returns a metrics DataFrame along with condition-aggregated spectra.
    """
    records: List[Dict[str, Any]] = []
    condition_spectra: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for label, dir_path in data_dirs.items():
        p = Path(dir_path)
        if not p.is_dir():
            print(f"[Warning] Directory not found for '{label}': {dir_path}. Skipping.")
            continue

        tif_files = sorted(list(p.glob("*.tif")) + list(p.glob("*.tiff")))
        if not tif_files:
            print(f"[Warning] No .tif/.tiff images found in '{label}' ({dir_path}).")
            continue

        print(f"\nProcessing '{label}' ({len(tif_files)} image{'s' if len(tif_files) != 1 else ''})...")
        cond_mean_specs: List[np.ndarray] = []
        cond_wns: Optional[np.ndarray] = None

        for img_file in tif_files:
            try:
                stack, wavenumbers, shift_ch, shift_score, _ = load_and_preprocess_stack(str(img_file))
                cond_wns = wavenumbers
                res = analyze_single_image(
                    stack,
                    wavenumbers,
                    img_file.name,
                    label,
                    shift_ch=shift_ch,
                    shift_score=shift_score,
                )
                cond_mean_specs.append(res["_mean_spectrum"])
                records.append(res)

                # Format dynamic ratio logging
                r_summary = " | ".join(
                    f"{t[2].split()[0]}={res.get(t[2], np.nan):.3f}" for t in PEAK_RATIO_TARGETS
                )
                print(f"  + {img_file.name} | shift={shift_ch:+d} | SNR={res['SNR']:.2f} | {r_summary}")
            except Exception as e:
                print(f"  [Error] Failed to process {img_file.name}: {e}")

        if cond_mean_specs and cond_wns is not None:
            # Aggregate across images in this condition
            specs_array = np.vstack(cond_mean_specs)  # (N_images, C)
            grand_mean = np.nanmean(specs_array, axis=0)
            grand_std = np.std(specs_array, axis=0)
            condition_spectra[label] = (cond_wns, grand_mean, grand_std)

    df_metrics = pd.DataFrame(records)
    return df_metrics, condition_spectra


# ============================================================================
# Statistical Analysis (Kruskal-Wallis + Post-hoc Dunn's)
# ============================================================================

def run_group_statistics(
    df: pd.DataFrame,
    metric_cols: List[str],
    group_col: str = "condition",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """
    Computes Kruskal-Wallis omnibus test and post-hoc Dunn's test with BH FDR
    for each specified metric.
    """
    kw_results: Dict[str, Dict[str, Any]] = {}
    dunns_results: Dict[str, pd.DataFrame] = {}

    for metric in metric_cols:
        if metric not in df.columns:
            continue

        kw = run_kruskal_wallis(df, value_col=metric, group_col=group_col)
        if kw is not None:
            kw_results[metric] = kw

        dunns = run_dunns_posthoc_fdr(df, value_col=metric, group_col=group_col, alpha=0.05)
        if dunns is not None:
            dunns_results[metric] = dunns

    return kw_results, dunns_results


# ============================================================================
# Plotting Suite
# ============================================================================

def render_spectral_panel(
    ax: plt.Axes,
    condition_spectra: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    cond_color_map: Dict[str, Any],
):
    """Renders combined spectral comparison with +/- 1 STD shaded ribbons on given axis."""
    for cond, (wn, mean_s, std_s) in condition_spectra.items():
        max_val = np.max(mean_s)
        if max_val > 1e-8:
            mean_s = mean_s / max_val
            std_s = std_s / max_val
        color = cond_color_map.get(cond, "black")
        ax.plot(wn, mean_s, label=f"{cond} (mean)", color=color, linewidth=2.0)
        ax.fill_between(
            wn,
            mean_s - std_s,
            mean_s + std_s,
            color=color,
            alpha=0.22,
            label=f"{cond} (+/-1 STD)",
        )

    # Annotate silent baseline cutoff and prominent Raman bands
    # ax.axvline(SILENT_CUTOFF_WN, color="gray", linestyle="--", alpha=0.7, label=f"Silent Cutoff ({SILENT_CUTOFF_WN:.0f})")
    annotations = [2850.0, 2880.0, 2930.0, 3015.0]
    for p_wn in annotations:
        if WN_START <= p_wn <= WN_END:
            ax.axvline(p_wn, color="darkred", linestyle=":", alpha=0.45)
            ax.text(
                p_wn,
                ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1.0,
                f" {int(p_wn)}",
                rotation=90,
                fontsize=8,
                verticalalignment="top",
                color="darkred",
            )

    ax.set_title("Hyperspectral Profile Comparison across Conditions", fontsize=11, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm^-1)", fontsize=10)
    ax.set_ylabel("Intensity (a.u.)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85)


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
    """Renders single metric scatter points + condition mean/SEM + statistical annotations."""
    sub_df = df_metrics.dropna(subset=[metric_key, "condition"])
    if sub_df.empty:
        ax.text(0.5, 0.5, f"No data for {metric_key}", ha="center", va="center")
        return

    x_coords = np.arange(len(conditions))
    group_means = []
    group_sems = []

    np.random.seed(42)  # consistent jitter

    for i, cond in enumerate(conditions):
        vals = sub_df.loc[sub_df["condition"] == cond, metric_key].values
        color = cond_color_map.get(cond, "tab:blue")
        if len(vals) > 0:
            m = float(np.mean(vals))
            sem = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            group_means.append(m)
            group_sems.append(sem)

            # Individual image scatter points with horizontal jitter
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

    # Plot condition average point with error bar
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

    # Add Kruskal-Wallis subtitle
    kw_info = kw_results.get(metric_key)
    kw_subtitle = ""
    if kw_info:
        kw_subtitle = f"\nKruskal-Wallis: H={kw_info['H_stat']:.2f}, p={format_p_value(kw_info['p_unc'])} ({p_to_stars(kw_info['p_unc'])})"
    ax.set_title(f"{title_str}{kw_subtitle}", fontsize=10, fontweight="bold")
    ax.set_xticks(x_coords)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel(ylabel_str, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Annotate Dunn's post-hoc pairwise significance if available
    dunns_df = dunns_results.get(metric_key)
    if dunns_df is not None and not dunns_df.empty:
        annotate_posthoc_bars(ax, dunns_df, conditions, sub_df[metric_key].values)


def save_all_plots(
    condition_spectra: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    df_metrics: pd.DataFrame,
    kw_results: Dict[str, Dict[str, Any]],
    dunns_results: Dict[str, pd.DataFrame],
    output_dir: Path,
    show_plot: bool = False,
) -> List[Path]:
    """
    Saves each plot as an individual separate PNG in output_dir, plus a combined dashboard PNG.
    Returns the list of saved plot paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    # Prepare palette and conditions
    conditions = list(condition_spectra.keys())
    if not conditions and "condition" in df_metrics.columns:
        conditions = sorted(df_metrics["condition"].unique().tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(conditions), 3)))
    cond_color_map = {cond: colors[i % len(colors)] for i, cond in enumerate(conditions)}

    ratio_names = [target[2] for target in PEAK_RATIO_TARGETS if target[2] in df_metrics.columns]

    # -----------------------------------------------------------------
    # 1. Individual PNG: Spectral Comparison Plot
    # -----------------------------------------------------------------
    fig_spec, ax_spec = plt.subplots(figsize=(8, 5.5), dpi=300)
    render_spectral_panel(ax_spec, condition_spectra, cond_color_map)
    spec_path = output_dir / "qc_spectral_comparison.png"
    fig_spec.savefig(spec_path, dpi=300, bbox_inches="tight")
    plt.close(fig_spec)
    saved_files.append(spec_path)

    # -----------------------------------------------------------------
    # 2. Individual PNG: SNR Comparison Plot
    # -----------------------------------------------------------------
    fig_snr, ax_snr = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    render_metric_panel(
        ax_snr,
        df_metrics=df_metrics,
        metric_key="SNR",
        title_str="Signal-to-Noise Ratio (SNR)",
        ylabel_str="SNR (Peak Signal / Silent STD)",
        conditions=conditions,
        cond_color_map=cond_color_map,
        kw_results=kw_results,
        dunns_results=dunns_results,
    )
    snr_path = output_dir / "qc_metric_SNR.png"
    fig_snr.savefig(snr_path, dpi=300, bbox_inches="tight")
    plt.close(fig_snr)
    saved_files.append(snr_path)

    # -----------------------------------------------------------------
    # 3. Individual PNGs: Each Peak Ratio Comparison Plot
    # -----------------------------------------------------------------
    for r_name in ratio_names:
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", r_name).strip("_")
        fig_r, ax_r = plt.subplots(figsize=(6.5, 5.5), dpi=300)
        render_metric_panel(
            ax_r,
            df_metrics=df_metrics,
            metric_key=r_name,
            title_str=f"Peak Ratio: {r_name}",
            ylabel_str="Peak Ratio",
            conditions=conditions,
            cond_color_map=cond_color_map,
            kw_results=kw_results,
            dunns_results=dunns_results,
        )
        r_path = output_dir / f"qc_metric_ratio_{clean_name}.png"
        fig_r.savefig(r_path, dpi=300, bbox_inches="tight")
        plt.close(fig_r)
        saved_files.append(r_path)

    # -----------------------------------------------------------------
    # 4. Combined Dashboard PNG: All Panels in One Overview Figure
    # -----------------------------------------------------------------
    metric_configs = [("SNR", "Signal-to-Noise Ratio (SNR)", "SNR (Peak Signal / Silent STD)")]
    for r_name in ratio_names:
        metric_configs.append((r_name, f"Peak Ratio: {r_name}", "Ratio Value"))

    total_panels = 1 + len(metric_configs)
    n_cols = min(3, total_panels)
    n_rows = int(np.ceil(total_panels / n_cols))

    fig_dash = plt.figure(figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
    gs = fig_dash.add_gridspec(n_rows, n_cols)

    ax_dash_spec = fig_dash.add_subplot(gs[0, 0])
    render_spectral_panel(ax_dash_spec, condition_spectra, cond_color_map)

    for idx, (m_key, title, ylabel) in enumerate(metric_configs, start=1):
        row = idx // n_cols
        col = idx % n_cols
        ax_m = fig_dash.add_subplot(gs[row, col])
        render_metric_panel(
            ax_m,
            df_metrics=df_metrics,
            metric_key=m_key,
            title_str=title,
            ylabel_str=ylabel,
            conditions=conditions,
            cond_color_map=cond_color_map,
            kw_results=kw_results,
            dunns_results=dunns_results,
        )

    dash_path = output_dir / "qc_multicondition_dashboard.png"
    fig_dash.savefig(dash_path, dpi=300, bbox_inches="tight")
    saved_files.append(dash_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig_dash)

    return saved_files


def annotate_posthoc_bars(
    ax,
    dunns_df: pd.DataFrame,
    conditions: List[str],
    all_values: np.ndarray,
):
    """Draws significance comparison brackets with p-value asterisks above bars."""
    y_max = np.nanmax(all_values) if len(all_values) > 0 else 1.0
    y_min = np.nanmin(all_values) if len(all_values) > 0 else 0.0
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
        q_val = row.get("q_fdr_bh", row.get("pval", row.get("p_adj_bh", 1.0)))
        stars = row.get("significance", p_to_stars(q_val))
        if stars == "ns" and len(dunns_df) > 3:
            # Avoid cluttering with too many ns bars if many comparisons
            continue

        bar_y = y_start + drawn * h_step
        ax.plot([idx1, idx1, idx2, idx2], [bar_y - h_step * 0.2, bar_y, bar_y, bar_y - h_step * 0.2], color="black", lw=1.2)
        ax.text((idx1 + idx2) * 0.5, bar_y, stars, ha="center", va="bottom", fontsize=8, fontweight="bold")
        drawn += 1

    if drawn > 0:
        ax.set_ylim(top=y_start + drawn * h_step + h_step * 0.5)


# ============================================================================
# Synthetic Demonstration Generator (for testing without live data)
# ============================================================================

def generate_mock_datasets(base_dir: Path) -> Dict[str, str]:
    """Generates synthetic 3D hyperspectral TIFF stacks for testing."""
    mock_dirs: Dict[str, str] = {}
    wns = np.linspace(WN_START, WN_END, 64)
    conditions = ["Round1_Control", "Round2_Treated", "Round3_Perturbed"]

    print(f"\n[Demo Mode] Creating synthetic 3D hyperspectral images in: {base_dir}")
    for cond_idx, cond in enumerate(conditions):
        cdir = base_dir / cond
        cdir.mkdir(parents=True, exist_ok=True)
        mock_dirs[cond] = str(cdir)

        for img_i in range(3):
            # Ground truth spectrum: silent region (<2800) + CH peaks (2850, 2880, 2930)
            baseline = 50.0 + np.random.normal(0, 1.5, size=len(wns))
            peak_2850 = 300.0 * np.exp(-((wns - 2850) / 18.0) ** 2)
            peak_2880 = 180.0 * np.exp(-((wns - 2880) / 15.0) ** 2)
            # Add condition effect on 2930 (CH3)
            ch3_scale = 220.0 + cond_idx * 90.0
            peak_2930 = ch3_scale * np.exp(-((wns - 2930) / 20.0) ** 2)

            true_spec = baseline + peak_2850 + peak_2880 + peak_2930

            # Create 3D stack (C, H, W) = (64, 32, 32)
            c_len = len(wns)
            h, w = 32, 32
            stack = np.zeros((c_len, h, w), dtype=np.float32)
            for c in range(c_len):
                stack[c, :, :] = true_spec[c] + np.random.normal(0, 5.0, size=(h, w))

            # Raw files on microscope are typically flipped spectral axis
            raw_unflipped = np.flip(stack, axis=0)

            tif_name = f"sample_{img_i + 1:02d}.tif"
            tifffile.imwrite(str(cdir / tif_name), raw_unflipped.astype(np.float32))

    return mock_dirs


# ============================================================================
# Main CLI & Workflow
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hyperspectral SRS Multi-Condition Quality Control (QC) & Statistical Testing Suite"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode using synthetic 3D hyperspectral TIFF stacks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=SAVE_FOLDER,
        help="Directory to save QC figures and summary tables (defaults to SAVE_FOLDER).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive matplotlib window (useful for headless / automated runs).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine directories to process
    active_dirs = DATA_DIRS.copy()
    if args.demo or not active_dirs:
        if not active_dirs:
            print("Notice: No paths configured in DATA_DIRS dictionary. Launching in DEMO mode.")
        demo_dir = out_dir / "demo_stacks"
        active_dirs = generate_mock_datasets(demo_dir)

    print("=" * 80)
    print("HYPERSPECTRAL SRS QUALITY CONTROL & GROUP STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Output Directory      : {out_dir.resolve()}")
    print(f"Spectral Range        : {WN_START:.1f} to {WN_END:.1f} cm^-1")
    print(f"Silent Region Cutoff  : < {SILENT_CUTOFF_WN:.1f} cm^-1")
    print(f"Conditions Selected   : {list(active_dirs.keys())}")
    print("=" * 80)

    # 1. Process directories
    df_metrics, condition_spectra = process_directories(active_dirs)

    if df_metrics.empty:
        print("\n[Error] No valid images were processed. Please check your DATA_DIRS paths.")
        sys.exit(1)

    # Clean spectral columns from exported metrics table for readability
    export_df = df_metrics.drop(columns=["_mean_spectrum", "_std_spectrum"], errors="ignore")
    metrics_csv_path = out_dir / "qc_individual_image_metrics.csv"
    export_df.to_csv(metrics_csv_path, index=False)
    print(f"\nExported per-image QC metrics to: {metrics_csv_path}")

    # 2. Run Non-Parametric Statistics
    test_metrics = ["SNR"] + [target[2] for target in PEAK_RATIO_TARGETS if target[2] in df_metrics.columns]
    kw_results, dunns_results = run_group_statistics(df_metrics, metric_cols=test_metrics, group_col="condition")

    # 3. Print Statistical Summary Tables
    print("\n" + "=" * 80)
    print("STATISTICAL TESTING SUMMARY (Kruskal-Wallis Omnibus & Post-hoc Dunn's)")
    print("=" * 80)
    for m in test_metrics:
        kw = kw_results.get(m)
        if kw:
            print(f"\n>> Metric: {m}")
            print(f"   Kruskal-Wallis: H = {kw['H_stat']:.3f}, df = {kw['df']}, p = {format_p_value(kw['p_unc'])} ({p_to_stars(kw['p_unc'])}), eta^2_H = {kw['eta_sq_h']:.3f}")

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
            dunns_csv_path = out_dir / f"qc_dunns_posthoc_{m.replace('/', '_').replace(' ', '_')}.csv"
            dunns.to_csv(dunns_csv_path, index=False)

    # 4. Generate & Save Individual PNGs + Combined Dashboard
    print("\n" + "=" * 80)
    print("GENERATING & SAVING FIGURES")
    print("=" * 80)
    saved_plots = save_all_plots(
        condition_spectra=condition_spectra,
        df_metrics=df_metrics,
        kw_results=kw_results,
        dunns_results=dunns_results,
        output_dir=out_dir,
        show_plot=not args.no_show,
    )
    for p in saved_plots:
        print(f"  + Saved figure: {p.name} -> {p}")

    print("\nQC Analysis successfully completed!")


if __name__ == "__main__":
    main()
